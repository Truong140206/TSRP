"""Score-probe mechanics on synthetic CPU tensors, not experimental results."""
import json
import sys
from types import SimpleNamespace
import numpy as np
import pytest
import torch

from engines import hrm_lora_wtp_and_tap_engine as engine
from engines.prediction_trace import TraceWriter
from tools import prediction_score_probe as probe
from tools import run_prediction_score_audit as runner
from tools.compare_prediction_traces import load_task
from tests.test_prediction_stage_reference import IndexedAdapters, IndexedOriginal


def config(path, weight=.7):
    return SimpleNamespace(eval=True, train_inference_task_only=False, world_size=1,
        num_tasks=10, dataset='synthetic', seed=42, print_freq=1000, train_mask=True,
        task_inc=False, nb_classes=100, En='gen', tau=-10., task_routing_mode='class',
        rp_head=True, rp_route_fusion_drm=True, rp_feature_source='lora', rp_lora_task=0,
        rp_route_fusion_weight=weight, rp_class_fusion_weight=.5,
        rp_class_fusion_gate='margin', rp_class_fusion_min_tasks=1,
        rp_class_fusion_sharpen=1., rp_fusion_ramp=0., classifier_union_audit=False,
        prediction_trace_dir=str(path))


def manifest(path):
    path.mkdir(parents=True)
    (path / 'run.json').write_text(json.dumps(dict(dataset_name='synthetic', seed=42)))


@pytest.mark.parametrize('scale', [.001, .1, 1., 20.])
def test_score_reconstruction_matches_engine_without_mutating_rng_or_tensors(monkeypatch, scale):
    generator = torch.Generator().manual_seed(19)
    primary = torch.randn(13, 100, generator=generator) * scale
    rp = torch.randn(13, 100, generator=generator)
    monkeypatch.setattr(engine, 'args_ref', [config('unused')])
    monkeypatch.setattr(engine, 'gate_stats', {})
    final = engine.fuse_class_scores(primary, rp, .5, 10)
    before = (primary.clone(), rp.clone(), final.clone(), torch.get_rng_state().clone())
    for i in range(len(primary)):
        result = probe.describe_scores(primary[i].tolist(), rp[i].tolist(), final[i].tolist(),
                                        0, int(primary[i].argmax()), int(final[i].argmax()))
        assert 0 <= result['effective_rp_weight'] <= .5
        assert result['reconstruction_max_abs_error'] < 1e-4
    for a, b in zip(before[:3], (primary, rp, final)):
        assert torch.equal(a, b)
    assert torch.equal(before[3], torch.get_rng_state())


def test_bad_final_scores_rejected(monkeypatch):
    monkeypatch.setattr(engine, 'args_ref', [config('unused')])
    monkeypatch.setattr(engine, 'gate_stats', {})
    primary, rp = torch.arange(100).float()[None, :], torch.arange(99, -1, -1).float()[None, :]
    final = engine.fuse_class_scores(primary, rp, .5, 10) + 1
    with pytest.raises(ValueError, match='reproduce'):
        probe.describe_scores(primary[0].tolist(), rp[0].tolist(), final[0].tolist(), 0, 99, 99)


def test_counterfactual_reports_ties_as_ambiguous():
    assert probe.mix_prediction([0., 0.], [1., 1.], .5) == dict(predicted_class=None, tied_classes=[0, 1])


@pytest.mark.parametrize('field,value', [('rp_class_fusion_gate', 'none'),
                                        ('rp_class_fusion_weight', .7), ('eval', False)])
def test_configuration_guard_rejects_changed_fusion(field, value):
    historical = vars(config('old'))
    current = dict(historical, prediction_trace_dir='new')
    probe.check_configuration(current, historical)
    current[field] = value
    with pytest.raises(ValueError, match='differs'):
        probe.check_configuration(current, historical)


def test_observer_validates_nonselected_rows(tmp_path):
    from tests import test_prediction_trace_tools as fixtures
    helper = fixtures.TraceTests()
    row = helper.row(0, 0, 0, 0)
    historical = [dict(type='sample', **row)]
    base = TraceWriter(tmp_path / 'trace.jsonl', dict(samples=1, eval_task=0))
    base.pending = [dict(row)]
    base.pending[0].pop('final_class')
    base.pending[0].pop('final_logits_sha256')
    observer = probe.ScoreObserver(base, historical, set(), [])
    # Selection is empty, but every historical row must still be checked.
    with pytest.raises(ValueError, match='Historical sample mismatch'):
        observer.end(torch.tensor([[1., 0, 0, 0, 0, 0]]))


def test_probe_preserves_full_sequence_and_adds_only_final_class_gate(monkeypatch, tmp_path):
    import main as application
    import utils
    from trainers import lora_trainer as trainer
    monkeypatch.setattr(utils, 'cleanup_distributed', lambda: None)
    monkeypatch.setattr(engine, 'args_ref', [None])
    monkeypatch.setattr(engine, 'gate_stats', {})
    generator = torch.Generator().manual_seed(23)
    original = torch.randn(20, 100, generator=generator)
    bank = torch.randn(20, 10, 100, generator=generator)
    rp = torch.randn(20, 100, generator=generator)
    monkeypatch.setattr(engine, 'rp_head_predict', lambda features, *a: rp[features[:, 0].long()].clone())
    loaders = [dict(val=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(
        torch.tensor([[2 * t], [2 * t + 1]]).float(), torch.tensor([10 * t, 10 * t + 1])),
        batch_size=2)) for t in range(10)]
    masks = [list(range(10 * t, 10 * t + 10)) for t in range(10)]
    mapping = {c: c // 10 for c in range(100)}
    history = tmp_path / 'history'
    destination = tmp_path / 'new'
    for root in (history, destination):
        for arm in ('full', 'class_gate'):
            manifest(root / arm)
    model, original_model = IndexedAdapters(bank), IndexedOriginal(original)
    device = torch.device('cpu')
    old_evaluate = trainer.evaluate_till_now
    # Historical traces generated independently with the original observer.
    for arm, weight in (('full', .7), ('class_gate', 1.)):
        args = config(history / arm, weight)
        for task in range(10):
            engine.evaluate(model, original_model, loaders[task]['val'], device,
                            i=task, task_id=9, class_mask=masks, target_task_map=mapping, args=args)
    selection = {str(t): [0, 1] for t in range(10)}
    (destination / 'plan.json').write_text(json.dumps(dict(selected=selection)))
    seen_stages = []
    def fake_main(args):
        matrix = np.zeros((10, 10))
        for stage in range(10):
            # Represents each untouched accumulation/solve boundary in trainer.
            seen_stages.append(stage)
            trainer.evaluate_till_now(model, original_model, loaders, device, stage,
                                      masks, mapping, matrix, args)
    monkeypatch.setattr(application, 'get_args', lambda: config(destination / 'full'))
    monkeypatch.setattr(application, 'main', fake_main)
    original_start = engine.start_trace
    argv = sys.argv[:]
    probe.run_probe(history, destination, ['unused'])
    assert seen_stages == list(range(10))
    assert trainer.evaluate_till_now is old_evaluate
    assert engine.start_trace is original_start
    assert sys.argv == argv
    for arm in ('full', 'class_gate'):
        records = json.loads((destination / arm / 'scores.json').read_text())
        assert len(records) == 20
        for task in range(10):
            _, old = load_task(history / arm / ('task_%02d.jsonl' % task))
            _, new = load_task(destination / arm / ('task_%02d.jsonl' % task))
            assert old == new
    result = runner.combine_scores(destination, {(t, i) for t in range(10) for i in (0, 1)})
    assert len(result) == 20
    assert 'report_only_counterfactuals' in result[0]
    # Stored vectors must not be accepted after RP evidence is tampered with.
    path = destination / 'full' / 'scores.json'
    data = json.loads(path.read_text())
    data[0]['scores']['rp_scores'][0] += 1
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='RP score vectors'):
        runner.combine_scores(destination, {(t, i) for t in range(10) for i in (0, 1)})


def test_child_command_keeps_eval_configuration_and_torchrun(tmp_path):
    args = SimpleNamespace(dataset='cifar100', seed=42, batch_size=24, port=29558,
                           resolved_data_path=tmp_path / 'data', history=tmp_path / 'old')
    cmd = runner.child_command(args, tmp_path / 'tii', tmp_path / 'lora', tmp_path / 'new')
    assert cmd[:3] == [sys.executable, '-m', 'torch.distributed.run']
    assert '--nproc_per_node=1' in cmd
    app = cmd[cmd.index('--') + 1:]
    assert app[0] == 'cifar100_lora'
    assert '--eval' in app
    assert app[app.index('--prediction_trace_dir') + 1] == str(tmp_path / 'new' / 'full')
    old_argv = sys.argv
    try:
        import main
        sys.argv = ['main.py', *app]
        parsed = main.get_args()
    finally:
        sys.argv = old_argv
    assert parsed.eval and parsed.seed == 42 and parsed.rp_route_fusion_weight == .7
    assert parsed.En == 'gen' and parsed.tau == -10 and parsed.num_tasks == 10


@pytest.mark.parametrize('problem', [None, 'source', 'checkpoints', 'runtime', 'data'])
def test_read_only_preflight_never_starts_gpu_or_writes_output(monkeypatch, tmp_path, problem):
    out, data, history = (tmp_path / name for name in ('out', 'data', 'old'))
    for path in (out, data, history):
        path.mkdir()
    runtime = {name: 'test' for name in ('torch', 'torchvision', 'timm', 'numpy')}
    runtime.update(python=sys.version, torch_cuda=torch.version.cuda,
                   cuda_visible_devices=runner.os.environ.get('CUDA_VISIBLE_DEVICES'))
    common = dict(dataset_path=str(data), source_sha256='same', checkpoints=[], runtime=runtime)
    if problem == 'source':
        common['source_sha256'] = 'changed'
    elif problem == 'checkpoints':
        common['checkpoints'] = [dict(path='changed', sha256='changed')]
    elif problem == 'runtime':
        common['runtime'] = dict(runtime, torch='changed')
    elif problem == 'data':
        common['dataset_path'] = 'changed'
    monkeypatch.setattr(sys, 'argv', ['audit', '--output-root', str(out), '--data-root', str(data),
                                     '--history', str(history), '--tag', 'scores'])
    monkeypatch.setattr(runner, 'fingerprint_history', lambda p: {})
    monkeypatch.setattr(runner.stages, 'audit', lambda p: dict(dataset='Split-CIFAR100', seed=42,
        changed_class_samples=[dict(eval_task=3, index=702)]))
    monkeypatch.setattr(runner.compare, 'load_task', lambda p: (dict(run=dict(common=common)), []))
    monkeypatch.setattr(runner.base, 'resolve_data', lambda args: data)
    monkeypatch.setattr(runner.base.verify, 'source_digest', lambda: 'same')
    monkeypatch.setattr(runner.base.grid, 'manifest', lambda *a: ([], {}))
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(runner.base.verify, 'read_stages', lambda path: {})
    def forbidden(*a, **kw):
        raise AssertionError('preflight must not launch GPU or create reports')
    monkeypatch.setattr(runner.base, 'idle_gpu', forbidden)
    monkeypatch.setattr(runner.base, 'execute', forbidden)
    monkeypatch.setattr(runner.base, 'save_new', forbidden)
    if problem:
        with pytest.raises(ValueError, match='differ'):
            runner.main()
    else:
        runner.main()
    assert list(out.iterdir()) == []
