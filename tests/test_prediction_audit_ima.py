"""ImageNet-A pilot mechanics; synthetic fixtures, not experimental results."""
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from tools import run_prediction_audit as runner
from tools.compare_prediction_traces import compare_directories, load_task


def options(root, seed=42):
    return SimpleNamespace(dataset='ima', seed=seed, batch_size=24, port=29558,
                           data_root=root, data_path=None, resolved_data_path=root)


def prepare(root, classes=200):
    for split in ('train', 'test'):
        for i in range(classes):
            folder = root / 'imagenet-a' / split / ('n%08d' % i)
            folder.mkdir(parents=True)
            (folder / 'sample.jpg').write_bytes(b'layout fixture, not an image')
    return root


def snapshot(root):
    return {p.relative_to(root).as_posix(): p.read_bytes()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('seed', [42, 43, 44, 45])
def test_ima_command_matches_verified_grid_and_only_changes_routing(seed, tmp_path):
    args = options(tmp_path, seed)
    tii, lora = tmp_path / 'tii', tmp_path / 'lora'
    full = runner.build_command(args, tii, lora, 'full')
    expected = runner.grid.command(None, 'ima', 'sup', 'full', tii, lora, tmp_path)
    runner.finish.set_flag(expected, '--seed', seed)
    assert full == expected
    cfg = runner.grid.parsed_config(full)
    assert cfg.dataset == 'Split-Imagenet-A' and cfg.seed == seed
    assert cfg.num_tasks == cfg.size == 10 and cfg.lora_rank == 8
    assert cfg.eval and cfg.strict_exemplar_free and not cfg.train_inference_task_only
    assert cfg.model == cfg.original_model == 'vit_base_patch16_224'
    assert (cfg.rp_dim, cfg.rp_lambda, cfg.rp_seed) == (10000, 10000., 1993)
    assert cfg.rp_feature_source == 'lora' and cfg.rp_lora_task == 0
    assert cfg.rp_class_fusion_gate == 'margin' and cfg.rp_class_fusion_weight == .5
    assert not cfg.shuffle
    gate = runner.build_command(args, tii, lora, 'class_gate')
    runner.finish.set_flag(gate, '--rp_route_fusion_weight', .7)
    assert gate == full
    assert 'ima' not in runner.verify.DATASETS  # No change to old audit grids.


def test_ima_real_main_parser_trace_flag_and_custom_launch(monkeypatch, tmp_path):
    import main
    args = options(tmp_path)
    args.batch_size, args.port = 7, 29570
    cmd = runner.build_command(args, tmp_path / 'tii', tmp_path / 'lora', 'full', tmp_path / 'trace')
    assert '--master_port=29570' in cmd
    monkeypatch.setattr(sys, 'argv', cmd[cmd.index('main.py'):])
    parsed = main.get_args()
    assert parsed.config == parsed.subparser_name == 'ima_lora'
    assert parsed.prediction_trace_dir == str(tmp_path / 'trace') and parsed.batch_size == 7
    assert parsed.opt_betas == (.9, .999)
    with pytest.raises(ValueError, match='only full and class_gate'):
        runner.build_command(args, tmp_path / 'tii', tmp_path / 'lora', 'rp_only')


def test_existing_dataset_commands_unchanged(tmp_path):
    for dataset in runner.verify.DATASETS:
        args = options(tmp_path)
        args.dataset = dataset
        for arm in ('full', 'class_gate'):
            assert runner.build_command(args, tmp_path / 'tii', tmp_path / 'lora', arm) == (
                runner.finish.variant_command(args, 42, tmp_path / 'tii', tmp_path / 'lora',
                                              'vit_base_patch16_224', arm))


@pytest.mark.parametrize('field,value', [('dataset', 'Split-Imagenet-R'), ('seed', 1),
                                        ('model', 'vit_base_patch16_224_mocov3'), ('num_tasks', 5)])
def test_ima_checkpoint_guard_rejects_wrong_experiment(tmp_path, field, value):
    cmd = runner.build_command(options(tmp_path), tmp_path / 'tii', tmp_path / 'lora', 'full')
    expected = runner.grid.parsed_config(cmd)
    saved = vars(expected).copy()
    runner.grid.checkpoint_args(saved, expected, 'lora')
    saved[field] = value
    with pytest.raises(ValueError, match=field):
        runner.grid.checkpoint_args(saved, expected, 'lora')


@pytest.mark.parametrize('layout', ['parent', 'nested', 'dataset_folder', 'exact'])
def test_ima_prepared_layout_is_read_only(tmp_path, layout):
    actual = tmp_path / 'imagenet-a' if layout == 'nested' else tmp_path
    prepare(actual)
    args = options(tmp_path)
    if layout == 'dataset_folder':
        args.data_root = tmp_path / 'imagenet-a'
    if layout == 'exact':
        args.data_path = actual
    before = snapshot(tmp_path)
    assert runner.resolve_data(args) == actual.resolve()
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize('exact', [False, True])
@pytest.mark.parametrize('problem', ['raw_unsplit', 'one_split', 'class_mismatch', 'empty_class', '100_classes'])
def test_ima_bad_layout_refuses_without_moving_images(tmp_path, problem, exact):
    if problem == 'raw_unsplit':
        raw = tmp_path / 'imagenet-a' / 'n00000000'
        raw.mkdir(parents=True)
        (raw / 'raw.jpg').write_bytes(b'must not be moved')
    elif problem == 'one_split':
        (tmp_path / 'imagenet-a' / 'train').mkdir(parents=True)
    else:
        prepare(tmp_path, 100 if problem == '100_classes' else 200)
        path = tmp_path / 'imagenet-a' / 'test' / 'n00000000'
        if problem == 'class_mismatch':
            path.rename(path.with_name('n99999999'))
        elif problem == 'empty_class':
            (path / 'sample.jpg').rename(path / 'not_an_image.txt')
    args = options(tmp_path)
    args.data_path = tmp_path if exact else None
    before = snapshot(tmp_path)
    with pytest.raises(ValueError, match='ImageNet-A'):
        runner.resolve_data(args)
    assert snapshot(tmp_path) == before


def test_ima_ambiguous_layout_needs_exact_parent(tmp_path):
    prepare(tmp_path)
    prepare(tmp_path / 'imagenet-a')
    args = options(tmp_path)
    with pytest.raises(ValueError, match='Need one existing'):
        runner.resolve_data(args)
    args.data_path = tmp_path
    assert runner.resolve_data(args) == tmp_path.resolve()
    args.data_path = tmp_path / 'imagenet-a'
    assert runner.resolve_data(args) == args.data_path.resolve()


@pytest.mark.parametrize('bad_data', [False, True])
def test_ima_preflight_stays_cpu_only_and_checks_data_before_checkpoints(monkeypatch, tmp_path, bad_data):
    import torch
    out, data = tmp_path / 'out', tmp_path / 'data'
    out.mkdir(); data.mkdir()
    monkeypatch.setattr(sys, 'argv', ['audit', '--dataset', 'ima', '--seed', '42',
        '--output-root', str(out), '--data-root', str(data), '--tag', 'ima_pilot'])
    calls = []
    def check_manifest(tii, lora, expected, pretrained):
        calls.append(expected.dataset)
        assert tii.name == 'ima_tii_original_10tasks_seed42'
        assert lora.name == 'ima_lora_rank8_baseline_10tasks_seed42'
        assert expected.dataset == 'Split-Imagenet-A' and expected.seed == 42
        return [], {}
    def forbidden(*args, **kwargs):
        raise AssertionError('Read-only preflight must not create output or use the GPU')
    if not bad_data:
        monkeypatch.setattr(runner, 'resolve_data', lambda args: data)
    monkeypatch.setattr(runner.grid, 'manifest', check_manifest)
    monkeypatch.setattr(runner.verify, 'source_digest', lambda: 'unchanged')
    monkeypatch.setattr(runner.importlib.metadata, 'version', lambda name: 'test')
    monkeypatch.setattr(runner, 'idle_gpu', forbidden)
    monkeypatch.setattr(runner, 'execute', forbidden)
    monkeypatch.setattr(runner, 'save_new', forbidden)
    monkeypatch.setattr(torch.cuda, 'is_available', forbidden)
    if bad_data:
        with pytest.raises(ValueError, match='ImageNet-A'):
            runner.main()
        assert calls == []
    else:
        runner.main()
        assert calls == ['Split-Imagenet-A']
    assert list(out.iterdir()) == []


def evaluation_rows():
    return ''.join('[Average accuracy till task%d] Acc@task: 80 Acc@1: 75 '
                   'Acc@5: 90 Loss: 1 Forgetting: 3 Backward: -2\n' % t for t in range(1, 11))


@pytest.mark.parametrize('exit_code', [0, 1])
def test_subprocess_timing_preserves_completion_and_failure_guards(monkeypatch, tmp_path, exit_code):
    path = tmp_path / 'console.log'
    process = SimpleNamespace(stdout=io.StringIO(evaluation_rows()), wait=lambda: exit_code)
    ticks = iter((100., 112.25))
    monkeypatch.setattr(runner.time, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr(runner, 'idle_gpu', lambda: None)
    monkeypatch.setattr(runner.subprocess, 'Popen', lambda *args, **kwargs: process)
    if exit_code:
        with pytest.raises(RuntimeError, match='Preserve log'):
            runner.execute(path, ['fake'], {})
        with pytest.raises(ValueError, match='completed evaluation'):
            runner.evaluation_seconds(path)
    else:
        assert len(runner.execute(path, ['fake'], {})) == 10
        assert runner.evaluation_seconds(path) == 12.25
    assert path.read_text().splitlines()[-1] == 'VERIFICATION_EXIT_CODE=%d' % exit_code
    before = path.read_bytes()
    def forbidden(*args, **kwargs):
        raise AssertionError('Existing output must never relaunch a process')
    monkeypatch.setattr(runner, 'idle_gpu', forbidden)
    monkeypatch.setattr(runner.subprocess, 'Popen', forbidden)
    if exit_code:
        with pytest.raises(ValueError, match='unfinished'):
            runner.execute(path, ['fake'], {})
    else:
        runner.execute(path, ['fake'], {})
    assert path.read_bytes() == before


@pytest.mark.parametrize('value', ['nan', 'inf', '-1', '1\nAUDIT_EVAL_ELAPSED_SECONDS=2'])
def test_bad_timing_rejected(tmp_path, value):
    path = tmp_path / 'console.log'
    path.write_text('AUDIT_EVAL_ELAPSED_SECONDS=' + value + '\nVERIFICATION_EXIT_CODE=0\n')
    with pytest.raises(ValueError):
        runner.evaluation_seconds(path)


@pytest.mark.parametrize('old_log', [False, True])
def test_timing_report_is_separate_reusable_and_never_overwritten(tmp_path, old_log):
    for arm in ('full_plain', 'full', 'class_gate'):
        folder = tmp_path / arm
        folder.mkdir()
        timing = '' if old_log and arm == 'full_plain' else 'AUDIT_EVAL_ELAPSED_SECONDS=2.5\n'
        (folder / 'console.log').write_text(timing + 'VERIFICATION_EXIT_CODE=0\n')
    result, path = runner.save_timings(tmp_path)
    assert result['total_evaluation_seconds'] == (None if old_log else 7.5)
    if old_log:
        assert result['evaluation_seconds']['full_plain'] is None
    before = path.read_bytes()
    runner.save_timings(tmp_path)
    assert path.read_bytes() == before
    path.write_text('{}')
    with pytest.raises(ValueError, match='Existing timing report differs'):
        runner.save_timings(tmp_path)
    assert path.read_text() == '{}'


def test_200_class_ten_task_observer_pair_with_unequal_sizes(monkeypatch, tmp_path):
    import torch
    from engines import hrm_lora_wtp_and_tap_engine as engine
    from tests.test_prediction_stage_reference import IndexedAdapters, IndexedOriginal
    masks = [list(range(20 * t, 20 * (t + 1))) for t in range(10)]
    sizes = [2 + t % 3 for t in range(10)]
    n = sum(sizes)
    generator = torch.Generator().manual_seed(310)
    original = torch.randn(n, 200, generator=generator)
    bank = torch.randn(n, 10, 200, generator=generator)
    rp = torch.randn(n, 200, generator=generator)
    monkeypatch.setattr(engine, 'args_ref', [None])
    monkeypatch.setattr(engine, 'gate_stats', {})
    monkeypatch.setattr(engine, 'rp_head_predict', lambda features, *a: rp[features[:, 0].long()].clone())
    loaders, offset = [], 0
    for task, count in enumerate(sizes):
        data = torch.utils.data.TensorDataset(torch.arange(offset, offset + count).float()[:, None],
                                              torch.arange(20 * task, 20 * task + count))
        loaders.append(torch.utils.data.DataLoader(data, batch_size=2))
        offset += count
    logs, calls, rngs = {}, {}, {}
    common = dict(source='same', checkpoints=['same'])
    for name, variant, trace in (('full_plain', 'full', False), ('full', 'full', True),
                                 ('class_gate', 'class_gate', True)):
        cmd = runner.build_command(options(tmp_path), tmp_path / 'tii', tmp_path / 'lora', variant)
        args = runner.grid.parsed_config(cmd)
        args.world_size, args.nb_classes, args.print_freq = 1, 200, 1000
        args.prediction_trace_dir = None
        if trace:
            folder = tmp_path / name
            folder.mkdir()
            (folder / 'run.json').write_text(json.dumps(dict(dataset_name='Split-Imagenet-A',
                seed=42, variant=variant, common=common)))
            args.prediction_trace_dir = str(folder)
        model = IndexedAdapters(bank)
        torch.manual_seed(42)
        stats = [engine.evaluate(model, IndexedOriginal(original), loader, torch.device('cpu'),
                  i=t, task_id=9, class_mask=masks, target_task_map={c: c // 20 for c in range(200)},
                  args=args) for t, loader in enumerate(loaders)]
        logs[name] = {10: {key: sum(s[key] for s in stats) / 10 for key in runner.verify.CORE}}
        calls[name], rngs[name] = model.calls, torch.get_rng_state()
    assert logs['full_plain'] == logs['full']
    assert calls['full_plain'] == calls['full']
    assert torch.equal(rngs['full_plain'], rngs['full'])
    result = compare_directories(tmp_path / 'class_gate', tmp_path / 'full')
    assert result['samples'] == n and sum(result['counts'].values()) == n
    runner.check_trace_metrics(result, logs)
    for task in range(10):
        header, rows = load_task(tmp_path / 'full' / ('task_%02d.jsonl' % task))
        assert header['dataset'] == 'Split-Imagenet-A' and len(rows) == sizes[task]
        assert all(0 <= row['final_class'] < 200 for row in rows)
