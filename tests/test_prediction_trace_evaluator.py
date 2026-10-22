"""CPU tensor tests of observer on/off in the ACTUAL evaluator; no real data."""
import json
import sys
from types import SimpleNamespace
import pytest
import torch
from engines import hrm_lora_wtp_and_tap_engine as engine
from engines.prediction_trace import TraceWriter, start_trace
from tools.compare_prediction_traces import load_task


class ToyOriginal(torch.nn.Module):
    def forward(self, inputs):
        return dict(logits=inputs.new_tensor([3, 0, 0, 0, 0, 0]).repeat(len(inputs), 1),
                    pre_logits=inputs)


class ToyAdapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, inputs, task_id=0, **kwargs):
        tasks = (torch.full((len(inputs),), task_id, dtype=torch.long)
                 if isinstance(task_id, int) else task_id)
        self.calls.append(tasks.tolist())
        bank = inputs.new_tensor([[0, 0, 2, 0, 1.9, 0], [0, 0, 3, 0, 1, 0],
                                  [0, 0, 0, 0, 4, 1]])
        return dict(logits=bank[tasks].clone(), pre_logits=inputs)


@pytest.mark.parametrize('weight', [1., .7])
@pytest.mark.parametrize('tau', [-1., 1.1])
def test_actual_evaluator_observer_inert(monkeypatch, tmp_path, weight, tau):
    args = SimpleNamespace(eval=True, world_size=1, num_tasks=3, dataset='synthetic', seed=42,
                           print_freq=100, train_mask=True, task_inc=False, nb_classes=6,
                           En='msp', tau=tau, task_routing_mode='class',
                           rp_head=True, rp_route_fusion_drm=True, rp_feature_source='lora',
                           rp_lora_task=0, rp_route_fusion_weight=weight,
                           rp_class_fusion_weight=.5, rp_class_fusion_gate='margin',
                           rp_class_fusion_min_tasks=1, rp_fusion_ramp=0.,
                           classifier_union_audit=False, prediction_trace_dir=None)
    monkeypatch.setattr(engine, 'args_ref', [None])
    monkeypatch.setattr(engine, 'gate_stats', {})
    monkeypatch.setattr(engine, 'rp_head_predict', lambda features, *a:
                        features.new_tensor([0, 0, 0, 0, 10, 0]).repeat(len(features), 1))
    dataset = torch.utils.data.TensorDataset(torch.zeros(3, 2), torch.tensor([4, 5, 4]))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
    def run():
        model = ToyAdapter()
        torch.manual_seed(42)
        stats = engine.evaluate(model, ToyOriginal(), loader, torch.device('cpu'), i=2,
                                task_id=2, class_mask=[[0, 1], [2, 3], [4, 5]],
                                target_task_map={i: i // 2 for i in range(6)}, args=args)
        return stats, model.calls, torch.get_rng_state()
    before, calls, rng = run()
    args.prediction_trace_dir = str(tmp_path)
    (tmp_path / 'run.json').write_text(json.dumps(dict(dataset_name=args.dataset, seed=42)))
    after, traced_calls, traced_rng = run()
    assert before == after
    assert calls == traced_calls
    assert torch.equal(rng, traced_rng)
    header, rows = load_task(tmp_path / 'task_02.jsonl')
    assert len(rows) == 3
    assert all(row['true_task'] == 2 for row in rows)
    assert all(row['crm_evaluated'] == (tau > 0) for row in rows)
    assert 100 * sum(row['final_class'] == row['target'] for row in rows) / 3 == pytest.approx(after['Acc@1'])


def test_crm_records_actual_retained_adapter(tmp_path):
    writer = TraceWriter(tmp_path / 'trace.jsonl', dict(samples=2, eval_task=1))
    writer.pending = [dict(proposal_after=1), dict(proposal_after=1)]
    writer.crm(torch.tensor([[0., 0., 3., 0., 0., 0.], [0., 0., 0., 3., 0., 0.]]),
               torch.tensor([0, 1]), torch.tensor([0, 1]), torch.tensor([2, 2]))
    assert [row['final_adapter'] for row in writer.pending] == [1, 2]
    assert [row['crm_used_alternative'] for row in writer.pending] == [False, True]


def test_tied_prediction_matches_evaluator_accuracy_rule():
    logits = torch.zeros(3, 8)
    assert TraceWriter.classes(logits) == logits.topk(5, 1, True, True).indices[:, 0].tolist()


def test_trace_rejects_nonsequential_loader(tmp_path):
    args = SimpleNamespace(prediction_trace_dir=str(tmp_path), eval=True, world_size=1,
                           num_tasks=1, task_routing_mode='class')
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.zeros(2, 1)),
                                       shuffle=True)
    with pytest.raises(ValueError, match='sequential'):
        start_trace(args, loader, 0, 1, [[0]])


def test_main_parser_accepts_opt_in_trace_flag(monkeypatch, tmp_path):
    import main
    monkeypatch.setattr(sys, 'argv', ['main.py', 'imr_lora', '--eval',
                                     '--prediction_trace_dir', str(tmp_path)])
    args = main.get_args()
    assert args.eval and args.prediction_trace_dir == str(tmp_path)


def test_main_rejects_tracing_before_training_can_start(tmp_path):
    import main
    args = SimpleNamespace(prediction_trace_dir=str(tmp_path), eval=False)
    with pytest.raises(ValueError, match='evaluation-only'):
        main.main(args)
