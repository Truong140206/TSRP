"""Independent scalar reference for the actual evaluator, using synthetic scores.

No pretrained weights, real dataset, engine routing/CRM/gate helper, or trace
writer is used to calculate the expected decisions. This checks implementation
mechanics, not whether routing is beneficial on real images.
"""
import json
import math
from types import SimpleNamespace
import pytest
import torch

from engines import hrm_lora_wtp_and_tap_engine as engine
from peft.lora.hide_lora import HideLoraPool
from tools.compare_prediction_traces import load_task


def argmax(values):
    return max(range(len(values)), key=values.__getitem__)


def standardize(values):
    mean = sum(values) / len(values)
    std = max(1e-6, math.sqrt(sum((x - mean) ** 2 for x in values) / len(values)))
    return [(x - mean) / std for x in values]


def softmax(values):
    exp = [math.exp(x - max(values)) for x in values]
    return [x / sum(exp) for x in exp]


def reference(original, bank, rp, weight, mode, tau):
    # Four seen tasks of five classes, plus five unseen classes with large scores.
    original, bank, rp = original[:20], [row[:20] for row in bank], rp[:20]
    tii = argmax(original) // 5
    primary = bank[tii]
    top = sorted(range(20), key=primary.__getitem__, reverse=True)
    before, alternative = top[0] // 5, top[1] // 5
    primary_tasks = standardize([max(primary[5 * t:5 * t + 5]) for t in range(4)])
    rp_tasks = standardize([max(rp[5 * t:5 * t + 5]) for t in range(4)])
    proposal = argmax([weight * p + (1 - weight) * r for p, r in zip(primary_tasks, rp_tasks)])
    drm = bank[proposal]
    probabilities = softmax(primary)
    if mode == 'msp':
        evaluate_crm = max(probabilities) < tau
    else:
        entropy = -sum((p ** .1) * ((1 - p) ** .1) for p in probabilities)
        evaluate_crm = entropy < tau
    def energy(row):
        peak = max(row)
        return peak + .1 * math.log(sum(math.exp((x - peak) / .1) for x in row))
    replace = evaluate_crm and energy(bank[alternative]) > energy(drm)
    adapter = alternative if replace else proposal
    crm = bank[adapter]
    top_logits = sorted(crm, reverse=True)[:2]
    ratio = math.exp(top_logits[1] - top_logits[0])
    margin = (1 - ratio) / (1 + ratio)
    share = .5 * (1 - margin)
    fused = [(1 - share) * a + share * b for a, b in zip(standardize(crm), standardize(rp))]
    return dict(tii_task=tii, proposal_before=before, proposal_after=proposal,
                drm_class=argmax(drm), crm_class=argmax(crm), final_class=argmax(fused),
                final_adapter=adapter, crm_evaluated=evaluate_crm,
                crm_used_alternative=replace), alternative


class IndexedOriginal(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores

    def forward(self, inputs):
        return dict(logits=self.scores[inputs[:, 0].long()].clone(), pre_logits=inputs)


class IndexedAdapters(torch.nn.Module):
    def __init__(self, scores):
        super().__init__()
        self.scores = scores
        self.calls = []

    def forward(self, inputs, task_id=0, **kwargs):
        ids = inputs[:, 0].long()
        task_ids = torch.full_like(ids, task_id) if isinstance(task_id, int) else task_id
        self.calls.append((ids.tolist(), task_ids.tolist()))
        return dict(logits=self.scores[ids, task_ids].clone(), pre_logits=inputs)


@pytest.mark.parametrize('weight', [1., .7])
@pytest.mark.parametrize('mode,tau', [('gen', -10.), ('msp', .6)])
@pytest.mark.parametrize('batch_size', [1, 7, 32])
def test_actual_evaluator_matches_independent_per_image_reference(
        monkeypatch, tmp_path, weight, mode, tau, batch_size):
    generator = torch.Generator().manual_seed(104)
    n = 67
    original = torch.randn(n, 25, generator=generator)
    bank = torch.randn(n, 4, 25, generator=generator)
    bank *= torch.tensor([.2, 1., 4., 10.])[torch.arange(n) % 4, None, None]
    rp = torch.randn(n, 25, generator=generator)
    # An unseen class must not win TII, DRM, CRM, or the final class gate.
    original[:, 20:] = 1000
    bank[:, :, 20:] = 1000
    rp[:, 20:] = -1000
    args = SimpleNamespace(eval=True, world_size=1, num_tasks=4, dataset='synthetic', seed=42,
                           print_freq=1000, train_mask=True, task_inc=False, nb_classes=25,
                           En=mode, tau=tau, task_routing_mode='class',
                           rp_head=True, rp_route_fusion_drm=True, rp_feature_source='lora',
                           rp_lora_task=0, rp_route_fusion_weight=weight,
                           rp_class_fusion_weight=.5, rp_class_fusion_gate='margin',
                           rp_class_fusion_min_tasks=1, rp_fusion_ramp=0.,
                           classifier_union_audit=False, prediction_trace_dir=str(tmp_path))
    monkeypatch.setattr(engine, 'args_ref', [None])
    monkeypatch.setattr(engine, 'gate_stats', {})
    monkeypatch.setattr(engine, 'rp_head_predict', lambda features, *a: rp[features[:, 0].long()].clone())
    (tmp_path / 'run.json').write_text(json.dumps(dict(dataset_name='synthetic', seed=42)))
    # Permute input identity so an accidental position-based lookup is detectable.
    order = torch.randperm(n, generator=generator)
    targets = order % 5
    inputs = order[:, None].float()
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(inputs, targets),
                                        batch_size=batch_size, shuffle=False)
    model = IndexedAdapters(bank)
    stats = engine.evaluate(model, IndexedOriginal(original), loader, torch.device('cpu'),
                            i=0, task_id=3, class_mask=[list(range(t * 5, t * 5 + 5)) for t in range(5)],
                            target_task_map={c: c // 5 for c in range(25)}, args=args)
    _, rows = load_task(tmp_path / 'task_00.jsonl')
    expected = [reference(original[i].tolist(), bank[i].tolist(), rp[i].tolist(), weight, mode, tau)
                for i in order]
    for row, (decision, _) in zip(rows, expected):
        assert {key: row[key] for key in decision} == decision
    assert stats['Acc@1'] == pytest.approx(
        100 * sum(e[0]['final_class'] == int(y) for e, y in zip(expected, targets)) / n, abs=1e-5)
    assert stats['Acc@task'] == pytest.approx(
        100 * sum(e[0]['proposal_after'] == 0 for e in expected) / n, abs=1e-5)
    # Check every batched adapter call, including empty DRM/CRM subsets.
    expected_calls = []
    for start in range(0, n, batch_size):
        ids = order[start:start + batch_size].tolist()
        batch = expected[start:start + batch_size]
        expected_calls.extend([(ids, [e['tii_task'] for e, _ in batch]), (ids, [0] * len(ids))])
        drm = [(i, e['proposal_after']) for i, (e, _) in zip(ids, batch)
               if e['proposal_after'] != e['tii_task']]
        crm = [(i, alternative) for i, (e, alternative) in zip(ids, batch) if e['crm_evaluated']]
        for subset in (drm, crm):
            expected_calls.append(([i for i, _ in subset], [task for _, task in subset]))
    assert model.calls == expected_calls
    assert {e['crm_evaluated'] for e, _ in expected} == {False, True}
    assert {e['crm_used_alternative'] for e, _ in expected} == {False, True}
    assert any(alternative == e['proposal_after'] for e, alternative in expected)


@pytest.mark.parametrize('depth', [0, 1])
def test_real_lora_pool_routes_each_sample_like_independent_matrix_products(depth):
    torch.manual_seed(17)
    pool = HideLoraPool(pool_size=4, depth=2, dim=6, rank=2, lora_alpha=3)
    with torch.no_grad():
        for name in ('k_lora_A', 'k_lora_B', 'v_lora_A', 'v_lora_B'):
            getattr(pool, name).copy_(torch.randn_like(getattr(pool, name)))
    inputs = torch.randn(9, 3, 6)
    tasks = torch.tensor([3, 0, 2, 1, 3, 2, 0, 1, 2])
    expected = []
    for sample, task in zip(inputs, tasks):
        key = (sample @ pool.k_lora_A[task, depth]) @ pool.k_lora_B[task, depth]
        value = (sample @ pool.v_lora_A[task, depth]) @ pool.v_lora_B[task, depth]
        expected.append(torch.cat([torch.zeros_like(sample), key, value], dim=-1) * 1.5)
    expected = torch.stack(expected)
    actual = pool(inputs, task_id=tasks, depth_id=depth)['lora_value']
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    for indices in (torch.tensor([6, 1, 8, 2, 5, 0, 4, 3, 7]), torch.tensor([8, 2, 0])):
        selected = pool(inputs[indices], task_id=tasks[indices], depth_id=depth)['lora_value']
        torch.testing.assert_close(selected, expected[indices], atol=1e-5, rtol=1e-5)
    for i, task in enumerate(tasks.tolist()):
        scalar = pool(inputs[i:i + 1], task_id=task, depth_id=depth, train=True)['lora_value']
        torch.testing.assert_close(scalar, expected[i:i + 1], atol=1e-5, rtol=1e-5)


def test_correct_task_is_not_a_class_constraint_in_actual_evaluator(monkeypatch, tmp_path):
    # Deliberately constructed counterexample, NOT an explanation of real CIFAR
    # samples: the task-0 adapter has a confident within-task class error. The
    # task-1 adapter and RP head classify this task-0 sample correctly.
    original = torch.zeros(1, 20)
    original[0, 10] = 10  # TII chooses adapter 2 for both arms.
    bank = torch.zeros(1, 4, 20)
    bank[0, 2, 5], bank[0, 2, 0] = 50, 49.9
    bank[0, 1, 0] = 10
    bank[0, 0, 1] = 10
    rp = torch.zeros(1, 20)
    rp[0, 0] = 10
    monkeypatch.setattr(engine, 'rp_head_predict', lambda features, *a: rp.clone())
    monkeypatch.setattr(engine, 'args_ref', [None])
    monkeypatch.setattr(engine, 'gate_stats', {})
    decisions = {}
    for arm, weight in (('class_gate', 1.), ('full', .7)):
        root = tmp_path / arm
        root.mkdir()
        (root / 'run.json').write_text(json.dumps(dict(dataset_name='synthetic', seed=42)))
        args = SimpleNamespace(eval=True, world_size=1, num_tasks=4, dataset='synthetic', seed=42,
                               print_freq=100, train_mask=True, task_inc=False, nb_classes=20,
                               En='gen', tau=-10., task_routing_mode='class',
                               rp_head=True, rp_route_fusion_drm=True, rp_feature_source='lora',
                               rp_lora_task=0, rp_route_fusion_weight=weight,
                               rp_class_fusion_weight=.5, rp_class_fusion_gate='margin',
                               rp_class_fusion_min_tasks=1, rp_fusion_ramp=0.,
                               classifier_union_audit=False, prediction_trace_dir=str(root))
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.zeros(1, 1), torch.tensor([0])))
        engine.evaluate(IndexedAdapters(bank), IndexedOriginal(original), loader,
                        torch.device('cpu'), i=0, task_id=3,
                        class_mask=[list(range(5 * t, 5 * t + 5)) for t in range(4)],
                        target_task_map={c: c // 5 for c in range(20)}, args=args)
        _, rows = load_task(root / 'task_00.jsonl')
        expected, _ = reference(original[0].tolist(), bank[0].tolist(), rp[0].tolist(),
                                weight, 'gen', -10.)
        assert {k: rows[0][k] for k in expected} == expected
        decisions[arm] = rows[0]
    assert decisions['class_gate']['final_adapter'] == 1
    assert decisions['class_gate']['final_class'] == 0
    assert decisions['full']['final_adapter'] == 0
    assert decisions['full']['final_class'] == 1
    assert decisions['full']['rp_class'] == decisions['class_gate']['rp_class'] == 0
    assert not decisions['full']['crm_evaluated']
