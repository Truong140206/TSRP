"""Same-batch invariants against real train_one_epoch; no dataset accuracy claim."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from engines import hrm_lora_wtp_and_tap_engine as engine
from tools import ctird_alignment_experiment as aligned
from tools import train_ctird_experiment as runner


def test_selection_uses_actual_old_ids_and_highest_or_lowest_explicitly():
    logits = torch.arange(15).float()[None]
    masks = [list(range(5, 10)), list(range(5)), list(range(10, 15))]
    assert aligned.select_teachers(logits, masks, 1, 5, 'online_highest').tolist() == [[0]]
    assert aligned.select_teachers(logits, masks, 2, 1, 'online_highest').tolist() == [[0]]
    assert aligned.select_teachers(logits, masks, 2, 1, 'online_lowest').tolist() == [[1]]
    # Retain class-based selection: both best classes may map to the same task.
    assert aligned.select_teachers(logits, masks, 2, 5, 'online_highest').tolist() == [[0, 0]]
    for bad in ([[], [], []], [[5, 5], [0], [1]], [[99], [0], [1]]):
        with pytest.raises(ValueError):
            aligned.select_teachers(logits, bad, 1, 1, 'online_highest')
    logits[0, 5] = float('-inf')
    with pytest.raises(ValueError, match='finite'):
        aligned.select_teachers(logits, masks, 1, 1, 'online_highest')


class Student(torch.nn.Module):
    def __init__(self):
        super().__init__()
        first = torch.nn.Embedding(12, 6)
        self.adapters = torch.nn.ModuleList([copy.deepcopy(first) for _ in range(3)])
        self.head = torch.nn.Linear(6, 15)
        self.dropout = torch.nn.Dropout(.5)
        self.calls = []

    def forward(self, inputs, task_id=None, train=False):
        ids = inputs[:, 0].long()
        self.calls.append((inputs.detach().clone(), self.training, torch.is_grad_enabled(), train))
        if isinstance(task_id, int):
            features = self.adapters[task_id](ids)
        else:
            features = torch.stack([self.adapters[t](i) for t, i in zip(task_id, ids)])
        return dict(features=features, logits=self.head(features))


class Router(torch.nn.Module):
    def forward(self, inputs):
        return dict(logits=torch.arange(15).float().expand(len(inputs), 15))


@pytest.mark.parametrize('mode', aligned.MODES)
def test_real_training_uses_same_current_batch_no_teacher_gradients(monkeypatch, mode):
    torch.manual_seed(31)
    model, router = Student(), Router()
    dataset = torch.utils.data.TensorDataset(torch.arange(12).float()[:, None], 10 + torch.arange(12) % 5)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)
    args = SimpleNamespace(eval=False, distributed=False, train_mask=True, nb_classes=15,
                           K=1, epochs=2, print_freq=1000, con=.2)
    masks = [list(range(t*5, t*5+5)) for t in range(3)]
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    original_kl, losses = F.kl_div, []
    def observe(logs, targets, **kwargs):
        assert not targets.requires_grad
        loss = original_kl(logs, targets, **kwargs)
        losses.append(float(loss.detach()))
        return loss
    monkeypatch.setattr(F, 'kl_div', observe)
    old_epoch, old_cache = engine.train_one_epoch, engine.get_old_features
    with aligned.installed(engine, mode):
        assert engine.get_old_features() is None
        for epoch in range(2):
            engine.train_one_epoch(model, router, torch.nn.CrossEntropyLoss(), loader,
                torch.optim.SGD(model.parameters(), lr=0), torch.device('cpu'), epoch,
                max_norm=2., task_id=2, class_mask=masks,
                target_task_map={i: i//5 for i in range(15)}, args=args)
    assert (engine.train_one_epoch, engine.get_old_features) == (old_epoch, old_cache)
    assert len(losses) == 6 and max(map(abs, losses)) < 1e-6
    assert len(model.calls) == 12
    for student, teacher in zip(model.calls[::2], model.calls[1::2]):
        torch.testing.assert_close(student[0], teacher[0], atol=0, rtol=0)
        assert student[1:] == (True, True, True)
        assert teacher[1:] == (False, False, False)
    assert all(p.grad is None for adapter in model.adapters[:2] for p in adapter.parameters())
    assert model.adapters[2].weight.grad is not None
    assert model.training


def test_modes_and_wrappers_restore_on_failure():
    model = Student().train()
    model.dropout.eval()
    before = [m.training for m in model.modules()]
    with pytest.raises(RuntimeError):
        with aligned.teacher_mode(model):
            assert not any(m.training for m in model.modules())
            raise RuntimeError('test')
    assert [m.training for m in model.modules()] == before
    old = (engine.train_one_epoch, engine.get_old_features)
    with pytest.raises(RuntimeError):
        with aligned.installed(engine, 'online_highest'):
            raise RuntimeError('test')
    assert (engine.train_one_epoch, engine.get_old_features) == old


def test_preflight_rejects_existing_or_mismatched_inputs_without_writing(monkeypatch, tmp_path):
    import main as application
    import sys
    from tools import run_prediction_audit
    monkeypatch.setattr(sys, 'argv', ['main.py', 'ima_lora', '--dataset', 'Split-Imagenet-A',
        '--lora_type', 'hide', '--max_train_tasks', '2', '--trained_original_model', str(tmp_path/'tii'),
        '--data-path', str(tmp_path/'data'), '--output_dir', str(tmp_path/'ctird_online_highest_trial')])
    args = application.get_args()
    monkeypatch.setattr(run_prediction_audit, 'resolve_data', lambda a: tmp_path/'data')
    checkpoint_dir = tmp_path/'tii'/'checkpoint'
    checkpoint_dir.mkdir(parents=True)
    for t in (1, 2):
        torch.save(dict(args=vars(args), model={}), checkpoint_dir/f'task{t}_checkpoint.pth')
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    plan = runner.preflight(args, 'online_highest')
    assert plan['tasks'] == 2 and plan['fresh_training']
    assert {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()} == before
    assert not (tmp_path/'ctird_online_highest_trial').exists()
    args.seed += 1
    with pytest.raises(ValueError, match='configuration mismatch'):
        runner.preflight(args, 'online_highest')
    args.seed -= 1
    (tmp_path/'ctird_online_highest_trial').mkdir()
    with pytest.raises(ValueError, match='fresh directory'):
        runner.preflight(args, 'online_highest')
