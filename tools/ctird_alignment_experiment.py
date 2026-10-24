"""Opt-in same-batch teacher targets. Normal training code is never rewritten.

online_lowest fixes alignment while retaining ascending score selection;
online_highest also changes it to descending scores. Both use actual old-class
IDs. K is still capped by old-task count, with class-based selection (duplicates
after mapping to tasks are deliberately not deduplicated).
"""
from contextlib import contextmanager
import inspect

from tools.fusion_logic_variants import checked_source

MODES = ('online_lowest', 'online_highest')


def select_teachers(logits, masks, task_id, top_k, mode):
    import torch
    if mode not in MODES or not 0 < task_id < len(masks) or top_k < 1:
        raise ValueError('Invalid CTIRD mode/task/K')
    classes = [int(c) for task in masks[:task_id] for c in task]
    if (not classes or len(set(classes)) != len(classes)
            or min(classes) < 0 or max(classes) >= logits.shape[1]):
        raise ValueError('Invalid old-class masks')
    index = torch.tensor(classes, dtype=torch.long, device=logits.device)
    old = logits.index_select(1, index)
    if not torch.isfinite(old).all():
        raise ValueError('Old-class teacher scores must be finite')
    k = min(task_id, top_k)
    selected = old.topk(k, dim=1, largest=mode == 'online_highest').indices
    tasks = torch.tensor([t for t in range(task_id) for _ in masks[t]], device=logits.device)
    return tasks[selected]


@contextmanager
def teacher_mode(model):
    # Preserve even deliberately mixed module modes, including on exceptions.
    modes = [(module, module.training) for module in model.modules()]
    try:
        model.eval()
        yield
    finally:
        for module, training in modes:
            module.training = training


def same_batch_targets(model, inputs, teacher_logits, masks, task_id, top_k, mode):
    import torch
    import torch.nn.functional as F
    selected = select_teachers(teacher_logits.detach(), masks, task_id, top_k, mode)
    targets = []
    with torch.no_grad(), teacher_mode(model):
        for rank in range(selected.shape[1]):
            features = model(inputs, task_id=selected[:, rank])['features']
            features = F.normalize(features, p=2, dim=1)
            # Retain the historical relation definition, KL direction and sum.
            relation = (features @ features.T).exp()
            targets.append((relation / relation.sum(1, keepdim=True)).detach())
    return targets


@contextmanager
def installed(engine, mode):
    if mode not in MODES:
        raise ValueError('Explicit CTIRD experiment mode required')
    checked_source(engine.__file__)
    old_epoch, old_cache = engine.train_one_epoch, engine.get_old_features
    source = inspect.getsource(old_epoch)
    start = source.index('        top_indices=None\n')
    end = source.index('        if args.train_mask and class_mask is not None:\n', start)
    replacement = ('        if task_id > 0:\n'
        '            robust_logits = experiment_same_batch_targets(\n'
        '                model, input, temp, class_mask, task_id, args.K)\n'
        '        \n')
    source = source[:start] + replacement + source[end:]
    namespace = dict(engine.__dict__)
    namespace['experiment_same_batch_targets'] = lambda *a: same_batch_targets(*a, mode)
    exec(compile(source, '<explicit-same-batch-ctird>', 'exec'), namespace)
    online_epoch = namespace['train_one_epoch']
    signature = inspect.signature(old_epoch)

    def dispatch(*args, **kwargs):
        bound = signature.bind(*args, **kwargs).arguments
        config = bound.get('args')
        if config is None or getattr(config, 'eval', False) or not config.train_mask:
            raise ValueError('CTIRD intervention requires masked training')
        if bound.get('old_features') is not None:
            raise ValueError('Positional teacher caches are forbidden in this experiment')
        return online_epoch(*args, **kwargs)

    engine.train_one_epoch, engine.get_old_features = dispatch, lambda *a, **k: None
    try:
        yield
    finally:
        engine.train_one_epoch, engine.get_old_features = old_epoch, old_cache
