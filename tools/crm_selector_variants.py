"""Candidate-selection experiments, with unchanged eligibility and candidates.

These confidence proxies are not calibrated and carry no performance guarantee.
No labels are accepted by the selector. Original controls call the old evaluator.
"""
from contextlib import contextmanager
import inspect
from tools.fusion_logic_variants import checked_source

FLAGS = ('experiment_crm_zenergy', 'experiment_crm_fused_margin')
ARMS = dict(full=(.7, False, False), class_gate=(1., False, False),
            full_zenergy=(.7, True, False), class_gate_zenergy=(1., True, False),
            full_fused_margin=(.7, False, True), class_gate_fused_margin=(1., False, True))


def changes_crm(arm):
    return any(ARMS[arm][1:])


def candidate_scores(engine, candidates, rp, args, seen_tasks):
    import torch
    if candidates.ndim != 3 or candidates.shape[1] != 2:
        raise ValueError('Expected two CRM candidates')
    flat = candidates.flatten(0, 1).float()
    valid = torch.isfinite(flat)
    if (torch.isnan(flat).any() or torch.isposinf(flat).any()
            or (valid.sum(1) < 2).any()
            or not torch.equal(valid[::2], valid[1::2])):
        raise ValueError('CRM candidates must share at least two finite seen classes')
    mode = [bool(getattr(args, key, False)) for key in FLAGS]
    if sum(mode) != 1:
        raise ValueError('Select exactly one CRM criterion')
    if mode[0]:
        z = engine._standardize_valid(flat, valid).masked_fill(~valid, float('-inf'))
        score = .1 * torch.logsumexp(z / .1, dim=1)
    else:
        if rp is None or rp.shape != candidates[:, 0].shape:
            raise ValueError('Fused-margin selection requires matching RP scores')
        if (args.rp_class_fusion_gate != 'margin' or args.rp_fusion_ramp != 0.
                or args.rp_class_fusion_min_tasks != 1 or args.rp_class_fusion_sharpen != 1.
                or args.rp_class_fusion_weight != .5):
            raise ValueError('Fused-margin selector requires the fixed paper class fusion')
        repeated = rp[:, None].expand(-1, 2, -1).flatten(0, 1)
        # Compute the existing fusion, then undo its positive affine rescaling.
        # This evaluates the same eventual class decision for each candidate.
        # The paper evaluator sets args_ref only later, at the final fusion.
        # Do not inherit a prior batch/arm's config or alter gate telemetry.
        previous_args, previous_stats = engine.args_ref[0], dict(engine.gate_stats)
        try:
            engine.args_ref[0] = args
            mixed = engine.fuse_class_scores(flat, repeated, .5, seen_tasks=seen_tasks)
        finally:
            engine.args_ref[0] = previous_args
            engine.gate_stats.clear()
            engine.gate_stats.update(previous_stats)
        mean, std = engine._valid_moments(flat, valid)
        mixed_z = torch.where(valid, (mixed - mean) / std, torch.zeros_like(mixed))
        score = engine._top2_margin(mixed_z, valid).squeeze(1)
    return score.reshape(-1, 2)


@contextmanager
def installed(engine):
    checked_source(engine.__file__)
    original = engine.evaluate
    anchor = '                    entropy = (0.1*torch.logsumexp( error_output/ 0.1, dim=2))\n'
    source = inspect.getsource(original)
    if source.count(anchor) != 1:
        raise ValueError('CRM selector insertion point changed')
    replacement = ('                    entropy = experiment_candidate_scores(\n'
        '                        error_output, fusion_rp_scores[filtered_index_tensor],\n'
        '                        args, task_id + 1)\n')
    namespace = dict(engine.__dict__)
    namespace['experiment_candidate_scores'] = lambda *a: candidate_scores(engine, *a)
    exec(compile(source.replace(anchor, replacement, 1), '<explicit-crm-selector>', 'exec'), namespace)
    modified = namespace['evaluate']
    signature = inspect.signature(original)

    def dispatch(*args, **kwargs):
        config = signature.bind(*args, **kwargs).arguments['args']
        if any(getattr(config, key, False) for key in FLAGS):
            if (not config.eval or config.task_inc or not config.rp_route_fusion_drm
                    or getattr(config, 'task_routing_mode', 'class') != 'class'):
                raise ValueError('CRM intervention requires task-agnostic fusion evaluation')
            return modified(*args, **kwargs)
        return original(*args, **kwargs)
    engine.evaluate = dispatch
    try:
        yield
    finally:
        engine.evaluate = original
