"""Explicit research interventions, installed only inside the experiment process.

No paper engine/config file is edited. The CRM variant is compiled from the
checked evaluator with exactly one inserted decision block; the paper default
always calls the original function. These hypotheses are not bug-fix claims.
"""
from contextlib import contextmanager
import hashlib
import inspect
from pathlib import Path


ENGINE_LF_SHA256 = 'f48523263f53a3447cf683d3824e474287c7dd8d2d442f757684a31e2490bbd7'
FLAGS = ('experiment_gate_zscore', 'experiment_crm_refresh')
ARMS = {
    'full': (.7, False, False),
    'class_gate': (1., False, False),
    'full_gate_z': (.7, True, False),
    'class_gate_gate_z': (1., True, False),
    'full_crm_refresh': (.7, False, True),
    'class_gate_crm_refresh': (1., False, True),
    'full_both': (.7, True, True),
    'class_gate_both': (1., True, True),
}
ANCHOR = '            corr_id=None\n            re_id = None\n'
INSERT = (
    '            if task_id > 0:\n'
    '                filtered_index_tensor, top5_id = experiment_refresh_crm(\n'
    '                    logits, prompt_id, class_mask, task_id + 1, args)\n'
)


def checked_source(path=None):
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'engines' / 'hrm_lora_wtp_and_tap_engine.py'
    source = Path(path).read_text(encoding='utf-8')
    if hashlib.sha256(source.encode()).hexdigest() != ENGINE_LF_SHA256:
        raise ValueError('Experimental evaluator requires the exact checked paper engine')
    if source.count(ANCHOR) != 1:
        raise ValueError('CRM insertion point changed')
    return source


def refresh_crm(engine, logits, proposal, class_mask, seen, args):
    """Recompute eligibility and a distinct alternative from POST-DRM logits.

Keep the original confidence function/threshold and CRM energy comparison.
The alternative is the highest-scoring class's task outside the selected task;
this avoids spending the alternative forward on the current adapter again.
No label/true task enters this decision.
"""
    import torch
    if seen < 2 or getattr(args, 'task_routing_mode', 'class') != 'class':
        raise ValueError('CRM refresh requires at least two tasks and class routing')
    probs = torch.softmax(logits, dim=1)
    if args.En == 'msp':
        indices = torch.where(probs.max(dim=1).values < args.tau)[0]
    elif args.En == 'gen':
        entropy = engine.generalized_entropy(probs.detach().cpu().numpy(), .1, 20)
        indices = torch.where(torch.as_tensor(entropy, device=logits.device) < args.tau)[0]
    else:
        raise ValueError('Unsupported CRM confidence function')
    scores = engine._task_scores_from_class_scores(logits, class_mask, seen, logits.device)
    scores.scatter_(1, proposal[:, None], float('-inf'))
    if not torch.isfinite(scores.max(dim=1).values).all():
        raise ValueError('No valid alternative task')
    alternative = scores.argmax(dim=1)
    return indices, torch.stack((proposal, alternative), dim=1)


@contextmanager
def installed(engine):
    """Scoped opt-in execution; restore the engine even when evaluation fails."""
    checked_source(engine.__file__)
    old_evaluate, old_gate = engine.evaluate, engine._fusion_gate
    source = inspect.getsource(old_evaluate)
    if source.count(ANCHOR) != 1:
        raise ValueError('Expected one CRM block in evaluate')
    modified = source.replace(ANCHOR, INSERT + ANCHOR, 1)
    namespace = dict(engine.__dict__)
    namespace['experiment_refresh_crm'] = lambda *args: refresh_crm(engine, *args)
    exec(compile(modified, '<explicit-crm-refresh-experiment>', 'exec'), namespace)
    crm_evaluate = namespace['evaluate']

    def dispatch(*args, **kwargs):
        config = kwargs.get('args', args[8] if len(args) > 8 else None)
        if getattr(config, 'experiment_crm_refresh', False):
            if not getattr(config, 'eval', False) or getattr(config, 'task_inc', False):
                raise ValueError('Experimental CRM is evaluation-only and task-agnostic')
            return crm_evaluate(*args, **kwargs)
        return old_evaluate(*args, **kwargs)

    def gate(routed_logits, valid, mode, rp_scores=None):
        if getattr(engine.args_ref[0], 'experiment_gate_zscore', False):
            if mode != 'margin':
                raise ValueError('Z-score gate experiment requires margin mode')
            routed_logits = engine._standardize_valid(routed_logits.float(), valid)
        return old_gate(routed_logits, valid, mode, rp_scores=rp_scores)

    engine.evaluate, engine._fusion_gate = dispatch, gate
    try:
        yield
    finally:
        engine.evaluate, engine._fusion_gate = old_evaluate, old_gate
