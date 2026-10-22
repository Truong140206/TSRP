"""Opt-in score observation during an unchanged saved-checkpoint evaluation.

Launched by run_prediction_score_audit.py, not as a training entry point.
The full historical sequence is preserved. Gated class-only is evaluated once
at the final stage using the same already-built RP state. No score intervention
is applied to the model; numerical counterfactuals are report-only calculations.
"""
import argparse
import copy
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools.compare_prediction_traces import load_task
from tools.run_prediction_audit import save_new


def normalized(values):
    mean = math.fsum(values) / len(values)
    std = max(1e-6, math.sqrt(math.fsum((x - mean) ** 2 for x in values) / len(values)))
    return [(x - mean) / std for x in values], mean, std


def mix_prediction(primary, rp, share):
    mixed = [(1 - share) * p + share * r for p, r in zip(primary, rp)]
    best = max(mixed)
    ties = [i for i, x in enumerate(mixed) if x == best]
    return dict(predicted_class=ties[0] if len(ties) == 1 else None, tied_classes=ties)


def describe_scores(primary, rp, final, target, primary_class, final_class):
    if (not len(primary) == len(rp) == len(final) or len(primary) < 2
            or not all(math.isfinite(x) for values in (primary, rp, final) for x in values)):
        raise ValueError('Expected complete finite final-stage class scores')
    if not all(0 <= c < len(primary) for c in (target, primary_class, final_class)):
        raise ValueError('Invalid score class index')
    if primary[primary_class] != max(primary) or final[final_class] != max(final):
        raise ValueError('Observed classes disagree with captured score vectors')
    top2 = sorted(primary, reverse=True)[:2]
    ratio = math.exp(top2[1] - top2[0])
    margin = (1 - ratio) / (1 + ratio)
    share = .5 * (1 - margin)
    pz, mean, std = normalized(primary)
    rz, _, _ = normalized(rp)
    reconstructed = [((1 - share) * p + share * r) * std + mean for p, r in zip(pz, rz)]
    error = max(abs(a - b) for a, b in zip(final, reconstructed))
    if error > 1e-4 + 1e-5 * max(map(abs, final)):
        raise ValueError('Captured final logits do not reproduce the paper margin fusion')
    primary_gap = pz[target] - pz[primary_class]
    rp_gap = rz[target] - rz[primary_class]
    return dict(primary_class=primary_class, final_class=final_class,
                rp_class=rp.index(max(rp)), margin=margin, effective_rp_weight=share,
                primary_target_logit=primary[target], primary_winner_logit=primary[primary_class],
                target_vs_primary_winner=dict(primary_z_gap=primary_gap, rp_z_gap=rp_gap,
                    weighted_primary_gap=(1 - share) * primary_gap,
                    weighted_rp_gap=share * rp_gap,
                    combined_z_gap=(1 - share) * primary_gap + share * rp_gap),
                reconstruction_max_abs_error=error,
                primary_logits=primary, rp_scores=rp, final_logits=final)


def check_configuration(current, historical):
    def recorded_form(config):
        # TraceWriter serializes tuples as JSON arrays. Compare that same
        # representation on both sides; preserve every value and sequence order.
        return json.loads(json.dumps({k: v for k, v in config.items()
                                     if k != 'prediction_trace_dir'}, allow_nan=False))
    current, historical = recorded_form(current), recorded_form(historical)
    if current != historical:
        differing = sorted(k for k in current.keys() | historical.keys()
                           if k not in current or k not in historical or current[k] != historical[k])
        raise ValueError('Replay configuration differs: ' + ', '.join(differing))
    required = dict(eval=True, num_tasks=10, rp_class_fusion_gate='margin',
                    rp_class_fusion_weight=.5, rp_class_fusion_sharpen=1., rp_fusion_ramp=0.,
                    rp_class_fusion_min_tasks=1, rp_route_fusion_drm=True)
    if any(current.get(k) != v for k, v in required.items()):
        raise ValueError('Score probe supports only the checked paper configuration')


class ScoreObserver:
    """Delegate every original observer call; copy only selected score rows."""
    def __init__(self, base, historical, selected, collected):
        self.base, self.historical = base, historical
        self.selected, self.collected = selected, collected
        self.indices, self.scores = [], {}

    def begin(self, *args):
        self.base.begin(*args)
        self.indices = [i for i, r in enumerate(self.base.pending) if r['index'] in self.selected]
        self.scores = {i: {} for i in self.indices}

    def capture(self, key, values):
        if self.indices:
            rows = values[self.indices].detach().cpu().tolist()
            for i, row in zip(self.indices, rows):
                self.scores[i][key] = row

    def routing(self, proposal, rp):
        self.base.routing(proposal, rp)
        self.capture('rp', rp)

    def drm(self, logits):
        self.base.drm(logits)

    def crm(self, logits, indices, choices, alternatives):
        self.base.crm(logits, indices, choices, alternatives)
        self.capture('primary', logits)

    def end(self, logits):
        self.capture('final', logits)
        rows = self.base.pending
        self.base.end(logits)
        for i, row in enumerate(rows):
            expected = {k: v for k, v in self.historical[row['index']].items() if k != 'type'}
            if row != expected:
                differences = sorted(k for k in row.keys() | expected.keys() if row.get(k) != expected.get(k))
                raise ValueError('Historical sample mismatch at task%d index%d: %s' %
                                 (row['true_task'], row['index'], ','.join(differences)))
            if i in self.scores:
                scores = self.scores[i]
                details = describe_scores(scores['primary'], scores['rp'], scores['final'],
                                          row['target'], row['crm_class'], row['final_class'])
                self.collected.append(dict(sample=dict(row), scores=details))

    def close(self, *args):
        self.base.close(*args)
        if self.base.count != len(self.historical):
            raise ValueError('Historical trace coverage differs')


def run_probe(history, destination, application_args):
    import main as application
    import utils
    from engines import hrm_lora_wtp_and_tap_engine as engine
    from trainers import lora_trainer as trainer

    plan = json.loads((destination / 'plan.json').read_text(encoding='utf-8'))
    selected = {int(t): set(indices) for t, indices in plan['selected'].items()}
    observed = {arm: [] for arm in ('full', 'class_gate')}
    closed = {arm: set() for arm in observed}
    old_start, old_evaluate = engine.start_trace, trainer.evaluate_till_now
    saved_argv = sys.argv
    finished = False

    def start(args, loader, task, stage, masks):
        base = old_start(args, loader, task, stage, masks)
        if base is None:
            return None
        arm = Path(args.prediction_trace_dir).name
        header, rows = load_task(history / arm / ('task_%02d.jsonl' % task))
        for key in ('dataset', 'seed', 'stage', 'eval_task', 'samples', 'class_mask'):
            if base.header[key] != header[key]:
                raise ValueError('Historical header mismatch: ' + key)
        check_configuration(base.header['configuration'], header['configuration'])
        observer = ScoreObserver(base, rows, selected.get(task, set()), observed[arm])
        original_close = observer.close
        def close(*values):
            original_close(*values)
            if task in closed[arm]:
                raise ValueError('Repeated final-stage task')
            closed[arm].add(task)
        observer.close = close
        return observer

    def evaluate(model, original_model, loaders, device, task_id, masks, task_map, matrix, args):
        nonlocal finished
        result = old_evaluate(model, original_model, loaders, device, task_id, masks, task_map, matrix, args)
        if task_id == args.num_tasks - 1:
            if finished:
                raise ValueError('Repeated final stage')
            alternate = copy.copy(args)
            alternate.rp_route_fusion_weight = 1.0
            alternate.prediction_trace_dir = str(destination / 'class_gate')
            print('SCORE_PROBE_CLASS_GATE_FINAL_ONLY: reuse completed RP state; no accumulation', flush=True)
            # Do not invent retention metrics using Full's earlier-stage matrix.
            for task in range(args.num_tasks):
                engine.evaluate(model, original_model, loaders[task]['val'], device, i=task,
                                task_id=task_id, class_mask=masks, target_task_map=task_map, args=alternate)
            finished = True
        return result

    try:
        sys.argv = ['main.py', *application_args]
        args = application.get_args()
        if not args.eval or args.train_inference_task_only:
            raise ValueError('Score probe is evaluation-only')
        if Path(args.prediction_trace_dir).resolve() != destination / 'full':
            raise ValueError('Unexpected score output path')
        engine.start_trace, trainer.evaluate_till_now = start, evaluate
        application.main(args)
        if not finished or any(tasks != set(range(10)) for tasks in closed.values()):
            raise ValueError('Incomplete score probe')
        for arm, records in observed.items():
            expected = {(t, i) for t, indices in selected.items() for i in indices}
            actual = [(r['sample']['true_task'], r['sample']['index']) for r in records]
            if len(actual) != len(set(actual)) or set(actual) != expected:
                raise ValueError('Incomplete selected-score coverage')
            save_new(destination / arm / 'scores.json', records)
        print('HISTORICAL_TRACE_ROWS_EXACT=PASS_BOTH_ARMS', flush=True)
    finally:
        engine.start_trace, trainer.evaluate_till_now = old_start, old_evaluate
        sys.argv = saved_argv
        utils.cleanup_distributed()


if __name__ == '__main__':
    separator = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    options = parser.parse_args(sys.argv[1:separator])
    run_probe(options.history.resolve(), options.destination.resolve(), sys.argv[separator + 1:])
