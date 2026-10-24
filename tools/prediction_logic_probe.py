"""Experimental final-stage 2x2 gate/CRM comparison for both routing arms.

Reproduce historical Full through all stages, then reuse that fixed final RP
state/checkpoint for the other seven arms. Only final-stage results are claimed;
no invented retention curves. Defaults/paper engine files remain unchanged.
"""
import argparse
import copy
import json
from pathlib import Path
import random
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import fusion_logic_variants as variants
from tools import audit_prediction_stages as stages
from tools import compare_prediction_traces as compare
from tools.prediction_score_probe import check_configuration
from tools.run_prediction_audit import save_new


def check_arm_configuration(current, historical, arm):
    weight, gate, crm = variants.ARMS[arm]
    expected = dict(historical, rp_route_fusion_weight=weight)
    if arm not in ('full', 'class_gate'):
        expected.update(experiment_gate_zscore=gate, experiment_crm_refresh=crm)
    check_configuration(current, expected)


class HistoricalObserver:
    def __init__(self, writer, rows):
        self.writer, self.rows = writer, rows

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def end(self, logits):
        pending = self.writer.pending
        self.writer.end(logits)
        for row in pending:
            expected = {k: v for k, v in self.rows[row['index']].items() if k != 'type'}
            if row != expected:
                raise ValueError('Original-arm historical trace mismatch at task%d index%d' %
                                 (row['true_task'], row['index']))


def run_probe(history, destination, application_args):
    import numpy as np
    import torch
    import main as application
    import utils
    from engines import hrm_lora_wtp_and_tap_engine as engine
    from trainers import lora_trainer as trainer

    old_start, old_evaluate = engine.start_trace, trainer.evaluate_till_now
    old_argv = sys.argv
    finished = False
    closed = {arm: set() for arm in variants.ARMS}
    timings, final_stats = {}, {}

    def start(args, loader, task, stage, masks):
        writer = old_start(args, loader, task, stage, masks)
        if writer is None:
            return None
        arm = Path(args.prediction_trace_dir).name
        if arm not in variants.ARMS:
            raise ValueError('Unknown experiment arm')
        old_header, old_rows = compare.load_task(history / 'full' / ('task_%02d.jsonl' % task))
        for key in ('dataset', 'seed', 'stage', 'eval_task', 'samples', 'class_mask'):
            if writer.header[key] != old_header[key]:
                raise ValueError('Historical header mismatch: ' + key)
        check_arm_configuration(writer.header['configuration'], old_header['configuration'], arm)
        observer = writer
        if arm in ('full', 'class_gate'):
            _, rows = compare.load_task(history / arm / ('task_%02d.jsonl' % task))
            observer = HistoricalObserver(writer, rows)
        close = observer.close
        def checked_close(*metrics):
            close(*metrics)
            if task in closed[arm]:
                raise ValueError('Repeated final-stage task')
            closed[arm].add(task)
        observer.close = checked_close
        return observer

    def evaluate(model, original_model, loaders, device, task_id, masks, task_map, matrix, args):
        nonlocal finished
        began = time.perf_counter()
        result = old_evaluate(model, original_model, loaders, device, task_id, masks, task_map, matrix, args)
        if task_id != args.num_tasks - 1:
            return result
        if finished:
            raise ValueError('Repeated final stage')
        timings['full'] = time.perf_counter() - began
        py_rng, np_rng, cpu_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None
        def restore_rng():
            random.setstate(py_rng)
            np.random.set_state(np_rng)
            torch.set_rng_state(cpu_rng)
            if cuda_rng is not None:
                torch.cuda.set_rng_state_all(cuda_rng)
        try:
            for arm, (weight, gate, crm) in variants.ARMS.items():
                if arm == 'full':
                    continue
                restore_rng()
                alternate = copy.copy(args)
                alternate.rp_route_fusion_weight = weight
                alternate.prediction_trace_dir = str(destination / arm)
                if arm != 'class_gate':
                    alternate.experiment_gate_zscore = gate
                    alternate.experiment_crm_refresh = crm
                print('LOGIC_ARM_START=' + arm + '; final stage only, no RP rebuild', flush=True)
                began = time.perf_counter()
                stats = [engine.evaluate(model, original_model, loaders[t]['val'], device, i=t,
                         task_id=task_id, class_mask=masks, target_task_map=task_map, args=alternate)
                         for t in range(args.num_tasks)]
                timings[arm] = time.perf_counter() - began
                final_stats[arm] = {key: statistics.mean(s[key] for s in stats)
                                   for key in ('Acc@1', 'Acc@5', 'Loss', 'Acc@task', 'GateShare')}
                print('LOGIC_ARM_COMPLETE=' + arm, flush=True)
        finally:
            restore_rng()
        finished = True
        return result

    try:
        sys.argv = ['main.py', *application_args]
        args = application.get_args()
        if (not args.eval or args.train_inference_task_only or args.num_tasks != 10
                or args.dataset != 'Split-Imagenet-A' or args.task_inc
                or args.rp_route_fusion_weight != .7
                or any(hasattr(args, flag) for flag in variants.FLAGS)):
            raise ValueError('Expected unchanged ImageNet-A Full evaluation configuration')
        if Path(args.prediction_trace_dir).resolve() != destination / 'full':
            raise ValueError('Unexpected experiment trace path')
        engine.start_trace, trainer.evaluate_till_now = start, evaluate
        with variants.installed(engine):
            application.main(args)
        if not finished or any(tasks != set(range(10)) for tasks in closed.values()):
            raise ValueError('Incomplete experimental final-stage coverage')
        save_new(destination / 'final_stats.json', final_stats)
        save_new(destination / 'timing.json', dict(final_pass_seconds=timings,
                 scope='Instrumented final-pass wall times, not method overhead. Full rebuild '
                       'and earlier stages are only included in the parent subprocess timer.'))
        print('LOGIC_HISTORICAL_ROWS_EXACT=PASS_BOTH_ARMS', flush=True)
    finally:
        engine.start_trace, trainer.evaluate_till_now = old_start, old_evaluate
        sys.argv = old_argv
        utils.cleanup_distributed()


def summarize(history, destination):
    """Validate every trace and report task-balanced, paired experimental changes."""
    headers, samples = {}, {}
    files = ['task_%02d.jsonl' % t for t in range(10)]
    old_headers, old_samples = {}, {}
    for arm in variants.ARMS:
        folder = destination / arm
        if sorted(p.name for p in folder.glob('task_*.jsonl')) != files:
            raise ValueError('Incomplete experiment traces: ' + arm)
        manifest = json.loads((folder / 'run.json').read_text(encoding='utf-8'))
        if manifest['variant'] != arm:
            raise ValueError('Wrong arm manifest')
        pairs = [compare.load_task(folder / name) for name in files]
        headers[arm], samples[arm] = [p[0] for p in pairs], [p[1] for p in pairs]
        for header, rows in pairs:
            stages.check_structure(header, rows)
            if header['run'] != manifest:
                raise ValueError('Trace/manifest provenance mismatch')
    for arm in ('full', 'class_gate'):
        pairs = [compare.load_task(history / arm / name) for name in files]
        old_headers[arm], old_samples[arm] = [p[0] for p in pairs], [p[1] for p in pairs]
        if samples[arm] != old_samples[arm]:
            raise ValueError('Original arm no longer reproduces historical rows: ' + arm)
    invariants = ('index', 'target', 'true_task', 'input_sha256', 'tii_task',
                  'proposal_before', 'initial_logits_sha256', 'rp_sha256', 'rp_class')
    report = {}
    for arm in variants.ARMS:
        control = 'full' if variants.ARMS[arm][0] == .7 else 'class_gate'
        for task, (head, rows) in enumerate(zip(headers[arm], samples[arm])):
            reference = old_headers['full'][task]
            for key in ('dataset', 'seed', 'stage', 'eval_task', 'samples', 'class_mask'):
                if head[key] != reference[key]:
                    raise ValueError('Arm header mismatch: ' + key)
            if head['run']['common'] != headers['full'][task]['run']['common']:
                raise ValueError('Arm provenance differs')
            check_arm_configuration(head['configuration'], reference['configuration'], arm)
            for row, before in zip(rows, old_samples[control][task]):
                if any(row[key] != before[key] for key in (*invariants, 'proposal_after', 'drm_class')):
                    raise ValueError('Experiment changed input/RP/routing/DRM evidence')
                if not variants.ARMS[arm][2] and any(row[k] != before[k] for k in
                        ('crm_class', 'final_adapter', 'crm_evaluated', 'crm_used_alternative')):
                    raise ValueError('Gate-only intervention changed CRM decisions')
        def paired_to(other, stage='final_class'):
            return stages.paired_stats([[(a[stage], b[stage], a['target'])
                   for a, b in zip(left, right)] for left, right in zip(samples[other], samples[arm])])
        against = {other: paired_to(other) for other in ('full', 'class_gate')}
        balanced = {}
        for stage in ('drm_class', 'crm_class', 'final_class'):
            balanced[stage] = statistics.mean(100 * sum(r[stage] == r['target'] for r in rows) / len(rows)
                                               for rows in samples[arm])
        report[arm] = dict(task_balanced_accuracy=balanced,
                          proposal_accuracy=statistics.mean(100 * sum(r['proposal_after'] == r['true_task']
                              for r in rows) / len(rows) for rows in samples[arm]),
                          final_adapter_accuracy=statistics.mean(100 * sum(r['final_adapter'] == r['true_task']
                              for r in rows) / len(rows) for rows in samples[arm]),
                          against_original=against)
    routing_effects = {}
    for label, left, right in (('original', 'class_gate', 'full'),
                              ('gate_z', 'class_gate_gate_z', 'full_gate_z'),
                              ('crm_refresh', 'class_gate_crm_refresh', 'full_crm_refresh'),
                              ('both', 'class_gate_both', 'full_both')):
        routing_effects[label] = stages.paired_stats([[(a['final_class'], b['final_class'], a['target'])
             for a, b in zip(l, r)] for l, r in zip(samples[left], samples[right])])
    return dict(arms=report, conditional_routing=routing_effects,
                limitations='Exploratory final-stage interventions on an already inspected test set. '
                'Not held-out validation, not new retention results, not evidence to select a new '
                'paper default. Controls must reproduce historical rows exactly. All eight arms '
                'share checkpoints, inputs and RP evidence; record losses as well as gains.')


if __name__ == '__main__':
    separator = sys.argv.index('--')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    options = parser.parse_args(sys.argv[1:separator])
    run_probe(options.history.resolve(), options.destination.resolve(), sys.argv[separator + 1:])
