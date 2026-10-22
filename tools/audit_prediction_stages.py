"""Read completed prediction traces on CPU; never rerun or modify the method.

Stage paths locate observed correctness changes, not independent causal effects.
The historical traces contain decisions, not the logits/margins needed to assign
a particular change to calibration or to the strength of the margin gate.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import compare_prediction_traces as compare


ARMS = ('class_gate', 'full')
STAGES = ('drm_class', 'crm_class', 'final_class')
DETAIL_FIELDS = ('proposal_before', 'proposal_after', 'final_adapter',
                 'crm_evaluated', 'crm_used_alternative', 'rp_class', *STAGES)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def check_summary(expected, stored, location='summary'):
    """Compare all reconstructed fields; allow runner-only completion metadata."""
    if isinstance(expected, dict):
        if not isinstance(stored, dict) or not expected.keys() <= stored.keys():
            raise ValueError('Stored summary mismatch: ' + location)
        for key, value in expected.items():
            check_summary(value, stored[key], location + '.' + key)
    elif isinstance(expected, list):
        if not isinstance(stored, list) or len(expected) != len(stored):
            raise ValueError('Stored summary mismatch: ' + location)
        for i, (a, b) in enumerate(zip(expected, stored)):
            check_summary(a, b, location + '[%d]' % i)
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if (type(stored) not in (int, float) or not math.isfinite(stored)
                or abs(expected - stored) > 1e-9):
            raise ValueError('Stored summary mismatch: ' + location)
    elif type(expected) is not type(stored) or expected != stored:
        raise ValueError('Stored summary mismatch: ' + location)


def check_structure(header, rows):
    masks = header['class_mask'][:header['stage']]
    seen = [c for mask in masks for c in mask]
    if (len(masks) != header['stage'] or any(not mask for mask in masks)
            or len(seen) != len(set(seen))):
        raise ValueError('Class masks must be nonempty and disjoint')
    for row in rows:
        if row['rp_class'] not in seen:
            raise ValueError('Invalid RP class')
        if any(type(row[k]) is not bool for k in ('crm_evaluated', 'crm_used_alternative')):
            raise ValueError('CRM flags must be Boolean')
        if row['crm_used_alternative'] and not row['crm_evaluated']:
            raise ValueError('CRM alternative selected without CRM evaluation')
        if not row['crm_used_alternative']:
            if row['final_adapter'] != row['proposal_after']:
                raise ValueError('Retained adapter differs without CRM replacement')
            if row['crm_class'] != row['drm_class']:
                raise ValueError('CRM class differs without CRM replacement')


def paired_stats(task_pairs):
    counts = {key: 0 for key in compare.OUTCOMES}
    task_accuracy, task_net = [], []
    for pairs in task_pairs:
        local = Counter(compare.outcome(a, b, y) for a, b, y in pairs)
        for key in counts:
            counts[key] += local[key]
        n = len(pairs)
        left = 100.0 * sum(a == y for a, _, y in pairs) / n
        right = 100.0 * sum(b == y for _, b, y in pairs) / n
        task_accuracy.append((left, right))
        task_net.append(100.0 * (local['repaired'] - local['harmed']) / n)
    n = sum(map(len, task_pairs))
    net = statistics.mean(task_net)
    accuracy = [statistics.mean(a[i] for a in task_accuracy) for i in (0, 1)]
    if abs(accuracy[1] - accuracy[0] - net) > 1e-9:
        raise AssertionError('Task-balanced net-change identity failed')
    return dict(counts=counts, samples=n, task_balanced_accuracy=accuracy,
                task_balanced_net_pp=net,
                pooled_net_pp=100.0 * (counts['repaired'] - counts['harmed']) / n)


def audit(run_dir):
    run_dir = Path(run_dir).resolve()
    summary_path = run_dir / 'summary.json'
    paths = [summary_path] + [p for arm in ARMS
                             for p in sorted((run_dir / arm).glob('task_*.jsonl'))]
    before = {str(p.relative_to(run_dir)): sha256(p) for p in paths}
    stored = json.loads(summary_path.read_text(encoding='utf-8'))
    if stored.get('observer_all_stages') != 'PASS_AT_1e-4_LOG_PRECISION':
        raise ValueError('Missing successful observer-on/off check in stored summary')
    original = compare.compare_directories(run_dir / 'class_gate', run_dir / 'full')
    check_summary(original, stored)

    tasks = []
    for path in sorted((run_dir / 'class_gate').glob('task_*.jsonl')):
        arms = {}
        for arm in ARMS:
            header, rows = compare.load_task(run_dir / arm / path.name)
            check_structure(header, rows)
            arms[arm] = rows
        tasks.append(arms)

    stage_pairs = {}
    for stage in STAGES:
        stage_pairs[stage] = paired_stats([
            [(a[stage], b[stage], a['target']) for a, b in zip(t['class_gate'], t['full'])]
            for t in tasks])
    if stage_pairs['final_class']['counts'] != original['counts']:
        raise AssertionError('Independent final counts disagree')

    within_arm = {}
    for arm in ARMS:
        within_arm[arm] = {}
        for first, second in zip(STAGES, STAGES[1:]):
            within_arm[arm][first + '_to_' + second] = paired_stats([
                [(r[first], r[second], r['target']) for r in t[arm]] for t in tasks])

    paths_by_outcome = {key: Counter() for key in ('repaired', 'harmed')}
    changed, corrected_adapter_harms = [], []
    flags = Counter(dict(same_proposal_different_final_class=0,
                         same_adapter_different_crm_class=0,
                         same_adapter_different_final_class=0))
    adapter_class = {arm: Counter(dict(task_right_class_right=0, task_right_class_wrong=0,
                                      task_wrong_class_right=0, task_wrong_class_wrong=0))
                     for arm in ARMS}
    for task in tasks:
        for a, b in zip(task['class_gate'], task['full']):
            for arm, row in (('class_gate', a), ('full', b)):
                tk = 'right' if row['final_adapter'] == row['true_task'] else 'wrong'
                cl = 'right' if row['final_class'] == row['target'] else 'wrong'
                adapter_class[arm]['task_%s_class_%s' % (tk, cl)] += 1
            same_adapter = a['final_adapter'] == b['final_adapter']
            flags['same_proposal_different_final_class'] += (
                a['proposal_after'] == b['proposal_after'] and a['final_class'] != b['final_class'])
            flags['same_adapter_different_crm_class'] += same_adapter and a['crm_class'] != b['crm_class']
            flags['same_adapter_different_final_class'] += same_adapter and a['final_class'] != b['final_class']
            result = compare.outcome(a['final_class'], b['final_class'], a['target'])
            if result not in paths_by_outcome:
                continue
            path = [compare.outcome(a[s], b[s], a['target']) for s in STAGES]
            paths_by_outcome[result][' -> '.join(path)] += 1
            detail = dict(eval_task=a['true_task'], index=a['index'], target=a['target'],
                          input_sha256=a['input_sha256'], final_outcome=result,
                          stage_outcomes=dict(zip(STAGES, path)),
                          proposal_transition=compare.route_group(a['proposal_after'], b['proposal_after'], a['true_task']),
                          final_adapter_transition=compare.route_group(a['final_adapter'], b['final_adapter'], a['true_task']))
            for arm, row in (('class_gate', a), ('full', b)):
                detail[arm] = {key: row[key] for key in DETAIL_FIELDS}
            changed.append(detail)
            if result == 'harmed' and detail['final_adapter_transition'] == 'corrected':
                corrected_adapter_harms.append(detail)

    after_paths = [summary_path] + [p for arm in ARMS
                                   for p in sorted((run_dir / arm).glob('task_*.jsonl'))]
    after = {str(p.relative_to(run_dir)): sha256(p) for p in after_paths}
    if before != after:
        raise ValueError('Input files changed during audit')
    return dict(schema=1, dataset=original['dataset'], seed=original['seed'],
                source_run=str(run_dir), samples=original['samples'],
                checks=dict(stored_summary='PASS', observer_trace_structure='PASS',
                            input_files_unchanged='PASS'), input_sha256=before,
                analyzer_sha256=sha256(__file__), comparator_sha256=sha256(compare.__file__),
                stage_pair_order=list(ARMS), within_arm_order='earlier stage, later stage',
                routing_accuracy={key: original['task_balanced'][key]
                                  for key in ('proposal_accuracy', 'final_adapter_accuracy')},
                stage_pairs=stage_pairs, within_arm=within_arm,
                final_error_paths={k: dict(sorted(v.items())) for k, v in paths_by_outcome.items()},
                final_adapter_class_counts={k: dict(v) for k, v in adapter_class.items()},
                consistency_flags=dict(flags),
                corrected_adapter_class_harms=corrected_adapter_harms,
                changed_class_samples=changed,
                limitations='One seed; conditional comparisons, not independent stage effects. '
                'Class paths do not establish a score/margin or calibration cause. '
                'Nonzero consistency flags require investigation, not automatic attribution to a bug. '
                'No training, GPU execution, or performance measurement was performed.')


def save_report(report, output, run_dir):
    output, run_dir = Path(output).resolve(), Path(run_dir).resolve()
    if output == run_dir or run_dir in output.parents:
        raise ValueError('Save the new report outside the historical run directory')
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
        stream.write('\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.run_dir)
        if args.output:
            save_report(report, args.output, args.run_dir)
        print('DATASET=%s SEED=%s SAMPLES=%s' % (report['dataset'], report['seed'], report['samples']))
        for key in ('checks', 'routing_accuracy', 'stage_pairs', 'within_arm', 'final_error_paths',
                    'final_adapter_class_counts', 'consistency_flags', 'corrected_adapter_class_harms'):
            print(key.upper() + '=' + json.dumps(report[key], sort_keys=True))
        if args.output:
            print('REPORT=' + str(args.output.resolve()))
        print('STAGE_AUDIT_COMPLETE=1/1')
    except (ValueError, KeyError, OSError, StopIteration) as error:
        raise SystemExit('STAGE_AUDIT_STOP: ' + str(error))


if __name__ == '__main__':
    main()
