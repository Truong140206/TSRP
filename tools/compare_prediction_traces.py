"""Compare full vs gated class-only on the same final-stage test samples.

Fail closed on partial, mismatched or corrupted traces. Counts are observations,
not a claim of independent causal stage effects. No training or GPU work here.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


OUTCOMES = ('both_correct', 'repaired', 'harmed', 'both_wrong')
ROUTES = ('unchanged_correct', 'unchanged_wrong', 'corrected', 'harmed', 'changed_wrong')


def outcome(left, right, target):
    if left == target:
        return 'both_correct' if right == target else 'harmed'
    return 'repaired' if right == target else 'both_wrong'


def route_group(left, right, target):
    if left == right:
        return 'unchanged_correct' if left == target else 'unchanged_wrong'
    if right == target:
        return 'corrected'
    return 'harmed' if left == target else 'changed_wrong'


def load_task(path):
    with Path(path).open(encoding='utf-8') as stream:
        header = json.loads(next(stream))
        if header.get('type') != 'header' or header.get('schema') != 1:
            raise ValueError('Unknown trace schema: ' + str(path))
        rows, footer, digest = [], None, hashlib.sha256()
        for line in stream:
            record = json.loads(line)
            if footer is not None:
                raise ValueError('Data after completion marker')
            if record.get('type') == 'complete':
                footer = record
                continue
            if record.get('type') != 'sample' or record['index'] != len(rows):
                raise ValueError('Trace has nonsequential or invalid samples')
            if record['true_task'] != header['eval_task']:
                raise ValueError('Trace task mismatch')
            masks = header['class_mask'][:header['stage']]
            seen = {cls for task in masks for cls in task}
            if record['target'] not in masks[record['true_task']]:
                raise ValueError('Trace target outside evaluation task')
            for name in ('final_class', 'drm_class', 'crm_class'):
                if record[name] not in seen:
                    raise ValueError('Trace predicts an unseen class')
            for name in ('tii_task', 'proposal_before', 'proposal_after', 'final_adapter'):
                if not 0 <= record[name] < header['stage']:
                    raise ValueError('Trace contains an invalid task')
            rows.append(record)
            digest.update(line.encode('utf-8'))
        if not rows or footer is None:
            raise ValueError('Incomplete trace: ' + str(path))
        if len(rows) != header['samples'] or footer['samples'] != len(rows):
            raise ValueError('Incomplete sample coverage')
        if digest.hexdigest() != footer['sha256']:
            raise ValueError('Trace checksum mismatch')
        acc = 100.0 * sum(r['final_class'] == r['target'] for r in rows) / len(rows)
        if not math.isfinite(footer['acc1']) or abs(acc - footer['acc1']) > 1e-8:
            raise ValueError('Trace accuracy mismatch')
        route = 100.0 * sum(r['proposal_after'] == r['true_task'] for r in rows) / len(rows)
        if (not math.isfinite(footer['proposal_accuracy'])
                or abs(route - footer['proposal_accuracy']) > 1e-8):
            raise ValueError('Trace proposal accuracy mismatch')
        return header, rows


def comparable_configuration(header):
    result = dict(header['configuration'])
    result.pop('prediction_trace_dir', None)
    result.pop('rp_route_fusion_weight', None)
    return result


def compare_tasks(left_path, right_path):
    left_header, left = load_task(left_path)
    right_header, right = load_task(right_path)
    for field in ('dataset', 'seed', 'stage', 'eval_task', 'samples', 'class_mask'):
        if left_header[field] != right_header[field]:
            raise ValueError('Pair mismatch: ' + field)
    if left_header['run']['common'] != right_header['run']['common']:
        raise ValueError('Different source/checkpoint provenance')
    if (left_header['run']['variant'], right_header['run']['variant']) != ('class_gate', 'full'):
        raise ValueError('Expected class_gate on the left and full on the right')
    if comparable_configuration(left_header) != comparable_configuration(right_header):
        raise ValueError('Configuration differs beyond routing weight')
    weights = tuple(h['configuration']['rp_route_fusion_weight'] for h in (left_header, right_header))
    if weights != (1.0, 0.7):
        raise ValueError('Expected paper routing weights (1.0, 0.7)')
    counts = {key: 0 for key in OUTCOMES}
    cross = {route: {key: 0 for key in OUTCOMES} for route in ROUTES}
    adapter_cross = {route: {key: 0 for key in OUTCOMES} for route in ROUTES}
    stage2 = {arm: {key: 0 for key in OUTCOMES} for arm in ('class_gate', 'full')}
    transitions = dict(proposal_changed=0, final_adapter_changed=0,
                       changed_proposal_same_final_adapter=0,
                       changed_proposal_same_final_class=0)
    invariants = ('index', 'target', 'true_task', 'input_sha256', 'tii_task',
                  'proposal_before', 'initial_logits_sha256', 'rp_sha256', 'rp_class')
    for a, b in zip(left, right):
        if any(a[k] != b[k] for k in invariants):
            raise ValueError('Inputs/initial logits/RP evidence differ at task %d index %d' %
                             (left_header['eval_task'], a['index']))
        if a['rp_sha256'] is None:
            raise ValueError('Missing RP evidence; not a class-fusion comparison')
        key = outcome(a['final_class'], b['final_class'], a['target'])
        group = route_group(a['proposal_after'], b['proposal_after'], a['true_task'])
        counts[key] += 1
        cross[group][key] += 1
        adapter_cross[route_group(a['final_adapter'], b['final_adapter'], a['true_task'])][key] += 1
        for arm, row in (('class_gate', a), ('full', b)):
            stage2[arm][outcome(row['crm_class'], row['final_class'], row['target'])] += 1
        changed = a['proposal_after'] != b['proposal_after']
        same_adapter = a['final_adapter'] == b['final_adapter']
        transitions['proposal_changed'] += changed
        transitions['final_adapter_changed'] += not same_adapter
        transitions['changed_proposal_same_final_adapter'] += changed and same_adapter
        transitions['changed_proposal_same_final_class'] += changed and a['final_class'] == b['final_class']
    n = len(left)
    scores = {arm: 100.0 * sum(r['final_class'] == r['target'] for r in rows) / n
              for arm, rows in (('class_gate', left), ('full', right))}
    routing = {arm: 100.0 * sum(r['proposal_after'] == r['true_task'] for r in rows) / n
               for arm, rows in (('class_gate', left), ('full', right))}
    adapters = {arm: 100.0 * sum(r['final_adapter'] == r['true_task'] for r in rows) / n
                for arm, rows in (('class_gate', left), ('full', right))}
    repaired = 100.0 * counts['repaired'] / n
    harmed = 100.0 * counts['harmed'] / n
    if abs((scores['full'] - scores['class_gate']) - (repaired - harmed)) > 1e-10:
        raise AssertionError('Net-change identity failed')
    return dict(task=left_header['eval_task'], samples=n, counts=counts,
                proposal_vs_class=cross, final_adapter_vs_class=adapter_cross,
                stage2_within_arm=stage2, final_adapter_accuracy=adapters,
                transitions=transitions, accuracy=scores, proposal_accuracy=routing,
                repair_pp=repaired, harm_pp=harmed, net_pp=repaired - harmed), left_header


def compare_directories(left, right):
    left, right = Path(left), Path(right)
    files = sorted(left.glob('task_*.jsonl'))
    if not files or [p.name for p in files] != sorted(p.name for p in right.glob('task_*.jsonl')):
        raise ValueError('Trace task files missing/mismatched')
    tasks, identity = [], None
    for path in files:
        result, header = compare_tasks(path, right / path.name)
        current = {key: header[key] for key in ('dataset', 'seed', 'stage', 'class_mask')}
        current['provenance'] = header['run']['common']
        if identity is not None and current != identity:
            raise ValueError('Mixed seeds/provenance within trace directory')
        identity = current
        tasks.append(result)
    if [task['task'] for task in tasks] != list(range(identity['stage'])):
        raise ValueError('Not all final-stage tasks are covered')
    counts = {key: sum(t['counts'][key] for t in tasks) for key in OUTCOMES}
    cross = {r: {key: sum(t['proposal_vs_class'][r][key] for t in tasks)
                 for key in OUTCOMES} for r in ROUTES}
    balanced = {key: statistics.mean(t[key] for t in tasks)
                for key in ('repair_pp', 'harm_pp', 'net_pp')}
    balanced['accuracy'] = {arm: statistics.mean(t['accuracy'][arm] for t in tasks)
                            for arm in ('class_gate', 'full')}
    balanced['proposal_accuracy'] = {
        arm: statistics.mean(t['proposal_accuracy'][arm] for t in tasks)
        for arm in ('class_gate', 'full')}
    balanced['final_adapter_accuracy'] = {
        arm: statistics.mean(t['final_adapter_accuracy'][arm] for t in tasks)
        for arm in ('class_gate', 'full')}
    adapter_cross = {r: {key: sum(t['final_adapter_vs_class'][r][key] for t in tasks)
                         for key in OUTCOMES} for r in ROUTES}
    transitions = {key: sum(t['transitions'][key] for t in tasks) for key in tasks[0]['transitions']}
    stage2 = {arm: {key: sum(t['stage2_within_arm'][arm][key] for t in tasks)
                    for key in OUTCOMES} for arm in ('class_gate', 'full')}
    return dict(**identity, samples=sum(t['samples'] for t in tasks),
                counts=counts, proposal_vs_class=cross, final_adapter_vs_class=adapter_cross,
                stage2_within_arm=stage2, transitions=transitions, task_balanced=balanced,
                tasks=tasks, interpretation='Conditional full-minus-gated-class-only; '
                'raw counts are not task-balanced percentage points. '
                'Tracing overhead is not a performance measurement.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--class-gate', type=Path, required=True)
    parser.add_argument('--full', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compare_directories(args.class_gate, args.full), indent=2))


if __name__ == '__main__':
    main()
