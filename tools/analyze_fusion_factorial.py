"""CPU-only 2-by-2 score/gate counterfactuals over every saved final-stage sample."""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import crm_candidate_probe as candidates
from tools.analyze_crm_candidate_errors import input_snapshot, prediction_at, SEEDS
from tools.prediction_score_probe import describe_scores, normalized

CELLS = {
    'cg_scores_cg_gate': ('class_gate', 'class_gate'),
    'full_scores_full_gate': ('full', 'full'),
    'full_scores_cg_gate': ('full', 'class_gate'),
    'cg_scores_full_gate': ('class_gate', 'full'),
}
CONTROLS = dict(class_gate='cg_scores_cg_gate', full='full_scores_full_gate')
STRATA = ('both_correct', 'both_wrong', 'full_repaired', 'full_harmed')
EFFECTS = {
    'full_minus_class_gate': {'full_scores_full_gate': 1, 'cg_scores_cg_gate': -1},
    'scores_at_cg_gate': {'full_scores_cg_gate': 1, 'cg_scores_cg_gate': -1},
    'gate_on_full_scores': {'full_scores_full_gate': 1, 'full_scores_cg_gate': -1},
    'gate_on_cg_scores': {'cg_scores_full_gate': 1, 'cg_scores_cg_gate': -1},
    'scores_at_full_gate': {'full_scores_full_gate': 1, 'cg_scores_full_gate': -1},
    'interaction': {'full_scores_full_gate': 1, 'full_scores_cg_gate': -1,
                    'cg_scores_full_gate': -1, 'cg_scores_cg_gate': 1},
}


def cell_predictions(full, cg):
    records = dict(full=full, class_gate=cg)
    target = full['sample']['target']
    if (any(full['sample'][k] != cg['sample'][k] for k in ('target', 'index', 'input_sha256'))
            or full['rp_scores'] != cg['rp_scores']):
        raise ValueError('Paired inputs/labels/RP differ')
    rp_z, _, _ = normalized(full['rp_scores'])
    primary, betas = {}, {}
    for arm, record in records.items():
        selected = record['candidates'][record['selected']]
        scores = selected['primary_logits']
        description = describe_scores(scores, record['rp_scores'], record['actual_final_logits'],
            target, selected['primary_class'], record['sample']['final_class'])
        primary[arm] = normalized(scores)[0]
        betas[arm] = description['effective_rp_weight']
    cells = {}
    for name, (score_arm, gate_arm) in CELLS.items():
        result = prediction_at(primary[score_arm], rp_z, betas[gate_arm])
        near = result['near_maximum_classes']
        if score_arm == gate_arm:
            observed = records[score_arm]['sample']['final_class']
            if observed not in result['exact_maximum_classes']:
                raise ValueError('Reconstructed control class differs: '+score_arm)
            # Use the recorded evaluator tie decision only for original controls.
            # It is never used to resolve a new counterfactual's ambiguous class.
            low = high = int(observed == target)
            predicted = observed
        else:
            predicted = result['unique_class_if_clear']
            low = int(near == [target])
            high = int(target in near)
        cells[name] = dict(predicted_class=predicted, correct_low=low, correct_high=high,
            prediction_ambiguous=len(near) > 1, correctness_ambiguous=low != high,
            near_maximum_classes=near)
    return cells, betas


def pair_bounds(low, high, reference):
    """Correctness uncertainty, not a statistical confidence interval."""
    return dict(repaired_low=low if not reference else 0,
                repaired_high=high if not reference else 0,
                harmed_low=1-high if reference else 0,
                harmed_high=1-low if reference else 0)


def empty_stats():
    return dict(samples=0, cells={cell:dict(correct_low=0, correct_high=0,
        prediction_ambiguous=0, correctness_ambiguous=0,
        versus={arm:pair_bounds(0,0,False) for arm in CONTROLS}) for cell in CELLS})


def add_row(stats, cells, reference):
    stats['samples'] += 1
    for name, result in cells.items():
        count = stats['cells'][name]
        for key in ('correct_low','correct_high','prediction_ambiguous','correctness_ambiguous'):
            count[key] += int(result[key])
        for arm, correct in reference.items():
            for key, value in pair_bounds(result['correct_low'],result['correct_high'],correct).items():
                count['versus'][arm][key] += value


def merge_stats(reports):
    total = empty_stats()
    for report in reports:
        total['samples'] += report['samples']
        for name, count in report['cells'].items():
            for key in ('correct_low','correct_high','prediction_ambiguous','correctness_ambiguous'):
                total['cells'][name][key] += count[key]
            for arm in CONTROLS:
                for key, value in count['versus'][arm].items():
                    total['cells'][name]['versus'][arm][key] += value
    return total


def value_bounds(low, high):
    return dict(value=low if low == high else None, low=low, high=high)


def effect_bounds(accuracy):
    effects = {}
    for name, coefficients in EFFECTS.items():
        low = sum(c*accuracy[k]['low' if c > 0 else 'high'] for k,c in coefficients.items())
        high = sum(c*accuracy[k]['high' if c > 0 else 'low'] for k,c in coefficients.items())
        effects[name] = value_bounds(low,high)
    return effects


def summarize_tasks(task_reports):
    if not task_reports or any(t['samples'] <= 0 for t in task_reports):
        raise ValueError('Expected nonempty tasks with full sample coverage')
    accuracy = {name:value_bounds(*[
        statistics.mean(100*t['cells'][name]['correct_'+side]/t['samples'] for t in task_reports)
        for side in ('low','high')]) for name in CELLS}
    pairs = {}
    for name in CELLS:
        pairs[name] = {}
        for arm in CONTROLS:
            rates = {key:statistics.mean(100*t['cells'][name]['versus'][arm][key]/t['samples']
                     for t in task_reports) for key in pair_bounds(0,0,False)}
            net_low, net_high = rates['repaired_low']-rates['harmed_high'], rates['repaired_high']-rates['harmed_low']
            control = accuracy[CONTROLS[arm]]['value']
            if (abs(net_low-(accuracy[name]['low']-control)) > 1e-9
                    or abs(net_high-(accuracy[name]['high']-control)) > 1e-9):
                raise ValueError('Task-balanced paired accounting differs from accuracy')
            pairs[name][arm] = dict(**rates, net=value_bounds(net_low,net_high))
    return dict(counts=merge_stats(task_reports), task_balanced_accuracy=accuracy,
        task_balanced_paired_pp=pairs, effects_pp=effect_bounds(accuracy), per_task=task_reports)


def load_records(folder, arm, task, expected_hash):
    path = folder/arm/('candidate_task_%02d.json' % task)
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected_hash:
        raise ValueError('Scores changed after verification: '+str(path))
    captured = json.loads(raw)
    result = {r['sample']['index']:r for r in captured['samples']}
    if len(result) != len(captured['samples']):
        raise ValueError('Duplicate sample indices')
    return result


def analyze_batch(root):
    root = Path(root).resolve()
    before = input_snapshot(root)
    print('Verifying all stored traces, scores and four-seed reports...',flush=True)
    with redirect_stdout(io.StringIO()):
        verified = candidates.audit_batch(root)
    print('CRM_CANDIDATE_ALL_SEEDS_VERIFIED=4/4',flush=True)
    seeds, changed_rows = {}, []
    strata = {s:empty_stats() for s in STRATA}
    for seed in SEEDS:
        folder = root/'ima'/('seed%d' % seed)
        evidence = verified['seeds'][seed]
        tasks = []
        for task in range(10):
            records = {arm:load_records(folder,arm,task,
                evidence['score_sha256'][arm+'/candidate_task_%02d.json' % task]) for arm in CONTROLS}
            if records['full'].keys() != records['class_gate'].keys():
                raise ValueError('Arms have different sample identities')
            stats = empty_stats()
            for index in sorted(records['full']):
                full, cg = records['full'][index], records['class_gate'][index]
                try:
                    cells, betas = cell_predictions(full,cg)
                except ValueError as exc:
                    raise ValueError('seed=%d task=%d index=%d: %s' % (seed,task,index,exc)) from exc
                reference = {arm:bool(cells[cell]['correct_low']) for arm,cell in CONTROLS.items()}
                if reference['full'] == reference['class_gate']:
                    group = 'both_correct' if reference['full'] else 'both_wrong'
                else:
                    group = 'full_repaired' if reference['full'] else 'full_harmed'
                add_row(stats,cells,reference)
                add_row(strata[group],cells,reference)
                if any(c['correct_low'] != int(reference['class_gate']) or c['correct_high'] != int(reference['class_gate'])
                       for c in cells.values()):
                    changed_rows.append(dict(seed=seed,eval_task=task,index=index,target=full['sample']['target'],
                        original_group=group, beta=betas, cells=cells))
            for arm, cell in CONTROLS.items():
                expected = evidence['arms'][arm]['per_task_counts'][task]
                if stats['samples'] != expected['samples'] or stats['cells'][cell]['correct_low'] != expected['actual_correct']:
                    raise ValueError('Reconstructed control task count differs')
            tasks.append(stats)
        summary = summarize_tasks(tasks)
        for arm, cell in CONTROLS.items():
            expected = evidence['arms'][arm]['task_balanced_pp_of_all_samples']['actual_correct']
            if abs(summary['task_balanced_accuracy'][cell]['value']-expected) > 1e-9:
                raise ValueError('Reconstructed control accuracy differs')
        seeds[seed] = summary
        print('FUSION_FACTORIAL_SEED_COMPLETE='+str(seed),flush=True)
        print('SEED_ACCURACY='+json.dumps(dict(seed=seed,**summary['task_balanced_accuracy'])),flush=True)
    accuracy = {name:value_bounds(*[statistics.mean(s['task_balanced_accuracy'][name][side] for s in seeds.values())
                for side in ('low','high')]) for name in CELLS}
    totals = merge_stats([s['counts'] for s in seeds.values()])
    paired_means = {}
    for cell in CELLS:
        paired_means[cell] = {}
        for arm in CONTROLS:
            rates = {key:statistics.mean(s['task_balanced_paired_pp'][cell][arm][key] for s in seeds.values())
                     for key in pair_bounds(0,0,False)}
            paired_means[cell][arm] = dict(**rates, net=value_bounds(
                rates['repaired_low']-rates['harmed_high'], rates['repaired_high']-rates['harmed_low']))
    if totals != merge_stats(list(strata.values())):
        raise ValueError('Original strata do not partition the full data')
    if input_snapshot(root) != before:
        raise ValueError('Input files changed during analysis')
    return dict(schema=1, batch_dir=str(root), input_sha256=before,
        verification='EXACT_CONTROL_CLASSES_AND_COUNTS; INPUT_FILES_UNCHANGED',
        cell_definitions={k:dict(primary_scores=v[0],effective_rp_weight=v[1]) for k,v in CELLS.items()},
        seeds=seeds, aggregate=dict(repeated_prediction_counts=totals,
            equal_seed_task_balanced_accuracy=accuracy,
            equal_seed_task_balanced_paired_pp=paired_means, effects_pp=effect_bounds(accuracy)),
        original_correctness_strata=strata, counterfactual_correctness_changes=changed_rows,
        limitations=[
            'Same fixed RP vector; selected primary scores are before class fusion, after routing and CRM.',
            'This is a final-stage diagnostic using both saved arms, not a deployable selector or retention evaluation.',
            'No forward pass, training, GPU access, label-based selection, parameter search or input modification.',
            'Counts repeat images over seeds; accuracies average tasks, then seeds, not pooled samples.',
            'Low/high bounds concern numerical near ties, not statistical confidence intervals.',
            'Crossed gates are not independent causal stage effects; interaction is reported.',
            'No hyperparameters are selected from these inspected test results.'])


def print_report(report):
    aggregate = report['aggregate']
    print('\nFOUR_SEED_TASK_BALANCED_ACCURACY')
    for cell, score in aggregate['equal_seed_task_balanced_accuracy'].items():
        print('%-26s %s' % (cell, '%.6f'%score['value'] if score['value'] is not None else json.dumps(score)))
    print('FACTORIAL_EFFECTS_PP='+json.dumps(aggregate['effects_pp'],sort_keys=True))
    print('PAIRED_TASK_BALANCED_PP='+json.dumps(aggregate['equal_seed_task_balanced_paired_pp'],sort_keys=True))
    counts = aggregate['repeated_prediction_counts']
    print('REPEATED_PREDICTION_SAMPLES='+str(counts['samples']))
    print('ALL_SAMPLE_COUNTS='+json.dumps(counts,sort_keys=True))
    for name, stats in report['original_correctness_strata'].items():
        print('ORIGINAL_STRATUM='+json.dumps(dict(group=name,**stats),sort_keys=True))
    print('Counts repeat images. Near-tie bounds are not confidence intervals. No new model was evaluated.')
    print('FUSION_FACTORIAL_COMPLETE=4/4',flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args(argv)
    root, output = args.batch_dir.resolve(), args.output.resolve()
    if root == output or root in output.parents:
        parser.error('Output must be outside the input batch')
    if output.exists() or not output.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    try:
        report = analyze_batch(root)
        with output.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print_report(report)
        print('REPORT='+str(output),flush=True)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit('FUSION_FACTORIAL_STOP: '+str(exc)) from exc


if __name__ == '__main__':
    main()
