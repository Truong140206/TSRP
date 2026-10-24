"""Exploratory fixed-rule selection between the two saved inference paths.

The rule was designed after inspecting ImageNet-A test diagnostics. It is not a
validated improvement. No training, threshold sweep or default evaluator change.
"""
import argparse
from contextlib import redirect_stdout
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import crm_candidate_probe as candidates
from tools.analyze_crm_candidate_errors import input_snapshot, prediction_at, SEEDS
from tools.analyze_fusion_factorial import cell_predictions, load_records
from tools.prediction_score_probe import normalized

RULE = 'full_primary_rp_consensus_else_class_gate_v1'
ARMS = ('class_gate', 'full', 'consensus')
REASONS = ('same_adapter', 'unclear_head', 'heads_disagree', 'heads_agree')


def choose_path(reference_adapter, proposed_adapter, proposed_primary, rp_scores):
    """Unlabelled inference decision. No target, task label, image ID or seed.

    Keep class-gate unless the proposed final adapter differs and its primary
    head and the fixed RP head have the same numerically clear top class.
    Agreement is evidence for this hypothesis, not a correctness guarantee.
    """
    if (type(reference_adapter) is not int or type(proposed_adapter) is not int
            or min(reference_adapter, proposed_adapter) < 0):
        raise ValueError('Expected nonnegative adapter IDs')
    if (len(proposed_primary) < 2 or len(proposed_primary) != len(rp_scores)
            or not all(math.isfinite(x) for x in (*proposed_primary, *rp_scores))):
        raise ValueError('Expected matching finite class vectors')
    if reference_adapter == proposed_adapter:
        return dict(path='class_gate', reason='same_adapter', agreed_class=None)
    primary_z = normalized(proposed_primary)[0]
    rp_z = normalized(rp_scores)[0]
    # The fixed tolerance is the existing float32 near-tie check, not a swept
    # confidence threshold. Standardization removes score offset/scale here.
    p = prediction_at(primary_z, primary_z, 0.)['unique_class_if_clear']
    r = prediction_at(rp_z, rp_z, 0.)['unique_class_if_clear']
    if p is None or r is None:
        return dict(path='class_gate', reason='unclear_head', agreed_class=None)
    if p != r:
        return dict(path='class_gate', reason='heads_disagree', agreed_class=None)
    return dict(path='full', reason='heads_agree', agreed_class=p)


def analyze_pair(full, cg):
    controls, _ = cell_predictions(full,cg)
    f = full['candidates'][full['selected']]
    g = cg['candidates'][cg['selected']]
    decision = choose_path(g['adapter'],f['adapter'],f['primary_logits'],full['rp_scores'])
    predictions = dict(full=controls['full_scores_full_gate']['predicted_class'],
        class_gate=controls['cg_scores_cg_gate']['predicted_class'])
    if decision['path'] == 'full' and predictions['full'] != decision['agreed_class']:
        raise ValueError('Two-head agreement does not reproduce captured Full fusion')
    predictions['consensus'] = predictions[decision['path']]
    # Labels first affect this policy's accounting here, after its choice.
    target = full['sample']['target']
    correct = {arm:int(prediction == target) for arm,prediction in predictions.items()}
    return dict(decision=decision, predictions=predictions, correct=correct,
        reference_adapter=g['adapter'], proposed_adapter=f['adapter'],
        target=target, index=full['sample']['index'])


def empty_counts():
    return dict(samples=0, correct=dict.fromkeys(ARMS,0),
        reasons=dict.fromkeys(REASONS,0), accepted_full=0, changed_class_from_class_gate=0,
        versus={arm:dict(repaired=0,harmed=0) for arm in ('class_gate','full')},
        original_full_repaired=0, retained_full_repairs=0, rejected_full_repairs=0,
        original_full_harmed=0, accepted_full_harms=0, avoided_full_harms=0,
        accepted_both_wrong=0)


def add_case(counts, row):
    correct, decision = row['correct'], row['decision']
    counts['samples'] += 1
    for arm in ARMS:
        counts['correct'][arm] += correct[arm]
    counts['reasons'][decision['reason']] += 1
    counts['accepted_full'] += int(decision['path'] == 'full')
    counts['changed_class_from_class_gate'] += int(row['predictions']['consensus'] != row['predictions']['class_gate'])
    for arm in counts['versus']:
        counts['versus'][arm]['repaired'] += int(not correct[arm] and correct['consensus'])
        counts['versus'][arm]['harmed'] += int(correct[arm] and not correct['consensus'])
    if correct['full'] and not correct['class_gate']:
        counts['original_full_repaired'] += 1
        counts['retained_full_repairs' if correct['consensus'] else 'rejected_full_repairs'] += 1
    elif correct['class_gate'] and not correct['full']:
        counts['original_full_harmed'] += 1
        counts['avoided_full_harms' if correct['consensus'] else 'accepted_full_harms'] += 1
    elif not correct['full'] and not correct['class_gate'] and decision['path'] == 'full':
        counts['accepted_both_wrong'] += 1
    if correct['full'] == correct['class_gate'] and correct['consensus'] != correct['full']:
        raise ValueError('Selecting an existing path cannot change shared correctness')


def sum_counts(reports):
    def merge(values):
        first=values[0]
        if isinstance(first,dict):
            return {key:merge([v[key] for v in values]) for key in first}
        return sum(values)
    result=merge(reports)
    if sum(result['reasons'].values()) != result['samples']:
        raise ValueError('Decision reasons do not cover every sample')
    for arm,pair in result['versus'].items():
        if pair['repaired']-pair['harmed'] != result['correct']['consensus']-result['correct'][arm]:
            raise ValueError('Paired counts do not reproduce accuracy changes')
    return result


def summarize_tasks(tasks):
    if not tasks or any(t['samples'] <= 0 for t in tasks):
        raise ValueError('Expected complete nonempty tasks')
    accuracy={arm:statistics.mean(100*t['correct'][arm]/t['samples'] for t in tasks) for arm in ARMS}
    pairs={}
    for arm in ('class_gate','full'):
        pair={key:statistics.mean(100*t['versus'][arm][key]/t['samples'] for t in tasks)
              for key in ('repaired','harmed')}
        pair['net']=pair['repaired']-pair['harmed']
        if abs(pair['net']-(accuracy['consensus']-accuracy[arm])) > 1e-9:
            raise ValueError('Task-balanced paired rates do not match accuracy')
        pairs[arm]=pair
    return dict(task_balanced_accuracy=accuracy,paired_task_balanced_pp=pairs,
        counts=sum_counts(tasks),per_task=tasks)


def analyze_batch(root):
    root=Path(root).resolve()
    before=input_snapshot(root)
    print('EXPLORATORY_ONLY: rule designed after inspecting these test results; not held-out validation.',flush=True)
    print('RULE='+RULE,flush=True)
    with redirect_stdout(io.StringIO()):
        verified=candidates.audit_batch(root)
    print('CRM_CANDIDATE_ALL_SEEDS_VERIFIED=4/4',flush=True)
    seeds,decisions={},[]
    for seed in SEEDS:
        folder=root/'ima'/('seed%d'%seed)
        evidence=verified['seeds'][seed]
        tasks=[]
        for task in range(10):
            records={arm:load_records(folder,arm,task,evidence['score_sha256'][arm+'/candidate_task_%02d.json'%task])
                     for arm in ('class_gate','full')}
            if records['full'].keys() != records['class_gate'].keys():
                raise ValueError('Paired sample indices differ')
            counts=empty_counts()
            for index in sorted(records['full']):
                try:
                    row=analyze_pair(records['full'][index],records['class_gate'][index])
                except ValueError as exc:
                    raise ValueError('seed=%d task=%d index=%d: %s'%(seed,task,index,exc)) from exc
                add_case(counts,row)
                decisions.append(dict(seed=seed,eval_task=task,**row))
            for arm in ('class_gate','full'):
                original=evidence['arms'][arm]['per_task_counts'][task]
                if counts['samples'] != original['samples'] or counts['correct'][arm] != original['actual_correct']:
                    raise ValueError('Original controls do not reproduce task counts')
            tasks.append(counts)
        seeds[seed]=summarize_tasks(tasks)
        for arm in ('class_gate','full'):
            if abs(seeds[seed]['task_balanced_accuracy'][arm]-evidence['arms'][arm]['task_balanced_pp_of_all_samples']['actual_correct']) > 1e-9:
                raise ValueError('Original controls do not reproduce task-balanced accuracy')
        print('ROUTE_ACCEPTANCE_SEED_COMPLETE='+str(seed),flush=True)
        print('SEED_RESULT='+json.dumps(dict(seed=seed,accuracy=seeds[seed]['task_balanced_accuracy'],
            paired_pp=seeds[seed]['paired_task_balanced_pp'],counts=seeds[seed]['counts'])),flush=True)
    accuracy={arm:statistics.mean(s['task_balanced_accuracy'][arm] for s in seeds.values()) for arm in ARMS}
    pairs={arm:{key:statistics.mean(s['paired_task_balanced_pp'][arm][key] for s in seeds.values())
                for key in ('repaired','harmed','net')} for arm in ('class_gate','full')}
    if input_snapshot(root) != before:
        raise ValueError('Input files changed during analysis')
    return dict(schema=1,rule=RULE,rule_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        status='POST_HOC_EXPLORATORY_NOT_VALIDATED',batch_dir=str(root),input_sha256=before,
        verification='ORIGINAL_CONTROL_CLASSES_COUNTS_ACCURACY_MATCH; INPUT_FILES_UNCHANGED',
        design_context='One fixed consensus rule, designed after inspecting ImageNet-A test diagnostics. '
                       'No fitted parameters or outcome-based rule selection during this run.',
        seeds=seeds,aggregate=dict(equal_seed_task_balanced_accuracy=accuracy,
            paired_task_balanced_pp=pairs,counts=sum_counts([s['counts'] for s in seeds.values()]),
            positive_seeds={arm:sum(s['paired_task_balanced_pp'][arm]['net']>0 for s in seeds.values())
                            for arm in ('class_gate','full')}),decisions=decisions,
        limitations=[
            'Agreement of two heads is not a correctness guarantee; both can agree on a wrong class.',
            'The decision function receives no labels, image identifiers or seed; labels enter accounting only.',
            'The rule uses both saved inference paths and may need additional forwards in a live evaluator.',
            'Costs, intermediate stages, retention and other datasets have not been evaluated by this tool.',
            'The current loader called val reads the test split; this is not an independent validation result.',
            'A different seed on the same test images is not an independent validation split.',
            'Counts repeat images over seeds; accuracy averages tasks then seeds.',
            'No model loading, GPU execution, training, default algorithm or manuscript change.'])


def print_report(report):
    aggregate=report['aggregate']
    print('\nFOUR_SEED_TASK_BALANCED_ACCURACY')
    for arm,accuracy in aggregate['equal_seed_task_balanced_accuracy'].items():
        print('%-15s %.6f'%(arm,accuracy))
    print('ROUTE_ACCEPTANCE_PAIRED_PP='+json.dumps(aggregate['paired_task_balanced_pp'],sort_keys=True))
    print('ROUTE_ACCEPTANCE_COUNTS='+json.dumps(aggregate['counts'],sort_keys=True))
    print('POSITIVE_SEEDS='+json.dumps(aggregate['positive_seeds'],sort_keys=True))
    print('EXPLORATORY_ONLY: do not replace paper results or claim independent validation from this run.')
    print('ROUTE_ACCEPTANCE_COMPLETE=4/4',flush=True)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root,output=args.batch_dir.resolve(),args.output.resolve()
    if root == output or root in output.parents:
        parser.error('Output must be outside the input batch')
    if output.exists() or not output.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    try:
        report=analyze_batch(root)
        with output.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print_report(report)
        print('REPORT='+str(output),flush=True)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit('ROUTE_ACCEPTANCE_STOP: '+str(exc)) from exc


if __name__ == '__main__':
    main()
