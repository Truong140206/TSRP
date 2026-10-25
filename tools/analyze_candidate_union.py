"""Label-aware CPU diagnosis of saved class-gate/Full candidate overlap.

Oracle availability uses labels for reporting only. This is not a new selector,
inference rule, validated accuracy result, or measurement of deployment cost.
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

from tools import crm_candidate_probe as capture
from tools.analyze_crm_candidate_errors import input_snapshot, SEEDS
from tools.analyze_fusion_factorial import load_records

EVENTS = (
    'cg_actual_correct', 'full_actual_correct', 'final_output_oracle_correct',
    'cg_available', 'full_available', 'union_available',
    'available_both', 'available_cg_only', 'available_full_only', 'available_neither',
    'cg_selector_missed', 'full_selector_missed',
    'union_missed_by_cg', 'union_missed_by_full',
    'union_primary_available', 'rp_correct',
    'unavailable_primary_only', 'unavailable_rp_only',
    'unavailable_primary_and_rp', 'unavailable_neither_primary_nor_rp',
    'has_added_full_adapter', 'has_removed_cg_adapter',
    'has_shared_adapter', 'primary_ties_present', 'fused_ties_present',
)
SIZES = ('cg_unique', 'full_unique', 'union_unique', 'shared', 'added_full',
         'removed_cg', 'cg_duplicate_slots', 'full_duplicate_slots')


def analyze_pair(full, cg):
    """Compare verified captures; target is intentionally used for diagnosis."""
    if (any(full['sample'][k] != cg['sample'][k]
            for k in ('target', 'index', 'input_sha256'))
            or full['rp_scores'] != cg['rp_scores']):
        raise ValueError('Paired inputs/labels/RP differ')
    records = dict(class_gate=cg, full=full)
    eligible = cg['sample']['crm_evaluated']
    if full['sample']['crm_evaluated'] != eligible:
        raise ValueError('Original arms must share CRM eligibility')
    target, classes = cg['sample']['target'], len(cg['rp_scores'])
    if (classes < 2 or not all(math.isfinite(x) for x in cg['rp_scores'])
            or type(target) is not int or not 0 <= target < classes):
        raise ValueError('Invalid target class')
    entries, sets, actual = {}, {}, {}
    largest_difference = 0.
    for arm, record in records.items():
        if len(record['candidates']) != (2 if eligible else 1):
            raise ValueError('Do not invent candidates for ineligible samples')
        selected = record['selected']
        if type(selected) is not int or not 0 <= selected < len(record['candidates']):
            raise ValueError('Invalid selected candidate')
        sets[arm] = set()
        for slot, candidate in enumerate(record['candidates']):
            adapter = candidate['adapter']
            if type(adapter) is not int or adapter < 0:
                raise ValueError('Invalid adapter ID')
            for field, prediction in (('primary_logits','primary_class'), ('fused_logits','fused_class')):
                values = candidate[field]
                if (len(values) != classes or not all(math.isfinite(x) for x in values)
                        or not capture.winner_valid(values,candidate[prediction])):
                    raise ValueError('Invalid candidate scores or captured class')
            sets[arm].add(adapter)
            if adapter in entries:
                reference = entries[adapter]['reference']
                for field in ('primary_logits','fused_logits'):
                    if not capture.vector_close(reference[field],candidate[field]):
                        raise ValueError('Same adapter has different scores; cannot deduplicate')
                    largest_difference = max(largest_difference,
                        max(abs(a-b) for a,b in zip(reference[field],candidate[field])))
                if any(reference[k] != candidate[k] for k in ('primary_class','fused_class')):
                    raise ValueError('Same adapter has different predictions; cannot deduplicate')
            else:
                entries[adapter] = dict(reference=candidate, occurrences=[],
                    primary_tied_classes=[], fused_tied_classes=[])
            entry = entries[adapter]
            for field, key in (('primary_logits','primary_tied_classes'),('fused_logits','fused_tied_classes')):
                entry[key] = sorted(set(entry[key]) | set(capture.tied_classes(candidate[field])))
            entry['occurrences'].append(dict(arm=arm,slot=slot,selected=slot==selected))
        chosen = record['candidates'][selected]
        actual[arm] = record['sample']['final_class']
        if (actual[arm] != chosen['fused_class']
                or record['sample']['final_adapter'] != chosen['adapter']
                or not capture.vector_close(record['actual_final_logits'],chosen['fused_logits'])):
            raise ValueError('Selected candidate does not match original final output')
    if eligible and full['candidates'][1]['adapter'] != cg['candidates'][1]['adapter']:
        raise ValueError('Original arms must share the pre-routing alternative')
    a, b = sets['class_gate'], sets['full']
    union, added, removed, shared = a | b, b-a, a-b, a & b
    # Original arms differ only in the current proposal; the alternative stays
    # shared. Enforce this before attributing an added adapter to routing.
    if any(adapter != full['candidates'][0]['adapter'] for adapter in added):
        raise ValueError('Added adapter is not the Full routing proposal')
    if len(union) > (3 if eligible else 2):
        raise ValueError('Unexpected original candidate topology')
    correct_fused = {i for i,e in entries.items() if e['reference']['fused_class']==target}
    correct_primary = {i for i,e in entries.items() if e['reference']['primary_class']==target}
    av_a, av_b = bool(a & correct_fused), bool(b & correct_fused)
    av_u, primary = bool(correct_fused), bool(correct_primary)
    if (full['sample']['rp_class'] != cg['sample']['rp_class']
            or not capture.winner_valid(cg['rp_scores'],cg['sample']['rp_class'])):
        raise ValueError('Shared RP prediction mismatch')
    rp = cg['sample']['rp_class']==target
    cg_ok, full_ok = actual['class_gate']==target, actual['full']==target
    events = dict(
        cg_actual_correct=cg_ok,full_actual_correct=full_ok,
        final_output_oracle_correct=cg_ok or full_ok,
        cg_available=av_a,full_available=av_b,union_available=av_u,
        available_both=av_a and av_b,available_cg_only=av_a and not av_b,
        available_full_only=av_b and not av_a,available_neither=not av_u,
        cg_selector_missed=av_a and not cg_ok,full_selector_missed=av_b and not full_ok,
        union_missed_by_cg=av_u and not cg_ok,union_missed_by_full=av_u and not full_ok,
        union_primary_available=primary,rp_correct=rp,
        unavailable_primary_only=not av_u and primary and not rp,
        unavailable_rp_only=not av_u and rp and not primary,
        unavailable_primary_and_rp=not av_u and primary and rp,
        unavailable_neither_primary_nor_rp=not av_u and not primary and not rp,
        has_added_full_adapter=bool(added),has_removed_cg_adapter=bool(removed),
        has_shared_adapter=bool(shared),
        primary_ties_present=any(len(e['primary_tied_classes'])>1 for e in entries.values()),
        fused_ties_present=any(len(e['fused_tied_classes'])>1 for e in entries.values()))
    sizes = dict(cg_unique=len(a),full_unique=len(b),union_unique=len(union),
        shared=len(shared),added_full=len(added),removed_cg=len(removed),
        cg_duplicate_slots=len(cg['candidates'])-len(a),
        full_duplicate_slots=len(full['candidates'])-len(b))
    details = [dict(adapter=i,primary_class=e['reference']['primary_class'],
        fused_class=e['reference']['fused_class'],occurrences=e['occurrences'],
        primary_tied_classes=e['primary_tied_classes'],fused_tied_classes=e['fused_tied_classes'])
        for i,e in sorted(entries.items())]
    return dict(index=cg['sample']['index'],input_sha256=cg['sample']['input_sha256'],
        target=target,events=events,sizes=sizes,
        actual_classes=actual,cg_adapters=sorted(a),full_adapters=sorted(b),
        added_full_adapters=sorted(added),removed_cg_adapters=sorted(removed),
        correct_fused_adapters=sorted(correct_fused),correct_primary_adapters=sorted(correct_primary),
        unique_candidates=details,duplicate_score_max_abs_difference=largest_difference)


def empty_counts():
    return dict(samples=0,events=dict.fromkeys(EVENTS,0),candidate_totals=dict.fromkeys(SIZES,0),
        duplicate_score_max_abs_difference=0.)


def add_case(counts, row):
    counts['samples'] += 1
    for key in EVENTS:
        counts['events'][key] += int(row['events'][key])
    for key in SIZES:
        counts['candidate_totals'][key] += row['sizes'][key]
    counts['duplicate_score_max_abs_difference'] = max(counts['duplicate_score_max_abs_difference'],
        row['duplicate_score_max_abs_difference'])


def check_counts(counts):
    n, e, c = counts['samples'],counts['events'],counts['candidate_totals']
    if n <= 0 or any(not 0 <= v <= n for v in e.values()):
        raise ValueError('Invalid event counts')
    identities = (
        (e['available_both']+e['available_cg_only']+e['available_full_only']+e['available_neither'],n),
        (e['union_available']-e['cg_available'],e['available_full_only']),
        (e['union_available']-e['full_available'],e['available_cg_only']),
        (e['cg_available']-e['cg_actual_correct'],e['cg_selector_missed']),
        (e['full_available']-e['full_actual_correct'],e['full_selector_missed']),
        (e['union_available']-e['cg_actual_correct'],e['union_missed_by_cg']),
        (e['union_available']-e['full_actual_correct'],e['union_missed_by_full']),
        (sum(e[k] for k in EVENTS if k.startswith('unavailable_')),e['available_neither']),
        (c['cg_unique']+c['added_full'],c['union_unique']),
        (c['full_unique']+c['removed_cg'],c['union_unique']),
        (c['cg_unique']+c['full_unique']-c['shared'],c['union_unique']),
    )
    if any(left != right for left,right in identities):
        raise ValueError('Candidate union accounting mismatch')
    if not max(e['cg_actual_correct'],e['full_actual_correct']) <= e['final_output_oracle_correct'] <= e['union_available']:
        raise ValueError('Invalid oracle nesting')


def merge_counts(items):
    result=empty_counts()
    for item in items:
        check_counts(item)
        result['samples'] += item['samples']
        for field in ('events','candidate_totals'):
            for key in result[field]:
                result[field][key] += item[field][key]
        result['duplicate_score_max_abs_difference']=max(result['duplicate_score_max_abs_difference'],
            item['duplicate_score_max_abs_difference'])
    check_counts(result)
    return result


def summarize_tasks(tasks):
    counts=merge_counts(tasks)
    rates={key:statistics.mean(100*t['events'][key]/t['samples'] for t in tasks) for key in EVENTS}
    sizes={key:statistics.mean(t['candidate_totals'][key]/t['samples'] for t in tasks) for key in SIZES}
    return dict(counts=counts,task_balanced_pp=rates,task_balanced_candidates_per_image=sizes,per_task=tasks)


def analyze_batch(root):
    root=Path(root).resolve()
    before=input_snapshot(root)
    print('LABEL_AWARE_DIAGNOSTIC_ONLY: oracle availability is not deployed accuracy.',flush=True)
    with redirect_stdout(io.StringIO()):
        verified=capture.audit_batch(root)
    print('CRM_CANDIDATE_ALL_SEEDS_VERIFIED=4/4',flush=True)
    seeds,cases={},[]
    # Match both controls and their already reported candidate ceilings exactly.
    mappings={'class_gate':{'cg_actual_correct':'actual_correct','cg_available':'available_correct',
                           'cg_selector_missed':'selected_wrong_other_correct'},
              'full':{'full_actual_correct':'actual_correct','full_available':'available_correct',
                      'full_selector_missed':'selected_wrong_other_correct'}}
    for seed in SEEDS:
        evidence=verified['seeds'][seed]
        folder=root/'ima'/('seed%d'%seed)
        tasks=[]
        for task in range(10):
            records={arm:load_records(folder,arm,task,evidence['score_sha256'][arm+'/candidate_task_%02d.json'%task])
                     for arm in mappings}
            if records['full'].keys() != records['class_gate'].keys():
                raise ValueError('Paired sample indices differ')
            counts=empty_counts()
            for index in sorted(records['full']):
                try:
                    row=analyze_pair(records['full'][index],records['class_gate'][index])
                except (ValueError,KeyError,TypeError) as exc:
                    raise ValueError('seed=%d task=%d index=%d: %s'%(seed,task,index,exc)) from exc
                add_case(counts,row)
                cases.append(dict(seed=seed,eval_task=task,**row))
            check_counts(counts)
            for arm,mapping in mappings.items():
                original=evidence['arms'][arm]['per_task_counts'][task]
                if (counts['samples'] != original['samples'] or any(
                        counts['events'][ours] != original[theirs] for ours,theirs in mapping.items())):
                    raise ValueError('Original control/candidate counts differ')
            tasks.append(counts)
        summary=summarize_tasks(tasks)
        for arm,mapping in mappings.items():
            rates=evidence['arms'][arm]['task_balanced_pp_of_all_samples']
            if any(abs(summary['task_balanced_pp'][ours]-rates[theirs])>1e-9 for ours,theirs in mapping.items()):
                raise ValueError('Original task-balanced rates differ')
        seeds[seed]=summary
        print('CANDIDATE_UNION_SEED_RESULT='+json.dumps(dict(seed=seed,
            task_balanced_pp=summary['task_balanced_pp'],counts=summary['counts'],
            candidates_per_image=summary['task_balanced_candidates_per_image'])),flush=True)
    aggregate=dict(counts=merge_counts([s['counts'] for s in seeds.values()]),
        equal_seed_task_balanced_pp={k:statistics.mean(s['task_balanced_pp'][k] for s in seeds.values()) for k in EVENTS},
        equal_seed_task_balanced_candidates_per_image={k:statistics.mean(
            s['task_balanced_candidates_per_image'][k] for s in seeds.values()) for k in SIZES})
    if input_snapshot(root) != before:
        raise ValueError('Input files changed during analysis')
    return dict(schema=1,status='LABEL_AWARE_DIAGNOSTIC_NOT_INFERENCE_RULE',batch_dir=str(root),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),input_sha256=before,
        verification='ORIGINAL_CONTROLS_AND_CANDIDATE_COUNTS_MATCH; INPUT_FILES_UNCHANGED',
        seeds=seeds,aggregate=aggregate,cases=cases,
        definitions={
            'available_full_only':'No correct fused candidate in class-gate, but at least one in Full; added union availability.',
            'available_cg_only':'Class-gate has a correct fused candidate and Full has none; opportunity lost by replacing its candidate set.',
            'cg_selector_missed':'Class-gate selected output is wrong but its other candidate is correct after the same fusion.',
            'union_available':'At least one captured fused class in the deduplicated union is correct; label-aware oracle only.'},
        limitations=[
            'No inference rule or new measured method accuracy; labels are used for diagnostic counting.',
            'Only candidates already evaluated are included; no candidate is invented for an ineligible sample.',
            'Identical adapter IDs require matching primary/fused scores within existing capture tolerance and identical captured classes.',
            'Ties follow recorded class choices, never a target-favouring choice; tie presence is reported.',
            'Oracle ceilings apply to hard selection of fixed captured classes, not arbitrary score mixtures or new experts.',
            'When all primary and RP top classes are wrong, a new mixture could still predict a different, correct class.',
            'Counts repeat images across seeds; reported rates average tasks then seeds.',
            'Test diagnostics do not supply independent validation for selecting a new method.',
            'No GPU, model forwards, training, normal evaluator or manuscript changes; live costs and retention remain unmeasured.'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args(argv)
    root,output=args.batch_dir.resolve(),args.output.resolve()
    if root==output or root in output.parents:
        parser.error('Output must be outside the input batch')
    if output.exists() or not output.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    try:
        report=analyze_batch(root)
        with output.open('x',encoding='utf-8') as stream:
            stream.write(json.dumps(report,indent=2,allow_nan=False)+'\n')
        rates=report['aggregate']['equal_seed_task_balanced_pp']
        print('\nFOUR_SEED_TASK_BALANCED_PERCENT_OR_PP')
        for key in ('cg_actual_correct','full_actual_correct','final_output_oracle_correct',
                    'cg_available','full_available','union_available',
                    'available_full_only','available_cg_only','cg_selector_missed','full_selector_missed'):
            print('%-30s %.6f'%(key,rates[key]))
        print('CANDIDATE_UNION_FOUR_SEED_MEANS='+json.dumps(report['aggregate'],sort_keys=True))
        print('ORACLE_ONLY: candidate availability is not deployed accuracy or an inference rule.')
        print('CANDIDATE_UNION_COMPLETE=4/4',flush=True)
        print('REPORT='+str(output),flush=True)
    except (OSError,ValueError,KeyError,TypeError) as exc:
        raise SystemExit('CANDIDATE_UNION_STOP: '+str(exc)) from exc


if __name__=='__main__':
    main()
