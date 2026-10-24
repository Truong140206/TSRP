"""Read saved CRM scores on CPU; explain paired errors without changing inference."""
import argparse
from collections import Counter
from contextlib import redirect_stdout
import io
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools import crm_candidate_probe as candidates
from tools.audit_prediction_stages import check_summary, sha256
from tools.prediction_score_probe import describe_scores, normalized

SEEDS = (42, 43, 44, 45)
ARMS = ('full', 'class_gate')


def beta_interval(primary_z, rp_z, target):
    """Closed interval where the target is a maximum, allowing boundary ties.

    Solve every target-versus-rival linear inequality on beta in [0, 1].
    This uses the target label for diagnosis, never for an inference decision.
    Inputs are already standardized; do not standardize them a second time.
    """
    if (len(primary_z) != len(rp_z) or len(primary_z) < 2
            or not 0 <= target < len(primary_z)
            or not all(math.isfinite(x) for x in (*primary_z, *rp_z))):
        raise ValueError('Expected finite paired class scores and a valid target')
    lower, upper = 0., 1.
    lower_classes, upper_classes = [], []
    impossible = []
    always_tied = []
    for rival in range(len(primary_z)):
        if rival == target:
            continue
        p = primary_z[target] - primary_z[rival]
        r = rp_z[target] - rp_z[rival]
        slope = r - p
        if slope == 0:
            if p < 0:
                impossible.append(rival)
            elif p == 0:
                always_tied.append(rival)
        elif slope > 0:
            bound = -p / slope
            if bound > lower:
                lower, lower_classes = bound, [rival]
            elif bound == lower:
                lower_classes.append(rival)
        else:
            bound = -p / slope
            if bound < upper:
                upper, upper_classes = bound, [rival]
            elif bound == upper:
                upper_classes.append(rival)
    feasible = not impossible and lower <= upper
    interval = [lower, upper] if feasible else None
    operational = [lower, min(upper, .5)] if feasible and lower <= min(upper, .5) else None
    return dict(target_is_maximum_beta_interval_0_1=interval,
        target_is_maximum_beta_interval_0_half=operational,
        lower_boundary_rivals=lower_classes, upper_boundary_rivals=upper_classes,
        always_tied_rivals=always_tied, impossible_constant_rivals=impossible,
        boundary_note='Closed intervals allow ties; endpoints need not give a unique correct prediction.')


def prediction_at(primary_z, rp_z, beta):
    if not 0 <= beta <= 1:
        raise ValueError('RP weight outside [0, 1]')
    values = [(1-beta)*p + beta*r for p, r in zip(primary_z, rp_z)]
    best = max(values)
    # Captures originate in float32. Do not call near ties a reliable repair.
    tolerance = 1e-7 + 1e-6 * max(abs(x) for x in values)
    near = [i for i, x in enumerate(values) if best-x <= tolerance]
    return dict(unique_class_if_clear=near[0] if len(near) == 1 else None,
        exact_maximum_classes=[i for i, x in enumerate(values) if x == best],
        near_maximum_classes=near, standardized_score_tolerance=tolerance)


def arm_scores(record, target, rival, other_beta):
    selected = record['candidates'][record['selected']]
    primary, rp = selected['primary_logits'], record['rp_scores']
    description = describe_scores(primary, rp, record['actual_final_logits'], target,
        selected['primary_class'], record['sample']['final_class'])
    pz, _, _ = normalized(primary)
    rz, _, _ = normalized(rp)
    beta = description['effective_rp_weight']
    p, r = pz[target]-pz[rival], rz[target]-rz[rival]
    crossing = p/(p-r) if p != r and 0 <= p/(p-r) <= 1 else None
    interval = beta_interval(pz, rz, target)
    bounds = interval['target_is_maximum_beta_interval_0_1']
    return dict(primary_class=selected['primary_class'], final_class=record['sample']['final_class'],
        rp_class=record['sample']['rp_class'], selected_adapter=selected['adapter'],
        candidate_adapters=[c['adapter'] for c in record['candidates']],
        margin=description['margin'], effective_rp_weight=beta,
        primary_target_minus_rival=p, rp_target_minus_rival=r,
        weighted_primary_gap=(1-beta)*p, weighted_rp_gap=beta*r,
        combined_target_minus_rival=(1-beta)*p+beta*r,
        target_rival_equal_at_beta=crossing,
        all_class_beta_interval=interval,
        actual_beta_minus_upper_boundary=beta-bounds[1] if bounds else None,
        reconstructed_prediction=prediction_at(pz, rz, beta),
        use_other_arm_beta=dict(beta=other_beta, **prediction_at(pz, rz, other_beta)),
        reconstruction_max_abs_error=description['reconstruction_max_abs_error'])


def analyze_pair(seed, task, full, cg):
    sample, reference = full['sample'], cg['sample']
    if (any(sample[k] != reference[k] for k in ('index', 'target', 'input_sha256'))
            or full['rp_scores'] != cg['rp_scores']):
        raise ValueError('Paired image, target or RP scores differ')
    target = sample['target']
    correct = {arm: r['sample']['final_class'] == target for arm, r in zip(ARMS, (full, cg))}
    if correct['full'] == correct['class_gate']:
        raise ValueError('Expected a paired final-class change')
    outcome = 'repaired' if correct['full'] else 'harmed'
    wrong_arm = 'class_gate' if correct['full'] else 'full'
    records = dict(full=full, class_gate=cg)
    rival = records[wrong_arm]['sample']['final_class']
    betas, events = {}, {}
    for arm, record in records.items():
        flags, detail = candidates.inspect_record(record, record['sample'], len(record['rp_scores']), 10)
        events[arm] = flags
        betas[arm] = detail['candidate_scores'][record['selected']]['effective_rp_weight']
    arms = {arm: arm_scores(record, target, rival, betas['class_gate' if arm == 'full' else 'full'])
            for arm, record in records.items()}
    f, c = arms['full'], arms['class_gate']
    # An exact ordered decomposition of this single rival gap, not independent
    # causal effects or a statement about all classes' prediction decisions.
    primary_change = (1-betas['class_gate'])*(f['primary_target_minus_rival']-c['primary_target_minus_rival'])
    beta_change = (betas['full']-betas['class_gate'])*(f['rp_target_minus_rival']-f['primary_target_minus_rival'])
    delta = f['combined_target_minus_rival']-c['combined_target_minus_rival']
    if not math.isclose(primary_change+beta_change, delta, abs_tol=1e-10, rel_tol=1e-10):
        raise ValueError('Paired gap decomposition failed')
    return dict(seed=seed, eval_task=task, index=sample['index'], target=target, outcome=outcome,
        rival_class=rival, rival_is_wrong_final_of=wrong_arm, arms=arms, events=events,
        full_minus_class_gate=dict(beta=betas['full']-betas['class_gate'],
            primary_target_minus_rival=f['primary_target_minus_rival']-c['primary_target_minus_rival'],
            mixed_target_minus_rival=delta,
            primary_change_at_class_gate_beta=primary_change,
            beta_change_on_full_primary=beta_change))


def group_counts(cases):
    counts = Counter(samples=len(cases))
    for case in cases:
        f, c = case['arms']['full'], case['arms']['class_gate']
        delta = case['full_minus_class_gate']
        target = case['target']
        flags = dict(
            full_primary_correct=f['primary_class'] == target,
            class_gate_primary_correct=c['primary_class'] == target,
            both_primary_correct=f['primary_class'] == c['primary_class'] == target,
            rp_correct=f['rp_class'] == target,
            full_beta_higher=delta['beta'] > 1e-10,
            full_beta_lower=delta['beta'] < -1e-10,
            full_beta_equal=abs(delta['beta']) <= 1e-10,
            full_primary_gap_smaller=delta['primary_target_minus_rival'] < -1e-10,
            full_primary_gap_larger=delta['primary_target_minus_rival'] > 1e-10,
            full_primary_gap_equal=abs(delta['primary_target_minus_rival']) <= 1e-10,
            full_higher_beta_and_smaller_primary_gap=(delta['beta'] > 1e-10 and delta['primary_target_minus_rival'] < -1e-10),
            full_correct_with_class_gate_beta=f['use_other_arm_beta']['unique_class_if_clear'] == target,
            full_ambiguous_with_class_gate_beta=f['use_other_arm_beta']['unique_class_if_clear'] is None,
            class_gate_correct_with_full_beta=c['use_other_arm_beta']['unique_class_if_clear'] == target,
            class_gate_ambiguous_with_full_beta=c['use_other_arm_beta']['unique_class_if_clear'] is None)
        for arm in ARMS:
            for key in ('crm_evaluated', 'same_adapter_candidates', 'fusion_harmed', 'fusion_repaired',
                        'both_candidates_wrong', 'selected_wrong_other_correct'):
                flags[arm+'_'+key] = case['events'][arm][key]
        flags['full_duplicate_and_fusion_harmed'] = (
            case['events']['full']['same_adapter_candidates'] and case['events']['full']['fusion_harmed'])
        flags['full_both_candidates_wrong_and_fusion_harmed'] = (
            case['events']['full']['both_candidates_wrong'] and case['events']['full']['fusion_harmed'])
        counts.update({k: int(v) for k, v in flags.items()})
    return dict(counts)


def aggregate_cases(cases):
    harmed = [c for c in cases if c['outcome'] == 'harmed']
    repaired = [c for c in cases if c['outcome'] == 'repaired']
    fusion_harms = [c for c in harmed if c['events']['full']['fusion_harmed']]
    return dict(paired_harmed=group_counts(harmed), paired_repaired=group_counts(repaired),
        paired_harmed_full_fusion_harmed=group_counts(fusion_harms),
        paired_harmed_both_primary_correct=group_counts([
            c for c in harmed if all(c['events'][a]['primary_correct'] for a in ARMS)]))


def input_snapshot(root):
    names = []
    for seed in SEEDS:
        folder = root/'ima'/('seed%d' % seed)
        names.append(folder/'summary.json')
        for arm in ARMS:
            for task in range(10):
                names.extend((folder/arm/('task_%02d.jsonl' % task),
                              folder/arm/('candidate_task_%02d.json' % task)))
    return {p.relative_to(root).as_posix(): sha256(p) for p in names}


def analyze_batch(root):
    root = Path(root).resolve()
    before = input_snapshot(root)
    print('Verifying all four seeds, traces and captured scores...', flush=True)
    with redirect_stdout(io.StringIO()):
        verified = candidates.audit_batch(root)
    print('CRM_CANDIDATE_ALL_SEEDS_VERIFIED=4/4', flush=True)
    cases = []
    for seed, summary in verified['seeds'].items():
        folder = root/'ima'/('seed%d' % seed)
        keys_by_task = {}
        for case in summary['paired_class_changes']:
            keys_by_task.setdefault(case['eval_task'], {})[case['index']] = case
        for task, expected in sorted(keys_by_task.items()):
            records = {}
            for arm in ARMS:
                path = folder/arm/('candidate_task_%02d.json' % task)
                captured = json.loads(path.read_text(encoding='utf-8'))
                if sha256(path) != summary['score_sha256'][path.relative_to(folder).as_posix()]:
                    raise ValueError('Scores changed after validation: '+str(path))
                records[arm] = {r['sample']['index']: r for r in captured['samples']
                                if r['sample']['index'] in expected}
                if records[arm].keys() != expected.keys():
                    raise ValueError('Missing paired score record')
            for index, stored in sorted(expected.items()):
                for arm in ARMS:
                    record = records[arm][index]
                    _, detail = candidates.inspect_record(record, record['sample'], 200, 10)
                    check_summary(detail, stored[arm], 'paired score detail')
                case = analyze_pair(seed, task, records['full'][index], records['class_gate'][index])
                if case['outcome'] != stored['outcome']:
                    raise ValueError('Paired outcome differs from verified report')
                cases.append(case)
        print('CRM_ERROR_SEED_ANALYZED=%d' % seed, flush=True)
    if input_snapshot(root) != before:
        raise ValueError('Input files changed during analysis')
    return dict(schema=1, batch_dir=str(root), input_sha256=before,
        verification='FOUR_SEEDS_RECOMPUTED_AND_INPUT_FILES_UNCHANGED',
        original_aggregate=verified['aggregate'], groups=aggregate_cases(cases), cases=cases,
        limitations=[
            'Counts are prediction occurrences repeated across seeds, not distinct independent images.',
            'Wrong-to-correct and correct-to-wrong refer to final classes, comparing class-gate with Full.',
            'Both arms use the same wrong final class as rival within each paired comparison.',
            'Beta intervals use all classes, allow boundary ties and use labels only for reporting.',
            'Weight swaps use saved selected-adapter scores; no adapter or CRM decision is recomputed.',
            'The two gap components are an ordered arithmetic decomposition, not independent causal effects.',
            'Selected-error counterfactuals do not estimate dataset accuracy or validate a new gate.',
            'No training, evaluation, GPU access, paper change or algorithm change is performed.'])


def print_report(report):
    print('\nCOUNTS_REPEAT_IMAGES_ACROSS_SEEDS=1')
    for group, counts in report['groups'].items():
        print('ERROR_GROUP='+json.dumps(dict(group=group, **counts), sort_keys=True))
    print('\nPAIRED_CASES: target vs the SAME wrong final class in both arms; P and R are standardized gaps.')
    for case in report['cases']:
        print('CASE seed=%d task=%d index=%d outcome=%s target=%d rival=%d' % (
            case['seed'], case['eval_task'], case['index'], case['outcome'], case['target'], case['rival_class']))
        for arm in ARMS:
            a = case['arms'][arm]
            bounds = a['all_class_beta_interval']['target_is_maximum_beta_interval_0_half']
            print('  %s primary=%d final=%d RP=%d beta=%.6f P=%+.6f R=%+.6f mixed=%+.6f '
                  'target_max_beta=%s swapped_beta_class=%s duplicate=%s' % (
                      arm, a['primary_class'], a['final_class'], a['rp_class'], a['effective_rp_weight'],
                      a['primary_target_minus_rival'], a['rp_target_minus_rival'], a['combined_target_minus_rival'],
                      json.dumps(bounds), str(a['use_other_arm_beta']['unique_class_if_clear']),
                      str(case['events'][arm]['same_adapter_candidates'])))
    print('Intervals include ties. Swapped weights are report-only; these selected cases are not a new accuracy result.')
    print('CRM_ERROR_ANALYSIS_COMPLETE=4/4', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file outside the input batch; never overwritten.')
    args = parser.parse_args(argv)
    root, output = args.batch_dir.resolve(), args.output.resolve()
    if output == root or root in output.parents:
        parser.error('Output must be outside the input batch')
    if output.exists() or not output.parent.is_dir():
        parser.error('Output must be a new file in an existing directory')
    try:
        report = analyze_batch(root)
        with output.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps(report, indent=2, allow_nan=False)+'\n')
        print_report(report)
        print('REPORT='+str(output), flush=True)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise SystemExit('CRM_ERROR_ANALYSIS_STOP: '+str(exc)) from exc


if __name__ == '__main__':
    main()
