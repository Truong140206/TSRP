"""Read completed eight-arm experiments; recompute stage counts, never use GPU."""
import argparse
from collections import Counter
import json
from pathlib import Path
import statistics
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import audit_prediction_stages as stages
from tools import compare_prediction_traces as compare
from tools import fusion_logic_variants as variants
from tools import prediction_logic_probe as probe

PAIRS = dict(original=('class_gate', 'full'),
             gate_z=('class_gate_gate_z', 'full_gate_z'),
             crm_refresh=('class_gate_crm_refresh', 'full_crm_refresh'),
             both=('class_gate_both', 'full_both'))


def audit(run_dir):
    root = Path(run_dir).resolve()
    paths = [root / 'summary.json'] + [p for arm in variants.ARMS
             for p in sorted((root / arm).glob('*')) if p.is_file()]
    before = {p.relative_to(root).as_posix(): stages.sha256(p) for p in paths}
    stored = json.loads((root / 'summary.json').read_text(encoding='utf-8'))
    if (stored.get('dataset') != 'Split-Imagenet-A'
            or stored.get('historical_controls') != 'EXACT_TRACE_ROWS_BOTH_ARMS'
            or stored.get('historical_full_all_stages') != 'PASS_AT_1e-4_LOG_PRECISION'):
        raise ValueError('Expected a completed ImageNet-A logic experiment')
    expected_files = {f'{arm}/task_{t:02d}.jsonl' for arm in variants.ARMS for t in range(10)}
    if set(stored.get('trace_sha256', {})) != expected_files:
        raise ValueError('Incomplete stored trace fingerprints')
    if any(before.get(p) != digest for p, digest in stored['trace_sha256'].items()):
        raise ValueError('Trace differs from completed experiment')
    # Reconstruct experimental metrics using its own unchanged control rows.
    # Original history attestation is checked above, not independently rerun.
    reconstructed = probe.summarize(root, root)
    stages.check_summary(reconstructed, stored)
    rows = {}
    for arm in variants.ARMS:
        rows[arm] = []
        for task in range(10):
            header, samples = compare.load_task(root / arm / f'task_{task:02d}.jsonl')
            if header['seed'] != stored['seed'] or header['dataset'] != stored['dataset']:
                raise ValueError('Mixed dataset/seed')
            rows[arm].append(samples)
    result = {}
    for name, (left, right) in PAIRS.items():
        paired = {s: stages.paired_stats([
            [(a[s], b[s], a['target']) for a, b in zip(l, r)]
            for l, r in zip(rows[left], rows[right])]) for s in stages.STAGES}
        within = {arm: {f'{a}_to_{b}': stages.paired_stats([
            [(r[a], r[b], r['target']) for r in task] for task in rows[arm]])
            for a, b in zip(stages.STAGES, stages.STAGES[1:])} for arm in (left, right)}
        paths_by_outcome = {k: Counter() for k in ('repaired', 'harmed')}
        for l, r in zip(rows[left], rows[right]):
            for a, b in zip(l, r):
                path = [compare.outcome(a[s], b[s], a['target']) for s in stages.STAGES]
                if path[-1] in paths_by_outcome:
                    paths_by_outcome[path[-1]][' -> '.join(path)] += 1
        result[name] = dict(left=left, right=right, stage_pairs=paired, within_arm=within,
                           final_error_paths={k: dict(v) for k, v in paths_by_outcome.items()})
    if before != {p.relative_to(root).as_posix(): stages.sha256(p) for p in paths}:
        raise ValueError('Input changed during audit')
    return dict(seed=stored['seed'], dataset=stored['dataset'], conditions=result,
                checks='TRACE_HASHES_AND_RECOMPUTED_SUMMARY_PASS', input_sha256=before,
                limitation='Descriptive paths, not independent causal effects. '
                'No candidate logits or calibrated confidence are inferred from class IDs.')


def audit_batch(batch):
    reports = [audit(Path(batch) / 'ima' / f'seed{seed}') for seed in (42, 43, 44, 45)]
    if [r['seed'] for r in reports] != [42, 43, 44, 45]:
        raise ValueError('Expected seeds 42, 43, 44, 45 exactly once')
    means = {name: {stage: statistics.mean(
        r['conditions'][name]['stage_pairs'][stage]['task_balanced_net_pp'] for r in reports)
        for stage in stages.STAGES} for name in PAIRS}
    return dict(seeds=reports, mean_full_minus_class_gate=means,
                note='Equal task weights within each seed; equal seed weights. '
                'Do not pool repeated images across seeds as independent samples.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--batch-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = audit_batch(args.batch_dir)
    if args.output:
        stages.save_report(report, args.output, args.batch_dir)
    for item in report['seeds']:
        print('\nSEED', item['seed'], item['checks'])
        print('CONDITION          AFTER_DRM    AFTER_CRM    FINAL_CLASS')
        for name, data in item['conditions'].items():
            values = [data['stage_pairs'][s]['task_balanced_net_pp'] for s in stages.STAGES]
            print(f'{name:16s} ' + ' '.join(f'{v:+12.6f}' for v in values))
            print('CLASS_COUNTS=' + json.dumps({s: data['stage_pairs'][s]['counts'] for s in stages.STAGES}))
            print('FINAL_ERROR_PATHS=' + json.dumps(data['final_error_paths']))
    print('FOUR_SEED_MEANS=' + json.dumps(report['mean_full_minus_class_gate']))
    print('LOGIC_STAGE_AUDIT_COMPLETE=4/4')
    if args.output:
        print('REPORT=' + str(args.output.resolve()))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, KeyError, OSError, StopIteration) as error:
        raise SystemExit('LOGIC_STAGE_AUDIT_STOP: ' + str(error))
