"""CIFAR seed42 score audit of completed traces; evaluation only, no retraining.

Rebuild RP once via the full historical ten-stage evaluation, then evaluate
gated class-only at the final stage on that state. Refuse changed historical
inputs/configuration/checkpoints/runtime or any sample-level hash mismatch.
Default is read-only CPU preflight; --run needs a free GPU.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import audit_prediction_stages as stages
from tools import compare_prediction_traces as compare
from tools import run_prediction_audit as base
from tools import prediction_score_probe as probe


def fingerprint_history(history):
    files = [history / 'summary.json', history / 'full' / 'console.log']
    files += [p for arm in ('full', 'class_gate') for p in sorted((history / arm).glob('task_*.jsonl'))]
    return {str(p): stages.sha256(p) for p in files}


def child_command(args, tii, lora, destination):
    cmd = base.build_command(args, tii, lora, 'full', destination / 'full')
    at = cmd.index('main.py')
    return (cmd[:at] + [str(ROOT / 'tools' / 'prediction_score_probe.py'),
                       '--history', str(args.history), '--destination', str(destination), '--']
            + cmd[at + 1:])


def check_launch_configuration(cmd, historical):
    """Check the CIFAR single-process launch before any checkpoint/GPU work.

    main adds config names; torchrun/init_distributed_mode supplies process
    fields; the CIFAR loader adds nb_classes and class_names. Only class_names
    is deferred here. The final-stage observer still checks every field.
    """
    current = vars(base.grid.parsed_config(cmd))
    config_name = cmd[cmd.index('main.py') + 1]
    if (config_name != 'cifar100_lora' or current['dataset'] != 'Split-CIFAR100'
            or '--nproc_per_node=1' not in cmd):
        raise ValueError('Unsupported score-audit launch configuration')
    current.update(config=config_name, subparser_name=config_name,
                   world_size=1, rank=0, gpu=0, distributed=True, dist_backend='nccl',
                   nb_classes=100)
    names = historical.get('class_names')
    if not isinstance(names, list) or len(names) != 100 or not all(isinstance(n, str) for n in names):
        raise ValueError('Historical CIFAR class_names are incomplete')
    probe.check_configuration(current, {k: v for k, v in historical.items() if k != 'class_names'})


def combine_scores(destination, expected_samples):
    arms = {}
    for arm in ('class_gate', 'full'):
        records = json.loads((destination / arm / 'scores.json').read_text(encoding='utf-8'))
        arms[arm] = {(r['sample']['true_task'], r['sample']['index']): r for r in records}
        if len(records) != len(arms[arm]) or set(arms[arm]) != set(expected_samples):
            raise ValueError('Missing/duplicate selected score records')
        for task in {t for t, _ in expected_samples}:
            _, traced = compare.load_task(destination / arm / ('task_%02d.jsonl' % task))
            for key in (key for key in expected_samples if key[0] == task):
                expected = {k: v for k, v in traced[key[1]].items() if k != 'type'}
                if arms[arm][key]['sample'] != expected:
                    raise ValueError('Captured score identity differs from verified trace')
    result = []
    for key in sorted(expected_samples):
        left, right = (arms[a][key] for a in ('class_gate', 'full'))
        if left['scores']['rp_scores'] != right['scores']['rp_scores']:
            raise ValueError('RP score vectors differ across arms')
        for record in (left, right):
            s, row = record['scores'], record['sample']
            check = probe.describe_scores(s['primary_logits'], s['rp_scores'], s['final_logits'],
                                           row['target'], row['crm_class'], row['final_class'])
            stages.check_summary(check, s, 'captured_scores')
        entry = dict(eval_task=key[0], index=key[1], target=left['sample']['target'],
                     class_gate=left, full=right, report_only_counterfactuals={})
        for arm, own, other in (('class_gate', left, right), ('full', right, left)):
            pz, _, _ = probe.normalized(own['scores']['primary_logits'])
            rz, _, _ = probe.normalized(own['scores']['rp_scores'])
            entry['report_only_counterfactuals'][arm] = dict(
                use_other_arms_effective_rp_weight=probe.mix_prediction(
                    pz, rz, other['scores']['effective_rp_weight']),
                ungated_beta_half=probe.mix_prediction(pz, rz, .5))
        result.append(entry)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Invalid tag')
    for name in ('output_root', 'data_root', 'history'):
        setattr(args, name, getattr(args, name).resolve())
        if not getattr(args, name).is_dir():
            parser.error('Existing directory required: ' + name)
    destination = args.output_root / args.tag / 'cifar100' / 'seed42'
    if destination.exists() or args.history in destination.parents:
        raise ValueError('Preserve old output; choose a new tag outside the historical run')
    historical_hashes = fingerprint_history(args.history)
    audit = stages.audit(args.history)
    if audit['dataset'] != 'Split-CIFAR100' or audit['seed'] != 42:
        raise ValueError('This probe is scoped to the completed CIFAR-100 seed42 audit')
    header, _ = compare.load_task(args.history / 'full' / 'task_00.jsonl')
    old_common = header['run']['common']
    args.dataset, args.seed, args.batch_size, args.port, args.data_path = 'cifar100', 42, 24, 29558, None
    args.resolved_data_path = base.resolve_data(args)
    if str(args.resolved_data_path) != old_common['dataset_path']:
        raise ValueError('Historical data path differs')
    tii = args.output_root / 'cifar100_tii_original_10tasks_seed42'
    lora = args.output_root / 'cifar100_lora_rank8_baseline_10tasks_seed42'
    plain = base.build_command(args, tii, lora, 'full')
    for arm in ('full', 'class_gate'):
        old_header, _ = compare.load_task(args.history / arm / 'task_00.jsonl')
        check_launch_configuration(base.build_command(args, tii, lora, arm),
                                   old_header['configuration'])
    print('SCORE_AUDIT_LAUNCH_CONFIG=PASS_BOTH_ARMS', flush=True)
    source = base.verify.source_digest()
    if source != old_common['source_sha256']:
        raise ValueError('Evaluator source differs from historical run; do not bypass this check')
    records, stamps = base.grid.manifest(tii, lora, base.grid.parsed_config(plain), None)
    if records != old_common['checkpoints']:
        raise ValueError('Historical checkpoint contents/paths differ')
    runtime = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy')}
    import torch
    runtime.update(python=sys.version, torch_cuda=torch.version.cuda,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if runtime != old_common['runtime']:
        raise ValueError('Historical runtime or CUDA_VISIBLE_DEVICES differs')
    selected = {}
    for sample in audit['changed_class_samples']:
        selected.setdefault(sample['eval_task'], []).append(sample['index'])
    expected_samples = {(t, i) for t, indices in selected.items() for i in indices}
    if not expected_samples:
        raise ValueError('No historical correctness changes to inspect')
    historical_metrics = base.verify.read_stages(args.history / 'full' / 'console.log')
    drivers = {p.name: stages.sha256(p) for p in sorted((ROOT / 'tools').glob('*.py'))}
    common = dict(source_sha256=source, drivers=drivers, checkpoints=records,
                  runtime=runtime, dataset_path=str(args.resolved_data_path))
    print('SCORE_AUDIT_PREFLIGHT=PASS', flush=True)
    print('SELECTED_SAMPLES=%d; original Full sequence once + final class_gate only' % len(expected_samples), flush=True)
    print('NO_ADAPTER_TRAINING; RP statistics must be rebuilt. DESTINATION=' + str(destination), flush=True)
    if not args.run:
        print('READ_ONLY_PREFLIGHT_ONLY; use --run when GPU is free', flush=True)
        return
    import fcntl
    with (args.output_root / '.tsrp_prediction_audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.idle_gpu()
        if shutil.disk_usage(args.output_root).free < 4 * 1024**3:
            raise RuntimeError('Need at least 4 GiB free disk')
        if fingerprint_history(args.history) != historical_hashes:
            raise ValueError('Historical input changed during preflight')
        destination.mkdir(parents=True, exist_ok=False)
        base.save_new(destination / 'plan.json', dict(selected=selected, historical_sha256=historical_hashes))
        for arm in ('full', 'class_gate'):
            folder = destination / arm
            folder.mkdir()
            base.save_new(folder / 'run.json', dict(dataset_name='Split-CIFAR100', seed=42,
                                                   variant=arm, common=common))
        cmd = child_command(args, tii, lora, destination)
        base.grid.assert_unchanged(source, drivers, stamps)
        new_metrics = base.execute(destination / 'console.log', cmd, dict(command=cmd, common=common))
        if not base.verify.compare(historical_metrics, new_metrics, range(1, 11),
                                   base.verify.CORE + base.verify.RETENTION, label='HISTORICAL_FULL_ALL_STAGES'):
            raise ValueError('Full evaluation changed; do not interpret scores')
        result = compare.compare_directories(destination / 'class_gate', destination / 'full')
        old_result = compare.compare_directories(args.history / 'class_gate', args.history / 'full')
        stages.check_summary({k: v for k, v in old_result.items() if k != 'provenance'}, result)
        details = combine_scores(destination, expected_samples)
        base.grid.assert_unchanged(source, drivers, stamps)
        if (fingerprint_history(args.history) != historical_hashes
                or any(stages.sha256(Path(r['path'])) != r['sha256'] for r in records)):
            raise ValueError('Historical input/checkpoint changed during probe')
        focus_keys = {(s['eval_task'], s['index']) for s in audit['corrected_adapter_class_harms']}
        focus = [r for r in details if (r['eval_task'], r['index']) in focus_keys]
        report = dict(historical_sha256=historical_hashes, provenance=common,
                      captured_score_sha256={arm: stages.sha256(destination / arm / 'scores.json')
                                             for arm in ('class_gate', 'full')},
                      historical_full_all_stages='PASS_AT_1e-4_LOG_PRECISION',
                      historical_trace_rows='EXACT_BOTH_ARMS', comparison=result,
                      samples=details, corrected_adapter_class_harms=focus,
                      limitations='Selected errors from one seed, not a validation set or a new accuracy result. '
                      'Swapped gate weights and ungated outputs are fixed-score numerical counterfactuals, '
                      'not trained variants, deployment recommendations, or independent stage effects. '
                      'Tracing is not a performance measurement.')
        base.save_new(destination / 'score_report.json', report)
        for sample in focus:
            compact = {k: sample[k] for k in ('eval_task', 'index', 'target', 'report_only_counterfactuals')}
            for arm in ('class_gate', 'full'):
                compact[arm] = {k: v for k, v in sample[arm]['scores'].items()
                                if k not in ('primary_logits', 'rp_scores', 'final_logits')}
            print('CORRECTED_ADAPTER_CLASS_HARM_SCORES=' + json.dumps(compact, sort_keys=True), flush=True)
        print('SCORE_AUDIT_COMPLETE=1/1', flush=True)
        print('REPORT=' + str(destination / 'score_report.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('SCORE_AUDIT_STOP: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
