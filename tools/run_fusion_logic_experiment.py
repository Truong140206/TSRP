"""Read-only preflight by default; --run explicitly runs the IMA logic experiment.

One historical Full reconstruction plus seven final-stage evaluations. Eight
arms: original/z-score gate/refreshed CRM/both for Full and class-gate. Never
overwrite historical data, retrain adapters or change the normal evaluator.
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
from tools import run_prediction_audit as base
from tools import audit_prediction_stages as stages
from tools import compare_prediction_traces as compare
from tools import prediction_logic_probe as probe
from tools import fusion_logic_variants as variants


def history_hashes(history):
    paths = [history / 'summary.json']
    for arm in ('full', 'class_gate'):
        paths += [history / arm / 'console.log', history / arm / 'run.json']
        paths += sorted((history / arm).glob('task_*.jsonl'))
    return {str(p): stages.sha256(p) for p in paths}


def check_launch(cmd, historical, arm):
    cfg = vars(base.grid.parsed_config(cmd))
    name = cmd[cmd.index('main.py') + 1]
    if name != 'ima_lora' or cfg['dataset'] != 'Split-Imagenet-A' or '--nproc_per_node=1' not in cmd:
        raise ValueError('Expected ImageNet-A single-process launch')
    cfg.update(config=name, subparser_name=name, world_size=1, rank=0, gpu=0,
               distributed=True, dist_backend='nccl', nb_classes=200)
    names = historical.get('class_names')
    if not isinstance(names, list) or len(names) != 200 or not all(isinstance(n, str) for n in names):
        raise ValueError('Historical ImageNet-A class names are incomplete')
    # Verify the actual labels again against historical trace headers in child.
    probe.check_arm_configuration(cfg, {k: v for k, v in historical.items() if k != 'class_names'}, arm)


def child_command(args, tii, lora, destination):
    cmd = base.build_command(args, tii, lora, 'full', destination / 'full')
    at = cmd.index('main.py')
    return (cmd[:at] + [str(ROOT / 'tools' / 'prediction_logic_probe.py'),
                       '--history', str(args.history), '--destination', str(destination), '--']
            + cmd[at + 1:])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--history', type=Path, required=True)
    parser.add_argument('--seed', type=int, choices=(42, 43, 44, 45), required=True)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Unsafe tag')
    for key in ('output_root', 'data_root', 'history'):
        setattr(args, key, getattr(args, key).resolve())
        if not getattr(args, key).is_dir():
            parser.error('Existing directory required: ' + key)
    destination = args.output_root / args.tag / 'ima' / ('seed%d' % args.seed)
    if destination.exists() or args.history == destination or args.history in destination.parents:
        raise ValueError('Preserve existing results; use a new tag outside the history directory')
    variants.checked_source()
    histories = history_hashes(args.history)
    audit = stages.audit(args.history)
    if audit['dataset'] != 'Split-Imagenet-A' or audit['seed'] != args.seed:
        raise ValueError('Historical dataset/seed differs')
    header, _ = compare.load_task(args.history / 'full' / 'task_00.jsonl')
    old_common = header['run']['common']
    args.dataset, args.batch_size, args.port = 'ima', 24, 29558
    args.data_path = Path(old_common['dataset_path'])
    args.resolved_data_path = base.resolve_data(args)
    if str(args.resolved_data_path) != old_common['dataset_path']:
        raise ValueError('Historical data path differs')
    tii = args.output_root / ('ima_tii_original_10tasks_seed%d' % args.seed)
    lora = args.output_root / ('ima_lora_rank8_baseline_10tasks_seed%d' % args.seed)
    plain = base.build_command(args, tii, lora, 'full')
    for arm in ('full', 'class_gate'):
        old_header, _ = compare.load_task(args.history / arm / 'task_00.jsonl')
        check_launch(base.build_command(args, tii, lora, arm), old_header['configuration'], arm)
    print('LOGIC_LAUNCH_CONFIG=PASS_BOTH_CONTROLS', flush=True)
    source = base.verify.source_digest()
    if source != old_common['source_sha256']:
        raise ValueError('Paper evaluator source differs from history')
    records, stamps = base.grid.manifest(tii, lora, base.grid.parsed_config(plain), None)
    if records != old_common['checkpoints']:
        raise ValueError('Historical checkpoint paths/contents differ')
    runtime = {name: importlib.metadata.version(name) for name in ('torch', 'torchvision', 'timm', 'numpy')}
    import torch
    runtime.update(python=sys.version, torch_cuda=torch.version.cuda,
                   cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if runtime != old_common['runtime']:
        raise ValueError('Historical runtime or CUDA_VISIBLE_DEVICES differs')
    drivers = {p.name: stages.sha256(p) for p in sorted((ROOT / 'tools').glob('*.py'))}
    common = dict(source_sha256=source, drivers=drivers, checkpoints=records,
                  runtime=runtime, dataset_path=str(args.resolved_data_path),
                  experiment='gate_zscore_and_post_drm_crm_v1', settings=variants.ARMS,
                  historical_sha256=histories)
    print('LOGIC_PREFLIGHT=PASS; NO_ADAPTER_TRAINING; 8 final arms', flush=True)
    print('DESTINATION=' + str(destination), flush=True)
    if not args.run:
        print('READ_ONLY_PREFLIGHT_ONLY; append --run when GPU is free', flush=True)
        return
    import fcntl
    with (args.output_root / '.tsrp_prediction_audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        base.idle_gpu()
        if shutil.disk_usage(args.output_root).free < 4 * 1024**3:
            raise RuntimeError('Need at least 4 GiB free disk')
        if history_hashes(args.history) != histories:
            raise ValueError('Historical files changed during preflight')
        destination.mkdir(parents=True, exist_ok=False)
        base.save_new(destination / 'plan.json', dict(common=common, arms=variants.ARMS))
        for arm in variants.ARMS:
            folder = destination / arm
            folder.mkdir()
            base.save_new(folder / 'run.json', dict(dataset_name='Split-Imagenet-A', seed=args.seed,
                                                   variant=arm, common=common))
        cmd = child_command(args, tii, lora, destination)
        base.grid.assert_unchanged(source, drivers, stamps)
        new_metrics = base.execute(destination / 'console.log', cmd, dict(command=cmd, common=common))
        historical = base.verify.read_stages(args.history / 'full' / 'console.log')
        if not base.verify.compare(historical, new_metrics, range(1, 11),
                                   base.verify.CORE + base.verify.RETENTION, label='LOGIC_HISTORICAL_FULL'):
            raise ValueError('Full sequence differs; do not interpret experimental results')
        report = probe.summarize(args.history, destination)
        stats = json.loads((destination / 'final_stats.json').read_text(encoding='utf-8'))
        for arm, values in stats.items():
            if abs(values['Acc@1'] - report['arms'][arm]['task_balanced_accuracy']['final_class']) > .00011:
                raise ValueError('Trace/evaluator accuracy mismatch: ' + arm)
        base.grid.assert_unchanged(source, drivers, stamps)
        if (history_hashes(args.history) != histories
                or any(stages.sha256(Path(r['path'])) != r['sha256'] for r in records)):
            raise ValueError('Historical input/checkpoint changed during experiment')
        report.update(dataset='Split-Imagenet-A', seed=args.seed, provenance=common,
                      historical_controls='EXACT_TRACE_ROWS_BOTH_ARMS',
                      historical_full_all_stages='PASS_AT_1e-4_LOG_PRECISION',
                      trace_sha256={str(p.relative_to(destination)): stages.sha256(p)
                                    for p in sorted(destination.glob('*/task_*.jsonl'))},
                      final_evaluator_stats=stats,
                      elapsed_seconds=base.evaluation_seconds(destination / 'console.log'))
        base.save_new(destination / 'summary.json', report)
        for arm, values in report['arms'].items():
            print('LOGIC_RESULT=' + json.dumps(dict(arm=arm,
                top1=values['task_balanced_accuracy']['final_class'],
                versus_old_full=values['against_original']['full']['task_balanced_net_pp'],
                versus_old_class_gate=values['against_original']['class_gate']['task_balanced_net_pp'])), flush=True)
        print('LOGIC_ROUTING_EFFECTS=' + json.dumps({k: v['task_balanced_net_pp']
              for k, v in report['conditional_routing'].items()}), flush=True)
        print('LOGIC_EXPERIMENT_COMPLETE=1/1', flush=True)
        print('SUMMARY=' + str(destination / 'summary.json'), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('LOGIC_EXPERIMENT_STOP: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
