#!/usr/bin/env python3
"""One matched diagnostic case, using saved checkpoints (no adapter training).

Default: read-only CPU preflight. --run explicitly authorises evaluation.
First run compares full without/with tracing, then gated class-only. RP statistics
are rebuilt using the existing evaluation path; no old checkpoint/log is replaced.
This diagnostic is NOT a timing or memory benchmark. Only trusted checkpoints.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import finish_soict_checks as finish
from tools import verify_paper_results as verify
from tools import verify_paper_backbone_grid as grid
from tools.compare_prediction_traces import compare_directories


def save_new(path, data):
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def resolve_data(args):
    root = (args.data_path or args.data_root).resolve()
    if args.dataset == 'imr':
        return verify.resolve_imr_data_path(args.data_root, args.data_path)
    if args.dataset == 'cifar100':
        from torchvision.datasets import CIFAR100
        for train in (True, False):
            CIFAR100(str(root), train=train, download=False)
    else:
        # Do not call the CUB constructor: it may extract/re-split missing data.
        from torchvision.datasets import ImageFolder
        if not (root / 'CUB_200_2011.tgz').is_file():
            raise ValueError('CUB loader requires its existing archive; no download authorised')
        labels = []
        for split in ('train', 'test'):
            obj = ImageFolder(str(root / 'CUB_200_2011' / split))
            if len(obj.classes) != 200:
                raise ValueError('Expected existing 200-class CUB train/test split')
            labels.append(obj.classes)
        if labels[0] != labels[1]:
            raise ValueError('CUB train/test labels differ')
    return root


def idle_gpu():
    # Refuse a busy shared GPU. Never stop another process or wait invisibly.
    result = subprocess.run(['nvidia-smi', '--query-compute-apps=pid',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    if result.stdout.strip():
        raise RuntimeError('GPU has compute jobs. Retry when free; no jobs were stopped.\n'
                           + result.stdout)
    memory = subprocess.run(['nvidia-smi', '--query-gpu=memory.free',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True)
    if any(int(line.strip()) < 16384 for line in memory.stdout.splitlines()):
        raise RuntimeError('Need at least 16 GiB free GPU memory; retry when free')


def execute(path, cmd, meta):
    if path.exists():
        return grid.read_verified(path, meta, 10)
    idle_gpu()
    env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONDONTWRITEBYTECODE='1')
    print('RUN=' + str(path), flush=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('VERIFICATION_META=' + json.dumps(meta, sort_keys=True) + '\n')
        stream.write('COMMAND=' + json.dumps(cmd) + '\n')
        stream.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            stream.write(line)
            stream.flush()
            print(line, end='', flush=True)
        code = process.wait()
        stream.write('VERIFICATION_EXIT_CODE=%d\n' % code)
    if code:
        raise RuntimeError('Evaluation failed. Preserve log and use a new tag: ' + str(path))
    return grid.read_verified(path, meta, 10)


def build_command(args, tii, lora, variant, trace=None):
    cmd = finish.variant_command(args, args.seed, tii, lora,
                                 'vit_base_patch16_224', variant)
    if trace is not None:
        cmd += ['--prediction_trace_dir', str(trace)]
    return cmd


def check_trace_metrics(result, logs):
    balanced = result['task_balanced']
    for arm in ('full', 'class_gate'):
        for key, metric in (('accuracy', 'Acc@1'), ('proposal_accuracy', 'Acc@task')):
            if abs(balanced[key][arm] - logs[arm][10][metric]) > 0.00011:
                raise ValueError('Trace/log mismatch: ' + arm + '/' + metric)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--data-path', type=Path)
    parser.add_argument('--dataset', choices=verify.DATASETS, default='imr')
    parser.add_argument('--seed', type=int, choices=(42, 43, 44, 45), default=42)
    parser.add_argument('--tag', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.tag):
        parser.error('Tag may contain only letters, digits, underscores and hyphens')
    args.output_root, args.data_root = args.output_root.resolve(), args.data_root.resolve()
    if not args.output_root.is_dir() or not args.data_root.is_dir():
        parser.error('Existing output-root and data-root required')
    args.resolved_data_path = resolve_data(args)
    args.batch_size, args.port = 24, 29558
    lora = args.output_root / ('%s_lora_rank8_baseline_10tasks_seed%d' % (args.dataset, args.seed))
    tii = args.output_root / ('%s_tii_original_10tasks_seed%d' % (args.dataset, args.seed))
    destination = args.output_root / args.tag / args.dataset / ('seed%d' % args.seed)
    plain = build_command(args, tii, lora, 'full')
    expected = grid.parsed_config(plain)
    source = verify.source_digest()
    drivers = {p.name: finish.file_hash(p) for p in sorted((ROOT / 'tools').glob('*.py'))}
    records, stamps = grid.manifest(tii, lora, expected, None)
    runtime = {name: importlib.metadata.version(name)
               for name in ('torch', 'torchvision', 'timm', 'numpy')}
    runtime['python'] = sys.version
    import torch
    runtime['torch_cuda'] = torch.version.cuda
    runtime['cuda_visible_devices'] = os.environ.get('CUDA_VISIBLE_DEVICES')
    common = dict(source_sha256=source, drivers=drivers, checkpoints=records,
                  runtime=runtime, dataset_path=str(args.resolved_data_path))
    print('PREDICTION_AUDIT_PREFLIGHT=PASS', flush=True)
    print('CASE=%s seed%d; 3 evaluations: full plain, full traced, class_gate traced' %
          (args.dataset, args.seed), flush=True)
    print('No adapter training; RP statistics will be rebuilt. DESTINATION=' + str(destination), flush=True)
    if not args.run:
        print('READ_ONLY_PREFLIGHT_ONLY; append --run when the GPU is free', flush=True)
        return
    import fcntl  # Execution target is the Linux 4090 machine.
    with (args.output_root / '.tsrp_prediction_audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_gpu()
        if shutil.disk_usage(args.output_root).free < 4 * 1024**3:
            raise RuntimeError('Need at least 4 GiB free disk for diagnostic output')
        destination.mkdir(parents=True, exist_ok=True)
        logs = {}
        for name, variant, traced in (('full_plain', 'full', False),
                                      ('full', 'full', True), ('class_gate', 'class_gate', True)):
            grid.assert_unchanged(source, drivers, stamps)
            arm = destination / name
            manifest = dict(dataset_name=expected.dataset, seed=args.seed, variant=variant, common=common)
            if arm.exists():
                if json.loads((arm / 'run.json').read_text(encoding='utf-8')) != manifest:
                    raise ValueError('Existing run has different provenance; use a new tag')
                if not (arm / 'console.log').is_file():
                    raise ValueError('Existing incomplete directory; preserve it and use a new tag')
            else:
                arm.mkdir()
                save_new(arm / 'run.json', manifest)
            cmd = build_command(args, tii, lora, variant, arm if traced else None)
            meta = dict(manifest=manifest, command=cmd)
            logs[name] = execute(arm / 'console.log', cmd, meta)
            if name == 'full':
                if not verify.compare(logs['full_plain'], logs['full'], range(1, 11),
                                      verify.CORE + verify.RETENTION, label='OBSERVER_ALL_STAGES'):
                    raise ValueError('Observation changed results; do not interpret counts')
        grid.assert_unchanged(source, drivers, stamps)
        # Full hashes again at completion, not just file modification times.
        if any(finish.file_hash(Path(r['path'])) != r['sha256'] for r in records):
            raise ValueError('Checkpoint contents changed during evaluation')
        result = compare_directories(destination / 'class_gate', destination / 'full')
        check_trace_metrics(result, logs)
        result['observer_all_stages'] = 'PASS_AT_1e-4_LOG_PRECISION'
        result['single_seed_only'] = True
        summary = destination / 'summary.json'
        if summary.exists():
            if json.loads(summary.read_text(encoding='utf-8')) != result:
                raise ValueError('Existing summary differs; preserve it and use a new tag')
        else:
            save_new(summary, result)
        print('PREDICTION_AUDIT_COMPLETE=1/1', flush=True)
        print('COUNTS=' + json.dumps(result['counts'], sort_keys=True), flush=True)
        print('PROPOSAL_VS_CLASS=' + json.dumps(result['proposal_vs_class'], sort_keys=True), flush=True)
        print('FINAL_ADAPTER_VS_CLASS=' + json.dumps(result['final_adapter_vs_class'], sort_keys=True), flush=True)
        print('TASK_BALANCED=' + json.dumps(result['task_balanced'], sort_keys=True), flush=True)
        print('SUMMARY=' + str(summary), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('PREDICTION_AUDIT_STOP: ' + str(error), file=sys.stderr, flush=True)
        raise SystemExit(1)
