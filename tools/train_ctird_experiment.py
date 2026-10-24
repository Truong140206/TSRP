"""Explicit fresh-run CTIRD experiment. Default: read-only preflight, no GPU.

With --run use single-process torchrun. Application arguments follow --.
Never resumes/overwrites an existing output or trains the historical baseline
directory. This is training, not an evaluation repair for old checkpoints.
"""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from tools import ctird_alignment_experiment as intervention
from tools.audit_prediction_stages import sha256


def preflight(args, mode):
    intervention.checked_source()
    if (mode not in intervention.MODES or args.config != 'ima_lora'
            or args.dataset != 'Split-Imagenet-A' or args.eval
            or args.train_inference_task_only or not args.train_mask
            or args.task_inc or args.lora_type != 'hide' or args.rp_head
            or args.prediction_trace_dir or args.num_tasks != 10
            or args.K < 1 or args.con <= 0 or args.epochs < 1):
        raise ValueError('Expected explicit ImageNet-A HideLoRA training with CTIRD')
    if not args.output_dir or not args.trained_original_model:
        raise ValueError('New output and existing TII directory are required')
    output = Path(args.output_dir).resolve()
    if (output.exists() or not output.parent.is_dir()
            or 'ctird_' + mode not in output.name):
        raise ValueError('Use a fresh directory named with ctird_' + mode + ' and an existing parent')
    count = args.max_train_tasks or args.num_tasks
    if not 2 <= count <= args.num_tasks:
        raise ValueError('CTIRD experiment needs between two and ten training tasks')
    tii = Path(args.trained_original_model).resolve()
    if output == tii or tii in output.parents or output in tii.parents:
        raise ValueError('Keep experimental outputs separate from TII inputs')
    from tools.run_prediction_audit import resolve_data
    data = resolve_data(SimpleNamespace(dataset='ima', data_path=Path(args.data_path),
                                        data_root=Path(args.data_path)))
    if data.resolve() != Path(args.data_path).resolve():
        raise ValueError('Pass the exact prepared data parent as --data-path')
    checkpoints = [tii / 'checkpoint' / f'task{t}_checkpoint.pth' for t in range(1, count + 1)]
    for path in checkpoints:
        if not path.is_file():
            raise ValueError('Missing TII checkpoint: ' + str(path))
    import torch
    hashes = {}
    for path in checkpoints:
        saved = torch.load(path, map_location='cpu', weights_only=False)
        configuration = saved.get('args')
        configuration = configuration if isinstance(configuration, dict) else vars(configuration)
        for key in ('dataset', 'seed', 'num_tasks', 'model', 'shuffle', 'input_size'):
            if configuration.get(key) != getattr(args, key):
                raise ValueError('TII checkpoint configuration mismatch: ' + key)
        hashes[str(path)] = sha256(path)
        del saved
    from tools.verify_paper_results import source_digest
    return dict(mode=mode, tasks=count, fresh_training=True, checkpoints=hashes,
                paper_source_sha256=source_digest(), entrypoint_sha256=sha256(__file__),
                runtime=dict(python=sys.version, packages={name: importlib.metadata.version(name)
                    for name in ('torch', 'torchvision', 'timm', 'numpy')}),
                engine_sha256=sha256(ROOT / 'engines' / 'hrm_lora_wtp_and_tap_engine.py'),
                intervention_sha256=sha256(intervention.__file__), arguments=vars(args))


def main():
    separator = sys.argv.index('--') if '--' in sys.argv else len(sys.argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=intervention.MODES, required=True)
    parser.add_argument('--run', action='store_true')
    options = parser.parse_args(sys.argv[1:separator])
    if separator == len(sys.argv):
        parser.error('Pass ima_lora application arguments after --')
    import main as application
    original_argv = sys.argv
    try:
        sys.argv = ['main.py', *original_argv[separator + 1:]]
        args = application.get_args()
    finally:
        sys.argv = original_argv
    plan = preflight(args, options.mode)
    print('CTIRD_PREFLIGHT=PASS; TASKS=%d; FRESH_TRAINING' % plan['tasks'], flush=True)
    if not options.run:
        print('READ_ONLY_PREFLIGHT_ONLY; no model training launched', flush=True)
        return
    if os.environ.get('WORLD_SIZE') != '1' or os.environ.get('LOCAL_RANK') != '0':
        raise ValueError('Use torchrun with --nproc_per_node=1 for --run')
    import fcntl
    import shutil
    import utils
    from engines import hrm_lora_wtp_and_tap_engine as engine
    from tools.run_prediction_audit import idle_gpu
    output = Path(args.output_dir).resolve()
    with (output.parent / '.tsrp_prediction_audit.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        idle_gpu()
        if shutil.disk_usage(output.parent).free < 12 * 1024**3:
            raise ValueError('Need 12 GiB free for a separate training run')
        if any(sha256(p) != digest for p, digest in plan['checkpoints'].items()):
            raise ValueError('TII inputs changed after preflight')
        output.mkdir(exist_ok=False)
        with (output / 'ctird_experiment.json').open('x', encoding='utf-8') as stream:
            json.dump(plan, stream, indent=2, allow_nan=False)
        args.ctird_experiment_mode = options.mode
        try:
            with intervention.installed(engine, options.mode):
                application.main(args)
        finally:
            utils.cleanup_distributed()
        for task in range(1, plan['tasks'] + 1):
            if not (output / 'checkpoint' / f'task{task}_checkpoint.pth').is_file():
                raise ValueError('Training returned without all expected checkpoints')
        if any(sha256(p) != digest for p, digest in plan['checkpoints'].items()):
            raise ValueError('TII input changed during training')
        print('CTIRD_TRAINING_COMPLETE=%d/%d' % (plan['tasks'], plan['tasks']), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        raise SystemExit('CTIRD_EXPERIMENT_STOP: ' + str(error))
