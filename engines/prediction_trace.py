"""Opt-in, final-stage observation only. Not a timing or memory benchmark.

Stores class/task decisions and hashes, never images or feature vectors.
No RNG calls, model forwards, parameter updates, or changes to logits.
"""
import hashlib
import json
from pathlib import Path


def encoded(value):
    return json.dumps(value, sort_keys=True, allow_nan=False, separators=(',', ':')) + '\n'


class TraceWriter:
    """An incomplete file has no completion footer and must not be analysed."""

    def __init__(self, path, header):
        self.path = Path(path)
        self.header = header
        self.count = self.correct = self.route_correct = 0
        self.digest = hashlib.sha256()
        self.pending = None
        with self.path.open('x', encoding='utf-8', newline='\n') as out:
            out.write(encoded(dict(type='header', schema=1, **header)))

    def append(self, records):
        with self.path.open('a', encoding='utf-8', newline='\n') as out:
            for row in records:
                if row['index'] != self.count:
                    raise ValueError('Nonsequential trace index')
                line = encoded(dict(type='sample', **row))
                out.write(line)
                self.digest.update(line.encode('utf-8'))
                self.count += 1
                self.correct += row['final_class'] == row['target']
                self.route_correct += row['proposal_after'] == row['true_task']

    def close(self, acc1, route_acc):
        if self.count != self.header['samples'] or self.count == 0:
            raise ValueError('Trace sample coverage is incomplete')
        observed = 100.0 * self.correct / self.count
        routing = 100.0 * self.route_correct / self.count
        if abs(observed - acc1) > 1e-4 or abs(routing - route_acc) > 1e-4:
            raise ValueError('Trace does not reproduce evaluator metrics')
        with self.path.open('a', encoding='utf-8', newline='\n') as out:
            out.write(encoded(dict(type='complete', samples=self.count,
                                   sha256=self.digest.hexdigest(), acc1=observed,
                                   proposal_accuracy=routing)))

    @staticmethod
    def hashes(tensor):
        cpu = tensor.detach().cpu().contiguous()
        description = str((tuple(cpu.shape[1:]), str(cpu.dtype))).encode()
        return [hashlib.sha256(description + row.numpy().tobytes()).hexdigest()
                for row in cpu]

    def begin(self, inputs, targets, tii, proposal, initial_logits, task_map):
        if self.pending is not None:
            raise RuntimeError('Previous trace batch not finished')
        labels = targets.detach().cpu().tolist()
        tii = tii.detach().cpu().tolist()
        proposal = proposal.detach().cpu().tolist()
        images = self.hashes(inputs)
        initial = self.hashes(initial_logits)
        self.pending = []
        for j, label in enumerate(labels):
            true_task = int(task_map[int(label)])
            if true_task != self.header['eval_task']:
                raise ValueError('Evaluation task and target mapping disagree')
            self.pending.append(dict(index=self.count + j, target=int(label),
                                     true_task=true_task, input_sha256=images[j],
                                     tii_task=int(tii[j]), proposal_before=int(proposal[j]),
                                     initial_logits_sha256=initial[j]))

    def routing(self, proposal, rp_scores):
        proposal = proposal.detach().cpu().tolist()
        hashes = self.hashes(rp_scores) if rp_scores is not None else [None] * len(proposal)
        classes = (rp_scores.argmax(1).detach().cpu().tolist()
                   if rp_scores is not None else [None] * len(proposal))
        for row, task, digest, cls in zip(self.pending, proposal, hashes, classes):
            row.update(proposal_after=int(task), rp_sha256=digest, rp_class=cls)

    @staticmethod
    def classes(logits):
        # Match timm accuracy(topk=(1, 5)), including its tie-breaking.
        return logits.topk(5, 1, True, True).indices[:, 0].detach().cpu().tolist()

    def drm(self, logits):
        for row, cls in zip(self.pending, self.classes(logits)):
            row['drm_class'] = int(cls)

    def crm(self, logits, indices, choices, alternatives):
        for row, cls in zip(self.pending, self.classes(logits)):
            row.update(crm_class=int(cls), final_adapter=row['proposal_after'],
                       crm_evaluated=False, crm_used_alternative=False)
        if choices is not None:
            for j, choice, alternative in zip(indices.detach().cpu().tolist(),
                                               choices.detach().cpu().tolist(),
                                               alternatives.detach().cpu().tolist()):
                row = self.pending[j]
                row['crm_evaluated'] = True
                row['crm_used_alternative'] = bool(choice)
                if choice == 1:
                    row['final_adapter'] = int(alternative)

    def end(self, logits):
        classes = self.classes(logits)
        hashes = self.hashes(logits)
        for row, cls, digest in zip(self.pending, classes, hashes):
            row.update(final_class=int(cls), final_logits_sha256=digest)
        self.append(self.pending)
        self.pending = None


def start_trace(args, loader, eval_task, stage, class_mask):
    directory = getattr(args, 'prediction_trace_dir', None)
    if not directory:
        return None
    import torch
    if not getattr(args, 'eval', False) or getattr(args, 'world_size', 1) != 1:
        raise ValueError('Prediction traces require evaluation and one process')
    if getattr(args, 'task_routing_mode', 'class') != 'class' or getattr(args, 'task_inc', False):
        raise ValueError('Prediction traces support the paper class-routing path only')
    if getattr(args, 'rp_head', False) and not getattr(args, 'rp_route_fusion_drm', False):
        raise ValueError('RP-only trace is not implemented')
    if not isinstance(loader.sampler, torch.utils.data.SequentialSampler) or loader.drop_last:
        raise ValueError('Prediction traces require sequential, complete test sampling')
    if stage != args.num_tasks:
        return None
    root = Path(directory)
    manifest = json.loads((root / 'run.json').read_text(encoding='utf-8'))
    if manifest['dataset_name'] != args.dataset or manifest['seed'] != args.seed:
        raise ValueError('Trace manifest does not match the evaluation')
    # All parser options are plain JSON values. Keep the complete effective config
    # so the pair checker can reject differences other than routing weight/path.
    config = dict(vars(args))
    header = dict(dataset=args.dataset, seed=args.seed, stage=stage,
                  eval_task=eval_task, samples=len(loader.dataset),
                  class_mask=class_mask, configuration=config, run=manifest)
    return TraceWriter(root / ('task_%02d.jsonl' % eval_task), header)
