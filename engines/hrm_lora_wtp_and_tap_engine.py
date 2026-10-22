import copy
import math
import sys
import os
import datetime
import json
from typing import Iterable
from pathlib import Path
import torch
import torch.distributed as dist
import numpy as np
from torch.nn import functional as F
from timm.utils import accuracy
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from torch import optim
import utils
from engines.random_projection_head import accumulate_rp_statistics, fit_rp_temperature, rp_head_predict
from engines.prediction_trace import start_trace



def generalized_entropy(softmax_id_val, gamma, M):
        probs =  softmax_id_val 
        probs_sorted = np.sort(probs, axis=1)[:,-M:]
        scores = np.sum(probs_sorted**gamma * (1 - probs_sorted)**(gamma), axis=1)
           
        return -scores 

def compute_task_energy_scores(logits, class_mask, num_tasks, temperature=0.1):
    temperature = max(float(temperature), 1e-6)
    task_scores = []
    for task_idx in range(num_tasks):
        class_ids = torch.tensor(class_mask[task_idx], dtype=torch.long, device=logits.device)
        task_logits = logits.index_select(1, class_ids)
        task_scores.append(temperature * torch.logsumexp(task_logits / temperature, dim=1))
    return torch.stack(task_scores, dim=1)


def get_old_features(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    model.eval()
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    # metric_logger = utils.MetricLogger(delimiter="  ")
    # metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    # metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    # header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    all_res = []
    for input, target in data_loader:
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        temp = old_logits.index_fill(dim=1, index=torch.tensor(class_mask[task_id], dtype=torch.int64).to(device), value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

        # here is the trick to mask out classes of non-current tasks
        top_indices=None
        if task_id>0:
            probabilities = temp[:,:task_id*len(class_mask[0])]
            m = min(task_id, args.K)
            _, top_indices = torch.topk(probabilities, k=m, dim=1, largest=False)
            top5_id = []
            for i in range(top_indices.shape[0]):
                top5_id.append(torch.tensor([target_task_map[v.item()] for v in top_indices[i]]))
            top5_id = torch.stack(top5_id, dim=0).to(device, non_blocking=True)
        
        if task_id>0:
            # robust_logits = robust_loss(model, input, output['features'], target,device,task_id,class_mask,top5_id)
            all_old_logits = []
            for k in range(top5_id.shape[1]):
                prompt_id = top5_id[:,k]
                with torch.no_grad():
                    output = model(input, task_id=prompt_id)
                    old_logits = output['features']
                    old_norm_features = F.normalize(output['features'], p=2, dim=1)
                    old_similarity_matrix = torch.mm(old_norm_features, old_norm_features.t())
                    old_similarity_matrix = torch.exp(old_similarity_matrix)
                    old_similarity_matrix = old_similarity_matrix / old_similarity_matrix.sum(1, keepdim=True)
                    all_old_logits.append(old_similarity_matrix)

            all_res.append(all_old_logits)
                
    return all_res



def train_one_epoch(model: torch.nn.Module, original_model: torch.nn.Module,
                    criterion, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0,
                    set_training_mode=True, task_id=-1, class_mask=None, target_task_map=None,
                    args=None, old_features=None):
    model.train(set_training_mode)
    original_model.eval()

    if args.distributed and utils.get_world_size() > 1:
        data_loader.sampler.set_epoch(epoch)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))
    header = f'Train: Epoch[{epoch + 1:{int(math.log10(args.epochs)) + 1}}/{args.epochs}]'
    global_index = 0
    for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
        input = input.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    logits = output['logits']
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                        temp = old_logits.index_fill(dim=1, index=torch.tensor(class_mask[task_id], dtype=torch.int64).to(device), value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

            
        output = model(input, task_id=task_id, train=set_training_mode)
        logits = output['logits']
        # here is the trick to mask out classes of non-current tasks
        top_indices=None
        if task_id>0:
            m = min(task_id, args.K)
            probabilities = temp[:,:task_id*len(class_mask[0])]
            _, top_indices = torch.topk(probabilities, k=m, dim=1, largest=False)
            top5_id = []
            for i in range(top_indices.shape[0]):
                top5_id.append(torch.tensor([target_task_map[v.item()] for v in top_indices[i]]))
            top5_id = torch.stack(top5_id, dim=0).to(device, non_blocking=True)
        
        if task_id>0:
            robust_logits = old_features[global_index]
        
        if args.train_mask and class_mask is not None:
            mask = class_mask[task_id]
            not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
            not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
            logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
 
        
        loss_ctird = 0
        if task_id>0:

            loss = criterion(logits, target)
            for k in range(len(robust_logits)):
                norm_features = F.normalize(output['features'], p=2, dim=1)
                
                # 计算相似度矩阵
                similarity_matrix = torch.mm(norm_features, norm_features.t())
                similarity_matrix = torch.exp(similarity_matrix)
                similarity_matrix = similarity_matrix / similarity_matrix.sum(1, keepdim=True)
                pos_mask = (similarity_matrix > 0.038).float()
                relation_target = robust_logits[k]
                loss_ctird = F.kl_div(torch.log(similarity_matrix.clamp_min(1e-12)), relation_target, reduction='batchmean')
                
                loss = loss + args.con * loss_ctird
        else:
            loss = criterion(logits, target)+args.con*loss_ctird
            

        acc1, acc5 = accuracy(logits, target, topk=(1, 5))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()))
            sys.exit(1)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        torch.cuda.synchronize()
        metric_logger.update(Loss=loss.item())
        metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
        metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
        metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
        global_index = global_index+1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

con_num=0
incon_num=0
con_all=0
incon_all=0
max_p = []
cls_mean = {}
cls_cov = {}
_rp_frozen_extractor = {'model': None}


def _rp_inputs(inputs, args):
    """Match the pretrained ViT's expected input range for the frozen head.

    build_transform() emits ToTensor() output in [0, 1] with no Normalize, but
    the ImageNet-21K ViT-B/16 checkpoint was trained on [-1, 1] (mean=std=0.5).
    HRM-PET is unaffected because it trains its LoRA and classifier on whatever
    features it gets, but a training-free head inherits the mismatch directly.
    Only the RP head's extraction path is rescaled; the LoRA path keeps the
    range its adapters were trained on.
    """
    mode = str(getattr(args, 'rp_input_norm', 'none'))
    if mode == 'half':
        return (inputs - 0.5) / 0.5
    if mode == 'cifar_half':
        # build_cifar_transform() already normalizes with CIFAR statistics, so
        # undo that first; applying the [-1, 1] rescale on top of it would be a
        # double normalization.
        mean = torch.tensor([0.5071, 0.4867, 0.4408], device=inputs.device)
        std = torch.tensor([0.2675, 0.2565, 0.2761], device=inputs.device)
        raw = inputs * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)
        return (raw - 0.5) / 0.5
    return inputs


def pin_rp_extractor(original_model, args):
    """Snapshot a fixed feature extractor for the RP head.

    Default: the task-1 TII model. Evaluation reloads the TII checkpoint for
    every task, so accumulating the Gram matrix straight from `original_model`
    would mix ten different extractors; pinning the first-task snapshot keeps
    the feature space constant.

    With --rp_bare_extractor the snapshot is a freshly created *pretrained*
    backbone that has never seen the task sequence. This is the only way to
    measure RanPAC's "no first-session adaptation" condition here, because
    `original_model` is NOT the raw pretrained ViT: its weights are overwritten
    wholesale from the TII checkpoint (trainers/lora_trainer.py:171). So a run
    with rp_feature_source=original still reads features off an *adapted*
    backbone, which is why our 56.36 on ImageNet-R was never comparable to
    RanPAC's 71.8 for that ablation.
    """
    if not bool(getattr(args, 'rp_pin_extractor', False)):
        return None
    if _rp_frozen_extractor['model'] is None:
        if bool(getattr(args, 'rp_bare_extractor', False)):
            from timm.models import create_model
            # --rp_bare_model overrides the checkpoint for THIS snapshot
            # only. Everything else keeps args.original_model, which is
            # what the LoRA and TII checkpoints were trained against.
            bare_name = (str(getattr(args, 'rp_bare_model', '') or '')
                         .strip() or args.original_model)
            if bare_name != args.original_model:
                print('RP head: bare snapshot uses', bare_name,
                      'instead of', args.original_model)
            snapshot = create_model(
                bare_name,
                pretrained=True,
                num_classes=args.nb_classes,
                mlp_structure=args.original_model_mlp_structure,
            )
            snapshot.to(next(original_model.parameters()).device)
            print('RP head: BARE pretrained extractor -- the TII checkpoint is '
                  'deliberately NOT loaded into this snapshot.')
        else:
            snapshot = copy.deepcopy(original_model)
        snapshot.eval()
        for parameter in snapshot.parameters():
            parameter.requires_grad = False
        _rp_frozen_extractor['model'] = snapshot
    return _rp_frozen_extractor['model']


def rp_extractor(original_model, args):
    """Pinned extractor when enabled, otherwise the current TII model."""
    pinned = _rp_frozen_extractor['model']
    if bool(getattr(args, 'rp_pin_extractor', False)) and pinned is not None:
        return pinned
    return original_model


def _task_scores_from_class_scores(scores, class_mask, seen_task_count, device):
    """Reduce per-class scores to one score per task (max over its classes)."""
    out = torch.full((scores.shape[0], seen_task_count), float('-inf'),
                     device=device, dtype=torch.float32)
    for task in range(seen_task_count):
        index = torch.as_tensor(
            [int(c) for c in class_mask[task]], dtype=torch.long, device=device)
        out[:, task] = scores.index_select(1, index).max(dim=1).values
    return out


def _stage_ramp(seen_tasks, total_tasks, gamma):
    """How much of the second opinion this stage has earned, in (0, 1].

    Both fusions currently lend the same weight at every stage, and that is why
    Forgetting and Backward barely move: they compare a task's final accuracy
    against its own earlier peak, so a correction applied equally at every stage
    cancels out of both. Constant weight is also wrong on its own terms. The RP
    head's Gram estimate needs classes before it is trustworthy -- while TII has
    more tasks to tell apart as they accumulate. Both say the second opinion
    should be trusted more as the sequence grows.

    NOTE 2026-08-28: the figure this docstring used to quote (-1.38 and -0.95 at
    stages 1-2 on ImageNet-R) was a single seed and does NOT generalise. Twelve
    (dataset, seed) cells now say the harm is confined to stage 1; stage 2 is
    already net positive, and withholding the blend there costs up to -1.5 on
    CUB-200. See rp_class_fusion_min_tasks below.

    gamma = 0 reproduces the constant weight exactly, and the ramp reaches 1.0
    at the final stage, so the reported final row is untouched either way.

    Which fusion to ramp is not a free choice, and ramping both was wrong.
    Per-stage accuracy on seed 42 shows the RP head hurting the *classifier* at
    stage 1 (ImageNet-R 86.0 against the baseline's 87.5) exactly as the Gram
    argument predicts, so ramping the class blend recovers real accuracy. But
    the same head is the *better router* from the first stage (77.1 against
    TII's 63.8), so ramping the route weight only threw routing away: average
    incremental Acc@task fell on all three datasets (86.30 -> 85.66 on
    ImageNet-R at gamma=2). Hence the scope flag, and hence 'class' rather than
    'both'.
    """
    if gamma <= 0.0:
        return 1.0
    total = max(1, int(total_tasks))
    seen = min(max(1, int(seen_tasks)), total)
    return (float(seen) / float(total)) ** float(gamma)


def fuse_routers(rp_scores, tii_logits, class_mask, seen_task_count, args,
                 device):
    """Route by combining the RP head and TII, which fail on different samples.

    Measured on seed 42: the RP head routes better than raw TII everywhere
    (ImageNet-R 77.1 vs 63.8, CIFAR 89.5 vs 81.2, CUB 94.0 vs 93.0), and the
    union of the two exceeds HRM-PET's own post-DRM/CRM routing on all three
    (83.0 vs 77.8, 92.9 vs 89.8, 96.3 vs 93.2). Both are mapped to
    log-probabilities over tasks before mixing, since their raw scales differ.
    """
    weight = float(getattr(args, 'rp_route_fusion_weight', 0.5))
    # Weight sits on TII, so the RP share is 1 - weight; ramp that share.
    if getattr(args, 'rp_fusion_ramp_scope', 'both') in ('both', 'route'):
        weight = 1.0 - (1.0 - weight) * _stage_ramp(
            seen_task_count, getattr(args, 'num_tasks', seen_task_count),
            getattr(args, 'rp_fusion_ramp', 0.0))
    rp_task = _task_scores_from_class_scores(
        torch.nan_to_num(rp_scores.float(), neginf=-1e4),
        class_mask, seen_task_count, device)
    tii_task = _task_scores_from_class_scores(
        torch.nan_to_num(tii_logits.float(), neginf=-1e4),
        class_mask, seen_task_count, device)

    def standardize(scores):
        # log_softmax preserves the spread of its input, and TII logits are on a
        # far wider scale than ridge scores, so mixing them directly lets TII
        # dominate at any weight. Standardizing per sample first makes the
        # weight mean what it says.
        # unbiased=False on purpose. At stage 1 there is exactly one task, so
        # the Bessel-corrected std divides by n-1 = 0 and yields NaN, which
        # clamp_min cannot repair -- NaN compares false against everything. The
        # NaN was harmless only because argmax over a single column returns 0
        # whatever it holds, i.e. the answer was right by luck. The population
        # std gives 0 there, which clamp_min does repair. For every stage past
        # the first the two differ by sqrt(n/(n-1)), a constant shared by both
        # routers, so it cancels in the argmax and no measured number moves.
        centered = scores - scores.mean(dim=1, keepdim=True)
        return centered / centered.std(
            dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)

    fused = (weight * standardize(tii_task)
             + (1.0 - weight) * standardize(rp_task))
    return fused.argmax(dim=1)


args_ref = [None]


def _valid_moments(scores, valid):
    """Per-sample mean and std over the valid classes only."""
    masked = torch.where(valid, scores, torch.zeros_like(scores))
    count = valid.sum(dim=1, keepdim=True).clamp_min(1).float()
    mean = masked.sum(dim=1, keepdim=True) / count
    centered = torch.where(valid, scores - mean, torch.zeros_like(scores))
    std = (centered.pow(2).sum(dim=1, keepdim=True) / count).sqrt()
    return mean, std.clamp_min(1e-6)


def _standardize_valid(scores, valid):
    """Zero-mean unit-variance over a sample's valid classes only."""
    mean, std = _valid_moments(scores, valid)
    return torch.where(valid, scores - mean, torch.zeros_like(scores)) / std


def _top2_margin(scores, valid):
    """Normalised gap between a sample's top two classes, in [0, 1].

    Dividing by the sum of the top two rather than by 1 makes the quantity a
    function of the ratio p2/p1 alone: (1 - r) / (1 + r). It therefore measures
    how threatened the winner is by the runner-up, independently of how much
    probability mass the pair holds -- which is what we want, since fusion can
    only change a prediction by unseating the winner.
    """
    probs = torch.softmax(
        torch.where(valid, scores.float(),
                    torch.full_like(scores, -1e4, dtype=torch.float32)),
        dim=1)
    top2 = probs.topk(2, dim=1).values
    return ((top2[:, :1] - top2[:, 1:2])
            / top2.sum(dim=1, keepdim=True).clamp_min(1e-12)).clamp(0.0, 1.0)


gate_stats = {}


def _fusion_gate(routed_logits, valid, mode, rp_scores=None):
    """How much this sample should listen to the second classifier, in [0, 1].

    A fixed blend spends the same 30% of the decision on the RP head for a
    sample the routed head calls at p=0.97 as for one it is torn on. The first
    kind is the overwhelming majority on CIFAR-100, and paying the blend there
    is what made Loss regress (+0.019 +- 0.003 over three seeds) while accuracy
    still improved: flattening an already-correct peak costs cross-entropy and
    buys no argmax. Fusion can only change a prediction by unseating the top
    class, and its most likely challenger is the runner-up, so gate on how
    decided the routed head is between its own top two.

    'margin' looks at one side only. It hands the RP head up to 46% of the
    decision whenever the routed head is torn -- even when the RP head is torn
    too and has nothing to contribute. 'margin_both' closes that gap by scaling
    the same quantity by the RP head's own margin:

        margin:       beta_i = beta * (1 - conf(routed))
        margin_both:  beta_i = beta * (1 - conf(routed)) * conf(rp)

    so the blend opens only when the routed head is undecided AND the RP head
    is decided. Setting conf(rp) = 1 recovers 'margin' exactly, which gives the
    mode an identity check of the same kind as w = 1.0 and beta = 0.
    """
    # The mean gate is recorded so a run can be read without guessing. The
    # first margin_both attempt was inconclusive precisely because the gate's
    # magnitude was invisible: its numbers rose monotonically with beta, which
    # is the signature of under-fusion, but nothing in the log said by how much
    # the gate had shrunk. GateShare and ConfRP make that direct.
    gate_stats.clear()
    if mode in (None, 'none'):
        return None
    if mode == 'margin':
        gate = (1.0 - _top2_margin(routed_logits, valid)).clamp(0.0, 1.0)
        gate_stats['gate'] = gate.mean().item()
        return gate
    if mode == 'margin_both':
        if rp_scores is None:
            raise ValueError(
                "rp_class_fusion_gate='margin_both' needs the RP scores; "
                'the caller passed none')
        # Put the RP scores on the routed logits' scale before measuring their
        # margin. Ridge scores span about 1 unit while the routed logits span
        # about 35, so a softmax on the raw scores leaves every RP margin near
        # 0.05 -- the gate then comes out ~20x too small and the mode silently
        # under-fuses. Measured 2026-08-28 before this fix: 74.59 / 74.63 /
        # 74.67 at beta 0.5 / 0.8 / 1.0 on ImageNet-R seed 42, against 75.51
        # for one-sided 'margin', rising monotonically with beta -- the
        # signature of under-fusion, not of a wrong hypothesis.
        #
        # This is the same affine map the mixing below uses, so conf(rp) = 1
        # still recovers 'margin' exactly.
        mean, std = _valid_moments(routed_logits.float(), valid)
        rp_on_scale = _standardize_valid(rp_scores.float(), valid) * std + mean
        undecided = 1.0 - _top2_margin(routed_logits, valid)
        conf_rp = _top2_margin(rp_on_scale, valid)
        gate = (undecided * conf_rp).clamp(0.0, 1.0)
        gate_stats['gate'] = gate.mean().item()
        gate_stats['conf_rp'] = conf_rp.mean().item()
        gate_stats['undecided'] = undecided.mean().item()
        return gate
    if mode == 'entropy':
        probs = torch.softmax(
            torch.where(valid, routed_logits.float(),
                        torch.full_like(routed_logits, -1e4, dtype=torch.float32)),
            dim=1)
        entropy = -(probs * probs.clamp_min(1e-12).log()).sum(dim=1, keepdim=True)
        classes = valid.sum(dim=1, keepdim=True).clamp_min(2).float()
        gate = (entropy / classes.log()).clamp(0.0, 1.0)
        gate_stats['gate'] = gate.mean().item()
        return gate
    raise ValueError('unknown rp_class_fusion_gate: %s' % mode)


def fuse_class_scores(routed_logits, rp_scores, weight, seen_tasks=None):
    """Mix the routed HRM-PET classifier with the RP head as a second classifier.

    Routing gains convert into Acc@1 poorly -- the shared head often already
    predicts the right class despite a wrong route, so fixing that route buys
    nothing and can even flip a correct prediction. The RP head is a full
    classifier that routing discards, and it is the stronger of the two on
    CUB-200 (87.96 against the baseline's 86.53) while weaker on ImageNet-R, so
    a blend can beat either alone.

    Ridge scores and softmax logits live on different scales, so each is
    standardized over the sample's valid classes before mixing; masked classes
    stay masked.
    """
    valid = torch.isfinite(routed_logits)
    routed = routed_logits.float()
    if (seen_tasks is not None
            and getattr(args_ref[0], 'rp_fusion_ramp_scope', 'both')
            in ('both', 'class')):
        weight = weight * _stage_ramp(
            seen_tasks, getattr(args_ref[0], 'num_tasks', seen_tasks),
            getattr(args_ref[0], 'rp_fusion_ramp', 0.0))
    # neginf=-1e4 here, not 0.0 as in the mixing below: the gate feeds these
    # straight into a softmax, so an unseen class must stay far away rather
    # than land in the middle of the distribution.
    gate = _fusion_gate(
        routed, valid,
        getattr(args_ref[0], 'rp_class_fusion_gate', 'none'),
        rp_scores=torch.nan_to_num(rp_scores.float(), neginf=-1e4))
    share = weight if gate is None else weight * gate
    mixed = ((1.0 - share) * _standardize_valid(routed, valid)
             + share * _standardize_valid(
                 torch.nan_to_num(rp_scores.float(), neginf=0.0), valid))
    # Standardizing puts the mixture at unit variance, which is the wrong scale
    # for cross-entropy and made Loss regress even while accuracy improved. Map
    # it back onto the routed logits' own scale: an affine map per sample, so
    # the ranking -- and therefore accuracy -- is untouched, while the loss
    # becomes comparable to the baseline's again. (This also explains why
    # --rp_calibrate was a no-op here: standardization cancels any scalar
    # temperature applied to the RP scores.)
    mean, std = _valid_moments(routed, valid)
    # Sharpening factor: mixing two classifiers shrinks the winner's margin
    # relative to the rest, which cross-entropy penalises even when the ranking
    # improves. Scaling is monotone per sample, so accuracy is bit-identical and
    # only the loss moves.
    sharpen = max(1e-3, float(getattr(args_ref[0], 'rp_class_fusion_sharpen', 1.0)))
    mixed = mixed * sharpen * std + mean
    return torch.where(
        valid, mixed, torch.full_like(mixed, float('-inf')))


def _blend_scores(head_scores, classifier_logits, weight):
    """Combine head scores with the HRM-PET classifier on a common scale.

    The two live on different scales (ridge regression values vs softmax
    logits), so each is converted to log-probabilities before mixing; masked
    classes stay at -inf.
    """
    finite = torch.isfinite(head_scores)
    head_log = torch.full_like(head_scores, float('-inf'))
    head_log[finite] = head_scores[finite]
    head_log = torch.log_softmax(head_log, dim=1)
    classifier_log = torch.log_softmax(
        classifier_logits.masked_fill(~finite, float('-inf')), dim=1)
    return head_log + weight * classifier_log


def _mask_evaluation_logits(logits, class_mask, seen_task_count, args,
                            evaluation_task=None):
    """Mask every candidate BEFORE DRM/CRM selection and class fusion."""
    if class_mask is None:
        return logits
    if bool(getattr(args, 'task_inc', False)) and evaluation_task is not None:
        allowed = class_mask[evaluation_task]
    elif bool(getattr(args, 'train_mask', False)):
        allowed = [c for task in class_mask[:seen_task_count] for c in task]
    else:
        return logits
    excluded = torch.ones(logits.shape[1], dtype=torch.bool, device=logits.device)
    excluded[torch.as_tensor(allowed, dtype=torch.long, device=logits.device)] = False
    return logits.masked_fill(excluded.unsqueeze(0), float('-inf'))


@torch.no_grad()
def evaluate(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
             device, i=-1, task_id=-1, class_mask=None, target_task_map=None, args=None, ):
    global con_num, incon_num ,max_p, con_all, incon_all
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test: [Task {}]'.format(i + 1)
    prediction_trace = start_trace(args, data_loader, i, task_id + 1, class_mask)

    # switch to evaluation mode
    model.eval()
    original_model.eval()

    # Explicit sentinel rather than a `'fusion_rp_scores' in dir()` probe. dir()
    # reads the function's locals, so once any batch assigns the name it stays
    # visible to every later batch -- a batch that failed to compute its own
    # scores would silently be judged against the previous batch's. That cannot
    # happen today because the branch is gated on an args flag that is constant
    # across batches, but the guard should not depend on that staying true.
    fusion_rp_scores = None
    with torch.no_grad():
        for input, target in metric_logger.log_every(data_loader, args.print_freq, header):
            fusion_rp_scores = None
            input = input.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            # compute output
            with torch.no_grad():
                if original_model is not None:
                    output = original_model(input)
                    shared_features = output.get('pre_logits')
                    logits = output['logits']
                    old_logits = logits
                    if args.train_mask and class_mask is not None:
                        mask = []
                        for id in range(task_id + 1):
                            mask.extend(class_mask[id])
                        not_mask = np.setdiff1d(np.arange(args.nb_classes), mask)
                        not_mask = torch.tensor(not_mask, dtype=torch.int64).to(device)
                        old_logits = logits.index_fill(dim=1, index=not_mask, value=float('-inf'))
                    
                else:
                    raise NotImplementedError("original model is None")

            if (bool(getattr(args, 'rp_head', False))
                    and not bool(getattr(args, 'rp_route_fusion_drm', False))):
                blend_weight = float(getattr(args, 'rp_logit_blend', 0.0))
                adapter_logits = None
                if str(getattr(args, 'rp_feature_source', 'original')) == 'original':
                    if (bool(getattr(args, 'rp_pin_extractor', False))
                            or str(getattr(args, 'rp_input_norm', 'none')) != 'none'):
                        features = rp_extractor(original_model, args)(
                            _rp_inputs(input, args))['pre_logits']
                    else:
                        features = shared_features
                    if blend_weight != 0.0:
                        adapter_logits = old_logits
                else:
                    adapter_output = model(
                        input, task_id=int(getattr(args, 'rp_lora_task', 0)),
                        train=True)
                    features = adapter_output['pre_logits']
                    # Same forward, so blending the HRM-PET classifier costs
                    # no extra adapter call.
                    adapter_logits = adapter_output['logits']
                logits = rp_head_predict(features, args, device)
                if bool(getattr(args, 'rp_route_fusion', False)):
                    routed = fuse_routers(
                        logits, old_logits, class_mask, task_id + 1, args,
                        device)
                    routed_logits = model(input, task_id=routed)['logits']
                    if class_mask is not None:
                        seen = []
                        for seen_task in range(task_id + 1):
                            seen.extend(class_mask[seen_task])
                        unseen = np.setdiff1d(np.arange(args.nb_classes), seen)
                        routed_logits = routed_logits.index_fill(
                            dim=1,
                            index=torch.tensor(
                                unseen, dtype=torch.int64, device=device),
                            value=float('-inf'))
                    loss = criterion(routed_logits, target)
                    acc1, acc5 = accuracy(routed_logits, target, topk=(1, 5))
                    task_inference_acc = utils.task_inference_accuracy(
                        routed.unsqueeze(-1), target, target_task_map,
                        torch.empty(0, dtype=torch.long, device=device), None)
                    metric_logger.meters['Loss'].update(loss.item())
                    metric_logger.meters['Acc@1'].update(
                        acc1.item(), n=input.shape[0])
                    metric_logger.meters['Acc@5'].update(
                        acc5.item(), n=input.shape[0])
                    metric_logger.meters['Acc@task'].update(
                        task_inference_acc.item(), n=input.shape[0])
                    metric_logger.meters['LoRA/sample'].update(
                        2.0, n=input.shape[0])
                    metric_logger.meters['ForwardCalls/sample'].update(
                        2.0, n=input.shape[0])
                    continue
                if blend_weight != 0.0 and adapter_logits is not None:
                    logits = _blend_scores(
                        logits, adapter_logits.float(), blend_weight)
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                predicted_class = logits.argmax(dim=1)
                prompt_id = torch.tensor(
                    [target_task_map[int(c)] for c in predicted_class.cpu()],
                    device=device)
                task_inference_acc = utils.task_inference_accuracy(
                    prompt_id.unsqueeze(-1), target, target_task_map,
                    torch.empty(0, dtype=torch.long, device=device), None)
                if bool(getattr(args, 'rp_route_audit', False)):
                    # Are the RP head and TII wrong on the SAME samples? If
                    # their routing errors are decorrelated, fusing them can
                    # recover accuracy that neither reaches alone.
                    true_task = torch.tensor(
                        [target_task_map[int(t)] for t in target.cpu()],
                        device=device)
                    tii_class = old_logits.argmax(dim=1)
                    tii_task = torch.tensor(
                        [target_task_map[int(c)] for c in tii_class.cpu()],
                        device=device)
                    tii_ok = tii_task.eq(true_task)
                    rp_ok = prompt_id.eq(true_task)
                    metric_logger.meters['RouteTII'].update(
                        tii_ok.float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteRP'].update(
                        rp_ok.float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteUnion'].update(
                        (tii_ok | rp_ok).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteBoth'].update(
                        (tii_ok & rp_ok).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteAgree'].update(
                        tii_task.eq(prompt_id).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                    metric_logger.meters['RouteRPOnly'].update(
                        (rp_ok & ~tii_ok).float().mean().mul(100.0).item(),
                        n=input.shape[0])
                metric_logger.meters['Loss'].update(loss.item())
                metric_logger.meters['Acc@1'].update(
                    acc1.item(), n=input.shape[0])
                metric_logger.meters['Acc@5'].update(
                    acc5.item(), n=input.shape[0])
                metric_logger.meters['Acc@task'].update(
                    task_inference_acc.item(), n=input.shape[0])
                metric_logger.meters['LoRA/sample'].update(
                    0.0 if str(getattr(
                        args, 'rp_feature_source', 'original')) == 'original'
                    else 1.0, n=input.shape[0])
                metric_logger.meters['ForwardCalls/sample'].update(
                    1.0, n=input.shape[0])
                continue








            lora_id = torch.max(old_logits, dim=1)[1]
            lora_id = torch.tensor([target_task_map[v.item()] for v in lora_id], device=device)
            
            output = model(input, task_id=lora_id)
            logits = output['logits']
            

            logits = _mask_evaluation_logits(
                logits, class_mask, task_id + 1, args, evaluation_task=i)
            id_logits = logits
            routing_mode = str(getattr(args, 'task_routing_mode', 'class')).lower()
            if routing_mode == 'task_energy':
                task_scores = compute_task_energy_scores(
                    id_logits, class_mask, task_id + 1,
                    temperature=getattr(args, 'task_routing_temperature', 0.1))
                candidate_count = min(2, task_scores.shape[1])
                top5_id = torch.topk(task_scores, k=candidate_count, dim=1, largest=True).indices
                prompt_id = top5_id[:, 0]
            else:
                prompt_class = torch.max(id_logits, dim=1)[1]
                _, top_5_indices = torch.topk(id_logits, 2, dim=1)
                task_map_keys = list(target_task_map.keys())
                task_map_values = torch.tensor([target_task_map[k] for k in task_map_keys], device=device)
                task_map_tensor = torch.zeros((max(task_map_keys) + 1,), dtype=torch.long, device=device)
                task_map_tensor[task_map_keys] = task_map_values
                top5_id = torch.index_select(task_map_tensor, 0, top_5_indices.view(-1)).view_as(top_5_indices)
                prompt_id = torch.tensor([target_task_map[v.item()] for v in prompt_class], device=device)
            if prediction_trace is not None:
                prediction_trace.begin(input, target, lora_id, prompt_id, logits, target_task_map)
            if bool(getattr(args, 'rp_route_fusion_drm', False)):
                # Replace only HRM-PET's first stage. Its DRM/CRM refinement is
                # what carries the baseline on ImageNet-R, so it is kept; the
                # fused router just hands it a better starting task.
                if str(getattr(args, 'rp_feature_source', 'original')) == 'original':
                    rp_features = rp_extractor(original_model, args)(
                        _rp_inputs(input, args))['pre_logits']
                else:
                    rp_features = model(
                        input, task_id=int(getattr(args, 'rp_lora_task', 0)),
                        train=True)['pre_logits']
                fusion_rp_scores = rp_head_predict(rp_features, args, device)
                prompt_id = fuse_routers(
                    fusion_rp_scores, id_logits,
                    class_mask, task_id + 1, args, device)
            if prediction_trace is not None:
                prediction_trace.routing(prompt_id, fusion_rp_scores)
            ##############
            # target_id = torch.tensor([target_task_map[v.item()] for v in target], device=device)

            
            ###########
            id_logits = F.softmax(id_logits,dim=1)
            values, _ = torch.topk(id_logits, 1, dim=1)
            if args.En == 'msp':
                low_confidence_indices = torch.where(values <args.tau)[0]
            else:
                softmax_ood = id_logits
                softmax_ood = softmax_ood.cpu().numpy()
                #energy = (0.1*torch.logsumexp(inp / 0.1, dim=1))
                en = generalized_entropy(softmax_ood, 0.1, 20)
                
                en = torch.tensor(en)
                en = en.to(device, non_blocking=True)
                low_confidence_indices = torch.where(en <args.tau)[0]


            filtered_index_tensor = low_confidence_indices    

            
            equal_drm = torch.nonzero(prompt_id != lora_id).flatten()
            output_drm = model(input[equal_drm], task_id=prompt_id[equal_drm])
            output_drm_logits = _mask_evaluation_logits(
                output_drm['logits'], class_mask, task_id + 1, args,
                evaluation_task=i)
            logits[equal_drm] = output_drm_logits
            if prediction_trace is not None:
                prediction_trace.drm(logits)
            #promtp_idx = output['prompt_idx']  # tensor B x topk
            
            corr_id=None
            re_id = None
            if task_id>0:
                    error_input = input[filtered_index_tensor]
                    # error_target = target[filtered_index_tensor]
                    ensemble_id = top5_id[filtered_index_tensor,:2]
                    #ensemble_id = second_largest_classes
                    error_output = []
                    error_output.append(logits[filtered_index_tensor,:])
                    # for i in range(ensemble_id.shape[1]):
                    out = model(error_input, task_id=ensemble_id[:,1])
                    alternative_logits = _mask_evaluation_logits(
                        out['logits'], class_mask, task_id + 1, args,
                        evaluation_task=i)
                    error_output.append(alternative_logits)
                    error_output = torch.stack(error_output,dim=1)
                   
                    entropy = (0.1*torch.logsumexp( error_output/ 0.1, dim=2))
                    corr_id = torch.max(entropy, dim=1)[1]
                    # print(corr_id)
                    corr = corr_id.unsqueeze(1).unsqueeze(2)
                    result = torch.gather(error_output, 1, corr.expand(-1, 1, len(class_mask[0])*len(class_mask)))

                    result = result.squeeze(1)
                    logits[filtered_index_tensor]=result
                    if routing_mode == 'task_energy':
                        selected_tasks = torch.gather(ensemble_id, 1, corr_id.unsqueeze(1)).squeeze(1)
                        prompt_id[filtered_index_tensor] = selected_tasks
                        re_id = selected_tasks
                    # re_id = []
                    # for k in range(len(corr_id)):
                    #     #print(corr_id[k].item())
                    #     re_id.append(ensemble_id[k,corr_id[k].item()].item())
                    # re_id = torch.tensor(re_id)
                    # re_id = re_id.to(device, non_blocking=True)
                    # try:
                    #     prompt_id[filtered_index_tensor] = re_id
                    # except:
                    #     pass
                    
            
            if prediction_trace is not None:
                prediction_trace.crm(logits, filtered_index_tensor, corr_id,
                                     ensemble_id[:, 1] if task_id > 0 else None)

            # HRM-PET (DRM+CRM) cost: 1 base LoRA per sample, +1 for DRM-rerouted
            # samples, +1 for CRM-rematched low-confidence samples (task_id > 0).
            default_lora_counts = torch.ones(
                input.shape[0], dtype=torch.float32, device=device)
            default_lora_counts[equal_drm] += 1.0
            if task_id > 0:
                default_lora_counts[filtered_index_tensor] += 1.0
            if bool(getattr(args, 'rp_route_fusion_drm', False)):
                # Router fusion runs one extra adapter forward per sample to
                # extract the RP head's features; count it so the reported cost
                # is the method's real cost, not the baseline's.
                default_lora_counts += 1.0

            metric_logger.meters['LoRA/sample'].update(
                default_lora_counts.mean().item(), n=input.shape[0])
            metric_logger.meters['ForwardCalls/sample'].update(
                default_lora_counts.mean().item(), n=input.shape[0])

            loss = criterion(logits, target)

            acc1, acc5 = accuracy(logits, target, topk=(1, 5))

            task_inference_acc = utils.task_inference_accuracy(prompt_id.unsqueeze(-1), target, target_task_map, filtered_index_tensor,re_id)

            # Capture primary correctness before stage 2 replaces the logits.
            routed_ok = None
            if (bool(getattr(args, 'classifier_union_audit', False))
                    and fusion_rp_scores is not None):
                routed_ok = logits.argmax(dim=1).eq(target)
            class_weight = float(getattr(args, 'rp_class_fusion_weight', 0.0))
            # The RP head needs enough classes before its Gram estimate is
            # reliable, so the blend can be held back for the first tasks.
            #
            # Measured over 12 (dataset, seed) cells, 2026-08-28: the harm is
            # confined to stage 1, where withholding the blend gains +0.1 to
            # +1.5 and never loses. By stage 2 the blend is already net
            # positive -- withholding it there costs -0.3 to -0.4 on CIFAR-100
            # and -1.1 to -1.5 on CUB-200. Stages 3+ are bit-identical either
            # way, and so is the final row.
            #
            # min_tasks=3 is therefore WORSE than the default (AIA@1 down in 9
            # of 12 cells). min_tasks=2 is a Pareto gain but only +0.04 AIA@1 on
            # average, below the noise floor of everything we report, so the
            # shipped default stays 1.
            if task_id + 1 < int(getattr(args, 'rp_class_fusion_min_tasks', 1)):
                class_weight = 0.0
            if class_weight != 0.0 and fusion_rp_scores is None:
                # Fail loudly. fusion_rp_scores only exists on the
                # --rp_route_fusion_drm path, so asking for class fusion
                # without it used to skip stage 2 in silence -- and a run that
                # silently measured the baseline would look like a measurement
                # of the blend.
                raise RuntimeError(
                    '--rp_class_fusion_weight is %.3f but the RP scores were '
                    'never computed; class fusion requires '
                    '--rp_route_fusion_drm. Refusing to skip stage 2 silently.'
                    % class_weight)
            if class_weight != 0.0 and fusion_rp_scores is not None:
                args_ref[0] = args
                logits = fuse_class_scores(
                    logits, fusion_rp_scores, class_weight, task_id + 1)
                loss = criterion(logits, target)
                acc1, acc5 = accuracy(logits, target, topk=(1, 5))
                # GateShare is the effective per-sample weight the RP head
                # actually receives, beta times the gate -- not the nominal
                # beta. Reading it next to Acc@1 tells a mode that lost because
                # it mixed too little apart from one that lost because mixing
                # more was the wrong thing to do.
                if 'gate' in gate_stats:
                    metric_logger.meters['GateShare'].update(
                        class_weight * gate_stats['gate'], n=input.shape[0])
                if 'conf_rp' in gate_stats:
                    metric_logger.meters['ConfRP'].update(
                        gate_stats['conf_rp'], n=input.shape[0])
                    metric_logger.meters['Undecided'].update(
                        gate_stats['undecided'], n=input.shape[0])
            if (bool(getattr(args, 'classifier_union_audit', False))
                    and fusion_rp_scores is not None):
                # Routing gains convert poorly into Acc@1 because the shared
                # classifier often gets the class right despite a wrong route.
                # This measures whether the RP head, used as a classifier in its
                # own right, is correct where the routed head is wrong.
                rp_ok = fusion_rp_scores.argmax(dim=1).eq(target)
                metric_logger.meters['ClsRouted'].update(
                    routed_ok.float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['ClsRP'].update(
                    rp_ok.float().mean().mul(100.0).item(), n=input.shape[0])
                metric_logger.meters['ClsUnion'].update(
                    (routed_ok | rp_ok).float().mean().mul(100.0).item(),
                    n=input.shape[0])
                metric_logger.meters['ClsRPOnly'].update(
                    (rp_ok & ~routed_ok).float().mean().mul(100.0).item(),
                    n=input.shape[0])
            metric_logger.meters['Loss'].update(loss.item())
            metric_logger.meters['Acc@1'].update(acc1.item(), n=input.shape[0])
            metric_logger.meters['Acc@5'].update(acc5.item(), n=input.shape[0])
            metric_logger.meters['Acc@task'].update(task_inference_acc.item(), n=input.shape[0])
            if prediction_trace is not None:
                prediction_trace.end(logits)

    if prediction_trace is not None:
        prediction_trace.close(metric_logger.meters['Acc@1'].global_avg,
                               metric_logger.meters['Acc@task'].global_avg)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(
        '* Acc@task {task_acc.global_avg:.3f} Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
        .format(task_acc=metric_logger.meters['Acc@task'],
                top1=metric_logger.meters['Acc@1'], top5=metric_logger.meters['Acc@5'],
                losses=metric_logger.meters['Loss']))
    if 'GateShare' in metric_logger.meters:
        line = '* GateShare {g.global_avg:.4f}'.format(
            g=metric_logger.meters['GateShare'])
        if 'ConfRP' in metric_logger.meters:
            line += (' Undecided {u.global_avg:.4f} ConfRP {c.global_avg:.4f}'
                     .format(u=metric_logger.meters['Undecided'],
                             c=metric_logger.meters['ConfRP']))
        print(line)
    if bool(getattr(args, 'stage_drift_audit', False)):
        print(
            '* StageDriftAudit OwnLocalAcc@1 {local_acc.global_avg:.3f} '
            'OwnSeenAcc@1 {seen_acc.global_avg:.3f} '
            'OwnLocalLoss {local_loss.global_avg:.4f} '
            'OwnSeenLoss {seen_loss.global_avg:.4f} '
            'OwnSeenTaskAcc {task_acc.global_avg:.3f} '
            'LocalToSeenFailure {failure.global_avg:.3f}'
            .format(
                local_acc=metric_logger.meters['OwnLocalAcc@1'],
                seen_acc=metric_logger.meters['OwnSeenAcc@1'],
                local_loss=metric_logger.meters['OwnLocalLoss'],
                seen_loss=metric_logger.meters['OwnSeenLoss'],
                task_acc=metric_logger.meters['OwnSeenTaskAcc'],
                failure=metric_logger.meters['LocalToSeenFailure']))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate_till_now(model: torch.nn.Module, original_model: torch.nn.Module, data_loader,
                      device, task_id=-1, class_mask=None, target_task_map=None, acc_matrix=None, args=None, ):
    global con_num, incon_num ,con_all, incon_all
    
    stat_matrix = np.zeros((160, args.num_tasks))

    for i in range(task_id + 1):
        con_num=0
        incon_num=0
        con_all = 0
        incon_all=0
        test_stats = evaluate(model=model, original_model=original_model, data_loader=data_loader[i]['val'],
                              device=device, i=i, task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                              args=args)

        stat_matrix[0, i] = test_stats['Acc@1']
        stat_matrix[1, i] = test_stats['Acc@5']
        stat_matrix[2, i] = test_stats['Loss']
        stat_matrix[3, i] = test_stats['Acc@task']
        stat_matrix[4, i] = test_stats.get('CandidateRecall', 0.0)
        stat_matrix[5, i] = test_stats.get('LoRA/sample', 0.0)
        stat_matrix[6, i] = test_stats.get('FallbackRate', 0.0)
        if bool(getattr(args, 'rp_route_audit', False)):
            for offset, name in enumerate((
                    'RouteTII', 'RouteRP', 'RouteUnion', 'RouteBoth',
                    'RouteAgree', 'RouteRPOnly')):
                stat_matrix[124 + offset, i] = test_stats.get(name, 0.0)
        if bool(getattr(args, 'classifier_union_audit', False)):
            for offset, name in enumerate(
                    ('ClsRouted', 'ClsRP', 'ClsUnion', 'ClsRPOnly')):
                stat_matrix[155 + offset, i] = test_stats.get(name, 0.0)
        if bool(getattr(args, 'report_conventional_cost', False)):
            stat_matrix[150, i] = test_stats.get('LoRA/sample', 0.0)
            stat_matrix[151, i] = test_stats.get('ForwardCalls/sample', 0.0)
        if bool(getattr(args, 'router_recall_audit', False)):
            router_base = {'max': 130, 'energy': 135, 'margin': 140,
                           'mean': 145}
            for router_name, base in router_base.items():
                stat_matrix[base, i] = test_stats.get(
                    'Router_{}_MeanRank'.format(router_name), 0.0)
                for offset, recall_k in enumerate((1, 2, 3, 4), start=1):
                    stat_matrix[base + offset, i] = test_stats.get(
                        'Router_{}_Recall@{}'.format(router_name, recall_k),
                        0.0)
        if bool(getattr(args, 'stage_drift_audit', False)):
            stat_matrix[98, i] = test_stats.get('OwnLocalAcc@1', 0.0)
            stat_matrix[99, i] = test_stats.get('OwnSeenAcc@1', 0.0)
            stat_matrix[100, i] = test_stats.get('OwnLocalLoss', 0.0)
            stat_matrix[101, i] = test_stats.get('OwnSeenLoss', 0.0)
            stat_matrix[102, i] = test_stats.get('OwnSeenTaskAcc', 0.0)
            stat_matrix[103, i] = test_stats.get(
                'LocalToSeenFailure', 0.0)

        acc_matrix[i, task_id] = test_stats['Acc@1']

    avg_stat = np.divide(np.sum(stat_matrix, axis=1), task_id + 1)

    diagonal = np.diag(acc_matrix)

    result_str = "[Average accuracy till task{}]\tAcc@task: {:.4f}\tAcc@1: {:.4f}\tAcc@5: {:.4f}\tLoss: {:.4f}".format(
        task_id + 1,
        avg_stat[3],
        avg_stat[0],
        avg_stat[1],
        avg_stat[2])
    if bool(getattr(args, 'rp_route_audit', False)):
        result_str += "	RouteTII: {:.4f}	RouteRP: {:.4f}	RouteUnion: {:.4f}	RouteBoth: {:.4f}	RouteAgree: {:.4f}	RouteRPOnly: {:.4f}".format(
            *[np.mean(stat_matrix[124 + k, :task_id + 1]) for k in range(6)])
    if bool(getattr(args, 'classifier_union_audit', False)):
        result_str += "	ClsRouted: {:.4f}	ClsRP: {:.4f}	ClsUnion: {:.4f}	ClsRPOnly: {:.4f}".format(
            *[np.mean(stat_matrix[155 + k, :task_id + 1]) for k in range(4)])
    if bool(getattr(args, 'report_conventional_cost', False)):
        result_str += "\tLoRA/sample: {:.4f}\tForwardCalls/sample: {:.4f}".format(
            avg_stat[150], avg_stat[151])
    if bool(getattr(args, 'router_recall_audit', False)):
        router_base = {'max': 130, 'energy': 135, 'margin': 140, 'mean': 145}
        for router_name, base in router_base.items():
            result_str += (
                "\tRouter_{0}_MeanRank: {1:.4f}"
                "\tRouter_{0}_Recall@1: {2:.4f}\tRouter_{0}_Recall@2: {3:.4f}"
                "\tRouter_{0}_Recall@3: {4:.4f}\tRouter_{0}_Recall@4: {5:.4f}"
            ).format(
                router_name, avg_stat[base], avg_stat[base + 1],
                avg_stat[base + 2], avg_stat[base + 3], avg_stat[base + 4])
    if bool(getattr(args, 'stage_drift_audit', False)):
        result_str += (
            "\tOwnLocalAcc@1: {:.4f}\tOwnSeenAcc@1: {:.4f}"
            "\tOwnLocalLoss: {:.4f}\tOwnSeenLoss: {:.4f}"
            "\tOwnSeenTaskAcc: {:.4f}\tLocalToSeenFailure: {:.4f}"
        ).format(
            avg_stat[98], avg_stat[99], avg_stat[100], avg_stat[101],
            avg_stat[102], avg_stat[103])
    if task_id > 0:
        forgetting = np.mean((np.max(acc_matrix, axis=1) -
                              acc_matrix[:, task_id])[:task_id])
        backward = np.mean((acc_matrix[:, task_id] - diagonal)[:task_id])

        result_str += "\tForgetting: {:.4f}\tBackward: {:.4f}".format(forgetting, backward)
    print(result_str)

    return test_stats


def train_and_evaluate(model: torch.nn.Module, model_without_ddp: torch.nn.Module, original_model: torch.nn.Module,
                       criterion, data_loader: Iterable, data_loader_per_cls: Iterable,
                       optimizer: torch.optim.Optimizer,
                       lr_scheduler,
                       device: torch.device,
                       class_mask=None, target_task_map=None, args=None, ):
    # create matrix to save end-of-task accuracies
    acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    pre_ca_acc_matrix = np.zeros((args.num_tasks, args.num_tasks))
    global cls_mean
    global cls_cov
    cls_mean = dict()
    cls_cov = dict()
    norm_blend_enabled = bool(getattr(args, 'continual_norm_blend', False))
    norm_update_ratio = min(
        1.0, max(0.0, float(getattr(args, 'continual_norm_update_ratio', 0.25))))

    task_count = args.num_tasks
    max_train_tasks = int(getattr(args, 'max_train_tasks', 0))
    if max_train_tasks > 0:
        task_count = min(task_count, max_train_tasks)
        if utils.is_main_process():
            print('Limiting run to', task_count, 'of', args.num_tasks, 'tasks')
    for task_id in range(task_count):
        previous_fc_norm = None
        if norm_blend_enabled and task_id > 0:
            previous_fc_norm = {
                name: parameter.detach().clone()
                for name, parameter in model_without_ddp.fc_norm.named_parameters()
            }
        # Create new optimizer for each task to clear optimizer status
        if task_id > 0 and args.reinit_optimizer:
            optimizer = create_optimizer(args, model)
            
            if args.sched != 'constant':
                lr_scheduler, _ = create_scheduler(args, optimizer)
            elif args.sched == 'constant':
                lr_scheduler = None

        # load original model checkpoint
        if args.trained_original_model:
            original_checkpoint_path = os.path.join(args.trained_original_model,
                                                    'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            if os.path.exists(original_checkpoint_path):
                print('Loading checkpoint from:', original_checkpoint_path)
                original_checkpoint = utils.load_checkpoint(original_checkpoint_path, map_location=device)
                original_model.load_state_dict(original_checkpoint['model'], strict=False)
            else:
                print('No checkpoint found at:', original_checkpoint_path)
                return
       
        if task_id > 0:
            with torch.no_grad():
                if args.distributed:
                    model.module.lora_layer.k_lora_A.grad.zero_()
                    model.module.lora_layer.k_lora_A[task_id] = model.module.lora_layer.k_lora_A[task_id-1]
                    model.module.lora_layer.k_lora_B.grad.zero_()
                    model.module.lora_layer.k_lora_B[task_id] = model.module.lora_layer.k_lora_B[task_id-1]
                    model.module.lora_layer.v_lora_A.grad.zero_()
                    model.module.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                    model.module.lora_layer.v_lora_B.grad.zero_()
                    model.module.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]
                else:
                    model.lora_layer.k_lora_A.grad.zero_()
                    model.lora_layer.k_lora_A[task_id] = model.lora_layer.k_lora_A[task_id-1]
                    model.lora_layer.k_lora_B.grad.zero_()
                    model.lora_layer.k_lora_B[task_id] = model.lora_layer.k_lora_B[task_id-1]
                    model.lora_layer.v_lora_A.grad.zero_()
                    model.lora_layer.v_lora_A[task_id] = model.module.lora_layer.v_lora_A[task_id-1]
                    model.lora_layer.v_lora_B.grad.zero_()
                    model.lora_layer.v_lora_B[task_id] = model.module.lora_layer.v_lora_B[task_id-1]

        if task_id > 0:
            old_features = get_old_features(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=0, max_norm=args.clip_grad,
                                            set_training_mode=False, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, )
        else:
            old_features = None

        for epoch in range(args.epochs):
            # model.module.init_weights_proj()
            train_stats = train_one_epoch(model=model, original_model=original_model, criterion=criterion,
                                            data_loader=data_loader[task_id]['train'], optimizer=optimizer,
                                            device=device, epoch=epoch, max_norm=args.clip_grad,
                                            set_training_mode=True, task_id=task_id, class_mask=class_mask,
                                            target_task_map=target_task_map, args=args, old_features=old_features)

            if lr_scheduler:
                lr_scheduler.step(epoch)
        model_without_ddp.after_task(task_id=task_id, device=device)

        if previous_fc_norm is not None:
            squared_update_before = 0.0
            squared_update_after = 0.0
            with torch.no_grad():
                for name, parameter in model_without_ddp.fc_norm.named_parameters():
                    previous = previous_fc_norm[name].to(parameter.device)
                    update = parameter - previous
                    squared_update_before += update.float().pow(2).sum().item()
                    parameter.mul_(norm_update_ratio).add_(
                        previous, alpha=1.0 - norm_update_ratio)
                    retained_update = parameter - previous
                    squared_update_after += retained_update.float().pow(2).sum().item()
            if utils.is_main_process():
                print(
                    'Continual norm blend:',
                    'update_ratio=', norm_update_ratio,
                    'delta_before=', math.sqrt(squared_update_before),
                    'delta_after=', math.sqrt(squared_update_after),
                )

        if args.lora_momentum > 0 and task_id > 0:
            with torch.no_grad():
                model.module.lora_layer.k_lora_A[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.k_lora_A[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.k_lora_A[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.k_lora_B[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.k_lora_B[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.k_lora_B[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.v_lora_A[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.v_lora_A[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.v_lora_A[0:task_id].detach().clone().mean(dim=0))
                model.module.lora_layer.v_lora_B[task_id].copy_(
                    (1 - args.lora_momentum) * model.module.lora_layer.v_lora_B[task_id].detach().clone()
                    + args.lora_momentum * model.module.lora_layer.v_lora_B[0:task_id].detach().clone().mean(dim=0))


        # compute mean and variance
        _compute_mean(model=model, data_loader=data_loader_per_cls, device=device, task_id=task_id,
                      class_mask=class_mask[task_id], args=args)


        if task_id > 0 and not args.not_train_ca:
            pre_ca_test_stats = evaluate_till_now(
                model=model, original_model=original_model,
                data_loader=data_loader, device=device,
                task_id=task_id, class_mask=class_mask,
                target_task_map=target_task_map,
                acc_matrix=pre_ca_acc_matrix, args=args)
            #train_dis(model, args, device,data_loader[task_id]['train'], class_mask,target_task_map, task_id)
            train_task_adaptive_prediction(
                model, args, device, class_mask, task_id,
                data_loader_per_cls=data_loader_per_cls)

        
        test_stats = evaluate_till_now(model=model, original_model=original_model, data_loader=data_loader,
                                       device=device,
                                       task_id=task_id, class_mask=class_mask, target_task_map=target_task_map,
                                       acc_matrix=acc_matrix, args=args)
        if args.output_dir and utils.is_main_process():
            Path(os.path.join(args.output_dir, 'checkpoint')).mkdir(parents=True, exist_ok=True)

            checkpoint_path = os.path.join(args.output_dir, 'checkpoint/task{}_checkpoint.pth'.format(task_id + 1))
            state_dict = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'args': args,
            }
            if args.sched is not None and args.sched != 'constant':
                state_dict['lr_scheduler'] = lr_scheduler.state_dict()

            utils.save_on_master(state_dict, checkpoint_path)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                    **{f'test_{k}': v for k, v in test_stats.items()},
                    }

        if args.output_dir and utils.is_main_process():
            with open(os.path.join(args.output_dir,
                                '{}_stats.txt'.format(datetime.datetime.now().strftime('log_%Y_%m_%d_%H_%M'))),
                    'a') as f:
                f.write(json.dumps(log_stats) + '\n')


@torch.no_grad()
@torch.no_grad()
def compute_rp_statistics(model, original_model, data_loader, device, task_id,
                          class_mask=None, args=None):
    """Accumulate Gram matrix and class prototypes for the routing-free head.

    Only aggregate second-order statistics are kept (no per-example features),
    so the strict exemplar-free protocol holds. With rp_feature_source=original
    the features come from the frozen backbone, which means the head needs no
    task inference at all -- the bottleneck that caps HRM-PET on CUB/CIFAR.
    """
    model.eval()
    original_model.eval()
    source = str(getattr(args, 'rp_feature_source', 'original'))
    for cls_id in class_mask:
        for inputs, targets in data_loader[cls_id]['train']:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            if source == 'original':
                features = rp_extractor(original_model, args)(
                    _rp_inputs(inputs, args))['pre_logits']
            else:
                # Fixed adapter for every task: this is RanPAC's first-session
                # adaptation, except the adaptation is HRM-PET's own trained
                # LoRA. The index must not vary or the Gram matrix would mix
                # incompatible feature spaces.
                features = model(
                    inputs, task_id=int(getattr(args, 'rp_lora_task', 0)),
                    train=True)['pre_logits']
            accumulate_rp_statistics(
                features, targets, args, device, args.nb_classes)


@torch.no_grad()
def _collect_rp_calibration_scores(model, original_model, data_loader, device,
                                   task_id, class_mask, args):
    """Transient scores over current-task training data, for temperature fit."""
    source = str(getattr(args, 'rp_feature_source', 'original'))
    score_chunks = []
    target_chunks = []
    for cls_id in class_mask:
        for inputs, targets in data_loader[cls_id]['train']:
            inputs = inputs.to(device, non_blocking=True)
            if source == 'original':
                features = rp_extractor(original_model, args)(
                    _rp_inputs(inputs, args))['pre_logits']
            else:
                features = model(
                    inputs, task_id=int(getattr(args, 'rp_lora_task', 0)),
                    train=True)['pre_logits']
            score_chunks.append(rp_head_predict(features, args, device))
            target_chunks.append(targets.to(device, non_blocking=True))
    return torch.cat(score_chunks, dim=0), torch.cat(target_chunks, dim=0)


def calibrate_rp_head(model, original_model, data_loader, device, task_id,
                      class_mask, args):
    """Fit the scalar temperature that makes the head's Loss comparable."""
    scores, targets = _collect_rp_calibration_scores(
        model, original_model, data_loader, device, task_id, class_mask, args)
    return fit_rp_temperature(scores.float(), targets.long(), args, device)


@torch.no_grad()
def _compute_mean(model: torch.nn.Module, data_loader: Iterable, device: torch.device, task_id, class_mask=None,
                  args=None, ):
    model.eval()

    for cls_id in class_mask:
        data_loader_cls = data_loader[cls_id]['train']
        features_per_cls = []
        for i, (inputs, targets) in enumerate(data_loader_cls):
            inputs = inputs.to(device, non_blocking=True)
            features = model(inputs, task_id=task_id, train=True)['pre_logits']
            features_per_cls.append(features)
        features_per_cls = torch.cat(features_per_cls, dim=0)
        features_per_cls_list = [torch.zeros_like(features_per_cls, device=device) for _ in range(args.world_size)]

        utils.distributed_barrier()
        dist.all_gather(features_per_cls_list, features_per_cls)
        gathered_features_per_cls = torch.cat(features_per_cls_list, dim=0)
        if args.ca_storage_efficient_method == 'covariance':
            features_per_cls = gathered_features_per_cls
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device)
        
        if args.ca_storage_efficient_method == 'variance':
            features_per_cls = gathered_features_per_cls
            # print(features_per_cls.shape)
            cls_mean[cls_id] = features_per_cls.mean(dim=0)
            cls_cov[cls_id] = torch.diag(torch.cov(features_per_cls.T) + (torch.eye(cls_mean[cls_id].shape[-1]) * 1e-4).to(device))
        if args.ca_storage_efficient_method == 'multi-centroid':
            from sklearn.cluster import KMeans
            n_clusters = args.n_centroids
            features_per_cls = gathered_features_per_cls.cpu().numpy()
            kmeans = KMeans(n_clusters=n_clusters)
            kmeans.fit(features_per_cls)
            cluster_lables = kmeans.labels_
            cluster_means = []
            cluster_vars = []
            for i in range(n_clusters):
               cluster_data = features_per_cls[cluster_lables == i]
               cluster_mean = torch.tensor(np.mean(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_var = torch.tensor(np.var(cluster_data, axis=0), dtype=torch.float64).to(device)
               cluster_means.append(cluster_mean)
               cluster_vars.append(cluster_var)
            
            cls_mean[cls_id] = cluster_means
            cls_cov[cls_id] = cluster_vars




@torch.no_grad()
def _sample_crct_class_features(mean, cov, count):
    distribution = utils.stable_multivariate_normal(
        mean.float(), cov.float(), 'class_alignment_gaussian')
    return distribution.sample(sample_shape=(count,))


def train_task_adaptive_prediction(model: torch.nn.Module, args, device,
                                   class_mask=None, task_id=-1,
                                   data_loader_per_cls=None):
    """Correct the classifier using Gaussian features, as in the paper runs."""
    model.train()
    run_epochs = args.crct_epochs
    param_list = [
        parameter for name, parameter in model.named_parameters()
        if parameter.requires_grad and 'lora' not in name
    ]
    if not param_list:
        raise ValueError('CRCT has no trainable parameters')
    network_params = [{
        'params': param_list, 'lr': args.ca_lr,
        'weight_decay': args.weight_decay,
    }]
    if 'mae' in args.model or 'beit' in args.model:
        optimizer = optim.AdamW(
            network_params, lr=args.ca_lr / 10,
            weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(
            network_params, lr=args.ca_lr, momentum=0.9,
            weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=run_epochs)
    criterion = torch.nn.CrossEntropyLoss().to(device)
    old_classes = [
        int(class_id)
        for seen_task in range(task_id)
        for class_id in class_mask[seen_task]
    ]
    replay_iterations = len(old_classes)

    for epoch in range(run_epochs):
        sampled_data = []
        sampled_label = []
        num_sampled_pcls = args.batch_size * 5
        metric_logger = utils.MetricLogger(delimiter="  ")
        metric_logger.add_meter(
            'Lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
        metric_logger.add_meter(
            'Loss', utils.SmoothedValue(window_size=1, fmt='{value:.4f}'))

        if args.ca_storage_efficient_method in ('covariance', 'variance'):
            for seen_task in range(task_id + 1):
                for class_id in class_mask[seen_task]:
                    mean = torch.tensor(
                        cls_mean[class_id], dtype=torch.float64).to(device)
                    cov = cls_cov[class_id].to(device)
                    if args.ca_storage_efficient_method == 'variance':
                        cov = torch.diag(cov)
                    sampled = _sample_crct_class_features(
                        mean, cov, num_sampled_pcls)
                    sampled_data.append(sampled)
                    sampled_label.extend([class_id] * sampled.shape[0])
        elif args.ca_storage_efficient_method == 'multi-centroid':
            for seen_task in range(task_id + 1):
                for class_id in class_mask[seen_task]:
                    for cluster in range(len(cls_mean[class_id])):
                        mean = cls_mean[class_id][cluster]
                        var = cls_cov[class_id][cluster]
                        if var.mean() == 0:
                            continue
                        cov = (
                            torch.diag(var)
                            + 1e-4 * torch.eye(mean.shape[0]).to(mean.device)
                        ).float()
                        sampled = _sample_crct_class_features(
                            mean, cov, num_sampled_pcls)
                        sampled_data.append(sampled)
                        sampled_label.extend([class_id] * sampled.shape[0])
        else:
            raise NotImplementedError

        inputs = torch.cat(sampled_data, dim=0).float().to(device)
        targets = torch.tensor(sampled_label).long().to(device)
        print(inputs.shape)
        order = torch.randperm(inputs.size(0))
        inputs = inputs[order]
        targets = targets[order]
        print(
            'CRCT replay iterations:', replay_iterations, 'of',
            int(math.ceil(inputs.size(0) / float(num_sampled_pcls))))

        for step in range(replay_iterations):
            inp = inputs[
                step * num_sampled_pcls:(step + 1) * num_sampled_pcls]
            tgt = targets[
                step * num_sampled_pcls:(step + 1) * num_sampled_pcls]
            logits = model(inp, fc_only=True)['logits']
            if args.train_mask and class_mask is not None:
                seen = [
                    class_id
                    for seen_task in range(task_id + 1)
                    for class_id in class_mask[seen_task]
                ]
                not_mask = np.setdiff1d(
                    np.arange(args.nb_classes), seen)
                not_mask = torch.tensor(
                    not_mask, dtype=torch.int64).to(device)
                logits = logits.index_fill(
                    dim=1, index=not_mask, value=float('-inf'))

            loss = criterion(logits, tgt)
            acc1, acc5 = accuracy(logits, tgt, topk=(1, 5))
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()))
                sys.exit(1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            torch.cuda.synchronize()
            metric_logger.update(Loss=loss.item())
            metric_logger.update(Lr=optimizer.param_groups[0]["lr"])
            metric_logger.meters['Acc@1'].update(
                acc1.item(), n=inp.shape[0])
            metric_logger.meters['Acc@5'].update(
                acc5.item(), n=inp.shape[0])

        metric_logger.synchronize_between_processes()
        print("Averaged stats:", metric_logger)
        scheduler.step()



