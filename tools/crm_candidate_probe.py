"""Observe both original CRM candidates; never use labels to select a candidate.

Only the two historical controls are evaluated. Counterfactual candidate fusion
is computed from already available logits, without another model forward.
"""
from contextlib import contextmanager
import inspect
import json
import math
from pathlib import Path
import statistics

from tools.fusion_logic_variants import checked_source
from tools.prediction_score_probe import describe_scores
from tools import audit_prediction_stages as stages
from tools import compare_prediction_traces as compare
from tools.run_prediction_audit import save_new


ARMS = {'full': (.7, False, False), 'class_gate': (1., False, False)}
FLAGS = ()
ANCHOR = '                    corr_id = torch.max(entropy, dim=1)[1]\n'
INSERT = (
    '                    if prediction_trace is not None:\n'
    '                        prediction_trace.candidates(\n'
    '                            error_output, filtered_index_tensor,\n'
    '                            prompt_id[filtered_index_tensor], ensemble_id[:, 1],\n'
    '                            entropy, corr_id)\n'
)


def changes_crm(arm):
    return False


def compact_scores(primary, rp, final, target, primary_class, final_class):
    description = describe_scores(primary, rp, final, target, primary_class, final_class)
    return {k: v for k, v in description.items()
            if k not in ('primary_logits', 'rp_scores', 'final_logits')}


def vector_close(left, right):
    return (len(left) == len(right) and all(math.isfinite(x) for x in (*left, *right))
            and all(abs(a-b) <= 1e-5 + 1e-6 * max(abs(a), abs(b)) for a,b in zip(left,right)))


def winner_valid(values, cls):
    return type(cls) is int and 0 <= cls < len(values) and values[cls] == max(values)


def tied_classes(values):
    best = max(values)
    return [i for i,value in enumerate(values) if value == best]


def energy(values):
    peak = max(values)
    return peak + .1 * math.log(math.fsum(math.exp((v-peak)/.1) for v in values))


class CandidateObserver:
    def __init__(self, writer, engine, args):
        self.writer, self.engine, self.args = writer, engine, args
        self.records = []
        self.pending_candidates = None

    def __getattr__(self, name):
        return getattr(self.writer, name)

    def begin(self, *values):
        self.writer.begin(*values)
        self.pending_candidates = {}
        self.rp = self.drm_scores = self.primary_scores = None

    def routing(self, proposal, rp):
        self.writer.routing(proposal, rp)
        if rp is None:
            raise ValueError('Candidate observation requires the historical RP scores')
        self.rp = rp.detach().clone()

    def drm(self, logits):
        self.writer.drm(logits)
        self.drm_scores = logits.detach().cpu().tolist()

    def candidates(self, logits, indices, current, alternative, criterion, choices):
        import torch
        if not len(indices):
            return
        if self.pending_candidates:
            raise ValueError('Repeated candidate capture in one batch')
        if logits.shape[1:] != (2, self.rp.shape[1]) or not torch.isfinite(logits).all():
            raise ValueError('Only complete finite final-stage candidate scores are supported')
        # The actual choice has already been made. Observe two hypothetical
        # class fusions without changing configuration, telemetry or logits.
        flat = logits.detach().clone().flatten(0, 1)
        rp = self.rp[indices, None].expand(-1, 2, -1).flatten(0, 1)
        before_args, before_stats = self.engine.args_ref[0], dict(self.engine.gate_stats)
        try:
            self.engine.args_ref[0] = self.args
            mixed = self.engine.fuse_class_scores(flat, rp, .5, self.args.num_tasks)
        finally:
            self.engine.args_ref[0] = before_args
            self.engine.gate_stats.clear()
            self.engine.gate_stats.update(before_stats)
        primary_classes = flat.topk(5, 1, True, True).indices[:, 0].reshape(-1, 2).cpu().tolist()
        final_classes = mixed.topk(5, 1, True, True).indices[:, 0].reshape(-1, 2).cpu().tolist()
        primary = flat.reshape_as(logits).cpu().tolist()
        final = mixed.reshape_as(logits).cpu().tolist()
        for j, batch_index in enumerate(indices.cpu().tolist()):
            self.pending_candidates[batch_index] = dict(
                selected=int(choices[j].item()), criterion=criterion[j].cpu().tolist(),
                candidates=[dict(adapter=int(ids[j].item()), primary_logits=primary[j][k],
                    fused_logits=final[j][k], primary_class=primary_classes[j][k],
                    fused_class=final_classes[j][k]) for k, ids in enumerate((current, alternative))])

    def crm(self, logits, *values):
        self.writer.crm(logits, *values)
        self.primary_scores = logits.detach().cpu().tolist()

    def end(self, logits):
        rows = self.writer.pending
        final = logits.detach().cpu().tolist()
        rp = self.rp.cpu().tolist()
        self.writer.end(logits)  # Includes exact historical-row checks.
        for j, row in enumerate(rows):
            captured = self.pending_candidates.get(j)
            if row['crm_evaluated'] != (captured is not None):
                raise ValueError('CRM eligibility differs from candidate capture')
            if captured is None:
                captured = dict(selected=0, criterion=None, candidates=[dict(
                    adapter=row['proposal_after'], primary_logits=self.primary_scores[j],
                    fused_logits=final[j], primary_class=row['crm_class'],
                    fused_class=row['final_class'])])
            selected = captured['candidates'][captured['selected']]
            if (captured['candidates'][0]['primary_logits'] != self.drm_scores[j]
                    or selected['primary_logits'] != self.primary_scores[j]
                    or not vector_close(selected['fused_logits'], final[j])
                    or selected['fused_class'] != row['final_class']):
                raise ValueError('Counterfactual selected candidate differs from actual evaluation')
            self.records.append(dict(sample=dict(row), rp_scores=rp[j],
                actual_final_logits=final[j], **captured))

    def close(self, *values):
        self.writer.close(*values)
        if len(self.records) != self.writer.header['samples']:
            raise ValueError('Incomplete candidate-score coverage')
        path = self.writer.path.with_name('candidate_' + self.writer.path.stem + '.json')
        save_new(path, dict(schema=1, trace_header=self.writer.header, samples=self.records))


@contextmanager
def installed(engine):
    checked_source(engine.__file__)
    old_evaluate, old_start = engine.evaluate, engine.start_trace
    source = inspect.getsource(old_evaluate)
    if source.count(ANCHOR) != 1:
        raise ValueError('Candidate observation insertion point changed')

    def start(args, loader, task, stage, masks):
        writer = old_start(args, loader, task, stage, masks)
        if writer is None:
            return None
        if (not args.eval or args.task_inc or args.rp_class_fusion_gate != 'margin'
                or args.rp_class_fusion_weight != .5 or args.rp_class_fusion_sharpen != 1.
                or args.rp_fusion_ramp != 0. or args.rp_class_fusion_min_tasks != 1
                or getattr(args, 'task_routing_mode', 'class') != 'class'
                or sorted(c for mask in masks[:stage] for c in mask) != list(range(args.nb_classes))):
            raise ValueError('Candidate observation requires the original final-stage class fusion')
        return CandidateObserver(writer, engine, args)

    namespace = dict(engine.__dict__, start_trace=start)
    exec(compile(source.replace(ANCHOR, ANCHOR + INSERT, 1), '<crm-candidate-observer>', 'exec'), namespace)
    engine.evaluate, engine.start_trace = namespace['evaluate'], start
    try:
        yield
    finally:
        engine.evaluate, engine.start_trace = old_evaluate, old_start


def inspect_record(record, row, classes, tasks):
    """Validate captures; labels are used only here, after inference, for analysis."""
    expected = {k:v for k,v in row.items() if k != 'type'}
    if record['sample'] != expected:
        raise ValueError('Candidate record differs from historical-verified sample')
    target, rp, actual = row['target'], record['rp_scores'], record['actual_final_logits']
    candidates, selected = record['candidates'], record['selected']
    eligible = row['crm_evaluated']
    if (len(candidates) != (2 if eligible else 1) or type(selected) is not int
            or selected != int(row['crm_used_alternative'])):
        raise ValueError('Candidate count/selection mismatch')
    if candidates[0]['adapter'] != row['proposal_after'] or candidates[selected]['adapter'] != row['final_adapter']:
        raise ValueError('Captured adapter differs from actual routing/CRM')
    if candidates[0]['primary_class'] != row['drm_class']:
        raise ValueError('First candidate differs from DRM prediction')
    vectors = [rp, actual] + [c[k] for c in candidates for k in ('primary_logits','fused_logits')]
    if any(len(v) != classes or not all(math.isfinite(x) for x in v) for v in vectors):
        raise ValueError('Incomplete/nonfinite candidate score vector')
    if not winner_valid(rp, row['rp_class']):
        raise ValueError('RP prediction differs from captured vector')
    descriptions = []
    for candidate in candidates:
        if type(candidate['adapter']) is not int or not 0 <= candidate['adapter'] < tasks:
            raise ValueError('Invalid candidate adapter')
        descriptions.append(compact_scores(candidate['primary_logits'], rp,
            candidate['fused_logits'], target, candidate['primary_class'], candidate['fused_class']))
    own = candidates[selected]
    if (own['primary_class'] != row['crm_class'] or own['fused_class'] != row['final_class']
            or not vector_close(own['fused_logits'], actual)):
        raise ValueError('Selected candidate does not reproduce the actual prediction')
    compact_scores(own['primary_logits'], rp, actual, target, row['crm_class'], row['final_class'])
    if eligible:
        scores = record['criterion']
        if (not isinstance(scores,list) or len(scores) != 2
                or not vector_close(scores, [energy(c['primary_logits']) for c in candidates])
                or selected != int(scores[1] > scores[0])):
            raise ValueError('Captured CRM criterion/choice differs from original energy')
    elif record['criterion'] is not None:
        raise ValueError('CRM criterion present on an unevaluated sample')
    correct = row['final_class'] == target
    primary_correct = row['crm_class'] == target
    other_correct = eligible and candidates[1-selected]['fused_class'] == target
    ties = any(c['fused_logits'].count(max(c['fused_logits'])) > 1 for c in candidates)
    events = dict(samples=True, crm_evaluated=eligible, no_crm=not eligible,
        same_adapter_candidates=eligible and candidates[0]['adapter'] == candidates[1]['adapter'],
        actual_correct=correct, primary_correct=primary_correct, rp_correct=row['rp_class']==target,
        fusion_repaired=not primary_correct and correct,
        fusion_harmed=primary_correct and not correct,
        fusion_harmed_rp_correct=primary_correct and not correct and row['rp_class']==target,
        fusion_harmed_rp_wrong=primary_correct and not correct and row['rp_class']!=target,
        selected_wrong_other_correct=eligible and not correct and other_correct,
        both_candidates_wrong=eligible and not correct and not other_correct,
        selected_correct_other_wrong=eligible and correct and not other_correct,
        both_candidates_correct=eligible and correct and other_correct,
        available_correct=correct or other_correct,
        hypothetical_fusion_top_ties=ties)
    detail = dict(selected=selected, target=target, crm_evaluated=eligible,
        candidate_adapters=[c['adapter'] for c in candidates],
        candidate_primary_classes=[c['primary_class'] for c in candidates],
        candidate_fused_classes=[c['fused_class'] for c in candidates],
        candidate_fused_ties=[tied_classes(c['fused_logits']) for c in candidates],
        candidate_scores=descriptions, events=events)
    return events, detail


def summarize_candidates(destination, comparison):
    result, hashes, details = {}, {}, {}
    reference_header,_ = compare.load_task(destination/'full'/'task_00.jsonl')
    if (reference_header['dataset'] != 'Split-Imagenet-A' or reference_header['stage'] != 10
            or reference_header['configuration']['nb_classes'] != 200
            or comparison.get('dataset',reference_header['dataset']) != reference_header['dataset']
            or comparison.get('seed',reference_header['seed']) != reference_header['seed']):
        raise ValueError('Expected matching final ImageNet-A seed and configuration')
    for arm in ('class_gate', 'full'):
        task_counts, details[arm] = [], {}
        expected_paths = [destination/arm/('candidate_task_%02d.json'%t) for t in range(10)]
        if sorted((destination/arm).glob('candidate_task_*.json')) != expected_paths:
            raise ValueError('Incomplete candidate-score task coverage')
        for task,path in enumerate(expected_paths):
            before = stages.sha256(path)
            captured = json.loads(path.read_text(encoding='utf-8'))
            header,rows = compare.load_task(destination/arm/('task_%02d.jsonl'%task))
            if (header['eval_task'] != task or any(header[k] != reference_header[k]
                    for k in ('dataset','seed','stage','class_mask'))
                    or compare.comparable_configuration(header) != compare.comparable_configuration(reference_header)):
                raise ValueError('Candidate task configuration differs across arms/tasks')
            expected_header = {k:v for k,v in header.items() if k not in ('type','schema')}
            if (captured.get('schema') != 1 or captured.get('trace_header') != expected_header
                    or len(captured['samples']) != len(rows)):
                raise ValueError('Captured candidate header/coverage mismatch')
            counts = None
            for record,row in zip(captured['samples'],rows):
                events, detail = inspect_record(record,row,header['configuration']['nb_classes'],header['stage'])
                if counts is None:
                    counts = dict.fromkeys(events,0)
                for key,value in events.items():
                    counts[key] += int(value)
                details[arm][task,row['index']] = detail
                if arm == 'full':
                    # Same image and fixed RP vector must also match across arms.
                    reference_sample = details['class_gate'][task,row['index']]
                    if (record['rp_scores'] != reference_sample.pop('_rp')
                            or row['input_sha256'] != reference_sample.pop('_input')
                            or row['target'] != reference_sample['target']):
                        raise ValueError('Paired input/RP/target differs')
                else:
                    detail.update(_rp=record['rp_scores'],_input=row['input_sha256'])
            if stages.sha256(path) != before:
                raise ValueError('Candidate scores changed while being read')
            hashes[path.relative_to(destination).as_posix()] = before
            task_counts.append(counts)
        keys = task_counts[0].keys()
        total = {k:sum(task[k] for task in task_counts) for k in keys}
        rates = {k:statistics.mean(100*task[k]/task['samples'] for task in task_counts) for k in keys}
        if abs(rates['actual_correct']-comparison['arms'][arm]['task_balanced_accuracy']['final_class'])>1e-9:
            raise ValueError('Candidate scores do not reproduce verified task-balanced accuracy')
        result[arm] = dict(counts=total, task_balanced_pp_of_all_samples=rates, per_task_counts=task_counts)
    paired = []
    for key, full in details['full'].items():
        cg = details['class_gate'][key]
        if full['events']['actual_correct'] != cg['events']['actual_correct']:
            paired.append(dict(eval_task=key[0], index=key[1],
                outcome='repaired' if full['events']['actual_correct'] else 'harmed',
                full=full, class_gate=cg))
    for name in ('harmed','repaired'):
        if sum(r['outcome']==name for r in paired) != comparison['conditional_routing']['original']['counts'][name]:
            raise ValueError('Paired candidate outcomes differ from verified traces')
    return dict(arms=result, paired_class_changes=paired, score_sha256=hashes,
        limitations='Oracle availability is label-based reporting over the candidates already evaluated, '
        'not a deployable selector or validation result. No candidate is invented for ineligible samples. '
        'Ties are recorded; counts use the evaluator topk convention. Scores are instrumented, not timings.')


def audit_batch(root):
    """Read and verify four completed seeds; never load models or use a GPU."""
    seeds = {}
    for seed in (42,43,44,45):
        folder = Path(root)/'ima'/('seed%d'%seed)
        path = folder/'summary.json'
        before = stages.sha256(path)
        stored = json.loads(path.read_text(encoding='utf-8'))
        if (stored.get('dataset') != 'Split-Imagenet-A' or stored.get('seed') != seed
                or stored.get('provenance',{}).get('experiment') != 'crm_candidate_scores_observer_v1'
                or stored.get('historical_controls') != 'EXACT_TRACE_ROWS_BOTH_ARMS'
                or stored.get('historical_full_all_stages') != 'PASS_AT_1e-4_LOG_PRECISION'):
            raise ValueError('Incomplete/wrong candidate audit: ' + str(path))
        trace_hashes = {str(k).replace('\\','/'):v for k,v in stored['trace_sha256'].items()}
        expected = {arm+('/task_%02d.jsonl'%t) for arm in ARMS for t in range(10)}
        if set(trace_hashes) != expected:
            raise ValueError('Incomplete stored trace hash list')
        for name,digest in trace_hashes.items():
            if stages.sha256(folder/name) != digest:
                raise ValueError('Stored trace hash mismatch: ' + name)
        computed = summarize_candidates(folder,stored)
        stages.check_summary(computed,stored['candidate_diagnostics'],'candidate_diagnostics')
        if stages.sha256(path) != before:
            raise ValueError('Summary changed during read')
        seeds[seed] = computed
        print('CRM_CANDIDATE_SEED_VERIFIED=%d'%seed,flush=True)
        for arm,report in computed['arms'].items():
            print('ARM=%s TOP1=%.6f COUNTS=%s'%(arm,
                report['task_balanced_pp_of_all_samples']['actual_correct'],
                json.dumps(report['counts'],sort_keys=True)),flush=True)
    aggregate = {}
    for arm in ARMS:
        reports = [data['arms'][arm] for data in seeds.values()]
        aggregate[arm] = dict(
            repeated_prediction_counts={k:sum(r['counts'][k] for r in reports) for k in reports[0]['counts']},
            equal_seed_task_balanced_pp={k:statistics.mean(r['task_balanced_pp_of_all_samples'][k]
                for r in reports) for k in reports[0]['task_balanced_pp_of_all_samples']})
    paired = {}
    for outcome in ('harmed','repaired'):
        cases = [r for data in seeds.values() for r in data['paired_class_changes'] if r['outcome']==outcome]
        paired[outcome] = dict(samples=len(cases), **{arm:{key:sum(r[arm]['events'][key] for r in cases)
            for key in ('selected_wrong_other_correct','both_candidates_wrong','fusion_harmed',
                        'fusion_repaired','rp_correct','hypothetical_fusion_top_ties')} for arm in ARMS})
    print('CRM_CANDIDATE_FOUR_SEED_MEANS='+json.dumps(aggregate,sort_keys=True),flush=True)
    print('CRM_CANDIDATE_PAIRED_PATHS='+json.dumps(paired,sort_keys=True),flush=True)
    print('Counts repeat images across seeds; rates average tasks, then seeds. '
          'Oracle alternatives are report-only, not a deployable method.',flush=True)
    print('CRM_CANDIDATE_ALL_SEEDS_VERIFIED=4/4',flush=True)
    return dict(seeds=seeds, aggregate=aggregate, paired=paired)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Read-only CPU summary of completed CRM candidate audits.')
    parser.add_argument('--batch-dir',type=Path,required=True)
    options = parser.parse_args()
    audit_batch(options.batch_dir)
