"""Synthetic trace/command checks. These are not experimental results."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

_path = Path(__file__).resolve().parents[1] / 'engines' / 'prediction_trace.py'
_spec = importlib.util.spec_from_file_location('prediction_trace_test', _path)
_trace = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_trace)
TraceWriter, start_trace = _trace.TraceWriter, _trace.start_trace
from tools import compare_prediction_traces as compare
from tools import run_prediction_audit as runner


# Exact additions from 42ea682 to 8e48ed7, including the two added blank lines.
# Remove only these hooks, then hash source rather than Python-version-specific
# ast.dump output. All other source, including indentation and strings, is kept.
_OBSERVER_HOOKS = (
    'from engines.prediction_trace import start_trace\n',
    '    prediction_trace = start_trace(args, data_loader, i, task_id + 1, class_mask)\n',
    '            if prediction_trace is not None:\n'
    '                prediction_trace.begin(input, target, lora_id, prompt_id, logits, target_task_map)\n',
    '            if prediction_trace is not None:\n'
    '                prediction_trace.routing(prompt_id, fusion_rp_scores)\n',
    '            if prediction_trace is not None:\n'
    '                prediction_trace.drm(logits)\n',
    '            if prediction_trace is not None:\n'
    '                prediction_trace.crm(logits, filtered_index_tensor, corr_id,\n'
    '                                     ensemble_id[:, 1] if task_id > 0 else None)\n\n',
    '            if prediction_trace is not None:\n'
    '                prediction_trace.end(logits)\n\n',
    '    if prediction_trace is not None:\n'
    "        prediction_trace.close(metric_logger.meters['Acc@1'].global_avg,\n"
    "                               metric_logger.meters['Acc@task'].global_avg)\n",
)


def engine_source_without_observer(source):
    source = source.replace('\r\n', '\n').replace('\r', '\n')
    for hook in _OBSERVER_HOOKS:
        if source.count(hook) != 1:
            raise AssertionError('Expected exactly one unchanged observer hook: ' + hook)
        source = source.replace(hook, '', 1)
    return source


class TraceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.left, self.right = self.root / 'class_gate', self.root / 'full'
        self.left.mkdir()
        self.right.mkdir()

    def header(self, task, n, arm):
        return dict(dataset='synthetic', seed=42, stage=2, eval_task=task, samples=n,
                    class_mask=[[0, 1, 2], [3, 4, 5]],
                    configuration=dict(rp_route_fusion_weight=1.0 if arm == 'class_gate' else .7,
                                       prediction_trace_dir=arm, rp_class_fusion_gate='margin'),
                    run=dict(variant=arm, common=dict(source='same', checkpoints=['same'])))

    def row(self, task, index, final, proposal):
        return dict(index=index, target=3 * task, true_task=task,
                    input_sha256='input%d:%d' % (task, index), tii_task=0,
                    proposal_before=0, proposal_after=proposal,
                    initial_logits_sha256='initial', rp_sha256='rp', rp_class=3 * task,
                    drm_class=final, crm_class=final, final_class=final,
                    final_adapter=proposal, crm_evaluated=False, crm_used_alternative=False,
                    final_logits_sha256='logits')

    def write(self, directory, task, rows, header=None):
        path = directory / ('task_%02d.jsonl' % task)
        header = header or self.header(task, len(rows), directory.name)
        writer = TraceWriter(path, header)
        writer.append(rows)
        writer.close(100 * sum(r['final_class'] == r['target'] for r in rows) / len(rows),
                     100 * sum(r['proposal_after'] == task for r in rows) / len(rows))
        return path

    def fixture(self):
        # Task 0 has two repairs; task 1 has one harm, with unequal test sizes.
        # Pooled change is +20pp; task-balanced change is -25pp. Do not confuse.
        for arm in (self.left, self.right):
            is_full = arm == self.right
            rows = [self.row(0, i, 0 if is_full and i < 2 else 1, 0 if is_full else 1)
                    for i in range(4)]
            self.write(arm, 0, rows)
            self.write(arm, 1, [self.row(1, 0, 4 if is_full else 3, 1 if is_full else 0)])

    def test_counts_cross_table_and_balanced_identity(self):
        self.fixture()
        result = compare.compare_directories(self.left, self.right)
        self.assertEqual(result['counts'], dict(both_correct=0, repaired=2, harmed=1, both_wrong=2))
        self.assertEqual(result['proposal_vs_class']['corrected']['repaired'], 2)
        self.assertEqual(result['proposal_vs_class']['corrected']['harmed'], 1)
        balanced = result['task_balanced']
        self.assertEqual((balanced['repair_pp'], balanced['harm_pp'], balanced['net_pp']),
                         (25.0, 50.0, -25.0))
        self.assertEqual(balanced['accuracy']['full'] - balanced['accuracy']['class_gate'], -25)

    def test_outcomes_include_all_route_categories(self):
        self.assertEqual(compare.route_group(0, 0, 0), 'unchanged_correct')
        self.assertEqual(compare.route_group(0, 0, 1), 'unchanged_wrong')
        self.assertEqual(compare.route_group(0, 1, 1), 'corrected')
        self.assertEqual(compare.route_group(0, 1, 0), 'harmed')
        self.assertEqual(compare.route_group(0, 1, 2), 'changed_wrong')
        self.assertEqual(compare.outcome(0, 0, 0), 'both_correct')

    def test_pair_rejects_mismatched_inputs_rp_initial_logits(self):
        for field in ('input_sha256', 'rp_sha256', 'initial_logits_sha256', 'rp_class'):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as tmp:
                base = self.row(0, 0, 0, 0)
                altered = dict(base, **{field: 'different' if field != 'rp_class' else 1})
                left, right = Path(tmp) / 'left', Path(tmp) / 'right'
                left.mkdir(); right.mkdir()
                a = self.write(left, 0, [base], self.header(0, 1, 'class_gate'))
                b = self.write(right, 0, [altered], self.header(0, 1, 'full'))
                with self.assertRaisesRegex(ValueError, 'Inputs/initial'):
                    compare.compare_tasks(a, b)

    def test_pair_rejects_provenance_gate_or_weight_change(self):
        row = self.row(0, 0, 0, 0)
        left = self.write(self.left, 0, [row])
        for i, field in enumerate(('source', 'gate', 'weight')):
            header = self.header(0, 1, 'full')
            if field == 'source':
                header['run']['common']['source'] = 'different'
            elif field == 'gate':
                header['configuration']['rp_class_fusion_gate'] = 'none'
            else:
                header['configuration']['rp_route_fusion_weight'] = .6
            folder = self.root / str(i); folder.mkdir()
            right = self.write(folder, 0, [row], header)
            with self.assertRaises(ValueError):
                compare.compare_tasks(left, right)

    def test_partial_corrupt_and_trailing_data_rejected(self):
        row = self.row(0, 0, 0, 0)
        path = self.write(self.left, 0, [row])
        text = path.read_text()
        for broken in ('\n'.join(text.splitlines()[:-1]) + '\n',
                       text.replace('"input0:0"', '"changed"'), text + '{}\n'):
            path.write_text(broken)
            with self.assertRaises(ValueError):
                compare.load_task(path)

    def test_missing_task_rejected(self):
        row = self.row(0, 0, 0, 0)
        self.write(self.left, 0, [row]); self.write(self.right, 0, [row])
        with self.assertRaisesRegex(ValueError, 'Not all'):
            compare.compare_directories(self.left, self.right)

    def test_writer_never_overwrites_and_checks_metrics(self):
        path = self.write(self.left, 0, [self.row(0, 0, 0, 0)])
        original = path.read_bytes()
        with self.assertRaises(FileExistsError):
            TraceWriter(path, self.header(0, 1, 'class_gate'))
        self.assertEqual(path.read_bytes(), original)
        writer = TraceWriter(self.right / 'bad.jsonl', self.header(0, 1, 'full'))
        writer.append([self.row(0, 0, 0, 0)])
        with self.assertRaisesRegex(ValueError, 'metrics'):
            writer.close(0, 100)

    def test_trace_is_disabled_without_flag(self):
        self.assertIsNone(start_trace(SimpleNamespace(), None, 0, 10, None))

    def test_log_metric_crosscheck(self):
        result = dict(task_balanced=dict(accuracy={'full': 75., 'class_gate': 74.},
                                        proposal_accuracy={'full': 80., 'class_gate': 78.}))
        logs = {arm: {10: {'Acc@1': result['task_balanced']['accuracy'][arm],
                          'Acc@task': result['task_balanced']['proposal_accuracy'][arm]}}
                for arm in ('full', 'class_gate')}
        runner.check_trace_metrics(result, logs)
        logs['full'][10]['Acc@1'] += .001
        with self.assertRaises(ValueError):
            runner.check_trace_metrics(result, logs)

    def test_commands_keep_paper_parameters_and_eval_only(self):
        for dataset in runner.verify.DATASETS:
            args = SimpleNamespace(dataset=dataset, seed=42, batch_size=24, port=29558,
                                   resolved_data_path=Path('/data'))
            full = runner.build_command(args, Path('/tii'), Path('/lora'), 'full')
            gate = runner.build_command(args, Path('/tii'), Path('/lora'), 'class_gate')
            cfg = runner.grid.parsed_config(full)
            self.assertTrue(cfg.eval and cfg.strict_exemplar_free and cfg.rp_pin_extractor)
            self.assertFalse(cfg.train_inference_task_only)
            self.assertEqual((cfg.rp_dim, cfg.rp_lambda, cfg.rp_seed), (10000, 10000., 1993))
            self.assertEqual((cfg.rp_class_fusion_weight, cfg.rp_class_fusion_gate), (.5, 'margin'))
            self.assertEqual(cfg.dataset, runner.verify.DATASETS[dataset][0])
            runner.finish.set_flag(gate, '--rp_route_fusion_weight', .7)
            self.assertEqual(full, gate)
            traced = runner.build_command(args, Path('/tii'), Path('/lora'), 'full', Path('/trace'))
            self.assertEqual(traced[:-2], full)
            self.assertEqual(traced[-2:], ['--prediction_trace_dir', str(Path('/trace'))])

    def test_busy_gpu_stops_before_evaluation(self):
        with patch.object(runner.subprocess, 'run', return_value=SimpleNamespace(stdout='1234\n')) as run:
            with self.assertRaisesRegex(RuntimeError, 'compute jobs'):
                runner.idle_gpu()
            self.assertEqual(run.call_count, 1)

    def test_low_memory_stops(self):
        with patch.object(runner.subprocess, 'run', side_effect=[SimpleNamespace(stdout=''),
                                                               SimpleNamespace(stdout='8000\n')]):
            with self.assertRaisesRegex(RuntimeError, '16 GiB'):
                runner.idle_gpu()

    def test_preflight_does_not_run_gpu_or_write_output(self):
        data = self.root / 'data'; data.mkdir()
        out = self.root / 'out'; out.mkdir()
        argv = ['audit', '--output-root', str(out), '--data-root', str(data), '--tag', 'test']
        fake_torch = SimpleNamespace(version=SimpleNamespace(cuda=None))
        with patch.object(sys, 'argv', argv), patch.dict(sys.modules, {'torch': fake_torch}), \
                patch.object(runner, 'resolve_data', return_value=data), \
                patch.object(runner.grid, 'manifest', return_value=([], {})), \
                patch.object(runner.importlib.metadata, 'version', return_value='test'), \
                patch.object(runner, 'execute') as execute, patch.object(runner, 'idle_gpu') as gpu:
            runner.main()
        execute.assert_not_called()
        gpu.assert_not_called()
        self.assertEqual(list(out.iterdir()), [])

    def test_engine_algorithm_is_unchanged_outside_observer_hooks(self):
        source = (_path.parent / 'hrm_lora_wtp_and_tap_engine.py').read_text(encoding='utf-8')
        restored = engine_source_without_observer(source)
        digest = hashlib.sha256(restored.encode('utf-8')).hexdigest()
        # SHA256 of the original UTF-8/LF Git blob at 42ea682, not an AST dump.
        self.assertEqual(digest, '99509a73fba96d9bdbaf58b55cccb41bd150e0f3df284065daa5529dd1a49dc6')

    def test_engine_source_guard_is_line_ending_independent(self):
        source = (_path.parent / 'hrm_lora_wtp_and_tap_engine.py').read_text(encoding='utf-8')
        self.assertEqual(engine_source_without_observer(source),
                         engine_source_without_observer(source.replace('\n', '\r\n')))

    def test_engine_source_guard_preserves_algorithm_changes(self):
        source = (_path.parent / 'hrm_lora_wtp_and_tap_engine.py').read_text(encoding='utf-8')
        mutations = (
            source.replace('logits[equal_drm] = output_drm_logits',
                           'logits[equal_drm] = output_drm_logits * 0', 1),
            source.replace(_OBSERVER_HOOKS[3],
                           _OBSERVER_HOOKS[3] + '                logits.zero_()\n', 1),
        )
        for changed in mutations:
            self.assertNotEqual(source, changed)
            self.assertNotEqual(engine_source_without_observer(source),
                                engine_source_without_observer(changed))

    def test_engine_source_guard_rejects_changed_or_duplicate_hooks(self):
        source = (_path.parent / 'hrm_lora_wtp_and_tap_engine.py').read_text(encoding='utf-8')
        for hook in _OBSERVER_HOOKS:
            for replacement in ('', hook + hook, hook.replace('prediction_trace', 'changed_trace')):
                changed = source.replace(hook, replacement, 1)
                with self.assertRaises(AssertionError):
                    engine_source_without_observer(changed)


if __name__ == '__main__':
    unittest.main()
