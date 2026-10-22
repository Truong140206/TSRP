"""CPU/offline audit fixtures are synthetic, not CIFAR or ImageNet results."""
import json
from pathlib import Path
import subprocess
import sys
import pytest

from tests import test_prediction_trace_tools as trace_fixtures
from tools import audit_prediction_stages as audit
from tools import compare_prediction_traces as compare


def make_run(root, mutate=None):
    helper = trace_fixtures.TraceTests()
    for arm in audit.ARMS:
        (root / arm).mkdir(parents=True)
        full = arm == 'full'
        rows = [helper.row(0, i, 0 if full and i < 2 else 1,
                           0 if full and i == 0 else 1) for i in range(4)]
        # Different correctness paths through DRM and CRM, before the final gate.
        rows[1].update(drm_class=1 if full else 0, crm_evaluated=True,
                       crm_used_alternative=True, final_adapter=0 if full else 1,
                       proposal_after=1 if full else 0)
        last = helper.row(1, 0, 4 if full else 3, 1 if full else 0)
        last.update(drm_class=3 if full else 4, crm_class=3 if full else 4)
        if mutate:
            mutate(arm, rows, last)
        helper.write(root / arm, 0, rows)
        helper.write(root / arm, 1, [last])
    summary = compare.compare_directories(root / 'class_gate', root / 'full')
    summary['observer_all_stages'] = 'PASS_AT_1e-4_LOG_PRECISION'
    summary['single_seed_only'] = True
    (root / 'summary.json').write_text(json.dumps(summary), encoding='utf-8')
    return root


def test_stage_paths_counts_and_unequal_task_weighting(tmp_path):
    run = make_run(tmp_path / 'run')
    before = {p: p.read_bytes() for p in run.rglob('*') if p.is_file()}
    result = audit.audit(run)
    assert result['checks'] == dict(stored_summary='PASS', observer_trace_structure='PASS',
                                    input_files_unchanged='PASS')
    assert result['stage_pairs']['final_class']['counts'] == dict(
        both_correct=0, repaired=2, harmed=1, both_wrong=2)
    assert result['stage_pairs']['final_class']['task_balanced_net_pp'] == -25
    assert result['stage_pairs']['final_class']['pooled_net_pp'] == 20
    assert result['stage_pairs']['drm_class']['task_balanced_net_pp'] == 50
    assert result['stage_pairs']['crm_class']['task_balanced_net_pp'] == 75
    assert result['final_error_paths'] == dict(
        repaired={'repaired -> repaired -> repaired': 1, 'harmed -> repaired -> repaired': 1},
        harmed={'repaired -> repaired -> harmed': 1})
    focus = result['corrected_adapter_class_harms']
    assert len(focus) == 1
    assert (focus[0]['eval_task'], focus[0]['index'], focus[0]['target']) == (1, 0, 3)
    assert focus[0]['class_gate']['crm_class'] == 4
    assert focus[0]['full']['crm_class'] == 3
    assert len(result['changed_class_samples']) == 3
    assert result['within_arm']['full']['crm_class_to_final_class']['task_balanced_net_pp'] == -50
    assert result['within_arm']['class_gate']['crm_class_to_final_class']['task_balanced_net_pp'] == 50
    assert {p: p.read_bytes() for p in before} == before
    assert len(result['input_sha256']) == 5


@pytest.mark.parametrize('field,value', [
    ('crm_evaluated', 'false'), ('crm_used_alternative', 0),
    ('crm_class', 2), ('final_adapter', 0), ('crm_used_alternative', True),
])
def test_invalid_observer_structure_rejected(tmp_path, field, value):
    def mutate(arm, rows, last):
        if arm == 'class_gate':
            rows[0][field] = value
    run = make_run(tmp_path / 'run', mutate)
    with pytest.raises(ValueError, match='CRM|Retained'):
        audit.audit(run)


def test_invalid_rp_class_rejected(tmp_path):
    def mutate(arm, rows, last):
        rows[0]['rp_class'] = 99
    run = make_run(tmp_path / 'run', mutate)
    with pytest.raises(ValueError, match='Invalid RP'):
        audit.audit(run)


@pytest.mark.parametrize('change', ['counts', 'balanced', 'observer', 'provenance'])
def test_stored_summary_disagreement_rejected(tmp_path, change):
    run = make_run(tmp_path / 'run')
    path = run / 'summary.json'
    data = json.loads(path.read_text())
    if change == 'counts':
        data['counts']['harmed'] += 1
    elif change == 'balanced':
        data['task_balanced']['net_pp'] += .01
    elif change == 'observer':
        del data['observer_all_stages']
    else:
        data['provenance']['source'] = 'another checkout'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='summary'):
        audit.audit(run)


def test_corrupt_trace_rejected(tmp_path):
    run = make_run(tmp_path / 'run')
    path = run / 'full' / 'task_00.jsonl'
    path.write_text(path.read_text().replace('input0:0', 'different'))
    with pytest.raises(ValueError, match='checksum'):
        audit.audit(run)


def test_change_during_read_rejected(tmp_path, monkeypatch):
    run = make_run(tmp_path / 'run')
    original = compare.compare_directories
    def change(left, right):
        result = original(left, right)
        path = run / 'summary.json'
        path.write_text(path.read_text() + '\n')
        return result
    monkeypatch.setattr(compare, 'compare_directories', change)
    with pytest.raises(ValueError, match='changed during'):
        audit.audit(run)


def test_output_never_overwrites_or_enters_historical_run(tmp_path):
    run = make_run(tmp_path / 'run')
    report = audit.audit(run)
    with pytest.raises(ValueError, match='outside'):
        audit.save_report(report, run / 'new.json', run)
    output = tmp_path / 'report.json'
    audit.save_report(report, output, run)
    content = output.read_bytes()
    with pytest.raises(FileExistsError):
        audit.save_report(report, output, run)
    assert output.read_bytes() == content


def test_cli_is_cpu_only_and_prints_compact_report(tmp_path):
    run = make_run(tmp_path / 'run')
    output = tmp_path / 'report.json'
    tool = Path(audit.__file__)
    result = subprocess.run([sys.executable, '-B', '-S', str(tool), '--run-dir', str(run),
                             '--output', str(output)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert 'STAGE_AUDIT_COMPLETE=1/1' in result.stdout
    assert 'CORRECTED_ADAPTER_CLASS_HARMS=' in result.stdout
    assert len(json.loads(output.read_text())['changed_class_samples']) == 3


def test_nonempty_disjoint_masks_required():
    for masks in ([[0], [0]], [[0], []]):
        with pytest.raises(ValueError, match='disjoint'):
            audit.check_structure(dict(stage=2, class_mask=masks), [])
