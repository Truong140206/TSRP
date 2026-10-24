"""Analytical boundaries, paired gap accounting and the real CPU CLI."""
import copy
from pathlib import Path
import random
import subprocess
import sys

import pytest

from tools import analyze_crm_candidate_errors as analysis
from tests.test_crm_candidate_probe import build_record


def test_threshold_checks_all_rivals_not_only_the_final_wrong_class():
    result = analysis.beta_interval([3., 2., 0.], [0., 3., 10.], 0)
    assert result['target_is_maximum_beta_interval_0_1'] == pytest.approx([0., 3/13])
    assert result['upper_boundary_rivals'] == [2]
    # The class-1 crossing is later (1/4); using it as a global bound is wrong.
    assert analysis.prediction_at([3., 2., 0.], [0., 3., 10.], .24)['unique_class_if_clear'] == 2


@pytest.mark.parametrize('p,r,target,interval', [
    ([0., 2., -3.], [0., -3., 2.], 0, [.4, .6]),  # Neither head alone is correct.
    ([0., 1.], [0., 1.], 0, None),
    ([1., 0.], [2., 0.], 0, [0., 1.]),
    ([0., 1.], [1., 0.], 0, [.5, 1.]),
    ([0., 0.], [0., 0.], 0, [0., 1.]),
    ([0., 1., -1.], [0., -1., 1.], 0, [.5, .5]),
    ([0., 2., -1.], [0., -1., 2.], 0, None),
])
def test_target_maximum_interval_handles_repairs_impossibility_and_ties(p, r, target, interval):
    result = analysis.beta_interval(p, r, target)
    actual = result['target_is_maximum_beta_interval_0_1']
    if interval is None:
        assert actual is None
    else:
        assert actual == pytest.approx(interval)
    if p == r == [0., 0.]:
        assert result['always_tied_rivals'] == [1]
        assert analysis.prediction_at(p, r, .3)['unique_class_if_clear'] is None


def test_closed_boundary_is_not_mislabeled_as_unique_correct_and_near_ties_are_visible():
    result = analysis.prediction_at([1., 0.], [0., 1.], .5)
    assert result['unique_class_if_clear'] is None
    assert result['exact_maximum_classes'] == [0, 1]
    assert analysis.prediction_at([1., 0.], [0., 1.], .5+1e-8)['unique_class_if_clear'] is None
    assert analysis.prediction_at([1., 0.], [0., 1.], .51)['unique_class_if_clear'] == 1


def test_analytic_interval_matches_direct_all_class_inequalities():
    rng = random.Random(136)
    for _ in range(80):
        p = [rng.uniform(-3, 3) for _ in range(7)]
        r = [rng.uniform(-3, 3) for _ in range(7)]
        target = rng.randrange(7)
        bounds = analysis.beta_interval(p, r, target)['target_is_maximum_beta_interval_0_1']
        for i in range(101):
            beta = i/100
            scores = [(1-beta)*x+beta*y for x,y in zip(p,r)]
            direct = scores[target] == max(scores)
            assert direct == (bounds is not None and bounds[0] <= beta <= bounds[1])


def pair(full_primary, cg_primary, rp):
    full, _ = build_record([full_primary], rp, 0, eligible=False)
    cg, _ = build_record([cg_primary], rp, 0, eligible=False)
    full['sample']['input_sha256'] = cg['sample']['input_sha256'] = 'same-image'
    return full, cg


def test_correct_primary_uses_wrong_final_rival_not_its_own_zero_gap():
    full, cg = pair([1., .9, 0., 0., 0.], [3., 0., 0., 0., 0.], [0., 10., 0., 0., 0.])
    result = analysis.analyze_pair(42, 3, full, cg)
    f, c = result['arms']['full'], result['arms']['class_gate']
    assert result['outcome'] == 'harmed'
    assert result['rival_class'] == 1
    assert f['primary_class'] == c['primary_class'] == 0
    assert f['primary_target_minus_rival'] > 0
    assert f['rp_target_minus_rival'] < 0
    assert f['combined_target_minus_rival'] < 0 < c['combined_target_minus_rival']
    assert f['effective_rp_weight'] > c['effective_rp_weight']
    assert f['actual_beta_minus_upper_boundary'] > 0
    assert f['use_other_arm_beta']['unique_class_if_clear'] == 0
    delta = result['full_minus_class_gate']
    assert delta['mixed_target_minus_rival'] == pytest.approx(
        delta['primary_change_at_class_gate_beta']+delta['beta_change_on_full_primary'])
    aggregate = analysis.aggregate_cases([result])
    group = aggregate['paired_harmed_full_fusion_harmed']
    assert group['samples'] == group['both_primary_correct'] == group['full_correct_with_class_gate_beta'] == 1
    assert group['full_same_adapter_candidates'] == group['full_crm_evaluated'] == 0
    assert aggregate['paired_repaired']['samples'] == 0
    # Reverse comparison covers the opposite direction, not only harmed cases.
    repaired = analysis.analyze_pair(43, 3, cg, full)
    assert repaired['outcome'] == 'repaired'
    assert repaired['rival_is_wrong_final_of'] == 'class_gate'
    assert repaired['full_minus_class_gate']['mixed_target_minus_rival'] == pytest.approx(-delta['mixed_target_minus_rival'])


def test_a_harm_can_come_from_a_weaker_primary_gap_despite_lower_rp_weight():
    full, cg = pair([1., .9, -100., -100., -100.], [.01, 0., 0., 0., 0.], [.99, 1., 0., 0., 0.])
    result = analysis.analyze_pair(42, 0, full, cg)
    assert result['outcome'] == 'harmed'
    assert result['full_minus_class_gate']['beta'] < 0
    assert result['full_minus_class_gate']['primary_target_minus_rival'] < 0
    assert result['arms']['full']['use_other_arm_beta']['unique_class_if_clear'] != 0
    group = analysis.aggregate_cases([result])['paired_harmed_full_fusion_harmed']
    assert group['full_beta_lower'] == group['full_primary_gap_smaller'] == 1
    assert group['full_higher_beta_and_smaller_primary_gap'] == 0


def test_rejects_mismatched_rp_or_identity_and_nonchanging_pairs():
    full, cg = pair([1., .9, 0., 0., 0.], [3., 0., 0., 0., 0.], [0., 10., 0., 0., 0.])
    for key in ('input_sha256', 'target', 'index'):
        changed = copy.deepcopy(cg)
        changed['sample'][key] = 'different'
        with pytest.raises(ValueError, match='Paired image'):
            analysis.analyze_pair(42, 0, full, changed)
    changed = copy.deepcopy(cg)
    changed['rp_scores'][0] += .1
    with pytest.raises(ValueError, match='Paired image'):
        analysis.analyze_pair(42, 0, full, changed)
    with pytest.raises(ValueError, match='final-class change'):
        analysis.analyze_pair(42, 0, full, full)


def test_real_cli_help_and_errors_need_no_site_packages_or_gpu(tmp_path):
    script = Path(analysis.__file__)
    def run(*args):
        return subprocess.run([sys.executable, '-B', '-S', str(script), *map(str,args)],
            capture_output=True, text=True, timeout=30, cwd=tmp_path)
    help_result = run('--help')
    assert help_result.returncode == 0 and '--batch-dir' in help_result.stdout
    missing = tmp_path/'missing'
    output = tmp_path/'report.json'
    result = run('--batch-dir', missing, '--output', output)
    assert result.returncode != 0 and 'CRM_ERROR_ANALYSIS_STOP' in result.stderr
    assert not output.exists()
    result = run('--batch-dir', missing, '--output', missing/'report.json')
    assert result.returncode != 0 and 'outside the input batch' in result.stderr
    output.write_text('keep this')
    result = run('--batch-dir', missing, '--output', output)
    assert result.returncode != 0 and 'new file' in result.stderr
    assert output.read_text() == 'keep this'


def test_new_output_is_not_written_if_analysis_rejects_input(monkeypatch, tmp_path):
    def fail(_):
        raise ValueError('checksum mismatch')
    monkeypatch.setattr(analysis, 'analyze_batch', fail)
    output = tmp_path/'result.json'
    with pytest.raises(SystemExit, match='checksum mismatch'):
        analysis.main(['--batch-dir', str(tmp_path/'input'), '--output', str(output)])
    assert not output.exists()
