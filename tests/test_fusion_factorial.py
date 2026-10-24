"""Full-population accounting, unequal tasks, near ties, and the actual CPU CLI."""
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from tools import analyze_fusion_factorial as factorial
from tests.test_crm_candidate_probe import build_record


def records(full_p, cg_p, rp, target=0):
    full,_ = build_record([full_p],rp,target,eligible=False)
    cg,_ = build_record([cg_p],rp,target,eligible=False)
    full['sample']['input_sha256'] = cg['sample']['input_sha256'] = 'shared'
    return full,cg


def test_four_cells_use_prefusion_scores_and_actual_per_image_weights():
    full,cg = records([1.,.9,0.,0.,0.],[3.,0.,0.,0.,0.],[0.,10.,0.,0.,0.])
    before = copy.deepcopy((full,cg))
    cells,betas = factorial.cell_predictions(full,cg)
    assert (full,cg) == before
    assert betas['full'] > betas['class_gate']
    assert cells['cg_scores_cg_gate']['predicted_class'] == 0
    assert cells['full_scores_full_gate']['predicted_class'] == 1
    assert cells['full_scores_cg_gate']['predicted_class'] == 0
    assert cells['cg_scores_full_gate']['predicted_class'] == 0
    reverse,_ = factorial.cell_predictions(cg,full)
    assert reverse['cg_scores_full_gate'] == cells['full_scores_cg_gate']
    assert reverse['full_scores_cg_gate'] == cells['cg_scores_full_gate']


def test_all_four_predictions_match_independent_scalar_mixing():
    import math
    import random
    rng = random.Random(38)
    def standardize(values):
        avg=sum(values)/len(values)
        sd=max(1e-6,math.sqrt(sum((x-avg)**2 for x in values)/len(values)))
        return [(x-avg)/sd for x in values]
    for _ in range(100):
        p = {arm:[rng.uniform(-5,5) for _ in range(7)] for arm in ('full','class_gate')}
        rp=[rng.uniform(-5,5) for _ in range(7)]
        full,cg=records(p['full'],p['class_gate'],rp)
        cells,_=factorial.cell_predictions(full,cg)
        for name,(score_arm,gate_arm) in factorial.CELLS.items():
            top=sorted(p[gate_arm],reverse=True)
            beta=1/(1+math.exp(top[0]-top[1]))
            z=[(1-beta)*a+beta*b for a,b in zip(standardize(p[score_arm]),standardize(rp))]
            expected=max(range(len(z)),key=z.__getitem__)
            assert cells[name]['predicted_class'] == expected
            assert cells[name]['correct_low'] == cells[name]['correct_high'] == int(expected==0)


def test_ambiguous_crossed_cell_is_bounded_and_not_given_a_favorable_tie_break():
    full,cg=records([0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.])
    cells,_=factorial.cell_predictions(full,cg)
    for name in ('full_scores_cg_gate','cg_scores_full_gate'):
        assert cells[name]['predicted_class'] is None
        assert cells[name]['correct_low']==0 and cells[name]['correct_high']==1
        assert cells[name]['correctness_ambiguous']
    # Controls retain their recorded evaluator decision, not an invented new tie rule.
    assert cells['cg_scores_cg_gate']['correct_low']==cells['cg_scores_cg_gate']['correct_high']==1
    stats=factorial.empty_stats()
    factorial.add_row(stats,cells,dict(full=True,class_gate=True))
    report=factorial.summarize_tasks([stats])
    assert report['task_balanced_accuracy']['full_scores_cg_gate']==dict(value=None,low=0.,high=100.)
    bounds=report['task_balanced_paired_pp']['full_scores_cg_gate']['full']
    assert bounds['harmed_low']==0 and bounds['harmed_high']==100
    assert bounds['net']==dict(value=None,low=-100.,high=0.)


def definite_cells(**correct):
    return {name:dict(correct_low=int(value),correct_high=int(value),
        prediction_ambiguous=False,correctness_ambiguous=False) for name,value in correct.items()}


def test_unequal_task_sizes_and_all_original_correctness_strata_are_counted():
    first,second=factorial.empty_stats(),factorial.empty_stats()
    # A crossed gate harms an image for which BOTH old arms were correct.
    factorial.add_row(first,definite_cells(cg_scores_cg_gate=True,full_scores_full_gate=True,
        full_scores_cg_gate=False,cg_scores_full_gate=True),dict(full=True,class_gate=True))
    # It repairs three images for which BOTH old arms were wrong.
    for _ in range(3):
        factorial.add_row(second,definite_cells(cg_scores_cg_gate=False,full_scores_full_gate=False,
            full_scores_cg_gate=True,cg_scores_full_gate=False),dict(full=False,class_gate=False))
    summary=factorial.summarize_tasks([first,second])
    acc=summary['task_balanced_accuracy']
    assert acc['cg_scores_cg_gate']['value']==acc['full_scores_cg_gate']['value']==50.
    # Pooled count would give 75%; the required equally weighted tasks give 50%.
    count=summary['counts']['cells']['full_scores_cg_gate']
    assert count['correct_low']==3 and summary['counts']['samples']==4
    assert count['versus']['full']['repaired_low']==3 and count['versus']['full']['harmed_low']==1
    assert summary['task_balanced_paired_pp']['full_scores_cg_gate']['full']['net']['value']==0.
    effects=summary['effects_pp']
    assert effects['scores_at_cg_gate']['value']+effects['gate_on_full_scores']['value']==effects['full_minus_class_gate']['value']


def test_counterfactual_effects_include_an_interaction():
    acc={k:factorial.value_bounds(v,v) for k,v in zip(factorial.CELLS,[60.,63.,65.,61.])}
    effects=factorial.effect_bounds(acc)
    assert effects['full_minus_class_gate']['value']==3.
    assert effects['scores_at_cg_gate']['value']==5.
    assert effects['gate_on_full_scores']['value']==-2.
    assert effects['gate_on_cg_scores']['value']==1.
    assert effects['scores_at_full_gate']['value']==2.
    assert effects['interaction']['value']==-3.


def test_controls_and_pair_identity_are_not_silently_overridden():
    full,cg=records([1.,.9,0.,0.,0.],[3.,0.,0.,0.,0.],[0.,10.,0.,0.,0.])
    for key in ('target','index','input_sha256'):
        other=copy.deepcopy(cg)
        other['sample'][key]='different'
        with pytest.raises(ValueError,match='Paired'):
            factorial.cell_predictions(full,other)
    other=copy.deepcopy(cg)
    other['rp_scores'][0]+=.1
    with pytest.raises(ValueError,match='Paired'):
        factorial.cell_predictions(full,other)
    other=copy.deepcopy(full)
    other['sample']['final_class']=0
    with pytest.raises(ValueError,match='Observed classes'):
        factorial.cell_predictions(other,cg)


def test_real_cli_runs_without_site_packages_and_rejects_bad_paths(tmp_path):
    command=[sys.executable,'-B','-S',str(Path(factorial.__file__))]
    help_result=subprocess.run(command+['--help'],capture_output=True,text=True,timeout=30,cwd=tmp_path)
    assert help_result.returncode==0 and '--batch-dir' in help_result.stdout
    batch,output=tmp_path/'missing',tmp_path/'report.json'
    result=subprocess.run(command+['--batch-dir',str(batch),'--output',str(output)],
        capture_output=True,text=True,timeout=30,cwd=tmp_path)
    assert result.returncode != 0 and 'FUSION_FACTORIAL_STOP' in result.stderr
    assert not output.exists()
    for path,expected in ((batch/'out.json','outside the input batch'),(output,'new file')):
        output.write_text('preserve')
        result=subprocess.run(command+['--batch-dir',str(batch),'--output',str(path)],
            capture_output=True,text=True,timeout=30,cwd=tmp_path)
        assert result.returncode != 0 and expected in result.stderr
        assert output.read_text()=='preserve'
