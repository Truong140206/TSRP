"""Fixed unlabelled rule, explicit harm example, and matched accounting."""
import copy
import inspect
from pathlib import Path
import subprocess
import sys

import pytest

from tools import analyze_route_acceptance as policy
from tests.test_crm_candidate_probe import build_record


def make_pair(full_primary,cg_primary,rp,target):
    full,_=build_record([full_primary],rp,target,eligible=False)
    cg,_=build_record([cg_primary],rp,target,eligible=False)
    full['candidates'][0]['adapter']=1
    full['sample']['proposal_after']=full['sample']['final_adapter']=1
    full['sample']['input_sha256']=cg['sample']['input_sha256']='same-image'
    return full,cg


def test_rule_accepts_only_clear_head_agreement_with_a_different_adapter():
    assert tuple(inspect.signature(policy.choose_path).parameters)==(
        'reference_adapter','proposed_adapter','proposed_primary','rp_scores')
    same=policy.choose_path(1,1,[0.,3.,1.],[0.,5.,1.])
    assert same['path']=='class_gate' and same['reason']=='same_adapter'
    agreed=policy.choose_path(0,1,[0.,3.,1.],[0.,5.,1.])
    assert agreed==dict(path='full',reason='heads_agree',agreed_class=1)
    disagreed=policy.choose_path(0,1,[3.,0.,1.],[0.,5.,1.])
    assert disagreed['path']=='class_gate' and disagreed['reason']=='heads_disagree'
    unclear=policy.choose_path(0,1,[3.,3.,1.],[0.,5.,1.])
    assert unclear['path']=='class_gate' and unclear['reason']=='unclear_head'
    near=policy.choose_path(0,1,[3.,3.-1e-8,1.],[5.,0.,1.])
    assert near['path']=='class_gate' and near['reason']=='unclear_head'


def test_class_permutation_and_positive_affine_transforms_preserve_selection():
    p,r=[1.,3.,0.,2.],[0.,8.,2.,1.]
    decision=policy.choose_path(0,1,p,r)
    transformed=policy.choose_path(0,1,[7*x-12 for x in p],[.3*x+4 for x in r])
    assert transformed==decision
    order=[2,0,3,1]
    permuted=policy.choose_path(0,1,[p[i] for i in order],[r[i] for i in order])
    assert permuted['path']==decision['path']
    assert permuted['agreed_class']==order.index(decision['agreed_class'])


@pytest.mark.parametrize('p,r',[
    ([1.],[1.]),([1.,2.],[1.,2.,3.]),([float('nan'),0.],[1.,0.]),
    ([1.,2.],[float('inf'),0.]),
])
def test_invalid_scores_are_rejected(p,r):
    with pytest.raises(ValueError,match='finite class vectors'):
        policy.choose_path(0,1,p,r)


def test_agreement_can_be_wrong_and_does_not_use_the_target_for_selection():
    full,cg=make_pair([0.,2.,0.,0.,0.],[4.,0.,0.,0.,0.],[0.,10.,0.,0.,0.],0)
    original=copy.deepcopy((full,cg))
    row=policy.analyze_pair(full,cg)
    assert (full,cg)==original
    assert row['decision']['path']=='full'
    assert row['correct']==dict(full=0,class_gate=1,consensus=0)
    counts=policy.empty_counts()
    policy.add_case(counts,row)
    assert counts['accepted_full_harms']==1
    assert counts['versus']['class_gate']['harmed']==1
    # Changing labels and image metadata can change accounting, not the rule.
    for rec in (full,cg):
        rec['sample']['target']=1
        rec['sample']['index']=99
        rec['sample']['input_sha256']='another-image'
    relabelled=policy.analyze_pair(full,cg)
    assert relabelled['decision']==row['decision']
    assert relabelled['correct']==dict(full=1,class_gate=0,consensus=1)


def test_fallback_retains_baseline_when_rp_and_proposed_primary_disagree():
    full,cg=make_pair([1.,.9,0.,0.,0.],[3.,0.,0.,0.,0.],[0.,10.,0.,0.,0.],0)
    row=policy.analyze_pair(full,cg)
    assert row['correct']==dict(full=0,class_gate=1,consensus=1)
    counts=policy.empty_counts()
    policy.add_case(counts,row)
    assert counts['avoided_full_harms']==1 and counts['accepted_full']==0


def test_strict_rule_can_reject_a_useful_full_fusion_repair():
    full,cg=make_pair([1.,.9,0.,0.,0.],[3.,0.,0.,0.,0.],[0.,10.,0.,0.,0.],1)
    row=policy.analyze_pair(full,cg)
    assert row['correct']==dict(full=1,class_gate=0,consensus=0)
    counts=policy.empty_counts()
    policy.add_case(counts,row)
    assert counts['rejected_full_repairs']==1 and counts['retained_full_repairs']==0


def test_task_balancing_and_both_wrong_acceptances_are_explicit():
    full,cg=make_pair([0.,2.,0.,0.,0.],[4.,0.,0.,0.,0.],[0.,10.,0.,0.,0.],1)
    first=policy.empty_counts()
    policy.add_case(first,policy.analyze_pair(full,cg))
    second=policy.empty_counts()
    for r in (full,cg):
        r['sample']['target']=2
    for _ in range(3):
        policy.add_case(second,policy.analyze_pair(full,cg))
    report=policy.summarize_tasks([first,second])
    assert report['counts']['samples']==4
    assert report['counts']['correct']['consensus']==1
    assert report['task_balanced_accuracy']['consensus']==50.  # not pooled 25%
    assert report['paired_task_balanced_pp']['class_gate']['net']==50.
    assert report['counts']['accepted_both_wrong']==3
    assert report['counts']['accepted_full']==4


def test_real_cli_is_stdlib_only_and_preserves_existing_or_invalid_output(tmp_path):
    command=[sys.executable,'-B','-S',str(Path(policy.__file__))]
    result=subprocess.run(command+['--help'],capture_output=True,text=True,timeout=30,cwd=tmp_path)
    assert result.returncode==0 and 'Exploratory' in result.stdout
    batch,output=tmp_path/'missing',tmp_path/'out.json'
    result=subprocess.run(command+['--batch-dir',str(batch),'--output',str(output)],
        capture_output=True,text=True,timeout=30,cwd=tmp_path)
    assert result.returncode!=0 and 'ROUTE_ACCEPTANCE_STOP' in result.stderr
    assert not output.exists()
    for path,message in ((batch/'out.json','outside the input batch'),(output,'new file')):
        output.write_text('preserve')
        result=subprocess.run(command+['--batch-dir',str(batch),'--output',str(path)],
            capture_output=True,text=True,timeout=30,cwd=tmp_path)
        assert result.returncode!=0 and message in result.stderr
        assert output.read_text()=='preserve'
