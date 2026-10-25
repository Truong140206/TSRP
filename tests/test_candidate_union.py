"""Candidate coverage, missed choices, deduplication and CLI safety."""
import copy
from pathlib import Path
import subprocess
import sys

import pytest

from tools import analyze_candidate_union as union
from tests.test_crm_candidate_probe import build_record


def record(choices, rp, target):
    pairs=list(choices.items())
    item,_=build_record([p for _,p in pairs],rp,target,eligible=len(pairs)==2)
    for candidate,(adapter,_) in zip(item['candidates'],pairs):
        candidate['adapter']=adapter
    item['sample']['proposal_after']=pairs[0][0]
    item['sample']['final_adapter']=pairs[item['selected']][0]
    item['sample']['input_sha256']='same-input'
    return item


def test_new_correct_candidate_can_be_unused_by_both_original_outputs():
    rp=[0.,0.,0.,0.,0.]
    common=[0.,8.,0.,0.,0.]
    full=record({0:[4.,0.,0.,0.,0.],1:common},rp,0)
    cg=record({2:[0.,0.,5.,0.,0.],1:common},rp,0)
    original=copy.deepcopy((full,cg))
    row=union.analyze_pair(full,cg)
    assert (full,cg)==original
    assert row['events']['available_full_only']
    assert row['events']['full_selector_missed']
    assert not row['events']['final_output_oracle_correct']
    assert row['events']['union_available']
    assert row['correct_fused_adapters']==[0]
    assert row['added_full_adapters']==[0]
    assert row['removed_cg_adapters']==[2]
    assert row['sizes']['union_unique']==3 and row['sizes']['shared']==1


def test_union_preserves_a_correct_candidate_discarded_by_full():
    rp=[0.,0.,0.,1.,0.]
    common=[0.,8.,0.,0.,0.]
    cg=record({0:[10.,0.,0.,0.,0.],1:common},rp,0)
    full=record({2:[0.,0.,9.,0.,0.],1:common},rp,0)
    row=union.analyze_pair(full,cg)
    assert row['events']['cg_actual_correct']
    assert not row['events']['full_actual_correct']
    assert row['events']['available_cg_only']
    assert row['events']['union_available']
    assert row['correct_fused_adapters']==[0]


def test_existing_cg_option_is_not_credited_as_new_routing_coverage():
    rp=[0.,0.,0.,0.,0.]
    common=[0.,8.,0.,0.,0.]
    cg=record({0:[6.,0.,0.,0.,0.],1:common},rp,0)
    full=record({2:[10.,0.,0.,0.,0.],1:common},rp,0)
    row=union.analyze_pair(full,cg)
    assert row['events']['cg_selector_missed']
    assert row['events']['full_actual_correct']
    assert row['events']['available_both']
    assert not row['events']['available_full_only']
    assert row['added_full_adapters']==[2]


@pytest.mark.parametrize('primary,rp,expected',[
    ([1.,.9,0.,0.,0.],[0.,10.,0.,0.,0.],'unavailable_primary_only'),
    ([0.,10.,0.,0.,0.],[2.,0.,0.,0.,0.],'unavailable_rp_only'),
    ([0.,10.,0.,0.,0.],[0.,0.,2.,0.,0.],'unavailable_neither_primary_nor_rp'),
])
def test_no_correct_fused_candidate_is_split_by_remaining_primary_and_rp_evidence(primary,rp,expected):
    full,cg=record({0:primary},rp,0),record({1:primary},rp,0)
    row=union.analyze_pair(full,cg)
    assert row['events']['available_neither']
    assert row['events'][expected]
    assert sum(row['events'][k] for k in union.EVENTS if k.startswith('unavailable_'))==1
    assert row['sizes']['union_unique']==2  # no invented CRM alternative


def test_two_wrong_heads_can_still_have_a_correct_fused_candidate():
    p,rp=[2.,2.1,0.,0.,0.],[2.,0.,2.1,0.,0.]
    row=union.analyze_pair(record({0:p},rp,0),record({1:p},rp,0))
    assert not row['events']['union_primary_available'] and not row['events']['rp_correct']
    assert row['events']['union_available']


def test_duplicate_slots_and_cross_arm_duplicates_are_one_adapter():
    rp,p=[0.,1.,0.,0.,0.],[6.,0.,0.,0.,0.]
    item=record({0:p,1:p},rp,0)
    item['candidates'][1]['adapter']=0
    row=union.analyze_pair(copy.deepcopy(item),item)
    assert row['sizes']==dict(cg_unique=1,full_unique=1,union_unique=1,shared=1,
        added_full=0,removed_cg=0,cg_duplicate_slots=1,full_duplicate_slots=1)
    assert len(row['unique_candidates'][0]['occurrences'])==4
    assert not row['events']['has_added_full_adapter']


def test_different_scores_for_same_adapter_are_not_silently_merged():
    rp=[0.,0.,0.,0.,0.]
    full=record({0:[6.,0.,0.,0.,0.]},rp,0)
    cg=record({0:[7.,0.,0.,0.,0.]},rp,0)
    with pytest.raises(ValueError,match='different scores.*DETAIL='):
        union.analyze_pair(full,cg)


def test_duplicate_diagnostics_measure_drift_without_relaxing_the_guard():
    rp=[0.]*5
    full=record({0:[6.,0.,0.,0.,0.]},rp,0)
    cg=record({0:[7.,0.,0.,0.,0.]},rp,0)
    original=copy.deepcopy((full,cg))
    pairs=union.duplicate_comparisons(full,cg)
    assert (full,cg)==original
    assert len(pairs)==1
    row=pairs[0]
    assert row['score_mismatch'] and row['strict_reject']
    assert not row['prediction_mismatch']
    assert row['cross_arm']
    assert row['primary']['max_abs_difference']==1.
    assert row['primary']['max_tolerance_ratio']>1.
    assert row['primary']['coordinates_exceeding_tolerance']==1
    assert row['primary']['standardized_max_abs_difference']==pytest.approx(0.,abs=1e-12)
    assert row['left']['arm']=='class_gate' and row['right']['arm']=='full'
    assert row['left']['effective_rp_weight']<row['right']['effective_rp_weight']
    with pytest.raises(ValueError,match='cannot deduplicate'):
        union.analyze_pair(full,cg)


def test_fused_only_drift_and_prediction_flip_have_distinct_diagnostics():
    cg=record({0:[6.,0.,0.,0.,0.]},[0.]*5,0)
    full=copy.deepcopy(cg)
    full['candidates'][0]['fused_logits']=[v+5e-5 for v in cg['candidates'][0]['fused_logits']]
    row=union.duplicate_comparisons(full,cg)[0]
    assert row['primary']['within_existing_tolerance']
    assert not row['fused']['within_existing_tolerance']
    assert not row['prediction_mismatch']
    full=record({0:[0.,6.,0.,0.,0.]},[0.]*5,0)
    row=union.duplicate_comparisons(full,cg)[0]
    assert row['primary_class_changed'] and row['fused_class_changed']
    assert row['prediction_mismatch']


def test_diagnostic_covers_within_arm_and_cross_arm_pairs_not_only_first_duplicate():
    item=record({0:[6.,0.,0.,0.,0.],1:[6.,0.,0.,0.,0.]},[0.]*5,0)
    item['candidates'][1]['adapter']=0
    pairs=union.duplicate_comparisons(copy.deepcopy(item),item)
    assert len(pairs)==6
    assert sum(p['cross_arm'] for p in pairs)==4
    assert not any(p['strict_reject'] for p in pairs)
    other=record({2:[6.,0.,0.,0.,0.]},[0.]*5,0)
    single=record({0:[6.,0.,0.,0.,0.]},[0.]*5,0)
    assert union.duplicate_comparisons(other,single)==[]


def test_near_tie_class_change_is_reported_even_when_scores_are_within_tolerance():
    rp=[0.]*5
    cg=record({0:[1.,1.-1e-7,0.,0.,0.]},rp,0)
    full=record({0:[1.-1e-7,1.,0.,0.,0.]},rp,0)
    row=union.duplicate_comparisons(full,cg)[0]
    assert not row['score_mismatch']
    assert row['prediction_mismatch'] and row['strict_reject']
    with pytest.raises(ValueError,match='different predictions'):
        union.analyze_pair(full,cg)


@pytest.mark.parametrize('left,right', [([0.],[0.]),([0.,1.],[0.]),([0.,float('nan')],[0.,1.])])
def test_invalid_vectors_do_not_become_tolerance_diagnostics(left,right):
    with pytest.raises(ValueError,match='finite duplicate'):
        union.vector_difference(left,right)


def test_ties_follow_captured_classes_not_target_favouring_choices():
    item=record({0:[0.]*5},[0.]*5,1)
    row=union.analyze_pair(copy.deepcopy(item),item)
    assert row['events']['fused_ties_present']
    assert not row['events']['union_available']
    other=copy.deepcopy(item)
    other['candidates'][0]['fused_class']=1
    other['sample']['final_class']=1
    with pytest.raises(ValueError,match='different predictions'):
        union.analyze_pair(other,item)


@pytest.mark.parametrize('field', ['target','index','input_sha256'])
def test_paired_identity_must_match(field):
    item=record({0:[6.,0.,0.,0.,0.]},[0.]*5,0)
    other=copy.deepcopy(item)
    other['sample'][field]='different'
    with pytest.raises(ValueError,match='Paired'):
        union.analyze_pair(other,item)


def test_original_alternative_must_be_shared():
    p=[6.,0.,0.,0.,0.]
    cg=record({0:p,1:p},[0.]*5,0)
    full=record({0:p,2:p},[0.]*5,0)
    with pytest.raises(ValueError,match='pre-routing alternative'):
        union.analyze_pair(full,cg)


def test_task_balanced_oracle_is_not_pooled_accuracy():
    rp,common=[0.]*5,[0.,8.,0.,0.,0.]
    full=record({0:[4.,0.,0.,0.,0.],1:common},rp,0)
    cg=record({2:[0.,0.,5.,0.,0.],1:common},rp,0)
    first,second=union.empty_counts(),union.empty_counts()
    union.add_case(first,union.analyze_pair(full,cg))
    for item in (full,cg):
        item['sample']['target']=4
    for _ in range(3):
        union.add_case(second,union.analyze_pair(full,cg))
    report=union.summarize_tasks([first,second])
    assert report['counts']['samples']==4
    assert report['counts']['events']['available_full_only']==1
    assert report['task_balanced_pp']['available_full_only']==50.  # not pooled 25%
    assert report['task_balanced_pp']['union_available']==50.
    assert report['task_balanced_candidates_per_image']['union_unique']==3.


def test_cli_is_stdlib_only_and_preserves_inputs_and_existing_output(tmp_path):
    command=[sys.executable,'-B','-S',str(Path(union.__file__))]
    result=subprocess.run(command+['--help'],cwd=tmp_path,capture_output=True,text=True,timeout=30)
    assert result.returncode==0 and 'Label-aware' in result.stdout
    batch,output=tmp_path/'missing',tmp_path/'out.json'
    result=subprocess.run(command+['--batch-dir',str(batch),'--output',str(output)],
        capture_output=True,text=True,timeout=30)
    assert result.returncode!=0 and 'CANDIDATE_UNION_STOP' in result.stderr
    assert not output.exists()
    output.write_text('preserve')
    for destination,message in ((output,'new file'),(batch/'out.json','outside')):
        result=subprocess.run(command+['--batch-dir',str(batch),'--output',str(destination)],
            capture_output=True,text=True,timeout=30)
        assert result.returncode!=0 and message in result.stderr
        assert output.read_text()=='preserve'
