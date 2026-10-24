"""Observer invariance and counterfactual accounting; no real-image claims."""
import ast
import copy
import inspect
import json

import pytest
import torch

from engines import hrm_lora_wtp_and_tap_engine as engine
from tools import crm_candidate_probe as probe
from tools.compare_prediction_traces import load_task
from tests.test_crm_selector_variants import configuration, mixed
from tests.test_prediction_stage_reference import IndexedAdapters, IndexedOriginal, reference


def test_source_is_identical_after_removing_the_single_observer_hook():
    source = inspect.getsource(engine.evaluate)
    modified = source.replace(probe.ANCHOR, probe.ANCHOR + probe.INSERT, 1)
    class RemoveCapture(ast.NodeTransformer):
        def visit_If(self, node):
            if (len(node.body) == 1 and isinstance(node.body[0],ast.Expr)
                    and isinstance(node.body[0].value,ast.Call)
                    and isinstance(node.body[0].value.func,ast.Attribute)
                    and node.body[0].value.func.attr == 'candidates'):
                return None
            return self.generic_visit(node)
    assert ast.dump(RemoveCapture().visit(ast.parse(modified))) == ast.dump(ast.parse(source))
    before = engine.evaluate,engine.start_trace
    with pytest.raises(RuntimeError,match='intentional'):
        with probe.installed(engine):
            raise RuntimeError('intentional')
    assert (engine.evaluate,engine.start_trace) == before


@pytest.mark.parametrize('weight',[.7,1.])
@pytest.mark.parametrize('batch_size',[1,7])
@pytest.mark.parametrize('mode,tau',[('gen',-10.),('gen',-100.),('msp',.6)])
def test_actual_evaluator_is_bit_identical_and_candidates_match_scalar_reference(
        monkeypatch,tmp_path,weight,batch_size,mode,tau):
    gen=torch.Generator().manual_seed(727)
    n=19
    original=torch.randn(n,20,generator=gen)
    bank=torch.randn(n,4,20,generator=gen)*torch.tensor([.2,1.,4.,10.])[None,:,None]
    rp=torch.randn(n,20,generator=gen)
    monkeypatch.setattr(engine,'rp_head_predict',lambda f,*a:rp[f[:,0].long()].clone())
    monkeypatch.setattr(engine,'args_ref',[None])
    monkeypatch.setattr(engine,'gate_stats',{})
    order=torch.randperm(n,generator=gen)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(order[:,None].float(),order%5),batch_size=batch_size)
    masks=[list(range(5*t,5*t+5)) for t in range(4)]
    mapping={c:c//5 for c in range(20)}
    outputs=[]
    initial_rng=torch.get_rng_state().clone()
    for instrument in (False,True):
        folder=tmp_path/str(instrument)
        folder.mkdir()
        (folder/'run.json').write_text(json.dumps(dict(dataset_name='synthetic',seed=42)))
        args=configuration(folder,'original',weight,tau)
        args.nb_classes=20
        args.En=mode
        model=IndexedAdapters(bank)
        orig=IndexedOriginal(original)
        torch.set_rng_state(initial_rng)
        def evaluate():
            return engine.evaluate(model,orig,loader,torch.device('cpu'),i=0,task_id=3,
                                   class_mask=masks,target_task_map=mapping,args=args)
        if instrument:
            with probe.installed(engine):
                stats=evaluate()
        else:
            stats=evaluate()
        _,rows=load_task(folder/'task_00.jsonl')
        outputs.append((stats,rows,model.calls,dict(engine.gate_stats),torch.get_rng_state().clone()))
    assert outputs[0][:4] == outputs[1][:4]
    assert torch.equal(outputs[0][4],outputs[1][4])
    captured=json.loads((tmp_path/'True'/'candidate_task_00.json').read_text())
    assert len(captured['samples'])==n
    for row,record,idx in zip(outputs[1][1],captured['samples'],order.tolist()):
        p,alternative=reference(original[idx].tolist(),bank[idx].tolist(),rp[idx].tolist(),weight,mode,tau)
        assert all(row[k]==v for k,v in p.items())
        ids=[p['proposal_after'],alternative] if p['crm_evaluated'] else [p['proposal_after']]
        assert [c['adapter'] for c in record['candidates']]==ids
        for candidate,adapter in zip(record['candidates'],ids):
            assert candidate['primary_logits']==bank[idx,adapter].tolist()
            reference_mixed=mixed(bank[idx,adapter].tolist(),rp[idx].tolist())
            assert candidate['fused_class']==max(range(20),key=reference_mixed.__getitem__)
        events,details=probe.inspect_record(record,row,20,4)
        assert events['actual_correct']==(p['final_class']==row['target'])
        assert events['available_correct']==any(c['fused_class']==row['target'] for c in record['candidates'])
        assert 0 <= details['candidate_scores'][record['selected']]['effective_rp_weight'] <= .5
    # Every recorded candidate must be checked, including the unselected one.
    record=copy.deepcopy(captured['samples'][0])
    record['candidates'][-1]['fused_logits'][0]+=1
    with pytest.raises(ValueError):
        probe.inspect_record(record,outputs[1][1][0],20,4)


def build_record(primary, rp, target, eligible=True):
    """Independent scalar candidate construction for event/tie edge cases."""
    from tests.test_prediction_stage_reference import standardize
    import math
    candidates=[]
    for adapter,p in enumerate(primary):
        ordered=sorted(p,reverse=True)
        share=1/(1+math.exp(ordered[0]-ordered[1]))
        avg=sum(p)/len(p)
        std=max(1e-6,math.sqrt(sum((v-avg)**2 for v in p)/len(p)))
        fused=[((1-share)*a+share*b)*std+avg for a,b in zip(standardize(p),standardize(rp))]
        candidates.append(dict(adapter=adapter,primary_logits=p,fused_logits=fused,
            primary_class=p.index(max(p)),fused_class=fused.index(max(fused))))
    scores=[probe.energy(c['primary_logits']) for c in candidates]
    selected=int(scores[1]>scores[0]) if eligible else 0
    chosen=candidates[selected]
    row=dict(index=0,target=target,proposal_after=0,final_adapter=selected,
        crm_evaluated=eligible,crm_used_alternative=bool(selected),rp_class=rp.index(max(rp)),
        drm_class=candidates[0]['primary_class'],crm_class=chosen['primary_class'],final_class=chosen['fused_class'])
    return dict(sample=dict(row),selected=selected,criterion=scores if eligible else None,
        candidates=candidates,rp_scores=rp,actual_final_logits=chosen['fused_logits']),row


@pytest.mark.parametrize('target,primary,rp,expected',[
    (0, [[6.,0.,0.,0.,0.],[0.,8.,0.,0.,0.]], [0.,0.,0.,0.,0.], 'selected_wrong_other_correct'),
    (2, [[6.,0.,0.,0.,0.],[0.,8.,0.,0.,0.]], [0.,0.,0.,0.,0.], 'both_candidates_wrong'),
    (1, [[6.,0.,0.,0.,0.],[0.,8.,0.,0.,0.]], [0.,0.,0.,0.,0.], 'selected_correct_other_wrong'),
    (0, [[6.,0.,0.,0.,0.],[8.,0.,0.,0.,0.]], [0.,0.,0.,0.,0.], 'both_candidates_correct'),
    (0, [[1.,.99,0.,0.,0.],[.9,.8,0.,0.,0.]], [0.,10.,0.,0.,0.], 'fusion_harmed_rp_wrong'),
    (1, [[1.,.99,0.,0.,0.],[.9,.8,0.,0.,0.]], [0.,10.,0.,0.,0.], 'fusion_repaired'),
])
def test_event_partitions_have_known_correct_and_incorrect_cases(target,primary,rp,expected):
    record,row=build_record(primary,rp,target)
    events,_=probe.inspect_record(record,row,5,2)
    assert events[expected]
    assert sum(events[k] for k in ('selected_wrong_other_correct','both_candidates_wrong',
                                  'selected_correct_other_wrong','both_candidates_correct'))==1
    assert events['available_correct']-events['actual_correct']==events['selected_wrong_other_correct']


def test_no_crm_has_no_invented_alternative_and_ties_are_visible():
    record,row=build_record([[0.,0.,0.,0.,0.]],[0.,0.,0.,0.,0.],0,eligible=False)
    events,detail=probe.inspect_record(record,row,5,2)
    assert events['no_crm'] and not events['crm_evaluated']
    assert not events['selected_wrong_other_correct']
    assert events['hypothetical_fusion_top_ties']
    assert detail['candidate_fused_ties']==[[0,1,2,3,4]]
    record['candidates'].append(record['candidates'][0])
    with pytest.raises(ValueError,match='count/selection'):
        probe.inspect_record(record,row,5,2)
