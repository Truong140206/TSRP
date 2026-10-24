"""Independent scalar selector references and invariance, not accuracy claims."""
import json
import math
from types import SimpleNamespace

import pytest
import torch

from engines import hrm_lora_wtp_and_tap_engine as engine
from tools import crm_selector_variants as variants
from tools.compare_prediction_traces import load_task
from tests.test_prediction_stage_reference import IndexedAdapters, IndexedOriginal, standardize, softmax


def configuration(folder, mode, weight=.7, tau=-10.):
    return SimpleNamespace(eval=True,world_size=1,num_tasks=4,dataset='synthetic',seed=42,
        print_freq=1000,train_mask=True,task_inc=False,nb_classes=25,En='gen',tau=tau,
        task_routing_mode='class',rp_head=True,rp_route_fusion_drm=True,rp_feature_source='lora',
        rp_lora_task=0,rp_route_fusion_weight=weight,rp_class_fusion_weight=.5,
        rp_class_fusion_gate='margin',rp_class_fusion_min_tasks=1,rp_fusion_ramp=0.,
        rp_class_fusion_sharpen=1.,classifier_union_audit=False,prediction_trace_dir=str(folder),
        experiment_crm_zenergy=mode=='zenergy',experiment_crm_fused_margin=mode=='fused_margin')


def mixed(primary, rp):
    order=sorted(primary,reverse=True)
    share=1/(1+math.exp(order[0]-order[1]))
    return [(1-share)*p+share*r for p,r in zip(standardize(primary),standardize(rp))]


def selector_reference(primary, rp, mode):
    if mode=='zenergy':
        z=standardize(primary)
        peak=max(z)
        return peak+.1*math.log(sum(math.exp((v-peak)/.1) for v in z))
    z=sorted(mixed(primary,rp),reverse=True)
    return math.tanh((z[0]-z[1])/2)


@pytest.mark.parametrize('mode',['zenergy','fused_margin'])
@pytest.mark.parametrize('weight',[.7,1.])
@pytest.mark.parametrize('batch_size',[1,13])
@pytest.mark.parametrize('tau',[-10.,-100.])
def test_actual_evaluator_matches_scalar_selection(monkeypatch,tmp_path,mode,weight,batch_size,tau):
    gen=torch.Generator().manual_seed(124)
    n=19
    original=torch.randn(n,25,generator=gen)
    bank=torch.randn(n,4,25,generator=gen)*torch.tensor([.2,1.,4.,10.])[None,:,None]
    rp=torch.randn(n,25,generator=gen)
    original[:,20:]=bank[:,:,20:]=1000
    monkeypatch.setattr(engine,'rp_head_predict',lambda f,*a:rp[f[:,0].long()].clone())
    monkeypatch.setattr(engine,'args_ref',[None])
    monkeypatch.setattr(engine,'gate_stats',{})
    args=configuration(tmp_path,mode,weight,tau)
    (tmp_path/'run.json').write_text(json.dumps(dict(dataset_name='synthetic',seed=42)))
    order=torch.randperm(n,generator=gen)
    loader=torch.utils.data.DataLoader(torch.utils.data.TensorDataset(order[:,None].float(),order%5),batch_size=batch_size)
    before=engine.evaluate
    with variants.installed(engine):
        engine.evaluate(IndexedAdapters(bank),IndexedOriginal(original),loader,torch.device('cpu'),
            i=0,task_id=3,class_mask=[list(range(t*5,t*5+5)) for t in range(5)],
            target_task_map={c:c//5 for c in range(25)},args=args)
    assert engine.evaluate is before
    _,rows=load_task(tmp_path/'task_00.jsonl')
    winner=lambda values:max(range(len(values)),key=values.__getitem__)
    for row,i in zip(rows,order.tolist()):
        orig=original[i,:20].tolist()
        candidates=bank[i,:,:20].tolist()
        r=rp[i,:20].tolist()
        initial=candidates[winner(orig)//5]
        top=sorted(range(20),key=initial.__getitem__,reverse=True)
        left=standardize([max(initial[t*5:t*5+5]) for t in range(4)])
        right=standardize([max(r[t*5:t*5+5]) for t in range(4)])
        proposal=winner([weight*a+(1-weight)*b for a,b in zip(left,right)])
        alt=top[1]//5
        confidence=-sum(p**.1*(1-p)**.1 for p in softmax(initial))
        eligible=confidence<tau
        chosen=proposal
        if eligible and selector_reference(candidates[alt],r,mode)>selector_reference(candidates[proposal],r,mode):
            chosen=alt
        assert row['proposal_after']==proposal
        assert row['crm_evaluated']==eligible
        assert row['final_adapter']==chosen, (i, proposal, alt,
            selector_reference(candidates[proposal],r,mode), selector_reference(candidates[alt],r,mode))
        assert row['final_class']==winner(mixed(candidates[chosen],r))


@pytest.mark.parametrize('mode',['zenergy','fused_margin'])
def test_selector_removes_additive_offset_dependence_and_handles_empty(monkeypatch,tmp_path,mode):
    args=configuration(tmp_path,mode)
    monkeypatch.setattr(engine,'args_ref',[args])
    candidates=torch.tensor([[[2.,1.9,0.,0.,float('-inf')],
                              [1.7,1.8,0.,0.,float('-inf')]]])
    rp=torch.zeros(1,5)
    a=variants.candidate_scores(engine,candidates,rp,args,4)
    shifted=candidates+torch.tensor([3.,7.])[None,:,None]
    b=variants.candidate_scores(engine,shifted,rp,args,4)
    torch.testing.assert_close(a,b,atol=1e-6,rtol=1e-5)
    assert variants.candidate_scores(engine,candidates[:0],rp[:0],args,4).shape==(0,2)
    if mode=='zenergy':
        torch.testing.assert_close(a,variants.candidate_scores(engine,candidates*2,rp,args,4),atol=1e-6,rtol=1e-5)
    with pytest.raises(RuntimeError):
        old=engine.evaluate
        with variants.installed(engine):
            raise RuntimeError('test')
    assert engine.evaluate is old


def test_invalid_masks_and_multiple_interventions_rejected(tmp_path):
    args=configuration(tmp_path,'zenergy')
    candidates=torch.tensor([[[1.,2.],[1.,float('-inf')]]])
    with pytest.raises(ValueError,match='finite seen'):
        variants.candidate_scores(engine,candidates,torch.zeros(1,2),args,4)
    args.experiment_crm_fused_margin=True
    with pytest.raises(ValueError,match='exactly one'):
        variants.candidate_scores(engine,torch.ones(1,2,4),torch.zeros(1,4),args,4)
