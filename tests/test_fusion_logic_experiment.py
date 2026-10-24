"""Synthetic mechanics/guards, not evidence of accuracy gains on real images."""
import copy
import json
import math
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from engines import hrm_lora_wtp_and_tap_engine as engine
from tools import fusion_logic_variants as variants
from tools import prediction_logic_probe as probe
from tools import run_fusion_logic_experiment as runner
from tools.compare_prediction_traces import load_task
from tests.test_prediction_stage_reference import IndexedAdapters, IndexedOriginal, standardize, softmax


def reference(original, bank, rp, weight, gate_z, crm_refresh, mode, tau):
    # Independent scalar implementation; 4 tasks, 5 classes each, unseen tail.
    original, bank, rp = original[:20], [r[:20] for r in bank], rp[:20]
    winner = lambda values: max(range(len(values)), key=values.__getitem__)
    tii = winner(original) // 5
    initial = bank[tii]
    order = sorted(range(20), key=initial.__getitem__, reverse=True)
    before, alternative = order[0] // 5, order[1] // 5
    left = standardize([max(initial[5*t:5*t+5]) for t in range(4)])
    right = standardize([max(rp[5*t:5*t+5]) for t in range(4)])
    proposal = winner([weight*a+(1-weight)*b for a,b in zip(left,right)])
    drm = bank[proposal]
    confidence_logits = initial
    if crm_refresh:
        confidence_logits = drm
        alternative = max((t for t in range(4) if t != proposal),
                          key=lambda t: max(drm[5*t:5*t+5]))
    probabilities = softmax(confidence_logits)
    confidence = (max(probabilities) if mode == 'msp' else
                  -sum(p**.1 * (1-p)**.1 for p in probabilities))
    eligible = confidence < tau
    def energy(values):
        peak = max(values)
        return peak + .1 * math.log(sum(math.exp((x-peak)/.1) for x in values))
    replace = eligible and energy(bank[alternative]) > energy(drm)
    adapter = alternative if replace else proposal
    primary = bank[adapter]
    gate_input = standardize(primary) if gate_z else primary
    top = sorted(gate_input, reverse=True)
    share = 1 / (1 + math.exp(top[0]-top[1]))
    combined = [(1-share)*p+share*r for p,r in zip(standardize(primary),standardize(rp))]
    return dict(tii_task=tii, proposal_before=before, proposal_after=proposal,
                drm_class=winner(drm), crm_class=winner(primary), final_class=winner(combined),
                final_adapter=adapter, crm_evaluated=eligible, crm_used_alternative=replace)


@pytest.mark.parametrize('weight', [.7, 1.])
@pytest.mark.parametrize('gate_z,crm_refresh', [(False,False),(True,False),(False,True),(True,True)])
@pytest.mark.parametrize('mode,tau', [('gen',-10.),('msp',.6)])
@pytest.mark.parametrize('batch_size', [1, 13])
def test_actual_experimental_evaluator_against_scalar_reference(
        monkeypatch,tmp_path,weight,gate_z,crm_refresh,mode,tau,batch_size):
    generator = torch.Generator().manual_seed(71)
    n = 29
    original = torch.randn(n,25,generator=generator)
    bank = torch.randn(n,4,25,generator=generator)
    bank *= torch.tensor([.2,1.,4.,10.])[torch.arange(n)%4,None,None]
    rp = torch.randn(n,25,generator=generator)
    original[:,20:], bank[:,:,20:], rp[:,20:] = 1000,1000,-1000
    monkeypatch.setattr(engine,'rp_head_predict',lambda features,*a:rp[features[:,0].long()].clone())
    monkeypatch.setattr(engine,'args_ref',[None])
    monkeypatch.setattr(engine,'gate_stats',{})
    args = SimpleNamespace(eval=True,world_size=1,num_tasks=4,dataset='synthetic',seed=42,
        print_freq=1000,train_mask=True,task_inc=False,nb_classes=25,En=mode,tau=tau,
        task_routing_mode='class',rp_head=True,rp_route_fusion_drm=True,rp_feature_source='lora',
        rp_lora_task=0,rp_route_fusion_weight=weight,rp_class_fusion_weight=.5,
        rp_class_fusion_gate='margin',rp_class_fusion_min_tasks=1,rp_fusion_ramp=0.,
        classifier_union_audit=False,prediction_trace_dir=str(tmp_path),
        experiment_gate_zscore=gate_z,experiment_crm_refresh=crm_refresh)
    (tmp_path/'run.json').write_text(json.dumps(dict(dataset_name='synthetic',seed=42)))
    order = torch.randperm(n,generator=generator)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(order[:,None].float(),order%5),
                                        batch_size=batch_size)
    old_eval,old_gate = engine.evaluate,engine._fusion_gate
    with variants.installed(engine):
        engine.evaluate(IndexedAdapters(bank),IndexedOriginal(original),loader,torch.device('cpu'),
                        i=0,task_id=3,class_mask=[list(range(t*5,t*5+5)) for t in range(5)],
                        target_task_map={c:c//5 for c in range(25)},args=args)
    assert (engine.evaluate,engine._fusion_gate) == (old_eval,old_gate)
    _,rows = load_task(tmp_path/'task_00.jsonl')
    for row,idx in zip(rows,order.tolist()):
        expected = reference(original[idx].tolist(),bank[idx].tolist(),rp[idx].tolist(),
                             weight,gate_z,crm_refresh,mode,tau)
        assert {k:row[k] for k in expected} == expected


def test_z_gate_affine_invariance_masks_and_restoration(monkeypatch):
    config = SimpleNamespace(experiment_gate_zscore=True,rp_class_fusion_gate='margin')
    monkeypatch.setattr(engine,'args_ref',[config])
    logits = torch.tensor([[1.,0.,-1.,float('-inf')]])
    valid = torch.isfinite(logits)
    old_gate = engine._fusion_gate
    with pytest.raises(RuntimeError,match='intentional'):
        with variants.installed(engine):
            a = engine._fusion_gate(logits*.1,valid,'margin')
            b = engine._fusion_gate(logits*4+17,valid,'margin')
            torch.testing.assert_close(a,b)
            assert torch.isfinite(a).all() and (a>=0).all() and (a<=1).all()
            config.experiment_gate_zscore=False
            torch.testing.assert_close(engine._fusion_gate(logits,valid,'margin'),old_gate(logits,valid,'margin'),rtol=0,atol=0)
            config.experiment_gate_zscore=True
            zeros = torch.zeros(2,4)
            assert torch.equal(engine._fusion_gate(zeros,torch.ones_like(zeros,dtype=torch.bool),'margin'),torch.ones(2,1))
            raise RuntimeError('intentional')
    assert engine._fusion_gate is old_gate


def test_crm_refresh_changes_eligibility_and_excludes_current_adapter():
    scores = torch.tensor([[10.,9.,1.,0.,-1.,-2.], [0.,0.,0.,0.,0.,0.]])
    args = SimpleNamespace(En='msp',tau=.6,task_routing_mode='class')
    indices,candidates = variants.refresh_crm(engine,scores,torch.tensor([0,2]),[[0,1],[2,3],[4,5]],3,args)
    assert indices.tolist() == [1]
    assert candidates.tolist() == [[0,1],[2,0]]
    assert torch.all(candidates[:,0] != candidates[:,1])
    with pytest.raises(ValueError,match='at least two'):
        variants.refresh_crm(engine,scores,torch.tensor([0,0]),[[0,1]],1,args)


def test_source_guard_fails_closed_and_supports_crlf(tmp_path):
    source = variants.checked_source()
    copy_path = tmp_path/'engine.py'
    copy_path.write_bytes(source.replace('\n','\r\n').encode())
    assert variants.checked_source(copy_path) == source
    copy_path.write_text(source.replace('corr_id=None','corr_id=0',1),encoding='utf-8')
    with pytest.raises(ValueError,match='exact checked'):
        variants.checked_source(copy_path)


def config(root,trace,weight=.7):
    launch = SimpleNamespace(dataset='ima',seed=42,batch_size=24,port=29558,resolved_data_path=root)
    cmd = runner.base.build_command(launch,root/'tii',root/'lora','full')
    parsed = runner.base.grid.parsed_config(cmd)
    parsed.prediction_trace_dir = str(trace)
    parsed.config = parsed.subparser_name = 'ima_lora'
    parsed.world_size,parsed.nb_classes,parsed.print_freq = 1,200,1000
    parsed.rp_route_fusion_weight = weight
    return parsed


def manifest(folder,arm):
    folder.mkdir(parents=True)
    (folder/'run.json').write_text(json.dumps(dict(dataset_name='Split-Imagenet-A',seed=42,
                                               variant=arm,common=dict(source='synthetic'))))


@pytest.mark.parametrize('experiment', ['gate_crm', 'crm_selector'])
def test_full_sequence_replay_and_all_final_arms(monkeypatch,tmp_path,experiment):
    import main as application
    import utils
    from trainers import lora_trainer as trainer
    spec = probe.variant_spec(experiment)
    monkeypatch.setattr(utils,'cleanup_distributed',lambda:None)
    monkeypatch.setattr(engine,'args_ref',[None])
    monkeypatch.setattr(engine,'gate_stats',{})
    generator = torch.Generator().manual_seed(23)
    sizes = [2+t%3 for t in range(10)]
    n = sum(sizes)
    original = torch.randn(n,200,generator=generator)
    bank = torch.randn(n,10,200,generator=generator)
    rp = torch.randn(n,200,generator=generator)
    monkeypatch.setattr(engine,'rp_head_predict',lambda features,*a:rp[features[:,0].long()].clone())
    loaders,offset = [],0
    for t,count in enumerate(sizes):
        data = torch.utils.data.TensorDataset(torch.arange(offset,offset+count)[:,None].float(),
                                             torch.arange(20*t,20*t+count))
        loaders.append(dict(val=torch.utils.data.DataLoader(data,batch_size=2)))
        offset += count
    masks = [list(range(20*t,20*t+20)) for t in range(10)]
    mapping = {c:c//20 for c in range(200)}
    history,destination = tmp_path/'history',tmp_path/'new'
    for arm in ('full','class_gate'):
        manifest(history/arm,arm)
    for arm in spec.ARMS:
        manifest(destination/arm,arm)
    model,orig,device = IndexedAdapters(bank),IndexedOriginal(original),torch.device('cpu')
    for arm,weight in (('full',.7),('class_gate',1.)):
        args = config(tmp_path,history/arm,weight)
        for t in range(10):
            engine.evaluate(model,orig,loaders[t]['val'],device,i=t,task_id=9,
                            class_mask=masks,target_task_map=mapping,args=args)
    seen=[]
    def fake_main(args):
        matrix=np.zeros((10,10))
        for stage in range(10):
            seen.append(stage)
            trainer.evaluate_till_now(model,orig,loaders,device,stage,masks,mapping,matrix,args)
    old_eval,old_start,old_gate,old_till = engine.evaluate,engine.start_trace,engine._fusion_gate,trainer.evaluate_till_now
    monkeypatch.setattr(application,'get_args',lambda:config(tmp_path,destination/'full'))
    monkeypatch.setattr(application,'main',fake_main)
    probe.run_probe(history,destination,['unused'],spec)
    assert seen == list(range(10))
    assert (engine.evaluate,engine.start_trace,engine._fusion_gate,trainer.evaluate_till_now) == (old_eval,old_start,old_gate,old_till)
    result=probe.summarize(history,destination,spec)
    assert set(result['arms']) == set(spec.ARMS)
    assert set(result['conditional_routing']) == (
        {'original','gate_z','crm_refresh','both'} if experiment == 'gate_crm'
        else {'original','zenergy','fused_margin'})
    stats=json.loads((destination/'final_stats.json').read_text())
    assert len(stats)==len(spec.ARMS)-1
    for arm in stats:
        assert stats[arm]['Acc@1'] == pytest.approx(result['arms'][arm]['task_balanced_accuracy']['final_class'],abs=1e-5)
    assert set(json.loads((destination/'timing.json').read_text())['final_pass_seconds']) == set(spec.ARMS)
    if experiment == 'gate_crm':
        from tools import audit_fusion_logic_stages as stage_audit
        from tools.audit_prediction_stages import sha256
        stored = dict(result, dataset='Split-Imagenet-A', seed=42,
            historical_controls='EXACT_TRACE_ROWS_BOTH_ARMS',
            historical_full_all_stages='PASS_AT_1e-4_LOG_PRECISION',
            trace_sha256={str(p.relative_to(destination)).replace('\\','/'):sha256(p)
                          for p in destination.glob('*/task_*.jsonl')})
        summary_path = destination/'summary.json'
        summary_path.write_text(json.dumps(stored),encoding='utf-8')
        before = {p:p.read_bytes() for p in destination.rglob('*') if p.is_file()}
        stage_result = stage_audit.audit(destination)
        assert set(stage_result['conditions']) == {'original','gate_z','crm_refresh','both'}
        for name, values in stage_result['conditions'].items():
            assert values['stage_pairs']['final_class'] == result['conditional_routing'][name]
        assert {p:p.read_bytes() for p in before} == before
        stored['arms']['full']['task_balanced_accuracy']['final_class'] += 1
        summary_path.write_text(json.dumps(stored),encoding='utf-8')
        with pytest.raises(ValueError,match='summary mismatch'):
            stage_audit.audit(destination)
    # No silent partial/mutated trace acceptance.
    path=destination/list(spec.ARMS)[-1]/'task_00.jsonl'
    path.write_text(path.read_text().replace('"rp_sha256":','"bad_rp_sha256":',1))
    with pytest.raises(ValueError,match='checksum'):
        probe.summarize(history,destination,spec)


@pytest.mark.parametrize('problem',[None,'seed','source','checkpoints','runtime','data','config','missing_config'])
def test_preflight_is_read_only_and_rejects_mismatch(monkeypatch,tmp_path,problem):
    out,data,history=(tmp_path/name for name in ('out','data','old'))
    for p in (out,data,history):
        p.mkdir()
    runtime={name:'test' for name in ('torch','torchvision','timm','numpy')}
    runtime.update(python=sys.version,torch_cuda=torch.version.cuda,cuda_visible_devices=runner.os.environ.get('CUDA_VISIBLE_DEVICES'))
    common=dict(dataset_path=str(data),source_sha256='same',checkpoints=[],runtime=runtime)
    if problem=='source':common['source_sha256']='different'
    if problem=='data':common['dataset_path']='different'
    if problem=='checkpoints':common['checkpoints']=[dict(path='different')]
    if problem=='runtime':common['runtime']=dict(runtime,torch='different')
    launch=SimpleNamespace(dataset='ima',seed=42,batch_size=24,port=29558,resolved_data_path=data)
    configs={}
    for arm in ('full','class_gate'):
        cmd=runner.base.build_command(launch,out/'ima_tii_original_10tasks_seed42',
                                     out/'ima_lora_rank8_baseline_10tasks_seed42',arm)
        c=vars(runner.base.grid.parsed_config(cmd))
        c.update(config='ima_lora',subparser_name='ima_lora',world_size=1,rank=0,gpu=0,
                 distributed=True,dist_backend='nccl',nb_classes=200,class_names=['c%d'%i for i in range(200)])
        configs[arm]=json.loads(json.dumps(c))
    if problem=='config':configs['class_gate']['rp_class_fusion_weight']=.7
    if problem=='missing_config':del configs['full']['lr_noise']
    monkeypatch.setattr(sys,'argv',['experiment','--output-root',str(out),'--data-root',str(data),
                                    '--history',str(history),'--seed','42','--tag','trial']
                                    + (['--run'] if problem in ('config','missing_config') else []))
    monkeypatch.setattr(runner,'history_hashes',lambda p:{})
    monkeypatch.setattr(runner.stages,'audit',lambda p:dict(dataset='Split-Imagenet-A',seed=43 if problem=='seed' else 42))
    monkeypatch.setattr(runner.compare,'load_task',lambda p:(dict(run=dict(common=common),configuration=configs[p.parent.name]),[]))
    monkeypatch.setattr(runner.base,'resolve_data',lambda args:data)
    monkeypatch.setattr(runner.base.verify,'source_digest',lambda:'same')
    monkeypatch.setattr(runner.base.grid,'manifest',lambda *a:([],{}))
    monkeypatch.setattr(runner.importlib.metadata,'version',lambda name:'test')
    def forbidden(*a,**k):raise AssertionError('Read-only preflight must not launch/write')
    if problem in ('config','missing_config'):
        monkeypatch.setattr(runner.base.grid,'manifest',forbidden)
    for name in ('idle_gpu','execute','save_new'):
        monkeypatch.setattr(runner.base,name,forbidden)
    if problem:
        with pytest.raises(ValueError):runner.main()
    else:runner.main()
    assert list(out.iterdir())==[]


def test_child_cli_and_experimental_config_guards(monkeypatch,tmp_path):
    import main as application
    args=SimpleNamespace(dataset='ima',seed=44,batch_size=24,port=29558,
                         history=tmp_path/'history',resolved_data_path=tmp_path)
    cmd=runner.child_command(args,tmp_path/'tii',tmp_path/'lora',tmp_path/'new')
    assert cmd[:3]==[sys.executable,'-m','torch.distributed.run']
    assert '--nproc_per_node=1' in cmd
    at = cmd.index('--')
    monkeypatch.setattr(sys,'argv',['main.py',*cmd[at+1:]])
    parsed = application.get_args()
    assert parsed.dataset == 'Split-Imagenet-A' and parsed.seed == 44
    assert parsed.eval and parsed.num_tasks == 10
    assert parsed.prediction_trace_dir == str(tmp_path/'new'/'full')
    assert parsed.rp_route_fusion_weight == .7
    original=config(tmp_path,tmp_path/'trace')
    for arm,(weight,gate,crm) in variants.ARMS.items():
        current=dict(vars(original),rp_route_fusion_weight=weight)
        if arm not in ('full','class_gate'):
            current.update(experiment_gate_zscore=gate,experiment_crm_refresh=crm)
        probe.check_arm_configuration(current,json.loads(json.dumps(vars(original))),arm)
        current['rp_lambda']=42
        with pytest.raises(ValueError,match='configuration differs'):
            probe.check_arm_configuration(current,vars(original),arm)
