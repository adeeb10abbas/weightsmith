"""Fresh native paired captures and the historical pooled 72-key closed-form edit.

Construction uses only sealed development corrections. It does not select a
budget or update strength from rollout outcomes. All candidate workspaces and
source identities are exported before the separate held-out controller runs.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
from common import HERE, ROOT, REPO, canonical, load, protocol, read, save, sha
from weightsmith import config as C, distill

ARRAYS=('vlm.mlp_hidden','vlm.resid_out','vlm.img_mlp_hidden','vlm.img_resid_out')
LAYERS=tuple(range(17))

def verify(artifacts):
    for path,h in artifacts.items():
        if sha(path)!=h:raise ValueError(f'Sealed artifact changed: {path}')

def source(manifest_path):
    m=read(manifest_path);p=protocol()
    if m.get('cohort_id')!=p['cohort_id'] or m.get('split')!='development' or m['protocol_sha256']!=sha(HERE/'protocol.json'):
        raise ValueError('Wrong source manifest/cohort')
    verify(m['artifacts'])
    budgets=m['correction_budgets_available']
    if budgets!=[n for n in p['budgets'] if n<=m['eligible_pairs']] or not budgets:
        raise ValueError('No predeclared correction budget available or source ordering changed')
    spec_path=Path(m['specs'][str(max(budgets))]['correction']);spec=read(spec_path)
    if spec['source_manifest_sha256']!=sha(manifest_path) or len(spec['items'])!=max(budgets):
        raise ValueError('Correction spec differs from source manifest')
    if [r['pair_id'] for r in spec['items']] != [r['pair_id'] for r in m['pairs'][:max(budgets)]]:
        raise ValueError('Corrections are not the nested first eligible episode pairs')
    if len({r['obs_path'] for r in spec['items']})!=len(spec['items']):
        raise ValueError('Correction count must equal distinct original episodes')
    return m,spec_path,spec

def run_checked(command,log,*,env=None,input_text=None):
    with Path(log).open('x') as f:
        result=subprocess.run(command,cwd=REPO,env=env,input=input_text,text=True,stdout=f,stderr=subprocess.STDOUT)
    if result.returncode:raise RuntimeError(f'Construction stage failed ({result.returncode}); inspect {log}')

def prepare_capture(out,spec):
    episodes=out/'episodes';episodes.mkdir(exist_ok=True)
    for i,r in enumerate(spec['items']):
        ep=episodes/f'item-{i:04d}';ep.mkdir(exist_ok=True)
        path=ep/'obs.npz'
        if not path.exists():path.symlink_to(Path(r['obs_path']).resolve())
        if path.resolve()!=Path(r['obs_path']).resolve():raise ValueError('Capture source link differs')
        save(ep/'summary.json',{'prompt':r['recorded_prompt']},exclusive=True)
    for which in ('original','teacher'):
        request={'cand':'base','episodes_dir':str(episodes),'vlm_layers':list(LAYERS),
                 'image_tokens':True,'prompt_tokens':list(range(64)),'action_layers':[0],
                 'denoise_steps':[0],'action_tokens':'mean','action_attention':False,
                 'prompt':spec[which+'_prompt'],
                 'items':[{'episode':f'item-{i:04d}','k':r['decision_index'],'prompt':r[which+'_prompt']}
                          for i,r in enumerate(spec['items'])]}
        save(out/f'{which}-capture-spec.json',request,exclusive=True)

def capture(out,gpu):
    from weightsmith.locks import GpuLock
    env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),XLA_PYTHON_CLIENT_PREALLOCATE='false',
             XLA_PYTHON_CLIENT_MEM_FRACTION=C.ACTS_MEM_FRACTION,JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE))
    (out/'captures').mkdir(exist_ok=True)
    for which in ('original','teacher'):
        path=out/'captures'/which;receipt=out/'captures'/f'{which}-receipt.json'
        req=out/f'{which}-capture-spec.json'
        if receipt.exists():
            r=read(receipt);verify(r['artifacts'])
            if r['spec_sha256']!=sha(req):raise ValueError('Capture request differs')
            continue
        if path.exists():raise ValueError(f'Partial capture preserved; no automatic retry: {path}')
        start=time.time()
        with GpuLock(gpu):
            run_checked([str(C.OPENPI_PY),'-m','weightsmith.acts_worker','--output',str(path)],
                        out/'captures'/f'{which}.log',env=env,input_text=req.read_text())
        save(receipt,{'spec_sha256':sha(req),'wall_seconds':time.time()-start,
             'artifacts':{str(f):sha(f) for f in (path/'acts.npz',path/'index.json',REPO/'weightsmith/acts_worker.py')}})

def validate_capture(out,spec,dataset):
    data=read(dataset/'dataset.json')
    if data.get('arm')!='matched_decision' or data['train_inputs']!=len(spec['items']):
        raise ValueError('Fresh native correction targets are required')
    if read(dataset/'input-spec.json')!=spec:raise ValueError('Native targets and capture use different observations')
    gate=data['live_replay_gate']
    if gate.get('status')!='pass' or gate.get('tolerance')!=1e-6 or gate.get('maximum_error',1)>1e-6:
        raise ValueError('Exact fixed16 source replay has not passed')
    a={};index={};p=protocol()['edit'];rows=[]
    for which in ('original','teacher'):
        path=out/'captures'/which
        verify(read(out/'captures'/f'{which}-receipt.json')['artifacts'])
        index[which]=read(path/'index.json')
        if index[which]['vlm_layers']!=list(LAYERS) or index[which]['prompt_token_slots']!=list(range(64)):
            raise ValueError('Wrong captured support')
        if len(index[which]['items'])!=len(spec['items']):raise ValueError('Capture dropped source decisions')
        with np.load(path/'acts.npz',allow_pickle=False) as z:a[which]={key:z[key].copy() for key in (*ARRAYS,'actions','recorded_actions')}
        if not all(np.isfinite(v).all() for v in a[which].values()):raise ValueError('Nonfinite capture')
    for i,(it,record) in enumerate(zip(spec['items'],data['records'],strict=True)):
        for field in ('item_id','obs_path','decision_index','pair_id','live_context_sha256'):
            if it[field]!=record[field]:raise ValueError('Native target/capture input mismatch')
        with np.load(it['obs_path'],allow_pickle=False) as z:
            seed=int(z['seed'][it['decision_index']]);step=int(z['step'][it['decision_index']])
        for which in ('original','teacher'):
            r=index[which]['items'][i]
            if r['episode']!=f'item-{i:04d}' or r['k']!=it['decision_index'] or r['seed']!=seed or r['sim_step']!=step or r['prompt']!=it[which+'_prompt']:
                raise ValueError('Captured input identity differs')
        original,teacher=index['original']['items'][i],index['teacher']['items'][i]
        for left,right in zip(p['original_slots'],p['teacher_slots'],strict=True):
            if original['prompt_token_ids_at_slots'][left]!=teacher['prompt_token_ids_at_slots'][right] or original['prompt_token_valid_at_slots'][left]!=teacher['prompt_token_valid_at_slots'][right]:
                raise ValueError(f'Template/state alignment failed in correction {i}: {left}->{right}')
        path=dataset/record['file']
        if sha(path)!=record['sha256']:raise ValueError('Native teacher action file changed')
        with np.load(path,allow_pickle=False) as z:
            teacher_actions=z['teacher_actions_physical'];live=z['live_replay_actions_physical'];recorded=z['recorded_actions_physical']
        parity=float(np.max(np.abs(a['teacher']['actions'][i]-teacher_actions)))
        replay=float(np.max(np.abs(live-recorded)))
        if parity>.01 or replay>1e-6:raise ValueError(f'Unqualified teacher/source replay: teacher={parity}, live={replay}')
        rows.append({'pair_id':it['pair_id'],'teacher_max_abs_error':parity,'live_replay_max_abs_error':replay,
                     'singleton_live_original_max_abs_error':float(np.max(np.abs(a['original']['actions'][i]-recorded))),
                     'aligned_valid_token_slots':sum(original['prompt_token_valid_at_slots'][k] for k in p['original_slots'])})
    save(out/'parity.json',{'status':'pass','teacher_tolerance':.01,'source_replay_tolerance':1e-6,
         'records':rows,'note':'Fresh singleton native/instrumented teacher comparison; separate exact fixed16 source replay. Singleton-versus-live action drift is diagnostic only.'})
    return a

def pooled_key_values(original,teacher,n):
    p=protocol()['edit']
    if n not in protocol()['budgets'] or n>len(original[ARRAYS[0]]):raise ValueError('Unavailable/unregistered correction count')
    return distill.build_keys_values({k:original[k][:n] for k in ARRAYS},{k:teacher[k][:n] for k in ARRAYS},
             layers=LAYERS,mode='both',alpha=1.,slot_map=list(zip(p['original_slots'],p['teacher_slots'],strict=True)),
             cells=distill.IMAGE_CELLS,capture_layers=LAYERS)

def fit(K,V):
    if K.shape[0]!=72 or V.shape[0]!=72 or not np.isfinite(K).all() or not np.isfinite(V).all():raise ValueError('Expected finite historical72-key system')
    lam=1e-3*np.sum(K*K)/len(K)
    if lam<=0:raise ValueError('Degenerate key covariance')
    delta=K.T@np.linalg.solve(K@K.T+lam*np.eye(len(K)),V)
    return delta,{'keys':72,'rank_upper_bound':72,'ridge_relative':1e-3,'ridge_absolute':float(lam),
                  'fit_relative_error':float(np.linalg.norm(K@delta-V)/max(np.linalg.norm(V),1e-30))}

def solve(out,arrays,budgets,manifest_path):
    from weightsmith.lab.openpi_vlm_full import validate_workspace
    base=Path(C.BASE_WS).resolve();base_cfg=read(base/'workspace.json');candidates=[]
    for n in budgets:
        dest=out/f'n{n}';report=dest/'construction.json'
        if report.exists():
            old=read(report);verify(old['artifacts']);candidates.append(old);continue
        dest.mkdir(exist_ok=False);workspace=dest/'workspace';workspace.mkdir();(workspace/'edits').mkdir();(workspace/'scratch').mkdir()
        (workspace/'base').symlink_to(base/'base',target_is_directory=True)
        selected={distill.MLP_DOWN.format(l):base_cfg['tensors'][distill.MLP_DOWN.format(l)] for l in LAYERS}
        cfg={**base_cfg,'tensors':selected,'max_changed_fraction':1.,'max_changed_elements':base_cfg['total_parameters_and_buffers'],
             'include_action':False,'editable_parameters':sum(e['numel'] for e in selected.values()),
             'edit_scope':'Closed-form correction; VLM MLP down projections0–16 only',
             'frozen':['vision encoder','VLM embeddings and other projections','action expert and action heads']}
        save(workspace/'workspace.json',cfg)
        start=time.time();kv=pooled_key_values(arrays['original'],arrays['teacher'],n);layers=[]
        for l in LAYERS:
            delta,stats=fit(*kv[l]);name=distill.MLP_DOWN.format(l);entry=selected[name];base_path=base/'base'/entry['file']
            weight=np.load(base_path,allow_pickle=False)
            if weight.shape!=delta.shape:raise ValueError('Weight/capture shape mismatch')
            edited=distill.bf16_round(weight.astype(np.float64)+delta).astype(np.float32)
            target=workspace/'edits'/entry['file'];np.save(target,edited,allow_pickle=False)
            layers.append({'layer':l,'tensor':name,**stats,'changed_values':int(np.count_nonzero(edited!=weight)),
                 'relative_delta_l2':float(np.linalg.norm(delta)/max(np.linalg.norm(weight),1e-30)),
                 'base_sha256':sha(base_path),'edited_sha256':sha(target)})
            print({'budget':n,'layer':l,'changed_values':layers[-1]['changed_values']},flush=True)
        artifacts={str(f):sha(f) for f in [workspace/'workspace.json',*sorted((workspace/'edits').glob('*.npy'))]}
        candidate={'candidate_id':f'weightsmith_n{n}','budget':n,'workspace':str(workspace),'status':'complete',
            'method':'Historical pooled 72-key residual-increment closed-form recipe; fresh source episodes',
            'source_manifest':str(manifest_path),'source_manifest_sha256':sha(manifest_path),
            'parity_sha256':sha(out/'parity.json'),'layers':layers,'artifacts':artifacts,
            'workspace_validation':validate_workspace(workspace),'wall_seconds':time.time()-start,
            'construction_code_sha256':sha(__file__),'distill_code_sha256':sha(REPO/'weightsmith/distill.py')}
        save(report,candidate);candidates.append(candidate)
    registry={'status':'complete','cohort_id':protocol()['cohort_id'],'source_manifest':str(manifest_path),
              'source_manifest_sha256':sha(manifest_path),'candidates':candidates,
              'all_predeclared_available_budgets':budgets,'heldout_used':False}
    save(out/'candidates.json',registry)
    return registry

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--source-manifest',type=Path,required=True)
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True);ap.add_argument('--output',type=Path,default=ROOT/'edits')
    args=ap.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=True)
    m,spec_path,spec=source(args.source_manifest)
    files=[Path(__file__),HERE/'common.py',HERE/'protocol.json',HERE/'lora/fresh_lora.py',spec_path,args.source_manifest,
           REPO/'weightsmith/distill.py',REPO/'weightsmith/acts_worker.py',REPO/'weightsmith/lab/openpi_probe.py',
           C.BASE_WS/'workspace.json']
    frozen={'source_manifest':str(args.source_manifest.resolve()),'artifacts':{str(f.resolve()):sha(f) for f in files}}
    save(out/'construction-seal.json',frozen,exclusive=True);verify(frozen['artifacts'])
    status={'status':'running','stage':'native_source_replay','started_unix':time.time()};save(out/'status.json',status)
    try:
        dataset=out/'teacher-parity'
        if not (dataset/'dataset.json').exists():
            if dataset.exists():raise ValueError('Partial native preparation preserved; no automatic retry')
            run_checked([str(C.OPENPI_PY),str(HERE/'lora/fresh_lora.py'),'prepare','--spec',str(spec_path),
                         '--arm','matched_decision','--output',str(dataset),'--gpu',str(args.gpu)],out/'teacher-parity.log')
        prepare_capture(out,spec);status.update(stage='paired_activation_capture');save(out/'status.json',status)
        capture(out,args.gpu);verify(frozen['artifacts'])
        arrays=validate_capture(out,spec,dataset);status.update(stage='closed_form_solve');save(out/'status.json',status)
        result=solve(out,arrays,m['correction_budgets_available'],args.source_manifest.resolve())
        verify(frozen['artifacts']);status.update(status='complete',stage='complete',candidates=len(result['candidates']),completed_unix=time.time())
    except BaseException as exc:
        status.update(status='stopped',reason=repr(exc));raise
    finally:save(out/'status.json',status)

if __name__=='__main__':main()
