"""Certify punctuation-safe native serving and unchanged temporal candidates.

No policy rollouts: replay recorded observations with native fixed16 inference.
All twelve legacy temporal development candidates must have bit-identical
physical actions under the corrected implementation. The twelve registered
prompt variants additionally receive zero-injection and exact-prefix controls.
"""
from __future__ import annotations
import argparse
import copy
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
STUDY=HERE.parent.parent/'rss_few_pair_validation_20261120'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(STUDY));sys.path.insert(0,str(HERE))
import artifacts as A
import verify as V


def old_candidate(root):
    root=Path(root).resolve();spec=A.read(root/'baseline.json')
    if (not root.is_relative_to(A.LEGACY_ROOT) or spec.get('schema')!=A.LEGACY_SCHEMA
        or spec.get('runtime_code_sha256')!=A.LEGACY_RUNTIME_SHA256
        or spec.get('artifact_code_sha256')!=A.LEGACY_ARTIFACT_SHA256):
        raise ValueError('Qualification accepts only the frozen temporal baseline implementation')
    if A.sha(root/'direction.npy')!=spec['direction_sha256']:raise ValueError('Legacy direction changed')
    for path,digest in spec['construction'].get('artifacts',{}).items():
        if A.sha(path)!=digest:raise ValueError('Legacy construction input changed')
    direction=np.load(root/'direction.npy',allow_pickle=False)
    if direction.dtype!=np.float32 or not np.isfinite(direction).all():raise ValueError('Invalid legacy direction')
    return spec,direction


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True)
    ap.add_argument('--output',type=Path,default=A.COMPATIBILITY_GATE.parent)
    args=ap.parse_args()
    from weightsmith import config as C
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='true',
        XLA_PYTHON_CLIENT_MEM_FRACTION=C.SERVER_MEM_FRACTION,JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE),
        JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='1')
    out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    old_dir=STUDY/'baselines';temporal=REPO/'projects/rss-few-pair-validation-20261120/temporal'
    source_status=A.read(temporal/'source-status.json');manifest=A.read(source_status['manifest'])
    if A.sha(source_status['manifest'])!=source_status['manifest_sha256']:raise ValueError('Temporal source manifest changed')
    source_spec=Path(manifest['spec']);orig,identities=V.requests(source_spec)
    registry_path=temporal/'adaptation/baselines/candidates.json';registry=A.read(registry_path)
    candidates=registry['candidates']
    if len(candidates)!=12 or {c['family'] for c in candidates}!={'embedding','residual_g0','residual_g1','residual_g2'}:
        raise ValueError('Need all twelve registered temporal development candidates')
    for family in ('embedding','residual_g0','residual_g1','residual_g2'):
        if sorted(c['strength'] for c in candidates if c['family']==family)!=[.5,1.,2.]:raise ValueError('Legacy strength grid changed')
    copied={};legacy_artifacts={}
    for c in candidates:
        path=Path(c['workspace']);spec,direction=old_candidate(path)
        clone=out/'equivalent-v2-artifacts'/c['candidate_id']
        metadata=A.upgrade_tokenization(spec['tokenization']) if spec['method']=='embedding_mean' else None
        A.write_artifact(clone,spec['method'],direction,
            {**spec['construction'],'purpose':'technical equivalence gate; not a new selected candidate',
             'legacy_artifact':str(path),'legacy_baseline_sha256':A.sha(path/'baseline.json')},
            strength=spec['strength'],layer=spec['layer'],tokenization=metadata)
        copied[c['candidate_id']]=clone
        for p in (path/'baseline.json',path/'direction.npy'):legacy_artifacts[str(p.resolve())]=A.sha(p)
    embedding=next(c for c in candidates if c['family']=='embedding' and c['strength']==1.)
    residual=next(c for c in candidates if c['family']=='residual_g0' and c['strength']==1.)
    for name,c in [('embedding',embedding),('residual',residual)]:
        spec,direction=A.load_artifact(copied[c['candidate_id']],verify_sources=True)
        A.write_artifact(out/(name+'-zero'),spec['method'],direction,
            {**spec['construction'],'purpose':'technical zero-injection control'},strength=0.,
            layer=spec['layer'],tokenization=spec['tokenization'])
    from goals.registry import GOAL_IDS,get_goal
    cases=[{'goal_id':gid,'cell':cell,'prompt':get_goal(gid)[cell]}
           for gid in GOAL_IDS for cell in ('original','teacher','unseen','competing')]
    targets=[{**copy.deepcopy(orig[i%3]),'prompt':case['prompt']} for i,case in enumerate(cases)]
    source_requests=[copy.deepcopy(orig[i%3]) for i in range(len(cases))]
    result={'status':'running','stage':'waiting_for_gpu','started_unix':time.time(),
        'kernel_batch':16,'new_policy_episodes':0,'gpu':args.gpu,
        'source_spec':str(source_spec),'source_spec_sha256':A.sha(source_spec),
        'source_observations':identities,'registered_prompt_cases':cases,
        'legacy_candidates':[c['candidate_id'] for c in candidates],
        'legacy_parity_tolerance':0.,'zero_tolerance':1e-6,'positive_control_tolerance':1e-6}
    A.save(out/'status.json',result);start=time.monotonic()
    try:
        from weightsmith.locks import GpuLock
        import jax,jax.numpy as jnp
        from flax import nnx
        import server as S
        import native as N
        old_server=V.module('_punctuation_legacy_server',old_dir/'server.py')
        all_actions={};legacy_errors={};errors={}
        with GpuLock(args.gpu):
            result['stage']='native_warmup';A.save(out/'status.json',result)
            srv=S.Server();srv.load(None);srv.warmup()
            before,tree=jax.tree.flatten(srv.state)
            old=object.__new__(old_server.Server);old.__dict__=srv.__dict__.copy()
            old._baseline_functions={}
            result['stage']='all_twelve_temporal_candidates';A.save(out/'status.json',result)
            for c in candidates:
                cid=c['candidate_id']
                old.load(c['workspace']);a=V.physical(old.infer_batch(orig))
                srv.load(copied[cid]);b=V.physical(srv.infer_batch(orig))
                legacy_errors[cid]=V.maximum(a,b)
                all_actions['old_'+cid]=a;all_actions['v2_'+cid]=b
                print({'candidate':cid,'max_abs_action_difference':legacy_errors[cid]},flush=True)
                if not np.array_equal(a,b):raise ValueError('Legacy temporal serving changed: '+cid)
            srv.load(None);old.load(None)
            native=V.physical(srv.infer_batch(targets));reference=V.physical(old.infer_batch(targets))
            errors['fixed16_native_wrapper']=V.maximum(native,reference)
            all_actions['native_registered_prompts']=native
            result['stage']='all_twelve_prompt_zero_controls';A.save(out/'status.json',result)
            for name in ('embedding','residual'):
                srv.load(out/(name+'-zero'));actions=V.physical(srv.infer_batch(targets))
                errors[name+'_zero']=V.maximum(native,actions);all_actions[name+'_zero']=actions
            srv.load(copied[embedding['candidate_id']]);nonzero=V.physical(srv.infer_batch(targets))
            errors['embedding_nonzero']=V.maximum(native,nonzero);all_actions['embedding_nonzero']=nonzero
            srv.load(None);restored=V.physical(srv.infer_batch(targets))
            errors['restore_native']=V.maximum(native,restored)
            result['stage']='all_twelve_prompt_embedding_replacement';A.save(out/'status.json',result)
            obs,noise,state=V.pack(srv,source_requests);teacher,teacher_noise,_=V.pack(srv,targets)
            if not np.array_equal(np.asarray(noise),np.asarray(teacher_noise)):raise ValueError('Control changed noise')
            model=nnx.merge(srv.graphdef,srv.state)
            replacement=model.PaliGemma.llm(teacher.tokenized_prompt,method='embed')
            fn=jax.jit(N.make_embedding_replacement_function(srv.graphdef))
            values=np.asarray(fn(srv.state,jax.random.key(0),obs,noise,replacement,teacher.tokenized_prompt_mask))
            outputs=[srv.policy._output_transform({'state':state[i],'actions':values[i]}) for i in range(len(cases))]
            actual=V.physical(outputs);errors['clarified_embedding_replacement']=V.maximum(native,actual)
            all_actions['registered_prompt_embedding_replacement']=actual
            after,after_tree=jax.tree.flatten(srv.state)
            if tree!=after_tree or len(before)!=len(after) or not all(a is b for a,b in zip(before,after,strict=True)):
                raise ValueError('Original checkpoint arrays changed')
            result['checkpoint_state_array_leaves_unchanged']=len(before)
        for key in ('fixed16_native_wrapper','embedding_zero','residual_zero','restore_native','clarified_embedding_replacement'):
            if errors[key]>1e-6:raise ValueError('Native punctuation control failed: '+key)
        if errors['embedding_nonzero']<=1e-6:raise ValueError('Nonzero embedding smoke has no detectable action change')
        np.savez_compressed(out/'actions.npz',**all_actions)
        files=[source_spec,registry_path,Path(source_status['manifest']),old_dir/'artifacts.py',old_dir/'native.py',
            old_dir/'server.py',REPO/'weightsmith/batched_server.py',*sorted(HERE.glob('*.py')),
            out/'actions.npz',*sorted(out.glob('**/baseline.json')),*sorted(out.glob('**/direction.npy')),
            Path(C.OPENPI_DIR)/'src/openpi/models/pi0.py',Path(C.OPENPI_DIR)/'src/openpi/models/gemma.py']
        artifacts={str(p.resolve()):A.sha(p) for p in files};artifacts.update(legacy_artifacts)
        for name,digest in legacy_artifacts.items():
            if A.sha(name)!=digest:raise ValueError('Legacy candidate changed during qualification')
        result.update(status='pass',stage='complete',legacy_exact_parity=True,
            legacy_candidate_max_abs_errors=legacy_errors,max_abs_physical_action_errors=errors,
            legacy_artifacts=legacy_artifacts,artifacts=artifacts,wall_seconds=time.monotonic()-start,
            interpretation='Technical native-serving equivalence and punctuation qualification; zero robot episodes and no efficacy evidence.')
        A.save(out/'passed.json',result)
    except BaseException as exc:
        result.update(status='failed',reason=repr(exc),wall_seconds=time.monotonic()-start);raise
    finally:A.save(out/'status.json',result)
    print({'status':result['status'],'receipt':str(out/'passed.json'),'legacy_exact_parity':result['legacy_exact_parity'],
           'native_errors':errors,'new_policy_episodes':0},flush=True)

if __name__=='__main__':main()
