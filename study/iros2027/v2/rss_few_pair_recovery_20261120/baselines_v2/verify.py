"""Bounded native action parity and positive-control gate; no robot rollouts.

Run in the OpenPI environment on a free GPU. Three recorded source observations
are replayed with fixed16 padding. A zero-strength baseline must match the native
server, and exact clarified embeddings must reproduce the native clarified
actions. These checks validate inference plumbing, not behavioral efficacy.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import os
from pathlib import Path
import sys
import time
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0,str(REPO));sys.path.insert(0,str(HERE))
import artifacts as A


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def requests(source_spec):
    spec=A.read(source_spec)
    items=spec['items'][:3]
    if len(items)!=3 or len({(it['obs_path'],it['decision_index']) for it in items})!=3:
        raise ValueError('Gate requires three distinct recorded source observations')
    rows=[];out=[]
    for item in items:
        k=item['decision_index'];path=Path(item['obs_path']).resolve()
        with np.load(path,allow_pickle=False) as data:
            seed=int(data['seed'][k])
            if seed<0:raise ValueError('Unrecorded sampling seed')
            request={'observation/exterior_image_1_left':data['base_img'][k].copy(),
                     'observation/wrist_image_left':data['wrist_img'][k].copy(),
                     'observation/joint_position':data['joint'][k].copy(),
                     'observation/gripper_position':data['gripper'][k].copy(),
                     'prompt':item.get('original_prompt',spec['original_prompt']),
                     'sampling_seed':seed}
        rows.append({'obs_path':str(path),'obs_sha256':A.sha(path),'decision_index':k,'seed':seed,
                     'original_prompt':request['prompt'],
                     'clarified_prompt':item.get('teacher_prompt',spec['teacher_prompt'])})
        out.append(request)
    return out,rows


def physical(outputs):
    return np.stack([np.asarray(o['actions'],dtype=np.float32) for o in outputs])


def maximum(left,right):
    if left.shape!=right.shape or not np.isfinite(left).all() or not np.isfinite(right).all():
        raise ValueError('Invalid finite action comparison')
    return float(np.max(np.abs(left-right)))


def pack(server,reqs):
    import jax,jax.numpy as jnp
    padded=[copy.deepcopy(r) for r in reqs]
    padded.extend(copy.deepcopy(reqs[0]) for _ in range(16-len(reqs)))
    inputs=[];noise=[]
    for r in padded:
        seed=r.pop('sampling_seed')
        inputs.append(server.policy._input_transform(r))
        noise.append(jax.random.normal(jax.random.key(seed),(1,server.H,server.D)))
    stacked=jax.tree.map(lambda *xs:jnp.stack([jnp.asarray(x) for x in xs]),*inputs)
    return server._model.Observation.from_dict(stacked),jnp.concatenate(noise),np.asarray(stacked['state'])


def gate(args):
    from weightsmith import config as C
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu),XLA_PYTHON_CLIENT_PREALLOCATE='true',
                      XLA_PYTHON_CLIENT_MEM_FRACTION=C.SERVER_MEM_FRACTION,
                      JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE),JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS='1')
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    source=Path(args.source_spec).resolve()
    orig,identities=requests(source)
    clarified=[{**r,'prompt':identity['clarified_prompt']} for r,identity in zip(orig,identities,strict=True)]
    e_spec,e_direction=A.load_artifact(args.embedding,verify_sources=True)
    r_spec,r_direction=A.load_artifact(args.residual,verify_sources=True)
    if e_spec['method']!='embedding_mean' or r_spec['method']!='residual_cells':
        raise ValueError('Gate requires one baseline of each type')
    for name,spec,direction in [('embedding',e_spec,e_direction),('residual',r_spec,r_direction)]:
        A.write_artifact(out/(name+'-zero'),spec['method'],direction,
                         {**spec['construction'],'purpose':'technical zero-injection gate only'},
                         strength=0.,layer=spec['layer'],tokenization=spec['tokenization'])
    result={'status':'running','stage':'waiting_for_gpu','started_unix':time.time(),'source_spec':str(source),'source_spec_sha256':A.sha(source),
            'source_observations':identities,'kernel_batch':16,'new_policy_episodes':0,
            'zero_tolerance':1e-6,'positive_control_tolerance':1e-6,
            'nonzero_smoke_minimum_action_change':1e-6,'gpu':args.gpu}
    A.save(out/'status.json',result)
    start=time.monotonic()
    try:
        from weightsmith.locks import GpuLock
        import jax,jax.numpy as jnp
        from flax import nnx
        import server as S
        import native as N
        old_path=REPO/'study/iros2027/v2/rss_repair_comparison/runtime/batched_server.py'
        old_module=module('_fewpair_original_fixed16_server',old_path)
        all_actions={};errors={}
        with GpuLock(args.gpu):
            result['stage']='native_warmup';A.save(out/'status.json',result)
            srv=S.Server();srv.load(None);srv.warmup()
            before,tree=jax.tree.flatten(srv.state)
            # Independent old wrapper around the exact same unedited state and
            # native compiled sampler, avoiding a second loaded checkpoint.
            reference=object.__new__(old_module.Server);reference.__dict__=srv.__dict__.copy()
            all_actions['old_native_original']=physical(reference.infer_batch(orig))
            all_actions['native_original']=physical(srv.infer_batch(orig))
            errors['fixed16_native_wrapper']=maximum(all_actions['native_original'],all_actions['old_native_original'])
            all_actions['native_clarified']=physical(srv.infer_batch(clarified))
            result['stage']='zero_and_nonzero_hooks';A.save(out/'status.json',result)
            for name,path in [('embedding_zero',out/'embedding-zero'),('residual_zero',out/'residual-zero'),
                              ('embedding_nonzero',Path(args.embedding)),('residual_nonzero',Path(args.residual))]:
                srv.load(path)
                all_actions[name]=physical(srv.infer_batch(orig))
                errors[name]=maximum(all_actions[name],all_actions['native_original'])
            srv.load(None)
            all_actions['restored_native']=physical(srv.infer_batch(orig))
            errors['restore_native']=maximum(all_actions['restored_native'],all_actions['native_original'])
            result['stage']='clarified_embedding_positive_control';A.save(out/'status.json',result)
            observation,noise,state=pack(srv,orig)
            teacher_observation,teacher_noise,_=pack(srv,clarified)
            if not np.array_equal(np.asarray(noise),np.asarray(teacher_noise)):
                raise ValueError('Positive control changed source noise')
            model=nnx.merge(srv.graphdef,srv.state)
            replacement=model.PaliGemma.llm(teacher_observation.tokenized_prompt,method='embed')
            fn=jax.jit(N.make_embedding_replacement_function(srv.graphdef))
            actions=np.asarray(fn(srv.state,jax.random.key(0),observation,noise,replacement,teacher_observation.tokenized_prompt_mask))
            outputs=[srv.policy._output_transform({'state':state[i],'actions':actions[i]}) for i in range(len(orig))]
            all_actions['clarified_embedding_replacement']=physical(outputs)
            errors['clarified_embedding_replacement']=maximum(all_actions['clarified_embedding_replacement'],all_actions['native_clarified'])
            errors['native_clarification_action_change']=maximum(all_actions['native_clarified'],all_actions['native_original'])
            after,after_tree=jax.tree.flatten(srv.state)
            state_unchanged=tree==after_tree and len(before)==len(after) and all(a is b for a,b in zip(before,after,strict=True))
            if not state_unchanged:
                raise ValueError('Original immutable server-state array identities changed')
            result['checkpoint_state_array_leaves_unchanged']=len(before)
        np.savez_compressed(out/'actions.npz',**all_actions)
        result['max_abs_physical_action_errors']=errors
        for key in ('fixed16_native_wrapper','embedding_zero','residual_zero','restore_native'):
            if errors[key]>result['zero_tolerance']:
                raise ValueError(f'Native parity failed for {key}: {errors[key]}')
        if errors['clarified_embedding_replacement']>result['positive_control_tolerance']:
            raise ValueError('Exact clarified-prefix embedding substitution does not reproduce teacher actions')
        for key in ('embedding_nonzero','residual_nonzero','native_clarification_action_change'):
            if errors[key]<=result['nonzero_smoke_minimum_action_change']:
                raise ValueError('No detectable action effect in technical smoke check: '+key)
        paths=[source,Path(args.embedding)/'baseline.json',Path(args.embedding)/'direction.npy',
               Path(args.residual)/'baseline.json',Path(args.residual)/'direction.npy',out/'actions.npz',
               old_path,REPO/'weightsmith/batched_server.py',*sorted(HERE.glob('*.py')),
               Path(C.OPENPI_DIR)/'src/openpi/models/pi0.py',Path(C.OPENPI_DIR)/'src/openpi/models/gemma.py']
        result.update(status='pass',stage='complete',wall_seconds=time.monotonic()-start,
                      artifacts={str(p.resolve()):A.sha(p) for p in paths},
                      interpretation='Technical action/parity qualification only; no rollout outcomes or evidence of baseline efficacy.')
        A.save(out/'passed.json',result)
    except BaseException as error:
        result.update(status='failed',error=repr(error),wall_seconds=time.monotonic()-start)
        raise
    finally:
        A.save(out/'status.json',result)
    print({'status':result['status'],'output':str(out),'errors':result['max_abs_physical_action_errors'],
           'wall_seconds':result['wall_seconds'],'new_policy_episodes':0},flush=True)
    return result


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source-spec',required=True);ap.add_argument('--embedding',required=True)
    ap.add_argument('--residual',required=True);ap.add_argument('--output',required=True)
    ap.add_argument('--gpu',type=int,choices=(0,1),required=True)
    gate(ap.parse_args())


if __name__=='__main__':main()
