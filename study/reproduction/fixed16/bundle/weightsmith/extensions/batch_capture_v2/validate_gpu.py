#!/usr/bin/env python3
"""Operator-only offline validation of the NEW fixed16 runtime and exact worker.

No simulator episodes, no legacy provenance promotion, no automatic GPU dispatch.
Run only under the existing isolation guard, explicitly passing --execute-gpu.
"""
from __future__ import annotations
import argparse
import json
import os
import re
from pathlib import Path
import sys
import time
import traceback

RUNTIME = Path(__file__).resolve().parent/'runtime'
COUNTS = (1, 2, 4, 8, 16)


def write_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def checked_paths(project, source, isolation):
    """Keep every validation version and source read under the owned deployment."""
    if not isolation:raise ValueError('WSP_ISOLATED_ROOT is required')
    root=Path(isolation).resolve(strict=True);project=Path(project);source=Path(source)
    if not project.is_absolute() or not source.is_absolute():raise ValueError('absolute owned project paths required')
    for path in (project,source):
        if path.is_symlink() or path.parent.resolve()!=root/'projects':raise ValueError('project must be directly under owned deployment/projects')
    if not re.fullmatch(r'rq123-v2-validation-v[1-9][0-9]*',project.name):raise ValueError('explicit versioned validation-only project required')
    if source.name!='rq123-development-v1' or not source.is_dir():raise ValueError('only original owned development reference inputs allowed')
    for path in (project/'study',project/'data',source/'data',source/'data/episodes'):
        if path.is_symlink() or (path.exists() and root not in path.resolve().parents):raise ValueError('symlinked/outside artifact directory rejected')
    return project.resolve(),source.resolve(),root


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute-gpu',action='store_true')
    parser.add_argument('--source-project',required=True,type=Path)
    parser.add_argument('--project',required=True,type=Path)
    args=parser.parse_args(argv)
    if not args.execute_gpu:parser.error('offline GPU validation requires --execute-gpu')
    if os.environ.get('WSP_MANAGED_PROCESS_GROUP')!='1':parser.error('existing isolated process-group guard required')
    try:project,source,isolated=checked_paths(args.project,args.source_project,os.environ.get('WSP_ISOLATED_ROOT'))
    except (ValueError,OSError) as error:parser.error(str(error))
    if isolated!=RUNTIME.parents[2].resolve():parser.error('validation fork must be inside the same owned isolation root')
    if not (project/'v2_runtime_binding.json').is_file():parser.error('immutable runtime binding must exist BEFORE validation')
    if (project/'study/capture_validation.json').exists():parser.error('validation already attempted; preserve its report and use an explicit new version')
    if (project/'data/episodes').exists() or (project/'data/batch_packets').exists():parser.error('fresh diagnostic evidence directories required')
    os.environ['WSP_ROOT']=str(project)
    sys.path.insert(0,str(RUNTIME))
    sys.path.insert(0,str(RUNTIME/'scripts'))
    from verify_runtime import verify_binding
    source_checkpoint_verification=verify_binding(project/'v2_runtime_binding.json',verify_checkpoint=True)
    import numpy as np
    from weightsmith import config as C
    from weightsmith.lab.batch_context_v2 import PROTOCOL_ID, pack_requests, unpack_requests, sha, compare_actions
    from weightsmith.lab.batch_runtime_v2 import project_identity
    identity=project_identity('base')  # CPU binding checks before any JAX import.
    os.environ['CUDA_VISIBLE_DEVICES']=str(C.ACTS_GPU)
    os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
    from weightsmith.locking import GpuLock
    report_path=project/'study/capture_validation.json';started=time.monotonic()
    report={'status':'running','protocol_id':PROTOCOL_ID,'runtime_binding_sha256':identity['runtime_manifest_sha256'],
            'source_checkpoint_verification':source_checkpoint_verification,
            'purpose':'offline validation only; no new simulator episodes; not editor evidence',
            'source_project':str(source),'selection_rule':'first request from first16 numerically sorted original-base slots; actual heterogeneous image/proprioception inputs and recorded seeds',
            'real_batch_counts':list(COUNTS),'padded_batch_count':16,'native_forward_calls_attempted':0,'native_forward_calls_completed':0,
            'worker_padded_forward_items_attempted':0,'batches':[],'source_observations':[]}
    write_json(report_path,report)
    try:
        candidates=[]
        for summary in (source/'data/episodes').glob('*/summary.json'):
            if summary.is_symlink() or summary.parent.is_symlink() or (summary.parent/'obs.npz').is_symlink():raise ValueError('symlinked source evidence rejected')
            row=json.loads(summary.read_text())
            if row.get('cand')=='base' and row.get('fingerprint')=='base':
                slot=int(str(row.get('slot','s-1')).removeprefix('s'))
                if 20032<=slot<=20063:candidates.append((slot,summary.parent))
        candidates.sort(key=lambda x:(x[0],x[1].name))
        chosen=[];slots=set()
        for slot,path in candidates:
            if slot not in slots:chosen.append((slot,path));slots.add(slot)
            if len(chosen)==16:break
        if len(chosen)!=16:raise ValueError('need16 distinct original-base slots for heterogeneous input validation')
        requests=[]
        for slot,path in chosen:
            with np.load(path/'obs.npz',allow_pickle=False) as z:
                request={'observation/exterior_image_1_left':z['base_img'][0].copy(),
                         'observation/wrist_image_left':z['wrist_img'][0].copy(),
                         'observation/joint_position':z['joint'][0].copy(),
                         'observation/gripper_position':z['gripper'][0].copy(),
                         'prompt':str(z['prompt'][0]),'sampling_seed':int(z['seed'][0])}
            requests.append(request);report['source_observations'].append({'episode':path.name,'slot':slot,'k':0,'obs_sha256':sha(path/'obs.npz')})
        pack_requests(requests)
        if len({r['sampling_seed'] for r in requests})!=16:raise ValueError('validation requests must have distinct recorded seeds')
        from weightsmith.batched_server import Server
        from weightsmith.acts_worker import capture
        with GpuLock(C.ACTS_GPU):
            server=Server();server.load(None);first=[];items=[]
            for count in COUNTS:
                # The raw request snapshot is taken before live transforms.
                raw_requests=unpack_requests(pack_requests(requests[:count])[0])
                server.record_actual_batches=True;report['native_forward_calls_attempted']+=1;write_json(report_path,report)
                outputs=server.infer_batch(raw_requests)
                report['native_forward_calls_completed']+=1
                native=np.stack([out['actions'] for out in outputs]);first.append((count,raw_requests,native))
                server.record_actual_batches=False;report['native_forward_calls_attempted']+=1;write_json(report_path,report)
                repeated=np.stack([out['actions'] for out in server.infer_batch(raw_requests)])
                report['native_forward_calls_completed']+=1
                repeat=compare_actions(native,repeated,0.0)
                report['batches'].append({'real_count':count,'repeat':repeat,'packet_id':outputs[0]['_wsp_batch_v2']['packet_id']})
                for position,(request,out) in enumerate(zip(raw_requests,outputs)):
                    packet=out['_wsp_batch_v2'];eid=f'base.base.validation-b{count:02d}-p{position:02d}'
                    directory=project/'data/episodes'/eid;directory.mkdir(parents=True)
                    # These are offline request links, never simulator outcomes.
                    write_json(directory/'summary.json',{'eid':eid,'cand':'base','fingerprint':'base',
                        'diagnostic_only':True,'no_simulator_episode':True,'source_episode':chosen[position][1].name})
                    np.savez_compressed(directory/'obs.npz',base_img=request['observation/exterior_image_1_left'][None],
                        wrist_img=request['observation/wrist_image_left'][None],joint=request['observation/joint_position'][None],
                        gripper=request['observation/gripper_position'][None],seed=np.asarray([request['sampling_seed']],np.int64),
                        step=np.asarray([0],np.int64),prompt=np.asarray([request['prompt']]),actions=np.asarray(out['actions'],np.float32)[None],
                        batch_packet_id=np.asarray([packet['packet_id']]),batch_packet_sha256=np.asarray([packet['packet_sha256']]),
                        batch_position=np.asarray([position]),batch_real_count=np.asarray([count]),batch_padded_count=np.asarray([16]))
                    items.append({'episode':eid,'k':0})
                write_json(report_path,report)
            identity_report=server.load(str(C.BASE_WS))
            if identity_report.get('changed_elements')!=0:raise ValueError('identity workspace unexpectedly edits parameters')
            # Validate true no-edit load against native outputs, same explicit input seeds.
            for index,(count,raw_requests,native) in enumerate(first):
                report['native_forward_calls_attempted']+=1;write_json(report_path,report)
                after=np.stack([out['actions'] for out in server.infer_batch(raw_requests)])
                report['native_forward_calls_completed']+=1
                report['batches'][index]['identity_load']=compare_actions(native,after,0.0)
            spec={'cand':'base','episodes_dir':str(project/'data/episodes'),'require_parity':True,'parity_tolerance':.02,
                  'items':items,'action_layers':[16,17],'vlm_layers':[],'denoise_steps':[9],'action_tokens':'mean',
                  'prompt_tokens':list(range(24)),'project_capture_bytes':1<<30}
            write_json(project/'study/validation_capture_spec.json',spec)
            worker=capture(spec,project/'data/acts/offline-validation',server=server)
            report['worker_padded_forward_items_attempted']=worker['padded_forward_items_started']
            report['worker_forward_accounting']={key:worker[key] for key in ('padded_forward_items_reserved','padded_forward_items_started','padded_forward_items_completed','native_forward_calls_started','native_forward_calls_completed','instrumented_forward_calls_started','instrumented_forward_calls_completed')}
            report['capture_validation']=worker['capture_validation']
            report['capture_index_sha256']=sha(project/'data/acts/offline-validation/index.json')
            report['capture_arrays_sha256']=sha(project/'data/acts/offline-validation/acts.npz')
            report['identity_load_report']=identity_report
        if not all(row['repeat']['passed'] and row['identity_load']['passed'] for row in report['batches']):raise ValueError('native repeat or identity-load equality failed')
        if report['capture_validation']['status']!='passed':raise ValueError('exact v2 worker parity failed')
        if project_identity('base')!=identity:raise ValueError('runtime binding changed during validation')
        report['status']='passed'
    except BaseException as error:
        report['status']='failed';report['error']=f'{type(error).__name__}: {error}'
        report['traceback']=traceback.format_exc()
        p=project/'data/acts/offline-validation/index.json'
        if p.exists():
            index=json.loads(p.read_text());report['worker_padded_forward_items_attempted']=index.get('padded_forward_items_started',0)
            report['worker_forward_accounting']={key:index.get(key,0) for key in ('padded_forward_items_reserved','padded_forward_items_started','padded_forward_items_completed','native_forward_calls_started','native_forward_calls_completed','instrumented_forward_calls_started','instrumented_forward_calls_completed')}
        raise
    finally:
        report['elapsed_seconds']=time.monotonic()-started
        report['native_forward_calls_started']=report['native_forward_calls_attempted']
        report['total_padded_forward_items_attempted']=16*report['native_forward_calls_started']+report['worker_padded_forward_items_attempted']
        report['total_padded_forward_items_completed']=16*report['native_forward_calls_completed']+report.get('worker_forward_accounting',{}).get('padded_forward_items_completed',0)
        write_json(report_path,report)
    print(json.dumps({'status':report['status'],'runtime_binding_sha256':report['runtime_binding_sha256'],'report':str(report_path)}))


if __name__=='__main__':main()
