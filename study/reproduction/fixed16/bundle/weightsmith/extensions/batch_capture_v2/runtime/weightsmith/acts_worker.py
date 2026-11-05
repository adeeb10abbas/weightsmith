"""V2 actual-batch activation capture; no legacy replay or cross-policy skips."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
import time
import types
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weightsmith import config as C
from weightsmith.integrity import workspace_fingerprint
from weightsmith.replay import ParityCheck
from weightsmith.lab.batch_context_v2 import PROTOCOL_ID, compare_actions, sha, three_way_parity, write_traces
from weightsmith.lab.batch_runtime_v2 import project_identity, postprocess
from weightsmith.lab.capture_plan_v2 import CALL_BYTES, plan_capture
from weightsmith.lab.batch_probe_v2 import make_batch_probe


def capture(spec,output,workspace='',server=None):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    fingerprint=workspace_fingerprint(workspace or None);identity=project_identity(fingerprint)
    index={'protocol_id':PROTOCOL_ID,'cand':spec.get('cand','base'),'policy_fingerprint':fingerprint,
           'items':[],'packets':{},'padded_forward_items_reserved':0,
           'native_forward_calls_started':0,'native_forward_calls_completed':0,
           'instrumented_forward_calls_started':0,'instrumented_forward_calls_completed':0,
           'logical_items_attempted':len(spec.get('items',[])),
           'capture_validation':{'protocol_id':PROTOCOL_ID,'status':'failed','all_real_positions_checked':False,
                                 'parity_tolerance':.02,'zero_skipped':True}}
    plan=None
    def checkpoint():
        index['padded_forward_items_started']=16*(index['native_forward_calls_started']+index['instrumented_forward_calls_started'])
        index['padded_forward_items_completed']=16*(index['native_forward_calls_completed']+index['instrumented_forward_calls_completed'])
        index['padded_forward_items_attempted']=index['padded_forward_items_started']  # explicit started-call alias
        index['elapsed_seconds']=time.monotonic()-started
        (output/'index.json').write_text(json.dumps(index,indent=2,allow_nan=False)+'\n')
    try:
        plan=plan_capture(spec,C.PROJECT_ROOT,identity);settings=plan['settings']
        index.update(action_layers=settings['action_layers'],vlm_layers=settings['vlm_layers'],
                     denoise_steps=settings['denoise_steps'],denoise_t=[round(1-.1*x,2) for x in settings['denoise_steps']],
                     action_tokens=settings['action_tokens'],prompt_token_slots=settings['prompt_tokens'],
                     logical_items=plan['logical_items'],unique_packets=plan['unique_packets'],
                     predicted_persist_bytes=plan['predicted_bytes'],padded_forward_items_planned=plan['padded_forward_items_planned'])
        checkpoint()
        # All expensive imports and runtime initialization follow CPU provenance
        # and size validation. Outer owsp holds the GPU lock/private process guard.
        import jax
        from openpi.shared import nnx_utils
        from weightsmith.batched_server import Server
        server=Server() if server is None else server
        server.record_actual_batches=False
        server.load(workspace or None)
        if server.report['fingerprint']!=fingerprint:raise ValueError('loaded policy identity differs')
        model=server.nnx.merge(server.graphdef,server.state)
        probe=nnx_utils.module_jit(types.MethodType(make_batch_probe(settings),model))
        prepared={};all_real_passed=True;real_checked=0
        for packet_id,data in plan['packets'].items():
            # Reserve both native and instrumented padded computations before
            # dispatch, retaining cost even if a device exception follows.
            index['padded_forward_items_reserved']+=32;checkpoint()
            index['native_forward_calls_started']+=1;checkpoint()
            native_outputs,context=server.infer_context(data['requests'])
            index['native_forward_calls_completed']+=1;checkpoint()
            native=np.stack([x['actions'] for x in native_outputs])
            index['instrumented_forward_calls_started']+=1;checkpoint()
            raw,trace=probe(context['observation'],context['noise'])
            jax.tree.map(lambda value:value.block_until_ready(),(raw,trace))
            index['instrumented_forward_calls_completed']+=1;checkpoint()
            instrumented=np.stack([x['actions'] for x in postprocess(server,raw,context)])
            parity=three_way_parity(native,instrumented,data['recorded_actions'],.02)
            all_real_passed=all_real_passed and parity['eligible_capture']
            real_checked+=len(data['requests'])
            # All requested-layer/step/token raw traces, every actual and padded
            # position, original floating-point representation. No pooling.
            host_trace=jax.tree.map(np.asarray,trace)
            raw_schema=write_traces(output/'raw_batches'/packet_id,host_trace)
            action_file=output/'raw_batches'/packet_id/'actions.npz'
            np.savez_compressed(action_file,native=native,instrumented=instrumented,
                                server_recorded=data['recorded_actions'])
            index['packets'][packet_id]={'packet_sha256':data['sha256'],'layout':data['packet']['layout'],
                'identity':data['packet']['identity'],'parity':parity,'raw_trace_schema_sha256':sha(output/'raw_batches'/packet_id/'trace_schema.json'),
                'raw_trace_array_sha256':raw_schema['trace_npz_sha256'],'action_arrays_sha256':sha(action_file),
                'raw_trace_shape_scope':'all16 batch positions including mapped padding; requested sites only',
                'raw_array_bytes':raw_schema['raw_array_bytes']}
            prepared[packet_id]={'native':native,'instrumented':instrumented,'trace':host_trace}
            checkpoint()
            actual=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
            if actual>CALL_BYTES:raise ValueError('actual capture bytes exceed512MiB')
        store={};selected_parity=ParityCheck(.02)
        def add(name,array):store.setdefault(name,[]).append(np.asarray(array))
        for item in plan['items']:
            data=prepared[item['packet_id']];position=item['batch_position'];tr=data['trace'];act=tr['action'];vlm=tr['vlm']
            mh=np.asarray(act['mlp_down_input'][:,:,position],np.float32)
            ro=np.asarray(act['residual_output'][:,:,position],np.float32)
            if settings['action_tokens']=='mean':mh=mh.mean(axis=-2,keepdims=True);ro=ro.mean(axis=-2,keepdims=True)
            add('action.mlp_hidden',mh);add('action.resid_out',ro)
            add('action.head_input',np.asarray(act['head_input'][:,position],np.float32))
            if settings['vlm_layers']:
                add('vlm.mlp_hidden',np.asarray(vlm['mlp_down_input'][:,position],np.float32))
                add('vlm.resid_out',np.asarray(vlm['residual_output'][:,position],np.float32))
            actions=data['instrumented'][position];native=data['native'][position];recorded=item['historical_actions']
            parity_item=selected_parity.add(actions,recorded)
            add('actions',actions);add('native_actions',native);add('recorded_actions',recorded)
            row={k:v for k,v in item.items() if k!='historical_actions'};row['parity']=parity_item
            row['native_vs_episode_recorded']=compare_actions(native[None],recorded[None],.02)
            row['instrumented_vs_episode_recorded']=compare_actions(actions[None],recorded[None],.02)
            all_real_passed=all_real_passed and row['native_vs_episode_recorded']['passed'] and parity_item['passed']
            tids=np.asarray(tr['token_ids'])[position];mask=np.asarray(tr['token_mask'])[position]
            row['prompt_token_ids_at_slots']=[int(tids[j]) for j in settings['prompt_tokens'] if j<len(tids)]
            row['prompt_token_valid_at_slots']=[bool(mask[j]) for j in settings['prompt_tokens'] if j<len(mask)]
            index['items'].append(row)
        np.savez_compressed(output/'acts.npz',**{key:np.stack(values) for key,values in store.items()})
        index['shapes']={key:list(np.stack(values).shape) for key,values in store.items()}
        index['compatibility_precision']='float32 exact widening of model BF16/FP32 feature values; pooled only when action_tokens=mean; raw unpooled traces retained separately'
        index['parity']=selected_parity.report();index['base_parity_max_abs_action_error']=index['parity']['max_abs_action_error']
        index['prompt_token_pieces_item0']='not decoded in v2; exact token IDs retained'
        if workspace_fingerprint(workspace or None)!=fingerprint:raise ValueError('capture policy changed')
        if project_identity(fingerprint)!=identity:raise ValueError('capture runtime binding changed')
        actual=sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        if actual>CALL_BYTES or plan['project_existing_bytes']+actual>plan['project_capture_limit_bytes']:raise ValueError('actual capture persistence cap exceeded')
        index['actual_persist_bytes_before_final_index']=actual
        index['capture_validation'].update(status='passed' if all_real_passed and real_checked==plan['real_positions_to_check'] and selected_parity.require_all(plan['logical_items']) else 'failed',
              all_real_positions_checked=real_checked==plan['real_positions_to_check'],real_positions_checked=real_checked,
              unique_packets_checked=len(prepared),historical_packet_precision='original server output precision',
              episode_precision='recorded client precision; raw and cast-aware errors retained')
        checkpoint()
        if index['capture_validation']['status']!='passed':raise ValueError('v2 whole-batch native/instrumented/historical parity failed')
        return index
    except BaseException as error:
        index['error']=f'{type(error).__name__}: {error}';index['capture_validation']['status']='failed';checkpoint();raise


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);parser.add_argument('--workspace',default='');args=parser.parse_args()
    spec=json.loads(sys.stdin.read())
    report=capture(spec,args.output,args.workspace)
    print(json.dumps({'id':Path(args.output).name,'items':len(report['items']),'capture_validation':report['capture_validation'],
                      'padded_forward_items_reserved':report['padded_forward_items_reserved'],
                      'padded_forward_items_started':report['padded_forward_items_started'],
                      'padded_forward_items_completed':report['padded_forward_items_completed'],'sec':report['elapsed_seconds']}))


if __name__=='__main__':main()
