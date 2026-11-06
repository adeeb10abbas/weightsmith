"""GPU-free validation of exact episode-to-microbatch links and capture bounds."""
from __future__ import annotations
import json
from pathlib import Path
import re
import numpy as np
from .batch_context_v2 import PROTOCOL_ID, pack_requests, read_packet, sha

CALL_BYTES=512<<20


def selected_indices(values,maximum,name,default):
    values=default if values is None else values
    if not isinstance(values,list) or len(set(values))!=len(values) or any(type(x)is not int or not 0<=x<maximum for x in values):raise ValueError('invalid '+name)
    return values


def estimate_bytes(spec,unique_packets,logical_items):
    # Conservative float32 bound for every raw trace field at requested sites;
    # BF16 is retained losslessly and generally needs fewer bytes.
    a=len(spec['action_layers']);v=len(spec['vlm_layers']);s=len(spec['denoise_steps'])
    t=15 if spec['action_tokens']=='mean' else len(spec['action_tokens']);p=len(spec['prompt_tokens'])
    raw_action=16*s*(a*(t*(4096+7*1024+8*1040)+2*1024)+15*(1024+32)+1024)*4
    raw_vlm=16*v*p*(16384+6*2048+8*1040)*4
    raw=(raw_action+raw_vlm+(1<<20))*unique_packets
    reduced_t=1 if spec['action_tokens']=='mean' else t
    compat=logical_items*(s*a*reduced_t*(4096+1024)+v*p*(16384+2048)+s*15*1024+2*15*8)*4
    return int(raw+compat+(8<<20))


def plan_capture(spec,project,expected_identity):
    project=Path(project).resolve();episodes=project/'data/episodes'
    if Path(spec['episodes_dir']).resolve()!=episodes:raise ValueError('episode directory must be this project data/episodes')
    if spec.get('require_parity') is not True or spec.get('parity_tolerance')!=.02:raise ValueError('v2 requires all-policy parity at0.02')
    candidate=spec.get('cand','base');items=spec.get('items')
    if not isinstance(items,list) or not 1<=len(items)<=96:raise ValueError('1..96 requested items required')
    settings={'action_layers':selected_indices(spec.get('action_layers'),18,'action layers',list(range(18))),
              'vlm_layers':selected_indices(spec.get('vlm_layers'),18,'vlm layers',[]),
              'denoise_steps':selected_indices(spec.get('denoise_steps'),10,'denoising steps',list(range(10))),
              'prompt_tokens':selected_indices(spec.get('prompt_tokens'),200,'prompt tokens',list(range(24)))}
    if not settings['action_layers'] or not settings['denoise_steps']:raise ValueError('action layers and steps cannot be empty')
    tokens=spec.get('action_tokens','mean')
    settings['action_tokens']='mean' if tokens=='mean' else selected_indices(tokens,15,'action tokens',list(range(15)))
    if settings['action_tokens']!='mean' and not settings['action_tokens']:raise ValueError('action tokens cannot be empty')
    packets={};chosen=[]
    for number,item in enumerate(items):
        eid=item.get('episode')
        if not isinstance(eid,str) or not re.fullmatch('[a-zA-Z0-9_.-]+',eid) or eid in ('.','..'):raise ValueError('invalid episode identifier')
        directory=episodes/eid
        if directory.is_symlink() or directory.resolve().parent!=episodes:raise ValueError('episode outside project')
        summary=json.loads((directory/'summary.json').read_text())
        if summary.get('cand')!=candidate or summary.get('fingerprint')!=expected_identity['policy_fingerprint']:raise ValueError('same-policy capture is required, including edited candidates')
        with np.load(directory/'obs.npz',allow_pickle=False) as z:
            required={'base_img','wrist_img','joint','gripper','seed','step','actions','prompt',
                      'batch_packet_id','batch_packet_sha256','batch_position','batch_real_count','batch_padded_count'}
            if not required<=set(z.files):raise ValueError('legacy recording lacks authentic batch provenance')
            count=len(z['step'])
            if 'k' in item:
                k=item['k']
                if type(k)is not int or not 0<=k<count:raise ValueError('request index outside recording')
            else:
                step=item.get('step')
                if type(step)is not int or step<0:raise ValueError('nonnegative recorded step required')
                k=max(0,int(np.searchsorted(z['step'],step,side='right')-1))
            packet_id=str(z['batch_packet_id'][k]);packet_sha=str(z['batch_packet_sha256'][k]);position=int(z['batch_position'][k])
            if not re.fullmatch('[0-9a-f]{32}',packet_id):raise ValueError('invalid batch packet ID')
            if packet_id not in packets:
                packet,requests,actions=read_packet(project/'data/batch_packets'/packet_id,packet_sha)
                if packet['identity']!=expected_identity or packet['layout']['runtime_variant']!='fixed16_new_runtime':raise ValueError('packet runtime or policy identity differs')
                packets[packet_id]={'packet':packet,'requests':requests,'recorded_actions':actions,'sha256':packet_sha}
            data=packets[packet_id];packet=data['packet'];layout=packet['layout']
            if data['sha256']!=packet_sha or not 0<=position<layout['real_count'] or int(z['batch_real_count'][k])!=layout['real_count'] or int(z['batch_padded_count'][k])!=16:raise ValueError('batch position/shape/hash mismatch')
            request={'observation/exterior_image_1_left':z['base_img'][k], 'observation/wrist_image_left':z['wrist_img'][k],
                     'observation/joint_position':z['joint'][k],'observation/gripper_position':z['gripper'][k],
                     'prompt':str(z['prompt'][k]),'sampling_seed':int(z['seed'][k])}
            _,hashes=pack_requests([request])
            if hashes[0]!=packet['request_content_sha256'][position]:raise ValueError('episode input differs from actual recorded batch position')
            historical=z['actions'][k].copy()
            if historical.shape!=(15,8) or not np.array_equal(data['recorded_actions'][position].astype(historical.dtype),historical):raise ValueError('episode actions differ from recorded server response at recording precision')
            chosen.append({'i':number,'episode':eid,'k':k,'sim_step':int(z['step'][k]),'seed':request['sampling_seed'],
                           'prompt':request['prompt'],'gripper_state':float(request['observation/gripper_position'][0]),
                           'packet_id':packet_id,'packet_sha256':packet_sha,'batch_position':position,
                           'historical_actions':historical,'source_obs_sha256':sha(directory/'obs.npz')})
    predicted=estimate_bytes(settings,len(packets),len(items))
    if predicted>CALL_BYTES:raise ValueError(f'full raw+compat capture estimate {predicted} exceeds512MiB; request fewer layers/steps/packets')
    project_limit=spec.get('project_capture_bytes',1<<30)
    if type(project_limit)is not int or not CALL_BYTES<=project_limit<=4<<30:raise ValueError('invalid operator capture storage budget')
    acts=project/'data/acts';used=sum(p.stat().st_size for p in acts.rglob('*') if p.is_file()) if acts.exists() else 0
    if used+predicted>project_limit:raise ValueError('project capture persistence budget would be exceeded')
    return {'settings':settings,'packets':packets,'items':chosen,'predicted_bytes':predicted,
            'unique_packets':len(packets),'real_positions_to_check':sum(x['packet']['layout']['real_count'] for x in packets.values()),
            'padded_forward_items_planned':32*len(packets),'logical_items':len(items),'project_existing_bytes':used,
            'project_capture_limit_bytes':project_limit}
