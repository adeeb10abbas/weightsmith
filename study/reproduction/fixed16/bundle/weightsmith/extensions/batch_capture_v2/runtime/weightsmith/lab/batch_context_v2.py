"""CPU-only batch provenance, lossless storage and parity accounting."""
from __future__ import annotations
import hashlib
import json
import math
from pathlib import Path
import re
import numpy as np
PROTOCOL_ID = 'wsp-batch-capture-v2-fixed16'

HASH = re.compile(r'^[0-9a-f]{64}$')
FIELDS = {'observation/exterior_image_1_left': ('base_img', (224,224,3), np.uint8),
          'observation/wrist_image_left': ('wrist_img', (224,224,3), np.uint8),
          'observation/joint_position': ('joint', (7,), None),
          'observation/gripper_position': ('gripper', (1,), None)}
VARIANTS = ('native_power_of_two', 'fixed16_new_runtime')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def canonical(value): return json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()


def checked_identity(identity):
    required={'checkpoint_manifest_sha256','runtime_manifest_sha256','native_server_sha256','policy_fingerprint'}
    if not isinstance(identity,dict) or set(identity)!=required:raise ValueError('exact checkpoint/runtime/server/policy identity required')
    for key in required-{'policy_fingerprint'}:
        if not HASH.fullmatch(str(identity[key])):raise ValueError('invalid identity digest')
    if identity['policy_fingerprint']!='base' and not HASH.fullmatch(str(identity['policy_fingerprint'])):raise ValueError('invalid policy fingerprint')
    return dict(identity)


def batch_layout(real_count,variant):
    if type(real_count)is not int or not 1<=real_count<=16:raise ValueError('v2 prototype supports 1..16 actual requests')
    if variant not in VARIANTS:raise ValueError('explicit runtime variant required')
    padded=16 if variant=='fixed16_new_runtime' else 1<<(real_count-1).bit_length()
    return {'real_count':real_count,'padded_count':padded,'position_to_source':list(range(real_count))+[0]*(padded-real_count),
            'padding_rule':'repeat_first_request_and_its_noise','runtime_variant':variant}


def seeded_noise(seeds,layout,draw,concatenate=np.concatenate):
    """Draw normal(key(seed),(1,H,D)) PER REAL REQUEST; pad existing draws.

    Never replace this with normal(one_key,(B,H,D)): that changes the noise.
    draw/concatenate injection enables a meaningful GPU-free test of this rule.
    """
    if len(seeds)!=layout['real_count']:raise ValueError('one seed per real request required')
    draws=[draw(int(seed)) for seed in seeds]
    if any(x.shape[0]!=1 for x in draws):raise ValueError('draws must have leading singleton batch axis')
    return concatenate([draws[i] for i in layout['position_to_source']],axis=0)


def pack_requests(requests):
    if not isinstance(requests,list) or not requests:raise ValueError('ordered actual request list required')
    result={short:[] for short,_,_ in FIELDS.values()};seeds=[];prompts=[];ids=[]
    for request in requests:
        if set(request)!={*FIELDS,'prompt','sampling_seed'}:raise ValueError('raw model request keys must be exact')
        if not isinstance(request['prompt'],str) or not request['prompt']:raise ValueError('recorded prompt required')
        seed=request['sampling_seed']
        if type(seed)is not int or not 0<=seed<2**32:raise ValueError('explicit nonnegative uint32 sampling seed required')
        h=hashlib.sha256();h.update(canonical({'prompt':request['prompt'],'sampling_seed':seed}))
        for key,(short,shape,dtype) in FIELDS.items():
            value=np.asarray(request[key])
            if value.shape!=shape or value.dtype.hasobject or not np.isfinite(value).all():raise ValueError('invalid raw observation: '+key)
            if dtype is not None and value.dtype!=dtype:raise ValueError('camera bytes must remain uint8')
            if dtype is None and value.dtype.kind!='f':raise ValueError('proprioception must be floating point')
            value=value.copy()
            if result[short] and value.dtype!=result[short][0].dtype:raise ValueError('mixed observation dtypes need an explicit codec')
            result[short].append(value)
            h.update(canonical({'field':key,'dtype':value.dtype.str,'shape':list(value.shape)}));h.update(value.tobytes())
        seeds.append(seed);prompts.append(request['prompt']);ids.append(h.hexdigest())
    result={key:np.stack(values) for key,values in result.items()}
    result.update(seeds=np.asarray(seeds,np.uint32),prompts=np.asarray(prompts))
    return result,ids


def unpack_requests(arrays):
    requests=[]
    for i in range(len(arrays['seeds'])):
        r={key:arrays[short][i].copy() for key,(short,_,_) in FIELDS.items()}
        r.update(prompt=str(arrays['prompts'][i]),sampling_seed=int(arrays['seeds'][i]));requests.append(r)
    pack_requests(requests) # exact dtype/shape/value validation
    return requests


def write_packet(directory,requests,recorded_actions,identity,*,variant='native_power_of_two',
                 provenance='observed_live_microbatch',source_links=None):
    """Called AT the actual server batch boundary, after native outputs exist.

    source_links are optional episode/request references; content hashes are always
    present. An unlinked packet preserves actual batching but does not invent a
    connection to a legacy episode. Legacy observations are never accepted here.
    """
    if provenance!='observed_live_microbatch':raise ValueError('legacy/duplicate inputs are not observed batch provenance')
    identity=checked_identity(identity);layout=batch_layout(len(requests),variant)
    arrays,ids=pack_requests(requests);actions=np.asarray(recorded_actions)
    if actions.shape!=(len(requests),15,8) or actions.dtype.kind!='f' or not np.isfinite(actions).all():raise ValueError('record every actual response at original precision')
    if source_links is not None and (len(source_links)!=len(requests) or any(not isinstance(x,dict) for x in source_links)):raise ValueError('source links must align to actual positions')
    directory=Path(directory)
    if directory.exists():raise ValueError('batch packets are immutable; fresh directory required')
    directory.mkdir(parents=True)
    np.savez_compressed(directory/'requests.npz',**arrays)
    np.savez_compressed(directory/'recorded_actions.npz',actions=actions)
    packet={'schema_version':2,'protocol_id':PROTOCOL_ID,'provenance':provenance,'layout':layout,
            'identity':identity,'request_content_sha256':ids,'source_links':source_links,
            'recorded_positions':list(range(len(requests))),'recorded_action_dtype':actions.dtype.str,
            'padding_outputs_recorded':False,'files':{name:{'sha256':sha(directory/name),'bytes':(directory/name).stat().st_size}
            for name in ('requests.npz','recorded_actions.npz')}}
    (directory/'packet.json').write_bytes(canonical(packet)+b'\n')
    for p in directory.iterdir():p.chmod(0o444)
    return {'path':str(directory.resolve()),'packet_sha256':sha(directory/'packet.json')}


def read_packet(directory,expected_sha256):
    directory=Path(directory)
    if directory.is_symlink() or (directory/'packet.json').is_symlink() or not HASH.fullmatch(str(expected_sha256)) or sha(directory/'packet.json')!=expected_sha256:raise ValueError('packet identity mismatch')
    packet=json.loads((directory/'packet.json').read_text())
    if packet.get('schema_version')!=2 or packet.get('protocol_id')!=PROTOCOL_ID or packet.get('provenance')!='observed_live_microbatch':raise ValueError('authentic v2 batch provenance required; legacy capture cannot be promoted')
    checked_identity(packet['identity'])
    layout=packet['layout']
    if layout!=batch_layout(layout['real_count'],layout['runtime_variant']):raise ValueError('batch padding/order metadata mismatch')
    if set(packet['files'])!={'requests.npz','recorded_actions.npz'}:raise ValueError('unexpected packet files')
    for name,entry in packet['files'].items():
        p=directory/name
        if p.is_symlink() or p.stat().st_size!=entry['bytes'] or sha(p)!=entry['sha256']:raise ValueError('packet file changed')
    with np.load(directory/'requests.npz',allow_pickle=False) as z:arrays={k:z[k] for k in z.files}
    requests=unpack_requests(arrays);_,ids=pack_requests(requests)
    with np.load(directory/'recorded_actions.npz',allow_pickle=False) as z:actions=z['actions']
    if len(requests)!=layout['real_count'] or ids!=packet['request_content_sha256'] or packet['recorded_positions']!=list(range(len(requests))):raise ValueError('request order/identity mismatch')
    if actions.shape!=(len(requests),15,8) or actions.dtype.str!=packet['recorded_action_dtype'] or not np.isfinite(actions).all():raise ValueError('recorded outputs invalid')
    return packet,requests,actions


def compare_actions(left,right,tolerance=.02):
    a,b=np.asarray(left),np.asarray(right)
    if a.shape!=b.shape or a.ndim!=3 or a.shape[1:]!=(15,8) or a.dtype.kind!='f' or b.dtype.kind!='f' or not np.isfinite(a).all() or not np.isfinite(b).all():raise ValueError('complete finite action batches required')
    if not math.isfinite(tolerance) or tolerance<0:raise ValueError('finite nonnegative tolerance required')
    diff=np.abs(a.astype(np.float64)-b.astype(np.float64));rows=[]
    for i,d in enumerate(diff):
        rows.append({'position':i,'max_abs':float(d.max()),'mean_abs':float(d.mean()),
                     'joint_max_abs':float(d[:,:7].max()),'raw_gripper_max_abs':float(d[:,7].max()),
                     'gripper_flips_tau_0p5':np.flatnonzero((a[i,:,7]>.5)!=(b[i,:,7]>.5)).tolist(),
                     'passed':bool(d.max()<=tolerance)})
    return {'checked_count':len(rows),'skipped_count':0,'invalid_count':0,'tolerance':tolerance,
            'passed':all(r['passed'] for r in rows),'max_abs':float(diff.max()),'items':rows,
            'left_dtype':str(a.dtype),'right_dtype':str(b.dtype),
            'exact_at_right_recording_dtype':bool(np.array_equal(a.astype(b.dtype),b))}


def three_way_parity(native,instrumented,recorded,tolerance=.02):
    checks={'native_vs_recorded':compare_actions(native,recorded,tolerance),
            'instrumented_vs_native':compare_actions(instrumented,native,tolerance),
            'instrumented_vs_recorded':compare_actions(instrumented,recorded,tolerance)}
    return {'protocol_id':PROTOCOL_ID,'eligible_capture':all(x['passed'] for x in checks.values()),
            'recorded_real_positions_only':True,'checks':checks,
            'interpretation':'A numerical capture check on these requests; not evidence of closed-loop improvement or retroactive validation.'}


def write_traces(directory,traces):
    """Preserve every supplied batch position/token/feature; no averaging/casting.

    bfloat16 is saved as its uint16 words plus an explicit dtype descriptor,
    avoiding the lossy float16 conversion used by the older capture output.
    """
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=False);arrays={};schema=[]
    def visit(node,path):
        if isinstance(node,dict):
            for key,value in sorted(node.items()):visit(value,path+[str(key)])
            return
        value=np.asarray(node)
        if value.dtype.hasobject or value.dtype.kind not in ('b','i','u','f','V'):raise ValueError('trace leaves must be numeric arrays')
        if value.dtype.kind=='f' and not np.isfinite(value).all():raise ValueError('nonfinite trace')
        name=f'a{len(schema):05d}';logical=str(value.dtype)
        if logical=='bfloat16':
            if not np.isfinite(value.astype(np.float32)).all():raise ValueError('nonfinite bfloat16 trace')
            stored=value.view(np.uint16);encoding='bfloat16_uint16_bits'
        elif value.dtype.kind=='V':raise ValueError('unsupported opaque dtype')
        else:stored=value;encoding='native_numpy'
        arrays[name]=stored
        schema.append({'array':name,'path':path,'shape':list(value.shape),'logical_dtype':logical,
                       'stored_dtype':stored.dtype.str,'encoding':encoding,'bytes':value.nbytes})
    visit(traces,[])
    np.savez_compressed(directory/'traces.npz',**arrays)
    meta={'protocol_id':PROTOCOL_ID,'pooling':'none','precision_conversion':'none','arrays':schema,
          'trace_npz_sha256':sha(directory/'traces.npz'),'raw_array_bytes':sum(e['bytes'] for e in schema)}
    (directory/'trace_schema.json').write_bytes(canonical(meta)+b'\n')
    return meta


def verify_episode_packet_links(row, project, expected_identity, instruction, task, *,
                                expected_threshold=.5, packet_cache=None):
    """Verify EVERY recorded candidate/base request before any outcome attribution.

    The caller pins the actual loaded server/candidate fingerprint. This function
    checks the row, complete chronological sampling stream, and every NPZ link
    against authentic full-microbatch inputs/outputs at client recording precision.
    It imports no model/runtime code and never repairs or rewrites evidence.
    """
    identity=checked_identity(expected_identity);project=Path(project).resolve()
    slot=row.get('slot')
    if type(slot)is not int or slot<0 or row.get('ok') is not True:raise ValueError('completed nonnegative integer slot required')
    terminal_steps=row.get('steps')
    if type(terminal_steps)is not int or not 1<=terminal_steps<=900:raise ValueError('terminal simulator step count required')
    if row.get('fingerprint')!=identity['policy_fingerprint']:raise ValueError('row deployment fingerprint differs')
    if row.get('task_name')!=task or row.get('prompt')!=instruction:raise ValueError('row task or prompt differs')
    if type(row.get('success'))is not bool:raise ValueError('strict boolean simulator outcome required')
    if type(row.get('gripper_threshold'))not in (int,float) or row['gripper_threshold']!=expected_threshold:raise ValueError('row gripper threshold differs')
    path=Path(row.get('obs',''))
    if not path.is_file() or path.is_symlink():raise ValueError('recorded observations missing or symlinked')
    if sha(project/'v2_runtime_binding.json')!=identity['runtime_manifest_sha256']:raise ValueError('runtime binding changed')
    raw={'base_img','wrist_img','joint','gripper','seed','step','prompt','actions'}
    links={'batch_packet_id','batch_packet_sha256','batch_position','batch_real_count','batch_padded_count'}
    cache={} if packet_cache is None else packet_cache;seen=set();packets=[]
    with np.load(path,allow_pickle=False) as z:
        required=raw|links
        if not required<=set(z.files):raise ValueError('complete actual batch links and model inputs required')
        if z['step'].ndim!=1:raise ValueError('recorded step vector required')
        n=len(z['step'])
        if n!=(terminal_steps+14)//15:raise ValueError('recorded request count does not cover terminal simulator step')
        if not 1<=n<=60 or any(z[key].ndim<1 or len(z[key])!=n for key in required):raise ValueError('all recording fields must cover the same1..60 requests')
        shapes={'base_img':(n,224,224,3),'wrist_img':(n,224,224,3),'joint':(n,7),'gripper':(n,1),'actions':(n,15,8)}
        if any(z[key].shape!=shape for key,shape in shapes.items()):raise ValueError('model input/action shapes differ')
        vectors={'seed','step','prompt',*links}
        if any(z[key].shape!=(n,) for key in vectors):raise ValueError('request metadata must be one vector entry per request')
        for key in ('seed','step','batch_position','batch_real_count','batch_padded_count'):
            if not np.issubdtype(z[key].dtype,np.integer):raise ValueError('integer request metadata required')
        if z['actions'].dtype!=np.float32 or not np.isfinite(z['actions']).all():raise ValueError('client action recording must preserve declared float32 precision')
        origin=((slot//16)*64+slot%16)*100
        if not np.array_equal(z['seed'],origin+np.arange(n)):raise ValueError('recorded sampling stream differs from slot/replan formula')
        if not np.array_equal(z['step'],15*np.arange(n)):raise ValueError('recorded requests must retain complete ordered15-step stream')
        if set(map(str,z['prompt']))!={instruction}:raise ValueError('actual recorded prompt differs')
        for k in range(n):
            packet_id=str(z['batch_packet_id'][k]);digest=str(z['batch_packet_sha256'][k]);position=int(z['batch_position'][k])
            if not re.fullmatch('[0-9a-f]{32}',packet_id) or not HASH.fullmatch(digest):raise ValueError('invalid actual packet identifier/hash')
            if (packet_id,position) in seen:raise ValueError('one packet position cannot represent repeated requests')
            seen.add((packet_id,position))
            if packet_id not in cache:
                cache[packet_id]=(digest,read_packet(project/'data/batch_packets'/packet_id,digest))
            cached_hash,(packet,requests,actions)=cache[packet_id]
            if cached_hash!=digest:raise ValueError('conflicting packet hashes')
            layout=packet['layout']
            if (packet['identity']!=identity or layout['runtime_variant']!='fixed16_new_runtime'
                    or int(z['batch_padded_count'][k])!=16 or int(z['batch_real_count'][k])!=layout['real_count']
                    or not 0<=position<layout['real_count']):raise ValueError('packet identity, position, or padding differs')
            request={key:z[short][k] for key,(short,_,_) in FIELDS.items()}
            request.update(prompt=str(z['prompt'][k]),sampling_seed=int(z['seed'][k]))
            _,ids=pack_requests([request])
            if ids[0]!=packet['request_content_sha256'][position]:raise ValueError('recorded request does not match actual packet position')
            if not np.array_equal(actions[position].astype(np.float32),z['actions'][k]):raise ValueError('recorded action differs from native response at client precision')
            packets.append({'k':k,'packet_id':packet_id,'packet_sha256':digest,'position':position,'request_content_sha256':ids[0]})
    return {'protocol_id':PROTOCOL_ID,'status':'passed','slot':slot,'policy_fingerprint':identity['policy_fingerprint'],
            'runtime_binding_sha256':identity['runtime_manifest_sha256'],'obs_sha256':sha(path),
            'requests_checked':n,'all_recorded_requests_checked':True,'packets':packets}
