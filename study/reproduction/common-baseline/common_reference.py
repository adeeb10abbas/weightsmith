#!/usr/bin/env python3
"""Separate fixed16 common-base transfer: exact files only, no remote writes/GPU.

Fetch is fixed to one completed original-host baseline. Verify/extract is local.
No import rewrites a runtime identity, recording, success label, or checkpoint.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path,PurePosixPath
import re
import shlex
import subprocess
import tarfile

ROOT='/tmp/weightsmith-codex-20261105/projects/rq123-v2-development-v1'
PREFIX='common-fixed16-baseline-v1'
MANIFEST='study/reference_base_manifest.json'
MANIFEST_SHA='579543173d6fa7f119f6ba6d39e6abb725d403b8e258b64968c12602c695771f'
NATIVE_SHA='c982b471408232b1a9d5361b320f90932c83e92cefce15d433df65783254ffa0'
VALIDATION_SHA='49882deb43e51fec6c4a180ce48073be63c7f64ffafc3eaa7ac4b059c58b5fba'
SOURCE_SHA='017e7ab14bdfd330bb11b7addb23f6a04fa4a07374442e41aa253617e4def754'
CODEC_SHA='b526f941ec312f3209224481cafb239466113f7d51d9eb3d2247687c8430e510'
SLOTS=list(range(30016,30048))
EXTRAS={MANIFEST,'config.json','study/status.json','study/identity_report.json','study/capture_validation.json'}


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def allowed(name):
    p=PurePosixPath(name)
    if p.is_absolute() or any(x in ('','.','..') for x in name.split('/')):return False
    return (name in EXTRAS|{'data/episodes/index.jsonl','study/reference_base_rows.json','v2_runtime_binding.json'}
            or re.fullmatch(r'data/episodes/base\.base\.s(300(?:1[6-9]|[23][0-9]|4[0-7]))/(obs\.npz|summary\.json)',name) is not None
            or re.fullmatch(r'data/batch_packets/[0-9a-f]{32}/(packet\.json|requests\.npz|recorded_actions\.npz)',name) is not None)


REMOTE=r'''
import datetime,hashlib,io,json,pathlib,re,sys,tarfile
root=pathlib.Path('/tmp/weightsmith-codex-20261105/projects/rq123-v2-development-v1')
expected='579543173d6fa7f119f6ba6d39e6abb725d403b8e258b64968c12602c695771f'
prefix='common-fixed16-baseline-v1';manifest_name='study/reference_base_manifest.json'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as stream:
  for block in iter(lambda:stream.read(1048576),b''):h.update(block)
 return h.hexdigest()
def regular(name):
 p=root/name
 if pathlib.PurePosixPath(name).is_absolute() or any(x in ('','.','..') for x in name.split('/')):raise ValueError('unsafe member')
 for node in (p,*p.parents):
  if node.is_symlink():raise ValueError('symlink rejected')
  if node==root:break
 if not p.is_file() or root.resolve() not in p.resolve().parents:raise ValueError('regular owned file required')
 return p
manifest_path=regular(manifest_name)
if digest(manifest_path)!=expected:raise ValueError('baseline manifest changed')
manifest=json.loads(manifest_path.read_text())
if manifest.get('protocol')!='wsp-manipulation-batch-aware-v2' or manifest.get('slots')!=list(range(30016,30048)) or manifest.get('threshold')!=.5:raise ValueError('wrong baseline')
status=json.loads(regular('study/status.json').read_text())
if status.get('status')!='completed' or status.get('completed_episodes')!=32 or status.get('reference_base_manifest_sha256')!=expected:raise ValueError('baseline not complete')
if status.get('runtime_binding_sha256')!='c982b471408232b1a9d5361b320f90932c83e92cefce15d433df65783254ffa0':raise ValueError('wrong native binding')
extras={manifest_name,'config.json','study/status.json','study/identity_report.json','study/capture_validation.json'}
names=set(manifest['files'])|extras;files={}
for name in sorted(names):
 fixed=name in extras|{'data/episodes/index.jsonl','study/reference_base_rows.json','v2_runtime_binding.json'}
 episode=re.fullmatch(r'data/episodes/base\.base\.s(300(?:1[6-9]|[23][0-9]|4[0-7]))/(obs\.npz|summary\.json)',name)
 packet=re.fullmatch(r'data/batch_packets/[0-9a-f]{32}/(packet\.json|requests\.npz|recorded_actions\.npz)',name)
 if not(fixed or episode or packet):raise ValueError('foreign evidence member')
 p=regular(name);actual=digest(p)
 if name in manifest['files'] and actual!=manifest['files'][name]:raise ValueError('baseline evidence changed')
 files[name]={'bytes':p.stat().st_size,'sha256':actual}
metadata={'schema_version':1,'artifact_kind':'common-fixed16-baseline-only','source_host':'runpod2','source_project':str(root),
 'recorded_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reference_manifest_sha256':expected,
 'files':files,'remote_writes':0,'GPU_calls':0,'includes_candidate_evidence':False,'includes_heldout':False}
raw=json.dumps(metadata,indent=2).encode()+b'\n'
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz',compresslevel=4) as target:
 info=tarfile.TarInfo(prefix+'/TRANSPORT_MANIFEST.json');info.size=len(raw);info.mode=0o444;target.addfile(info,io.BytesIO(raw))
 for name in sorted(files):
  p=regular(name);info=tarfile.TarInfo(prefix+'/'+name);info.size=files[name]['bytes'];info.mode=0o444
  with p.open('rb') as stream:target.addfile(info,stream)
if digest(manifest_path)!=expected or digest(regular('study/status.json'))!=files['study/status.json']['sha256']:raise ValueError('completed baseline changed during stream')
'''


def verify_archive(archive,*,destination=None):
    destination=Path(destination) if destination is not None else None
    if destination is not None:
        if destination.exists() or destination.is_symlink():raise ValueError('Fresh operator staging directory required')
        destination.mkdir(parents=True)
    transport=None;seen=set();computed={};metadata={};total=0
    with tarfile.open(archive,'r|gz') as stream:
        for member in stream:
            path=PurePosixPath(member.name)
            if path.as_posix()!=member.name or path.is_absolute() or '..' in path.parts or path.parts[:1]!=(PREFIX,) or not member.isfile():raise ValueError('Nonregular or unsafe archive member')
            name=path.relative_to(PREFIX).as_posix()
            if name in seen:raise ValueError('Duplicate archive member')
            seen.add(name)
            if transport is None:
                if name!='TRANSPORT_MANIFEST.json' or member.size>4<<20:raise ValueError('First transport manifest required')
                raw=stream.extractfile(member).read();transport=json.loads(raw)
                if (transport.get('reference_manifest_sha256')!=MANIFEST_SHA or transport.get('source_project')!=ROOT
                    or not all(allowed(n) for n in transport.get('files',{}))):raise ValueError('Wrong or foreign transport inventory')
                if destination:(destination/name).write_bytes(raw)
                metadata[name]=raw;continue
            if name not in transport['files'] or not allowed(name):raise ValueError('Unexpected archive member')
            if member.size>256<<20:raise ValueError('Unexpected member size')
            h=hashlib.sha256();count=0;parts=[] if not name.endswith('.npz') else None
            target=destination/name if destination else None
            if target:target.parent.mkdir(parents=True,exist_ok=True)
            out=target.open('xb') if target else None
            try:
                with stream.extractfile(member) as inp:
                    for block in iter(lambda:inp.read(1<<20),b''):
                        h.update(block);count+=len(block)
                        if out:out.write(block)
                        if parts is not None:parts.append(block)
            finally:
                if out:out.close();target.chmod(0o444)
            actual={'bytes':count,'sha256':h.hexdigest()}
            if member.size!=count or actual!=transport['files'][name]:raise ValueError('Member checksum/size mismatch')
            computed[name]=actual;total+=count
            if parts is not None:metadata[name]=b''.join(parts)
    if transport is None or set(computed)!=set(transport['files']):raise ValueError('Incomplete common evidence archive')
    manifest=json.loads(metadata[MANIFEST])
    if hashlib.sha256(metadata[MANIFEST]).hexdigest()!=MANIFEST_SHA:raise ValueError('Original baseline manifest differs')
    if set(computed)!=set(manifest['files'])|EXTRAS or any(computed[n]['sha256']!=h for n,h in manifest['files'].items()):raise ValueError('Baseline membership/hash mismatch')
    expected={f'data/episodes/base.base.s{i}/{name}' for i in SLOTS for name in ('obs.npz','summary.json')}
    if {n for n in computed if n.startswith('data/episodes/') and n!='data/episodes/index.jsonl'}!=expected:raise ValueError('Exactly32 base recordings required')
    native=json.loads(metadata['v2_runtime_binding.json']);status=json.loads(metadata['study/status.json']);validation=json.loads(metadata['study/capture_validation.json'])
    if (computed['v2_runtime_binding.json']['sha256']!=NATIVE_SHA or computed['study/capture_validation.json']['sha256']!=VALIDATION_SHA
        or status.get('status')!='completed' or status.get('completed_episodes')!=32 or status.get('attempted_episodes')!=32
        or status.get('reference_base_manifest_sha256')!=MANIFEST_SHA or status.get('runtime_binding_sha256')!=NATIVE_SHA
        or native.get('source_manifest_sha256')!=SOURCE_SHA or validation.get('status')!='passed' or validation.get('runtime_binding_sha256')!=NATIVE_SHA):
        raise ValueError('Original-host readiness/binding metadata differs')
    if (status.get('config_sha256')!=computed['config.json']['sha256'] or status.get('identity_report_sha256')!=computed['study/identity_report.json']['sha256']
        or status.get('capture_validation_sha256')!=VALIDATION_SHA):raise ValueError('Readiness metadata checksum differs')
    index=[json.loads(line) for line in metadata['data/episodes/index.jsonl'].splitlines() if line.strip()]
    rows=json.loads(metadata['study/reference_base_rows.json'])
    for cohort in (index,rows):
        if len(cohort)!=32 or {r.get('slot') for r in cohort}!={f's{i}' for i in SLOTS}:raise ValueError('Wrong development slots')
        for r in cohort:
            if r.get('cand')!='base' or r.get('fingerprint')!='base' or type(r.get('success'))is not bool or r.get('gripper_threshold')!=.5:raise ValueError('Only original base evidence is allowed')
            if r.get('obs_sha256')!=computed[f'data/episodes/{r["eid"]}/obs.npz']['sha256']:raise ValueError('Observation fingerprint differs')
    full={r['eid']:r for r in rows}
    for r in index:
        if r!={k:v for k,v in full[r['eid']].items() if k not in ('events','video')} or json.loads(metadata[f'data/episodes/{r["eid"]}/summary.json'])!=full[r['eid']]:raise ValueError('Original JSON records differ')
    packet_ids={n.split('/')[2] for n in computed if n.startswith('data/batch_packets/')}
    if any({f'data/batch_packets/{pid}/{n}' for n in ('packet.json','requests.npz','recorded_actions.npz')}-set(computed) for pid in packet_ids):raise ValueError('Incomplete native packet')
    return {'status':'verified_exact_common_baseline','archive_sha256':sha(archive),'archive_bytes':Path(archive).stat().st_size,
        'reference_manifest_sha256':MANIFEST_SHA,'native_runtime_binding_sha256':NATIVE_SHA,
        'capture_validation_sha256':VALIDATION_SHA,'source_manifest_sha256':SOURCE_SHA,
        'verified_evidence_files':len(manifest['files']),'verified_metadata_extras':len(EXTRAS),'regular_archive_members':len(seen),
        'native_packets':len(packet_ids),'episodes':32,'baseline_successes':sum(r['success'] for r in rows),
        'uncompressed_evidence_bytes':total,'original_bytes_unchanged':True,'readiness_scope':'original host only; no teammate-host GPU or provider readiness implied',
        'source_host':'runpod2','source_project':ROOT,'candidate_or_heldout_evidence':False}


def verify_packets(destination,codec_path):
    """Recheck every request and complete actual real-position membership locally."""
    root=Path(destination);codec_path=Path(codec_path)
    if sha(codec_path)!=CODEC_SHA:raise ValueError('Only the sealed CPU packet codec may execute')
    if root.is_symlink() or sha(root/MANIFEST)!=MANIFEST_SHA or sha(root/'v2_runtime_binding.json')!=NATIVE_SHA:raise ValueError('Original extracted baseline identities required')
    inventory=json.loads((root/'TRANSPORT_MANIFEST.json').read_text())['files']
    pinned=json.loads((root/MANIFEST).read_text())['files']
    if set(inventory)!=set(pinned)|EXTRAS:raise ValueError('Original extracted inventory required')
    for name,entry in inventory.items():
        if not allowed(name):raise ValueError('Foreign extracted evidence')
        path=root/name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():raise ValueError('Only regular extracted evidence is permitted')
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256'] or (name in pinned and entry['sha256']!=pinned[name]):raise ValueError('Extracted evidence changed after archive verification')
    spec=importlib.util.spec_from_file_location('fixed16_common_packet_codec',codec_path);codec=importlib.util.module_from_spec(spec);spec.loader.exec_module(codec)
    native=json.loads((root/'v2_runtime_binding.json').read_text());rows=json.loads((root/'study/reference_base_rows.json').read_text());cache={};seen=set();reports=[]
    identity={'checkpoint_manifest_sha256':native['checkpoint_manifest_sha256'],'runtime_manifest_sha256':sha(root/'v2_runtime_binding.json'),'native_server_sha256':native['native_server_sha256'],'policy_fingerprint':'base'}
    for row in rows:
        adapted={'slot':int(row['slot'][1:]),'ok':True,'fingerprint':'base','task_name':row['sim_task'],'prompt':row['prompt'],
            'success':row['success'],'steps':row['steps'],'gripper_threshold':row['gripper_threshold'],'obs':str(root/'data/episodes'/row['eid']/'obs.npz')}
        result=codec.verify_episode_packet_links(adapted,root,identity,'Stack the yellow block on the red block','StackYellowOnRedTask',packet_cache=cache)
        for item in result['packets']:
            key=(item['packet_id'],item['position'])
            if key in seen:raise ValueError('Native real position mapped more than once')
            seen.add(key)
        reports.append({k:v for k,v in result.items() if k!='packets'})
    for pid,(_, (packet,_,_)) in cache.items():
        if any((pid,pos) not in seen for pos in range(packet['layout']['real_count'])):raise ValueError('Native packet real member lacks a baseline request')
    actual={p.name for p in (root/'data/batch_packets').iterdir()}
    if actual!=set(cache):raise ValueError('Unreferenced or missing baseline packet')
    return {'status':'all_baseline_packet_links_verified','episodes':len(reports),'requests':len(seen),'native_packets':len(cache),
        'all_actual_real_positions_mapped_once':True,'codec_sha256':sha(codec_path),'episode_checks':reports}


def fetch(output):
    output=Path(output);partial=output.with_suffix(output.suffix+'.partial')
    if output.exists() or partial.exists():raise FileExistsError('Existing transfer/partial cannot be overwritten')
    with partial.open('xb') as stream:
        process=subprocess.run(['ssh','-o','BatchMode=yes','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=2','runpod2','python3 -c '+shlex.quote(REMOTE)],stdout=stream,stderr=subprocess.PIPE,timeout=1800)
    if process.returncode:raise RuntimeError('Read-only baseline transfer failed: '+process.stderr.decode(errors='replace')[-4000:])
    receipt=verify_archive(partial);partial.rename(output);return receipt


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    p=sub.add_parser('fetch');p.add_argument('--output',type=Path,required=True)
    p=sub.add_parser('verify');p.add_argument('--archive',type=Path,required=True);p.add_argument('--extract-to',type=Path)
    p=sub.add_parser('verify-packets');p.add_argument('--root',type=Path,required=True);p.add_argument('--codec',type=Path,required=True)
    args=parser.parse_args()
    result=fetch(args.output) if args.command=='fetch' else verify_archive(args.archive,destination=args.extract_to) if args.command=='verify' else verify_packets(args.root,args.codec)
    print(json.dumps(result,indent=2))
