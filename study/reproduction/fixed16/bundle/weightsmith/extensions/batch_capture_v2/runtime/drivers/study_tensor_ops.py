#!/usr/bin/env python3
"""Declarative NumPy operations for the study; no dynamic code or user paths.

Runs on the GPU host's CPU. The operator binds WSP_ROOT and BASE_WORKSPACE;
input is one JSON object on stdin. Only practice observations/captures are read.
"""
from __future__ import annotations
import base64
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import sys
import zlib

import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from weightsmith import config as C
from weightsmith.integrity import workspace_fingerprint
from weightsmith.lab.openpi_head import _effective_bfloat16
from weightsmith.lab.openpi_vlm_full import validate_workspace

from study_validation import validate_episode_id

NAME=re.compile(r'^[a-zA-Z0-9_.@-]+$')
TENSOR=re.compile(r'^action\.blocks\.(\d{2})\.mlp_1\.linear$')

def child(root,name):
    if not isinstance(name,str) or not NAME.fullmatch(name) or name in ('.','..'):
        raise ValueError('invalid identifier')
    path=root/name
    if path.is_symlink() or path.resolve().parent!=root.resolve():
        raise ValueError('identifier cannot escape its declared root')
    return path


def png_rgb(image):
    image=np.ascontiguousarray(image,dtype=np.uint8)
    h,w,c=image.shape
    if c!=3:raise ValueError('expected RGB')
    def chunk(name,payload):
        return struct.pack('>I',len(payload))+name+payload+struct.pack('>I',zlib.crc32(name+payload)&0xffffffff)
    raw=b''.join(b'\0'+image[y].tobytes() for y in range(h))
    return b'\x89PNG\r\n\x1a\n'+chunk(b'IHDR',struct.pack('>IIBBBBB',w,h,8,2,0,0,0))+chunk(b'IDAT',zlib.compress(raw))+chunk(b'IEND',b'')


class TensorOps:
    def __init__(self,root,base):
        self.root=Path(root);self.base=Path(base)
        self.episodes=self.root/'data/episodes';self.acts=self.root/'data/acts'
        self.directions=self.root/'notes/study_directions'

    def workspace(self,candidate):
        if candidate=='base':return self.base
        if candidate not in ('p1','p2','p3'):raise ValueError('candidate must be base or p1,p2,p3')
        return child(self.root/'cands',candidate)

    def tensor(self,candidate,name):
        ws=self.workspace(candidate)
        manifest=json.loads((ws/'workspace.json').read_text())
        if not manifest.get('edit_constraints'):raise ValueError('restricted study workspace required')
        match=TENSOR.fullmatch(name)
        if not match or not 0<=int(match[1])<18 or name not in manifest['tensors']:
            raise ValueError('only action MLP down-projection tensors are exposed')
        entry=manifest['tensors'][name]
        file=entry['file']
        if Path(file).name!=file:raise ValueError('invalid manifest file')
        original=np.load(ws/'base'/file,allow_pickle=False)
        edit_path=ws/'edits'/file
        if edit_path.is_symlink():raise ValueError('symlinked edit is forbidden')
        current=np.load(edit_path,allow_pickle=False) if edit_path.is_file() else original.copy()
        if original.shape!=(4096,1024) or current.shape!=original.shape:raise ValueError('unexpected tensor shape')
        return ws,edit_path,original,current

    def rows(self,values):
        if not isinstance(values,list) or not 1<=len(values)<=128 or any(type(v) is not int or not 0<=v<4096 for v in values):
            raise ValueError('rows must be 1..128 integer indices in 0..4095')
        if len(set(values))!=len(values):raise ValueError('duplicate rows')
        return np.asarray(values,dtype=np.int64)

    def checked_capture(self, identifier):
        directory = child(self.acts, identifier)
        for name in ('index.json', 'acts.npz'):
            path = directory / name
            if path.is_symlink() or not path.is_file():
                raise ValueError('A complete regular v2 capture artifact is required')
        idx = json.loads((directory / 'index.json').read_text())
        marker = idx.get('capture_validation', {})
        required = {'protocol_id': 'wsp-batch-capture-v2-fixed16', 'status': 'passed',
                    'all_real_positions_checked': True, 'parity_tolerance': 0.02,
                    'zero_skipped': True}
        if any(type(marker.get(k)) is not type(v) or marker.get(k) != v
               for k, v in required.items()):
            raise ValueError('Only passed whole-batch v2 captures may produce contrasts or directions')
        return directory, idx

    def contrast(self,spec):
        aid=spec['capture'];directory,idx=self.checked_capture(aid)
        if spec.get('base_only') and idx.get('cand')!='base':raise ValueError('nonadaptive contrasts must use base captures')
        top_k=spec.get('top_k',64)
        if type(top_k)is not int or not 1<=top_k<=128:raise ValueError('top_k must be an integer in 1..128')
        key=spec.get('key','action.mlp_hidden')
        if key not in ('action.mlp_hidden','action.resid_out'):raise ValueError('only action hidden/residual contrasts exposed')
        layer=spec['layer'];step=spec.get('denoise_step',0)
        li=idx['action_layers'].index(layer);si=idx['denoise_steps'].index(step)
        with np.load(directory/'acts.npz',allow_pickle=False) as z:
            values=np.asarray(z[key][:,si,li],dtype=np.float64).mean(axis=-2)
        if len(values)!=len(idx['items']):raise ValueError('capture item count mismatch')
        if not np.isfinite(values).all():raise ValueError('nonfinite activations')
        # One vector per episode: multiple sampled requests do not inflate group size.
        grouped={}
        for item,v in zip(idx['items'],values):grouped.setdefault(validate_episode_id(item['episode']),[]).append(v)
        successes=[];failures=[]
        for eid,vs in grouped.items():
            summary=json.loads((child(self.episodes,eid)/'summary.json').read_text())
            (successes if summary['success'] else failures).append(np.mean(vs,axis=0))
        if not successes or not failures:raise ValueError('capture must include success and failure episodes')
        positive=np.stack(successes);negative=np.stack(failures)
        delta=positive.mean(0)-negative.mean(0)
        pooled=np.sqrt((positive.var(0)+negative.var(0))/2+1e-12)
        effect=delta/pooled
        top=np.argsort(np.abs(effect))[-top_k:][::-1]
        result={'capture':aid,'key':key,'layer':layer,'denoise_step':step,
                'success_episodes':len(successes),'failure_episodes':len(failures),
                'positive_minus_negative_norm':float(np.linalg.norm(delta)),
                'top_features':[{'index':int(i),'success_mean':float(positive[:,i].mean()),
                                 'failure_mean':float(negative[:,i].mean()),'standardized_contrast':float(effect[i])} for i in top],
                'interpretation':'Descriptive episode-averaged association, not causal localization.',
                'capture_index_sha256':hashlib.sha256((directory/'index.json').read_bytes()).hexdigest(),
                'capture_arrays_sha256':hashlib.sha256((directory/'acts.npz').read_bytes()).hexdigest(),
                'captured_policy_fingerprint':idx.get('policy_fingerprint','base' if idx.get('cand')=='base' else None),
                'episode_summary_sha256':{eid:hashlib.sha256((child(self.episodes,eid)/'summary.json').read_bytes()).hexdigest() for eid in grouped}}
        if key=='action.resid_out':
            norm=np.linalg.norm(delta)
            if norm<=1e-12:raise ValueError('residual contrast has zero norm')
            direction=(delta/norm).astype(np.float32)
            identifier=hashlib.sha256(direction.tobytes()+json.dumps({k:result[k] for k in ('capture','layer','denoise_step')},sort_keys=True).encode()).hexdigest()
            self.directions.mkdir(parents=True,exist_ok=True)
            np.save(self.directions/(identifier+'.npy'),direction,allow_pickle=False)
            result['direction_array_sha256']=hashlib.sha256((self.directions/(identifier+'.npy')).read_bytes()).hexdigest()
            (self.directions/(identifier+'.json')).write_text(json.dumps(result,indent=2)+'\n')
            result['direction_id']=identifier
            result['direction_sign']='unit vector from failure mean toward success mean'
        return result

    def direction(self,identifier, expected_layer=None):
        if not re.fullmatch('[0-9a-f]{64}',identifier):raise ValueError('direction ID must be a returned SHA256')
        path=self.directions/(identifier+'.npy')
        if path.is_symlink():raise ValueError('symlinked direction')
        metadata=json.loads(path.with_suffix('.json').read_text())
        if hashlib.sha256(path.read_bytes()).hexdigest()!=metadata['direction_array_sha256']:raise ValueError('direction hash mismatch')
        capture, _ = self.checked_capture(metadata['capture'])
        if (hashlib.sha256((capture/'index.json').read_bytes()).hexdigest()!=metadata['capture_index_sha256']
                or hashlib.sha256((capture/'acts.npz').read_bytes()).hexdigest()!=metadata['capture_arrays_sha256']):
            raise ValueError('Direction source capture changed after validation')
        if expected_layer is not None and metadata['layer']!=expected_layer:raise ValueError('direction was derived from a different layer')
        value=np.load(path,allow_pickle=False)
        if value.shape!=(1024,) or not np.isfinite(value).all():raise ValueError('invalid direction')
        return value

    def run(self,spec):
        op=spec.get('op')
        if op=='activation_contrast':return self.contrast(spec)
        if op in ('observation','observation_summary'):
            directory=child(self.episodes,validate_episode_id(spec['episode']))
            with np.load(directory/'obs.npz',allow_pickle=False) as z:
                if op=='observation_summary':
                    return {'episode':spec['episode'],'requests':len(z['step']),'steps':z['step'].tolist(),
                            'gripper':z['gripper'].reshape(-1).tolist(),
                            'action_gripper_mean':z['actions'][:,:,7].mean(axis=1).tolist(),
                            'action_gripper_min':z['actions'][:,:,7].min(axis=1).tolist(),
                            'action_gripper_max':z['actions'][:,:,7].max(axis=1).tolist()}
                k=spec.get('request_index',0)
                if type(k)is not int or not 0<=k<len(z['step']):raise ValueError('invalid recorded request index')
                image=np.concatenate([z['base_img'][k],z['wrist_img'][k]],axis=1)
                return {'_image':base64.b64encode(png_rgb(image)).decode(),'mimeType':'image/png',
                        'episode':spec['episode'],'request_index':k,'step':int(z['step'][k])}
        candidate=spec.get('candidate','base')
        ws,path,original,current=self.tensor(candidate,spec['tensor'])
        if op=='row_stats':
            rows=self.rows(spec['rows']) if spec.get('rows') is not None else np.arange(4096)
            values=current[rows].astype(np.float64)
            norms=np.linalg.norm(values,axis=1)
            order=np.argsort(norms)[-min(128,len(rows)):][::-1]
            result={'tensor':spec['tensor'],'rows_total':len(rows),'row_norm_order':'descending',
                    'rows':[{'row':int(rows[i]),'norm':float(norms[i]),'mean':float(values[i].mean()),
                             'std':float(values[i].std())} for i in order]}
            if spec.get('direction_id'):
                u=self.direction(spec['direction_id'],int(TENSOR.fullmatch(spec['tensor'])[1]))
                for row in result['rows']:row['projection']=float(current[row['row']]@u)
            return result
        if candidate=='base':raise ValueError('base weights are immutable')
        if op not in ('row_scale','row_add_direction','row_project_out'):raise ValueError('unknown declarative operation')
        rows=self.rows(spec['rows'])
        reference=spec.get('reference','current')
        if reference not in ('base','current'):raise ValueError('reference must be base or current')
        source=original if reference=='base' else current
        amount=spec.get('factor',1.0)
        if isinstance(amount,bool) or not isinstance(amount,(int,float)) or not math.isfinite(amount) or abs(amount)>1e6:
            raise ValueError('factor must be finite and within [-1e6,1e6]')
        selected=source[rows].astype(np.float64)
        if op=='row_scale':updated=selected*amount
        else:
            u=self.direction(spec['direction_id'],int(TENSOR.fullmatch(spec['tensor'])[1])).astype(np.float64)
            if op=='row_add_direction':updated=selected+amount*u[None,:]
            else:updated=selected-amount*(selected@u)[:,None]*u[None,:]
        current[rows]=_effective_bfloat16(updated)
        if not np.isfinite(current).all():raise ValueError('operation produced nonfinite values')
        previous=path.read_bytes() if path.exists() else None
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_suffix('.pending.npy')
        np.save(temporary,current,allow_pickle=False);temporary.replace(path)
        try:report=validate_workspace(ws)
        except Exception:
            if previous is None:path.unlink()
            else:path.write_bytes(previous)
            raise
        report['fingerprint']=workspace_fingerprint(ws)
        report['operation']=op
        return report


if __name__=='__main__':
    try:
        payload=sys.stdin.read(200_001)
        if len(payload)>200_000:raise ValueError('request too large')
        print(json.dumps(TensorOps(C.PROJECT_ROOT,C.BASE_WS).run(json.loads(payload))))
    except Exception as error:
        sys.exit(str(error))
