"""Meaningful CPU-only provenance, explicit noise, precision and planning checks."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
RUNTIME=Path(__file__).resolve().parents[1]/'runtime'
sys.path.insert(0,str(RUNTIME))
from weightsmith.lab.batch_context_v2 import (PROTOCOL_ID,batch_layout,seeded_noise,write_packet,read_packet,
    pack_requests,sha,three_way_parity,compare_actions,write_traces)
from weightsmith.lab.capture_plan_v2 import plan_capture,estimate_bytes

IDENTITY={'checkpoint_manifest_sha256':'a'*64,'runtime_manifest_sha256':'b'*64,
          'native_server_sha256':'c'*64,'policy_fingerprint':'base'}


def requests(n=3):
    return [{'observation/exterior_image_1_left':np.full((224,224,3),i,np.uint8),
             'observation/wrist_image_left':np.full((224,224,3),i+40,np.uint8),
             'observation/joint_position':np.arange(7,dtype=np.float32)+i/8,
             'observation/gripper_position':np.asarray([i/4],np.float32),
             'prompt':f'ordered request {i}','sampling_seed':1000+i*137} for i in range(n)]


def fixture(root):
    req=requests();actions=np.arange(3*15*8,dtype=np.float64).reshape(3,15,8)/1000
    pid='d'*32;written=write_packet(root/'data/batch_packets'/pid,req,actions,IDENTITY,variant='fixed16_new_runtime')
    items=[]
    for i,r in enumerate(req):
        eid=f'base.base.s{30016+i}';folder=root/'data/episodes'/eid;folder.mkdir(parents=True)
        (folder/'summary.json').write_text(json.dumps({'cand':'base','fingerprint':'base'}))
        np.savez_compressed(folder/'obs.npz',base_img=r['observation/exterior_image_1_left'][None],
            wrist_img=r['observation/wrist_image_left'][None],joint=r['observation/joint_position'][None],
            gripper=r['observation/gripper_position'][None],seed=np.asarray([r['sampling_seed']]),step=np.asarray([0]),
            prompt=np.asarray([r['prompt']]),actions=actions[i:i+1].astype(np.float32),batch_packet_id=np.asarray([pid]),
            batch_packet_sha256=np.asarray([written['packet_sha256']]),batch_position=np.asarray([i]),
            batch_real_count=np.asarray([3]),batch_padded_count=np.asarray([16]))
        items.append({'episode':eid,'k':0})
    spec={'cand':'base','episodes_dir':str(root/'data/episodes'),'items':items,
          'require_parity':True,'parity_tolerance':.02,'action_layers':[16,17],'denoise_steps':[9],'vlm_layers':[],
          'action_tokens':'mean','prompt_tokens':list(range(24))}
    return spec,pid,written


class BatchContextTests(unittest.TestCase):
    def test_per_request_noise_and_padding_preserve_explicit_draws(self):
        seeds=[7,101,5];calls=[]
        def draw(seed):calls.append(seed);return np.random.default_rng(seed).normal(size=(1,15,32))
        actual=seeded_noise(seeds,batch_layout(3,'fixed16_new_runtime'),draw)
        self.assertEqual(calls,seeds);self.assertEqual(actual.shape,(16,15,32))
        for i,seed in enumerate(seeds):np.testing.assert_array_equal(actual[i],np.random.default_rng(seed).normal(size=(1,15,32))[0])
        for i in range(3,16):np.testing.assert_array_equal(actual[i],actual[0])
        self.assertFalse(np.array_equal(actual,np.random.default_rng(seeds[0]).normal(size=(16,15,32))))

    def test_packet_retains_heterogeneous_order_and_float64_outputs(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,pid,written=fixture(root)
            packet,req,actions=read_packet(root/'data/batch_packets'/pid,written['packet_sha256'])
            self.assertEqual([r['sampling_seed'] for r in req],[1000,1137,1274])
            self.assertEqual(packet['layout']['position_to_source'],[0,1,2]+[0]*13)
            self.assertEqual(actions.dtype,np.float64)
            self.assertEqual(pack_requests(req)[1],packet['request_content_sha256'])

    def test_legacy_or_duplicate_claim_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(ValueError):write_packet(Path(t)/'x',requests(1),np.zeros((1,15,8)),IDENTITY,provenance='reconstructed_duplicate_B16')

    def test_tampered_array_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,pid,written=fixture(root);p=root/'data/batch_packets'/pid/'requests.npz'
            p.chmod(0o644)
            with p.open('ab') as f:f.write(b'changed')
            with self.assertRaises(ValueError):read_packet(p.parent,written['packet_sha256'])

    def test_tampered_order_even_with_new_manifest_hash_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,pid,written=fixture(root);p=root/'data/batch_packets'/pid/'packet.json';d=json.loads(p.read_text())
            d['request_content_sha256'].reverse();p.chmod(0o644);p.write_text(json.dumps(d))
            with self.assertRaises(ValueError):read_packet(p.parent,sha(p))

    def test_symlink_array_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,pid,written=fixture(root);p=root/'data/batch_packets'/pid/'requests.npz'
            target=root/'external.npz';p.rename(target);p.symlink_to(target)
            with self.assertRaises(ValueError):read_packet(p.parent,written['packet_sha256'])

    def test_full_real_batch_gate_detects_unselected_failure_without_binary_flip(self):
        native=np.zeros((3,15,8));recorded=native.copy();instrumented=native.copy();instrumented[2,4,7]=.09
        result=three_way_parity(native,instrumented,recorded)
        self.assertFalse(result['eligible_capture'])
        check=result['checks']['instrumented_vs_native'];self.assertEqual(check['checked_count'],3)
        self.assertEqual(check['items'][2]['gripper_flips_tau_0p5'],[])
        self.assertTrue(check['items'][0]['passed'])

    def test_cast_aware_equality_keeps_raw_error(self):
        raw=np.full((1,15,8),.123456789123,np.float64);stored=raw.astype(np.float32)
        result=compare_actions(raw,stored)
        self.assertGreater(result['max_abs'],0);self.assertTrue(result['exact_at_right_recording_dtype'])

    def test_raw_traces_preserve_all_positions_tokens_precision(self):
        values=np.arange(2*16*15*3,dtype=np.float64).reshape(2,16,15,3)/7
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'traces';schema=write_traces(d,{'hidden':values})
            with np.load(d/'traces.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['a00000'],values)
            self.assertEqual(schema['arrays'][0]['logical_dtype'],'float64');self.assertEqual(schema['pooling'],'none')

    def test_bfloat16_preserves_bits_when_dtype_available(self):
        try:import ml_dtypes
        except ImportError:self.skipTest('optional CPU bfloat16 dtype unavailable')
        values=np.asarray([1,1.0078125,10000],dtype=ml_dtypes.bfloat16)
        with tempfile.TemporaryDirectory() as t:
            d=Path(t)/'traces';schema=write_traces(d,{'bf16':values})
            with np.load(d/'traces.npz',allow_pickle=False) as z:np.testing.assert_array_equal(z['a00000'],values.view(np.uint16))
            self.assertEqual(schema['arrays'][0]['encoding'],'bfloat16_uint16_bits')
            for number,value in enumerate((float('nan'),float('inf'),-float('inf'))):
                bad=np.asarray([value],dtype=ml_dtypes.bfloat16)
                with self.assertRaisesRegex(ValueError,'nonfinite'):
                    write_traces(Path(t)/f'bad-{number}',{'bf16':bad})


class CapturePlanTests(unittest.TestCase):
    def test_deduplicates_packet_but_counts_all_real_positions(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,_,_=fixture(root);spec['items']=spec['items'][:2]
            plan=plan_capture(spec,root,IDENTITY)
            self.assertEqual((plan['unique_packets'],plan['logical_items'],plan['real_positions_to_check'],plan['padded_forward_items_planned']),(1,2,3,32))

    def test_strict_same_policy_and_runtime(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,_,_=fixture(root)
            with self.assertRaises(ValueError):plan_capture({**spec,'cand':'p1'},root,IDENTITY)
            with self.assertRaises(ValueError):plan_capture(spec,root,{**IDENTITY,'runtime_manifest_sha256':'e'*64})
            with self.assertRaises(ValueError):plan_capture({**spec,'require_parity':False},root,IDENTITY)

    def test_legacy_and_corrupted_position_input_or_actions_rejected(self):
        for field in ('batch_packet_id','batch_position','base_img','actions'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
                root=Path(t);spec,_,_=fixture(root);p=root/'data/episodes'/spec['items'][0]['episode']/'obs.npz'
                with np.load(p,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
                if field=='batch_packet_id':del arrays[field]
                else:arrays[field].flat[0]+=1
                np.savez_compressed(p,**arrays)
                with self.assertRaises(ValueError):plan_capture(spec,root,IDENTITY)

    def test_default_full_capture_rejected_before_gpu(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,_,_=fixture(root);spec.pop('action_layers');spec.pop('denoise_steps')
            with self.assertRaisesRegex(ValueError,'exceeds512MiB'):plan_capture(spec,root,IDENTITY)

    def test_failed_capture_storage_still_consumes_project_budget(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);spec,_,_=fixture(root);folder=root/'data/acts/failed';folder.mkdir(parents=True)
            with (folder/'partial.npz').open('wb') as f:f.truncate((1<<30)-1024)
            with self.assertRaisesRegex(ValueError,'project capture'):plan_capture(spec,root,IDENTITY)

    def test_import_has_no_gpu_import_or_environment_mutation(self):
        code='import os,sys; before=dict(os.environ); sys.path.insert(0,sys.argv[1]); from weightsmith.lab import batch_context_v2,capture_plan_v2,batch_probe_v2; assert "jax" not in sys.modules; assert "openpi" not in sys.modules; assert dict(os.environ)==before'
        subprocess.run([sys.executable,'-c',code,str(RUNTIME)],check=True,capture_output=True,text=True)



class AttemptAccountingTests(unittest.TestCase):
    def test_startup_failure_is_charged_and_retained(self):
        import ast, io, types
        from unittest.mock import Mock
        source=(RUNTIME/'weightsmith/owsp.py').read_text();tree=ast.parse(source)
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=='cmd_acts')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);episodes=root/'episodes';e=episodes/'base.base.s30016';e.mkdir(parents=True);(e/'obs.npz').write_bytes(b'fixture')
            events=[]
            def charge(**kwargs):events.append(('charge',kwargs))
            def dispatch(*args,**kwargs):
                self.assertEqual(events,[('charge',{'acts_calls':1,'acts_items':1})]);raise OSError('simulated startup failure')
            class Lock:
                def __init__(self,*args):pass
                def __enter__(self):return self
                def __exit__(self,*args):pass
            spec={'id':'startup-test','items':[{'episode':e.name,'k':0}],'project_capture_bytes':2**50}
            namespace={'json':json,'sys':types.SimpleNamespace(stdin=io.StringIO(json.dumps(spec)),exit=lambda message:(_ for _ in ()).throw(RuntimeError(message))),
                'CFG':{'prompt_overrides':False,'limits':{'acts_items_per_call':96},'instruction':'fixture','study':{'project_capture_bytes':1<<30}},
                'episode_dir':lambda eid:episodes/eid,'check_budget':Mock(),'cand_dir':lambda name:None,'validate':Mock(),
                'NAME_RE':__import__('re').compile(r'^[a-zA-Z0-9_.-]+$'),'ACTS':root/'acts','charge':charge,
                'EPI':episodes,'GpuLock':Lock,'C':types.SimpleNamespace(ACTS_GPU=0,JAX_CACHE=root/'jax',ACTS_MEM_FRACTION='.9'),
                'os':__import__('os'),'subprocess':types.SimpleNamespace(run=dispatch),'OPENPI_PY':'not-executed','HARNESS':RUNTIME/'weightsmith'}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'owsp-isolated-function','exec'),namespace)
            with self.assertRaisesRegex(OSError,'simulated startup'):namespace['cmd_acts'](None)
            attempt=json.loads((root/'acts/startup-test/attempt.json').read_text())
            self.assertEqual(attempt['status'],'startup_or_dispatch_failed')
            self.assertEqual(attempt['spec']['project_capture_bytes'],1<<30)
            self.assertEqual(len(events),1)

class ValidationScopeTests(unittest.TestCase):
    def test_versions_stay_in_owned_root_with_original_reference(self):
        import importlib.util
        spec=importlib.util.spec_from_file_location('v2_validation_helper',RUNTIME.parent/'validate_gpu.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as t:
            root=Path(t).resolve();source=root/'projects/rq123-development-v1';source.mkdir(parents=True)
            for version in (1,2,10):
                project=root/f'projects/rq123-v2-validation-v{version}'
                self.assertEqual(module.checked_paths(project,source,str(root))[0],project)
            invalid=[root/'projects/rq123-v2-validation-v0',root/'other/rq123-v2-validation-v2',root/'projects/rq123-v2-adaptive-r1']
            for project in invalid:
                with self.assertRaises(ValueError):module.checked_paths(project,source,str(root))
            with self.assertRaises(ValueError):module.checked_paths(root/'projects/rq123-v2-validation-v2',root/'projects/foreign-project',str(root))
            link=root/'projects/rq123-v2-validation-v3';link.symlink_to(root/'elsewhere')
            with self.assertRaises(ValueError):module.checked_paths(link,source,str(root))
            with self.assertRaises(ValueError):module.checked_paths(root/'projects/rq123-v2-validation-v2',source,None)

if __name__=='__main__':unittest.main()
