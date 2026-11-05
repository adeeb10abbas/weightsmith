"""Candidate-cohort provenance must validate all requests before acceptance."""
import ast
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch
import numpy as np
RUNTIME=Path(__file__).resolve().parents[1]/'runtime';sys.path.insert(0,str(RUNTIME))
from weightsmith.lab.batch_context_v2 import write_packet,sha,verify_episode_packet_links


def candidate_fixture(root):
    (root/'v2_runtime_binding.json').write_text('{"fixture":true}\n')
    identity={'checkpoint_manifest_sha256':'a'*64,'runtime_manifest_sha256':sha(root/'v2_runtime_binding.json'),
              'native_server_sha256':'c'*64,'policy_fingerprint':'d'*64}
    instruction='Stack the yellow block on the red block';slot=30016;steps=[]
    for k in range(2):
        requests=[]
        for pos in range(3):
            s=slot+pos
            requests.append({'observation/exterior_image_1_left':np.full((224,224,3),k+pos,np.uint8),
                'observation/wrist_image_left':np.full((224,224,3),k+pos+10,np.uint8),
                'observation/joint_position':np.arange(7,dtype=np.float32)+k+pos,
                'observation/gripper_position':np.asarray([.25],np.float32),'prompt':instruction,
                'sampling_seed':((s//16)*64+s%16)*100+k})
        actions=np.arange(3*15*8,dtype=np.float64).reshape(3,15,8)/10000+k/100
        pid=f'{k+1:032x}';saved=write_packet(root/'data/batch_packets'/pid,requests,actions,identity,variant='fixed16_new_runtime')
        r=requests[0]
        steps.append(dict(base_img=r['observation/exterior_image_1_left'],wrist_img=r['observation/wrist_image_left'],
            joint=r['observation/joint_position'],gripper=r['observation/gripper_position'],prompt=np.asarray(instruction),
            seed=np.asarray(r['sampling_seed']),step=np.asarray(15*k),actions=actions[0].astype(np.float32),
            batch_packet_id=np.asarray(pid),batch_packet_sha256=np.asarray(saved['packet_sha256']),
            batch_position=np.asarray(0),batch_real_count=np.asarray(3),batch_padded_count=np.asarray(16)))
    obs=root/'candidate-obs.npz';np.savez_compressed(obs,**{key:np.stack([s[key] for s in steps]) for key in steps[0]})
    row={'slot':slot,'ok':True,'fingerprint':identity['policy_fingerprint'],'task_name':'StackYellowOnRedTask',
         'prompt':instruction,'success':False,'steps':30,'gripper_threshold':.5,'obs':str(obs)}
    return row,identity,instruction


class CandidatePackets(unittest.TestCase):
    def test_edited_policy_all_requests_verified(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);row,identity,prompt=candidate_fixture(root)
            report=verify_episode_packet_links(row,root,identity,prompt,row['task_name'])
            self.assertEqual(report['requests_checked'],2);self.assertTrue(report['all_recorded_requests_checked'])
            self.assertEqual(len(report['packets']),2);self.assertEqual(report['policy_fingerprint'],'d'*64)

    def test_second_record_corruption_cannot_hide_behind_valid_first_record(self):
        for field in ('seed','step','base_img','actions','batch_position','batch_real_count','batch_padded_count'):
            with self.subTest(field=field),tempfile.TemporaryDirectory() as t:
                root=Path(t);row,identity,prompt=candidate_fixture(root);obs=Path(row['obs'])
                with np.load(obs,allow_pickle=False) as z:arrays={key:z[key] for key in z.files}
                arrays[field][1].flat[0]+=1 if arrays[field].ndim>1 else 0
                if arrays[field].ndim==1:arrays[field][1]+=1
                np.savez_compressed(obs,**arrays)
                with self.assertRaises(ValueError):verify_episode_packet_links(row,root,identity,prompt,row['task_name'])

    def test_missing_candidate_links_or_packet_fail(self):
        for missing in ('field','packet'):
            with self.subTest(missing=missing),tempfile.TemporaryDirectory() as t:
                root=Path(t);row,identity,prompt=candidate_fixture(root);obs=Path(row['obs'])
                if missing=='field':
                    with np.load(obs,allow_pickle=False) as z:arrays={key:z[key] for key in z.files if key!='batch_packet_id'}
                    np.savez_compressed(obs,**arrays)
                else:(root/'data/batch_packets'/f'{2:032x}'/'packet.json').unlink()
                with self.assertRaises((ValueError,FileNotFoundError)):verify_episode_packet_links(row,root,identity,prompt,row['task_name'])

    def test_truncated_record_prefix_cannot_cover_full_terminal_episode(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);row,identity,prompt=candidate_fixture(root);obs=Path(row['obs'])
            with np.load(obs,allow_pickle=False) as z:arrays={key:z[key][:1] for key in z.files}
            np.savez_compressed(obs,**arrays)
            with self.assertRaisesRegex(ValueError,'terminal simulator step'):verify_episode_packet_links(row,root,identity,prompt,row['task_name'])

    def test_policy_runtime_slot_and_boolean_binding(self):
        changes=({'fingerprint':'base'},{'slot':30017},{'success':'false'},{'gripper_threshold':.4},{'prompt':'other'})
        for change in changes:
            with self.subTest(change=change),tempfile.TemporaryDirectory() as t:
                root=Path(t);row,identity,prompt=candidate_fixture(root)
                with self.assertRaises(ValueError):verify_episode_packet_links({**row,**change},root,identity,prompt,row['task_name'])
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);row,identity,prompt=candidate_fixture(root)
            (root/'v2_runtime_binding.json').write_text('{"changed":true}')
            with self.assertRaises(ValueError):verify_episode_packet_links(row,root,identity,prompt,row['task_name'])

    def test_invalid_candidate_is_charged_but_never_packaged(self):
        from weightsmith.lab import batch_runtime_v2,batch_context_v2
        tree=ast.parse((RUNTIME/'weightsmith/owsp.py').read_text())
        function=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cmd_eval')
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);events=[];identity={'policy_fingerprint':'d'*64}
            def run(*args,**kwargs):
                self.assertEqual(events,[{'episodes':16,'evals':1}])
                return [{'slot':30016+k,'ok':True} for k in range(16)]
            namespace={'cand_dir':lambda name:None,'validate':Mock(),'CFG':{'envs':16,'practice_seed_base':30016,'practice_slots':32,'prompt_overrides':False,'instruction':'fixture'},
                'check_budget':Mock(),'fingerprint':lambda name:'d'*64,'STATE':root/'state','charge':lambda **kwargs:events.append(kwargs),
                'json':json,'time':__import__('time'),'run_eval':run,'C':types.SimpleNamespace(PROJECT_ROOT=root),'TASK':'StackYellowOnRedTask',
                'sys':sys,'EPI':root/'episodes'}
            exec(compile(ast.Module(body=[function],type_ignores=[]),'candidate-eval-under-test','exec'),namespace)
            args=types.SimpleNamespace(name='p1',n=16,offset=0,prompt=None)
            with patch.object(batch_runtime_v2,'project_identity',return_value=identity),patch.object(batch_context_v2,'verify_episode_packet_links',side_effect=ValueError('corrupt actual packet')):
                with self.assertRaisesRegex(ValueError,'corrupt actual packet'):namespace['cmd_eval'](args)
            self.assertFalse((root/'episodes').exists());self.assertEqual(len(events),1)
            attempts=list((root/'state/eval_attempts').glob('*.json'));self.assertEqual(len(attempts),1)
            audit=json.loads(attempts[0].read_text());self.assertEqual(audit['status'],'failed');self.assertEqual(audit['episodes_reserved'],16)
            self.assertEqual(len(audit['rows']),16)


class ForwardAccounting(unittest.TestCase):
    def test_reserved_started_and_completed_distinguish_failed_stages(self):
        spec=importlib.util.spec_from_file_location('v2_acts_for_accounting',RUNTIME/'weightsmith/acts_worker.py')
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        settings={'action_layers':[0],'vlm_layers':[],'denoise_steps':[0],'action_tokens':'mean','prompt_tokens':[0]}
        plan={'settings':settings,'logical_items':1,'unique_packets':1,'predicted_bytes':1024,'padded_forward_items_planned':32,
              'packets':{'one':{'requests':[{}]}}}
        for fail,started,completed in [('native',16,0),('instrumented',32,16)]:
            with self.subTest(fail=fail),tempfile.TemporaryDirectory() as t:
                fake=types.SimpleNamespace(record_actual_batches=True,report={'fingerprint':'base'},graphdef=None,state=None,
                    nnx=types.SimpleNamespace(merge=lambda *args:types.SimpleNamespace()),load=lambda *args:None)
                def native(requests):
                    if fail=='native':raise RuntimeError('native failure')
                    return [{'actions':np.zeros((15,8))}],{'observation':None,'noise':None}
                fake.infer_context=native
                def probe(*args):raise RuntimeError('instrumented failure')
                nnx_utils=types.SimpleNamespace(module_jit=lambda method:probe)
                shared=types.ModuleType('openpi.shared');shared.nnx_utils=nnx_utils
                with patch.object(module,'workspace_fingerprint',return_value='base'),patch.object(module,'project_identity',return_value={}),patch.object(module,'plan_capture',return_value=plan),patch.dict(sys.modules,{'jax':types.ModuleType('jax'),'openpi.shared':shared}):
                    with self.assertRaisesRegex(RuntimeError,fail+' failure'):module.capture({'items':[{}]},Path(t),server=fake)
                report=json.loads((Path(t)/'index.json').read_text())
                self.assertEqual(report['padded_forward_items_reserved'],32)
                self.assertEqual(report['padded_forward_items_started'],started)
                self.assertEqual(report['padded_forward_items_completed'],completed)
                self.assertEqual(report['capture_validation']['status'],'failed')

if __name__=='__main__':unittest.main()
