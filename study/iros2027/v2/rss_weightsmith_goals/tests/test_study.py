import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

scoring=module('new_goal_scoring',HERE/'runtime/scoring.py')
selector=module('new_goal_selector',HERE/'adaptation/prepare.py')
pipeline=module('new_goal_pipeline',HERE/'adaptation/pipeline.py')
layout=module('new_goal_layout',HERE/'runtime/layouts.py')
evaluation=module('new_goal_evaluation',HERE/'evaluate.py')
audit=module('new_goal_report',HERE/'report.py')
worker=module('new_goal_worker',HERE/'worker.py')

class ProtocolTests(unittest.TestCase):
    def test_goal_and_budget_identity(self):
        p=json.loads((HERE/'protocol.json').read_text())
        self.assertEqual(set(p['case_order']),set(scoring.CASES))
        self.assertEqual(p['qualification']['max_policy_episodes_before_adaptation'],3*(32+32+16))
        self.assertEqual(p['heldout']['episodes_per_goal'],2*5*8*2)
        self.assertEqual(p['adaptation']['preservation_keys'],0)
        self.assertFalse(p['adaptation']['rank_truncation'])
        for case,cfg in p['cases'].items():
            self.assertEqual(len(set(cfg['unseen'])),3)
            self.assertFalse({cfg['original'],cfg['teacher']} & set(cfg['unseen']))
            for stage in ('initial','repeat','competing'):
                self.assertGreaterEqual(p['qualification']['slot_start'][stage][case],200000)
            dev=layout.load(HERE/'runtime/layouts'/f'{case}_development.json')
            test=layout.load(HERE/'runtime/layouts'/f'{case}_heldout.json')
            self.assertEqual((len(dev['layouts']),len(test['layouts'])),(4,8))
            self.assertFalse({x['id'] for x in dev['layouts']} & {x['id'] for x in test['layouts']})

class PhaseTests(unittest.TestCase):
    def test_contact_at_replan_boundary_belongs_to_preceding_chunk(self):
        grip=np.zeros(70,dtype=bool);grip[30:]=True
        height=np.zeros(70);height[42:]=.04
        result=selector.phase_indices([0,15,30,45,60],grip,height)
        self.assertEqual(result,{'precontact':0,'grasp':1,'transport':3})
    def test_failed_no_grasp_uses_failure_local_last_decision(self):
        result=selector.phase_indices([0,15,30],[False]*40,[0.]*40,failed_original=True)
        self.assertEqual(result,{'precontact':2})
    def test_no_transport_without_clearance(self):
        grip=np.zeros(40,dtype=bool);grip[18:]=True
        result=selector.phase_indices([0,15,30],grip,np.zeros(40))
        self.assertNotIn('transport',result)
    def test_missing_phase_coverage_is_not_fabricated(self):
        with self.assertRaises(ValueError):selector.phase_indices([0,15,30],[False]*20,[0.]*20)

class ScoreTests(unittest.TestCase):
    def make(self):
        initial={'banana':np.zeros((2,3)),'plate_large':np.array([[.2,.2,0.],[.2,.2,0.]]),'bowl':np.ones((2,3))}
        return scoring.Scorer('banana_left_plate',initial,.25),initial
    def predicates(self):
        return {'target':np.ones(2,bool),'preservation':np.zeros(2,bool),'mover_table':np.ones(2,bool),
            'mover_reference':np.zeros(2,bool),'detached_banana':np.ones(2,bool),'detached_plate_large':np.ones(2,bool),
            'detached_bowl':np.ones(2,bool),'mover_gripped':np.zeros(2,bool)}
    def test_requires_movement_stability_and_full_dwell(self):
        s,p=self.make();v=self.predicates()
        for t in range(8):s.update(t,p,v)
        self.assertFalse(s.current['banana_left_plate'].any())
        p={k:x.copy() for k,x in p.items()};p['banana'][:,0]=.05
        s.update(8,p,v)
        for t in range(9,13):s.update(t,p,v)
        self.assertFalse(s.current['banana_left_plate'].any())
        s.update(13,p,v);self.assertTrue(s.current['banana_left_plate'].all())
    def test_reference_motion_disqualifies_and_frozen_components_stay_fixed(self):
        s,p=self.make();v=self.predicates();s.update(0,p,v)
        p={k:x.copy() for k,x in p.items()};p['banana'][:,0]=.1;p['plate_large'][0,0]+=.03
        for t in range(1,9):s.update(t,p,v)
        self.assertFalse(s.current['banana_left_plate'][0]);self.assertTrue(s.current['banana_left_plate'][1])
        changed=self.predicates();changed['target'][:]=False
        s.update(9,p,changed,active=[True,False])
        self.assertTrue(s.current['banana_left_plate'][1]);self.assertTrue(s.parts['target'][1])

class FitTests(unittest.TestCase):
    def test_full_phase_rows_match_untruncated_ridge_solution(self):
        rng=np.random.default_rng(3);K=rng.normal(size=(96,110));V=rng.normal(size=(96,5));H=np.empty((0,110))
        delta,info=pipeline.fit(K,V,H)
        lam=.001*np.sum(K*K)/96
        np.testing.assert_allclose((K.T@K+lam*np.eye(110))@delta,K.T@V,atol=1e-9)
        self.assertEqual(info['preservation_keys'],0);self.assertFalse(info['rank_truncation'])
        with self.assertRaises(ValueError):pipeline.fit(K,V,np.ones((32,110)))

class FreezeTests(unittest.TestCase):
    def test_freeze_retains_source_coverage_failure_and_all_unseen_pairs(self):
        protocol=json.loads((HERE/'protocol.json').read_text())
        completed,failed=protocol['case_order'][:2]
        qualification={'selected':[completed,failed],
            'repeat':{case:{'passes_gate':True} for case in (completed,failed)}}
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            def write(relative,value):
                path=root/relative;path.parent.mkdir(parents=True,exist_ok=True)
                path.write_text(json.dumps(value));return path
            write('qualification-status.json',qualification)
            write('qualification-source-freeze.json',{})
            write(f'{failed}-adaptation/status.json',{'status':'source_coverage_failed','heldout_used':False})
            write(f'{completed}-adaptation/status.json',{'status':'complete'})
            write(f'qualification/{completed}/qualified.json',{'preservation_competence_qualified':True})
            adaptation=root/f'{completed}-adaptation'
            for relative,value in [('spec.json',{}),('identity.json',{}),
                ('edit-fit/solve.json',{'replay_parity':{'status':'pass'},'teacher_action_parity':{'status':'pass'}}),
                ('native-targets/dataset.json',{}),('edit-prepared/prepared.json',{})]:
                write(f'{completed}-adaptation/{relative}',value)
            write(f'{completed}-adaptation/candidate-seal.json',{'case':completed,'split':'development',
                'artifacts':{},'spec_sha256':evaluation.sha(adaptation/'spec.json'),
                'solve_sha256':evaluation.sha(adaptation/'edit-fit/solve.json'),
                'source_identity_sha256':evaluation.sha(adaptation/'identity.json')})
            with patch.object(evaluation,'verify_qualification',return_value=(qualification,[])), \
                 patch.object(evaluation,'verify_gate',return_value={}), \
                 patch.object(evaluation,'source_files',return_value=[]):
                manifest=evaluation.freeze(root,root/'frozen')
            self.assertEqual(manifest['qualified_cases'],[completed,failed])
            self.assertEqual(manifest['cases'],[completed])
            self.assertEqual(manifest['planned_episodes'],160)
            self.assertIn(str(root/f'{failed}-adaptation/status.json'),manifest['artifacts'])
            plan=json.loads((root/f'frozen/{completed}-plan.json').read_text())
            self.assertEqual(len(plan['batches']),10)
            for cell in protocol['heldout']['cells']:
                pair=[row for row in plan['batches'] if row['cell']==cell]
                self.assertEqual({row['condition'] for row in pair},{'base','weightsmith'})
                self.assertEqual(pair[0]['slot_start'],pair[1]['slot_start'])
                self.assertEqual(pair[0]['input_prompt'],pair[1]['input_prompt'])
                self.assertTrue(all(row['input_prompt']==row['effective_prompt'] for row in pair))

    def test_pairing_rejects_changed_noise(self):
        rows=[{'layout_id':f'L{i//2}','slot':i,'policy_seed':i,'success':bool(i%2),
               'first_camera_sha256':{'base_img':'x','wrist_img':'y'}} for i in range(16)]
        edited=[{**row,'success':True} for row in rows]
        result=audit.paired(rows,edited)
        self.assertEqual((result['layouts'],result['fixed'],result['broken']),(8,8,0))
        edited[0]['policy_seed']=999
        with self.assertRaises(ValueError):audit.paired(rows,edited)

class ExecutionReceiptTests(unittest.TestCase):
    def test_wrong_workspace_or_missing_identity_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            batch=Path(directory);(batch/'sim-l0-a0.log').touch()
            spec={'workspace':'/candidate/workspace'}
            with self.assertRaisesRegex(ValueError,'workspace'):
                worker.verify_batch_execution(spec,json.dumps({'workspace':None}),batch)
            with self.assertRaisesRegex(ValueError,'workspace'):
                worker.verify_batch_execution({'workspace':None},{},batch)
            self.assertEqual(len(worker.verify_batch_execution(spec,json.dumps(spec),batch)),1)

    def test_extra_simulator_attempt_and_incorrect_receipt_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            batch=Path(directory);first=batch/'sim-l0-a0.log';first.touch()
            spec={'workspace':None}
            with self.assertRaisesRegex(ValueError,'preserved logs'):
                worker.verify_batch_execution(spec,json.dumps(spec),batch,[])
            (batch/'sim-l0-a1.log').touch()
            with self.assertRaisesRegex(ValueError,'Exactly one'):
                worker.verify_batch_execution(spec,json.dumps(spec),batch,[str(first)])

if __name__=='__main__':unittest.main()
