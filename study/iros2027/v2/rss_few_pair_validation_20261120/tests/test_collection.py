import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import common
import collect
import preflight
import prepare_sources

class CollectionTests(unittest.TestCase):
    def test_disjoint_source_groups_are_reproducible(self):
        ids=[f'pair{i}' for i in range(9)]
        a=prepare_sources.partition_pairs(ids,120)
        self.assertEqual(a,prepare_sources.partition_pairs(ids,120))
        self.assertEqual(sorted(x for g in a for x in g['pair_ids']),ids)
        self.assertTrue(all(len(g['indices'])==3 for g in a))
        with self.assertRaises(ValueError):prepare_sources.partition_pairs(ids[:8],120)
        with self.assertRaises(ValueError):prepare_sources.partition_pairs(['same']*9,120)

    def test_pair_gate_requires_failed_original_and_successful_teacher(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'obs.npz'
            np.savez(path,actions=np.zeros((5,16,7)),gripper=np.zeros(5))
            row={'slot':300000,'layout_id':'layout','policy_seed':7,'obs':str(path),'success':False}
            teacher={**row,'success':True}
            with patch.object(prepare_sources,'select_failure_decision',return_value=2):
                self.assertEqual(prepare_sources.eligibility('cube_behind',row,teacher),(2,[]))
                self.assertIn('original_succeeded',prepare_sources.eligibility('cube_behind',teacher,teacher)[1])
                self.assertIn('clarified_failed',prepare_sources.eligibility('cube_behind',row,row)[1])
                with self.assertRaises(ValueError):prepare_sources.eligibility('cube_behind',row,{**teacher,'policy_seed':8})

    def test_plan_and_freeze_reject_source_changes(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common,'ROOT',Path(tmp)):
            g=common.goal('temporal');layout=common.goal_root('temporal')/'design/collection-layouts.json'
            layout.parent.mkdir(parents=True);layout.write_text('{}')
            plan={'cohort_id':common.protocol()['cohort_id'],'protocol_sha256':common.sha(HERE/'protocol.json'),
                  'goal_id':'temporal','case':g['runtime_case'],'runtime_dir':g['runtime_dir'],
                  'split':'development','stage':'collection','inference_batching':'fixed16',
                  'batches':[{'task':g['task'],'goal':g['goal'],'layouts':str(layout),'n':16,'slot_start':300000,'workspace':None}]}
            common.validate_plan(plan)
            broken=copy.deepcopy(plan);broken['batches'][0]['n']=8
            with self.assertRaises(ValueError):common.validate_plan(broken)
            path=common.freeze_plan(Path(tmp)/'plan.json',plan,{str(layout):common.sha(layout)})
            worker=common.load('fewpair_worker_test',HERE/'worker.py')
            worker.verify_global_freeze(common.read(path))
            layout.write_text('{"changed": true}')
            with self.assertRaises(ValueError):worker.verify_global_freeze(common.read(path))

    def test_reset_certificate_requires_each_layout_twice_and_no_actions(self):
        with tempfile.TemporaryDirectory() as tmp:
            layout=Path(tmp)/'layouts.json';ids=[f'id{i}' for i in range(16)]
            layout.write_text(json.dumps({'layouts':[{'id':x} for x in ids],'objects':['cube']}))
            value={'status':'complete','goal_id':'temporal','reset_cycles':2,'policy_calls':0,'action_calls':0,
                   'forbidden_modules_loaded':[],'layout_ids':ids,
                   'resets':[{'layout_ids':ids,'vertical_displacement_checked':True,'table_contact':{'cube':[True]*16}}]*2,
                   'source_sha256':{str(layout.resolve()):common.sha(layout)}}
            preflight.validate_summary(value,'temporal',layout)
            for change in ({'action_calls':1},{'policy_calls':1},{'reset_cycles':1},{'resets':[{'layout_ids':ids}]}):
                with self.assertRaises(ValueError):preflight.validate_summary({**value,**change},'temporal',layout)

    def test_fixed_episode_caps_account_for_all_conditions(self):
        p=common.protocol()
        self.assertEqual(544,64*(3+3+1+1)+32)
        self.assertEqual(208,16*(1+3+3*3))
        self.assertEqual(p['limits']['max_total_new_policy_episodes'],3*(96+208+544))

if __name__=='__main__':unittest.main()
