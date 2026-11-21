import copy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import evaluate as E
from common import goal,protocol,sha,HERE

def candidates():
    return [{'candidate_id':f'{family}-s{strength}','family':family,'strength':strength,
        'method':'embedding' if family=='embedding' else 'residual','workspace':'somewhere',
        'group_id':None if family=='embedding' else family[-2:]}
        for family in ('embedding','residual_g0','residual_g1','residual_g2') for strength in (.5,1.,2.)]

def rows(n):
    return [{'slot':i,'layout_id':f'L{i}','policy_seed':10*i,'success':i<n} for i in range(16)]

class EvaluationTests(unittest.TestCase):
    def test_serving_fingerprint_matches_native_names_and_values(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'edits').mkdir()
            manifest={'tensors':{'tensor.z':{'file':'first.npy'},'tensor.a':{'file':'second.npy'}}}
            arrays={'tensor.z':np.array([2,3],dtype=np.float32),'tensor.a':np.array([7,9],dtype=np.float32)}
            for name,array in arrays.items():np.save(root/'edits'/manifest['tensors'][name]['file'],array)
            native=hashlib.sha256()
            for name in sorted(arrays):native.update(name.encode());native.update(arrays[name].tobytes())
            self.assertEqual(E.serving_fingerprint(root,manifest,list(arrays)),native.hexdigest())
            self.assertNotEqual(E.serving_fingerprint(root,manifest,['tensor.a']),native.hexdigest())

    def test_full_schedule_and_competing_goals(self):
        for gid in protocol()['goal_ids']:
            g=goal(gid)
            conditions=['base','clarified','embedding',*[f'weightsmith_g{i}' for i in range(3)],*[f'residual_g{i}' for i in range(3)]]
            batches=[b for c in conditions for b in E.test_cells(gid,c,None)]
            self.assertEqual(sum(b['n'] for b in batches),544)
            for b in batches:
                if b['cell']=='competing':self.assertEqual((b['task'],b['goal'],b['effective_prompt']),(g['competing_task'],g['competing_goal'],g['competing']))
            originals=E.test_cells(gid,'base',None)[:2]
            self.assertEqual(originals[1]['slot_start']-originals[0]['slot_start'],16)
            self.assertEqual(originals[0]['layouts'],originals[1]['layouts'])

    def test_selection_ties_and_zero_fallback(self):
        cs=candidates();groups={c['candidate_id']:rows(4) for c in cs};groups['base']=rows(4)
        out=E.select_candidates(cs,groups)
        self.assertTrue(all(x['strength']==0 and x['workspace'] is None for x in out))
        for c in cs:groups[c['candidate_id']]=rows(6)
        out=E.select_candidates(cs,groups)
        self.assertTrue(all(x['strength']==.5 for x in out))
        groups['residual_g1-s2.0']=rows(7)
        out=E.select_candidates(cs,groups)
        self.assertEqual(next(x for x in out if x['candidate_id']=='residual_g1')['strength'],2.)

    def test_selection_rejects_missing_or_mismatched_trials(self):
        cs=candidates();groups={c['candidate_id']:rows(4) for c in cs};groups['base']=rows(4)
        bad=copy.deepcopy(groups);bad[cs[0]['candidate_id']].pop()
        with self.assertRaises(ValueError):E.select_candidates(cs,bad)
        bad=copy.deepcopy(groups);bad[cs[0]['candidate_id']][0]['policy_seed']+=1
        with self.assertRaises(ValueError):E.select_candidates(cs,bad)
        bad=copy.deepcopy(groups);bad.pop('base')
        with self.assertRaises(ValueError):E.select_candidates(cs,bad)

    def test_plan_rejects_prompt_and_split_substitution(self):
        gid='cube_behind';g=goal(gid)
        plan={'cohort_id':protocol()['cohort_id'],'goal_id':gid,'case':g['runtime_case'],
            'runtime_dir':g['runtime_dir'],'stage':'heldout','split':'heldout','inference_batching':'fixed16',
            'protocol_sha256':sha(HERE/'protocol.json'),'batches':E.test_cells(gid,'base',None)}
        E.validate_evaluation_plan(plan)
        bad=copy.deepcopy(plan);bad['batches'][0]['effective_prompt']=g['teacher']
        with self.assertRaises(ValueError):E.validate_evaluation_plan(bad)
        bad=copy.deepcopy(plan);bad['split']='development'
        with self.assertRaises(ValueError):E.validate_evaluation_plan(bad)

if __name__=='__main__':unittest.main()
