"""Meaningful source-count, pairing, split, and outcome-filter checks."""
import importlib.util
from pathlib import Path
import sys
import ast
import tempfile
import unittest
import numpy as np

HERE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE))
import common
import prepare_sources as S

class CollectionTests(unittest.TestCase):
    def test_worker_sets_port_before_runtime_imports(self):
        module=ast.parse((HERE/'worker.py').read_text())
        main=next(n for n in module.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        envline=next(n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and ast.unparse(n.func)=='os.environ.update')
        loadlines=[n.lineno for n in ast.walk(main) if isinstance(n,ast.Call) and ast.unparse(n.func)=='load']
        self.assertTrue(all(envline<line for line in loadlines))

    def test_frozen_budget_and_disjoint_layouts(self):
        p=common.protocol()
        self.assertEqual(p['budgets'],[1,3,6,12])
        self.assertEqual(p['collection']['max_pairs'],48)
        blobs=[common.layouts(k) for k in ('collection','near','far')]
        poses=[common.canonical(r['poses']) for b in blobs for r in b['layouts']]
        self.assertEqual(len(set(poses)),len(poses))
        self.assertEqual([b['split'] for b in blobs],['development','development','heldout'])
        far=blobs[-1]['layouts']
        self.assertEqual(sum(r['poses']['rubiks_cube']['pos'][0]>r['poses']['banana']['pos'][0] for r in far),8)
        for row in far:
            self.assertGreater(abs(row['poses']['bowl']['pos'][0]-.465),.02)
            self.assertGreater(np.linalg.norm(np.array(row['poses']['rubiks_cube']['pos'])[:2]-np.array(row['poses']['banana']['pos'])[:2]),.09)

    def test_one_episode_yields_one_first_release_and_pair_outcomes_gate(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'obs.npz'
            actions=np.zeros((8,15,8),dtype=np.float32)
            actions[[1,4],:,7]=1
            np.savez(path,actions=actions,gripper=np.full((8,1),.3))
            original={'slot':1,'layout_id':'A','policy_seed':99,'success':False,'obs':str(path)}
            teacher={**original,'success':True}
            self.assertEqual(S.eligible(original,teacher),(2,[]))
            self.assertIn('original_succeeded',S.eligible({**original,'success':True},teacher)[1])
            self.assertIn('clarified_failed',S.eligible(original,{**teacher,'success':False})[1])
            with self.assertRaises(ValueError):S.eligible(original,{**teacher,'slot':2})

    def test_no_release_is_not_replaced_with_arbitrary_frame(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'obs.npz';np.savez(path,actions=np.ones((8,15,8)),gripper=np.full((8,1),.3))
            o={'slot':1,'layout_id':'A','policy_seed':99,'success':False,'obs':str(path)}
            self.assertEqual(S.eligible(o,{**o,'success':True}),(None,['no_early_release_decision']))

if __name__=='__main__':unittest.main()
