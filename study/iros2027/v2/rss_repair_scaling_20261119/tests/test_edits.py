"""Numerical checks of the frozen ridge estimator and nested source use."""
from pathlib import Path
import sys
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import build_edits as E

class EditTests(unittest.TestCase):
    def test_ridge_solution_satisfies_primal_normal_equations(self):
        rng=np.random.default_rng(19);K=rng.normal(size=(72,11));V=rng.normal(size=(72,7))
        delta,report=E.fit(K,V);lam=report['ridge_absolute']
        residual=(K.T@K+lam*np.eye(11))@delta-K.T@V
        self.assertLess(np.linalg.norm(residual)/np.linalg.norm(K.T@V),1e-10)
        self.assertEqual(report['rank_upper_bound'],72)

    def test_one_correction_is_exact_prefix_not_all_available_sources(self):
        rng=np.random.default_rng(31)
        def arrays():return {'vlm.mlp_hidden':rng.normal(size=(3,17,64,4)),
             'vlm.resid_out':rng.normal(size=(3,17,64,3)),
             'vlm.img_mlp_hidden':rng.normal(size=(3,17,2,17,4)),
             'vlm.img_resid_out':rng.normal(size=(3,17,2,17,3))}
        original,teacher=arrays(),arrays();first=E.pooled_key_values(original,teacher,1)
        for v in original.values():v[1:]=1e6
        for v in teacher.values():v[1:]=-1e6
        second=E.pooled_key_values(original,teacher,1)
        for layer in E.LAYERS:
            self.assertEqual(first[layer][0].shape[0],72)
            np.testing.assert_array_equal(first[layer][0],second[layer][0])
            np.testing.assert_array_equal(first[layer][1],second[layer][1])
        with self.assertRaises(ValueError):E.pooled_key_values(original,teacher,2)

    def test_zero_keys_cannot_emit_candidate(self):
        with self.assertRaises(ValueError):E.fit(np.zeros((72,4)),np.ones((72,3)))

if __name__=='__main__':unittest.main()
