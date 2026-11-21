"""Checks against pooling artifacts and changed target construction."""
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import local_activation_fit as L
import numpy as np


class LocalActivationFitTests(unittest.TestCase):
    def test_pooled_perfect_fit_can_hide_bad_individual_fit(self):
        prediction = np.array([[[1.0]], [[1.0]]])
        target = np.array([[[0.5]], [[1.5]]])
        states = L.per_state_metrics(prediction, target)
        pooled_error = float(np.sum((prediction.mean(0)-target.mean(0))**2))
        self.assertEqual(pooled_error, 0.0)
        self.assertGreater(L.aggregate(states)["relative_error"], 0.0)

    def test_null_and_exact_targets(self):
        target = np.arange(1, 9, dtype=np.float64).reshape(2, 2, 2)
        for row in L.per_state_metrics(np.zeros_like(target), target):
            self.assertEqual(row["relative_error"], 1.0)
            self.assertEqual(row["target_energy_reduction"], 0.0)
        self.assertEqual(L.aggregate(L.per_state_metrics(target, target))["relative_error"], 0.0)
        with self.assertRaises(ValueError): L.metrics(0, 0)

    def test_single_state_definition_matches_exact_construction_pool(self):
        rng = np.random.default_rng(561)
        original, teacher = {}, {}
        for key in L.recipe.ARRAYS:
            width = 3 if "hidden" in key else 2
            shape = (9, 17, 2, 17, width) if ".img_" in key else (9, 17, 40, width)
            original[key] = rng.normal(size=shape).astype(np.float32)
            teacher[key] = rng.normal(size=shape).astype(np.float32)
        slots = [(i, i) for i in range(40)]
        group = [8, 0, 4]
        pooled = L.recipe.pooled_key_values(original, teacher, group, slots)
        for layer in (0, 1, 16):
            keys, values = L.local_systems(original, teacher, slots, layer)
            self.assertEqual(keys.shape, (9, 72, 3))
            L.check_pooled_identity(keys, values, group, pooled[layer])
        keys, values = L.local_systems(original, teacher, slots, 0)
        bad = (pooled[0][0]+1, pooled[0][1])
        with self.assertRaises(ValueError): L.check_pooled_identity(keys, values, group, bad)


if __name__ == "__main__":
    unittest.main()
