import importlib.util
from pathlib import Path
import unittest
import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
spec = importlib.util.spec_from_file_location("rss_scoring", RUNTIME / "scoring.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
Scorer = module.Scorer


class RuntimeScoringTests(unittest.TestCase):
    def temporal(self):
        initial = {"rubiks_cube": [[0., 0., 0.]], "banana": [[1., 0., 0.]], "bowl": [[2., 0., 0.]]}
        return Scorer("temporal", initial, .1), initial

    def sample(self, scorer, pos, tick, cube=False, banana=False):
        return scorer.update(tick, pos, {"rubiks_cube_placed": [cube], "banana_placed": [banana]})

    def test_temporal_requires_correct_first_order_and_full_stable_final(self):
        s, p = self.temporal()
        self.sample(s, p, 1, cube=True)
        for t in range(2, 12):
            self.sample(s, p, t, cube=True, banana=True)
        self.assertFalse(s.current["cube_then_banana"][0])  # Only .9 seconds.
        self.sample(s, p, 12, cube=True, banana=True)
        self.assertTrue(s.current["cube_then_banana"][0])
        self.assertFalse(s.current["banana_then_cube"][0])
        self.sample(s, p, 13, cube=False, banana=True)
        self.assertFalse(s.current["cube_then_banana"][0])
        self.assertTrue(s.ever["cube_then_banana"][0])

    def test_wrong_first_placement_cannot_be_undone(self):
        s, p = self.temporal()
        self.sample(s, p, 1, banana=True)
        self.sample(s, p, 2, cube=True, banana=False)
        for t in range(3, 20):
            self.sample(s, p, t, cube=True, banana=True)
        self.assertFalse(s.current["cube_then_banana"][0])
        self.assertTrue(s.current["banana_then_cube"][0])

    def test_simultaneous_placement_is_stock_but_not_strict_success(self):
        s, p = self.temporal()
        for t in range(1, 20):
            self.sample(s, p, t, cube=True, banana=True)
        self.assertTrue(s.stock["cube_then_banana"][0])
        self.assertFalse(s.current["cube_then_banana"][0])
        self.assertFalse(s.current["banana_then_cube"][0])

    def test_cube_only_rejects_banana_moved_then_returned(self):
        s, p = self.temporal()
        p["banana"] = [[1.04, 0., 0.]]
        self.sample(s, p, 1)
        p["banana"] = [[1., 0., 0.]]
        for t in range(2, 20):
            self.sample(s, p, t, cube=True)
        self.assertFalse(s.current["cube_only"][0])

    def spatial(self):
        p = {"mustard_bottle": [[0., 0., 0.]], "raisin_box": [[1., 0., 0.]]}
        pred = {"right_of": [True], "left_of": [False], "on_raisin": [False],
                "mustard_table_contact": [True], "mustard_raisin_contact": [False], "mustard_detached": [True]}
        return Scorer("spatial", p, .1), p, pred

    def test_spatial_no_op_is_not_success_and_reference_return_does_not_recover(self):
        s, p, pred = self.spatial()
        for t in range(1, 20):
            s.update(t, p, pred)
        self.assertTrue(s.stock["mustard_right"][0])
        self.assertFalse(s.current["mustard_right"][0])
        p["mustard_bottle"] = [[0., -.1, 0.]]
        p["raisin_box"] = [[1.03, 0., 0.]]
        s.update(20, p, pred)
        p["raisin_box"] = [[1., 0., 0.]]
        for t in range(21, 40):
            s.update(t, p, pred)
        self.assertFalse(s.current["mustard_right"][0])
        self.assertFalse(s.parts["reference_fixed"][0])

    def test_spatial_detachment_and_dwell_must_be_continuous(self):
        s, p, pred = self.spatial()
        p["mustard_bottle"] = [[0., -.1, 0.]]
        for t in range(1, 12):
            s.update(t, p, pred)
        self.assertFalse(s.current["mustard_right"][0])
        pred["mustard_detached"] = [False]
        s.update(12, p, pred)
        pred["mustard_detached"] = [True]
        for t in range(13, 23):
            s.update(t, p, pred)
        self.assertFalse(s.current["mustard_right"][0])
        s.update(23, p, pred)
        self.assertTrue(s.current["mustard_right"][0])

    def test_duplicate_tick_does_not_accumulate_dwell_and_frozen_env_is_immutable(self):
        s, p, pred = self.spatial()
        p["mustard_bottle"] = [[0., -.1, 0.]]
        s.update(1, p, pred)
        for _ in range(20):
            s.update(2, p, pred)
        self.assertEqual(s.count["mustard_right"][0], 1)
        for t in range(3, 13):
            s.update(t, p, pred)
        self.assertTrue(s.current["mustard_right"][0])
        p["raisin_box"] = [[4., 0., 0.]]
        pred["right_of"] = [False]
        s.update(13, p, pred, active=[False])
        self.assertTrue(s.current["mustard_right"][0])
        self.assertEqual(s.maximum["raisin_box"][0], 0.)

    def test_reset_requires_fresh_scorer_and_clears_event_latches(self):
        s, p = self.temporal()
        self.sample(s, p, 10, banana=True)
        with self.assertRaises(RuntimeError):
            self.sample(s, p, 1)
        fresh = Scorer("temporal", p, .1)
        self.sample(fresh, p, 1, cube=True)
        self.assertEqual(fresh.first["banana"][0], -1)


if __name__ == "__main__":
    unittest.main()
