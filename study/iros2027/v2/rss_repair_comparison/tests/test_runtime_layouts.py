import copy
import importlib.util
import json
from pathlib import Path
import unittest
import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"


def module(name):
    spec = importlib.util.spec_from_file_location("rss_" + name, RUNTIME / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


layouts = module("layouts")
generator = module("generate_layouts")


class RuntimeLayoutTests(unittest.TestCase):
    def test_frozen_layouts_are_deterministic_disjoint_and_initially_nontrivial(self):
        for case in ("temporal", "spatial"):
            seen = set()
            for split, count in (("development", 4), ("heldout", 8)):
                spec = json.loads((RUNTIME / "layouts" / f"{case}_{split}.json").read_text())
                self.assertEqual(spec, generator.build(case, split))
                self.assertEqual(len(spec["layouts"]), count)
                for layout in spec["layouts"]:
                    key = json.dumps(layout["poses"], sort_keys=True)
                    self.assertNotIn(key, seen)
                    seen.add(key)
                    positions = {k: np.array(v["pos"]) for k, v in layout["poses"].items()}
                    for p in positions.values():
                        self.assertTrue(.3 < p[0] < .6 and -.2 < p[1] < .3)
                    if case == "spatial":
                        delta = positions["mustard_bottle"] - positions["raisin_box"]
                        # Identity robot base: left/right 45 degree cones require |dy| >= |dx|.
                        self.assertLess(abs(delta[1]), abs(delta[0]) - .05)
                        self.assertGreater(np.linalg.norm(delta[:2]), .10)
                    else:
                        self.assertGreater(np.linalg.norm(positions["rubiks_cube"][:2] - positions["banana"][:2]), .10)
                        for name in ("rubiks_cube", "banana"):
                            self.assertGreater(np.linalg.norm(positions[name][:2] - positions["bowl"][:2]), .18)

    def test_slot_mapping_is_independent_of_pool_lane_partition(self):
        for n_layouts, repeats in ((4, 4), (8, 2)):
            ids = [layouts.layout_index(s, n_layouts) for s in range(16)]
            self.assertEqual(np.bincount(ids).tolist(), [repeats] * n_layouts)
            for group in (0, 7, 29):
                self.assertEqual([layouts.layout_index(group * 16 + e, n_layouts) for e in range(16)], ids)
        with self.assertRaises(ValueError):
            layouts.allowed([{"id": "D1"}], "H1")
        with self.assertRaises(ValueError):
            layouts.allowed([{"id": "D1"}], "D1,D1")

    def test_invalid_physics_reset_is_not_a_policy_failure(self):
        good = {"objects": {"bowl": {"written_pos": [[.4, 0, .08]], "placed_pos": [[.4, 0, .08]],
                "settled_pos": [[.4, 0, .08]], "written_quat": [[1, 0, 0, 0]],
                "placed_quat": [[1, 0, 0, 0]], "settled_quat": [[1, 0, 0, 0]],
                "drift_last_ticks_m": [0.], "speed_mps": [0.]}}}
        layouts.validate_record(good)
        for field, value in (("placed_pos", [[.42, 0, .08]]), ("settled_pos", [[.42, 0, .08]]),
                             ("drift_last_ticks_m", [.006]), ("speed_mps", [.04]),
                             ("placed_quat", [[0, 1, 0, 0]]), ("settled_quat", [[2, 0, 0, 0]])):
            broken = copy.deepcopy(good)
            broken["objects"]["bowl"][field] = value
            with self.subTest(field=field), self.assertRaises(RuntimeError):
                layouts.validate_record(broken)


if __name__ == "__main__":
    unittest.main()
