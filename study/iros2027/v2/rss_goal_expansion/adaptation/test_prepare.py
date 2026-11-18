"""Regression tests for actual fixed16 versus shrinking active membership."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import unittest

HERE = Path(__file__).resolve().parent


def module(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f"{name}.py")
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


native = module("native_targets")
selector = module("prepare")


def item(size):
    members = [{"env_id": e, "obs_path": f"/tmp/fixed16-fixture/obs_r0_e{e}.npz",
                "decision_index": 2, "obs_sha256": str(e) * 64} for e in range(size)]
    context = {"members": members, "source_run_id": 0, "source_step": 30,
        "source_batch_observation_sha256": {m["obs_path"]: m["obs_sha256"] for m in members},
        "inference_batching": "fixed16"}
    return {"live_batch_size": size, "live_batch_bucket": 16, "live_batch_members": members,
        "source_env_id": 0, "source_run_id": 0, "source_step": 30,
        "obs_path": members[0]["obs_path"], "decision_index": 2, "live_context": context,
        "inference_batching": "fixed16", "live_context_sha256": selector.canonical(context)}


class FixedKernelTests(unittest.TestCase):
    def test_active_membership_does_not_change_executed_bucket(self):
        for n in (1, 3, 8, 9, 16):
            with self.subTest(n=n):
                self.assertEqual(len(native.validate_live_metadata(item(n))["members"]), n)

    def test_dynamic_bucket_is_rejected_for_fixed16_source(self):
        for n in (1, 3, 8):
            record = item(n)
            record["live_batch_bucket"] = 1 << (n - 1).bit_length()
            with self.assertRaisesRegex(ValueError, "bucket"):
                native.validate_live_metadata(record)

    def test_unspecified_regime_is_not_assumed_fixed(self):
        record = item(16)
        del record["inference_batching"]
        with self.assertRaises(ValueError):
            native.validate_live_metadata(record)

    def test_changed_membership_is_detected_even_with_valid_bucket(self):
        record = item(3)
        record["live_batch_members"] = copy.deepcopy(record["live_batch_members"])
        record["live_batch_members"][1]["decision_index"] = 3
        with self.assertRaises(ValueError):
            native.validate_live_metadata(record)

    def test_selection_ignores_success_and_uses_distinct_replicas(self):
        ids = ["D01", "D02", "D03", "D04"]
        rows = [{"slot": s, "layout_id": ids[s % 4], "success": s > 7} for s in range(16)]
        selected = selector.chosen_rows(list(reversed(rows)), ids, 2)
        self.assertEqual(sorted(x["slot"] for x in selected), list(range(8)))
        for row in rows:
            row["success"] = not row["success"]
        self.assertEqual(sorted(x["slot"] for x in selector.chosen_rows(rows, ids, 2)), list(range(8)))


if __name__ == "__main__":
    unittest.main()
