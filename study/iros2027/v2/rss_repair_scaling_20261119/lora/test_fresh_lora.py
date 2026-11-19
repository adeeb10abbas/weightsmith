"""CPU guards for scientific data boundaries and action-label correctness."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fresh_lora as F


class PhysicalPolicy:
    def _input_transform(self, value):
        assert value["prompt"] == "original request"
        return {"state": np.pad(value["observation/joint_position"], (0, 25)),
                "actions": np.pad(value["actions"], ((0, 0), (0, 24)))}

    def _output_transform(self, value):
        action = value["actions"][:, :8].copy()
        action[:, :7] += value["state"][None, :7]
        return {"actions": action}


class FreshLoRATest(unittest.TestCase):
    def test_absolute_joints_and_gripper_survive_roundtrip(self):
        raw = {"base_img": np.zeros((2, 2, 3), dtype=np.uint8),
               "wrist_img": np.zeros((2, 2, 3), dtype=np.uint8),
               "joint": np.arange(7, dtype=np.float32) / 10,
               "gripper": np.array([.4], dtype=np.float32)}
        target = np.tile(np.r_[raw["joint"] + .05, np.float32(.7)], (15, 1)).astype(np.float32)
        original = copy.deepcopy(raw)
        inputs, normalized, physical, error = F.normalize_physical(PhysicalPolicy(), raw, target, "original request")
        np.testing.assert_allclose(normalized[:, :7], .05, atol=1e-7)
        np.testing.assert_array_equal(normalized[:, 7], target[:, 7])
        np.testing.assert_array_equal(normalized[:, 8:], 0)
        np.testing.assert_allclose(physical, target, atol=1e-7)
        self.assertLessEqual(error, F.ROUNDTRIP_TOLERANCE)
        self.assertNotIn("actions", inputs)
        for key in raw:
            np.testing.assert_array_equal(raw[key], original[key])

    def test_bad_normalization_is_rejected(self):
        class BrokenPolicy(PhysicalPolicy):
            def _output_transform(self, value):
                out = super()._output_transform(value)
                out["actions"][:, 7] += .1
                return out
        raw = {"base_img": np.zeros((2, 2, 3), dtype=np.uint8), "wrist_img": np.zeros((2, 2, 3), dtype=np.uint8),
               "joint": np.zeros(7, dtype=np.float32), "gripper": np.zeros(1, dtype=np.float32)}
        with self.assertRaisesRegex(ValueError, "roundtrip"):
            F.normalize_physical(BrokenPolicy(), raw, np.zeros((15, 8), dtype=np.float32), "original request")

    def fixture(self, root):
        obs = root / "obs_r0_e0.npz"
        np.savez(obs, seed=np.array([1, 2, 3]))
        episodes = root / "episodes.json"
        episodes.write_text(json.dumps([{"obs": str(obs), "strict_final": {"cube_then_banana": True}}]))
        base = {"obs_path": str(obs), "split": "train", "pair_id": "pair-1", "recorded_prompt": "clear",
                "original_prompt": "original", "teacher_prompt": "clear", "live_context": {"source_episodes_path": str(episodes)}}
        spec = {"cohort_id": "new-study", "split": "development", "arm": "full_trajectory", "correction_count": 1,
                "items": [{**base, "item_id": f"decision-{k}", "decision_index": k} for k in range(3)]}
        pair = {"pair_id": "pair-1", "eligible": True, "selected_decision": 1,
                "teacher": {"obs": str(obs), "obs_sha256": F.sha(obs), "effective_prompt": "clear"},
                "original": {"obs": str(obs), "obs_sha256": F.sha(obs), "effective_prompt": "original"}}
        manifest = root / "source_manifest.json"
        manifest.write_text(json.dumps({"cohort_id": "new-study", "split": "development",
                                       "correction_budgets_available": [1], "pairs": [pair], "artifacts": {}}))
        spec.update(source_manifest=str(manifest), source_manifest_sha256=F.sha(manifest), distinct_pair_ids=["pair-1"])
        path = root / "spec.json"
        path.write_text(json.dumps(spec))
        return path, spec, episodes

    def parse(self, path, spec, arm="full_trajectory"):
        path.write_text(json.dumps(spec))
        # Existing N.validate_live_context has its own receipt/hash/context tests.
        # These tests isolate the extra full-successful-trajectory requirements.
        with patch.object(F.N, "validate_live_context", return_value={}):
            return F.read_spec(path, arm)

    def test_complete_successful_episode_is_accepted(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            _, rows, _, count = self.parse(path, spec)
            self.assertEqual(count, 1)
            self.assertEqual(len(rows), 3)

    def test_failure_cannot_be_called_successful_trajectory(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, episodes = self.fixture(Path(temp))
            rows = json.loads(episodes.read_text())
            rows[0]["strict_final"]["cube_then_banana"] = False
            episodes.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, "did not succeed"):
                self.parse(path, spec)

    def test_truncated_trajectory_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            spec["items"].pop()
            with self.assertRaisesRegex(ValueError, "every recorded decision"):
                self.parse(path, spec)

    def test_duplicate_chunk_and_validation_source_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            duplicate = copy.deepcopy(spec)
            duplicate["items"].append(duplicate["items"][0])
            with self.assertRaisesRegex(ValueError, "duplicate"):
                self.parse(path, duplicate)
            spec["items"][0]["split"] = "validation"
            with self.assertRaisesRegex(ValueError, "Only training"):
                self.parse(path, spec)

    def test_heldout_source_rejected_before_replay(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            spec["items"][0]["obs_path"] = str(Path(temp) / "heldout" / "obs_r0_e0.npz")
            with self.assertRaisesRegex(ValueError, "Held-out"):
                self.parse(path, spec)

    def test_correction_alias_accepted_without_inventing_validation_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            spec["arm"] = "correction"
            spec["items"] = [spec["items"][1]]
            spec["items"][0]["recorded_prompt"] = "original"
            _, rows, _, count = self.parse(path, spec, "matched_decision")
            self.assertEqual((len(rows), count), (1, 1))

    def test_fixed_budget_sees_every_chunk_twice(self):
        for n in (1, 60, 180, 360, 720):
            steps = F.planned_steps(n)
            order = F.training_order(n, steps)
            frequencies = np.bincount(order, minlength=n)
            self.assertGreaterEqual(frequencies.min(), 2)
            self.assertEqual(steps, len(order))
            self.assertEqual(order, F.training_order(n, steps))

    def test_changed_source_manifest_or_prefix_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            spec["distinct_pair_ids"] = ["unselected-pair"]
            with self.assertRaisesRegex(ValueError, "nested prefix"):
                self.parse(path, spec)
            spec["distinct_pair_ids"] = ["pair-1"]
            Path(spec["source_manifest"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                self.parse(path, spec)

    def test_unselected_failure_decision_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            spec["arm"] = "correction"
            spec["items"] = [spec["items"][0]]
            spec["items"][0]["recorded_prompt"] = "original"
            with self.assertRaisesRegex(ValueError, "frozen selector"):
                self.parse(path, spec, "matched_decision")

    def test_unrecorded_student_instruction_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path, spec, _ = self.fixture(Path(temp))
            spec["items"][0]["original_prompt"] = "unseen evaluation instruction"
            with self.assertRaisesRegex(ValueError, "instruction differs"):
                self.parse(path, spec)


if __name__ == "__main__":
    unittest.main()
