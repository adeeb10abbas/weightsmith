"""CPU integrity and real-transform checks; no policy success claims."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("historical_lora", Path(__file__).with_name("historical_lora.py"))
H = importlib.util.module_from_spec(spec); spec.loader.exec_module(H)


class HistoricalSourceTests(unittest.TestCase):
    def index(self):
        rows = [{} for _ in range(18)]
        for i, (seed, k) in enumerate(H.DECISIONS):
            item = {"episode": f"base.base.s{seed}", "k": k, "seed": seed*100+5,
                    "sim_step": k*15, "gripper_state": .3}
            rows[i] = {**item, "prompt": H.ORIGINAL}
            rows[12+i] = {**item, "prompt": H.TEACHER}
        return {"cand": "base", "items": rows}

    def test_exact_six_pairs_and_source_order(self):
        result = H.source_items(self.index())
        self.assertEqual([(r["episode"], r["decision_index"]) for r in result],
                         [(f"base.base.s{s}", k) for s, k in H.DECISIONS])
        self.assertEqual([r["teacher_row"] for r in result], list(range(12, 18)))

    def test_mismatched_noise_or_prompt_is_rejected(self):
        for field, value in [("seed", 1), ("prompt", H.ORIGINAL), ("sim_step", 999), ("k", 99)]:
            index = self.index(); index["items"][12][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                H.source_items(index)

    def test_nonbase_teacher_is_rejected(self):
        index = self.index(); index["cand"] = "edited"
        with self.assertRaises(ValueError): H.source_items(index)

    def test_native_normalization_preserves_absolute_joint_targets(self):
        # Use installed checkpoint normalization and actual OpenPI transforms,
        # without loading model tensors or invoking inference.
        from openpi import transforms as T
        from openpi.training import config, checkpoints
        from weightsmith import config as C
        cfg = config.get_config("pi05_droid_jointpos")
        data = cfg.data.create(cfg.assets_dirs, cfg.model)
        stats = checkpoints.load_norm_stats(C.CHECKPOINT / "assets", data.asset_id)
        policy = SimpleNamespace(
            _input_transform=T.compose([*data.data_transforms.inputs, T.Normalize(stats, use_quantiles=data.use_quantile_norm),
                                        *data.model_transforms.inputs]),
            _output_transform=T.compose([*data.model_transforms.outputs, T.Unnormalize(stats, use_quantiles=data.use_quantile_norm),
                                         *data.data_transforms.outputs]))
        raw = {"base_img": np.zeros((224, 224, 3), np.uint8), "wrist_img": np.zeros((224, 224, 3), np.uint8),
               "joint": np.array([.2, -.3, .1, -2., .1, 2., .4], np.float32), "gripper": np.array([.3], np.float32)}
        actions = np.broadcast_to(np.r_[raw["joint"], .7], (15, 8)).astype(np.float32).copy()
        actions[:, :7] += .025
        before = actions.copy()
        _, target, actual, error = H.normalized_teacher(policy, raw, actions)
        np.testing.assert_array_equal(actions, before)
        np.testing.assert_allclose(actual, actions, atol=H.ROUNDTRIP_TOLERANCE, rtol=0)
        np.testing.assert_array_equal(target[:, 8:], np.zeros((15, 24)))
        self.assertLessEqual(error, H.ROUNDTRIP_TOLERANCE)


if __name__ == "__main__": unittest.main()
