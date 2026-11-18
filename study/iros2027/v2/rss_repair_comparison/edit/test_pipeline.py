"""CPU-only scientific-contract checks; these are not policy evidence."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location("repair_pipeline", Path(__file__).with_name("pipeline.py"))
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rng = np.random.default_rng(7331)

    def manifest(self):
        items = []
        for i, (split, role) in enumerate((("train", "repair"), ("train", "preservation"), ("validation", "repair"))):
            path = self.root / f"episode-{i}.npz"
            np.savez(path, seed=np.array([31 + i]), step=np.array([0]),
                     base_img=np.zeros((1, 4, 4, 3), np.uint8), wrist_img=np.zeros((1, 4, 4, 3), np.uint8),
                     joint=np.zeros((1, 7), np.float32), gripper=np.zeros((1, 1), np.float32),
                     actions=np.zeros((1, 15, 8), np.float32))
            members = [{"env_id": 0, "obs_path": str(path), "decision_index": 0, "obs_sha256": p.sha(path)}]
            context = {"source_step": 0, "source_run_id": 0, "members": members,
                       "source_batch_observation_sha256": {str(path): p.sha(path)}}
            context_hash = p.hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            items.append({"obs_path": str(path), "decision_index": 0, "split": split, "role": role,
                          "source_env_id": 0, "source_run_id": 0, "source_step": 0,
                          "live_batch_size": 1, "live_batch_bucket": 1, "live_batch_members": members,
                          "live_context": context, "live_context_sha256": context_hash,
                          "original_prompt": "repair" if role == "repair" else "preserve",
                          "teacher_prompt": "clear" if role == "repair" else "preserve", "recorded_prompt": "clear" if role == "repair" else "preserve"})
        manifest = self.root / "spec.json"
        p.dump(manifest, {"cohort_id": "cpu-test-only", "split": "development", "items": items})
        return manifest

    def captures(self, prepared):
        meta = json.loads((prepared / "prepared.json").read_text())
        shape = (3, len(p.CAPTURE_LAYERS), 2, 17)
        original = {p.ARRAY_KEYS[0]: self.rng.normal(size=(*shape, 6)).astype(np.float32),
                    p.ARRAY_KEYS[1]: self.rng.normal(size=(*shape, 4)).astype(np.float32)}
        teacher = {k: v.copy() for k, v in original.items()}
        teacher[p.ARRAY_KEYS[1]][0] += self.rng.normal(size=teacher[p.ARRAY_KEYS[1]][0].shape)
        for which, arrays in (("original", original), ("teacher", teacher)):
            dest = prepared / "captures" / which
            dest.mkdir(parents=True)
            np.savez(dest / "acts.npz", **arrays, actions=np.zeros((3, 15, 8)), recorded_actions=np.zeros((3, 15, 8)))
            p.dump(dest / "index.json", {"vlm_layers": list(p.CAPTURE_LAYERS), "items": [
                {"episode": r["episode"], "k": r["decision_index"], "seed": r["policy_seed"], "prompt": r[f"{which}_prompt"]} for r in meta["records"]]})
        return original, teacher, meta

    def test_frozen_input_identity_and_episode_split(self):
        manifest = self.manifest()
        prepared = self.root / "prepared"
        p.prepare(argparse.Namespace(spec=str(manifest), output=str(prepared)))
        _, meta = p.load_prepared(prepared)
        self.assertEqual(meta["records"][0]["policy_seed"], 31)
        self.assertNotEqual(meta["records"][0]["original_input_sha256"], meta["records"][0]["teacher_input_sha256"])
        self.assertEqual(meta["records"][1]["original_input_sha256"], meta["records"][1]["teacher_input_sha256"])
        src = Path(meta["records"][0]["obs_path"])
        with src.open("ab") as f:
            f.write(b"mutation")
        with self.assertRaisesRegex(ValueError, "changed after freeze"):
            p.load_prepared(prepared)
        data = json.loads(manifest.read_text())
        data["items"][-1].update(obs_path=data["items"][0]["obs_path"], decision_index=1)
        p.dump(manifest, data)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            p.read_spec(manifest)

    def test_validation_exclusion_and_preservation(self):
        prepared = self.root / "prepared"
        p.prepare(argparse.Namespace(spec=str(self.manifest()), output=str(prepared)))
        original, teacher, meta = self.captures(prepared)
        before, groups = p.key_values(original, teacher, meta["records"], "global")
        teacher[p.ARRAY_KEYS[1]][2] += 1e6
        original[p.ARRAY_KEYS[0]][2] += 1e6
        after, _ = p.key_values(original, teacher, meta["records"], "global")
        self.assertEqual(len(groups), 1)
        for layer in p.LAYERS:
            for a, b in zip(before[layer], after[layer]):
                np.testing.assert_array_equal(a, b)
            K, V, H = before[layer]
            self.assertEqual(K.shape, (32, 6))
            self.assertEqual(H.shape, (32, 6))
            delta, stats = p.fit(K, V, H)
            A = np.concatenate((K, H))
            B = np.concatenate((V, np.zeros((32, 4))))
            lam = 1e-3 * np.sum(K * K) / len(K)
            effective_relative_ridge = lam / (np.sum(A * A) / len(A))
            np.testing.assert_allclose(delta, p.distill.closed_form_edit(A, B, effective_relative_ridge), atol=1e-12)
            self.assertLessEqual(stats["rank"], 32)

    def test_sparse_workspace_and_teacher_parity(self):
        manifest = self.manifest()
        prepared = self.root / "prepared"
        p.prepare(argparse.Namespace(spec=str(manifest), output=str(prepared)))
        _, _, meta = self.captures(prepared)
        base = self.root / "base-workspace"
        (base / "base").mkdir(parents=True)
        tensors = {}
        for layer in p.LAYERS:
            name = p.distill.MLP_DOWN.format(layer)
            file = f"layer-{layer}.npy"
            np.save(base / "base" / file, p.distill.bf16_round(self.rng.normal(size=(6, 4))))
            tensors[name] = {"file": file, "shape": [6, 4], "numel": 24}
        p.dump(base / "workspace.json", {"tensors": tensors, "total_parameters_and_buffers": 216,
                                         "editable_parameters": 216, "base_vlm_sha256": "synthetic-test-only"})
        dataset = self.root / "lora"
        dataset.mkdir()
        records = []
        for i, record in enumerate(meta["records"]):
            file = f"item-{i}.npz"
            np.savez(dataset / file, teacher_actions_physical=np.zeros((15, 8)),
                     live_replay_actions_physical=np.zeros((15, 8), np.float32),
                     recorded_actions_physical=np.zeros((15, 8), np.float32))
            records.append({**record, "file": file, "sha256": p.sha(dataset / file),
                            "live_replay_max_abs_error": 0., "live_replay_bit_exact": True})
        dataset_report = {**meta, "records": records, "live_replay_gate": {
            "mode": "exact_pool_bucket", "status": "pass", "tolerance": 1e-6, "items": len(records), "maximum_error": 0.}}
        p.dump(dataset / "dataset.json", dataset_report)
        parity = p.teacher_parity(prepared, meta, dataset, 1e-2)
        self.assertTrue(parity["all_bit_exact"])
        out = self.root / "edit"
        p.solve(argparse.Namespace(prepared=str(prepared), output=str(out), lora_dataset=str(dataset),
                                  base_workspace=str(base), pooling="global", hold_weight=1., parity_tolerance=1e-2))
        report = json.loads((out / "solve.json").read_text())
        self.assertFalse(report["validation_used_in_fit"])
        self.assertEqual(len(list((out / "workspace/edits").glob("*.npy"))), 9)
        self.assertEqual(report["teacher_action_parity"]["status"], "pass")
        self.assertEqual(report["replay_parity"]["mode"], "exact_pool_bucket")
        self.assertEqual(report["replay_parity"]["tolerance"], 1e-6)
        self.assertEqual(report["singleton_recorded_drift"]["status"], "diagnostic_only")
        for file in (out / "workspace/edits").glob("*.npy"):
            x = np.load(file)
            np.testing.assert_array_equal(x, p.distill.bf16_round(x))
        dataset_report["live_replay_gate"]["status"] = "missing"
        p.dump(dataset / "dataset.json", dataset_report)
        with self.assertRaisesRegex(ValueError, "exact live-pool"):
            p.certified_live_replay(prepared, meta, dataset)
        dataset_report["live_replay_gate"]["status"] = "pass"
        dataset_report["records"][0]["live_replay_max_abs_error"] = 0.02
        p.dump(dataset / "dataset.json", dataset_report)
        with self.assertRaisesRegex(ValueError, "every selected input"):
            p.certified_live_replay(prepared, meta, dataset)
        dataset_report["records"][0]["live_replay_max_abs_error"] = 0.
        p.dump(dataset / "dataset.json", dataset_report)
        with self.assertRaisesRegex(ValueError, "no threshold inflation"):
            p.teacher_parity(prepared, meta, dataset, .05)
        np.savez(dataset / records[0]["file"], teacher_actions_physical=np.ones((15, 8)))
        with self.assertRaisesRegex(ValueError, "changed"):
            p.teacher_parity(prepared, meta, dataset, 1e-2)

    def test_singleton_drift_is_not_an_acceptance_gate(self):
        prepared = self.root / "prepared"
        p.prepare(argparse.Namespace(spec=str(self.manifest()), output=str(prepared)))
        _, _, meta = self.captures(prepared)
        dest = prepared / "captures/teacher/acts.npz"
        with np.load(dest) as z:
            arrays = {k: z[k].copy() for k in z.files}
        arrays["actions"][:] = .025
        np.savez(dest, **arrays)
        report = p.singleton_drift_report(prepared, meta)
        self.assertEqual(report["status"], "diagnostic_only")
        self.assertFalse(report["used_as_acceptance_gate"])
        self.assertGreater(report["max_abs_error"], .01)

    def test_preservation_rejects_different_teacher(self):
        manifest = self.manifest()
        data = json.loads(manifest.read_text())
        data["items"][1]["teacher_prompt"] = "change the goal"
        p.dump(manifest, data)
        with self.assertRaisesRegex(ValueError, "Preservation"):
            p.read_spec(manifest)

    def test_preservation_mean_penalty_is_count_normalized(self):
        K = self.rng.normal(size=(32, 6))
        V = self.rng.normal(size=(32, 4))
        H = self.rng.normal(size=(64, 6))
        delta, stats = p.fit(K, V, H)
        A = np.concatenate((K, H / np.sqrt(2)))
        B = np.concatenate((V, np.zeros((64, 4))))
        lam = 1e-3 * np.sum(K * K) / len(K)
        effective_relative_ridge = lam / (np.sum(A * A) / len(A))
        np.testing.assert_allclose(delta, p.distill.closed_form_edit(A, B, effective_relative_ridge), atol=1e-12)
        self.assertEqual(stats["preservation_items"], 2)
        self.assertAlmostEqual(stats["preservation_key_scale"], 1 / np.sqrt(2))
        repeated, more = p.fit(K, V, np.concatenate((H, H, H)))
        np.testing.assert_allclose(repeated, delta, atol=1e-12)
        self.assertEqual(more["preservation_items"], 6)
        self.assertEqual(more["ridge_absolute"], stats["ridge_absolute"])


if __name__ == "__main__":
    unittest.main()
