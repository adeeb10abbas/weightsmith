"""Synthetic CPU tests for pre-heldout candidate selection; no policy execution."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

spec = importlib.util.spec_from_file_location("candidate_selector", Path(__file__).with_name("select_candidate.py"))
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


class CandidateSelectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.here = self.root / "source"
        self.here.mkdir()
        self.adaptation = self.root / "temporal-adaptation"
        self.protocol = {"cases": {"temporal": {"cells": [
            {"id": "original", "task": "RSSCubeThenBananaTask", "goal": "cube_then_banana", "prompt": "repair request"},
            {"id": "competing_reverse", "task": "RSSBananaThenCubeTask", "goal": "banana_then_cube", "prompt": "known preservation request", "role": "primary_preservation"},
            {"id": "unseen_before", "task": "RSSCubeThenBananaTask", "goal": "cube_then_banana", "prompt": "HELDOUT PHRASE NEVER SELECTED"}]}}}
        dump(self.here / "protocol.json", self.protocol)
        dump(self.here / "runtime/layouts/temporal_development.json", {"case": "temporal", "split": "development", "layouts": [{"id": f"T-D{i:02d}"} for i in range(1, 5)]})
        self.source = self.here / "worker.py"
        self.source.write_text("# synthetic source\n")
        for p in (patch.object(m, "HERE", self.here), patch.object(m, "runtime_sources", return_value=[self.source])):
            p.start(); self.addCleanup(p.stop)

    def make_nomination(self, mses=(.2, .05, .08)):
        a = self.adaptation
        dump(a / "spec.json", {"case": "temporal", "split": "development"})
        dataset = a / "lora-dataset/dataset.json"
        dump(dataset, {"split": "development", "spec_sha256": m.sha(a / "spec.json")})
        source_identity = {str(self.source): m.sha(self.source)}
        dump(a / "lora-sweep/identity.json", source_identity)
        dump(a / "source-freeze.json", {"sources": source_identity})
        base = self.root / "base"
        base.mkdir()
        entries = []
        names = ("initial-1e-3", "lr-3e-5", "lr-1e-4")
        for index, (cid, rate, mse) in enumerate(zip(names, (.001, .00003, .0001), mses)):
            fit = a / "lora-fit" if index == 0 else a / "lora-sweep" / cid
            workspace = fit / "workspace"
            (workspace / "edits").mkdir(parents=True)
            (workspace / "base").symlink_to(base, target_is_directory=True)
            tensors = {}
            for layer in m.LAYERS:
                file = f"layer-{layer}.npy"
                tensors[f"language.blocks.{layer:02d}.mlp.linear"] = {"file": file,
                    "model_path": ["PaliGemma", "llm", "layers", "mlp", "linear"], "scan_index": layer}
                np.save(workspace / "edits" / file, np.full((2, 2), layer + index, np.float32))
            dump(workspace / "workspace.json", {"tensors": tensors})
            training = {"dataset_manifest_sha256": m.sha(dataset), "rank": 32, "layers": m.LAYERS,
                        "steps": 200, "learning_rate": rate, "seed": 17, "weight_decay": 0,
                        "final_validation_actions": {"by_role": {role: {"robot_action_mse": mse} for role in ("repair", "preservation")}},
                        "workspace_validation": {"edited_tensors": {name: {} for name in tensors}}}
            training_path = fit / "training.json"
            dump(training_path, training)
            artifacts = {str(p): m.sha(p) for p in m.workspace_files(workspace)}
            seal_path = a / "lora-sweep/initial-seal.json" if index == 0 else fit / "seal.json"
            dump(seal_path, {"candidate_id": cid, "case": "temporal", "workspace_artifacts": artifacts,
                             "artifacts": {**artifacts, str(training_path): m.sha(training_path)},
                             "source_identity_sha256": m.sha(a / "lora-sweep/identity.json"),
                             "dataset_manifest_sha256": m.sha(dataset)})
            entries.append({"candidate_id": cid, "learning_rate": rate, "fit_path": str(fit), "workspace": str(workspace),
                            "training_path": str(training_path), "training_sha256": m.sha(training_path),
                            "balanced_validation_action_mse": mse, "stage": "initial" if index == 0 else "new",
                            "seal_path": str(seal_path), "seal_sha256": m.sha(seal_path), "workspace_artifacts": artifacts,
                            "status": "complete"})
        best = min(entries[1:], key=lambda x: (x["balanced_validation_action_mse"], x["learning_rate"]))
        nom = {"case": "temporal", "status": "complete", "split": "development", "dataset_path": str(dataset.parent),
               "dataset_manifest_sha256": m.sha(dataset), "spec_path": str(a / "spec.json"), "spec_sha256": m.sha(a / "spec.json"),
               "source_freeze_path": str(a / "source-freeze.json"), "source_freeze_sha256": m.sha(a / "source-freeze.json"),
               "source_identity_path": str(a / "lora-sweep/identity.json"), "source_identity_sha256": m.sha(a / "lora-sweep/identity.json"),
               "selected_new_candidate_id": best["candidate_id"], "closed_loop_candidate_ids": [names[0], best["candidate_id"]],
               "candidates": entries, "costs": {"all_three_fit_steps": 600}, "failed_attempts": []}
        dump(a / "lora-sweep/nomination.json", nom)
        return nom

    def evaluate(self, plan, counts):
        root = self.root / "development-evaluation"
        root.mkdir()
        dump(root / "plan.json", plan)
        dump(root / "status.json", {"split": "development", "cohort_id": plan["cohort_id"], "status": "complete", "completed_batches": 4, "completed_episodes": 64})
        frozen = m.read(plan["global_freeze"]["path"])
        dump(root / "identity.json", frozen["artifacts"])
        for number, b in enumerate(plan["batches"]):
            batch = root / f"batch-{number:02d}-{b['cell']}-{b['condition']}"
            batch.mkdir()
            successes = [i < counts[b["cell"], b["condition"]] for i in range(16)]
            layouts = [f"T-D{i % 4 + 1:02d}" for i in range(16)]
            metrics = batch / "metrics_r0.json"
            dump(metrics, {"selected_goal": b["goal"], "strict_final": {b["goal"]: successes}, "layout": {"layout_ids": layouts}})
            rows = []
            for i in range(16):
                slot, seed = b["slot_start"] + i, 1000 + i
                obs = batch / f"obs_r0_e{i}.npz"
                np.savez(obs, seed=np.array([seed]))
                rows.append({"slot": slot, "policy_seed": seed, "layout_id": layouts[i], "split": "development", "case": "temporal",
                             "cohort_id": plan["cohort_id"], "cell": b["cell"], "condition": b["condition"], "inference_batching": "fixed16",
                             "task_name": b["task"], "input_prompt": b["input_prompt"], "effective_prompt": b["effective_prompt"],
                             "selected_goal": b["goal"], "ok": True, "success": successes[i], "strict_final": {b["goal"]: successes[i]},
                             "env": i, "obs": str(obs), "obs_sha256": m.sha(obs), "metrics": str(metrics), "metrics_sha256": m.sha(metrics)})
            dump(batch / "episodes.json", rows)
            dump(batch / "completed.json", {"batch": number, "cell": b["cell"], "condition": b["condition"], "n": 16,
                                             "episodes_sha256": m.sha(batch / "episodes.json"), "successes": sum(successes),
                                             "server": json.dumps({"workspace": b["workspace"], "fingerprint": b["server_fingerprint"]}),
                                             "wall_seconds": 3., "execution_seconds": 2., "queue_seconds": 1.})
        return root

    def counts(self, original_initial=8, preserve_initial=8, original_new=10, preserve_new=9):
        return {("original", "lora_initial"): original_initial, ("primary_preservation", "lora_initial"): preserve_initial,
                ("original", "lora_nominee"): original_new, ("primary_preservation", "lora_nominee"): preserve_new}

    def test_plan_has_only_declared_development_goals_and_paired_fixed_slots(self):
        self.make_nomination()
        plan = m.make_plan(self.adaptation, self.root / "plan")
        self.assertEqual(plan["split"], "development")
        self.assertEqual(plan["inference_batching"], "fixed16")
        self.assertEqual([b["condition"] for b in plan["batches"]], ["lora_initial", "lora_nominee", "lora_nominee", "lora_initial"])
        self.assertEqual(sum(b["n"] for b in plan["batches"]), 64)
        self.assertEqual({b["slot_start"] for b in plan["batches"]}, {122000})
        self.assertNotIn("HELDOUT PHRASE", json.dumps(plan))
        frozen = m.read(plan["global_freeze"]["path"])
        self.assertEqual(frozen["plan_sha256"]["temporal"], m.payload_sha(plan))
        self.assertEqual(sum("/workspace/edits/" in p for p in frozen["artifacts"]), 27)

    def test_export_uses_success_rule_and_preserves_all_fits(self):
        self.make_nomination()
        plan = m.make_plan(self.adaptation, self.root / "plan")
        evaluation = self.evaluate(plan, self.counts())
        initial_file = self.adaptation / "lora-fit/workspace/edits/layer-8.npy"
        before = m.sha(initial_file)
        result = m.select(self.adaptation, evaluation)
        self.assertEqual(result["selected_candidate_id"], "lr-3e-5")
        self.assertFalse(result["heldout_used"])
        self.assertEqual(result["all_training_costs"]["all_three_fit_steps"], 600)
        self.assertEqual(m.sha(initial_file), before)
        selected = self.adaptation / "lora-selected"
        m.check_artifacts(m.read(selected / "seal.json")["artifacts"])
        self.assertTrue((selected / "workspace/base").is_symlink())
        self.assertEqual(m.sha(selected / "training.json"), result["selected_training_sha256"])
        with self.assertRaises(FileExistsError):
            m.select(self.adaptation, evaluation)

    def test_success_outweighs_action_mse(self):
        self.make_nomination()
        plan = m.make_plan(self.adaptation, self.root / "plan")
        evaluation = self.evaluate(plan, self.counts(original_initial=16, preserve_initial=16))
        nom, candidates, _ = m.nomination(self.adaptation)
        winner, _ = m.verify_evaluation(evaluation, self.adaptation, nom, candidates)
        self.assertEqual(winner["candidate_id"], "initial-1e-3")

    def test_success_and_mse_tie_uses_lower_learning_rate(self):
        self.make_nomination(mses=(.2, .2, .3))
        plan = m.make_plan(self.adaptation, self.root / "plan")
        evaluation = self.evaluate(plan, self.counts(original_new=8, preserve_new=8))
        nom, candidates, _ = m.nomination(self.adaptation)
        winner, _ = m.verify_evaluation(evaluation, self.adaptation, nom, candidates)
        self.assertEqual(winner["candidate_id"], "lr-3e-5")

    def test_heldout_row_or_mismatched_seed_is_rejected(self):
        self.make_nomination()
        plan = m.make_plan(self.adaptation, self.root / "plan")
        evaluation = self.evaluate(plan, self.counts())
        nom, candidates, _ = m.nomination(self.adaptation)
        batch = evaluation / "batch-01-original-lora_nominee"
        rows = m.read(batch / "episodes.json")
        rows[0]["split"] = "heldout"
        dump(batch / "episodes.json", rows)
        receipt = m.read(batch / "completed.json")
        receipt["episodes_sha256"] = m.sha(batch / "episodes.json")
        dump(batch / "completed.json", receipt)
        with self.assertRaisesRegex(ValueError, "declared development"):
            m.verify_evaluation(evaluation, self.adaptation, nom, candidates)
        rows[0]["split"] = "development"
        rows[0]["policy_seed"] += 1
        np.savez(rows[0]["obs"], seed=np.array([rows[0]["policy_seed"]]))
        rows[0]["obs_sha256"] = m.sha(rows[0]["obs"])
        dump(batch / "episodes.json", rows)
        receipt["episodes_sha256"] = m.sha(batch / "episodes.json")
        dump(batch / "completed.json", receipt)
        with self.assertRaisesRegex(ValueError, "paired slots"):
            m.verify_evaluation(evaluation, self.adaptation, nom, candidates)

    def test_changed_weights_or_wrong_loaded_workspace_is_rejected(self):
        self.make_nomination()
        plan = m.make_plan(self.adaptation, self.root / "plan")
        evaluation = self.evaluate(plan, self.counts())
        nom, candidates, _ = m.nomination(self.adaptation)
        batch = evaluation / "batch-00-original-lora_initial"
        receipt = m.read(batch / "completed.json")
        receipt["server"] = {"workspace": str(self.root / "other"), "fingerprint": "other"}
        dump(batch / "completed.json", receipt)
        with self.assertRaisesRegex(ValueError, "Loaded model"):
            m.verify_evaluation(evaluation, self.adaptation, nom, candidates)
        tensor = self.adaptation / "lora-fit/workspace/edits/layer-8.npy"
        np.save(tensor, np.zeros((2, 2), np.float32))
        with self.assertRaisesRegex(ValueError, "differs from nomination"):
            m.nomination(self.adaptation)

    def test_nominee_cannot_be_changed_after_seeing_rollouts(self):
        nom = self.make_nomination()
        nom["selected_new_candidate_id"] = "lr-1e-4"
        nom["closed_loop_candidate_ids"][1] = "lr-1e-4"
        dump(self.adaptation / "lora-sweep/nomination.json", nom)
        with self.assertRaisesRegex(ValueError, "validation-MSE rule"):
            m.nomination(self.adaptation)


if __name__ == "__main__":
    unittest.main()
