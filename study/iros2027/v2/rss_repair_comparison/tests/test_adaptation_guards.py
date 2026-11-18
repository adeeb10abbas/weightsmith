"""CPU-only provenance checks, with synthetic artifacts and no policy execution."""
import copy
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

HERE = Path(__file__).resolve().parents[1]


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


selector = module("comparison_selector", HERE / "prepare_shared.py")
planner = module("comparison_planner", HERE / "make_eval_plan.py")


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


class GuardTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.here = self.root / "source"
        self.here.mkdir()
        self.protocol = json.loads((HERE / "protocol.json").read_text())
        shutil.copyfile(HERE / "rewrites.py", self.here / "rewrites.py")
        self.protocol["router_sha256"] = planner.sha(self.here / "rewrites.py")
        dump(self.here / "protocol.json", self.protocol)
        self.patches = [patch.object(selector, "HERE", self.here), patch.object(planner, "HERE", self.here)]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.counter = 0
        self.selections = {}
        original_loader = planner.load_module

        def load(name, path):
            if name != "heldout_candidate_selector":
                return original_loader(name, path)
            # The selector's own CPU suite exercises nomination and rollout
            # verification. Here we isolate the factory's binding/export guards.
            return SimpleNamespace(
                RULE="synthetic deterministic development rule",
                nomination=lambda adaptation: self.selections[str(adaptation)]["nomination"],
                verify_evaluation=lambda root, adaptation, nom, candidates: self.selections[str(adaptation)]["election"],
                workspace_files=lambda workspace: [workspace / "workspace.json", *sorted((workspace / "edits").glob("*.npy"))],
            )

        mocked_loader = patch.object(planner, "load_module", side_effect=load)
        mocked_loader.start()
        self.addCleanup(mocked_loader.stop)

    def cohort(self, case, cells=None, *, early_cube=False, prompt_override=None):
        self.counter += 1
        owner = self.root / f"cohort-{self.counter}"
        cells = cells or list(selector.TASKS[case])
        plan = {"split": "development", "case": case, "cohort_id": f"development-{self.counter}", "batches": []}
        for number, cell in enumerate(cells):
            task = selector.TASKS[case][cell]
            if early_cube and cell == "cube_only":
                task = "RSSCubeOnlyTask"
            prompt = self.protocol["cases"][case].get(cell, f"development {cell}")
            prompt = (prompt_override or {}).get(cell, prompt)
            declared = {"cell": cell, "condition": "base", "task": task, "input_prompt": prompt,
                        "effective_prompt": prompt, "slot_start": 120000, "n": 16, "workspace": None}
            plan["batches"].append(declared)
            batch = owner / f"batch-{number:02d}-{cell}-base"
            batch.mkdir(parents=True)
            rows = []
            for i in range(16):
                obs = batch / f"obs_r0_e{i}.npz"
                np.savez(obs, seed=np.arange(9) + 100 * i, step=np.arange(9) * 15)
                rows.append({"slot": 120000 + i, "layout_id": f"{'T' if case == 'temporal' else 'S'}-D{i % 4 + 1:02d}",
                             "env": i, "run": 0,
                             "case": case, "cohort_id": plan["cohort_id"], "split": "development", "cell": cell,
                             "condition": "base", "task_name": task, "input_prompt": prompt, "effective_prompt": prompt,
                             "ok": True, "success": bool(i % 2), "obs": str(obs), "obs_sha256": selector.sha(obs)})
            dump(batch / "episodes.json", rows)
            dump(batch / "completed.json", {"batch": number, "cell": cell, "condition": "base", "n": 16,
                                             "episodes_sha256": selector.sha(batch / "episodes.json")})
        dump(owner / "plan.json", plan)
        dump(owner / "status.json", {"status": "complete", "split": "development", "cohort_id": plan["cohort_id"],
                                      "completed_batches": len(cells)})
        return owner

    def test_selector_uses_owned_development_and_full_horizon_replacement(self):
        owner = self.cohort("temporal", early_cube=True)
        with self.assertRaisesRegex(ValueError, "full-horizon"):
            selector.prepare(owner, self.root / "bad.json")
        replacement = self.cohort("temporal", ["cube_only"])
        out = self.root / "shared.json"
        selector.prepare(owner, out, [replacement / "batch-00-cube_only-base"])
        spec = json.loads(out.read_text())
        self.assertEqual(len(spec["items"]), 32)
        self.assertEqual(sum(r["split"] == "train" for r in spec["items"]), 24)
        self.assertEqual(sum(r["role"] == "repair" for r in spec["items"]), 16)
        self.assertEqual({r["source_slot"] for r in spec["items"]}, set(range(120000, 120004)))
        self.assertTrue(all("cohort-2" in r["obs_path"] for r in spec["items"] if r["source_cell"] == "cube_only"))
        self.assertTrue(all(r["live_batch_size"] == 16 and r["live_batch_bucket"] == 16 for r in spec["items"]))
        self.assertTrue(all(r["live_context_sha256"] == selector.canonical_sha256(r["live_context"]) for r in spec["items"]))

    def test_live_batch_membership_shrinks_with_recorded_request_steps(self):
        owner = self.cohort("spatial", ["opposite_left"])
        batch = owner / "batch-00-opposite_left-base"
        rows = json.loads((batch / "episodes.json").read_text())
        for row in rows[7:]:
            path = Path(row["obs"])
            np.savez(path, seed=np.array([100]), step=np.array([0]))
            row["obs_sha256"] = selector.sha(path)
        dump(batch / "episodes.json", rows)
        receipt = json.loads((batch / "completed.json").read_text())
        receipt["episodes_sha256"] = selector.sha(batch / "episodes.json")
        dump(batch / "completed.json", receipt)
        index = selector.live_context_index(batch, rows)
        first = selector.live_context(index, rows[0], 0)
        later = selector.live_context(index, rows[0], 2)
        self.assertEqual((first["live_batch_size"], first["live_batch_bucket"]), (16, 16))
        self.assertEqual((later["live_batch_size"], later["live_batch_bucket"]), (7, 8))
        self.assertEqual([m["env_id"] for m in later["live_batch_members"]], list(range(7)))
        self.assertEqual(later["source_step"], 30)
        self.assertEqual(len(later["live_context"]["source_batch_observation_sha256"]), 16)
        np.savez(batch / "obs_r0_e99.npz", seed=np.array([1]), step=np.array([0]))
        with self.assertRaisesRegex(ValueError, "unreceipted"):
            selector.live_context_index(batch, rows)

    def test_generic_spatial_replacement_keeps_declared_clarifier(self):
        owner = self.cohort("spatial")
        replacement = self.cohort("spatial", ["opposite_left"], prompt_override={"opposite_left": "clearer left instruction"})
        out = self.root / "shared.json"
        selector.prepare(owner, out, [replacement / "batch-00-opposite_left-base"])
        rows = json.loads(out.read_text())["items"]
        self.assertEqual({r["teacher_prompt"] for r in rows if r["source_cell"] == "opposite_left"}, {"clearer left instruction"})
        self.assertTrue(all(r["original_prompt"] == r["teacher_prompt"] for r in rows if r["role"] == "preservation"))

    def test_replacement_rejects_heldout_owner_case_and_rows(self):
        owner = self.cohort("spatial")
        replacement = self.cohort("spatial", ["opposite_left"])
        batch = replacement / "batch-00-opposite_left-base"
        plan = json.loads((replacement / "plan.json").read_text())
        for field, value, error in (("split", "heldout", "development plan"), ("case", "temporal", "case mismatch")):
            changed = {**plan, field: value}
            dump(replacement / "plan.json", changed)
            with self.assertRaisesRegex(ValueError, error):
                selector.prepare(owner, self.root / "bad.json", [batch])
        dump(replacement / "plan.json", plan)
        rows = json.loads((batch / "episodes.json").read_text())
        rows[0]["split"] = "heldout"
        dump(batch / "episodes.json", rows)
        receipt = json.loads((batch / "completed.json").read_text())
        receipt["episodes_sha256"] = selector.sha(batch / "episodes.json")
        dump(batch / "completed.json", receipt)
        with self.assertRaisesRegex(ValueError, "Episode identity"):
            selector.prepare(owner, self.root / "bad.json", [batch])

    def test_replacement_rejects_unknown_layout_and_foreign_observation(self):
        replacement = self.cohort("spatial", ["opposite_left"])
        batch = replacement / "batch-00-opposite_left-base"
        rows = json.loads((batch / "episodes.json").read_text())
        original = copy.deepcopy(rows)
        rows[0]["layout_id"] = "S-H01"
        for altered, message in ((rows, "development layouts"),
                                 ([{**r, "obs": str(self.root / "foreign.npz")} if i == 0 else r for i, r in enumerate(original)], "owned")):
            dump(batch / "episodes.json", altered)
            receipt = json.loads((batch / "completed.json").read_text())
            receipt["episodes_sha256"] = selector.sha(batch / "episodes.json")
            dump(batch / "completed.json", receipt)
            with self.assertRaisesRegex(ValueError, message):
                selector.batch_rows(batch, "spatial", replacement=True)

    def adaptation(self, case):
        out = self.root / "runs" / f"{case}-adaptation"
        base = self.root / f"{case}-base"
        base.mkdir()
        manifest = {"base_vlm_sha256": "test-base", "tensors": {}}
        for i in range(8, 17):
            name, file = f"language.blocks.{i:02d}.mlp.linear", f"layer-{i}.npy"
            manifest["tensors"][name] = {"file": file}
            (base / "base").mkdir(exist_ok=True)
            (base / "base" / file).write_bytes(b"synthetic base")
        dump(base / "workspace.json", manifest)
        spec = {"case": case, "cohort_id": "development-adaptation", "split": "development", "source_observation_sha256": {"obs": "fake"}}
        dump(out / "spec.json", spec)
        dump(out / "status.json", {"case": case, "status": "complete"})
        dataset = {"cohort_id": spec["cohort_id"], "split": "development", "spec_sha256": planner.sha(out / "spec.json"),
                   "source_observation_sha256": spec["source_observation_sha256"], "base_workspace": str(base),
                   "base_workspace_manifest_sha256": planner.sha(base / "workspace.json")}
        dump(out / "lora-dataset/dataset.json", dataset)
        dataset_sha = planner.sha(out / "lora-dataset/dataset.json")
        fit = {"cohort_id": spec["cohort_id"], "dataset_manifest_sha256": dataset_sha,
               "rank": 32, "layers": list(range(8, 17)), "steps": 200, "learning_rate": .001, "seed": 17, "weight_decay": 0}
        dump(out / "lora-fit/training.json", fit)
        edit = {"cohort_id": spec["cohort_id"], "spec_sha256": dataset["spec_sha256"],
                "teacher_action_parity": {"status": "pass", "dataset_sha256": dataset_sha, "tolerance": .01},
                "replay_parity": {"status": "pass", "mode": "exact_pool_bucket", "tolerance": 1e-6,
                                  "dataset_sha256": dataset_sha},
                "source_observation_sha256": spec["source_observation_sha256"], "base_workspace": str(base),
                "base_manifest_sha256": dataset["base_workspace_manifest_sha256"], "rank_cap": 32, "pooling": "global",
                "hold_weight": 1, "validation_used_in_fit": False, "layers": []}
        for method in ("edit-fit", "lora-fit"):
            workspace = out / method / "workspace"
            dump(workspace / "workspace.json", manifest)
            (workspace / "edits").mkdir()
            for i in range(8, 17):
                name, file = f"language.blocks.{i:02d}.mlp.linear", f"layer-{i}.npy"
                path = workspace / "edits" / file
                path.write_bytes(f"synthetic {method} {i}".encode())
                if method == "edit-fit":
                    edit["layers"].append({"layer": i, "tensor": name, "rank": 32, "ridge_relative": .001,
                                           "edited_sha256": planner.sha(path)})
        dump(out / "edit-fit/solve.json", edit)
        seals = {}
        for method, folder, stage in (("closed_form", "edit-fit", "edit-solve"), ("lora", "lora-fit", "lora-train")):
            workspace = out / folder / "workspace"
            log = out / f"{stage}.log"
            log.write_text("synthetic stage log\n")
            files = [workspace / "workspace.json", *sorted((workspace / "edits").glob("*.npy"))]
            seals[method] = {"artifacts": {str(p): planner.sha(p) for p in files},
                             "sealed_after_stage": stage, "stage_log_sha256": planner.sha(log)}
        dump(out / "candidate-seals.json", seals)
        self.selected_fixture(out, case)
        return out

    def selected_fixture(self, out, case):
        selected, source = out / "lora-selected", out / "lora-fit"
        shutil.copytree(source / "workspace", selected / "workspace")
        shutil.copyfile(source / "training.json", selected / "training.json")
        evaluation = self.root / f"{case}-selection-evaluation"
        for name in ("status.json", "identity.json", "freeze.json", "completed.json", "episodes.json", "metrics.json"):
            dump(evaluation / name, {"synthetic": name})
        dump(evaluation / "plan.json", {"global_freeze": {"path": str(evaluation / "freeze.json")}})
        (evaluation / "obs.npz").write_bytes(b"synthetic observation, verified by mocked selector")
        source_artifacts = {str(p): planner.sha(p) for p in [source / "workspace/workspace.json", *sorted((source / "workspace/edits").glob("*.npy"))]}
        candidate = {"candidate_id": "initial-1e-3", "learning_rate": .001,
                     "workspace": str(source / "workspace"), "workspace_artifacts": source_artifacts,
                     "training_path": str(source / "training.json"), "training_sha256": planner.sha(source / "training.json"),
                     "seal_path": str(out / "lora-sweep/initial-seal.json")}
        dump(Path(candidate["seal_path"]), {"synthetic": "fit seal"})
        for name in ("source-freeze.json", "identity.json"):
            dump(out / "lora-sweep" / name, {"synthetic": name})
        nom = {"case": case, "dataset_manifest_sha256": planner.sha(out / "lora-dataset/dataset.json"),
               "spec_sha256": planner.sha(out / "spec.json"), "candidates": [candidate],
               "source_freeze_path": str(out / "lora-sweep/source-freeze.json"),
               "source_identity_path": str(out / "lora-sweep/identity.json")}
        dump(out / "lora-sweep/nomination.json", nom)
        evidence = {"scores": [{"candidate_id": "initial-1e-3", "total_strict_successes": 10}],
                    "source_receipt_sha256": {str(evaluation / name): planner.sha(evaluation / name) for name in ("completed.json", "episodes.json")},
                    "source_observation_metric_sha256": {str(evaluation / name): planner.sha(evaluation / name) for name in ("obs.npz", "metrics.json")}}
        winner = {"candidate_id": "initial-1e-3", "learning_rate": .001}
        receipt = {"case": case, "split": "development", "heldout_used": False, "status": "complete",
                   "selection_rule": "synthetic deterministic development rule", "evaluation_root": str(evaluation),
                   "selected_candidate_id": winner["candidate_id"], "selected_learning_rate": winner["learning_rate"],
                   "selected_source_workspace": candidate["workspace"], "selected_training_sha256": candidate["training_sha256"],
                   "dataset_manifest_sha256": nom["dataset_manifest_sha256"], "spec_sha256": nom["spec_sha256"],
                   "nomination_sha256": planner.sha(out / "lora-sweep/nomination.json"), **evidence}
        dump(selected / "selection.json", receipt)
        files = [selected / "training.json", selected / "selection.json", selected / "workspace/workspace.json", *sorted((selected / "workspace/edits").glob("*.npy"))]
        dump(selected / "seal.json", {"case": case, "split": "development", "selected_candidate_id": winner["candidate_id"],
                                     "artifacts": {str(p): planner.sha(p) for p in files}, "source_workspace_artifacts": source_artifacts})
        self.selections[str(out)] = {"nomination": (nom, {winner["candidate_id"]: candidate}, {}),
                                     "election": (winner, evidence)}

    def test_plan_rejects_stale_parity_current_spec_and_base(self):
        out = self.adaptation("spatial")
        planner.validate_case(out, "spatial", self.protocol)
        editpath = out / "edit-fit/solve.json"
        original = json.loads(editpath.read_text())
        changed = copy.deepcopy(original)
        changed["teacher_action_parity"]["dataset_sha256"] = "stale"
        dump(editpath, changed)
        with self.assertRaisesRegex(ValueError, "different LoRA targets"):
            planner.validate_case(out, "spatial", self.protocol)
        dump(editpath, original)
        specpath = out / "spec.json"
        spec = json.loads(specpath.read_text())
        dump(specpath, {**spec, "new_field": "changed"})
        with self.assertRaisesRegex(ValueError, "current frozen development"):
            planner.validate_case(out, "spatial", self.protocol)
        dump(specpath, spec)
        base = Path(original["base_workspace"])
        manifest = json.loads((base / "workspace.json").read_text())
        dump(base / "workspace.json", {**manifest, "base_vlm_sha256": "another base"})
        with self.assertRaisesRegex(ValueError, "Base workspace identity"):
            planner.validate_case(out, "spatial", self.protocol)

    def test_global_freeze_binds_both_cases_runtime_and_exact_tensor_receipts(self):
        temporal = self.adaptation("temporal")
        spatial = self.adaptation("spatial")
        worker = self.here / "worker.py"
        worker.write_text("# test worker\n")
        layout = self.here / "runtime/layouts/spatial_heldout.json"
        dump(layout, {"synthetic": "heldout layout specification, no observations"})
        runtime = [worker, layout, self.here / "protocol.json", self.here / "rewrites.py"]
        with patch.object(planner, "runtime_sources", return_value=runtime):
            planner.make(self.root / "runs", self.root / "frozen")
        freeze = self.root / "frozen/freeze.json"
        artifacts = json.loads(freeze.read_text())["artifacts"]
        self.assertIn(str(worker), artifacts)
        self.assertIn(str(layout), artifacts)
        self.assertIn(str(temporal / "candidate-seals.json"), artifacts)
        self.assertIn(str(spatial / "candidate-seals.json"), artifacts)
        self.assertEqual(sum("/workspace/edits/" in p for p in artifacts), 54)
        self.assertIn(str(spatial / "lora-selected/selection.json"), artifacts)
        self.assertIn(str(self.root / "spatial-selection-evaluation/episodes.json"), artifacts)
        self.assertIn(str(self.root / "spatial-selection-evaluation/metrics.json"), artifacts)
        self.assertNotIn(str(self.root / "spatial-selection-evaluation/obs.npz"), artifacts)
        for case in ("temporal", "spatial"):
            plan = json.loads((freeze.parent / f"{case}-plan.json").read_text())
            self.assertEqual(plan["global_freeze"]["sha256"], planner.sha(freeze))
            self.assertEqual(sum(b["n"] for b in plan["batches"]), 320)
            self.assertEqual(plan["rewrite_dispatch"]["calls"], 10)
            self.assertGreater(plan["rewrite_dispatch"]["cpu_wall_seconds"], 0)
            self.assertTrue(all("lora-selected/workspace" in b["workspace"] for b in plan["batches"] if b["condition"] == "lora"))
            expected = json.loads(freeze.read_text())["plan_sha256"][case]
            self.assertEqual(planner.plan_payload_sha256(plan), expected)
            plan["batches"][0]["effective_prompt"] += " changed"
            self.assertNotEqual(planner.plan_payload_sha256(plan), expected)
        tensor = temporal / "edit-fit/workspace/edits/layer-8.npy"
        tensor.write_bytes(b"mutated candidate")
        with patch.object(planner, "runtime_sources", return_value=runtime):
            with self.assertRaisesRegex(ValueError, "post-fit stage seal"):
                planner.make(self.root / "runs", self.root / "badfreeze")

    def test_lora_candidate_seal_rejects_changed_merged_tensor(self):
        self.adaptation("temporal")
        spatial = self.adaptation("spatial")
        tensor = spatial / "lora-selected/workspace/edits/layer-8.npy"
        tensor.write_bytes(b"changed after training")
        with patch.object(planner, "runtime_sources", return_value=[]):
            with self.assertRaisesRegex(ValueError, "export differs from its seal"):
                planner.make(self.root / "runs", self.root / "badfreeze")

    def test_selection_receipt_must_match_recomputed_development_winner(self):
        out = self.adaptation("spatial")
        planner.verify_selected_lora(out, "spatial")
        winner, evidence = self.selections[str(out)]["election"]
        self.selections[str(out)]["election"] = (winner, {**evidence, "scores": []})
        with self.assertRaisesRegex(ValueError, "recomputed development winner/evidence"):
            planner.verify_selected_lora(out, "spatial")

    def test_selected_export_must_match_source_even_with_resealed_copies(self):
        out = self.adaptation("temporal")
        selected = out / "lora-selected"
        tensor = selected / "workspace/edits/layer-8.npy"
        tensor.write_bytes(b"changed and resealed, but not evaluated")
        seal = json.loads((selected / "seal.json").read_text())
        seal["artifacts"][str(tensor)] = planner.sha(tensor)
        dump(selected / "seal.json", seal)
        with self.assertRaisesRegex(ValueError, "weights differ from the evaluated source"):
            planner.verify_selected_lora(out, "temporal")

    def test_dispatch_is_recomputed_from_prompt_not_trusted_from_protocol(self):
        self.adaptation("temporal")
        self.adaptation("spatial")
        protocol = json.loads((self.here / "protocol.json").read_text())
        protocol["cases"]["temporal"]["cells"][0]["rewrite"]["effective_prompt"] = "unverified replacement"
        dump(self.here / "protocol.json", protocol)
        with patch.object(planner, "runtime_sources", return_value=[]):
            with self.assertRaisesRegex(ValueError, "rewrite differs from raw-prompt dispatch"):
                planner.make(self.root / "runs", self.root / "badfreeze")

    def test_selected_learning_rate_must_belong_to_declared_grid(self):
        out = self.adaptation("spatial")
        path = out / "lora-selected/training.json"
        fit = json.loads(path.read_text())
        dump(path, {**fit, "learning_rate": 1e-4})
        planner.validate_case(out, "spatial", self.protocol)
        dump(path, {**fit, "learning_rate": .05})
        with self.assertRaisesRegex(ValueError, "recipe differs"):
            planner.validate_case(out, "spatial", self.protocol)


if __name__ == "__main__":
    unittest.main()
