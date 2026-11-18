"""A failed wording gap or repeated noise stream must never authorize fitting."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location("goal_adaptation_run", HERE / "run.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


class QualificationGateTests(unittest.TestCase):
    def fixture(self, root, *, repeat_teacher=11, repeat_start=151008, split="development"):
        cfg = {"original": "original wording", "teacher": "clearer wording", "task": "ExampleTask"}
        (root / "protocol.json").write_text(json.dumps({"cases": {"example": cfg}, "qualification": {
            "slot_start": {"initial": {"example": 150000}, "repeat": {"example": 151008}}}}))
        rounds = []
        for round_id, start in enumerate((150000, repeat_start)):
            round_ = {}
            stage = ("initial", "repeat")[round_id]
            owner = root / "qualification" / "example" / stage
            declarations = []
            for number, (cell, successes) in enumerate((("original", 1), ("teacher", 12 if round_id == 0 else repeat_teacher))):
                batch = owner / f"batch-{number:02d}-{cell}-base"
                batch.mkdir(parents=True)
                rows = [{"case": "example", "cell": cell, "split": split, "condition": "base",
                         "ok": True, "slot": start + i, "effective_prompt": cfg[cell],
                         "inference_batching": "fixed16", "task_name": cfg["task"], "success": i < successes} for i in range(16)]
                (batch / "episodes.json").write_text(json.dumps(rows))
                receipt = {"batch": number, "n": 16, "cell": cell, "episodes_sha256": build.sha(batch / "episodes.json")}
                (batch / "completed.json").write_text(json.dumps(receipt))
                declarations.append({"cell": cell, "condition": "base", "workspace": None,
                                     "effective_prompt": cfg[cell], "slot_start": start, "task": cfg["task"]})
                round_[cell] = str(batch)
            (owner / "plan.json").write_text(json.dumps({"case": "example", "split": split, "stage": stage,
                "inference_batching": "fixed16", "batches": declarations}))
            (owner / "status.json").write_text(json.dumps({"status": "complete", "completed_batches": 2, "completed_episodes": 32}))
            (owner / "identity.json").write_text(json.dumps({str(root / "protocol.json"): build.sha(root / "protocol.json")}))
            rounds.append(round_)
        path = root / "qualified.json"
        path.write_text(json.dumps({"case": "example", "split": split, "rounds": rounds,
                                   "protocol_sha256": build.sha(root / "protocol.json")}))
        return path

    def check(self, accepted, **kw):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            path = self.fixture(root, **kw)
            old = build.COHORT
            try:
                build.COHORT = root
                if accepted:
                    self.assertIn(str(path), build.verify_gate(path, "example"))
                else:
                    with self.assertRaises(ValueError):
                        build.verify_gate(path, "example")
            finally:
                build.COHORT = old

    def test_two_fresh_qualified_rounds_pass(self):
        self.check(True)

    def test_initial_success_does_not_override_failed_repeat(self):
        self.check(False, repeat_teacher=4)

    def test_repeat_cannot_reuse_initial_noise_slots(self):
        self.check(False, repeat_start=150000)

    def test_test_outcomes_cannot_qualify_adaptation(self):
        self.check(False, split="heldout")


if __name__ == "__main__":
    unittest.main()
