import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

path = Path(__file__).resolve().parents[1] / "worker.py"
spec = importlib.util.spec_from_file_location("rss_worker", path)
w = importlib.util.module_from_spec(spec)
spec.loader.exec_module(w)


class GlobalFreezeTests(unittest.TestCase):
    def fixture(self, root, split="heldout"):
        candidate = root / "candidate.bin"
        candidate.write_bytes(b"original merged weights")
        plan = {"case": "temporal", "split": split, "batches": [{"prompt": "frozen request"}]}
        payload_hash = w.hashlib.sha256(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        freeze = root / "freeze.json"
        freeze.write_text(json.dumps({"artifacts": {str(candidate): w.digest(candidate)},
                                      "plan_sha256": {"temporal": payload_hash}}))
        plan["global_freeze"] = {"path": str(freeze), "sha256": w.digest(freeze)}
        return plan, candidate, freeze

    def test_accepts_exact_preexisting_freeze_and_rejects_changed_candidate(self):
        with tempfile.TemporaryDirectory() as td:
            plan, candidate, _ = self.fixture(Path(td))
            self.assertIn(str(candidate), w.verify_global_freeze(plan))
            candidate.write_bytes(b"changed after plan construction")
            with self.assertRaisesRegex(ValueError, "globally frozen"):
                w.verify_global_freeze(plan)

    def test_case_request_and_manifest_cannot_change(self):
        with tempfile.TemporaryDirectory() as td:
            plan, _, freeze = self.fixture(Path(td))
            plan["batches"][0]["prompt"] = "different request"
            with self.assertRaisesRegex(ValueError, "Case plan"):
                w.verify_global_freeze(plan)
            freeze.write_text("{}")
            with self.assertRaisesRegex(ValueError, "manifest changed"):
                w.verify_global_freeze(plan)

    def test_heldout_requires_freeze_but_development_does_not(self):
        self.assertEqual(w.verify_global_freeze({"split": "development"}), {})
        with self.assertRaisesRegex(ValueError, "requires"):
            w.verify_global_freeze({"split": "heldout"})

    def test_development_selection_enforces_a_supplied_candidate_seal(self):
        with tempfile.TemporaryDirectory() as td:
            plan, candidate, _ = self.fixture(Path(td), split="development")
            self.assertIn(str(candidate), w.verify_global_freeze(plan))
            candidate.write_bytes(b"changed development candidate")
            with self.assertRaisesRegex(ValueError, "globally frozen"):
                w.verify_global_freeze(plan)


if __name__ == "__main__":
    unittest.main()
