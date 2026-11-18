"""Boundary tests for heldout separation, candidate identity, and analysis units."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import HERE, make_layouts, protocol, select_winner, sha, test_batches, validate_plan
from analyze import clustered_interval


class ProtocolTests(unittest.TestCase):
    def plan(self):
        p = protocol()
        spaces = {"weightsmith": p["historical_full"]["workspace"], "matched_random": p["matched_random"]["workspace"],
                  "teacher_action_lora": "/sealed/lora/workspace"}
        return {"cohort_id": p["cohort_id"], "case": "temporal", "split": "heldout", "inference_batching": "fixed16",
                "protocol_sha256": sha(HERE / "protocol.json"), "batches": test_batches(spaces)}

    def test_heldout_schedule_pairs_all_conditions_without_duplicate_slots(self):
        plan = self.plan(); validate_plan(plan)
        for arm in protocol()["conditions"]:
            rows = [b for b in plan["batches"] if b["condition"] == arm]
            self.assertEqual(sum(r["n"] for r in rows), 64)
            self.assertEqual(len({r["slot_start"] for r in rows}), 4)
        self.assertEqual(len({tuple(b["condition"] for b in plan["batches"][i:i+5]) for i in range(0, 20, 5)}), 4)

    def test_heldout_cannot_substitute_new_checkpoint_or_prompt(self):
        for field, value in (("workspace", "/unsealed/edit"), ("effective_prompt", "new wording"), ("slot_start", 190000)):
            plan = self.plan()
            target = next(b for b in plan["batches"] if b["condition"] == "weightsmith")
            target[field] = value
            with self.assertRaises(ValueError): validate_plan(plan)

    def test_heldout_refuses_missing_arm_and_partial_budget(self):
        for mutation in (lambda p: p["batches"].pop(), lambda p: p["batches"][0].update(n=8)):
            plan = self.plan(); mutation(plan)
            with self.assertRaises(ValueError): validate_plan(plan)

    def test_layouts_are_deterministic_and_disjoint(self):
        dev, test = make_layouts("development"), make_layouts("heldout")
        self.assertEqual(dev, make_layouts("development"))
        self.assertEqual(test, make_layouts("heldout"))
        self.assertEqual((len(dev["layouts"]), len(test["layouts"])), (8, 16))
        self.assertTrue({str(r["poses"]) for r in dev["layouts"]}.isdisjoint({str(r["poses"]) for r in test["layouts"]}))

    def test_selection_uses_complete_results_and_lower_rate_for_ties(self):
        rows = [{"learning_rate": lr, "successes": k, "n": 16} for lr, k in [(1e-3, 8), (3e-5, 8), (1e-4, 4)]]
        self.assertEqual(select_winner(rows)["learning_rate"], 3e-5)
        with self.assertRaises(ValueError): select_winner(rows[:2])
        rows[0]["n"] = 15
        with self.assertRaises(ValueError): select_winner(rows)

    def test_uncertainty_keeps_four_replicates_inside_layout(self):
        ids = [f"L{i}" for i in range(16) for _ in range(4)]
        # Perfect within-layout pairing has zero difference even when all
        # trial outcomes vary substantially between layouts.
        result = clustered_interval([0] * 64, ids)
        self.assertEqual((result["ci_low"], result["ci_high"]), (0., 0.))
        self.assertTrue(result["degenerate_ci"])
        # Four perfectly correlated replicas must not appear to be 64
        # independent environments: the 16-cluster interval remains broad.
        variable = clustered_interval([0] * 32 + [1] * 32, ids)
        self.assertGreaterEqual(variable["ci_high"] - variable["ci_low"], .4)
        with self.assertRaises(ValueError): clustered_interval([0] * 63, ids[:63])
        with self.assertRaises(ValueError): clustered_interval([0] * 64, [f"episode{i}" for i in range(64)])


if __name__ == "__main__":
    unittest.main()
