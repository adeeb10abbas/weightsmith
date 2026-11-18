"""Select fixed development decisions for new goals without selecting outcomes."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parents[1] / "rss_repair_comparison" / "prepare_shared.py"
_spec = importlib.util.spec_from_file_location("original_context_helpers", ORIGINAL)
helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(helpers)
sha = helpers.sha


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_batch(batch, case, cell, expected_prompt):
    batch = Path(batch).resolve()
    owner = batch.parent
    plan = json.loads((owner / "plan.json").read_text())
    status = json.loads((owner / "status.json").read_text())
    receipt = json.loads((batch / "completed.json").read_text())
    if (plan.get("split") != "development" or status.get("status") != "complete"
            or status.get("cohort_id") != plan.get("cohort_id") or plan.get("case") != case
            or plan.get("inference_batching") != "fixed16"
            or status.get("completed_batches") != len(plan["batches"])):
        raise ValueError("Only completed, fixed16 development cohorts from this goal are allowed")
    index = receipt["batch"]
    declared = plan["batches"][index]
    if (batch.name != f"batch-{index:02d}-{cell}-base" or declared["cell"] != cell
            or declared["condition"] != "base" or declared.get("workspace")
            or declared["effective_prompt"] != expected_prompt or receipt["n"] != 16
            or receipt.get("cell") != cell or receipt.get("condition") != "base"
            or sha(batch / "episodes.json") != receipt["episodes_sha256"]):
        raise ValueError("Source batch does not match its declared unchanged-policy development plan")
    layout_path = Path(declared["layouts"])
    layout = json.loads(layout_path.read_text())
    layout_ids = [x["id"] for x in layout["layouts"]]
    if layout.get("split") != "development" or layout.get("case") != case or len(set(layout_ids)) != 4:
        raise ValueError("Exactly four development layouts from this case are required")
    if declared.get("layout_ids") and declared["layout_ids"] != layout_ids:
        raise ValueError("Unexpected layout subset or permutation")
    rows = json.loads((batch / "episodes.json").read_text())
    if len(rows) != 16 or {r["slot"] for r in rows} != set(range(declared["slot_start"], declared["slot_start"] + 16)):
        raise ValueError("Development source slots are missing, repeated, or different from the plan")
    identity = json.loads((owner / "identity.json").read_text())
    if identity.get(str(layout_path.resolve())) != sha(layout_path):
        raise ValueError("Development layout identity differs from source execution")
    for row in rows:
        expected = {"case": case, "cell": cell, "split": "development", "condition": "base",
                    "cohort_id": plan["cohort_id"], "inference_batching": "fixed16",
                    "input_prompt": declared["input_prompt"], "effective_prompt": expected_prompt,
                    "task_name": declared["task"], "ok": True,
                    "layout_id": layout_ids[row["slot"] % 4]}
        if any(row.get(k) != v for k, v in expected.items()):
            raise ValueError("Source episode differs from its development plan")
        if not Path(row["obs"]).resolve().is_relative_to(batch):
            raise ValueError("Observation is outside its source batch")
    return rows, layout_ids, {str(p): sha(p) for p in
        (owner / "plan.json", owner / "status.json", owner / "identity.json", layout_path.resolve())}


def chosen_rows(rows, layout_ids, replicas):
    selected = []
    for layout in layout_ids:
        candidates = sorted((r for r in rows if r["layout_id"] == layout), key=lambda r: r["slot"])
        if len(candidates) != 4:
            raise ValueError("Each development layout must have four policy-noise replicates")
        selected.extend(candidates[:replicas])
    return selected


def prepare(case, protocol_path, teacher_batch, preservation_batch, output):
    protocol_path = Path(protocol_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    cfg = protocol["cases"][case]
    items, observations, receipts, owners = [], {}, {}, {}
    layout_reference = None
    for batch, cell, prompt, replicas, fractions in (
        (teacher_batch, "teacher", cfg["teacher"], 1, (0, .25, .5, .75)),
        (preservation_batch, "competing", cfg["competing"], 2, (0, .5)),
    ):
        batch = Path(batch).resolve()
        rows, layouts, owner_hashes = read_batch(batch, case, cell, prompt)
        if layout_reference is not None and layouts != layout_reference:
            raise ValueError("Repair and preservation use different development layouts")
        layout_reference = layouts
        context_index = helpers.live_context_index(batch, rows)
        receipts[str(batch)] = sha(batch / "completed.json")
        owners.update(owner_hashes)
        for row in chosen_rows(rows, layouts, replicas):
            path = Path(row["obs"]).resolve()
            with np.load(path, allow_pickle=False) as z:
                n = len(z["seed"])
            indices = [int(f * (n - 1)) for f in fractions]
            if n < 1 or len(set(indices)) != len(indices):
                raise ValueError("Source has too few decisions for the fixed selection rule")
            for k in indices:
                context = helpers.live_context(context_index, row, k)
                context["live_batch_bucket"] = 16
                context["inference_batching"] = "fixed16"
                context["live_context"]["inference_batching"] = "fixed16"
                context["live_context_sha256"] = canonical(context["live_context"])
                repair = cell == "teacher"
                items.append({"item_id": f"{case}-{cell}-{row['slot']}-k{k}",
                    "obs_path": str(path), "decision_index": k,
                    "split": "validation" if row["layout_id"] == layouts[-1] else "train",
                    "role": "repair" if repair else "preservation", "layout_id": row["layout_id"],
                    "source_cell": cell, "source_slot": row["slot"],
                    "original_prompt": cfg["original"] if repair else prompt,
                    "teacher_prompt": prompt, "recorded_prompt": prompt, **context})
            observations[str(path)] = row["obs_sha256"]
    counts = {f"{s}_{r}": sum(x["split"] == s and x["role"] == r for x in items)
              for s in ("train", "validation") for r in ("repair", "preservation")}
    if counts != {"train_repair": 12, "train_preservation": 12, "validation_repair": 4, "validation_preservation": 4}:
        raise ValueError(f"Unexpected fixed observation budget: {counts}")
    result = {"cohort_id": "rss-goal-expansion-20261117-adaptation-v1", "case": case,
        "split": "development", "inference_batching": "fixed16", "reuse_or_new": "new",
        "selection_rule": "Teacher: lowest slot per layout, fractions0,.25,.5,.75. Competing: two lowest slots per layout, fractions0,.5. All outcomes retained. Last declared layout is validation.",
        "protocol_sha256": sha(protocol_path), "selector_sha256": sha(__file__),
        "context_helper_sha256": sha(ORIGINAL), "source_observation_sha256": observations,
        "source_receipt_sha256": receipts, "source_owner_sha256": owners, "counts": counts, "items": items}
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as f:
        json.dump(result, f, indent=2)
        f.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "protocol", "teacher-batch", "preservation-batch", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    spec = prepare(args.case, args.protocol, args.teacher_batch, args.preservation_batch, args.output)
    print(json.dumps({"case": spec["case"], "counts": spec["counts"], "output": args.output}))
