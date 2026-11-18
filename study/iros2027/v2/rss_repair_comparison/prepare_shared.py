"""Select predeclared inputs from completed development receipts, never outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TASKS = {
    "temporal": {"original": "RSSCubeThenBananaTask", "teacher": "RSSCubeThenBananaTask",
                 "reverse_order": "RSSBananaThenCubeTask", "cube_only": "RSSCubeOnlyFullHorizonTask"},
    "spatial": {"original": "RSSMustardRightTask", "teacher": "RSSMustardRightTask",
                "opposite_left": "RSSMustardLeftTask", "on_raisin": "RSSMustardOnRaisinTask"},
}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha256(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def live_context_index(batch, rows):
    """Index the exact development requests submitted at each recorded sim step.

    The batched client sends all environments needing a replan together. Completed
    environments disappear from subsequent request batches, so the original
    cohort size alone is not the numerical inference bucket.
    """
    known, hashes = {}, {}
    for row in rows:
        path = Path(row["obs"]).resolve()
        env, run = int(row["env"]), int(row["run"])
        if path.name != f"obs_r{run}_e{env}.npz":
            raise ValueError("Observation filename does not identify its recorded run/environment")
        actual = sha(path)
        if actual != row["obs_sha256"]:
            raise ValueError("Changed replay-context observation")
        with np.load(path, allow_pickle=False) as z:
            steps = z["step"].copy()
            if len(steps) != len(z["seed"]) or len(steps) == 0 or np.any(np.diff(steps) <= 0):
                raise ValueError("Replay-context decision steps must be nonempty and strictly increasing")
        known[str(path)] = {"env_id": env, "run_id": run, "steps": steps, "obs_sha256": actual}
        hashes[str(path)] = actual
    for path, info in known.items():
        discovered = {str(p.resolve()) for p in Path(path).parent.glob(f"obs_r{info['run_id']}_e*.npz")}
        expected = {p for p, v in known.items() if Path(p).parent == Path(path).parent and v["run_id"] == info["run_id"]}
        if discovered != expected:
            raise ValueError("Replay context includes an unreceipted or missing environment recording")
    return {"observations": known, "source_batch_observation_sha256": hashes,
            "source_episodes_path": str((batch / "episodes.json").resolve()),
            "source_episodes_sha256": sha(batch / "episodes.json"),
            "source_receipt_path": str((batch / "completed.json").resolve()),
            "source_receipt_sha256": sha(batch / "completed.json")}


def live_context(index, row, decision_index):
    source_path = str(Path(row["obs"]).resolve())
    source = index["observations"][source_path]
    step = int(source["steps"][decision_index])
    members = []
    for path, info in index["observations"].items():
        if Path(path).parent != Path(source_path).parent or info["run_id"] != source["run_id"]:
            continue
        matches = np.flatnonzero(info["steps"] == step)
        if len(matches):
            members.append({"env_id": info["env_id"], "obs_path": path,
                            "decision_index": int(matches[0]), "obs_sha256": info["obs_sha256"]})
    members.sort(key=lambda m: m["env_id"])
    if not 1 <= len(members) <= 16 or len({m["env_id"] for m in members}) != len(members):
        raise ValueError("Invalid recorded live-batch membership")
    if not any(m["obs_path"] == source_path and m["decision_index"] == decision_index for m in members):
        raise ValueError("Selected request absent from its recorded live batch")
    context = {k: v for k, v in index.items() if k != "observations"}
    context.update(source_run_id=source["run_id"], source_step=step, members=members)
    bucket = 1 << (len(members) - 1).bit_length()
    return {"source_env_id": source["env_id"], "source_run_id": source["run_id"], "source_step": step,
            "live_batch_size": len(members), "live_batch_bucket": bucket, "live_batch_members": members,
            "live_context": context, "live_context_sha256": canonical_sha256(context)}


def owner_plan(root, case=None):
    """A batch belongs to its immediate cohort directory, never a loose receipt."""
    root = Path(root).resolve()
    status = json.loads((root / "status.json").read_text())
    plan = json.loads((root / "plan.json").read_text())
    if status.get("status") != "complete" or status.get("split") != "development" or plan.get("split") != "development":
        raise ValueError("Requires a complete development cohort and development plan")
    if plan.get("case") not in TASKS or case is not None and plan["case"] != case:
        raise ValueError("Source owner case mismatch")
    if not plan.get("cohort_id") or status.get("cohort_id") != plan["cohort_id"]:
        raise ValueError("Source owner cohort identity mismatch")
    if status.get("completed_batches") != len(plan["batches"]):
        raise ValueError("Source owner has incomplete batch accounting")
    return plan


def batch_rows(batch, case, *, replacement=False):
    batch = Path(batch).resolve()
    plan = owner_plan(batch.parent, case)
    receipt = json.loads((batch / "completed.json").read_text())
    number = receipt.get("batch")
    if type(number) is not int or not 0 <= number < len(plan["batches"]):
        raise ValueError("Batch is not an entry of its owning plan")
    declared = plan["batches"][number]
    cell = receipt["cell"]
    if cell not in TASKS[case] or declared.get("cell") != cell:
        raise ValueError("Unexpected source cell")
    if replacement and cell in {"original", "teacher"}:
        raise ValueError("Replacement is allowed only for declared preservation cells")
    if batch.name != f"batch-{number:02d}-{cell}-base":
        raise ValueError("Batch path does not match its owning plan")
    if receipt.get("condition") != "base" or declared.get("condition") != "base" or declared.get("workspace"):
        raise ValueError("Adaptation sources must use the unchanged base policy")
    if declared.get("task") != TASKS[case][cell]:
        raise ValueError("Unexpected task; cube_only requires full-horizon qualification")
    if sha(batch / "episodes.json") != receipt["episodes_sha256"]:
        raise ValueError("Changed development receipt")
    rows = json.loads((batch / "episodes.json").read_text())
    n = declared.get("n", 16)
    if n != 16 or receipt.get("n") != n or len(rows) != n:
        raise ValueError("Expected sixteen completed source episodes")
    expected_slots = set(range(declared["slot_start"], declared["slot_start"] + n))
    if {r["slot"] for r in rows} != expected_slots:
        raise ValueError("Source slots do not match the owning plan")
    expected_layouts = {f"{'T' if case == 'temporal' else 'S'}-D{i:02d}" for i in range(1, 5)}
    if {r.get("layout_id") for r in rows} != expected_layouts:
        raise ValueError("Only the exact four development layouts are allowed")
    for r in rows:
        expected_layout = f"{'T' if case == 'temporal' else 'S'}-D{r['slot'] % 4 + 1:02d}"
        if r["layout_id"] != expected_layout:
            raise ValueError("Source layout does not match the declared slot mapping")
        expected = {"split": "development", "case": case, "cohort_id": plan["cohort_id"],
                    "cell": cell, "condition": "base", "task_name": declared["task"],
                    "input_prompt": declared["input_prompt"], "effective_prompt": declared["effective_prompt"]}
        if any(r.get(k) != v for k, v in expected.items()) or r.get("ok") is not True:
            raise ValueError("Episode identity differs from its development owner plan")
        if not Path(r["obs"]).resolve().is_relative_to(batch):
            raise ValueError("Observation must be owned by the selected source batch")
    return receipt, rows, plan


def prepare(root, output, replacements=()):
    root, output = Path(root).resolve(), Path(output)
    plan = owner_plan(root)
    protocol = json.loads((HERE / "protocol.json").read_text())
    case = plan["case"]
    prompts = protocol["cases"][case]
    items, sources, receipt_sources = [], {}, {}
    replacements_by_cell = {}
    for b in replacements:
        b = Path(b).resolve()
        receipt, _, _ = batch_rows(b, case, replacement=True)
        if receipt["cell"] in replacements_by_cell:
            raise ValueError("Duplicate replacement cell")
        replacements_by_cell[receipt["cell"]] = b
    if len(plan["batches"]) != 4 or {b["cell"] for b in plan["batches"]} != set(TASKS[case]):
        raise ValueError("Main development plan requires each of the four declared cells once")
    batches = []
    for number, declared in enumerate(plan["batches"]):
        cell = declared["cell"]
        if cell == "original":
            continue
        batches.append(replacements_by_cell.get(cell, root / f"batch-{number:02d}-{cell}-base"))
    owner_sources = {}
    for batch in batches:
        receipt, rows, source_plan = batch_rows(batch, case)
        cell = receipt["cell"]
        if cell == "teacher" and source_plan["batches"][receipt["batch"]]["effective_prompt"] != prompts["teacher"]:
            raise ValueError("Recorded teacher instruction differs from the declared repair teacher")
        receipt_sources[str(batch.resolve())] = sha(batch / "completed.json")
        for p in (batch.parent / "plan.json", batch.parent / "status.json"):
            owner_sources[str(p)] = sha(p)
        context_index = live_context_index(batch, rows)
        # Fixed lowest slot for each development layout, regardless of success.
        selected = {}
        for row in sorted(rows, key=lambda r: r["slot"]):
            selected.setdefault(row["layout_id"], row)
        if len(selected) != 4:
            raise ValueError("Expected exactly four development layouts")
        repair = cell == "teacher"
        for layout, row in sorted(selected.items()):
            obs = Path(row["obs"])
            if sha(obs) != row["obs_sha256"]:
                raise ValueError("Changed source observation")
            with np.load(obs, allow_pickle=False) as z:
                n = len(z["seed"])
            if n < 1:
                raise ValueError("A source episode contains no recorded decisions")
            indices = sorted({int(f * (n - 1)) for f in ([0, .25, .5, .75] if repair else [0, .5])})
            for k in indices:
                item = {"item_id": f"{case}-{cell}-{layout}-k{k}", "obs_path": str(obs.resolve()),
                        "decision_index": k, "split": "validation" if layout.endswith("04") else "train",
                        "role": "repair" if repair else "preservation", "layout_id": layout,
                        "source_cell": cell, "source_slot": row["slot"],
                        "original_prompt": prompts["original"] if repair else row["effective_prompt"],
                        "teacher_prompt": prompts["teacher"] if repair else row["effective_prompt"],
                        "recorded_prompt": row["effective_prompt"]}
                item.update(live_context(context_index, row, k))
                items.append(item)
            sources[str(obs.resolve())] = row["obs_sha256"]
    spec = {"cohort_id": "rss-repair-comparison-20261117-adaptation-v1", "split": "development", "case": case,
            "selection_rule": protocol["development"]["data_selection"],
            "protocol_sha256": sha(HERE / "protocol.json"), "selector_sha256": sha(__file__),
            "source_plan_sha256": sha(root / "plan.json"), "source_observation_sha256": sources,
            "source_receipt_sha256": receipt_sources,
            "source_owner_sha256": owner_sources,
            "replacement_batches": {cell: str(path) for cell, path in replacements_by_cell.items()},
            "replacement_note": "Explicit development preservation replacements; cube-only always uses full horizon" if replacements else None,
            "items": items}
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(spec, indent=2) + "\n")
    print(json.dumps({"output": str(output), "items": len(items),
                      "train": sum(i["split"] == "train" for i in items),
                      "validation": sum(i["split"] == "validation" for i in items)}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--development-root", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--replacement-batch", action="append", default=[])
    args = ap.parse_args()
    prepare(args.development_root, args.output, args.replacement_batch)
