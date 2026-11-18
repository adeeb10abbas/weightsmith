"""Execute a frozen comparison plan; seal each completed batch and stop on gaps."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(8 << 20):
            h.update(chunk)
    return h.hexdigest()


def verify_global_freeze(plan):
    """Bind a worker to a supplied source/candidate seal and the complete plan."""
    reference = plan.get("global_freeze")
    if not reference:
        if plan["split"] == "heldout":
            raise ValueError("Heldout execution requires the pre-existing global freeze")
        return {}
    path = Path(reference["path"]).resolve()
    if digest(path) != reference["sha256"]:
        raise ValueError("Global freeze manifest changed")
    manifest = json.loads(path.read_text())
    payload = {k: v for k, v in plan.items() if k != "global_freeze"}
    actual = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if manifest["plan_sha256"].get(plan["case"]) != actual:
        raise ValueError("Case plan differs from its pre-test global freeze")
    expected = manifest["artifacts"]
    if not expected or any(digest(p) != checksum for p, checksum in expected.items()):
        raise ValueError("A globally frozen source, observation manifest, or candidate changed")
    return {**expected, str(path): reference["sha256"]}


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    globally_frozen = verify_global_freeze(plan)
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(WSP_POOL_PORT=str(8700 + args.gpu * 10),
                      WSP_SERVER_GPU=str(args.gpu), WSP_SIM_GPU=str(args.gpu),
                      WSP_EVAL_EXTRA_GPUS="", WSP_TASK_DIRS=str(HERE / "runtime"))
    from weightsmith import config as config
    from weightsmith import pool
    from weightsmith.locks import GpuLock
    if plan["split"] == "heldout" and plan.get("inference_batching") != "fixed16":
        raise ValueError("Comparison inference must keep its kernel batch at sixteen")
    if plan.get("inference_batching") == "fixed16":
        from weightsmith import rollout
        rollout.HARNESS = HERE / "runtime"

    # Keep the simulator cwd/assets unchanged while routing this cohort's output
    # away from the historical output symlink (its external disk may be absent).
    output = root / "sim-output"
    output.mkdir(exist_ok=True)
    os.environ["RSS_OUTPUT_DIR"] = str(output)
    class OutputRoot:
        def __fspath__(self):
            return str(config.ROBOLAB)
        def __truediv__(self, part):
            return output if part == "output" else config.ROBOLAB / part
    pool.ROBOLAB = OutputRoot()

    original_cmd = pool.sim_cmd
    def simulator_command(*pos, **kw):
        cmd = original_cmd(*pos, **kw)
        cmd[1] = str(HERE / "runtime/run_batched.py")
        return cmd
    pool.sim_cmd = simulator_command
    sources = [Path(__file__), args.plan, HERE / "protocol.json", HERE / "rewrites.py",
               REPO / "weightsmith/pool.py", REPO / "weightsmith/batched_server.py",
               REPO / "weightsmith/acts_worker.py", REPO / "weightsmith/lab/openpi_probe.py",
               config.OPENPI_DIR / "src/openpi/models/pi0.py",
               config.OPENPI_DIR / "src/openpi/models/gemma.py",
               config.BASE_WS / "workspace.json",
               REPO / "third_party/pi05_droid_jointpos.sha256",
               *sorted((HERE / "runtime").glob("*.py"))]
    sources += sorted({Path(b["layouts"]) for b in plan["batches"]})
    sources += sorted({Path(b["workspace"]) / "workspace.json" for b in plan["batches"] if b.get("workspace")})
    for workspace in sorted({b["workspace"] for b in plan["batches"] if b.get("workspace")}):
        sources += sorted((Path(workspace) / "edits").glob("*.npy"))
    identity = dict(globally_frozen)
    for p in sources:
        path = str(p.resolve())
        checksum = digest(p)
        if path in identity and identity[path] != checksum:
            raise ValueError("Local runtime/candidate differs from the pre-test freeze")
        identity[path] = checksum
    sealed = root / "identity.json"
    if sealed.exists() and json.loads(sealed.read_text()) != identity:
        raise RuntimeError("Frozen plan/runtime changed: use a new cohort directory")
    save(sealed, identity)
    for source in sorted({*sources, *(Path(p) for p in globally_frozen)}):
        if source.suffix == ".npy" or source.stat().st_size > 2 * 1024**2:
            continue  # Hash exact large tensor artifacts; keep their canonical copy.
        target = root / "frozen-sources" / digest(source) / source.name
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(source, target)
    save(root / "plan.json", plan)
    state = {"cohort_id": plan["cohort_id"], "split": plan["split"], "gpu": args.gpu,
             "status": "starting", "completed_episodes": 0, "completed_batches": 0,
             "planned_episodes": sum(b.get("n", 16) for b in plan["batches"])}
    try:
        for number, spec in enumerate(plan["batches"]):
            if any(digest(p) != checksum for p, checksum in identity.items()):
                raise RuntimeError("Plan, runtime, or candidate changed after freeze")
            batch = root / f"batch-{number:02d}-{spec['cell']}-{spec['condition']}"
            receipt = batch / "completed.json"
            n = spec.get("n", 16)
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["episodes_sha256"] != digest(batch / "episodes.json"):
                    raise RuntimeError(f"Receipt mismatch: {batch}")
                state["completed_episodes"] += old["n"]
                state["completed_batches"] += 1
                continue
            if batch.exists():
                raise RuntimeError(f"Incomplete batch preserved for inspection: {batch}")
            if shutil.disk_usage(root).free < 15 * 1024**3:
                raise RuntimeError("Less than 15 GiB free")
            batch.mkdir()
            state.update(status="waiting_gpu", batch=number, batch_path=str(batch),
                         cell=spec["cell"], condition=spec["condition"], batch_started_at=time.time())
            save(root / "status.json", state)
            print(f"START {number}: {spec['cell']} {spec['condition']}", flush=True)
            os.environ["WSP_START_LAYOUTS"] = spec["layouts"]
            os.environ["WSP_START_LAYOUT_IDS"] = ",".join(spec.get("layout_ids", []))
            start = spec["slot_start"]
            with GpuLock(args.gpu):
                execution_started = time.time()
                queue_seconds = execution_started - state["batch_started_at"]
                state.update(status="running", execution_started_at=execution_started,
                             queue_seconds=queue_seconds)
                save(root / "status.json", state)
                rows, server = pool.run_pool(
                    workspace=spec.get("workspace"), task=spec["task"],
                    tag=f"rss-{plan['case']}-g{args.gpu}-b{number}", slots=range(start, start + n),
                    envs=16, server_gpu=args.gpu, sim_gpu=args.gpu, env_seed=0,
                    rendering_type="performance", video_mode="sensor", log_root=batch,
                    prompt=spec["effective_prompt"], extra_gpus=())
                execution_seconds = time.time() - execution_started
            save(batch / "raw-rows.json", rows)
            if any(digest(p) != checksum for p, checksum in identity.items()):
                raise RuntimeError("Source/candidate changed during batch; preserved without result receipt")
            if len(rows) != n or not all(r["ok"] for r in rows):
                raise RuntimeError("Incomplete policy batch; no automatic cohort replay")
            episodes = []
            for row in rows:
                obs = Path(row["obs"])
                metrics_path = obs.parent / f"metrics_r{row['run']}.json"
                metrics = json.loads(metrics_path.read_text())
                e = row["env"]
                with np.load(obs, allow_pickle=False) as data:
                    camera_hashes = {k: hashlib.sha256(data[k][0].tobytes()).hexdigest()
                                     for k in ["base_img", "wrist_img"]}
                    seed = int(data["seed"][0])
                    decisions = len(data["seed"])
                selected = metrics["selected_goal"]
                strict = {k: bool(v[e]) for k, v in metrics["strict_final"].items()}
                if strict[selected] != row["success"]:
                    raise RuntimeError("Selected termination and strict final score disagree")
                episodes.append({**row, "cohort_id": plan["cohort_id"], "split": plan["split"],
                                 "case": plan["case"], "cell": spec["cell"],
                                 "inference_batching": plan.get("inference_batching", "dynamic_power_of_two"),
                                 "condition": spec["condition"], "input_prompt": spec["input_prompt"],
                                 "effective_prompt": spec["effective_prompt"], "policy_seed": seed,
                                 "layout_id": metrics["layout"]["layout_ids"][e],
                                 "strict_final": strict,
                                 "strict_ever": {k: bool(v[e]) for k, v in metrics["strict_ever"].items()},
                                 "stock_final": {k: bool(v[e]) for k, v in metrics["stock_final"].items()},
                                 "stock_ever": {k: bool(v[e]) for k, v in metrics["stock_ever"].items()},
                                 "selected_goal": selected, "decision_count": decisions,
                                 "metrics": str(metrics_path), "metrics_sha256": digest(metrics_path),
                                 "first_camera_sha256": camera_hashes, "obs_sha256": digest(obs)})
            save(batch / "episodes.json", episodes)
            value = {"batch": number, "cell": spec["cell"], "condition": spec["condition"],
                     "n": n, "successes": sum(r["success"] for r in rows), "server": server,
                     "wall_seconds": time.time() - state["batch_started_at"],
                     "queue_seconds": queue_seconds, "execution_seconds": execution_seconds,
                     "sim_attempt_logs": [str(p) for p in sorted(batch.glob("sim-*.log"))],
                     "episodes_sha256": digest(batch / "episodes.json")}
            save(receipt, value)
            with (root / "events.jsonl").open("a") as f:
                f.write(json.dumps(value) + "\n")
                f.flush()
                os.fsync(f.fileno())
            state["completed_episodes"] += n
            state["completed_batches"] += 1
            save(root / "status.json", state)
            print(f"DONE {number}: {value['successes']}/{n}", flush=True)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="stopped", reason=repr(exc))
        raise
    finally:
        save(root / "status.json", state)


if __name__ == "__main__":
    main()
