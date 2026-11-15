"""Bounded, resumable spatial development diagnostics. No editing or held-out access.

One process owns one case and GPU; every completed batch is sealed before advancing.
Interrupted/incomplete batches are preserved and require inspection rather than automatic replay.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def batches(case):
    start = case["slot_start"]
    if type(start) is not int or start % 16 or not 70000 <= start <= 79968:
        raise ValueError("spatial diagnostics must use aligned development slots in 70000..79999")
    return [("original", start), ("explicit", start), ("explicit", start + 16), ("original", start + 16)]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--case", choices=["S01", "S02"], required=True)
    ap.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    ap.add_argument("--root", type=Path, required=True)
    args = ap.parse_args()
    manifest = json.loads((HERE / "spatial-pilot.json").read_text())
    case = manifest["cases"][args.case]
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.environ.update(WSP_POOL_PORT=str(8380 + args.gpu * 10),
                      WSP_SERVER_GPU=str(args.gpu), WSP_SIM_GPU=str(args.gpu), WSP_EVAL_EXTRA_GPUS="")
    from weightsmith import config as config
    from weightsmith import pool
    from weightsmith.locks import GpuLock

    # Only this worker's simulator command uses the cohort-specific telemetry launcher.
    original_sim_cmd = pool.sim_cmd

    def simulator_command(*pos, **kw):
        cmd = original_sim_cmd(*pos, **kw)
        cmd[1] = str(HERE / "runtime/run_batched.py")
        return cmd

    pool.sim_cmd = simulator_command
    state = {"case_id": args.case, "gpu": args.gpu, "status": "starting", "completed_batches": 0,
             "completed_episodes": 0, "planned_episodes": 64, "heldout_episodes": 0,
             "formal_eligibility": "not assessed; human semantic review pending"}
    identity = {"manifest_sha256": digest(HERE / "spatial-pilot.json"),
                "worker_sha256": digest(__file__), "telemetry_sha256": digest(HERE / "scene_telemetry.py"),
                "launcher_sha256": digest(HERE / "runtime/run_batched.py"),
                "checkpoint_manifest_sha256": digest(REPO / "third_party/pi05_droid_jointpos.sha256"),
                "pool_sha256": digest(REPO / "weightsmith/pool.py"),
                "scoring_sha256": digest(config.ROBOLAB / "robolab/core/task/conditionals.py")}
    sealed = root / "identity.json"
    if sealed.exists() and json.loads(sealed.read_text()) != identity:
        raise RuntimeError("runtime/plan changed since this cohort started; preserve it and use a new cohort")
    save(sealed, identity)
    try:
        for number, (wording, start) in enumerate(batches(case)):
            batch = root / f"batch-{number:02d}-{wording}"
            receipt = batch / "completed.json"
            if receipt.exists():
                state["completed_batches"] += 1
                state["completed_episodes"] += 16
                continue
            if batch.exists():
                raise RuntimeError(f"preserved incomplete batch requires inspection: {batch}")
            if shutil.disk_usage(root).free < 10 * 1024**3:
                raise RuntimeError("less than 10 GiB free; stopping before another batch")
            batch.mkdir()
            state.update(status="running", batch=number, wording=wording, slot_start=start,
                         log_dir=str(batch), batch_started_at=time.time())
            save(root / "status.json", state)
            print(f"START {args.case} batch {number} {wording} slots {start}-{start+15}", flush=True)
            with GpuLock(args.gpu):
                rows, server = pool.run_pool(
                    workspace=None, task=case["task"], tag=f"iros27-{args.case.lower()}-b{number}",
                    slots=range(start, start + 16), envs=16, server_gpu=args.gpu, sim_gpu=args.gpu,
                    env_seed=0, rendering_type="performance", video_mode="sensor", log_root=batch,
                    prompt=case[f"{wording}_prompt"], extra_gpus=())
            save(batch / "raw-rows.json", rows)
            if len(rows) != 16 or not all(row["ok"] for row in rows):
                raise RuntimeError("incomplete batch; raw evidence preserved, no automatic replay")
            episodes = []
            for row in rows:
                obs = Path(row["obs"])
                scene_start, scene_end = obs.parent / "scene-start.json", obs.parent / "scene-final.json"
                if not scene_start.exists() or not scene_end.exists():
                    raise RuntimeError("scene telemetry missing; preserve batch for inspection")
                initial, final = json.loads(scene_start.read_text()), json.loads(scene_end.read_text())
                with np.load(obs, allow_pickle=False) as data:
                    cameras = {key: hashlib.sha256(data[key][0].tobytes()).hexdigest()
                               for key in ["base_img", "wrist_img"]}
                    seed = int(data["seed"][0])
                e = row["env"]
                episodes.append({**row, "case_id": args.case, "cohort_id": manifest["cohort_id"],
                                 "condition": f"base_{wording}", "actual_prompt": case[f"{wording}_prompt"],
                                 "policy_seed": seed, "scene_seed": 0, "eval_replica": 0, "order_block": number,
                                 "first_camera_sha256": cameras,
                                 "initial_object_poses": {k: {f: v[e] for f, v in item.items()}
                                                          for k, item in initial["objects"].items()},
                                 "max_bowl_displacement_m": final["max_bowl_displacement_m"][e],
                                 "semantic_review": "pending", "eligibility": "not_assessed"})
            save(batch / "episodes.json", episodes)
            receipt_value = {"batch": number, "wording": wording, "slot_start": start, "n": 16,
                             "successes": sum(row["success"] for row in rows), "server": server,
                             "wall_seconds": time.time() - state["batch_started_at"],
                             "episodes_sha256": digest(batch / "episodes.json")}
            with (root / "events.jsonl").open("a") as stream:
                stream.write(json.dumps(receipt_value) + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            save(receipt, receipt_value)
            state["completed_batches"] += 1
            state["completed_episodes"] += 16
            save(root / "status.json", state)
            print(f"DONE {args.case} batch {number}: {receipt_value['successes']}/16", flush=True)
        state["status"] = "complete"
    except BaseException as exc:
        state.update(status="stopped", reason=repr(exc))
        raise
    finally:
        save(root / "status.json", state)


if __name__ == "__main__":
    main()
