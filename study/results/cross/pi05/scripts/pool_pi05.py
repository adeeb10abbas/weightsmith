"""Track C (pi0.5 DROID, the WSP control) closed-loop evaluation worker for the cross-model comparison.

Run one worker per GPU on the GPU host (system python3, stdlib only):
    nohup setsid python3 pool_pi05.py --gpu 1 --port 8700 > worker-g1.log 2>&1 < /dev/null &
    nohup setsid python3 pool_pi05.py --gpu 0 --port 8720 > worker-g0.log 2>&1 < /dev/null &
On the rented pod (2x RTX 4080 SUPER) the same script runs with --root /workspace/cross/pi05
--weightsmith /workspace/weightsmith --lock-dirs ~/.cache/weightsmith/locks.

Each iteration a worker
  1. takes its GPU's flocks in the order Tracks A and B take them: the coordinator's
     ~/.cache/weightsmith/locks/gpu<g>.lock, then weightsmith's WSP_LOCK_DIR gpu<g>.lock (blocking; no lock is held
     while waiting for the other GPU, so there is no lock-order deadlock);
  2. claims the first job of the queue file that is neither done nor claimed (O_EXCL claim file);
  3. runs it through weightsmith's own pool (weightsmith.pool.run_pool) with exactly ONE 16-slot group, so the group
     is the first and only run of a fresh Isaac process (later runs of a process start from other scenes); a crashed
     group is retried by run_pool in another fresh process;
  4. releases the locks, so Tracks A and B get the GPU between our jobs.

Slots follow weightsmith/pool.py: slot s = g * 16 + env, group g = sampling-seed base of the process, environment
seed 0, per-env noise stream ((g * 64 + env) * 100 + k). Equal slots = equal start and noise across arms (paired).

Job (queue file = JSON list, re-read every iteration, so jobs can be appended while workers run):
  {"tag": "T1-v2mi-f-s30000", "code": "T1", "task": "RubiksCubeThenBananaTask", "slot_start": 30000, "n": 16,
   "arm": "v2mi-f", "workspace": "<project>/cands/v2mi-f" | null, "fp": "63db0c52a51306c9" | null,
   "prompt": null | "<instruction override>", "rendering_type": null | "performance", "video_mode": "sensor"}
rendering_type null = RoboLab's default preset (no --rendering-type flag: Isaac Lab's RenderCfg default, "balanced").
With video_mode "sensor" each slot's MP4 is copied to <root>/videos/<code>/<arm>/<slot>.mp4.
Before a candidate job the worker recomputes the candidate's fingerprint (owsp.fingerprint: sha256 over the sorted
edits/*.npy names and bytes) and refuses to run on a mismatch; after it, it checks that the policy server loaded
exactly that workspace (and no workspace for base jobs).
Output: runs/<tag>.json (one row per slot + server load report + timing), ledger.jsonl (GPU time per job).
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
import traceback
from pathlib import Path

import shutil

HOME = Path.home()
ROOT = HOME / "cross/pi05"               # ali-ms layout; the pod passes --root/--weightsmith/--lock-dirs
WS_REPO = ROOT / "weightsmith"           # weightsmith @ ad31044 (origin/main), wsp.env with WSP_TASK_DIRS for T8
LOCK_DIRS = [HOME / ".cache/weightsmith/locks", HOME / "vibes/opus-weight-surgery/state"]


def fingerprint(cand_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted((cand_dir / "edits").glob("*.npy")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def lock_gpu(gpu: int):
    fs = []
    for d in LOCK_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        f = open(d / f"gpu{gpu}.lock", "a")
        fcntl.flock(f, fcntl.LOCK_EX)
        fs.append(f)
    return fs


def unlock(fs):
    for f in reversed(fs):
        try:
            fcntl.flock(f, fcntl.LOCK_UN)
        finally:
            f.close()


def port_free(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def log(msg: str):
    print(time.strftime("%m-%d %H:%M:%S"), msg, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--queue", default=str(HOME / "cross/pi05/jobs/queue.json"))
    ap.add_argument("--max-fail", type=int, default=3)
    ap.add_argument("--video-mode", default="sensor", help="default for jobs without their own video_mode")
    ap.add_argument("--root", default=None, help="state directory (runs/, claims/, logs/, videos/, ledger.jsonl)")
    ap.add_argument("--weightsmith", default=None, help="weightsmith checkout whose run_pool and wsp.env to use")
    ap.add_argument("--lock-dirs", default=None, help="comma-separated flock directories, taken in this order")
    a = ap.parse_args()
    global ROOT, WS_REPO, LOCK_DIRS
    if a.root:
        ROOT = Path(a.root).expanduser()
    if a.weightsmith:
        WS_REPO = Path(a.weightsmith).expanduser()
    if a.lock_dirs:
        LOCK_DIRS = [Path(x).expanduser() for x in a.lock_dirs.split(",") if x.strip()]
    if a.queue == str(HOME / "cross/pi05/jobs/queue.json") and a.root:
        a.queue = str(ROOT / "jobs/queue.json")

    os.environ["WSP_POOL_PORT"] = str(a.port)  # before weightsmith.config is imported: run_pool's lane-0 port
    os.environ.setdefault("WSP_ENV_FILE", str(WS_REPO / "wsp.env"))
    sys.path.insert(0, str(WS_REPO))
    from weightsmith import config as C  # noqa: E402
    from weightsmith.pool import run_pool  # noqa: E402
    assert C.POOL_PORT == a.port, (C.POOL_PORT, a.port)

    runs, claims, logs = ROOT / "runs", ROOT / "claims", ROOT / "logs"
    for d in (runs, claims, logs):
        d.mkdir(parents=True, exist_ok=True)
    log(f"worker gpu={a.gpu} port={a.port} queue={a.queue} root={ROOT} weightsmith={WS_REPO} locks={LOCK_DIRS} "
        f"task_dirs={os.environ.get('WSP_TASK_DIRS')} vk_icd={C.VK_ICD}")

    def pending():
        jobs = json.loads(Path(a.queue).read_text())
        out = []
        for j in jobs:
            if (runs / f"{j['tag']}.json").exists() or (claims / j["tag"]).exists():
                continue
            if len(list(runs.glob(f"{j['tag']}.fail*.json"))) >= a.max_fail:
                continue
            out.append(j)
        return out

    while True:
        if (ROOT / "STOP").exists() or (ROOT / f"STOP-gpu{a.gpu}").exists():
            log("STOP file present: exiting")
            return
        if (ROOT / f"PAUSE-gpu{a.gpu}").exists() or not pending():
            time.sleep(60)
            continue
        t_req = time.time()
        locks = lock_gpu(a.gpu)
        t_lock = time.time()
        job, rec = None, None
        try:
            for j in pending():
                try:
                    fd = os.open(claims / j["tag"], os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                except FileExistsError:
                    continue
                os.write(fd, json.dumps({"gpu": a.gpu, "pid": os.getpid(), "t": time.time()}).encode())
                os.close(fd)
                job = j
                break
            if job is None:
                continue
            log(f"lock gpu{a.gpu} after {t_lock - t_req:.0f}s wait; job {job['tag']}")
            if job["n"] != 16 or job["slot_start"] % 16:
                raise ValueError("one job = exactly one 16-slot group")
            ws = job.get("workspace")
            fp = None
            if ws:
                fp = fingerprint(Path(ws))
                if fp != job["fp"]:
                    raise RuntimeError(f"fingerprint {fp} != expected {job['fp']} for {ws}")
            if not port_free(a.port):
                raise RuntimeError(f"port {a.port} busy")
            stamp = time.strftime("%m%d-%H%M%S")
            log_root = logs / f"{job['tag']}-{stamp}"
            t0 = time.time()
            video_mode = job.get("video_mode") or a.video_mode
            rendering = job.get("rendering_type") or None  # None: RoboLab's default preset (no flag)
            rows, rep = run_pool(workspace=ws, task=job["task"], tag=f"x05-{job['tag']}",
                                 slots=list(range(job["slot_start"], job["slot_start"] + job["n"])), envs=16,
                                 server_gpu=a.gpu, sim_gpu=a.gpu, env_seed=0, video_mode=video_mode,
                                 rendering_type=rendering, log_root=log_root, prompt=job.get("prompt"),
                                 extra_gpus=())
            t1 = time.time()
            try:
                server = json.loads(rep)
            except ValueError:
                server = {"raw": rep[:2000]}
            loaded = server.get("workspace")
            if ws and (loaded is None or Path(loaded).resolve() != Path(ws).resolve()):
                raise RuntimeError(f"server loaded {loaded!r}, expected {ws}")
            if not ws and loaded is not None:
                raise RuntimeError(f"base job but server loaded {loaded!r}")
            ok = [r for r in rows if r.get("ok")]
            videos = {}
            for r in ok:
                if r.get("video"):
                    dst = ROOT / "videos" / job["code"] / job["arm"] / f"{r['slot']}.mp4"
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(r["video"], dst)
                    videos[str(r["slot"])] = str(dst)
            sim_log = next(iter(sorted(log_root.glob("sim-l0a*.log"))), None)
            rec = {**job, "rendering_type_used": rendering or "default (no --rendering-type flag)",
                   "video_mode_used": video_mode, "videos": videos, "fp_checked": fp, "gpu": a.gpu, "port": a.port,
                   "server": server, "sim_cmd_log": str(sim_log) if sim_log else None,
                   "server_fingerprint": server.get("fingerprint"), "rows": sorted(rows, key=lambda r: r["slot"]),
                   "completed": len(ok), "success": sum(bool(r["success"]) for r in ok),
                   "lock_requested": t_req, "lock_acquired": t_lock, "started": t0, "finished": t1,
                   "lock_wait_s": round(t_lock - t_req, 1), "gpu_s": round(t1 - t_lock, 1),
                   "log_root": str(log_root), "host": socket.gethostname()}
            if len(ok) != job["n"]:
                raise RuntimeError(f"only {len(ok)}/{job['n']} slots completed")
            (runs / f"{job['tag']}.json").write_text(json.dumps(rec, indent=1))
            log(f"done {job['tag']}: {rec['success']}/{rec['completed']} in {t1 - t0:.0f}s (gpu{a.gpu})")
        except Exception as e:  # noqa: BLE001 - record, release the claim, let the queue retry it
            if job is not None:
                k = len(list(runs.glob(f"{job['tag']}.fail*.json")))
                fail = {**(rec or job), "error": repr(e), "traceback": traceback.format_exc(), "gpu": a.gpu,
                        "t": time.time()}
                (runs / f"{job['tag']}.fail{k}.json").write_text(json.dumps(fail, indent=1, default=str))
                log(f"FAILED {job['tag']} (attempt {k}): {e!r}")
                (claims / job["tag"]).unlink(missing_ok=True)
                time.sleep(30)
        finally:
            t_rel = time.time()
            unlock(locks)
            if job is not None:
                with open(ROOT / "ledger.jsonl", "a") as f:
                    f.write(json.dumps({"tag": job["tag"], "gpu": a.gpu, "lock_requested": t_req,
                                        "lock_acquired": t_lock, "released": t_rel,
                                        "held_s": round(t_rel - t_lock, 1)}) + "\n")
        time.sleep(2)  # let a waiting track take the GPU before we ask again


if __name__ == "__main__":
    main()
