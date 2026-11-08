"""Closed-loop evaluation of Cosmos3 Edge in RoboLab on GPU1 (system python3, stdlib only).

For each job: take the GPU1 flocks (the coordinator's ~/.cache/weightsmith/locks/gpu1.lock, then weightsmith's
WSP_LOCK_DIR gpu1.lock, the order every other user of these locks takes them), start edge_server.py (cosmos venv) and
run one Isaac Sim process per group of E slots on the same GPU, stop the server and release the locks. Locks are
released between jobs so other users of GPU1 get turns.

Slots follow weightsmith/pool.py with 16-slot groups: slot s = 16 g + j, group g is the sampling-seed base and slot s
gets the noise stream ((g * 64 + j) * 100 + k), k = replan index. With E < 16 envs per process (E = 8 here: 16 envs
beside the 9 GB policy server ran PhysX out of GPU memory), group g runs as 16/E processes whose env e holds slot
16 g + offset + e; OWSP_ENV_OFFSET keeps the seeds identical to a 16-env process. Every process runs one episode per
env (the FIRST episode of a fresh process): a process's later episodes start from a different initial scene
(study/results/opus/where/README.md), so a slot's start is identical across arms whatever else is run.

Job file: a JSON list of {"tag", "task", "slot_start", "n", "prompt"?, "edit"?}, re-read before every job (append to
it while the pool runs); results go to OUT/<tag>.json (one row per slot) and the GPU ledger to OUT/../gpu_ledger.jsonl.
A job whose OUT/<tag>.json exists is skipped; a job that fails twice is dropped for this pool run.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HOME = Path.home()
WS = HOME / "cross/cosmos"
SCRIPTS = WS / "scripts"
COSMOS_PY = WS / "cosmos-framework/.venv/bin/python"
ROBOLAB = HOME / "vibes/weight-space-policy-lab/robolab_checkout"
SIM_PY = HOME / "vibes/weight-space-policy-lab/robolab_env/.venv/bin/python"
LOCKS = [HOME / ".cache/weightsmith/locks", HOME / "vibes/opus-weight-surgery/state"]
VK_ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
GPU = 1
GROUP = 16  # slots per sampling-seed group (the shared slot scheme); a group may run as 16/envs processes


def lock_gpu(gpu):
    fs = []
    for d in LOCKS:
        d.mkdir(parents=True, exist_ok=True)
        f = open(d / f"gpu{gpu}.lock", "a")
        fcntl.flock(f, fcntl.LOCK_EX)
        fs.append(f)
    return fs


def unlock(fs):
    for f in reversed(fs):
        fcntl.flock(f, fcntl.LOCK_UN)
        f.close()


def wait_ready(proc, port, timeout=900):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if proc.poll() is not None:
            raise RuntimeError(f"server died rc={proc.returncode}")
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/healthz", timeout=1) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(2)
    raise TimeoutError("server not ready")


def stop(proc):
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def run_group(job, g, envs, off, port, out, task_dirs):
    """One sim process: envs 0..envs-1 hold slots g*GROUP + off + env (OWSP_ENV_OFFSET keeps their noise seeds
    identical to a 16-env process)."""
    tag = job["tag"]
    name = f"cosmosx-{tag}-g{g}o{off}-{time.strftime('%m%d-%H%M%S')}"
    senv = os.environ.copy()
    for k in ("OPENAI_API_KEY", "OWSP_PROMPT", "ANTHROPIC_API_KEY"):
        senv.pop(k, None)
    senv.update(OMNI_KIT_ACCEPT_EULA="Y", VK_ICD_FILENAMES=VK_ICD, OWSP_OBS_DIR=str(out / f"{tag}.obs" / f"g{g}o{off}"),
                WSP_TASK_DIRS=task_dirs, PYTHONUNBUFFERED="1", OWSP_ENV_OFFSET=str(off))
    if job.get("prompt"):
        senv["OWSP_PROMPT"] = job["prompt"]
    cmd = [str(SIM_PY), str(SCRIPTS / "run_cosmos_batched.py"), "--task", job["task"], "--environment-seed", "0",
           "--sampling-seed-base", str(g), "--remote-port", str(port), "--num-envs", str(envs), "--num-runs", "1",
           "--headless", "--video-mode", job.get("video_mode", "none"), "--output-folder-name", name,
           "--device", f"cuda:{GPU}", "--/rtx/verifyDriverVersion/enabled=false",
           "--/renderer/multiGpu/enabled=false", "--/renderer/multiGpu/autoEnable=false",
           f"--/renderer/activeGpu={GPU}", "--rendering-type", "performance"]
    t0 = time.time()
    with open(out / f"{tag}.g{g}o{off}.sim.log", "w") as f:
        p = subprocess.Popen(cmd, cwd=str(ROBOLAB), env=senv, stdout=f, stderr=subprocess.STDOUT)
        return p, name, t0


def run_job(job, envs, port, out, task_dirs, ledger):
    tag = job["tag"]
    slots = list(range(job["slot_start"], job["slot_start"] + job["n"]))
    if job["slot_start"] % GROUP or job["n"] % GROUP or GROUP % envs:
        raise SystemExit(f"{tag}: slots must be whole groups of {GROUP} and {GROUP} a multiple of envs={envs}")
    groups = [job["slot_start"] // GROUP + i for i in range(job["n"] // GROUP)]
    t_req = time.time()
    locks = lock_gpu(GPU)
    t_lock = time.time()
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(GPU), PYTHONUNBUFFERED="1", COSMOS_TRAINING="false")
    cmd = [str(COSMOS_PY), str(SCRIPTS / "edge_server.py"), "--port", str(port),
           "--calls-log", str(out / f"{tag}.calls.jsonl")]
    if job.get("edit"):
        cmd += ["--edit", job["edit"]]
    server = p = None
    group_info, rows, load = [], [], None
    try:
        server = subprocess.Popen(cmd, cwd=str(WS), env=env, stdout=open(out / f"{tag}.server.log", "w"),
                                  stderr=subprocess.STDOUT)
        wait_ready(server, port)
        t_ready = time.time()
        load = [l for l in (out / f"{tag}.server.log").read_text().splitlines() if l.startswith("loaded ")]
        for g, off in [(g, off) for g in groups for off in range(0, GROUP, envs)]:
            p, name, t0 = run_group(job, g, envs, off, port, out, task_dirs)
            while p.poll() is None:
                if server.poll() is not None:
                    p.kill()
                    raise RuntimeError("policy server died")
                time.sleep(3)
            res = ROBOLAB / "output" / name / "episode_results.jsonl"
            n_rows = 0
            if res.exists():
                for line in res.read_text().splitlines():
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    e = int(r["env_id"])
                    rows.append({"slot": g * GROUP + off + e, "group": g, "env": e, "env_offset": off, "ok": True,
                                 "success": bool(r["success"]), "steps": r.get("episode_step"),
                                 "instruction": r.get("instruction"), "reason": r.get("reason"), "score": r.get("score"),
                                 "obs": str(out / f"{tag}.obs" / f"g{g}o{off}" / f"obs_r0_e{e}.npz")})
                    n_rows += 1
            group_info.append({"group": g, "env_offset": off, "rc": p.returncode, "rows": n_rows, "sim_s": round(time.time() - t0, 1),
                               "robolab_output": str(ROBOLAB / "output" / name)})
    finally:
        if p is not None and p.poll() is None:
            p.kill()
            p.wait()
        stop(server)
        unlock(locks)
    t_end = time.time()
    seen = {r["slot"] for r in rows}
    rows += [{"slot": s, "ok": False} for s in slots if s not in seen]
    rows.sort(key=lambda r: r["slot"])
    ok = [r for r in rows if r["ok"]]
    summary = {"tag": tag, "task": job["task"], "prompt": job.get("prompt"), "edit": job.get("edit"),
               "slots": [slots[0], slots[-1]], "envs": envs, "n": len(rows), "completed": len(ok),
               "success": sum(r["success"] for r in ok), "groups": group_info,
               "server_load": json.loads(load[0][7:]) if load else None, "wait_lock_s": round(t_lock - t_req, 1),
               "gpu_s": round(t_end - t_lock, 1), "server_ready_s": round(t_ready - t_lock, 1),
               "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_lock)),
               "finished": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t_end)), "rows": rows}
    (out / f"{tag}.json").write_text(json.dumps(summary, indent=1))
    with open(ledger, "a") as f:
        f.write(json.dumps({"tag": tag, "gpu": GPU, "start": t_lock, "end": t_end, "gpu_s": round(t_end - t_lock, 1),
                            "kind": "eval", "n": len(rows)}) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", required=True, help="JSON list of jobs")
    ap.add_argument("--out", required=True)
    ap.add_argument("--envs", type=int, default=8, help="envs per sim process (16 ran out of GPU memory beside the server)")
    ap.add_argument("--port", type=int, default=8500)
    ap.add_argument("--task-dirs", default=str(WS / "tasks"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ledger = out.parent / "gpu_ledger.jsonl"
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    fails: dict[str, int] = {}
    while True:  # the job file is re-read before every job, so jobs can be appended while the pool runs
        jobs = json.loads(Path(a.jobs).read_text())
        pending = [j for j in jobs if not (out / f"{j['tag']}.json").exists() and fails.get(j["tag"], 0) < 2]
        if not pending:
            print("queue empty", flush=True)
            break
        job = pending[0]
        try:
            run_job(job, a.envs, a.port, out, a.task_dirs, ledger)
        except Exception as e:  # keep going with the next job; the failure is in the log
            fails[job["tag"]] = fails.get(job["tag"], 0) + 1
            print(f"job {job['tag']} failed: {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    main()
