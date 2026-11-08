"""Closed-loop evaluation of GR00T N1.7 DROID on RoboLab, one fresh simulator process per 16-slot group.

System python3, stdlib only. Holds GPU0's flocks (the coordinator's ~/.cache/weightsmith/locks and weightsmith's
WSP_LOCK_DIR) for one evaluation batch (one arm on one task) and releases them at the end, starts
groot_server.py (Isaac-GR00T venv) once, and runs every group of E slots as the FIRST and only run of a fresh
Isaac process on the same GPU (later runs of one process start from different initial scenes). Slot identity as in
weightsmith/pool.py: slot s = g * E + env, group g = the sampling-seed base of that process, environment seed 0.
Equal slots get equal initial scenes and equal noise seeds, so evaluations on equal slots are paired.
"""
from __future__ import annotations

import argparse, fcntl, json, os, shutil, signal, subprocess, sys, time
from pathlib import Path

HOME = Path.home()
# Machine layout; defaults are ali-ms. Other hosts set these in src/machine.env (sourced by worker.sh).
WS = Path(os.environ.get("GROOT_WS", HOME / "cross/groot"))
SRC = WS / "src"
GROOT_PY = WS / "Isaac-GR00T/.venv/bin/python"
ROBOLAB = Path(os.environ.get("GROOT_ROBOLAB", HOME / "vibes/weight-space-policy-lab/robolab_checkout"))
SIM_PY = Path(os.environ.get("GROOT_SIM_PY", HOME / "vibes/weight-space-policy-lab/robolab_env/.venv/bin/python"))
LOCKS = [Path(d) for d in os.environ.get(
    "GROOT_LOCK_DIRS", f"{HOME}/.cache/weightsmith/locks:{HOME}/vibes/opus-weight-surgery/state").split(":")]
MACHINE = os.environ.get("GROOT_MACHINE", "ali-ms-rtx3090")
VK_ICD = os.environ.get("GROOT_VK_ICD", "/usr/share/vulkan/icd.d/nvidia_icd.json")
TASK_DIRS = str(SRC / "tasks")
ENV_SEED = 0


def lock_gpu(gpu):
    fs = []
    for d in LOCKS:
        d.mkdir(parents=True, exist_ok=True)
        f = open(d / f"gpu{gpu}.lock", "w")
        fcntl.flock(f, fcntl.LOCK_EX)
        fs.append(f)
    return fs


def wait_ready(proc, log, timeout=1200):
    t = time.monotonic() + timeout
    while time.monotonic() < t:
        if proc.poll() is not None:
            raise RuntimeError(f"server died rc={proc.returncode}; see {log}")
        if "Server is ready and listening" in Path(log).read_text(errors="ignore"):
            return
        time.sleep(2)
    raise TimeoutError("server not ready")


def run_group(a, g, server, out, stamp, attempt, video):
    name = f"groot-{a.tag}-{stamp}-g{g}a{attempt}"
    senv = os.environ.copy()
    senv.pop("OPENAI_API_KEY", None)
    senv.update(OMNI_KIT_ACCEPT_EULA="Y", VK_ICD_FILENAMES=VK_ICD, WSP_TASK_DIRS=a.task_dirs, PYTHONUNBUFFERED="1")
    senv.pop("OWSP_PROMPT", None)
    senv.pop("OWSP_OBS_DIR", None)
    obs_dir = out / f"{a.tag}.obs" / f"g{g}"
    if a.record_obs:
        senv["OWSP_OBS_DIR"] = str(obs_dir)
    if a.prompt:
        senv["OWSP_PROMPT"] = a.prompt
    scmd = [str(SIM_PY), str(SRC / "adapters/run_groot_batched.py"), "--task", a.task,
            "--environment-seed", str(ENV_SEED), "--sampling-seed-base", str(g), "--remote-port", str(a.port),
            "--open-loop-horizon", "8", "--num-envs", str(a.envs), "--num-runs", "1", "--headless",
            "--video-mode", "sensor" if video else "none", "--output-folder-name", name, "--device", f"cuda:{a.gpu}",
            "--/rtx/verifyDriverVersion/enabled=false", "--/renderer/multiGpu/enabled=false",
            "--/renderer/multiGpu/autoEnable=false", f"--/renderer/activeGpu={a.gpu}"]
    if a.rendering != "default":
        scmd += ["--rendering-type", a.rendering]
    t0 = time.time()
    with open(out / f"{a.tag}.g{g}a{attempt}.sim.log", "w") as f:
        p = subprocess.Popen(scmd, cwd=str(ROBOLAB), env=senv, stdout=f, stderr=subprocess.STDOUT)
        while p.poll() is None:
            if server.poll() is not None:
                p.kill()
                raise RuntimeError("policy server died")
            time.sleep(3)
    rows = []
    res = ROBOLAB / "output" / name / "episode_results.jsonl"
    vdir = WS / "videos" / a.task / (a.arm or a.tag)
    if res.exists():
        for line in res.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                if int(r["run"]) != 0:
                    continue
                e = int(r["env_id"])
                vid = None
                if video:
                    src = [v for v in (ROBOLAB / "output" / name / a.task).glob(f"*_0_env{e}.mp4") if "viewport" not in v.name]
                    if src:
                        vdir.mkdir(parents=True, exist_ok=True)
                        vid = vdir / f"{g * a.envs + e}.mp4"
                        shutil.copyfile(src[0], vid)
                rows.append({"slot": g * a.envs + e, "group": g, "env": e, "ok": True, "success": bool(r["success"]),
                             "steps": r.get("episode_step"), "instruction": r.get("instruction"),
                             "reason": r.get("reason"), "score": r.get("score"), "attempt": attempt,
                             "robolab_output": str(ROBOLAB / "output" / name), "video": str(vid) if vid else None,
                             "obs": str(obs_dir / f"obs_r0_e{e}.npz") if a.record_obs else None})
    return rows, p.returncode, round(time.time() - t0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--slot-start", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--prompt", default=None, help="instruction override (else the task's own wording)")
    ap.add_argument("--edit", default=None)
    ap.add_argument("--task-dirs", default=TASK_DIRS)
    ap.add_argument("--out", required=True, help="results directory")
    ap.add_argument("--video-groups", type=int, default=1, help="record sensor video in the first N groups (5%% budget)")
    ap.add_argument("--rendering", default="performance", choices=["performance", "balanced", "quality", "default"],
                    help="RoboLab --rendering-type preset ('default' = unset, IsaacLab's fallback)")
    ap.add_argument("--max-batch", type=int, default=0, help="server micro-batch size (0 = whole request)")
    ap.add_argument("--arm", default=None, help="arm name for the video folder (default: tag)")
    ap.add_argument("--record-obs", action="store_true", help="save every request and chunk (OWSP_OBS_DIR)")
    a = ap.parse_args()
    slots = list(range(a.slot_start, a.slot_start + a.n))
    if a.slot_start % a.envs or a.n % a.envs:
        raise SystemExit("slots must be whole groups of envs")
    groups = list(range(a.slot_start // a.envs, (a.slot_start + a.n) // a.envs))
    out = Path(a.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%m%d-%H%M%S")
    t0 = time.time()
    locks = lock_gpu(a.gpu)
    t_lock = time.time()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(a.gpu), PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
    slog = out / f"{a.tag}.server.log"
    cmd = [str(GROOT_PY), str(SRC / "adapters/groot_server.py"), "--port", str(a.port),
           "--calls-log", str(out / f"{a.tag}.calls.jsonl"), "--max-batch", str(a.max_batch)]
    if a.edit:
        cmd += ["--edit", a.edit]
    server = subprocess.Popen(cmd, cwd=str(WS), env=env, stdout=open(slog, "w"), stderr=subprocess.STDOUT)
    rows, procs, t_ready, err = [], [], None, None
    try:
        wait_ready(server, slog)
        t_ready = time.time()
        for g in groups:
            for attempt in range(2):
                r, rc, secs = run_group(a, g, server, out, stamp, attempt, groups.index(g) < a.video_groups)
                procs.append({"group": g, "attempt": attempt, "rc": rc, "episodes": len(r), "secs": secs})
                if len(r) == a.envs:
                    rows += r
                    break
            else:
                rows += r  # partial results of the last attempt are kept
    except BaseException as e:  # noqa: BLE001 - partial results are kept and the error recorded
        err = repr(e)
    finally:
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
        for f in locks:
            fcntl.flock(f, fcntl.LOCK_UN)
            f.close()
    seen = {r["slot"] for r in rows}
    rows += [{"slot": s, "ok": False} for s in slots if s not in seen]
    rows.sort(key=lambda r: r["slot"])
    ok = [r for r in rows if r["ok"]]
    text = slog.read_text(errors="ignore").splitlines()
    load = [l for l in text if l.startswith("loaded ")]
    edit = [l for l in text if l.startswith("edit ")]
    summary = {"tag": a.tag, "machine": MACHINE, "task": a.task, "prompt": a.prompt, "edit": a.edit, "slots": [slots[0], slots[-1]],
               "envs": a.envs, "env_seed": ENV_SEED, "rendering": a.rendering, "max_batch": a.max_batch, "groups": groups, "n": len(rows), "completed": len(ok),
               "success": sum(r["success"] for r in ok), "error": err, "procs": procs,
               "server_load": json.loads(load[0][7:]) if load else None,
               "edit_report": json.loads(edit[0][5:]) if edit else None,
               "wait_lock_s": round(t_lock - t0, 1), "gpu_s": round(time.time() - t_lock, 1),
               "server_ready_s": round(t_ready - t_lock, 1) if t_ready else None, "rows": rows}
    (out / f"{a.tag}.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k not in ("rows", "edit_report")}))


if __name__ == "__main__":
    main()
