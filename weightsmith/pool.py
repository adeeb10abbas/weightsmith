"""RoboLab evaluation engine for pi0.5.

One batched policy server (WSP_SERVER_GPU) plus one Isaac process with E parallel envs
(WSP_SIM_GPU); on the reference box both shared one 24 GB GPU. Episode identity:
  group g  -> sampling-seed base for one sim "run" (all E envs step together)
  slot  s  = g * E + env_id
Per-env noise seed stream = ((g * 64 + env_id) * 100 + k), so a slot always receives the same
noise regardless of which other slots run with it -> evaluations on equal slots are paired.
"""
from __future__ import annotations

import argparse, json, os, signal, subprocess, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402
from weightsmith.rollout import start_server, wait_ready, ROBOLAB, SIM_PY, HARNESS  # noqa: E402


def sim_cmd(task, name, port, gpu, envs, runs, group0, env_seed, video_mode, rendering_type):
    cmd = [str(SIM_PY), str(HARNESS / "run_batched.py"), "--policy", "pi05", "--task", task,
           "--environment-seed", str(env_seed), "--sampling-seed-base", str(group0),
           "--remote-port", str(port), "--num-envs", str(envs), "--num-runs", str(runs),
           "--headless", "--video-mode", video_mode, "--output-folder-name", name, "--device", f"cuda:{gpu}",
           "--/rtx/verifyDriverVersion/enabled=false", "--/renderer/multiGpu/enabled=false",
           "--/renderer/multiGpu/autoEnable=false", f"--/renderer/activeGpu={gpu}"]
    if rendering_type:
        cmd += ["--rendering-type", rendering_type]
    return cmd


def _lane(lane, *, workspace, task, tag, stamp, groups, envs, server_gpu, sim_gpu, port, env_seed,
          video_mode, rendering_type, log_root, prompt, procs, out):
    """One server + one Isaac process working through a contiguous block of groups (with retries)."""
    server = start_server(workspace, port, server_gpu, log_root / f"server-l{lane}.log")
    procs.append(server)
    done, load_report = [], ""
    try:
        wait_ready(server, port, timeout=900)
        load_report = (log_root / f"server-l{lane}.log").read_text().split("loaded ", 1)[-1].split("\n")[0]
        todo = list(groups)
        attempt = 0
        while todo and attempt < 4:
            name = f"pool-{tag}-{stamp}-l{lane}a{attempt}"
            obs_dir = log_root / f"obs-l{lane}a{attempt}"
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.update(OMNI_KIT_ACCEPT_EULA="Y", VK_ICD_FILENAMES=C.VK_ICD, OWSP_OBS_DIR=str(obs_dir))
            env.pop("OWSP_PROMPT", None)
            if prompt:
                env["OWSP_PROMPT"] = prompt
            with open(log_root / f"sim-l{lane}a{attempt}.log", "w") as f:
                p = subprocess.Popen(sim_cmd(task, name, port, sim_gpu, envs, len(todo), todo[0], env_seed,
                                             video_mode, rendering_type),
                                     cwd=ROBOLAB, env=env, stdout=f, stderr=subprocess.STDOUT)
                procs.append(p)
                while p.poll() is None:
                    if server.poll() is not None:
                        p.kill()
                        raise RuntimeError(f"policy server (lane {lane}, gpu {server_gpu}) died")
                    time.sleep(3)
            res = ROBOLAB / "output" / name / "episode_results.jsonl"
            per_run = {}
            if res.exists():
                for line in res.read_text().splitlines():
                    if line.strip():
                        r = json.loads(line)
                        per_run[int(r["run"])] = per_run.get(int(r["run"]), 0) + 1
            runs_done = sorted(r for r, c in per_run.items() if c == envs)
            done.append((name, obs_dir, todo[0], runs_done))
            k = 0
            while k in runs_done:
                k += 1
            todo = todo[k:]
            attempt += 1
    except BaseException as e:  # noqa: BLE001 - reported to the caller, partial results kept
        out[lane] = {"done": done, "report": load_report, "error": repr(e)}
        raise
    finally:
        for q in (server,):
            if q.poll() is None:
                q.send_signal(signal.SIGINT)
                try:
                    q.wait(30)
                except subprocess.TimeoutExpired:
                    q.kill()
    out[lane] = {"done": done, "report": load_report, "error": None}


def run_pool(*, workspace, task, tag, slots, envs=16, server_gpu=C.SERVER_GPU, sim_gpu=C.SIM_GPU, env_seed=0,
             video_mode="sensor", rendering_type="performance", log_root=None, prompt=None, extra_gpus=(), **_):
    """Run one episode per slot. Lane 0 uses (server_gpu, sim_gpu); each GPU in `extra_gpus` adds a lane
    that hosts its own server + Isaac process. Groups are split into contiguous blocks across lanes."""
    import threading
    slots = sorted(slots)
    groups = sorted({s // envs for s in slots})
    if len(slots) != len(groups) * envs or len(set(slots)) != len(slots):
        raise ValueError(f"slots must be whole groups of {envs}")
    for a, b in zip(groups, groups[1:]):
        if b != a + 1:
            raise ValueError("groups must be consecutive")
    lanes = [(server_gpu, sim_gpu)] + [(g, g) for g in extra_gpus if g not in (server_gpu, sim_gpu)]
    lanes = lanes[:len(groups)]
    stamp = time.strftime("%m%d-%H%M%S")
    log_root = Path(log_root or (C.LOG_DIR / f"pool-{tag}-{stamp}"))
    log_root.mkdir(parents=True, exist_ok=True)
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # make finally-cleanup run on kill
    # contiguous, near-equal blocks of groups per lane
    q, r = divmod(len(groups), len(lanes))
    blocks, i = [], 0
    for li in range(len(lanes)):
        n = q + (1 if li < r else 0)
        blocks.append(groups[i:i + n]); i += n
    procs, out, threads = [], {}, []
    try:
        for li, ((sg, mg), blk) in enumerate(zip(lanes, blocks)):
            t = threading.Thread(target=lambda li=li, sg=sg, mg=mg, blk=blk: _lane(
                li, workspace=workspace, task=task, tag=tag, stamp=stamp, groups=blk, envs=envs,
                server_gpu=sg, sim_gpu=mg, port=C.POOL_PORT + li, env_seed=env_seed, video_mode=video_mode,
                rendering_type=rendering_type, log_root=log_root, prompt=prompt, procs=procs, out=out), daemon=True)
            t.start(); threads.append(t)
            time.sleep(5)  # stagger Isaac start-up
        for t in threads:
            while t.is_alive():
                t.join(timeout=5)
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
    errors = [v["error"] for v in out.values() if v.get("error")]
    load_report = out.get(0, {}).get("report", "")
    rows = []
    for li in sorted(out):
        for name, obs_dir, g0, runs_done in out[li]["done"]:
            outp = ROBOLAB / "output" / name
            res = outp / "episode_results.jsonl"
            if not res.exists():
                continue
            for line in res.read_text().splitlines():
                if not line.strip():
                    continue
                rr = json.loads(line)
                run, e = int(rr["run"]), int(rr["env_id"])
                if run not in runs_done:
                    continue
                g = g0 + run
                vids = sorted((outp / task).glob(f"*_{run}_env{e}.mp4"))
                rows.append({"slot": g * envs + e, "group": g, "env": e, "ok": True, "lane": li,
                             "success": bool(rr["success"]), "steps": rr.get("episode_step"),
                             "instruction": rr.get("instruction"), "task_name": rr.get("task_name"),
                             "score": rr.get("score"), "reason": rr.get("reason"),
                             "out_dir": str(outp / task), "run": run,
                             "obs": str(obs_dir / f"obs_r{run}_e{e}.npz"),
                             "video": str(vids[0]) if vids else "",
                             "log": str(outp / task / f"log_{run}_env{e}.json")})
    got = {x["slot"] for x in rows}
    for sl in slots:
        if sl not in got:
            rows.append({"slot": sl, "ok": False})
    reports = {li: out[li]["report"] for li in out}
    if len({v for v in reports.values() if v}) > 1:  # every lane must serve the identical weights
        raise RuntimeError(f"lanes loaded different weights: {reports}")
    (log_root / "rows.json").write_text(json.dumps({"server": load_report, "lanes": [list(l) for l in lanes],
                                                    "errors": errors, "rows": rows}, indent=1))
    return rows, load_report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--task", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--slot-start", type=int, required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--envs", type=int, default=16)
    ap.add_argument("--rendering-type", default="performance")
    ap.add_argument("--extra-gpus", default="", help="comma list of extra GPUs, each runs its own server+sim lane")
    a = ap.parse_args()
    t0 = time.time()
    rows, rep = run_pool(workspace=a.workspace, task=a.task, tag=a.tag, slots=range(a.slot_start, a.slot_start + a.n),
                         envs=a.envs, rendering_type=a.rendering_type or None,
                         extra_gpus=tuple(int(x) for x in a.extra_gpus.split(",") if x.strip()))
    ok = [r for r in rows if r["ok"]]
    print(json.dumps({"n": len(rows), "completed": len(ok), "success": sum(r["success"] for r in ok),
                      "wall_min": round((time.time() - t0) / 60, 1), "server": rep[:300]}))


if __name__ == "__main__":
    main()
