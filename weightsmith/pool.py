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


def run_pool(*, workspace, task, tag, slots, envs=16, server_gpu=C.SERVER_GPU, sim_gpu=C.SIM_GPU, env_seed=0,
             video_mode="sensor", rendering_type="performance", log_root=None, prompt=None, **_):
    slots = sorted(slots)
    groups = sorted({s // envs for s in slots})
    if len(slots) != len(groups) * envs or len(set(slots)) != len(slots):
        raise ValueError(f"slots must be whole groups of {envs}")
    for a, b in zip(groups, groups[1:]):
        if b != a + 1:
            raise ValueError("groups must be consecutive")
    stamp = time.strftime("%m%d-%H%M%S")
    log_root = Path(log_root or (C.LOG_DIR / f"pool-{tag}-{stamp}"))
    log_root.mkdir(parents=True, exist_ok=True)
    port = C.POOL_PORT
    import threading
    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))  # make finally-cleanup run on kill
    server = start_server(workspace, port, server_gpu, log_root / "server.log")
    p = None
    done = []  # (name, obs_dir, first_group, runs_done)
    load_report = ""
    try:
        wait_ready(server, port, timeout=900)
        load_report = (log_root / "server.log").read_text().split("loaded ", 1)[-1].split("\n")[0]
        todo = list(groups)
        attempt = 0
        while todo and attempt < 4:
            name = f"pool-{tag}-{stamp}-a{attempt}"
            obs_dir = log_root / f"obs-a{attempt}"
            env = os.environ.copy()
            env.pop("OPENAI_API_KEY", None)
            env.update(OMNI_KIT_ACCEPT_EULA="Y", VK_ICD_FILENAMES=C.VK_ICD,
                       OWSP_OBS_DIR=str(obs_dir))
            env.pop("OWSP_PROMPT", None)
            if prompt:
                env["OWSP_PROMPT"] = prompt
            with open(log_root / f"sim-a{attempt}.log", "w") as f:
                p = subprocess.Popen(sim_cmd(task, name, port, sim_gpu, envs, len(todo), todo[0], env_seed,
                                             video_mode, rendering_type),
                                     cwd=ROBOLAB, env=env, stdout=f, stderr=subprocess.STDOUT)
                while p.poll() is None:
                    if server.poll() is not None:
                        p.kill()
                        raise RuntimeError("policy server died")
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
    finally:
        if p is not None and p.poll() is None:
            p.kill()
        if server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
    rows = []
    for name, obs_dir, g0, runs_done in done:
        out = ROBOLAB / "output" / name
        res = out / "episode_results.jsonl"
        if not res.exists():
            continue
        for line in res.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            run, e = int(r["run"]), int(r["env_id"])
            if run not in runs_done:
                continue
            g = g0 + run
            vids = sorted((out / task).glob(f"*_{run}_env{e}.mp4"))
            rows.append({"slot": g * envs + e, "group": g, "env": e, "ok": True,
                         "success": bool(r["success"]), "steps": r.get("episode_step"),
                         "instruction": r.get("instruction"), "task_name": r.get("task_name"),
                         "score": r.get("score"), "reason": r.get("reason"),
                         "out_dir": str(out / task), "run": run,
                         "obs": str(obs_dir / f"obs_r{run}_e{e}.npz"),
                         "video": str(vids[0]) if vids else "",
                         "log": str(out / task / f"log_{run}_env{e}.json")})
    got = {r["slot"] for r in rows}
    for s in slots:
        if s not in got:
            rows.append({"slot": s, "ok": False})
    (log_root / "rows.json").write_text(json.dumps({"server": load_report, "rows": rows}, indent=1))
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
    a = ap.parse_args()
    t0 = time.time()
    rows, rep = run_pool(workspace=a.workspace, task=a.task, tag=a.tag, slots=range(a.slot_start, a.slot_start + a.n),
                         envs=a.envs, rendering_type=a.rendering_type or None)
    ok = [r for r in rows if r["ok"]]
    print(json.dumps({"n": len(rows), "completed": len(ok), "success": sum(r["success"] for r in ok),
                      "wall_min": round((time.time() - t0) / 60, 1), "server": rep[:300]}))


if __name__ == "__main__":
    main()
