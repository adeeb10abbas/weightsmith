"""Batched RoboLab rollouts for pi0.5 (base or edited workspace).

One policy server + one Isaac sim process. Sim runs num_envs parallel envs.
Reuses the vendored lab serve path (weightsmith.lab: exact edit application + hashing).
"""
from __future__ import annotations

import argparse, json, os, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402

CHECKPOINT = C.CHECKPOINT
OPENPI = C.OPENPI_DIR
ROBOLAB = C.ROBOLAB
SIM_PY = C.SIM_PY
OPENPI_PY = C.OPENPI_PY
HARNESS = Path(__file__).resolve().parent


def wait_ready(proc, port, timeout=300):
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


def start_server(workspace, port, gpu, log):
    cmd = [str(OPENPI_PY), str(HARNESS / "batched_server.py"), "--port", str(port)]
    if workspace is not None:
        cmd += ["--workspace", str(Path(workspace).resolve())]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.update(JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE), JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
               CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE="true", XLA_PYTHON_CLIENT_MEM_FRACTION=C.SERVER_MEM_FRACTION,
               PYTHONPATH=str(C.REPO) + os.pathsep + env.get("PYTHONPATH", ""))
    return subprocess.Popen(cmd, cwd=OPENPI, env=env, stdout=open(log, "w"), stderr=subprocess.STDOUT)


def run_sim(task, name, port, gpu, num_envs, num_runs, env_seed, samp_seed, log_dir, record_images=False, video_mode="all", rendering_type=None):
    cmd = [str(SIM_PY), str(HARNESS / "run_batched.py"), "--policy", "pi05", "--task", *task,
           "--environment-seed", str(env_seed), "--sampling-seed-base", str(samp_seed),
           "--remote-port", str(port), "--num-envs", str(num_envs), "--num-runs", str(num_runs),
           "--headless", "--video-mode", video_mode, "--output-folder-name", name, "--device", f"cuda:{gpu}",
           "--/rtx/verifyDriverVersion/enabled=false", "--/renderer/multiGpu/enabled=false",
           "--/renderer/multiGpu/autoEnable=false", f"--/renderer/activeGpu={gpu}"]
    if record_images:
        cmd.append("--record-image-data")
    if rendering_type:
        cmd += ["--rendering-type", rendering_type]
    env = os.environ.copy()
    env.pop("OPENAI_API_KEY", None)
    env.update(OMNI_KIT_ACCEPT_EULA="Y", VK_ICD_FILENAMES=C.VK_ICD)
    with open(log_dir / "sim.log", "w") as f:
        return subprocess.run(cmd, cwd=ROBOLAB, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=6 * 3600).returncode


def collect(name):
    out = ROBOLAB / "output" / name
    rows = [json.loads(l) for l in (out / "episode_results.jsonl").read_text().splitlines() if l.strip()]
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default=None)
    ap.add_argument("--task", nargs="+", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--port", type=int, default=8100)
    ap.add_argument("--server-gpu", type=int, default=0)
    ap.add_argument("--sim-gpu", type=int, default=1)
    ap.add_argument("--num-envs", type=int, default=4)
    ap.add_argument("--num-runs", type=int, default=1)
    ap.add_argument("--env-seed", type=int, default=0)
    ap.add_argument("--sampling-seed", type=int, required=True)
    ap.add_argument("--record-images", action="store_true")
    ap.add_argument("--video-mode", default="all")
    ap.add_argument("--rendering-type", default=None)
    ap.add_argument("--external-server", action="store_true", help="reuse an already-running server on --port")
    a = ap.parse_args()
    log_dir = C.LOG_DIR / a.name
    log_dir.mkdir(parents=True, exist_ok=False)
    t0 = time.time()
    server = None
    if not a.external_server:
        server = start_server(a.workspace, a.port, a.server_gpu, log_dir / "server.log")
    try:
        if server:
            wait_ready(server, a.port)
        t1 = time.time()
        rc = run_sim(a.task, a.name, a.port, a.sim_gpu, a.num_envs, a.num_runs, a.env_seed, a.sampling_seed, log_dir, a.record_images, a.video_mode, a.rendering_type)
        t2 = time.time()
    finally:
        if server and server.poll() is None:
            server.send_signal(signal.SIGINT)
            try:
                server.wait(30)
            except subprocess.TimeoutExpired:
                server.kill()
    rows = collect(a.name) if rc == 0 or (ROBOLAB / "output" / a.name / "episode_results.jsonl").exists() else []
    summary = {"name": a.name, "workspace": a.workspace, "task": a.task, "rc": rc,
               "server_s": round(t1 - t0, 1), "sim_s": round(t2 - t1, 1),
               "n": len(rows), "successes": sum(bool(r.get("success")) for r in rows),
               "episodes": [{k: r.get(k) for k in ("task_name", "run", "episode", "env_id", "success", "episode_step", "score", "reason")} for r in rows]}
    (log_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}))


if __name__ == "__main__":
    main()
