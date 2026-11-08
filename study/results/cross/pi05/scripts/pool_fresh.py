"""Track C' (fresh pi0.5 DROID control): closed-loop worker for ali-ms GPU0, default render preset.

    cd ~/cross/pi05 && setsid nohup python3 scripts/pool_fresh.py --gpu 0 --port 8720 \
        --queue fresh/jobs/queue.json --root fresh --until 2026-11-08T23:00Z > fresh/worker-g0.log 2>&1 < /dev/null &

Same slots, scenes and noise seeds as Tracks A (Cosmos3 Edge) and B (GR00T N1.7): slot s = 16 g + j, group g is the
sampling-seed base, slot s gets the noise stream ((g * 64 + j) * 100 + k) (weightsmith/batched_client.py), environment
seed 0, and every slot is the FIRST episode of a fresh Isaac process (one process per 16-slot group, num_runs 1; a later
episode of a process starts from another scene). With --envs 8 a group runs as two fresh 8-env processes and
OWSP_ENV_OFFSET (patched into this host's batched_client.py) keeps each slot's seeds identical to a 16-env process,
exactly as Track A runs Cosmos.

Render preset: RoboLab's default (no --rendering-type flag; Isaac Lab's RenderCfg default), recorded in every record.

Differences from pool_pi05.py (weightsmith run_pool, one call per group): the policy server is started once per run of
consecutive queue jobs with the same weights (workspace) and kept up across their groups, so only the Isaac process
is fresh per group; weights are verified once per server (candidate fingerprint recomputed on disk before start, the
server's load report checked after). The server is stateless across requests (every request carries its seed).

Locks: the coordinator's ~/.cache/weightsmith/locks/gpu<g>.lock, then weightsmith's WSP_LOCK_DIR gpu<g>.lock, held for
one server run (a batch of up to --batch jobs) and released between batches.
Output: <root>/runs/<tag>.json (one row per slot), <root>/runs/<tag>.failN.json, <root>/ledger.jsonl, <root>/logs/.
"""
from __future__ import annotations

import argparse
import datetime as dt
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
import traceback
from pathlib import Path

HOME = Path.home()
WS_REPO = HOME / "cross/pi05/weightsmith"
LOCK_DIRS = [HOME / ".cache/weightsmith/locks", HOME / "vibes/opus-weight-surgery/state"]
GROUP = 16
EP_SECONDS = {"T1": 60, "T2": 30, "T3": 30, "T4": 60, "T5": 30, "T6": 60, "T7": 30, "T8": 40, "R1": 40}


def log(msg):
    print(time.strftime("%m-%d %H:%M:%S", time.gmtime()) + "Z", msg, flush=True)


def fingerprint(cand_dir: Path) -> str:
    h = hashlib.sha256()
    for f in sorted((cand_dir / "edits").glob("*.npy")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def lock_gpu(gpu):
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


def port_free(port):
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) != 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--queue", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--envs", type=int, default=16, help="envs per Isaac process (16, or 8 = two processes per group)")
    ap.add_argument("--batch", type=int, default=4, help="max consecutive same-weights jobs per server run")
    ap.add_argument("--until", default=None, help="UTC deadline (ISO, e.g. 2026-11-08T23:00Z): no job starts that "
                                                 "would not finish by then (estimate from finished jobs)")
    ap.add_argument("--max-fail", type=int, default=2)
    a = ap.parse_args()
    root = Path(a.root).expanduser().resolve()
    os.environ["WSP_POOL_PORT"] = str(a.port)
    os.environ.setdefault("WSP_ENV_FILE", str(WS_REPO / "wsp.env"))
    sys.path.insert(0, str(WS_REPO))
    from weightsmith import config as C  # noqa: E402  (loads wsp.env into os.environ: WSP_TASK_DIRS for T8)
    from weightsmith.rollout import start_server, wait_ready  # noqa: E402
    from weightsmith.pool import sim_cmd  # noqa: E402
    if GROUP % a.envs:
        raise SystemExit("envs must divide 16")
    deadline = None
    if a.until:
        deadline = dt.datetime.fromisoformat(a.until.replace("Z", "+00:00")).timestamp()
    runs, logs = root / "runs", root / "logs"
    for d in (runs, logs):
        d.mkdir(parents=True, exist_ok=True)
    client_src = (WS_REPO / "weightsmith/batched_client.py").read_text()
    if a.envs != GROUP and "OWSP_ENV_OFFSET" not in client_src:
        raise SystemExit("batched_client.py lacks the OWSP_ENV_OFFSET patch; refusing to run with envs < 16")
    log(f"worker gpu={a.gpu} port={a.port} envs={a.envs} queue={a.queue} root={root} task_dirs="
        f"{os.environ.get('WSP_TASK_DIRS')} vk_icd={C.VK_ICD} server_mem={C.SERVER_MEM_FRACTION} deadline={a.until}")

    def pending():
        out = []
        for j in json.loads(Path(a.queue).read_text()):
            if (runs / f"{j['tag']}.json").exists():
                continue
            if len(list(runs.glob(f"{j['tag']}.fail*.json"))) >= a.max_fail:
                continue
            out.append(j)
        return out

    def est_seconds(code):
        """Wall time of one job of this task from finished jobs (per second of episode), else a prior."""
        rates = []
        for f in runs.glob("*.json"):
            if ".fail" in f.name:
                continue
            r = json.loads(f.read_text())
            if r.get("job_s"):
                rates.append(r["job_s"] / EP_SECONDS[r["code"]])
        rate = max(rates) if rates else 10.0
        return rate * EP_SECONDS[code] + 30

    def run_group(job, g, off, port, log_root, attempt):
        name = f"x05f-{job['tag']}-g{g}o{off}a{attempt}-{time.strftime('%m%d-%H%M%S')}"
        env = os.environ.copy()
        for k in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OWSP_PROMPT"):
            env.pop(k, None)
        env.update(OMNI_KIT_ACCEPT_EULA="Y", VK_ICD_FILENAMES=C.VK_ICD, PYTHONUNBUFFERED="1",
                   OWSP_OBS_DIR=str(log_root / f"obs-g{g}o{off}a{attempt}"), OWSP_ENV_OFFSET=str(off))
        if job.get("prompt"):
            env["OWSP_PROMPT"] = job["prompt"]
        cmd = sim_cmd(job["task"], name, port, a.gpu, a.envs, 1, g, 0, job.get("video_mode") or "sensor",
                      job.get("rendering_type") or None)
        slog = log_root / f"sim-g{g}o{off}a{attempt}.log"
        with open(slog, "w") as f:
            p = subprocess.Popen(cmd, cwd=C.ROBOLAB, env=env, stdout=f, stderr=subprocess.STDOUT,
                                 start_new_session=True)
        return p, name, slog, cmd

    def collect(job, g, off, name):
        outp = C.ROBOLAB / "output" / name
        res = outp / "episode_results.jsonl"
        rows = []
        if res.exists():
            for line in res.read_text().splitlines():
                if not line.strip():
                    continue
                rr = json.loads(line)
                if int(rr["run"]) != 0:
                    continue
                e = int(rr["env_id"])
                vids = sorted((outp / job["task"]).glob(f"*_0_env{e}.mp4"))
                rows.append({"slot": g * GROUP + off + e, "group": g, "env": e, "env_offset": off, "ok": True,
                             "success": bool(rr["success"]), "steps": rr.get("episode_step"),
                             "instruction": rr.get("instruction"), "score": rr.get("score"),
                             "reason": rr.get("reason"), "events": rr.get("events"), "out_dir": str(outp),
                             "video": str(vids[0]) if vids else ""})
        return rows

    while True:
        if (root / "STOP").exists():
            log("STOP file present: exiting")
            return
        todo = pending()
        if not todo:
            log("queue empty: exiting")
            return
        first = todo[0]
        if deadline and time.time() + est_seconds(first["code"]) + 90 > deadline:
            log(f"deadline: next job {first['tag']} (est {est_seconds(first['code']):.0f}s) would end after {a.until}")
            return
        batch = [first]
        for j in todo[1:]:
            if len(batch) >= a.batch or j.get("workspace") != first.get("workspace"):
                break
            batch.append(j)
        ws = first.get("workspace")
        t_req = time.time()
        locks = lock_gpu(a.gpu)
        t_lock = time.time()
        server = None
        try:
            log(f"locks gpu{a.gpu} after {t_lock - t_req:.0f}s; batch {[j['tag'] for j in batch]}")
            fp = None
            if ws:
                fp = fingerprint(Path(ws))
                if fp != first["fp"]:
                    raise RuntimeError(f"fingerprint {fp} != expected {first['fp']} for {ws}")
            if not port_free(a.port):
                raise RuntimeError(f"port {a.port} busy")
            stamp = time.strftime("%m%d-%H%M%S")
            srv_log = logs / f"server-{first['tag']}-{stamp}.log"
            t_s0 = time.time()
            server = start_server(ws, a.port, a.gpu, srv_log)
            wait_ready(server, a.port, timeout=900)
            t_s1 = time.time()
            rep_line = srv_log.read_text().split("batched_server:loaded ", 1)[-1].split("\n")[0]
            try:
                srv_rep = json.loads(rep_line)
            except ValueError:
                srv_rep = {"raw": rep_line[:2000]}
            loaded = srv_rep.get("workspace")
            if ws and (loaded is None or Path(loaded).resolve() != Path(ws).resolve()):
                raise RuntimeError(f"server loaded {loaded!r}, expected {ws}")
            if not ws and loaded is not None:
                raise RuntimeError(f"base batch but server loaded {loaded!r}")
            log(f"server up in {t_s1 - t_s0:.0f}s: {json.dumps(srv_rep)[:200]}")
            for job in batch:
                if (root / "STOP").exists():
                    break
                if (runs / f"{job['tag']}.json").exists():
                    continue
                if deadline and time.time() + est_seconds(job["code"]) > deadline:
                    log(f"deadline: skipping {job['tag']}")
                    break
                if job["n"] != GROUP or job["slot_start"] % GROUP or job.get("workspace") != ws:
                    raise ValueError(f"bad job {job['tag']}")
                g = job["slot_start"] // GROUP
                log_root = logs / f"{job['tag']}-{time.strftime('%m%d-%H%M%S')}"
                log_root.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                rows, procs_info = [], []
                try:
                    for off in range(0, GROUP, a.envs):
                        got = []
                        for attempt in range(2):  # a crashed process is rerun once, in another fresh process
                            p, name, slog, cmd = run_group(job, g, off, a.port, log_root, attempt)
                            while p.poll() is None:
                                if server.poll() is not None:
                                    os.killpg(p.pid, signal.SIGKILL)
                                    raise RuntimeError("policy server died")
                                time.sleep(3)
                            got = collect(job, g, off, name)
                            procs_info.append({"group": g, "env_offset": off, "attempt": attempt, "rc": p.returncode,
                                               "rows": len(got), "robolab_output": name, "sim_log": str(slog)})
                            if len(got) == a.envs:
                                break
                            log(f"{job['tag']} g{g}o{off} attempt {attempt}: {len(got)}/{a.envs} rows, rc={p.returncode}")
                        rows += got
                except Exception as e:  # noqa: BLE001
                    k = len(list(runs.glob(f"{job['tag']}.fail*.json")))
                    (runs / f"{job['tag']}.fail{k}.json").write_text(json.dumps(
                        {**job, "error": repr(e), "traceback": traceback.format_exc(), "procs": procs_info,
                         "rows": rows, "t": time.time()}, indent=1, default=str))
                    log(f"FAILED {job['tag']}: {e!r}")
                    raise
                t1 = time.time()
                ok = [r for r in rows if r["ok"]]
                videos = {}
                for r in ok:
                    if r.get("video"):
                        dst = root / "videos" / job["code"] / job["arm"] / f"{r['slot']}.mp4"
                        dst.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            if dst.exists():
                                dst.unlink()
                            os.link(r["video"], dst)
                        except OSError:
                            shutil.copy2(r["video"], dst)
                        videos[str(r["slot"])] = str(dst)
                rec = {**job, "status": "fresh", "render_preset": job.get("rendering_type") or "default",
                       "rendering_flag": (["--rendering-type", job["rendering_type"]] if job.get("rendering_type")
                                          else "none (RoboLab default preset)"),
                       "envs_per_process": a.envs, "processes": procs_info, "fp_checked": fp,
                       "server": srv_rep, "server_fingerprint": srv_rep.get("fingerprint"),
                       "server_log": str(srv_log), "server_started": t_s0, "server_ready_s": round(t_s1 - t_s0, 1),
                       "rows": sorted(rows, key=lambda r: r["slot"]), "completed": len(ok),
                       "success": sum(bool(r["success"]) for r in ok), "started": t0, "finished": t1,
                       "job_s": round(t1 - t0, 1), "gpu": a.gpu, "port": a.port, "host": socket.gethostname(),
                       "gpu_name": "NVIDIA GeForce RTX 3090", "log_root": str(log_root), "videos": videos,
                       "sim_cmd_example": cmd}
                if len(ok) != GROUP or sorted(r["slot"] for r in ok) != list(range(job["slot_start"],
                                                                                   job["slot_start"] + GROUP)):
                    k = len(list(runs.glob(f"{job['tag']}.fail*.json")))
                    (runs / f"{job['tag']}.fail{k}.json").write_text(json.dumps(
                        {**rec, "error": f"only {len(ok)}/16 slots"}, indent=1, default=str))
                    log(f"FAILED {job['tag']}: only {len(ok)}/16 slots; exiting (a monitored run: look first)")
                    (root / "STOP").touch()
                    break
                (runs / f"{job['tag']}.json").write_text(json.dumps(rec, indent=1, default=str))
                log(f"done {job['tag']}: {rec['success']}/16 in {t1 - t0:.0f}s")
        except Exception as e:  # noqa: BLE001
            log(f"batch error: {e!r}; exiting (a monitored run: look first)")
            (root / "STOP").touch()
        finally:
            if server is not None and server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(30)
                except subprocess.TimeoutExpired:
                    server.kill()
            t_rel = time.time()
            unlock(locks)
            with open(root / "ledger.jsonl", "a") as f:
                f.write(json.dumps({"batch": [j["tag"] for j in batch], "gpu": a.gpu, "lock_requested": t_req,
                                    "lock_acquired": t_lock, "released": t_rel,
                                    "held_s": round(t_rel - t_lock, 1)}) + "\n")
        time.sleep(2)


if __name__ == "__main__":
    main()
