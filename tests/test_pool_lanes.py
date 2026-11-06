"""Offline test of the eval-pool lane logic with a fake policy server and a fake Isaac process.
Run: python3 tests/test_pool_lanes.py  or  python3 -m pytest tests   (no GPU, no OpenPI/RoboLab needed)"""
import json, os, subprocess, sys, tempfile, time
from pathlib import Path

repo = Path(__file__).resolve().parents[1]
tmp = Path(tempfile.mkdtemp())
os.environ.update(WSP_LOG_DIR=str(tmp / "logs"), ROBOLAB_DIR=str(tmp / "robolab"), SIM_PYTHON=sys.executable,
                  WSP_LOCK_DIR=str(tmp / "locks"))
(tmp / "robolab" / "output").mkdir(parents=True)
sys.path.insert(0, str(repo))
from weightsmith import pool  # noqa: E402

# fake sim: argv parsing compatible with sim_cmd(); crashes once on lane 1 attempt 0 after 1 run (tests retry)
FAKE = tmp / "fake_sim.py"
FAKE.write_text(r'''
import sys, json, os, pathlib
a = sys.argv; g = lambda k: a[a.index(k) + 1]
name, envs, runs, g0 = g("--output-folder-name"), int(g("--num-envs")), int(g("--num-runs")), int(g("--sampling-seed-base"))
out = pathlib.Path(os.environ["ROBOLAB_DIR"]) / "output" / name
(out / g("--task")).mkdir(parents=True, exist_ok=True)
crash = "l1a0" in name
with open(out / "episode_results.jsonl", "w") as f:
    for r in range(runs):
        if crash and r == 1:
            sys.exit(3)
        for e in range(envs):
            f.write(json.dumps({"run": r, "env_id": e, "success": (g0 + r + e) % 3 == 0, "episode_step": 100,
                                "instruction": "x", "task_name": g("--task")}) + "\n")
''')
pool.HARNESS = tmp
(tmp / "run_batched.py").write_text(FAKE.read_text())


class FakeServer:
    def __init__(self, log): Path(log).write_text('loaded {"workspace": null}\n'); self.rc = None
    def poll(self): return self.rc
    def send_signal(self, s): self.rc = 0
    def wait(self, t=None): return 0
    def kill(self): self.rc = -9


pool.start_server = lambda ws, port, gpu, log: FakeServer(log)
pool.wait_ready = lambda *a, **k: None


def test_pool_lanes():
    rows, rep = pool.run_pool(workspace=None, task="T", tag="t", slots=range(160, 160 + 16 * 5), envs=16,
                              server_gpu=1, sim_gpu=1, extra_gpus=(0,), rendering_type=None)
    ok = [r for r in rows if r["ok"]]
    assert len(rows) == 80 and len(ok) == 80, (len(rows), len(ok))
    assert sorted(r["slot"] for r in ok) == list(range(160, 240))
    assert {r["lane"] for r in ok} == {0, 1}
    exp = {s: ((s // 16) + (s % 16)) % 3 == 0 for s in range(160, 240)}
    assert all(r["success"] == exp[r["slot"]] for r in ok), "slot->seed mapping broken across lanes/retries"
    print("lane test OK:", sum(r["success"] for r in ok), "/ 80 successes; lanes", sorted({r['lane'] for r in ok}), "; rep", rep)


if __name__ == "__main__":
    test_pool_lanes()
