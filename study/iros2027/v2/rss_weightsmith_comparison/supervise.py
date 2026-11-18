"""Durably sequence fixed LoRA selection, full candidate freeze, test, and report."""
from __future__ import annotations

import argparse
import fcntl
import subprocess
import sys
import time

from common import HERE, ROOT, read, save, sha, validate_plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    parser.add_argument("--poll-seconds", type=float, default=15)
    args = parser.parse_args()
    if args.poll_seconds < 1 or args.poll_seconds > 60:
        raise ValueError("Polling interval must be between 1 and 60 seconds")
    if not (ROOT / "design/design.json").is_file():
        raise ValueError("Prepare and review the comparison design before starting the supervisor")
    lock = (ROOT / "supervisor.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status = {"status": "waiting_for_lora_candidates", "gpu": args.gpu, "started_unix": time.time(),
              "max_selection_episodes": 48, "max_test_episodes": 320}
    def state(**updates):
        status.update(updates, updated_unix=time.time())
        save(ROOT / "supervisor-status.json", status)
    def run(stage, command):
        state(status="running", stage=stage, command=command)
        with (ROOT / f"supervisor-{stage}.log").open("a") as stream:
            subprocess.run(command, cwd=HERE.parents[3], stdout=stream, stderr=subprocess.STDOUT, check=True)
    try:
        state()
        while not (ROOT / "lora-adaptation/candidates.json").is_file():
            prep_state = ROOT / "lora-adaptation/status.json"
            if prep_state.is_file() and read(prep_state).get("status") in {"failed", "stopped"}:
                raise RuntimeError("LoRA preparation/training stopped; inspect preserved artifacts")
            time.sleep(args.poll_seconds)
            state()
        selection_plan = ROOT / "selection-frozen/plan.json"
        if not selection_plan.exists():
            run("prepare-selection", [sys.executable, str(HERE / "prepare.py"), "selection"])
        validate_plan(read(selection_plan))
        if not (ROOT / "selection/status.json").exists() or read(ROOT / "selection/status.json")["status"] != "complete":
            run("selection", [sys.executable, str(HERE / "worker.py"), "--plan", str(selection_plan),
                              "--root", str(ROOT / "selection"), "--gpu", str(args.gpu)])
        heldout_plan = ROOT / "heldout-frozen/plan.json"
        if not heldout_plan.exists():
            run("freeze-heldout", [sys.executable, str(HERE / "prepare.py"), "heldout"])
        validate_plan(read(heldout_plan))
        if not (ROOT / "evaluation/status.json").exists() or read(ROOT / "evaluation/status.json")["status"] != "complete":
            run("evaluation", [sys.executable, str(HERE / "worker.py"), "--plan", str(heldout_plan),
                               "--root", str(ROOT / "evaluation"), "--gpu", str(args.gpu)])
        report = ROOT / "analysis-v1/report.json"
        if not report.exists():
            run("analysis", [sys.executable, str(HERE / "analyze.py")])
        data = read(report)
        if data["status"] != "complete_and_verified" or data["new_test_policy_episodes"] != 320:
            raise ValueError("Final report is not complete")
        state(status="complete_with_verified_report", stage="complete", report=str(report), report_sha256=sha(report),
              completed_selection_episodes=48, completed_test_episodes=320)
    except BaseException as exc:
        state(status="stopped", reason=repr(exc))
        raise


if __name__ == "__main__":
    main()
