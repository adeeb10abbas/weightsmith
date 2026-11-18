"""Continue the bounded goal study after qualification, preserving every attempt."""
from __future__ import annotations
import argparse
import fcntl
import json
from pathlib import Path
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "advance.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    status_path = root / "advance-status.json"
    state = {"status": "waiting_qualification", "gpu": args.gpu, "started_at": time.time()}
    def save(**changes):
        state.update(changes, updated_at=time.time())
        tmp = status_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(status_path)
    def execute(stage, command):
        save(status="running", stage=stage)
        print("START " + stage, flush=True)
        subprocess.run(command, cwd=REPO, check=True)
        print("DONE " + stage, flush=True)
    try:
        save()
        while True:
            path = root / "qualification-status.json"
            if path.exists():
                q = read(path)
                if q["status"] == "stopped":
                    raise RuntimeError("Qualification stopped; preserve evidence and inspect before continuing")
                if q["status"] == "complete":
                    break
                save(qualification_case=q.get("active_case"), qualification_stage=q.get("active_stage"))
            time.sleep(30)
        selected = q["selected"]
        from report import verify_qualification
        q, _ = verify_qualification(root, read(HERE / "protocol.json"), {})
        selected = q["selected"]
        save(selected_cases=selected)
        for case in selected:
            gate_path = root / "qualification" / case / "qualified.json"
            gate = read(gate_path)
            output = root / f"{case}-adaptation"
            if output.exists():
                if not (output / "status.json").exists() or read(output / "status.json")["status"] not in ("complete", "source_coverage_failed"):
                    raise RuntimeError("Incomplete adaptation preserved; no automatic restart")
                continue
            execute(f"adapt-{case}", [sys.executable, str(HERE / "adaptation/run.py"),
                "--case", case, "--teacher-batch", gate["teacher_batch"],
                "--preservation-batch", gate["preservation_batch"], "--qualification", str(gate_path),
                "--output", str(output), "--gpu", str(args.gpu)])
        constructed = [case for case in selected if read(root / f"{case}-adaptation/status.json")["status"] == "complete"]
        save(constructed_cases=constructed)
        if constructed:
            frozen = root / "heldout-frozen-v1"
            if not frozen.exists():
                execute("freeze-all-goals", [sys.executable, str(HERE / "evaluate.py"),
                    "--root", str(root), "--output", str(frozen)])
            for case in constructed:
                owner = root / f"{case}-heldout-v1"
                if owner.exists() and not (owner / "status.json").exists():
                    raise RuntimeError("Incomplete test owner has no status; preserve for inspection")
                if owner.exists() and read(owner / "status.json")["status"] == "stopped":
                    raise RuntimeError("Stopped test attempt preserved; no automatic restart")
                execute(f"evaluate-{case}", [sys.executable, str(HERE / "worker.py"),
                    "--plan", str(frozen / f"{case}-plan.json"), "--root", str(owner), "--gpu", str(args.gpu)])
        output = root / "analysis-v1/report.json"
        if output.exists():
            raise FileExistsError("Preserve the existing final report")
        execute("audit-complete-study", [sys.executable, str(HERE / "report.py"), "--root", str(root), "--output", str(output)])
        save(status="complete", report=str(output), counts=read(output)["counts"])
    except BaseException as exc:
        save(status="stopped", reason=repr(exc))
        raise


if __name__ == "__main__":
    main()
