"""Build one predeclared new-goal edit; all input cohorts must be development."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
COHORT = HERE.parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
from weightsmith import config as C
from prepare import prepare, sha, CoverageError


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def source_files():
    old = COHORT.parent / "rss_repair_comparison" / "make_eval_plan.py"
    spec = importlib.util.spec_from_file_location("original_source_inventory", old)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = module.runtime_sources()
    files += list(COHORT.rglob("*.py"))
    files += list((COHORT / "runtime/layouts").glob("*.json"))
    files += [COHORT / "protocol.json", old,
              COHORT.parent / "rss_repair_comparison/prepare_shared.py", COHORT.parent / "rss_goal_expansion_v2/adaptation/prepare.py"]
    files += [C.ROBOLAB / "assets/scenes" / name for name in ("bagel_plate_banana_bowl.usda", "butter_raisin_box.usda", "bowls_2_table.usda")]
    return sorted(set(p.resolve() for p in files))


def verify_gate(path, case):
    """Recompute the declared language-gap gate from sealed development batches."""
    path = Path(path).resolve()
    gate = json.loads(path.read_text())
    protocol = json.loads((COHORT / "protocol.json").read_text())
    if gate.get("protocol_sha256") != sha(COHORT / "protocol.json"):
        raise ValueError("Qualification identifies a different frozen protocol")
    if gate.get("case") != case or gate.get("split") != "development":
        raise ValueError("A case-specific development gate is required")
    rounds = gate.get("rounds")
    if not isinstance(rounds, list) or len(rounds) != 2:
        raise ValueError("Both initial and repeat qualification rounds are required")
    hashes = {str(Path(path).resolve()): sha(path)}
    used = set()
    for stage, round_ in zip(("initial", "repeat"), rounds, strict=True):
        scores = {}
        round_slots = None
        for cell in ("original", "teacher"):
            batch = Path(round_[cell]).resolve()
            receipt = json.loads((batch / "completed.json").read_text())
            rows = json.loads((batch / "episodes.json").read_text())
            owner = json.loads((batch.parent / "plan.json").read_text())
            status = json.loads((batch.parent / "status.json").read_text())
            if (status.get("status") != "complete" or owner.get("split") != "development"
                    or owner.get("case") != case or owner.get("stage") != stage
                    or owner.get("inference_batching") != "fixed16"
                    or status.get("completed_batches") != 2 or status.get("completed_episodes") != 32
                    or len(owner["batches"]) != 2 or len(rows) != 16
                    or receipt["episodes_sha256"] != sha(batch / "episodes.json")
                    or receipt["n"] != 16 or receipt["cell"] != cell
                    or any(r.get("case") != case or r.get("split") != "development"
                           or r.get("cell") != cell or r.get("condition") != "base"
                           or r.get("inference_batching") != "fixed16" or not r.get("ok") for r in rows)):
                raise ValueError("Qualification is incomplete or is not an unchanged-policy development batch")
            number = receipt["batch"]
            declared = owner["batches"][number]
            cfg = protocol["cases"][case]
            if (batch.parent.name != stage or batch.parent.parent.name != case
                    or batch.parent.parent.parent.name != "qualification"
                    or declared["slot_start"] != protocol["qualification"]["slot_start"][stage][case]):
                raise ValueError("Qualification owner or noise slots differ from the predeclared stage")
            identity = json.loads((batch.parent / "identity.json").read_text())
            if not identity or any(sha(p) != h for p, h in identity.items()):
                raise ValueError("A qualification source changed after its execution")
            if (declared["cell"] != cell or declared["condition"] != "base" or declared.get("workspace")
                    or declared["task"] != cfg["task"] or declared["effective_prompt"] != cfg[cell]
                    or any(r["effective_prompt"] != cfg[cell] or r.get("task_name") != cfg["task"] for r in rows)):
                raise ValueError("Qualification instructions differ from the frozen goal pair")
            slots = {r["slot"] for r in rows}
            if len(slots) != 16 or slots != set(range(declared["slot_start"], declared["slot_start"] + 16)):
                raise ValueError("Qualification slots differ from its plan")
            if round_slots is not None and slots != round_slots:
                raise ValueError("Original and clearer qualification trials must share planned seeds")
            round_slots = slots
            scores[cell] = sum(bool(r["success"]) for r in rows)
            for p in (batch / "completed.json", batch / "episodes.json", batch.parent / "plan.json", batch.parent / "status.json", batch.parent / "identity.json"):
                hashes[str(p)] = sha(p)
        if used & round_slots:
            raise ValueError("Repeat qualification must use fresh noise slots")
        used |= round_slots
        if scores["teacher"] < 8 or scores["teacher"] - scores["original"] < 4 or scores["original"] > 10:
            raise ValueError(f"New goal failed the predeclared qualification rule: {scores}")
    return hashes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("case", "teacher-batch", "preservation-batch", "qualification", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    gate_identity = verify_gate(args.qualification, args.case)
    gate = json.loads(Path(args.qualification).read_text())
    if (Path(args.teacher_batch).resolve() != Path(gate["rounds"][0]["teacher"]).resolve()
            or Path(args.teacher_batch).resolve() != Path(gate["teacher_batch"]).resolve()
            or Path(args.preservation_batch).resolve() != Path(gate["preservation_batch"]).resolve()):
        raise ValueError("Adaptation must use the predeclared initial teacher and qualified preservation batch")
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    status = {"case": args.case, "split": "development", "status": "preparing",
        "gpu": args.gpu, "started_at": time.time(), "completed_stages": [],
        "heldout_used": False, "closed_loop_evaluation": "performed separately after this edit is frozen"}
    try:
        identity = {str(p): sha(p) for p in source_files()}
        identity.update(gate_identity)
        prepare(args.case, COHORT / "protocol.json", args.teacher_batch, args.preservation_batch, out / "spec.json")
        identity[str(out / "spec.json")] = sha(out / "spec.json")
        save(out / "identity.json", identity)
        for name, checksum in identity.items():
            source = Path(name)
            if source.stat().st_size <= 2 * 1024**2:
                dest = out / "frozen-sources" / checksum / source.name
                dest.parent.mkdir(parents=True, exist_ok=True)
                if not dest.exists():
                    shutil.copyfile(source, dest)
        stages = [
            ("edit-prepare", [sys.executable, str(HERE / "pipeline.py"), "prepare", "--spec", str(out / "spec.json"), "--output", str(out / "edit-prepared")]),
            ("native-targets", [C.OPENPI_PY, str(HERE / "native_targets.py"), "prepare", "--spec", str(out / "spec.json"), "--output", str(out / "native-targets"), "--gpu", str(args.gpu)]),
            ("edit-capture", [sys.executable, str(HERE / "pipeline.py"), "capture", "--prepared", str(out / "edit-prepared"), "--gpu", str(args.gpu), "--python", C.OPENPI_PY, "--lora-dataset", str(out / "native-targets")]),
            ("edit-solve", [sys.executable, str(HERE / "pipeline.py"), "solve", "--prepared", str(out / "edit-prepared"), "--output", str(out / "edit-fit"), "--lora-dataset", str(out / "native-targets")]),
        ]
        env = dict(os.environ, OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
            JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE), JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1")
        for name, command in stages:
            if any(sha(p) != h for p, h in identity.items()):
                raise ValueError("Adaptation sources changed after freeze")
            start = time.time()
            status.update(status="running", stage=name, stage_started_at=start)
            save(out / "status.json", status)
            print(f"START {args.case} {name}", flush=True)
            with (out / f"{name}.log").open("x") as log:
                subprocess.run(command, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                    timeout=3600, check=True)
            if any(sha(p) != h for p, h in identity.items()):
                raise ValueError("Adaptation sources changed during execution")
            status["completed_stages"].append({"stage": name, "wall_seconds": time.time() - start})
            print(f"DONE {args.case} {name}", flush=True)
        workspace = out / "edit-fit/workspace"
        files = [workspace / "workspace.json", *sorted((workspace / "edits").glob("*.npy"))]
        if len(files) != 10:
            raise ValueError("Expected exactly nine merged MLP tensors and the workspace manifest")
        save(out / "candidate-seal.json", {"case": args.case, "split": "development",
            "artifacts": {str(p): sha(p) for p in files}, "spec_sha256": sha(out / "spec.json"),
            "solve_sha256": sha(out / "edit-fit/solve.json"), "source_identity_sha256": sha(out / "identity.json")})
        status.update(status="complete", wall_seconds=time.time() - status["started_at"])
    except CoverageError as exc:
        status.update(status="source_coverage_failed", reason=str(exc), heldout_used=False)
    except BaseException as exc:
        status.update(status="stopped", reason=repr(exc))
        raise
    finally:
        status["wall_seconds"] = time.time() - status["started_at"]
        save(out / "status.json", status)


if __name__ == "__main__":
    main()
