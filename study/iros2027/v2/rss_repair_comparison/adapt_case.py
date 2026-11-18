"""Run both frozen adaptation recipes on a shared, completed development set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import shutil

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from weightsmith import config as C
from prepare_shared import prepare
from make_eval_plan import runtime_sources, sha


def save(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--case", choices=["temporal", "spatial"], required=True)
    p.add_argument("--gpu", type=int, choices=[0, 1], required=True)
    p.add_argument("--replacement-batch", type=Path, action="append", default=[])
    a = p.parse_args()
    run = a.run_root.resolve()
    out = run / f"{a.case}-adaptation"
    if out.exists():
        raise FileExistsError("Preserve interrupted adaptation; inspect before an explicit new run")
    out.mkdir()
    status = {"case": a.case, "gpu": a.gpu, "status": "preparing", "started_at": time.time(),
              "closed_loop_evaluation": "not performed by this command", "completed_stages": []}
    env = os.environ.copy()
    env.update(OPENBLAS_NUM_THREADS="4", OMP_NUM_THREADS="4",
               JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE),
               JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1")
    try:
        identity = {str(path): sha(path) for path in runtime_sources()}
        save(out / "identity.json", identity)
        for name, checksum in identity.items():
            source = Path(name)
            if source.stat().st_size <= 2 * 1024**2:
                target = out / "frozen-sources" / checksum / source.name
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copyfile(source, target)
        replacements = list(a.replacement_batch)
        if a.case == "temporal":
            q = run / "temporal-qualification-v1"
            qstatus = json.loads((q / "status.json").read_text())
            if qstatus["status"] != "complete":
                raise ValueError("Temporal full-horizon qualification must finish before adaptation")
            replacements.append(q / "batch-00-cube_only-base")
        prepare(run / f"{a.case}-development-v2", out / "spec.json", replacements)
        identity[str(out / "spec.json")] = sha(out / "spec.json")
        save(out / "identity.json", identity)
        stages = [
            ("edit-prepare", [sys.executable, str(HERE / "edit/pipeline.py"), "prepare", "--spec", str(out / "spec.json"), "--output", str(out / "edit-prepared")]),
            ("lora-targets", [C.OPENPI_PY, str(HERE / "lora/train_lora.py"), "prepare", "--spec", str(out / "spec.json"), "--output", str(out / "lora-dataset"), "--gpu", str(a.gpu)]),
            ("edit-capture", [sys.executable, str(HERE / "edit/pipeline.py"), "capture", "--prepared", str(out / "edit-prepared"), "--gpu", str(a.gpu), "--python", C.OPENPI_PY, "--lora-dataset", str(out / "lora-dataset")]),
            ("edit-solve", [sys.executable, str(HERE / "edit/pipeline.py"), "solve", "--prepared", str(out / "edit-prepared"), "--output", str(out / "edit-fit"), "--lora-dataset", str(out / "lora-dataset")]),
            ("lora-train", [C.OPENPI_PY, str(HERE / "lora/train_lora.py"), "train", "--dataset", str(out / "lora-dataset"), "--output", str(out / "lora-fit"), "--gpu", str(a.gpu)]),
        ]
        candidate_seals = {}
        for stage, cmd in stages:
            if any(sha(path) != checksum for path, checksum in identity.items()):
                raise ValueError("Adaptation source or input specification changed after freeze")
            t = time.time()
            status.update(status="running", stage=stage, stage_started_at=t)
            save(out / "status.json", status)
            print(f"START {a.case} {stage}", flush=True)
            with (out / f"{stage}.log").open("w") as f:
                subprocess.run(cmd, cwd=C.REPO, env=env, stdout=f, stderr=subprocess.STDOUT,
                               check=True, timeout=3600)
            if any(sha(path) != checksum for path, checksum in identity.items()):
                raise ValueError("Adaptation source/input changed during stage; preserve this attempt")
            status["completed_stages"].append({"stage": stage, "wall_seconds": time.time() - t})
            method = {"edit-solve": "closed_form", "lora-train": "lora"}.get(stage)
            if method:
                workspace = out / ("edit-fit" if method == "closed_form" else "lora-fit") / "workspace"
                hashes = {}
                for path in [workspace / "workspace.json", *sorted((workspace / "edits").glob("*.npy"))]:
                    h = hashlib.sha256()
                    with path.open("rb") as f:
                        while chunk := f.read(8 << 20):
                            h.update(chunk)
                    hashes[str(path)] = h.hexdigest()
                if len(hashes) != 10:
                    raise ValueError("Expected workspace metadata and nine merged tensors")
                candidate_seals[method] = {"artifacts": hashes, "sealed_after_stage": stage,
                                           "stage_log_sha256": hashlib.sha256((out / f"{stage}.log").read_bytes()).hexdigest()}
                save(out / "candidate-seals.json", candidate_seals)
            print(f"DONE {a.case} {stage}", flush=True)
        status.update(status="complete", wall_seconds=time.time() - status["started_at"])
    except BaseException as exc:
        status.update(status="stopped", reason=repr(exc))
        raise
    finally:
        save(out / "status.json", status)


if __name__ == "__main__":
    main()
