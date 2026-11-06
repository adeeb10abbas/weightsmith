"""Integration test: re-derive the published cube-then-banana edits from their capture with weightsmith.distill.

Skipped unless WSP_DISTILL_PROJECT names the cube-then-banana project root ON THE GPU HOST (the one holding capture
`c2-multi` and candidates `abl-imgl8`, `v2mi-f`); `bin/owsp` must reach the host (WSP_HOST / WSP_REMOTE_REPO in the
environment or wsp.env). For each target it

  1. creates a fresh candidate and builds the edit there with the rendered sandbox script (render_sandbox_script),
  2. re-derives the edit inside the stored candidate's sandbox and compares it value by value with the stored
     tensors (render_compare_script: max |difference|, number of differing values, content digests),
  3. compares `owsp diff` of the fresh and the stored candidate (changed values, max |delta|, L2 of the delta).

  WSP_DISTILL_PROJECT=/home/ali/vibes/opus-weight-surgery/projects/cube-then-banana \\
  WSP_HOST=ali-ms WSP_REMOTE_REPO=weightsmith python3 -m pytest tests/test_distill_reproduce.py -s

Optional: WSP_DISTILL_TARGETS=abl-imgl8,v2mi-f (default both), WSP_DISTILL_REPORT=path.json (write the numbers).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from weightsmith import distill as D  # noqa: E402

PROJECT = os.environ.get("WSP_DISTILL_PROJECT")
pytestmark = pytest.mark.skipif(not PROJECT, reason="set WSP_DISTILL_PROJECT to the cube-then-banana project root on the GPU host")

# How the stored candidates were built (study/results/opus/ctb/v2mi_edit.py, v2mi2_edit.py): capture c2-multi holds
# 6 premature-release decisions x 8 wordings; items 0-5 = original wording, items 12-17 = explicit wording.
TARGETS = {
    "abl-imgl8": dict(mode="image", layers=list(range(8, 17))),
    "v2mi-f": dict(mode="both", layers=list(range(17)), slot_map="pi05"),
}
COMMON = dict(orig_capture="c2-multi", expl_capture="c2-multi", orig_items=list(range(0, 6)),
              expl_items=list(range(12, 18)), alpha=1.0, lam_rel=1e-3,
              instruction="Put the cube then the banana in the bowl",
              explicit_instruction="Put the cube in the bowl then put the banana in the bowl")


def owsp(*args: str, script: str | None = None, tmp: Path | None = None) -> str:
    argv = [str(REPO / "bin" / "owsp"), *args]
    if script is not None:
        path = tmp / f"script-{time.time_ns()}.py"
        path.write_text(script)
        argv.append(str(path))
    p = subprocess.run(argv, env=dict(os.environ, WSP_ROOT=PROJECT), capture_output=True, text=True, timeout=3600)
    assert p.returncode == 0, p.stdout[-3000:] + p.stderr[-3000:]
    return p.stdout + p.stderr


@pytest.mark.parametrize("target", [t for t in os.environ.get("WSP_DISTILL_TARGETS", "abl-imgl8,v2mi-f").split(",") if t])
def test_reproduce_stored_candidate(target, tmp_path):
    cfg = D.BuildConfig(**COMMON, **TARGETS[target])
    fresh = f"rep-{target}-{time.strftime('%m%d-%H%M%S')}"
    owsp("new", fresh, "--note", f"re-derived {target} with weightsmith.distill (tests/test_distill_reproduce.py)")
    t0 = time.time()
    built = D.parse_result(owsp("py", fresh, script=D.render_sandbox_script(cfg), tmp=tmp_path))
    t_build = time.time() - t0
    cmp = D.parse_result(owsp("py", target, script=D.render_compare_script(cfg), tmp=tmp_path))
    diff_fresh = json.loads(owsp("diff", fresh))["edited_tensors"]
    diff_stored = json.loads(owsp("diff", target))["edited_tensors"]

    rows = []
    for b, c in zip(built["layers"], cmp["layers"]):
        name = b["tensor"]
        df, ds = diff_fresh[name], diff_stored[name]
        rows.append({"tensor": name, "fresh_sha256": b["sha256"], "stored_sha256": c["stored_sha256"],
                     "rederived_in_stored_sandbox_sha256": c["sha256"],
                     "max_abs_diff_vs_stored": c["max_abs_diff_vs_stored"], "n_diff_vs_stored": c["n_diff_vs_stored"],
                     "stored_max_abs_delta": c["stored_max_abs_delta"],
                     "owsp_diff_fresh": {k: df[k] for k in ("changed", "max_abs_delta", "l2_delta")},
                     "owsp_diff_stored": {k: ds[k] for k in ("changed", "max_abs_delta", "l2_delta")}})
    report = {"target": target, "fresh_candidate": fresh, "project": PROJECT, "build_seconds": round(t_build, 1),
              "config": cfg.to_dict(), "slot_positions": built.get("slot_positions"),
              "min_slot_consistency": built.get("min_slot_consistency"),
              "max_abs_diff_vs_stored": max(r["max_abs_diff_vs_stored"] for r in rows),
              "n_diff_vs_stored": sum(r["n_diff_vs_stored"] for r in rows),
              "bit_identical_tensors": sum(r["fresh_sha256"] == r["stored_sha256"] for r in rows),
              "tensors": len(rows), "layers": rows}
    print(json.dumps({k: v for k, v in report.items() if k != "layers"}, indent=1))
    if os.environ.get("WSP_DISTILL_REPORT"):
        p = Path(os.environ["WSP_DISTILL_REPORT"])
        allr = json.loads(p.read_text()) if p.exists() else {}
        allr[target] = report
        p.write_text(json.dumps(allr, indent=1))

    assert set(diff_fresh) == set(diff_stored) == {r["tensor"] for r in rows}
    total = sum(diff_stored[r["tensor"]]["changed"] for r in rows)
    # bf16-level agreement: differences only where float rounding of the solve tips a bf16 rounding (one ulp)
    assert report["n_diff_vs_stored"] <= 1e-4 * total
    for r in rows:
        assert r["max_abs_diff_vs_stored"] <= 2 ** -7 * max(1.0, r["stored_max_abs_delta"]) * 2
        assert r["fresh_sha256"] == r["rederived_in_stored_sandbox_sha256"]  # the build is deterministic
        assert abs(r["owsp_diff_fresh"]["changed"] - r["owsp_diff_stored"]["changed"]) <= 1e-4 * r["owsp_diff_stored"]["changed"]
