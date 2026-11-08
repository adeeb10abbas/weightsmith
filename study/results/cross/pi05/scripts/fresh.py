"""Track C' (fresh pi0.5, ali-ms GPU0, default render preset): metrics from the per-job records of pool_fresh.py.

    rsync -a ali-ms:cross/pi05/fresh/runs/ ../results/fresh/runs/     # (sync_fresh.sh)
    python3 fresh.py                                                  # -> ../results/fresh/summary.json,
                                                                      #    fresh records in ../../metrics_pi05.json

Every comparison pairs two arms on the same slots (same scene, same noise seeds) and uses only 16-slot groups that both
arms completed. Statistics from weightsmith.stats: exact two-sided McNemar on the discordant pairs, Wilson 95%
intervals, paired bootstrap of the difference (20,000 draws, seed 0). Records written here carry
"status": "fresh", "run": "C-prime-default" and replace earlier records with the same "run" (provisional records and the
earlier performance-preset partial record are left untouched).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = PI05.parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))
from weightsmith import stats as S  # noqa: E402
from tasks import TASKS, TEACHERS, CANDS  # noqa: E402

RUNS = Path(os.environ.get("XPI05_FRESH_RUNS") or PI05 / "results/fresh/runs")
METRICS = Path(os.environ.get("XPI05_METRICS") or PI05.parent / "metrics_pi05.json")
SUMMARY = Path(os.environ.get("XPI05_SUMMARY") or PI05 / "results/fresh/summary.json")
RUN_ID = "C-prime-default"
MODEL = "pi05_droid_jointpos (OpenPI pi0.5 DROID, joint position; gs://openpi-assets-simeval/pi05_droid_jointpos)"
HW = "ali-ms GPU0, RTX 3090 24 GB; weightsmith @ ad31044 (+ OWSP_ENV_OFFSET client patch, inactive at 16 envs)"
EDIT_INFO = {
    "v2mi-f": {"edit_location": "VLM (PaliGemma) MLPs, layers 0-16: prompt + state + image keys",
               "values_changed": 450468837,
               "values_changed_note": "13.4% of pi0.5's parameters as dense entries; closed-form distillation of the "
                                      "explicit wording 'Put the cube in the bowl then put the banana in the bowl'"},
    "ctl-rand": {"edit_location": "same layers as v2mi-f", "values_changed": 450468837,
                 "values_changed_note": "random-direction control: v2mi-f's layers, rank and singular values, random "
                                        "orthonormal directions (seed 0)"},
}


def load_runs():
    T = {}  # (code, arm) -> {slot: bool}
    recs = {}
    for f in sorted(RUNS.glob("*.json")):
        if ".fail" in f.name:
            continue
        r = json.loads(f.read_text())
        if r.get("status") != "fresh" or r.get("completed") != 16:
            continue
        key = (r["code"], r["arm"])
        d = T.setdefault(key, {})
        for row in r["rows"]:
            d[row["slot"]] = bool(row["success"])
        recs[r["tag"]] = r
    return T, recs


def block(d, lo, hi):
    return {s: v for s, v in d.items() if lo <= s <= hi}


def common_groups(*arms):
    gs = None
    for a in arms:
        g = {s // 16 for s in a}
        full = {x for x in g if all(x * 16 + j in a for j in range(16))}
        gs = full if gs is None else gs & full
    return sorted(gs or [])


def slots_of(groups):
    return [g * 16 + j for g in groups for j in range(16)]


def wil(k, n):
    lo, hi = S.wilson(k, n)
    return [round(100 * lo, 1), round(100 * hi, 1)]


def pair(ref, cand, slots):
    fixed = sum(1 for s in slots if cand[s] and not ref[s])
    broke = sum(1 for s in slots if ref[s] and not cand[s])
    both = sum(1 for s in slots if ref[s] and cand[s])
    neither = len(slots) - fixed - broke - both
    p = S.mcnemar_exact(fixed, broke)
    bs = S.paired_bootstrap(S.PairedTable(both=both, fixed=fixed, broke=broke, neither=neither))
    return {"fixed": fixed, "broke": broke, "both": both, "neither": neither, "p": p, "bootstrap": bs}


def span(slots):
    return [min(slots), max(slots)] if slots else None


def base_record(code, experiment, slots, wording):
    return {"experiment": experiment, "status": "fresh", "run": RUN_ID, "model": MODEL, "hardware": HW,
            "render_preset": "default", "envs_per_process": 16, "env_seed": 0,
            "task": TASKS[code][0], "code": code, "wording_pair": wording, "slots": span(slots), "n": len(slots),
            "unedited_k": None, "edited_k": None, "fixed": None, "broke": None, "p": None, "ci": None,
            "control_k": None, "teacher_k": None, "recovery_pct": None, "transfer": None, "retention": None,
            "edit": None, "edit_location": None, "values_changed": None, "values_changed_note": None, "notes": ""}


def bs_ci(b):
    lo, hi = b
    return [round(100 * lo, 1), round(100 * hi, 1)]


def main():
    T, recs = load_runs()
    out = []
    summary = {"run": RUN_ID, "render_preset": "default", "jobs": len(recs),
               "gpu_hours": round(sum(r["job_s"] for r in recs.values()) / 3600, 2), "cells": {}}

    # 1. failure map: unedited, default wording, practice 30000-30015 (first group of the shared practice block)
    for code in TASKS:
        b = block(T.get((code, "base"), {}), 30000, 30031)
        gs = common_groups(b)
        if not gs:
            continue
        sl = slots_of(gs)
        k = sum(b[s] for s in sl)
        r = base_record(code, "failure_map", sl, [TASKS[code][1]])
        r.update(unedited_k=k, ci={"wilson95_pct": wil(k, len(sl))},
                 per_slot={str(s): int(b[s]) for s in sl},
                 notes="unedited pi0.5, task's default instruction, practice slots, default render preset; every "
                       "16-slot group the first and only run of a fresh Isaac process")
        out.append(r)
        summary["cells"][f"{code}/base/practice"] = f"{k}/{len(sl)}"

    # 2. T1 held-out: v2mi-f vs unedited, the random-direction control, the teacher wording
    base = block(T.get(("T1", "base"), {}), 96000, 96063)
    ed = block(T.get(("T1", "v2mi-f"), {}), 96000, 96063)
    ctl = block(T.get(("T1", "ctl-rand"), {}), 96000, 96063)
    tch = block(T.get(("T1", "teacher"), {}), 96000, 96063)
    gs = common_groups(base, ed)
    if gs:
        sl = slots_of(gs)
        pe = pair(base, ed, sl)
        r = base_record("T1", "edit_target", sl, [TASKS["T1"][1]])
        ku, ke = sum(base[s] for s in sl), sum(ed[s] for s in sl)
        r.update(unedited_k=ku, edited_k=ke, fixed=pe["fixed"], broke=pe["broke"], p=pe["p"],
                 ci={"unedited_wilson95_pct": wil(ku, len(sl)), "edited_wilson95_pct": wil(ke, len(sl)),
                     "paired_diff_bootstrap95_pts": bs_ci(pe["bootstrap"])},
                 edit={"name": "v2mi-f", "fingerprint": CANDS["v2mi-f"]}, **EDIT_INFO["v2mi-f"],
                 per_slot={str(s): [int(base[s]), int(ed[s])] for s in sl},
                 notes="held-out slots (never used for any choice); instruction unchanged (task default); "
                       "per_slot = [unedited, edited]")
        gc = common_groups(base, ed, ctl)
        if gc:
            sc = slots_of(gc)
            kc = sum(ctl[s] for s in sc)
            pc = pair(base, ctl, sc)
            pec = pair(ctl, ed, sc)
            r["control_k"] = kc
            r["control"] = {"name": "ctl-rand", "fingerprint": CANDS["ctl-rand"], "slots": span(sc), "n": len(sc),
                            "k": kc, "wilson95_pct": wil(kc, len(sc)),
                            "unedited_k_same_slots": sum(base[s] for s in sc),
                            "edited_k_same_slots": sum(ed[s] for s in sc),
                            "vs_unedited": {k: pc[k] for k in ("fixed", "broke", "p")},
                            "edit_vs_control": {"edit_only": pec["fixed"], "control_only": pec["broke"],
                                                "p": pec["p"]},
                            **EDIT_INFO["ctl-rand"]}
            for s in sc:
                r["per_slot"][str(s)].append(int(ctl[s]))
            r["notes"] = r["notes"].replace("[unedited, edited]", "[unedited, edited, control]")
            summary["cells"]["T1/ctl-rand/heldout"] = f"{kc}/{len(sc)}"
        gt = common_groups(base, ed, tch)
        if gt:
            st = slots_of(gt)
            kt, kus, kes = (sum(tch[s] for s in st), sum(base[s] for s in st), sum(ed[s] for s in st))
            r["teacher_k"] = kt
            r["teacher"] = {"wording": TEACHERS["T1"], "slots": span(st), "n": len(st), "k": kt,
                            "vs_edit": {k: pair(ed, tch, st)[k] for k in ("fixed", "broke", "p")}}
            if kt != kus:
                r["recovery_pct"] = round(100 * (kes - kus) / (kt - kus), 1)
        out.append(r)
        summary["cells"]["T1/base/heldout"] = f"{ku}/{len(sl)}"
        summary["cells"]["T1/v2mi-f/heldout"] = f"{ke}/{len(sl)}"

    # 3. side effects of v2mi-f, held-out 96000-96031, task's own default wording
    for code in TASKS:
        if code == "T1":
            continue
        b = block(T.get((code, "base"), {}), 96000, 96031)
        e = block(T.get((code, "v2mi-f"), {}), 96000, 96031)
        gs = common_groups(b, e)
        if not gs:
            continue
        sl = slots_of(gs)
        pe = pair(b, e, sl)
        ku, ke = sum(b[s] for s in sl), sum(e[s] for s in sl)
        r = base_record(code, "side_effect", sl, [TASKS[code][1]])
        r.update(unedited_k=ku, edited_k=ke, fixed=pe["fixed"], broke=pe["broke"], p=pe["p"],
                 ci={"unedited_wilson95_pct": wil(ku, len(sl)), "edited_wilson95_pct": wil(ke, len(sl)),
                     "paired_diff_bootstrap95_pts": bs_ci(pe["bootstrap"])},
                 edit={"name": "v2mi-f", "fingerprint": CANDS["v2mi-f"]}, **EDIT_INFO["v2mi-f"],
                 per_slot={str(s): [int(b[s]), int(e[s])] for s in sl},
                 notes="side effect of the T1 edit v2mi-f on another task, held-out slots, the task's own default "
                       "instruction; per_slot = [unedited, edited]")
        if code == "R1":
            r["retention"] = {"unedited_k": ku, "edited_k": ke, "n": len(sl)}
        else:
            r["transfer"] = {"unedited_k": ku, "edited_k": ke, "n": len(sl)}
        out.append(r)
        summary["cells"][f"{code}/base/heldout"] = f"{ku}/{len(sl)}"
        summary["cells"][f"{code}/v2mi-f/heldout"] = f"{ke}/{len(sl)}"

    # 4. T8 wording gap: reference-first (task default) vs target-first, unedited weights, practice
    rf = block(T.get(("T8", "base"), {}), 30000, 30031)
    tf = block(T.get(("T8", "teacher"), {}), 30000, 30031)
    gs = common_groups(rf, tf)
    if gs:
        sl = slots_of(gs)
        pe = pair(rf, tf, sl)
        k1, k2 = sum(rf[s] for s in sl), sum(tf[s] for s in sl)
        r = base_record("T8", "wording", sl, [TASKS["T8"][1], TEACHERS["T8"]])
        r.update(unedited_k=k1, teacher_k=k2, fixed=pe["fixed"], broke=pe["broke"], p=pe["p"],
                 ci={"rf_wilson95_pct": wil(k1, len(sl)), "tf_wilson95_pct": wil(k2, len(sl))},
                 per_slot={str(s): [int(rf[s]), int(tf[s])] for s in sl},
                 notes="unedited weights; reference-first (RF, task default) vs target-first (TF) wording on the "
                       "same starts; fixed = TF-only, broke = RF-only; per_slot = [RF, TF]")
        out.append(r)
        summary["cells"]["T8/teacher(TF)/practice"] = f"{k2}/{len(sl)}"

    for r in out:
        for k in ("p",):
            if isinstance(r.get(k), float):
                r[k] = float(f"{r[k]:.4g}")
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    summary["records"] = [{k: r[k] for k in ("experiment", "code", "slots", "n", "unedited_k", "edited_k",
                                             "control_k", "teacher_k", "fixed", "broke", "p")} for r in out]
    SUMMARY.write_text(json.dumps(summary, indent=1) + "\n")
    M = json.loads(METRICS.read_text())
    M = [x for x in M if x.get("run") != RUN_ID] + out
    METRICS.write_text(json.dumps(M, indent=1, ensure_ascii=False) + "\n")
    for x in summary["records"]:
        print(json.dumps(x))
    print("jobs", summary["jobs"], "gpu_h", summary["gpu_hours"])


if __name__ == "__main__":
    main()
