"""Track C (pi0.5 DROID): tables, metrics and plots from the per-slot run records.

Reads results/provisional.json (Step 0, existing results; scripts/provisional.py) and the fresh per-job records
results/runs/<tag>.json (pod, RTX 4080 SUPER) and results/runs_ali-ms/<tag>.json (ali-ms, RTX 3090; the hardware
calibration only), and writes ../metrics_pi05.json (one record per experiment, provisional and fresh, each marked),
results/summary.json and plots/*.svg + *.png (2x). Statistics: weightsmith.stats (exact McNemar, Wilson, paired
bootstrap with 20,000 draws, seed 0), as in the paper.

A job tag is <code>-<arm>-s<slot_start>; an arm's outcomes are pooled over its 16-slot jobs and arms are paired on
the slots both ran.
    python3 study/results/cross/pi05/scripts/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[3]))
sys.path.insert(0, str(HERE / "scripts"))
from weightsmith import stats  # noqa: E402
from tasks import CANDS, TASKS, TEACHERS  # noqa: E402

RUNS = HERE / "results" / "runs"
RUNS_OLD = HERE / "results" / "runs_ali-ms"
PLOTS = HERE / "plots"
METRICS = HERE.parent / "metrics_pi05.json"
MODEL = "pi05_droid_jointpos (OpenPI pi0.5 DROID, joint position; gs://openpi-assets-simeval/pi05_droid_jointpos)"
HW_POD = "rented pod, 2x RTX 4080 SUPER 32 GB (driver 595), weightsmith @ ad31044"
HW_OLD = "ali-ms, RTX 3090 (driver of the reference box), weightsmith @ ad31044"
PRACTICE = range(30000, 30032)
HELDOUT = range(96000, 96064)
LABELS = {"T1": "T1 cube then banana", "T2": "T2 pick glasses", "T3": "T3 cube left of bowl",
          "T4": "T4 stack yellow on red", "T5": "T5 mustard in left bin", "T6": "T6 stack 3 cubes",
          "T7": "T7 pick green block", "T8": "T8 mustard right of raisin (RF)", "R1": "R1 cube in bowl"}
ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R1"]
EDITS = {
    "v2mi-f": {"location": "VLM MLP down-projections language.blocks.00-16.mlp.linear (17 tensors), rank <= 72/layer",
               "values_changed": 450468837, "note": "450,468,837 stored values (13.4%)"},
    "abl-imgl8": {"location": "VLM MLP down-projections language.blocks.08-16.mlp.linear (9 tensors), image-cell "
                              "keys only, rank <= 32/layer",
                  "values_changed": 248057499, "note": "248,057,499 stored values (7.4%); 5.3M numbers as factors "
                                                       "(0.16%)"},
}


ROWS: dict[tuple[str, str], dict[int, dict]] = {}


def load_runs(d: Path):
    arms: dict[tuple[str, str], dict[int, bool]] = {}
    meta: list[dict] = []
    if not d.exists():
        return arms, meta
    for f in sorted(d.glob("*.json")):
        if ".fail" in f.name:
            continue
        r = json.loads(f.read_text())
        key = (r["code"], r["arm"])
        rows = {x["slot"]: bool(x["success"]) for x in r["rows"] if x.get("ok")}
        arms.setdefault(key, {}).update(rows)
        if d == RUNS:
            for x in r["rows"]:
                if x.get("ok"):
                    ROWS.setdefault(key, {})[x["slot"]] = {"success": bool(x["success"]), "steps": x.get("steps"),
                                                           "reason": x.get("reason"),
                                                           "video": (r.get("videos") or {}).get(str(x["slot"]))}
        meta.append({k: r.get(k) for k in ("tag", "code", "task", "arm", "slot_start", "n", "prompt", "fp",
                                           "fp_checked", "server_fingerprint", "completed", "success", "gpu",
                                           "lock_wait_s", "gpu_s", "started", "finished", "host")})
    return arms, meta


def pct(x):
    return round(100 * x, 1)


def wil(k, n):
    lo, hi = stats.wilson(k, n)
    return [pct(lo), pct(hi)]


def paired(ref, cand, slots):
    common = sorted(set(ref) & set(cand) & set(slots))
    if not common:
        return None
    c = stats.compare({s: ref[s] for s in common}, {s: cand[s] for s in common})
    t = c.table
    lo, hi = c.difference_ci
    return {"n": t.n, "ref_k": t.ref_success, "cand_k": t.cand_success, "fixed": t.fixed, "broke": t.broke,
            "both": t.both, "neither": t.neither, "p": c.p_value, "diff_pts": pct(t.difference),
            "bootstrap95_diff_pts": [pct(lo), pct(hi)], "slots": [common[0], common[-1]],
            "per_slot": {str(s): [int(ref[s]), int(cand[s])] for s in common}}


def base_record(experiment, code, *, slots, n, notes, hw=HW_POD, **kw):
    rec = {"experiment": experiment, "status": "fresh", "model": MODEL, "hardware": hw, "task": TASKS[code][0],
           "code": code, "wording_pair": [TASKS[code][1]], "slots": slots, "n": n, "unedited_k": None,
           "edited_k": None, "fixed": None, "broke": None, "p": None, "ci": None, "control_k": None,
           "teacher_k": None, "recovery_pct": None, "transfer": None, "retention": None, "edit": None,
           "edit_location": None, "values_changed": None, "values_changed_note": None, "notes": notes}
    rec.update(kw)
    return rec


def fresh_records(arms):
    recs, table = [], {}
    for code in ORDER:
        base = arms.get((code, "base"), {})
        bs = {s: v for s, v in base.items() if s in PRACTICE}
        if bs:
            k, n = sum(bs.values()), len(bs)
            recs.append(base_record("failure_map", code, slots=[min(bs), max(bs)], n=n, unedited_k=k,
                                    ci={"wilson95_pct": wil(k, n)},
                                    notes="unedited policy, task's default instruction, shared practice slots; every "
                                          "16-slot group the first and only run of a fresh Isaac process"))
            table.setdefault(code, {})["base"] = (k, n)
        teach = arms.get((code, "teacher"))
        if teach and bs:
            c = paired(bs, teach, PRACTICE)
            if c:
                recs.append(base_record(
                    "wording", code, slots=c["slots"], n=c["n"], unedited_k=c["ref_k"], fixed=c["fixed"],
                    broke=c["broke"], p=c["p"], wording_pair=[TASKS[code][1], TEACHERS[code]],
                    teacher_k={"k": c["cand_k"], "n": c["n"], "slots": c["slots"], "wording": TEACHERS[code]},
                    ci={"wilson95_unedited_pct": wil(c["ref_k"], c["n"]),
                        "wilson95_teacher_pct": wil(c["cand_k"], c["n"]), "diff_pts": c["diff_pts"],
                        "bootstrap95_diff_pts": c["bootstrap95_diff_pts"]},
                    notes="same unedited weights, teacher wording via the prompt override; fixed = teacher-only pairs",
                    per_slot=c["per_slot"]))
                table.setdefault(code, {})["teacher"] = (c["cand_k"], c["n"])
        for arm in CANDS:
            ed = arms.get((code, arm))
            if not ed or not bs:
                continue
            c = paired(bs, ed, PRACTICE)
            if not c:
                continue
            t = arms.get((code, "teacher"), {})
            tk = paired(bs, t, c["slots"] and range(c["slots"][0], c["slots"][1] + 1)) if t else None
            rec_pct = None
            teacher = None
            if tk and set(map(int, tk["per_slot"])) == set(map(int, c["per_slot"])):
                teacher = {"k": tk["cand_k"], "n": tk["n"], "slots": tk["slots"], "wording": TEACHERS[code]}
                gap = tk["cand_k"] - c["ref_k"]
                rec_pct = round(100 * (c["cand_k"] - c["ref_k"]) / gap, 1) if gap > 0 else None
            recs.append(base_record(
                "edit_target" if code == "T1" else "side_effect", code, slots=c["slots"], n=c["n"],
                unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
                ci={"wilson95_unedited_pct": wil(c["ref_k"], c["n"]), "wilson95_edited_pct": wil(c["cand_k"], c["n"]),
                    "diff_pts": c["diff_pts"], "bootstrap95_diff_pts": c["bootstrap95_diff_pts"]},
                teacher_k=teacher, recovery_pct=rec_pct,
                edit={"name": arm, "fingerprint": CANDS[arm]}, edit_location=EDITS[arm]["location"],
                values_changed=EDITS[arm]["values_changed"], values_changed_note=EDITS[arm]["note"],
                notes=("the paper's T1 edit, byte copy (fingerprint checked before every job), task's own instruction; "
                       "paired with the unedited arm on the same slots"), per_slot=c["per_slot"]))
            table.setdefault(code, {})[arm] = c
    return recs, table


def video_picks(slots=range(30000, 30016)):
    """Per task, on the slots with sensor videos: clear failures of the unedited policy, slots the edit fixes
    (unedited fails, v2mi-f succeeds; shortest edited episode first) and slots it breaks."""
    out = {}
    for code in ORDER:
        b, e = ROWS.get((code, "base"), {}), ROWS.get((code, "v2mi-f"), {})
        common = [s for s in slots if s in b and s in e]
        if not common:
            continue
        fixes = sorted((s for s in common if not b[s]["success"] and e[s]["success"]), key=lambda s: e[s]["steps"])
        breaks = [s for s in common if b[s]["success"] and not e[s]["success"]]
        fails = [s for s in common if not b[s]["success"] and not e[s]["success"]]
        out[code] = {
            "clear_fix": [{"slot": s, "unedited": f"fail ({b[s]['steps']} steps): {b[s]['reason']}",
                           "edited": f"success in {e[s]['steps']} steps",
                           "videos": [b[s]["video"], e[s]["video"]]} for s in fixes[:3]],
            "clear_failure": [{"slot": s, "unedited": f"fail ({b[s]['steps']} steps): {b[s]['reason']}",
                               "edited": "fail" if not e[s]["success"] else "success",
                               "video": b[s]["video"]}
                              for s in (fixes[:1] + fails[:2])],
            "broke": [{"slot": s, "unedited": f"success in {b[s]['steps']} steps",
                       "edited": f"fail ({e[s]['steps']} steps): {e[s]['reason']}"} for s in breaks],
        }
    return out


def calibration(arms_pod, arms_old):
    out = []
    for (code, arm), old in arms_old.items():
        new = arms_pod.get((code, arm))
        if not new:
            continue
        c = paired(old, new, PRACTICE)
        if not c:
            continue
        out.append(base_record(
            "calibration", code, slots=c["slots"], n=c["n"], hw=f"{HW_OLD} vs {HW_POD}",
            unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
            ci={"wilson95_3090_pct": wil(c["ref_k"], c["n"]), "wilson95_4080s_pct": wil(c["cand_k"], c["n"]),
                "bootstrap95_diff_pts": c["bootstrap95_diff_pts"]},
            notes=(f"same weights ({arm}), same slots and runtime commit, RTX 3090 (unedited_k) vs RTX 4080 SUPER "
                   "(edited_k): a hardware check, not an edit; fixed = 4080S-only successes"),
            per_slot=c["per_slot"]))
    return out


# ------------------------------------------------------------------------------------------------- plots
INK, INK2, MUTE, LIGHT, PALE, RULE = "#1C1C1C", "#4D4D4D", "#858585", "#C6C6C6", "#ECECEC", "#A3A3A3"
TEAL, ORANGE = "#147D92", "#C0612B"


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": RULE,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "svg.fonttype": "none"})
    return plt


def _save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.png", dpi=200, bbox_inches="tight")  # 2x of 100 dpi


def plot_side_effects(table, arm, name):
    rows = [(c, table[c]) for c in ORDER if c in table and arm in table[c]]
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(rows) + 1.0))
    for i, (code, d) in enumerate(rows):
        y = len(rows) - 1 - i
        c = d[arm]
        for dy, k, col in ((0.13, c["ref_k"], LIGHT), (-0.13, c["cand_k"], TEAL)):
            ax.barh(y + dy, 100 * k / c["n"], height=0.24, color=col, edgecolor="none", zorder=2)
        hi = 100 * max(c["ref_k"], c["cand_k"]) / c["n"]
        ax.text(hi + 1.5, y, f"{c['ref_k']} → {c['cand_k']} of {c['n']}   {c['fixed']} fixed / {c['broke']} broke, "
                             f"p = {stats.format_p(c['p'])}", va="center", fontsize=7.8, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[c] for c, _ in reversed(rows)])
    ax.set_xlim(0, 175)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("success, % of the same paired starts (grey: unedited, teal: edited)")
    ax.grid(axis="x", color=PALE, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(f"pi0.5 DROID: the T1 edit {arm} on every task (slots 30000-30031, RTX 4080 SUPER)", fontsize=9.3,
                 color=INK, loc="left")
    _save(fig, name)
    plt.close(fig)


def plot_failure_map(table):
    rows = [(c, table[c]["base"]) for c in ORDER if c in table and "base" in table[c]]
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 0.9))
    for i, (code, (k, n)) in enumerate(rows):
        y = len(rows) - 1 - i
        lo, hi = wil(k, n)
        ax.barh(y, 100 * k / n, height=0.5, color=LIGHT, edgecolor="none", zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.1, zorder=3)
        ax.text(max(hi, 100 * k / n) + 1.5, y, f"{k}/{n}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[c] for c, _ in reversed(rows)])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("success, % of paired starts (bar) with Wilson 95% interval")
    ax.grid(axis="x", color=PALE, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("pi0.5 DROID, unedited, default wording (slots 30000-30031, RTX 4080 SUPER)", fontsize=9.5,
                 color=INK, loc="left")
    _save(fig, "failure_map")
    plt.close(fig)


def main():
    prov = json.loads((HERE / "results/provisional.json").read_text())["records"]
    arms, meta = load_runs(RUNS)
    arms_old, meta_old = load_runs(RUNS_OLD)
    fresh, table = fresh_records(arms)
    fresh += calibration(arms, arms_old)
    gpu_s = sum((m.get("gpu_s") or 0) for m in meta)
    summary = {"model": MODEL, "hardware": HW_POD, "video_picks": video_picks(), "jobs": meta,
               "jobs_ali_ms": meta_old,
               "gpu_hours_pod_jobs": round(gpu_s / 3600, 2),
               "table": {c: {a: (v if not isinstance(v, dict) else {k: x for k, x in v.items() if k != "per_slot"})
                             for a, v in d.items()} for c, d in table.items()}}
    (HERE / "results/summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    METRICS.write_text(json.dumps(prov + fresh, indent=1) + "\n")
    plot_failure_map(table)
    plot_side_effects(table, "v2mi-f", "side_effects_v2mi-f")
    plot_side_effects(table, "abl-imgl8", "side_effects_abl-imgl8")
    print(f"{len(prov)} provisional + {len(fresh)} fresh records -> {METRICS}")
    for r in fresh:
        print(f"  {r['experiment']:12s} {r['code']} {r['edit']['name'] if r['edit'] else '':10s} n={r['n']} "
              f"{r['unedited_k']} -> {r['edited_k']} fixed {r['fixed']} broke {r['broke']} p={r['p']}")


if __name__ == "__main__":
    main()
