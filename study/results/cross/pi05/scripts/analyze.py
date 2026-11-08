"""Track C (pi0.5 DROID): tables, metrics and plots from the per-slot run records.

Reads results/provisional.json (existing results; scripts/provisional.py), the fresh per-job records
results/runs/<tag>.json (pod, RTX 4080 SUPER) and results/runs_ali-ms/<tag>.json (ali-ms, RTX 3090: one T1 group
under the performance preset, used as a hardware check), and writes ../metrics_pi05.json (one record per experiment,
provisional and fresh, each marked in "status"), results/summary.json and plots/*.svg + *.png (2x). Statistics:
weightsmith.stats (exact two-sided McNemar, Wilson 95%, paired bootstrap with 20,000 draws, seed 0), as in the paper.

A job tag is <code>-<arm>-s<slot_start>; an arm's outcomes are pooled over its 16-slot jobs, and two arms are paired
on the slots both ran inside one block (practice 30000-30031 or held-out 96000-96063).
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
from tasks import CANDS, TASKS, TEACHERS, WORDINGS  # noqa: E402

RUNS = HERE / "results" / "runs"
RUNS_OLD = HERE / "results" / "runs_ali-ms"
PLOTS = HERE / "plots"
METRICS = HERE.parent / "metrics_pi05.json"
MODEL = "pi05_droid_jointpos (OpenPI pi0.5 DROID, joint position; gs://openpi-assets-simeval/pi05_droid_jointpos)"
HW_POD = "rented pod (vast.ai, Japan), 2x RTX 4080 SUPER 32 GB, driver 595.58; weightsmith @ ad31044"
HW_OLD = "ali-ms, RTX 3090; weightsmith @ ad31044"
BLOCKS = {"practice": range(30000, 30032), "heldout": range(96000, 96064)}
LABELS = {"T1": "T1 cube then banana", "T2": "T2 pick glasses", "T3": "T3 cube left of bowl",
          "T4": "T4 stack yellow on red", "T5": "T5 mustard in left bin", "T6": "T6 stack 3 cubes",
          "T7": "T7 pick green block", "T8": "T8 mustard right of raisin (RF)", "R1": "R1 cube in bowl"}
ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R1"]
EDIT_TASK = {"v2mi-f": "T1", "abl-imgl8": "T1", "ctl-rand": "T1", "s4r-grp3": "T8", "ctl-rand3": "T8"}
CONTROL_OF = {"v2mi-f": "ctl-rand", "s4r-grp3": "ctl-rand3"}
EDITS = {
    "v2mi-f": {"location": "VLM MLP down-projections language.blocks.00-16.mlp.linear (17 tensors), rank <= 72/layer",
               "values_changed": 450468837, "note": "450,468,837 stored values (13.4%)"},
    "abl-imgl8": {"location": "VLM MLP down-projections language.blocks.08-16.mlp.linear (9 tensors), image-cell "
                              "keys only, rank <= 32/layer",
                  "values_changed": 248057499, "note": "248,057,499 stored values (7.4%); 5.3M numbers as factors "
                                                       "(0.16%)"},
    "ctl-rand": {"location": "same tensors, ranks and singular values as v2mi-f; random orthonormal directions "
                             "(numpy default_rng(0))",
                 "values_changed": 502291925, "note": "about 502.3M stored values (from the control's build report)"},
    "s4r-grp3": {"location": "VLM MLP down-projections language.blocks.08-16.mlp.linear (9 tensors), 96 keys/layer "
                             "in 3 decision-phase groups",
                 "values_changed": 275482328, "note": "about 275.5M stored values; 15.9M numbers as factors (0.48%)"},
    "ctl-rand3": {"location": "same tensors, ranks and singular values as s4r-grp3; random orthonormal directions",
                  "values_changed": 290322032, "note": "about 290.3M stored values (from the control's build report)"},
}

ROWS: dict[tuple[str, str], dict[int, dict]] = {}


def load_runs(d: Path, old=False):
    arms: dict[tuple[str, str], dict[int, bool]] = {}
    meta: list[dict] = []
    if not d.exists():
        return arms, meta
    for f in sorted(d.glob("*.json")):
        if ".fail" in f.name:
            continue
        r = json.loads(f.read_text())
        arm = r["arm"]
        if old and not r.get("rendering_type_used"):  # the ali-ms job ran under weightsmith's performance preset
            arm += "-perf"
        key = (r["code"], arm)
        rows = {x["slot"]: bool(x["success"]) for x in r["rows"] if x.get("ok")}
        arms.setdefault(key, {}).update(rows)
        if not old:
            for x in r["rows"]:
                if x.get("ok"):
                    ROWS.setdefault(key, {})[x["slot"]] = {"success": bool(x["success"]), "steps": x.get("steps"),
                                                           "reason": x.get("reason"),
                                                           "video": (r.get("videos") or {}).get(str(x["slot"]))}
        meta.append({k: r.get(k) for k in ("tag", "code", "task", "arm", "slot_start", "n", "prompt", "fp",
                                           "fp_checked", "rendering_type_used", "completed", "success", "gpu",
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
            "bootstrap95_diff_pts": [pct(lo), pct(hi)], "slots": [common[0], common[-1]], "slot_list": common,
            "per_slot": {str(s): [int(ref[s]), int(cand[s])] for s in common}}


def rng_txt(c):
    return f"{c['slots'][0]}-{c['slots'][1]}" + ("" if c["slots"][1] - c["slots"][0] + 1 == c["n"] else
                                                 f" ({c['n']} slots)")


def base_record(experiment, code, *, slots, n, notes, hw=HW_POD, preset="default", **kw):
    rec = {"experiment": experiment, "status": "fresh", "model": MODEL, "hardware": hw,
           "render_preset": preset, "task": TASKS[code][0], "code": code, "wording_pair": [TASKS[code][1]],
           "slots": slots, "n": n, "unedited_k": None, "edited_k": None, "fixed": None, "broke": None, "p": None,
           "ci": None, "control_k": None, "teacher_k": None, "recovery_pct": None, "transfer": None,
           "retention": None, "edit": None, "edit_location": None, "values_changed": None,
           "values_changed_note": None, "notes": notes}
    rec.update(kw)
    return rec


def ci_pair(c, a="unedited", b="edited"):
    return {f"wilson95_{a}_pct": wil(c["ref_k"], c["n"]), f"wilson95_{b}_pct": wil(c["cand_k"], c["n"]),
            "diff_pts": c["diff_pts"], "bootstrap95_diff_pts": c["bootstrap95_diff_pts"]}


def teacher_word(code, arm):
    if arm == "teacher":
        return TEACHERS[code]
    return dict(WORDINGS.get(code, ()) and [WORDINGS[code]]).get(arm)


def fresh_records(arms):
    recs, table = [], {}
    for code in ORDER:
        for bname, block in BLOCKS.items():
            base = {s: v for s, v in arms.get((code, "base"), {}).items() if s in block}
            if not base:
                continue
            T = table.setdefault(code, {}).setdefault(bname, {})
            k, n = sum(base.values()), len(base)
            T["base"] = {"k": k, "n": n, "slots": [min(base), max(base)]}
            if bname == "practice":
                recs.append(base_record("failure_map", code, slots=[min(base), max(base)], n=n, unedited_k=k,
                                        ci={"wilson95_pct": wil(k, n)},
                                        notes="unedited policy, task's default instruction, shared practice slots; "
                                              "every 16-slot group the first and only run of a fresh Isaac process"))
            # wordings given to the unedited weights
            for arm in ["teacher"] + [w[0] for c2, w in WORDINGS.items() if c2 == code]:
                w = arms.get((code, arm))
                c = paired(base, w, block) if w else None
                if not c:
                    continue
                wording = TEACHERS[code] if arm == "teacher" else WORDINGS[code][1]
                T[arm] = c
                recs.append(base_record(
                    "wording" if bname == "practice" else "wording_heldout", code, slots=c["slots"], n=c["n"],
                    unedited_k=c["ref_k"], edited_k=None, fixed=c["fixed"], broke=c["broke"], p=c["p"],
                    wording_pair=[TASKS[code][1], wording],
                    teacher_k={"k": c["cand_k"], "n": c["n"], "slots": c["slots"], "wording": wording,
                               "role": "teacher (known-good wording)" if arm == "teacher" else
                               "alternative wording (parity with Track A)"},
                    ci=ci_pair(c, "default_wording", "other_wording"),
                    notes="same unedited weights, other wording via the prompt override; fixed = pairs only the "
                          "other wording solves, broke = pairs only the default wording solves",
                    per_slot=c["per_slot"]))
            # edits and controls
            for arm in CANDS:
                ed = arms.get((code, arm))
                c = paired(base, ed, block) if ed else None
                if not c:
                    continue
                T[arm] = c
                is_ctl = arm.startswith("ctl-")
                target = EDIT_TASK[arm] == code
                exp = ("control" if is_ctl else "edit_target" if target else
                       "retention" if code == "R1" else "side_effect")
                if bname == "heldout":
                    exp += "_heldout"
                same = set(c["slot_list"])
                ctl = None
                if arm in CONTROL_OF and (code, CONTROL_OF[arm]) in arms:
                    cc = paired(base, arms[(code, CONTROL_OF[arm])], c["slot_list"])
                    if cc and set(cc["slot_list"]) == same:
                        ctl = {"k": cc["cand_k"], "n": cc["n"], "slots": cc["slots"], "arm": CONTROL_OF[arm],
                               "fingerprint": CANDS[CONTROL_OF[arm]], "fixed": cc["fixed"], "broke": cc["broke"],
                               "p_vs_unedited": cc["p"]}
                        ce = paired(arms[(code, CONTROL_OF[arm])], ed, c["slot_list"])
                        ctl["edit_vs_control"] = {k2: ce[k2] for k2 in ("fixed", "broke", "p")}
                teacher, rec_pct = None, None
                if code in TEACHERS and (code, "teacher") in arms and not is_ctl:
                    tc = paired(base, arms[(code, "teacher")], c["slot_list"])
                    if tc and set(tc["slot_list"]) == same:
                        teacher = {"k": tc["cand_k"], "n": tc["n"], "slots": tc["slots"], "wording": TEACHERS[code]}
                        gap = tc["cand_k"] - c["ref_k"]
                        rec_pct = round(100 * (c["cand_k"] - c["ref_k"]) / gap, 1) if gap > 0 else None
                        te = paired(arms[(code, "teacher")], ed, c["slot_list"])
                        teacher["edit_vs_teacher"] = {"edit_only": te["fixed"], "teacher_only": te["broke"],
                                                      "p": te["p"]}
                delta = {"diff_pts": c["diff_pts"], "fixed": c["fixed"], "broke": c["broke"], "p": c["p"],
                         "direction": ("better" if c["p"] < 0.05 and c["fixed"] > c["broke"] else
                                       "worse" if c["p"] < 0.05 else "no detectable change at this n")}
                recs.append(base_record(
                    exp, code, slots=c["slots"], n=c["n"], unedited_k=c["ref_k"], edited_k=c["cand_k"],
                    fixed=c["fixed"], broke=c["broke"], p=c["p"], ci=ci_pair(c), control_k=ctl,
                    teacher_k=teacher, recovery_pct=rec_pct,
                    transfer=delta if (not target and code != "R1" and not is_ctl) else None,
                    retention=({"unedited": [c["ref_k"], c["n"]], "edited": [c["cand_k"], c["n"]], **delta}
                               if code == "R1" else None),
                    edit={"name": arm, "fingerprint": CANDS[arm], "built_for": EDIT_TASK[arm]},
                    edit_location=EDITS[arm]["location"], values_changed=EDITS[arm]["values_changed"],
                    values_changed_note=EDITS[arm]["note"],
                    notes=("byte copy of the published candidate (fingerprint recomputed before every job and the "
                           "server's loaded workspace checked after it), task's own default instruction; paired with "
                           "the unedited arm on the same slots" + ("; random-direction control" if is_ctl else "")),
                    per_slot=c["per_slot"]))
            # render-preset sensitivity (secondary)
            bp = arms.get((code, "base-perf"))
            c = paired(base, bp, block) if bp else None
            if c:
                T["base-perf"] = c
                recs.append(base_record(
                    "preset_sensitivity", code, slots=c["slots"], n=c["n"], preset="default vs performance",
                    unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
                    ci=ci_pair(c, "default_preset", "performance_preset"),
                    notes="unedited weights; unedited_k = default preset, edited_k = performance preset (not an edit)",
                    per_slot=c["per_slot"]))
            ep = arms.get((code, "v2mi-f-perf"))
            if bp and ep:
                c = paired({s: v for s, v in bp.items() if s in block}, ep, block)
                if c:
                    T["v2mi-f-perf"] = c
                    recs.append(base_record(
                        "edit_under_performance_preset", code, slots=c["slots"], n=c["n"], preset="performance",
                        unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
                        ci=ci_pair(c), edit={"name": "v2mi-f", "fingerprint": CANDS["v2mi-f"], "built_for": "T1"},
                        edit_location=EDITS["v2mi-f"]["location"], values_changed=EDITS["v2mi-f"]["values_changed"],
                        notes="secondary: both arms under RoboLab's performance preset (the paper's setting)",
                        per_slot=c["per_slot"]))
    return recs, table


def calibration(arms_pod, arms_old):
    out = []
    for (code, arm), old in arms_old.items():
        new = arms_pod.get((code, arm))
        c = paired(old, new, BLOCKS["practice"]) if new else None
        if not c:
            continue
        out.append(base_record(
            "hardware_check", code, slots=c["slots"], n=c["n"], hw=f"{HW_OLD} vs {HW_POD}", preset="performance",
            unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
            ci=ci_pair(c, "rtx3090", "rtx4080s"),
            notes=(f"same unedited weights, same slots, runtime commit and preset (performance), RTX 3090 "
                   "(unedited_k) vs RTX 4080 SUPER (edited_k): a hardware check, not an edit"),
            per_slot=c["per_slot"]))
    return out


PRIOR = HERE / "results" / "prior_slots_ali-ms.json"
HW_PRIOR = "ali-ms, RTX 3090 (weightsmith runtime; earlier studies)"
CTB = "cube-then-banana/heldout/"
PRIOR_SETS = {
    "T1": {"slots": range(90000, 90064), "source": "held-out runs of cube-then-banana (v2mi-f base/cand 90000, ctl1)",
           "base": [CTB + "v2mi-f-RubiksCubeThenBananaTask-90000-128/logs-base-90000"],
           "edit": [CTB + "v2mi-f-RubiksCubeThenBananaTask-90000-128/logs-cand-90000"],
           "control": [CTB + "ctl1-90000-64/logs-rand-90000"],
           "teacher": [CTB + "ctl1-90000-64/logs-teacher-90000"], "edit_name": "v2mi-f", "control_name": "ctl-rand"},
    "T8": {"slots": range(92000, 92064), "source": "held-out runs of sgdw-s4r (s4r-grp2 base arm; ctl-92000-64)",
           "base": [f"sgdw-s4r/heldout/s4r-grp2-MustardRightOfRaisinRFTask-92000-64/logs-base-{b}" for b in (92000, 92032)],
           "edit": [f"sgdw-s4r/heldout/ctl-92000-64/logs-g3-{b}" for b in (92000, 92032)],
           "control": [f"sgdw-s4r/heldout/ctl-92000-64/logs-rand3-{b}" for b in (92000, 92032)],
           "edit_name": "s4r-grp3", "control_name": "ctl-rand3"},
}


def prior_arm(R, keys):
    out = {}
    for k in keys:
        out.update({int(sl): bool(v[0]) for sl, v in R[k].items()})
    return out


def prov_controls():
    """PROVISIONAL control records: each random-direction control paired slot by slot with the unedited run and
    with the edit (and the teacher wording, T1), recomputed from the earlier runs' per-slot records."""
    if not PRIOR.exists():
        return []
    R = json.loads(PRIOR.read_text())["runs"]
    recs = []
    for code, S in PRIOR_SETS.items():
        A = {k: prior_arm(R, S[k]) for k in ("base", "edit", "control", "teacher") if k in S}
        c = paired(A["base"], A["control"], S["slots"])
        ce = paired(A["control"], A["edit"], S["slots"])
        cb = paired(A["base"], A["edit"], S["slots"])
        teacher = None
        if "teacher" in A:
            tb = paired(A["base"], A["teacher"], S["slots"])
            te = paired(A["teacher"], A["edit"], S["slots"])
            teacher = {"k": tb["cand_k"], "n": tb["n"], "slots": tb["slots"], "wording": TEACHERS[code],
                       "teacher_vs_unedited": {"fixed": tb["fixed"], "broke": tb["broke"], "p": tb["p"]},
                       "edit_vs_teacher": {"edit_only": te["fixed"], "teacher_only": te["broke"], "p": te["p"]}}
        name = S["control_name"]
        rec = base_record(
            "control", code, slots=c["slots"], n=c["n"], hw=HW_PRIOR, preset="performance",
            unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
            ci=ci_pair(c, "unedited", "control"),
            control_k={"k": c["cand_k"], "n": c["n"], "slots": c["slots"], "arm": name,
                       "control_vs_unedited": {"fixed": c["fixed"], "broke": c["broke"], "p": c["p"]},
                       "edit_vs_control": {"edit_only": ce["fixed"], "control_only": ce["broke"], "p": ce["p"],
                                           "edit_k": ce["cand_k"]},
                       "edit_vs_unedited": {"fixed": cb["fixed"], "broke": cb["broke"], "p": cb["p"],
                                            "edit_k": cb["cand_k"]}},
            teacher_k=teacher,
            edit={"name": name, "fingerprint": CANDS.get(name), "built_for": code, "control_of": S["edit_name"]},
            edit_location=EDITS[name]["location"], values_changed=EDITS[name]["values_changed"],
            values_changed_note=EDITS[name]["note"],
            notes=(f"PROVISIONAL (own slots, earlier run): random-direction control of {S['edit_name']}; unedited_k / "
                   f"edited_k = unedited / control on the same slots; recomputed from the per-slot records in "
                   f"results/prior_slots_ali-ms.json ({S['source']})"),
            per_slot=c["per_slot"])
        rec["status"] = "provisional"
        rec["source"] = "results/prior_slots_ali-ms.json: " + "; ".join(S["base"] + S["edit"] + S["control"] +
                                                                         S.get("teacher", []))
        recs.append(rec)
    return recs


def fresh_partial(arms_old, meta_old):
    """The one pi0.5 group on the shared slots: T1 unedited, 30000-30015, ali-ms before the move to the pod."""
    out = []
    for m in meta_old:
        key = (m["code"], m["arm"] + ("-perf" if not m.get("rendering_type_used") else ""))
        rows = arms_old.get(key, {})
        if not rows:
            continue
        k, n = sum(rows.values()), len(rows)
        out.append(base_record(
            "failure_map_partial", m["code"], slots=[min(rows), max(rows)], n=n, hw=HW_OLD, preset="performance",
            unedited_k=k, ci={"wilson95_pct": wil(k, n)},
            notes=("FRESH but partial: the only pi0.5 episodes on the shared slots, one 16-slot group (first and "
                   "only run of a fresh Isaac process, environment seed 0) on ali-ms GPU1 at 05:03 UTC before the "
                   "pi0.5 queue moved to the pod; RoboLab performance preset (Tracks A and B use the default "
                   "preset), so not a like-for-like failure-map entry"),
            per_slot={str(sl): [int(v)] for sl, v in sorted(rows.items())}))
    return out


def video_picks(slots=range(30000, 30032)):
    """Per task: slots the edit fixes (unedited fails, v2mi-f succeeds; shortest edited episode first), slots it
    breaks, and failures of both."""
    out = {}
    for code in ORDER:
        for arm in ("v2mi-f", "s4r-grp3"):
            b, e = ROWS.get((code, "base"), {}), ROWS.get((code, arm), {})
            common = [s for s in slots if s in b and s in e]
            if not common:
                continue
            fixes = sorted((s for s in common if not b[s]["success"] and e[s]["success"]),
                           key=lambda s: e[s]["steps"])
            breaks = [s for s in common if b[s]["success"] and not e[s]["success"]]
            fails = [s for s in common if not b[s]["success"] and not e[s]["success"]]
            out[f"{code}/{arm}"] = {
                "fixed": [{"slot": s, "unedited": f"fail ({b[s]['steps']} steps): {b[s]['reason']}",
                           "edited": f"success in {e[s]['steps']} steps"} for s in fixes],
                "broke": [{"slot": s, "unedited": f"success in {b[s]['steps']} steps",
                           "edited": f"fail ({e[s]['steps']} steps): {e[s]['reason']}"} for s in breaks],
                "both_fail": fails}
    return out


# ------------------------------------------------------------------------------------------------- plots
# Reference palette of the dataviz method (light mode): slot 1 blue for the edit, a neutral for the unedited
# policy, slot 2 orange for the wording (teacher); text in ink tokens, hairline grid.
SURF, INK, INK2, MUTE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NEUTRAL, BLUE, ORANGE = "#a19f99", "#2a78d6", "#eb6834"
COLOR_OF = {"v2mi-f": BLUE, "s4r-grp3": BLUE, "abl-imgl8": BLUE, "teacher": ORANGE}  # others: neutral


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "svg.fonttype": "none",
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})
    return plt


def _save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.png", dpi=200, bbox_inches="tight")  # 2x of 100 dpi


def _axes_clean(ax):
    ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", length=0)


def plot_failure_map(table, prov):
    rows = [(c, table[c]["practice"]["base"]) for c in ORDER if "base" in table.get(c, {}).get("practice", {})]
    if not rows:
        return
    old = {r["code"]: r for r in prov if r["experiment"] == "failure_map"}
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.0, 0.36 * len(rows) + 1.2))
    for i, (code, b) in enumerate(rows):
        y = len(rows) - 1 - i
        k, n = b["k"], b["n"]
        lo, hi = wil(k, n)
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.4, alpha=0.45, solid_capstyle="round", zorder=2)
        ax.plot([100 * k / n], [y], "o", ms=7, color=BLUE, mec=SURF, mew=1.5, zorder=4)
        if code in old:
            o = old[code]
            ax.plot([100 * o["unedited_k"] / o["n"]], [y], "o", ms=6, mfc="none", mec=MUTE, mew=1.2, zorder=3)
        ax.text(103, y, f"{k}/{n}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[c] for c, _ in reversed(rows)])
    ax.set_xlim(-2, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _axes_clean(ax)
    ax.plot([], [], "o", ms=7, color=BLUE, label="fresh: slots 30000-30031, default preset (line: Wilson 95%)")
    ax.plot([], [], "o", ms=6, mfc="none", mec=MUTE, mew=1.2,
            label="provisional: earlier runs, own slots, performance preset")
    ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=1, frameon=False, fontsize=8,
              labelcolor=INK2)
    ax.set_title("π0.5 DROID, unedited, each task's default wording: success rate", fontsize=10, color=INK,
                 loc="left")
    _save(fig, "failure_map")
    plt.close(fig)


def prov_table(prov, arm="v2mi-f"):
    """Provisional side-effect map: per shared task the largest earlier paired run of the edit (held-out first)."""
    out = {}
    for r in prov:
        if r["experiment"] not in ("side_effect", "edit_target") or not r.get("code") or not r.get("edit"):
            continue
        if r["edit"]["name"] != arm:
            continue
        cur = out.get(r["code"], {}).get("practice", {}).get(arm)
        if cur is None or r["n"] > cur["n"]:
            out.setdefault(r["code"], {}).setdefault("practice", {})[arm] = {
                "ref_k": r["unedited_k"], "cand_k": r["edited_k"], "n": r["n"], "fixed": r["fixed"],
                "broke": r["broke"], "p": r["p"], "slots": r["slots"]}
    return out


def plot_dumbbell(table, arm, name, title, block="practice", slot_labels=False):
    rows = [(c, table[c][block][arm]) for c in ORDER if arm in table.get(c, {}).get(block, {})]
    if not rows:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(8.0, 0.42 * len(rows) + 1.3))
    for i, (code, c) in enumerate(rows):
        y = len(rows) - 1 - i
        a, b = 100 * c["ref_k"] / c["n"], 100 * c["cand_k"] / c["n"]
        ax.plot([a, b], [y, y], color=BLUE if b >= a else ORANGE, lw=2, alpha=0.5, zorder=2)
        ax.plot([a], [y], "o", ms=7, color=NEUTRAL, mec=SURF, mew=1.5, zorder=3)
        ax.plot([b], [y], "o", ms=7, color=BLUE, mec=SURF, mew=1.5, zorder=4)
        sig = c["p"] < 0.05
        ax.text(104, y, f"{c['ref_k']} → {c['cand_k']} / {c['n']}", va="center", fontsize=8.3, color=INK,
                fontweight="bold" if sig else "normal")
        ax.text(136, y, f"{c['fixed']} fixed, {c['broke']} broke", va="center", fontsize=8.3, color=INK2)
        ax.text(176, y, f"p = {stats.format_p(c['p'])}", va="center", fontsize=8.3,
                color=INK if sig else INK2, fontweight="bold" if sig else "normal")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[c] + (f"\nslots {d['slots'][0]}-{d['slots'][1]}" if slot_labels else "")
                        for c, d in reversed(rows)], linespacing=1.1)
    ax.set_xlim(-2, 200)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _axes_clean(ax)
    ax.plot([], [], "o", ms=7, color=NEUTRAL, label="unedited")
    ax.plot([], [], "o", ms=7, color=BLUE, label=f"edited ({arm})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.3, -0.07), ncol=2, frameon=False, fontsize=8,
              labelcolor=INK2)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    _save(fig, name)
    plt.close(fig)


def plot_arms(T, arms, name, title, note):
    """Bars for several arms on the same paired slots of one task and block (Wilson 95% whiskers)."""
    bars = [(lab, T[a]) for a, lab in arms if a in T]
    arm_of = {lab: a for a, lab in arms}
    if len(bars) < 2:
        return
    plt = _plt()
    fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(bars) + 1.2))
    for i, (lab, c) in enumerate(bars):
        y = len(bars) - 1 - i
        k, n = (c["k"], c["n"]) if "k" in c else (c["cand_k"], c["n"])
        lo, hi = wil(k, n)
        col = COLOR_OF.get(arm_of[lab], NEUTRAL)
        ax.barh(y, 100 * k / n, height=0.56, color=col, edgecolor="none", zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1, zorder=3)
        ax.text(max(hi, 100 * k / n) + 2, y, f"{k}/{n}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(bars)))
    ax.set_yticklabels([lab for lab, _ in reversed(bars)])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _axes_clean(ax)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    ax.text(0, -0.75, note, transform=ax.get_yaxis_transform(), fontsize=7.6, color=INK2, va="top")
    _save(fig, name)
    plt.close(fig)


def plot_bars(groups, name, title, note, width=7.2):
    """Horizontal bars, one per arm, grouped by the slots they share; Wilson 95% whiskers; every bar labelled."""
    rows = []
    for g, bars in groups:
        rows.append(("H", g, None, None, None))
        rows += [("B", lab, k, n, col) for lab, k, n, col in bars]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(width, 0.34 * len(rows) + 1.0))
    yt, yl = [], []
    for i, (kind, lab, k, n, col) in enumerate(rows):
        y = len(rows) - 1 - i
        if kind == "H":
            ax.text(-1, y - 0.05, lab, fontsize=8.6, color=INK, fontweight="bold", va="center", ha="left",
                    transform=ax.get_yaxis_transform() if False else ax.transData)
            continue
        lo, hi = wil(k, n)
        ax.barh(y, 100 * k / n, height=0.62, color=col, edgecolor="none", zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1, zorder=3)
        ax.text(max(hi, 100 * k / n) + 1.5, y, f"{k}/{n}", va="center", fontsize=8.4, color=INK)
        yt.append(y)
        yl.append(lab)
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=8.4)
    ax.set_xlim(0, 112)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _axes_clean(ax)
    ax.set_title(title, fontsize=10, color=INK, loc="left")
    fig.text(0.01, -0.02, note, fontsize=7.4, color=INK2, va="top", ha="left", wrap=True)
    _save(fig, name)
    plt.close(fig)


def plot_prov_failure_map(prov, meas):
    """PROVISIONAL unedited success per task: headline run (filled, Wilson 95%) and every other earlier run."""
    head = {r["code"]: r for r in prov if r["experiment"] == "failure_map"}
    rows = [c for c in ORDER if c in head]
    plt = _plt()
    fig, ax = plt.subplots(figsize=(7.4, 0.4 * len(rows) + 1.3))
    for i, code in enumerate(rows):
        y = len(rows) - 1 - i
        r = head[code]
        k, n = r["unedited_k"], r["n"]
        lo, hi = wil(k, n)
        ax.plot([lo, hi], [y, y], color=BLUE, lw=1.6, alpha=0.45, solid_capstyle="round", zorder=2)
        for m in meas:
            if m["code"] == code and not (m["k"] == k and m["n"] == n and m["slots"] == r["slots"]):
                ax.plot([100 * m["k"] / m["n"]], [y], "o", ms=6, mfc="none", mec=MUTE, mew=1.1, zorder=3)
        ax.plot([100 * k / n], [y], "o", ms=8, color=BLUE, mec=SURF, mew=1.5, zorder=4)
        ax.text(104, y, f"{k}/{n}", va="center", fontsize=8.4, color=INK)
        ax.text(118, y, f"slots {r['slots'][0]}-{r['slots'][1]}", va="center", fontsize=7.8, color=INK2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[c] for c in reversed(rows)])
    ax.set_xlim(-2, 150)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
    _axes_clean(ax)
    ax.plot([], [], "o", ms=8, color=BLUE, label="largest earlier run (line: Wilson 95%)")
    ax.plot([], [], "o", ms=6, mfc="none", mec=MUTE, mew=1.1, label="other earlier unedited runs (other slots)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2, frameon=False, fontsize=7.8,
              labelcolor=INK2)
    ax.set_title("PROVISIONAL: π0.5 DROID unedited, default wording (earlier runs, RTX 3090, performance preset)",
                 fontsize=9.5, color=INK, loc="left")
    _save(fig, "provisional_failure_map")
    plt.close(fig)


def plot_provisional(prov, ctl):
    t1 = next((r for r in ctl if r["code"] == "T1"), None)
    if t1:
        ck = t1["control_k"]
        groups = [("T1 cube then banana, held-out slots 90000-90063 (every bar on the same 64 starts)", [
            ("unedited", t1["unedited_k"], t1["n"], NEUTRAL),
            ("random-direction control", ck["k"], ck["n"], NEUTRAL),
            ("edit v2mi-f (instruction unchanged)", ck["edit_vs_unedited"]["edit_k"], t1["n"], BLUE),
            ("explicit wording, unedited weights (teacher)", t1["teacher_k"]["k"], t1["n"], ORANGE)])]
        e = next(r for r in prov if r["experiment"] == "edit_target" and r["code"] == "T1"
                 and r["edit"]["name"] == "v2mi-f")
        groups.append((f"T1, all held-out slots {e['slots'][0]}-{e['slots'][1]}", [
            ("unedited", e["unedited_k"], e["n"], NEUTRAL), ("edit v2mi-f", e["edited_k"], e["n"], BLUE)]))
        plot_bars(groups, "provisional_t1_heldout",
                  "PROVISIONAL: the T1 edit against its random-direction control and the teacher wording",
                  f"Edit vs control on 90000-90063: {ck['edit_vs_control']['edit_only']} edit-only / "
                  f"{ck['edit_vs_control']['control_only']} control-only starts, p = "
                  f"{stats.format_p(ck['edit_vs_control']['p'])}; control vs unedited p = "
                  f"{stats.format_p(t1['p'])}; teacher vs edit {t1['teacher_k']['edit_vs_teacher']['teacher_only']} / "
                  f"{t1['teacher_k']['edit_vs_teacher']['edit_only']}, p = "
                  f"{stats.format_p(t1['teacher_k']['edit_vs_teacher']['p'])}. Earlier runs on ali-ms (RTX 3090, "
                  "performance preset). Whiskers: Wilson 95%.")
    where = [r for r in prov if r["experiment"] == "where_to_edit"]
    if where:
        w0 = where[0]
        bars = [("unedited", w0["unedited_k"], w0["n"], NEUTRAL)]
        for r in sorted(where, key=lambda r: r["edited_k"]):
            vc = r.get("values_changed")
            vtxt = (f"{vc / 1e6:.0f}M values" if vc and vc >= 1e6 else f"{vc:,} values" if vc else "")
            bars.append((f"{r['edit']['name']}" + (f"  ({vtxt})" if vtxt else ""), r["edited_k"], r["n"], BLUE))
        plot_bars([(f"T1, held-out slots {w0['slots'][0]}-{w0['slots'][1]}, all arms on the same starts", bars)],
                  "provisional_where_to_edit", "PROVISIONAL: where the T1 edit goes (one location per bar)",
                  "Values changed = stored weight values that differ from the unedited checkpoint. Earlier runs on "
                  "ali-ms (RTX 3090, performance preset). 12 of the 14 chunks ran on one GPU lane, so their second 16 "
                  "slots started from other scenes; study/results/opus/where/ repeats every comparison on the 32 "
                  "unaffected slots with the same ordering. Whiskers: Wilson 95%.", width=7.6)
    t8 = next((r for r in ctl if r["code"] == "T8"), None)
    w8 = next((r for r in prov if r["experiment"] == "wording" and r["code"] == "T8"), None)
    if t8 and w8:
        ck = t8["control_k"]
        groups = [(f"T8 mustard right of raisin, held-out slots {t8['slots'][0]}-{t8['slots'][1]}", [
            ("unedited, reference-first (RF) wording", t8["unedited_k"], t8["n"], NEUTRAL),
            ("RF + random-direction control", ck["k"], ck["n"], NEUTRAL),
            ("RF + edit s4r-grp3 (TF distilled)", ck["edit_vs_unedited"]["edit_k"], t8["n"], BLUE)]),
            (f"T8, practice slots {w8['slots'][0]}-{w8['slots'][1]} (wording screen)", [
                ("unedited, RF wording", w8["unedited_k"], w8["n"], NEUTRAL),
                ("unedited, target-first (TF) wording", w8["teacher_k"]["k"], w8["n"], ORANGE)])]
        plot_bars(groups, "provisional_t8", "PROVISIONAL: T8 wording gap and its own edit",
                  f"Held-out: edit vs control {ck['edit_vs_control']['edit_only']} / "
                  f"{ck['edit_vs_control']['control_only']}, p = {stats.format_p(ck['edit_vs_control']['p'])}. "
                  f"Practice: TF-only {w8['fixed']} / RF-only {w8['broke']}, p = {stats.format_p(w8['p'])}. "
                  "Earlier runs on ali-ms (RTX 3090, performance preset), study/results/sgdw/. "
                  "Whiskers: Wilson 95%.")


def main():
    prov = json.loads((HERE / "results/provisional.json").read_text())["records"]
    arms, meta = load_runs(RUNS)
    arms_old, meta_old = load_runs(RUNS_OLD, old=True)
    fresh, table = fresh_records(arms)
    fresh += calibration(arms, arms_old)
    fresh += fresh_partial(arms_old, meta_old)
    ctl = prov_controls()
    prov = prov + ctl
    gpu_s = sum((m.get("gpu_s") or 0) for m in meta)
    summary = {"model": MODEL, "hardware": HW_POD, "jobs_done": len(meta),
               "gpu_hours_pod_jobs": round(gpu_s / 3600, 2), "video_picks": video_picks(),
               "table": {c: {b: {a: ({k: x for k, x in v.items() if k not in ("per_slot", "slot_list")})
                                 for a, v in d.items()} for b, d in bl.items()} for c, bl in table.items()},
               "jobs": meta, "jobs_ali_ms": meta_old}
    (HERE / "results/summary.json").write_text(json.dumps(summary, indent=1) + "\n")
    METRICS.write_text(json.dumps(prov + fresh, indent=1) + "\n")
    plot_failure_map(table, prov)
    plot_prov_failure_map(prov, json.loads((HERE / "results/provisional.json").read_text())["unedited_measurements"])
    plot_provisional(prov, ctl)
    plot_dumbbell(table, "v2mi-f", "side_effects_v2mi-f",
                  "π0.5 DROID: the T1 edit v2mi-f on every task (same paired starts, slots 30000-30031)")
    plot_dumbbell(table, "abl-imgl8", "side_effects_abl-imgl8",
                  "π0.5 DROID: the 0.16% variant abl-imgl8 on every task (slots 30000-30031)")
    plot_dumbbell(prov_table(prov), "v2mi-f", "provisional_side_effects_v2mi-f",
                  "PROVISIONAL: earlier paired runs of v2mi-f (own slots, performance preset, RTX 3090)",
                  slot_labels=True)
    for b, slots in (("heldout", "96000-96063"), ("practice", "30000-30031")):
        if "T1" in table and b in table["T1"]:
            plot_arms(table["T1"][b], [("base", "unedited"), ("ctl-rand", "random-direction control"),
                                       ("v2mi-f", "edit v2mi-f"), ("teacher", "explicit wording (teacher)")],
                      f"t1_{b}", f"T1 cube then banana, {'held-out' if b == 'heldout' else 'practice'} slots {slots}",
                      "Same paired starts for every bar; default render preset; whiskers: Wilson 95%.")
        if "T8" in table and b in table["T8"]:
            plot_arms(table["T8"][b], [("base", "unedited, RF wording"), ("ctl-rand3", "RF + random control"),
                                       ("s4r-grp3", "RF + edit s4r-grp3"), ("v2mi-f", "RF + T1 edit v2mi-f"),
                                       ("teacher", "TF wording (teacher)")],
                      f"t8_{b}", f"T8 mustard right of raisin, {'held-out' if b == 'heldout' else 'practice'} "
                                 f"slots {slots}",
                      "RF = reference-first (the task's wording), TF = target-first. Whiskers: Wilson 95%.")
    print(f"{len(prov)} provisional + {len(fresh)} fresh records -> {METRICS}")
    for r in fresh:
        print(f"  {r['experiment']:30s} {r['code']} {r['edit']['name'] if r['edit'] else '':10s} n={r['n']:3d} "
              f"{r['unedited_k']} -> {r['edited_k']} fixed {r['fixed']} broke {r['broke']} p={r['p']}")


if __name__ == "__main__":
    main()
