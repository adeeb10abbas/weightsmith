"""Tables, metrics_groot.json and plots for Track B (GR00T N1.7 DROID). Run from the repo root:

    python3 study/results/cross/groot/scripts/analyze.py

Reads the per-run summaries in study/results/cross/groot/results/runs/*.json (one per evaluation, rows = slots), pairs
arms on equal slots, and writes study/results/cross/metrics_groot.json, results/tables.md and plots/*.svg|png.
Statistics come from weightsmith/stats.py: Wilson 95% intervals, exact two-sided McNemar on the discordant pairs,
paired bootstrap 95% interval of the difference (20,000 draws, default_rng(0)).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))
from weightsmith.stats import PairedTable, mcnemar_exact, paired_bootstrap, wilson  # noqa: E402

G = ROOT / "study/results/cross/groot"
RUNS = G / "results/runs"
PLOTS = G / "plots"
METRICS = ROOT / "study/results/cross/metrics_groot.json"
MODEL = "GR00T N1.7 DROID (nvidia/GR00T-N1.7-DROID@05e7cc97)"

TASKS = [  # (id, RoboLab class, default instruction)
    ("T1", "RubiksCubeThenBananaTask", "Put the cube then the banana in the bowl"),
    ("T2", "PickGlassesTask", "Pick up the eye glasses"),
    ("T3", "RubiksCubeLeftOfBowlTask", "Put the rubiks cube to the left of the bowl"),
    ("T4", "StackYellowOnRedTask", "Stack the yellow block on the red block"),
    ("T5", "MustardInLeftBinTask", "Put the mustard in the left bin"),
    ("T6", "Stack3RubiksCubeTask", "Stack the rubiks cubes in a tower"),
    ("T7", "PickUpGreenObjectTask", "Pick up the green vegetable block"),
    ("T8", "MustardRightOfRaisinRFTask", "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."),
    ("R1", "RubiksCubeTask", "Put the cube in the bowl"),
]
TID = {c: t for t, c, _ in TASKS}


FINDINGS = {  # observations about a run that the numbers alone do not show (written after looking at the run)
    "wd-T3-alt": "not wording-sensitive: under both wordings GR00T grasps the cube and holds it near the bowl until "
                 "the episode ends (final frames of slots 30000-30005)",
    "wd-T8-tf": "the target-first wording does not rescue GR00T (0/32 under both; Cosmos3 Edge went 0 -> 15/16 under "
                "TF in Track A); under TF every episode grasps the mustard (RF: 10/32 never grasp it) but none ends "
                "with it right of the raisin box",
    "wd-T1-expl": "GR00T's only wording-sensitive task among those screened (T1, T3, T8)",
}


def load_runs():
    runs = {}
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        runs[d["tag"]] = d
    return runs


def machine(*rs):
    ms = sorted({r.get("machine", "ali-ms-rtx3090") for r in rs if r})
    return ms[0] if len(ms) == 1 else "+".join(ms)


def outcomes(run):
    return {r["slot"]: bool(r["success"]) for r in run["rows"] if r.get("ok")}


def rate(run):
    o = outcomes(run)
    k, n = sum(o.values()), len(o)
    lo, hi = wilson(k, n) if n else (float("nan"), float("nan"))
    return k, n, lo, hi


def pair(ref, cand):
    a, b = outcomes(ref), outcomes(cand)
    slots = sorted(set(a) & set(b))
    both = sum(a[s] and b[s] for s in slots)
    fixed = sum((not a[s]) and b[s] for s in slots)
    broke = sum(a[s] and not b[s] for s in slots)
    neither = len(slots) - both - fixed - broke
    t = PairedTable(both=both, fixed=fixed, broke=broke, neither=neither)
    lo, hi = paired_bootstrap(t)
    return {"n": len(slots), "ref_k": both + broke, "cand_k": both + fixed, "fixed": fixed, "broke": broke,
            "p": mcnemar_exact(fixed, broke), "diff": (fixed - broke) / len(slots), "diff_ci95": [lo, hi],
            "slots": [slots[0], slots[-1]] if slots else None}


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def main():
    runs = load_runs()
    recs, lines = [], []
    # ---------------------------------------------------------------- metric 1: failure map
    lines += ["## Metric 1: failure map (unedited, task's own wording, slots 30000-30031)", "",
              "| Task | RoboLab task | Success | Rate | Wilson 95% |", "|---|---|---|---|---|"]
    fm = {}
    for tid, cls, instr in TASKS:
        r = runs.get(f"fm-{tid}")
        if not r:
            continue
        k, n, lo, hi = rate(r)
        fm[tid] = (k, n, lo, hi)
        lines.append(f"| {tid} | {cls} | {k}/{n} | {100*k/n:.0f}% | {100*lo:.0f}-{100*hi:.0f}% |")
        recs.append({"experiment": "failure_map", "model": MODEL, "machine": machine(r), "task": f"{tid} {cls}", "wording_pair": [instr, None],
                     "n": n, "unedited_k": k, "edited_k": None, "fixed": None, "broke": None, "p": None,
                     "ci": {"unedited_wilson95": [lo, hi]}, "control_k": None, "teacher_k": None, "recovery_pct": None,
                     "transfer": None, "retention": None, "edit_location": None, "values_changed": None,
                     "slots": r["slots"], "notes": f"tag {r['tag']}; envs {r['envs']}; env seed {r.get('env_seed', 0)}; "
                                                    f"one fresh simulator process per 16-slot group"})
    # ---------------------------------------------------------------- metric 2: wording sensitivity
    wd = [t for t in runs if t.startswith("wd-")]
    if wd:
        lines += ["", "## Metric 2: wording sensitivity (unedited, paired on slots 30000-30031)", "",
                  "| Task | Original | Alternative wording | Original k/n | Alternative k/n | alt only / orig only | exact p |",
                  "|---|---|---|---|---|---|---|"]
    for tag in sorted(wd):
        tid = tag.split("-")[1]
        base = runs.get(f"fm-{tid}")
        if not base:
            continue
        c = pair(base, runs[tag])
        orig = next(i for t, _, i in TASKS if t == tid)
        lines.append(f"| {tid} | {orig} | {runs[tag]['prompt']} | {c['ref_k']}/{c['n']} | {c['cand_k']}/{c['n']} | "
                     f"{c['fixed']} / {c['broke']} | {fmt_p(c['p'])} |")
        recs.append({"experiment": "wording_sensitivity", "model": MODEL, "machine": machine(base, runs[tag]), "task": f"{tid} {base['task']}",
                     "wording_pair": [orig, runs[tag]["prompt"]], "n": c["n"], "unedited_k": c["ref_k"],
                     "edited_k": None, "fixed": c["fixed"], "broke": c["broke"], "p": c["p"],
                     "ci": {"diff_bootstrap95": c["diff_ci95"]}, "control_k": None, "teacher_k": c["cand_k"],
                     "recovery_pct": None, "transfer": None, "retention": None, "edit_location": None,
                     "values_changed": None, "slots": c["slots"],
                     "notes": f"alternative wording given to the unedited model (teacher); fixed = succeeds only "
                              f"under the alternative wording; tags {base['tag']} vs {tag}"
                              + (f"; {FINDINGS[tag]}" if tag in FINDINGS else "")})
    # ---------------------------------------------------------------- render-preset sensitivity
    for tag in sorted(t for t in runs if t.startswith("rp-")):
        tid = tag.split("-")[1]
        base = runs.get(f"fm-{tid}")
        if not base:
            continue
        c = pair(runs[tag], base)  # reference = performance preset, candidate = default preset
        lines += ["", f"## Render-preset sensitivity ({tid}, unedited, paired slots {c['slots'][0]}-{c['slots'][1]})", "",
                  "| Preset | Success | default only / performance only | exact p |", "|---|---|---|---|",
                  f"| performance (`--rendering-type performance`) | {c['ref_k']}/{c['n']} | | |",
                  f"| default (unset) | {c['cand_k']}/{c['n']} | {c['fixed']} / {c['broke']} | {fmt_p(c['p'])} |"]
        recs.append({"experiment": "render_preset_sensitivity", "model": MODEL, "machine": machine(base, runs[tag]), "task": f"{tid} {base['task']}",
                     "wording_pair": None, "n": c["n"], "unedited_k": c["cand_k"], "edited_k": None,
                     "fixed": c["fixed"], "broke": c["broke"], "p": c["p"], "ci": {"diff_bootstrap95": c["diff_ci95"]},
                     "control_k": None, "teacher_k": None, "recovery_pct": None, "transfer": None, "retention": None,
                     "edit_location": None, "values_changed": None, "slots": c["slots"],
                     "notes": f"unedited model; default preset {c['cand_k']}/{c['n']} vs performance preset "
                              f"{c['ref_k']}/{c['n']} on the same slots; fixed = succeeds only under the default preset"})
    # ---------------------------------------------------------------- edits (practice, control, held-out, side effects)
    recs += edit_records(runs, lines)
    METRICS.write_text(json.dumps({"track": "B", "model": MODEL, "records": recs}, indent=1))
    (G / "results/tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if fm:
        plot_failure_map(fm)
    for e, m in EDITS.items():
        plot_edit(runs, e, m)


EDITS = {}  # filled from results/edits.json: {edit: {"task": tid, "location": ..., "values_changed": ..., "wording": [o, x]}}


def edit_records(runs, lines):
    meta_f = G / "results/edits.json"
    if not meta_f.exists():
        return []
    EDITS.update(json.loads(meta_f.read_text()))
    recs = []
    for e, m in EDITS.items():
        tid = m["task"]
        base, teach = runs.get(f"fm-{tid}"), runs.get(m.get("teacher_tag", ""))
        ed, ctl = runs.get(f"ed-{e}-{tid}"), runs.get(f"ctl-{e}-{tid}")
        if not (base and ed):
            continue
        c = pair(base, ed)
        cc = pair(base, ctl) if ctl else None
        if not m.get("values_changed") and ed.get("edit_report"):
            m["values_changed"] = ed["edit_report"].get("values_changed")
            m["values_total"] = ed["edit_report"].get("values_total")
        if ctl and ctl.get("edit_report"):
            m["control_values_changed"] = ctl["edit_report"].get("values_changed")
        tk = pair(base, teach)["cand_k"] if teach else None
        rec_pct = (100 * (c["cand_k"] - c["ref_k"]) / (tk - c["ref_k"])) if tk is not None and tk != c["ref_k"] else None
        lines += ["", f"## Edit `{e}` ({m['location']}) on {tid}, practice slots {c['slots'][0]}-{c['slots'][1]}", "",
                  "| Arm | Success | fixed / broke vs unedited | exact p | 95% CI of difference |", "|---|---|---|---|---|",
                  f"| unedited | {c['ref_k']}/{c['n']} | | | |",
                  f"| edited | {c['cand_k']}/{c['n']} | {c['fixed']} / {c['broke']} | {fmt_p(c['p'])} | "
                  f"{100*c['diff_ci95'][0]:+.0f} to {100*c['diff_ci95'][1]:+.0f} pts |"]
        if cc:
            lines.append(f"| random-direction control | {cc['cand_k']}/{cc['n']} | {cc['fixed']} / {cc['broke']} | "
                         f"{fmt_p(cc['p'])} | {100*cc['diff_ci95'][0]:+.0f} to {100*cc['diff_ci95'][1]:+.0f} pts |")
        if tk is not None:
            lines.append(f"| teacher (explicit wording, unedited) | {tk}/{c['n']} | | | |")
        recs.append({"experiment": "wsp_edit_practice", "model": MODEL, "machine": machine(base, ed, ctl), "task": f"{tid} {base['task']}",
                     "wording_pair": m.get("wording"), "n": c["n"], "unedited_k": c["ref_k"], "edited_k": c["cand_k"],
                     "fixed": c["fixed"], "broke": c["broke"], "p": c["p"], "ci": {"diff_bootstrap95": c["diff_ci95"]},
                     "control_k": cc["cand_k"] if cc else None, "control_fixed_broke": [cc["fixed"], cc["broke"]] if cc else None,
                     "control_p": cc["p"] if cc else None, "teacher_k": tk,
                     "recovery_pct": rec_pct, "transfer": None, "retention": None, "edit_location": m["location"],
                     "values_changed": m.get("values_changed"), "slots": c["slots"], "notes": m.get("notes", "")})
        # held-out
        hb, he = runs.get(f"ho-{tid}-base"), runs.get(f"ho-{tid}-{e}")
        if hb and he:
            h = pair(hb, he)
            hc = runs.get(f"ho-{tid}-ctl-{e}")
            hcc = pair(hb, hc) if hc else None
            lines += ["", f"### Held-out ({tid}, slots {h['slots'][0]}-{h['slots'][1]})", "",
                      f"edited {h['cand_k']}/{h['n']} vs unedited {h['ref_k']}/{h['n']}: {h['fixed']} fixed / {h['broke']} broke, "
                      f"exact p = {fmt_p(h['p'])}, difference {100*h['diff']:+.1f} pts "
                      f"[{100*h['diff_ci95'][0]:+.1f}, {100*h['diff_ci95'][1]:+.1f}]"
                      + (f"; random control {hcc['cand_k']}/{hcc['n']} ({hcc['fixed']}/{hcc['broke']}, p = {fmt_p(hcc['p'])})" if hcc else "")]
            lo, hi = wilson(h["cand_k"], h["n"])
            recs.append({"experiment": "wsp_edit_heldout", "model": MODEL, "machine": machine(hb, he, hc), "task": f"{tid} {hb['task']}",
                         "wording_pair": m.get("wording"), "n": h["n"], "unedited_k": h["ref_k"], "edited_k": h["cand_k"],
                         "fixed": h["fixed"], "broke": h["broke"], "p": h["p"],
                         "ci": {"diff_bootstrap95": h["diff_ci95"], "edited_wilson95": [lo, hi],
                                "unedited_wilson95": list(wilson(h["ref_k"], h["n"]))},
                         "control_k": hcc["cand_k"] if hcc else None, "teacher_k": None, "recovery_pct": None,
                         "transfer": None, "retention": None, "edit_location": m["location"],
                         "values_changed": m.get("values_changed"), "slots": h["slots"], "notes": "held-out slots never used for design"})
        # side effects / transfer / retention: the edited model on every task, paired with the failure map
        se = []
        for t2, cls2, instr2 in TASKS:
            r = runs.get(f"se-{e}-{t2}") or (ed if t2 == tid else None)
            b2 = runs.get(f"fm-{t2}")
            if r and b2:
                se.append((t2, pair(b2, r)))
        if se:
            lines += ["", f"### Side-effect map of `{e}` (edited vs unedited, each task's own wording, slots 30000-30031)", "",
                      "| Task | Unedited | Edited | fixed / broke | exact p |", "|---|---|---|---|---|"]
            for t2, c2 in se:
                lines.append(f"| {t2} | {c2['ref_k']}/{c2['n']} | {c2['cand_k']}/{c2['n']} | {c2['fixed']} / {c2['broke']} | {fmt_p(c2['p'])} |")
                if t2 != tid:
                    recs.append({"experiment": "wsp_side_effect" if t2 != "R1" else "wsp_retention", "model": MODEL, "machine": machine(runs[f"fm-{t2}"], runs.get(f"se-{e}-{t2}")),
                                 "task": f"{t2} {runs[f'fm-{t2}']['task']}", "wording_pair": None, "n": c2["n"],
                                 "unedited_k": c2["ref_k"], "edited_k": c2["cand_k"], "fixed": c2["fixed"],
                                 "broke": c2["broke"], "p": c2["p"], "ci": {"diff_bootstrap95": c2["diff_ci95"]},
                                 "control_k": None, "teacher_k": None, "recovery_pct": None,
                                 "transfer": f"edit `{e}` built on {tid}" if t2 != "R1" else None,
                                 "retention": f"{c2['cand_k']}/{c2['n']} vs {c2['ref_k']}/{c2['n']}" if t2 == "R1" else None,
                                 "edit_location": m["location"], "values_changed": m.get("values_changed"),
                                 "slots": c2["slots"], "notes": f"edit {e} unchanged, task's own wording"})
            m["_side_effects"] = se
    return recs


def _style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "svg.fonttype": "none"})
    return plt


TEAL, GREY, INK = "#147D92", "#9AA0A6", "#202124"


def plot_failure_map(fm):
    plt = _style()
    tids = [t for t, _, _ in TASKS if t in fm]
    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    xs = range(len(tids))
    vals = [fm[t][0] / fm[t][1] for t in tids]
    los = [vals[i] - fm[t][2] for i, t in enumerate(tids)]
    his = [fm[t][3] - vals[i] for i, t in enumerate(tids)]
    ax.bar(xs, vals, color=GREY, width=0.62, zorder=2)
    ax.errorbar(xs, vals, yerr=[los, his], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
    for i, t in enumerate(tids):
        ax.text(i, min(1.0, fm[t][3]) + 0.03, f"{fm[t][0]}/{fm[t][1]}", ha="center", va="bottom", fontsize=8.5, color=INK)
    ax.set_xticks(list(xs), tids)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
    ax.set_ylabel("Success (%)")
    ax.set_title("GR00T N1.7 DROID, unedited: success per task, k/n starts from slot 30000 (Wilson 95%)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#E8EAED", zorder=0)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "failure_map.svg")
    fig.savefig(PLOTS / "failure_map.png", dpi=200)
    plt.close(fig)



def plot_edit(runs, e, m):
    """Edit vs control (and teacher) on practice, held-out if present; side-effect map if present."""
    plt = _style()
    tid = m["task"]
    base = runs.get(f"fm-{tid}")
    arms = [("unedited", base, GREY), ("random control", runs.get(f"ctl-{e}-{tid}"), "#C9CDD2"),
            ("WSP edit", runs.get(f"ed-{e}-{tid}"), TEAL), ("explicit wording\n(teacher)", runs.get(m.get("teacher_tag", "")), "#7FB8C4")]
    arms = [a for a in arms if a[1]]
    groups = [("practice (30000-30031)", arms)]
    hb, he = runs.get(f"ho-{tid}-base"), runs.get(f"ho-{tid}-{e}")
    if hb and he:
        hold = [("unedited", hb, GREY), ("random control", runs.get(f"ho-{tid}-ctl-{e}"), "#C9CDD2"), ("WSP edit", he, TEAL)]
        groups.append(("held-out (96000-96063)", [a for a in hold if a[1]]))
    fig, axes = plt.subplots(1, len(groups), figsize=(3.6 * len(groups) + 0.6, 3.3), squeeze=False)
    for ax, (title, arms_) in zip(axes[0], groups):
        for i, (name, run, col) in enumerate(arms_):
            k, n, lo, hi = rate(run)
            ax.bar(i, k / n, color=col, width=0.66, zorder=2)
            ax.errorbar(i, k / n, yerr=[[k / n - lo], [hi - k / n]], fmt="none", ecolor=INK, elinewidth=1, capsize=3, zorder=3)
            ax.text(i, hi + 0.03, f"{k}/{n}", ha="center", fontsize=8.5, color=INK)
        ax.set_xticks(range(len(arms_)), [a[0] for a in arms_], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
        ax.set_title(title, fontsize=9.5, loc="left")
        ax.grid(axis="y", color="#E8EAED", zorder=0)
    axes[0][0].set_ylabel("Success (%)")
    fig.suptitle(f"GR00T N1.7 DROID, {tid}: closed-form edit `{e}` ({m['location']}), Wilson 95%", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / f"edit_vs_control_{e}.svg")
    fig.savefig(PLOTS / f"edit_vs_control_{e}.png", dpi=200)
    plt.close(fig)
    se = m.get("_side_effects")
    if se:
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        x = range(len(se))
        for i, (t2, c2) in enumerate(se):
            ax.bar(i - 0.18, c2["ref_k"] / c2["n"], width=0.34, color=GREY, zorder=2)
            ax.bar(i + 0.18, c2["cand_k"] / c2["n"], width=0.34, color=TEAL, zorder=2)
            star = "*" if c2["p"] < 0.05 else ""
            ax.text(i, max(c2["ref_k"], c2["cand_k"]) / c2["n"] + 0.04, f"+{c2['fixed']}/-{c2['broke']}{star}", ha="center", fontsize=7.5, color=INK)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(color=GREY, label="unedited"), Patch(color=TEAL, label="edited")], frameon=False,
                  fontsize=8, loc="upper right", ncol=2)
        ax.set_xticks(list(x), [t for t, _ in se])
        ax.set_ylim(0, 1.15)
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0], ["0", "25", "50", "75", "100"])
        ax.set_ylabel("Success (%)")
        ax.set_title(f"Side effects of `{e}` (built on {tid}): +fixed/-broke vs unedited, * exact p < 0.05",
                     fontsize=9, loc="left")
        ax.grid(axis="y", color="#E8EAED", zorder=0)
        fig.tight_layout()
        fig.savefig(PLOTS / f"side_effects_{e}.svg")
        fig.savefig(PLOTS / f"side_effects_{e}.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
