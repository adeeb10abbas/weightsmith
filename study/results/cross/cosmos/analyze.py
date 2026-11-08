"""Tables, metrics and plots for Track A (Cosmos3 Edge Policy DROID) from the per-slot run records.

Reads results/runs/<machine>/<tag>.json (one row per slot, copied from the GPU hosts by scripts/sync_results.sh),
writes results/summary.json, ../metrics_cosmos.json (one record per experiment and machine) and plots/*.svg + *.png
(2x). Statistics come from weightsmith.stats (exact McNemar, Wilson, paired bootstrap), as in the paper.

An arm is a tag without its "-s<slot_start>" suffix, per machine; arms are only ever paired within one machine and
on the slots both ran. Arm names: <code>-base (unedited, default wording), <code>-<wording> (unedited, another
wording), <code>-E<edit> (edited weights, default wording), <code>-C<edit> (the edit's random-direction control).
    python3 study/results/cross/cosmos/analyze.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[3]))
from weightsmith import stats  # noqa: E402

RUNS = HERE / "results" / "runs"
PLOTS = HERE / "plots"
METRICS = HERE.parent / "metrics_cosmos.json"
MODEL = "nvidia/Cosmos3-Edge-Policy-DROID@a7c7288"
GPU_NAMES = {"ali-ms": "RTX 3090 24 GB", "korea-rtxpro4000": "RTX PRO 4000 Blackwell 24 GB"}

sys.path.insert(0, str(HERE / "scripts"))
from tasks import TASKS as _T, WORDINGS as _W  # noqa: E402

TASKS = {c: (t[0], t[1]) for c, t in _T.items()}
LABELS = {"T1": "T1 cube then banana", "T2": "T2 pick glasses", "T3": "T3 cube left of bowl",
          "T4": "T4 stack yellow on red", "T5": "T5 mustard in left bin", "T6": "T6 stack 3 cubes",
          "T7": "T7 pick green block", "T8": "T8 mustard right of raisin (RF)", "R1": "R1 cube in bowl"}
PRACTICE = range(30000, 30032)
HELDOUT = range(96000, 96064)
WORDINGS = [(c, w[0], w[1], w[2]) for c, w in _W.items()]
EDITS_FILE = HERE / "edits.json"  # edit definitions (name, task, location, build report), written as edits are frozen

INK, INK2, MUTE = "#1C1C1C", "#4D4D4D", "#858585"
# arm colours (validated with the dataviz skill's validate_palette.js, light surface, adjacent pairs: CVD dE >= 11.1,
# normal-vision dE >= 17.6; grey is the mandated neutral for "unedited" and always carries a direct label)
LIGHT, PALE, RULE = "#9A9A9A", "#ECECEC", "#A3A3A3"   # LIGHT = unedited
TEAL = "#0A8EA6"                                       # edited
CTRL = "#EB6834"                                       # random-direction control
TEAL_PALE = "#6250D6"                                  # unedited under the known-good wording (teacher)
RED = "#B1504A"


def load_runs():
    arms: dict[tuple[str, str], dict[int, bool]] = {}
    meta: dict[str, list[dict]] = {}
    for f in sorted(RUNS.glob("*/*.json")):
        d = json.loads(f.read_text())
        if d.get("completed") != d.get("n"):
            continue
        m = d.get("machine") or f.parent.name
        arm = d["tag"].rsplit("-s", 1)[0]
        rows = {r["slot"]: r["success"] for r in d["rows"] if r.get("ok")}
        arms.setdefault((m, arm), {}).update(rows)
        meta.setdefault(f"{m}/{arm}", []).append({k: d.get(k) for k in (
            "tag", "task", "prompt", "edit", "slots", "envs", "render", "n", "completed", "success", "gpu_s",
            "started", "finished")})
    return arms, meta


def rate(arm, slots):
    got = [arm[s] for s in slots if s in arm]
    return sum(got), len(got)


def slot_range(arm, slots):
    s = sorted(x for x in arm if x in slots)
    return [s[0], s[-1]] if s else None


def paired(ref, cand, slots=None):
    common = sorted(set(ref) & set(cand) & (set(slots) if slots is not None else set(ref)))
    if not common:
        return {}
    c = stats.compare({s: ref[s] for s in common}, {s: cand[s] for s in common})
    t = c.table
    lo, hi = c.difference_ci
    return {"n": len(common), "ref_k": t.both + t.broke, "cand_k": t.both + t.fixed, "fixed": t.fixed,
            "broke": t.broke, "both": t.both, "neither": t.neither, "p": stats.mcnemar_exact(t.fixed, t.broke),
            "diff_pts": round(100 * (t.fixed - t.broke) / len(common), 2),
            "diff_ci95_pts": [round(100 * lo, 2), round(100 * hi, 2)], "slots": [common[0], common[-1]]}


def wilson_pct(k, n):
    lo, hi = stats.wilson(k, n)
    return [round(100 * lo, 1), round(100 * hi, 1)]


def record(**kw):
    base = {"experiment": None, "machine": None, "model": MODEL, "task": None, "code": None, "wording_pair": None,
            "n": None, "slots": None, "unedited_k": None, "edited_k": None, "fixed": None, "broke": None, "p": None,
            "ci": None, "control_k": None, "teacher_k": None, "recovery_pct": None, "transfer": None,
            "retention": None, "edit_location": None, "values_changed": None, "notes": ""}
    base.update(kw)
    return base


# --------------------------------------------------------------------------------------------- plots
def _style():
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


def _axes_common(ax, xlabel, xmax=112):
    ax.set_xlim(0, xmax)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel(xlabel)
    ax.grid(axis="x", color=PALE, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def _ci(ax, y, k, n, h=0.1):
    lo, hi = stats.wilson(k, n)
    ax.plot([100 * lo, 100 * hi], [y, y], color=INK2, lw=1.0, zorder=3, solid_capstyle="butt")
    for x in (100 * lo, 100 * hi):
        ax.plot([x, x], [y - h, y + h], color=INK2, lw=1.0, zorder=3)
    return 100 * hi


def plot_failure_map(fmap, machine):
    rows = [r for r in fmap if r["machine"] == machine]
    if not rows:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        ax.barh(y, 100 * r["k"] / r["n"], height=0.5, color=LIGHT, edgecolor="none", zorder=2)
        hi = _ci(ax, y, r["k"], r["n"], 0.12)
        ax.text(max(hi, 100 * r["k"] / r["n"]) + 1.5, y, f"{r['k']}/{r['n']}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["code"]] for r in reversed(rows)])
    _axes_common(ax, "success, % of paired starts (bar), Wilson 95% interval")
    ns = sorted({r["n"] for r in rows})
    ax.set_title(f"Cosmos3 Edge Policy DROID, unedited, default wording ({machine}, n = {'/'.join(map(str, ns))})",
                 fontsize=9.5, color=INK, loc="left")
    _save(fig, f"failure_map_{machine}")
    plt.close(fig)


def plot_pairs(rows, name, title, a_lab, b_lab, label_fn):
    """rows: dicts with code, n, ref_k, cand_k, fixed, broke, p. grey = a (reference), teal = b."""
    if not rows:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.6, 0.62 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, k, color, lab in ((0.17, r["ref_k"], LIGHT, a_lab), (-0.17, r["cand_k"], TEAL, b_lab)):
            ax.barh(y + dy, 100 * k / r["n"], height=0.3, color=color, edgecolor="none", zorder=2)
            hi = _ci(ax, y + dy, k, r["n"], 0.08)
            ax.text(max(hi, 100 * k / r["n"]) + 1.5, y + dy, f"{k}/{r['n']} {lab}", va="center", fontsize=7.8,
                    color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{label_fn(r)}\n{r['fixed']} fixed / {r['broke']} broke, p = {stats.format_p(r['p'])}"
                        for r in reversed(rows)], fontsize=8.2)
    _axes_common(ax, "success, % of the same paired starts, Wilson 95% interval", 135)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    _save(fig, name)
    plt.close(fig)


def plot_bars(groups, name, title):
    """groups: list of (label, [(arm label, k, n, color)])."""
    if not groups:
        return
    plt = _style()
    nb = sum(len(g[1]) for g in groups)
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * nb + 0.45 * len(groups) + 0.8))
    y, ticks = 0.0, []
    for glabel, bars in reversed(groups):
        y0 = y
        for lab, k, n, color in reversed(bars):
            ax.barh(y, 100 * k / n, height=0.24, color=color, edgecolor="none", zorder=2)
            hi = _ci(ax, y, k, n, 0.07)
            ax.text(max(hi, 100 * k / n) + 1.5, y, f"{lab}: {k}/{n}", va="center", fontsize=7.8, color=INK)
            y += 0.32
        ticks.append(((y0 + y - 0.32) / 2, glabel))
        y += 0.35
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=8.5)
    _axes_common(ax, "success, % of paired starts, Wilson 95% interval", 140)
    from matplotlib.patches import Patch
    seen = {}
    for _, bars in groups:
        for lab, _k, _n, color in bars:
            seen.setdefault(lab, color)
    ax.legend(handles=[Patch(color=c, label=l) for l, c in seen.items()], loc="lower left", fontsize=7.5,
              frameon=False, ncol=len(seen), bbox_to_anchor=(0.0, 1.0), borderaxespad=0.2, handlelength=1.2,
              columnspacing=1.2)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left", pad=22)
    _save(fig, name)
    plt.close(fig)


def plot_side_effects(rows, name, title):
    """Paired change (edited - unedited, points) per task with its bootstrap 95% interval."""
    if not rows:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 0.36 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        d, (lo, hi) = r["diff_pts"], r["diff_ci95_pts"]
        color = TEAL if d > 0 else (RED if d < 0 else MUTE)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.0, zorder=2)
        ax.plot([d], [y], "o", ms=6, color=color, zorder=3, mec="white", mew=1.2)
        ax.text(max(hi, d) + 3, y, f"{r['ref_k']}/{r['n']} -> {r['cand_k']}/{r['n']}  (+{r['fixed']} / -{r['broke']}, "
                f"p = {stats.format_p(r['p'])})", va="center", fontsize=7.8, color=INK)
    ax.axvline(0, color=RULE, lw=0.8, zorder=1)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["code"]] for r in reversed(rows)])
    ax.set_xlim(-100, 160)
    ax.set_xticks([-100, -50, 0, 50, 100])
    ax.set_xlabel("edited minus unedited success, points (paired bootstrap 95% interval)")
    ax.grid(axis="x", color=PALE, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title(title, fontsize=9.5, color=INK, loc="left")
    _save(fig, name)
    plt.close(fig)


# --------------------------------------------------------------------------------------------- main
def main():
    arms, meta = load_runs()
    machines = sorted({m for m, _ in arms})
    edits = json.loads(EDITS_FILE.read_text()) if EDITS_FILE.exists() else []
    records, fmap, wrows, side, edit_rows = [], [], [], [], []
    for m in machines:
        # metric 1: failure map on practice slots
        for code, (task, instr) in TASKS.items():
            arm = arms.get((m, f"{code}-base"))
            k, n = rate(arm or {}, PRACTICE)
            if not n:
                continue
            fmap.append({"machine": m, "code": code, "task": task, "k": k, "n": n, "wilson95": wilson_pct(k, n),
                         "slots": slot_range(arm, PRACTICE)})
            records.append(record(experiment="failure_map", machine=m, task=task, code=code, wording_pair=[instr],
                                  n=n, slots=slot_range(arm, PRACTICE), unedited_k=k,
                                  ci={"wilson95_pct": wilson_pct(k, n)},
                                  notes="unedited policy, task's default instruction, practice slots, default render"))
        # metric 2: wording screens on practice slots
        for code, suffix, alt, why in WORDINGS:
            base, alt_arm = arms.get((m, f"{code}-base")), arms.get((m, f"{code}-{suffix}"))
            c = paired(base or {}, alt_arm or {}, PRACTICE) if base and alt_arm else {}
            if not c:
                continue
            wrows.append({"machine": m, "code": code, "alt": alt, **c})
            records.append(record(experiment="wording", machine=m, task=TASKS[code][0], code=code,
                                  wording_pair=[TASKS[code][1], alt], n=c["n"], slots=c["slots"],
                                  unedited_k=c["ref_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
                                  ci={"diff_pts": c["diff_pts"], "bootstrap95_pts": c["diff_ci95_pts"],
                                      "wilson95_orig_pct": wilson_pct(c["ref_k"], c["n"]),
                                      "wilson95_alt_pct": wilson_pct(c["cand_k"], c["n"])},
                                  teacher_k=c["cand_k"],
                                  notes=f"same weights, wording via prompt override; fixed = only the alternative "
                                        f"wording succeeds. {why}"))
        # metric 3-5: edits
        for e in edits:
            code, name, teach = e["code"], e["name"], e.get("teacher_arm")
            base = arms.get((m, f"{code}-base"))
            ed, ctl = arms.get((m, f"{code}-E{name}")), arms.get((m, f"{code}-C{name}"))
            tch = arms.get((m, f"{code}-{teach}")) if teach else None
            for split, slots in (("practice", PRACTICE), ("heldout", HELDOUT)):
                c = paired(base or {}, ed or {}, slots) if base and ed else {}
                if not c:
                    continue
                cc = paired(base, ctl, slots) if ctl else {}
                ct = paired(base, tch, slots) if tch else {}
                ec = paired(ctl, ed, slots) if ctl else {}
                rec_pct = None
                if ct and ct["cand_k"] != ct["ref_k"] and ct["n"] == c["n"]:
                    rec_pct = round(100 * (c["cand_k"] - c["ref_k"]) / (ct["cand_k"] - ct["ref_k"]), 1)
                row = {"machine": m, "code": code, "edit": name, "split": split, "location": e["location"], **c,
                       "control": cc or None, "teacher": ct or None, "edit_vs_control": ec or None,
                       "recovery_pct": rec_pct}
                edit_rows.append(row)
                records.append(record(
                    experiment=f"edit_{split}", machine=m, task=TASKS[code][0], code=code,
                    wording_pair=[TASKS[code][1], e.get("good_wording")], n=c["n"], slots=c["slots"],
                    unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"], broke=c["broke"], p=c["p"],
                    ci={"diff_pts": c["diff_pts"], "bootstrap95_pts": c["diff_ci95_pts"],
                        "wilson95_unedited_pct": wilson_pct(c["ref_k"], c["n"]),
                        "wilson95_edited_pct": wilson_pct(c["cand_k"], c["n"])},
                    control_k=cc.get("cand_k") if cc else None, teacher_k=ct.get("cand_k") if ct else None,
                    control_n=cc.get("n") if cc else None, teacher_n=ct.get("n") if ct else None,
                    edit_vs_control={k: ec[k] for k in ("n", "ref_k", "cand_k", "fixed", "broke", "p", "slots")}
                    if ec else None,
                    control_slots=cc.get("slots") if cc else None, teacher_slots=ct.get("slots") if ct else None,
                    recovery_pct=rec_pct, edit_location=e["location"], values_changed=e.get("values_changed"),
                    notes=f"edit {name} ({e.get('summary', '')}); control: same rank and singular values, random "
                          f"directions" + (f"; control vs unedited fixed {cc['fixed']} broke {cc['broke']} p = "
                                           f"{cc['p']:.3g}" if cc else "") +
                          (f"; edit vs control p = {ec['p']:.3g}" if ec else "")))
            # side effects: the edited weights on every other task, practice slots, default wording
            for oc in TASKS:
                if oc == code:
                    continue
                b2, e2 = arms.get((m, f"{oc}-base")), arms.get((m, f"{oc}-E{name}"))
                c = paired(b2 or {}, e2 or {}, PRACTICE) if b2 and e2 else {}
                if not c:
                    continue
                side.append({"machine": m, "edit": name, "code": oc, **c})
                records.append(record(
                    experiment="side_effect", machine=m, task=TASKS[oc][0], code=oc, wording_pair=[TASKS[oc][1]],
                    n=c["n"], slots=c["slots"], unedited_k=c["ref_k"], edited_k=c["cand_k"], fixed=c["fixed"],
                    broke=c["broke"], p=c["p"], ci={"diff_pts": c["diff_pts"], "bootstrap95_pts": c["diff_ci95_pts"]},
                    transfer=c["diff_pts"] if oc != "R1" else None, retention=c["diff_pts"] if oc == "R1" else None,
                    edit_location=e["location"], values_changed=e.get("values_changed"),
                    notes=f"edit {name} built for {code}, run on {oc} with {oc}'s default instruction"))
    # where to edit, offline: replayed RF-failure decisions, action-chunk distance to the teacher (TF wording)
    where = []
    for m in machines:
        f = RUNS.parent / "offline" / m / "t8_offline_check.json"
        if not f.exists():
            continue
        oc = json.loads(f.read_text())
        loc = {e["name"]: e["location"] for e in edits}
        for g, gd in oc["groups"].items():
            if g == "all":
                continue
            for var, ed in (("O+und", "t8und"), ("O+gen", "t8gen")):
                if var not in gd:
                    continue
                v, c = gd[var], gd.get(var + "-ctl", {})
                row = {"machine": m, "edit": ed, "group": g, "n": v["n"],
                       "unedited_ee_cm": round(gd["student_vs_teacher"]["ee_end_cm"], 2),
                       "edited_ee_cm": round(v["vs_teacher"]["ee_end_cm"], 2),
                       "control_ee_cm": round(c["vs_teacher"]["ee_end_cm"], 2) if c else None,
                       "gap_closed_pct": round(v["gap_closed_ee_pct"], 1),
                       "control_gap_closed_pct": round(c["gap_closed_ee_pct"], 1) if c else None,
                       "closer_to_teacher": v["closer_to_teacher_ee"],
                       "control_closer_to_teacher": c.get("closer_to_teacher_ee")}
                where.append(row)
                records.append(record(
                    experiment="where_to_edit_offline", machine=m, task=TASKS["T8"][0], code="T8",
                    wording_pair=[TASKS["T8"][1], WORDINGS[[w[0] for w in WORDINGS].index("T8")][2]], n=v["n"],
                    edit_location=loc.get(ed), offline=row,
                    values_changed=next((e.get("values_changed") for e in edits if e["name"] == ed), None),
                    notes=f"offline replay of {v['n']} RF-failure decisions ({g} set: "
                          + ("decisions the gen edit was built from, slots 30000-30005" if g == "build" else
                             "held-aside decisions, slots 30006-30011, never used to build either edit")
                          + ", replans 1-5): chunk-end end-effector distance to the TF-wording action chunk, same "
                            "seed; gap closed = 1 - edited/unedited distance; control = same-size random directions"))
    summary = {"model": MODEL, "where_offline": where, "machines": {m: GPU_NAMES.get(m) for m in machines},
               "arms": {f"{m}/{a}": {"slots": sorted(v), "k": sum(v.values()), "n": len(v)} for (m, a), v in arms.items()},
               "failure_map": fmap, "wording": wrows, "edits": edit_rows, "side_effects": side, "runs": meta,
               "records": len(records)}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    METRICS.write_text(json.dumps(records, indent=1))
    for m in machines:
        plot_failure_map(fmap, m)
        plot_pairs([r for r in wrows if r["machine"] == m], f"wording_{m}",
                   f"Wording sensitivity, unedited weights, same starts ({m})", "original", "alternative",
                   lambda r: LABELS[r["code"]])
        for e in edits:
            groups = []
            for r in [r for r in edit_rows if r["machine"] == m and r["edit"] == e["name"]]:
                bars = [("unedited", r["ref_k"], r["n"], LIGHT)]
                if r["control"]:
                    bars.append(("random control", r["control"]["cand_k"], r["control"]["n"], CTRL))
                bars.append(("edited", r["cand_k"], r["n"], TEAL))
                if r["teacher"]:
                    bars.append(("unedited, good wording", r["teacher"]["cand_k"], r["teacher"]["n"], TEAL_PALE))
                groups.append((f"{r['split']}\n{r['slots'][0]}-{r['slots'][1]}", bars))
            plot_bars(groups, f"edit_{e['name']}_{m}", f"{LABELS[e['code']]}: edit {e['name']} ({e['location']}), {m}")
            plot_side_effects([r for r in side if r["machine"] == m and r["edit"] == e["name"]],
                              f"side_effects_{e['name']}_{m}",
                              f"Edit {e['name']} on the other tasks, practice starts ({m})")
    for r in fmap:
        print(f"{r['machine']:7s} {r['code']:3s} {r['task']:28s} {r['k']:3d}/{r['n']:<3d} Wilson95 {r['wilson95']}")
    for r in wrows:
        print(f"{r['machine']:7s} {r['code']} wording: {r['ref_k']}/{r['n']} -> {r['cand_k']}/{r['n']}  fixed "
              f"{r['fixed']} broke {r['broke']} p={r['p']:.3g}")
    for r in edit_rows:
        print(f"{r['machine']:7s} {r['code']} {r['edit']} {r['split']}: {r['ref_k']}/{r['n']} -> {r['cand_k']}/{r['n']} "
              f"fixed {r['fixed']} broke {r['broke']} p={r['p']:.3g} "
              f"ctl={r['control'] and str(r['control']['cand_k']) + '/' + str(r['control']['n'])} "
              f"teacher={r['teacher'] and str(r['teacher']['cand_k']) + '/' + str(r['teacher']['n'])} "
              f"recovery={r['recovery_pct']}")
    for r in where:
        print(f"{r['machine']:7s} where {r['edit']} {r['group']}: ee {r['unedited_ee_cm']} -> {r['edited_ee_cm']} cm "
              f"(gap closed {r['gap_closed_pct']}%, control {r['control_gap_closed_pct']}%), n={r['n']}")
    for r in side:
        print(f"{r['machine']:7s} side {r['edit']} {r['code']}: {r['ref_k']}/{r['n']} -> {r['cand_k']}/{r['n']} "
              f"+{r['fixed']}/-{r['broke']} p={r['p']:.3g}")


if __name__ == "__main__":
    main()
