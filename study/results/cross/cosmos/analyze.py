"""Tables, metrics and plots for Track A (Cosmos3 Edge Policy DROID) from the per-slot run records.

Reads results/runs/<tag>.json (one row per slot, copied from the GPU host by scripts/sync_results.sh), writes
results/summary.json, ../metrics_cosmos.json (one record per experiment) and plots/*.svg + *.png (2x).
Statistics come from weightsmith.stats (exact McNemar, Wilson, paired bootstrap), as in the paper.

An arm is a tag without its "-s<slot_start>" suffix; arms are paired on the slots both ran.
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

TASKS = {  # code: (RoboLab task class, default instruction)
    "T1": ("RubiksCubeThenBananaTask", "Put the cube then the banana in the bowl"),
    "T2": ("PickGlassesTask", "Pick up the eye glasses"),
    "T3": ("RubiksCubeLeftOfBowlTask", "Put the rubiks cube to the left of the bowl"),
    "T4": ("StackYellowOnRedTask", "Stack the yellow block on the red block"),
    "T5": ("MustardInLeftBinTask", "Put the mustard in the left bin"),
    "T6": ("Stack3RubiksCubeTask", "Stack the rubiks cubes in a tower"),
    "T7": ("PickUpGreenObjectTask", "Pick up the green vegetable block"),
    "T8": ("MustardRightOfRaisinRFTask",
           "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."),
    "R1": ("RubiksCubeTask", "Put the cube in the bowl"),
}
LABELS = {"T1": "T1 cube then banana", "T2": "T2 pick glasses", "T3": "T3 cube left of bowl",
          "T4": "T4 stack yellow on red", "T5": "T5 mustard in left bin", "T6": "T6 stack 3 cubes",
          "T7": "T7 pick green block", "T8": "T8 mustard right of raisin (RF)", "R1": "R1 cube in bowl"}
PRACTICE = range(30000, 30032)
HELDOUT = range(96000, 96064)

# Wording screens (metric 2): (code, alt-arm suffix, alternative wording, why chosen)
WORDINGS: list[tuple[str, str, str, str]] = []
# Edits (metric 3): filled as experiments finish; see README
EDITS: list[dict] = []

# --------------------------------------------------------------------------------------------- loading
INK, INK2, MUTE = "#1C1C1C", "#4D4D4D", "#858585"
LIGHT, PALE, RULE = "#C6C6C6", "#ECECEC", "#A3A3A3"
TEAL, TEAL_PALE = "#147D92", "#D5E8EC"


def load_runs():
    arms: dict[str, dict[int, bool]] = {}
    meta: dict[str, list[dict]] = {}
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        arm = d["tag"].rsplit("-s", 1)[0]
        rows = {r["slot"]: r["success"] for r in d["rows"] if r.get("ok")}
        arms.setdefault(arm, {}).update(rows)
        meta.setdefault(arm, []).append({k: d.get(k) for k in ("tag", "task", "prompt", "edit", "slots", "envs",
                                                                "n", "completed", "success", "gpu_s", "started",
                                                                "finished")})
    return arms, meta


def rate(arm: dict[int, bool], slots) -> tuple[int, int]:
    got = [arm[s] for s in slots if s in arm]
    return sum(got), len(got)


def paired(ref: dict[int, bool], cand: dict[int, bool], slots=None) -> dict:
    common = sorted(set(ref) & set(cand) & (set(slots) if slots is not None else set(ref)))
    if not common:
        return {}
    c = stats.compare({s: ref[s] for s in common}, {s: cand[s] for s in common})
    t = c.table
    lo, hi = c.diff_ci
    return {"n": len(common), "ref_k": t.both + t.broke, "cand_k": t.both + t.fixed, "fixed": t.fixed,
            "broke": t.broke, "both": t.both, "neither": t.neither, "p": stats.mcnemar_exact(t.fixed, t.broke),
            "diff_pts": 100 * (t.fixed - t.broke) / len(common), "diff_ci95_pts": [100 * lo, 100 * hi],
            "slots": [common[0], common[-1]]}


def wilson_pct(k, n):
    lo, hi = stats.wilson(k, n)
    return [round(100 * lo, 1), round(100 * hi, 1)]


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


def plot_failure_map(rows):
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 0.34 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        p, (lo, hi) = r["k"] / r["n"], r["wilson95"]
        ax.barh(y, 100 * p, height=0.5, color=LIGHT, edgecolor="none", zorder=2)
        ax.plot([lo, hi], [y, y], color=INK2, lw=1.1, zorder=3, solid_capstyle="butt")
        for x in (lo, hi):
            ax.plot([x, x], [y - 0.12, y + 0.12], color=INK2, lw=1.1, zorder=3)
        ax.text(max(hi, 100 * p) + 1.5, y, f"{r['k']}/{r['n']}", va="center", ha="left", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([LABELS[r["code"]] for r in reversed(rows)])
    ax.set_xlim(0, 112)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("success, % of paired starts (bar) with Wilson 95% interval")
    ax.grid(axis="x", color=PALE, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    n = sorted({r["n"] for r in rows})
    ax.set_title(f"Cosmos3 Edge Policy DROID, unedited, default wording (n = {'/'.join(map(str, n))} per task)",
                 fontsize=9.5, color=INK, loc="left")
    _save(fig, "failure_map")
    plt.close(fig)


def plot_wording(rows):
    if not rows:
        return
    plt = _style()
    fig, ax = plt.subplots(figsize=(6.4, 0.62 * len(rows) + 0.9))
    for i, r in enumerate(rows):
        y = len(rows) - 1 - i
        for dy, k, color, lab in ((0.17, r["orig_k"], MUTE, "original"), (-0.17, r["alt_k"], TEAL, "alternative")):
            ax.barh(y + dy, 100 * k / r["n"], height=0.3, color=color, edgecolor="none", zorder=2)
            ax.text(100 * k / r["n"] + 1.5, y + dy, f"{k}/{r['n']} {lab}", va="center", fontsize=8, color=INK)
        ax.text(101, y, "", va="center")
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{LABELS[r['code']]}\n{r['fixed']} fixed / {r['broke']} broke, p = {stats.format_p(r['p'])}"
                        for r in reversed(rows)], fontsize=8.5)
    ax.set_xlim(0, 125)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("success, % of the same paired starts (unedited weights)")
    ax.grid(axis="x", color=PALE, lw=0.8, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.set_title("Wording sensitivity: original vs alternative wording, same starts", fontsize=9.5, color=INK,
                 loc="left")
    _save(fig, "wording")
    plt.close(fig)


def plot_edit(groups, name, title):
    """groups: list of (label, [(arm label, k, n, color)])."""
    if not groups:
        return
    plt = _style()
    nb = sum(len(g[1]) for g in groups)
    fig, ax = plt.subplots(figsize=(6.4, 0.3 * nb + 0.45 * len(groups) + 0.8))
    y, ticks = 0.0, []
    for glabel, bars in reversed(groups):
        for lab, k, n, color in reversed(bars):
            ax.barh(y, 100 * k / n, height=0.24, color=color, edgecolor="none", zorder=2)
            lo, hi = stats.wilson(k, n)
            ax.plot([100 * lo, 100 * hi], [y, y], color=INK2, lw=0.9, zorder=3)
            ax.text(max(100 * hi, 100 * k / n) + 1.5, y, f"{lab}: {k}/{n}", va="center", fontsize=8, color=INK)
            y += 0.32
        ticks.append((y - 0.32 * len(bars) / 2 - 0.16, glabel))
        y += 0.35
    ax.set_yticks([t[0] for t in ticks])
    ax.set_yticklabels([t[1] for t in ticks], fontsize=8.5)
    ax.set_xlim(0, 135)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xlabel("success, % of paired starts (Wilson 95% interval)")
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
    records, summary = [], {"model": MODEL, "arms": {a: {"slots": sorted(v), "k": sum(v.values()), "n": len(v)}
                                                     for a, v in arms.items()}, "runs": meta}
    # metric 1: failure map on practice slots
    fmap = []
    for code, (task, instr) in TASKS.items():
        arm = arms.get(f"{code}-base")
        if not arm:
            continue
        k, n = rate(arm, PRACTICE)
        if n == 0:
            continue
        fmap.append({"code": code, "task": task, "k": k, "n": n, "wilson95": wilson_pct(k, n)})
        records.append({"experiment": "failure_map", "model": MODEL, "task": task, "code": code,
                        "wording_pair": [instr], "n": n, "unedited_k": k, "edited_k": None, "fixed": None,
                        "broke": None, "p": None, "ci": {"wilson95_pct": wilson_pct(k, n)}, "control_k": None,
                        "teacher_k": None, "recovery_pct": None, "transfer": None, "retention": None,
                        "edit_location": None, "values_changed": None,
                        "slots": [min(s for s in arm if s in PRACTICE), max(s for s in arm if s in PRACTICE)],
                        "notes": "unedited policy, task's default instruction, practice slots"})
    summary["failure_map"] = fmap
    # metric 2: wording screens on practice slots
    wrows = []
    for code, suffix, alt, why in WORDINGS:
        base, alt_arm = arms.get(f"{code}-base"), arms.get(f"{code}-{suffix}")
        if not base or not alt_arm:
            continue
        c = paired(base, alt_arm, PRACTICE)
        if not c:
            continue
        wrows.append({"code": code, "alt": alt, "orig_k": c["ref_k"], "alt_k": c["cand_k"], **c})
        records.append({"experiment": "wording", "model": MODEL, "task": TASKS[code][0], "code": code,
                        "wording_pair": [TASKS[code][1], alt], "n": c["n"], "unedited_k": c["ref_k"],
                        "edited_k": None, "fixed": c["fixed"], "broke": c["broke"], "p": c["p"],
                        "ci": {"diff_pts": c["diff_pts"], "bootstrap95_pts": c["diff_ci95_pts"],
                               "wilson95_orig_pct": wilson_pct(c["ref_k"], c["n"]),
                               "wilson95_alt_pct": wilson_pct(c["cand_k"], c["n"])},
                        "control_k": None, "teacher_k": c["cand_k"], "recovery_pct": None, "transfer": None,
                        "retention": None, "edit_location": None, "values_changed": None, "slots": c["slots"],
                        "notes": f"same weights, wording via prompt override; fixed = only the alternative succeeds. {why}"})
    summary["wording"] = wrows
    for e in EDITS:
        rec = e["record"](arms) if callable(e.get("record")) else None
        if rec:
            records.append(rec)
    summary["records"] = len(records)
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "summary.json").write_text(json.dumps(summary, indent=1))
    METRICS.write_text(json.dumps(records, indent=1))
    if fmap:
        plot_failure_map(fmap)
    plot_wording(wrows)
    for r in fmap:
        print(f"{r['code']:3s} {r['task']:28s} {r['k']:3d}/{r['n']:<3d} Wilson95 {r['wilson95']}")
    for r in wrows:
        print(f"{r['code']} wording: {r['orig_k']}/{r['n']} -> {r['alt_k']}/{r['n']}  fixed {r['fixed']} broke "
              f"{r['broke']} p={r['p']:.3g}")


if __name__ == "__main__":
    main()
