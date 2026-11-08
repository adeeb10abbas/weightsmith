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


def load_runs():
    runs = {}
    for f in sorted(RUNS.glob("*.json")):
        d = json.loads(f.read_text())
        runs[d["tag"]] = d
    return runs


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
        recs.append({"experiment": "failure_map", "model": MODEL, "task": f"{tid} {cls}", "wording_pair": [instr, None],
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
        recs.append({"experiment": "wording_sensitivity", "model": MODEL, "task": f"{tid} {base['task']}",
                     "wording_pair": [orig, runs[tag]["prompt"]], "n": c["n"], "unedited_k": c["ref_k"],
                     "edited_k": None, "fixed": c["fixed"], "broke": c["broke"], "p": c["p"],
                     "ci": {"diff_bootstrap95": c["diff_ci95"]}, "control_k": None, "teacher_k": c["cand_k"],
                     "recovery_pct": None, "transfer": None, "retention": None, "edit_location": None,
                     "values_changed": None, "slots": c["slots"],
                     "notes": f"alternative wording given to the unedited model (teacher); fixed = succeeds only "
                              f"under the alternative wording; tags {base['tag']} vs {tag}"})
    METRICS.write_text(json.dumps({"track": "B", "model": MODEL, "records": recs}, indent=1))
    (G / "results/tables.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    if fm:
        plot_failure_map(fm)


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
    ax.set_title("GR00T N1.7 DROID, unedited: success on 32 paired starts (Wilson 95%)", fontsize=10, loc="left")
    ax.grid(axis="y", color="#E8EAED", zorder=0)
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "failure_map.svg")
    fig.savefig(PLOTS / "failure_map.png", dpi=200)
    plt.close(fig)


if __name__ == "__main__":
    main()
