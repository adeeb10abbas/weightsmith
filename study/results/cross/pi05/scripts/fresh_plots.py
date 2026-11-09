"""Track C' plots (fresh pi0.5, ali-ms GPU0, default preset) from the fresh records in ../../metrics_pi05.json.

    python3 fresh_plots.py   # -> ../plots/fresh_failure_map.{svg,png}, fresh_t1_heldout.*, fresh_side_effects.*

Same style as analyze.py (light surface, thin bars, Wilson 95% whiskers, direct k/n labels). PNG at 2x.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
PI05 = HERE.parent
REPO = PI05.parents[3]
sys.path.insert(0, str(REPO))
from weightsmith import stats as S  # noqa: E402

PLOTS = PI05 / "plots"
SURF, INK, INK2, MUTE, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
NEUTRAL, BLUE, LIGHT = "#a19f99", "#2a78d6", "#d3d1cb"
RUN = "C-prime-default"


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": GRID,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK, "svg.fonttype": "none",
                         "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF})


def save(fig, name):
    PLOTS.mkdir(exist_ok=True)
    fig.savefig(PLOTS / f"{name}.svg", bbox_inches="tight")
    fig.savefig(PLOTS / f"{name}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def axes_common(ax, n_rows):
    ax.set_xlim(0, 100)
    ax.set_ylim(n_rows - 0.4, -0.6)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_xticklabels(["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)


def bar(ax, y, k, n, color, h=0.34, label=None):
    lo, hi = S.wilson(k, n)
    ax.barh(y, 100 * k / n, height=h, color=color, label=label, edgecolor=SURF, linewidth=1)
    ax.plot([100 * lo, 100 * hi], [y, y], color=INK2, lw=0.9, solid_capstyle="butt")
    ax.text(max(100 * hi, 100 * k / n) + 1.5, y, f"{k}/{n}", va="center", fontsize=8.2, color=INK)


def main():
    M = [r for r in json.loads((PI05.parent / "metrics_pi05.json").read_text()) if r.get("run") == RUN]
    style()
    cos = {}
    cf = PI05 / "results/fresh/cosmos_same_slots_ali-ms.json"
    if cf.exists():
        cos = json.loads(cf.read_text())

    fm = [r for r in M if r["experiment"] == "failure_map"]
    if fm:
        order = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R1"]
        fm.sort(key=lambda r: order.index(r["code"]))
        fig, ax = plt.subplots(figsize=(6.4, 0.42 * len(fm) + 1.0))
        for i, r in enumerate(fm):
            bar(ax, i - 0.19, r["unedited_k"], r["n"], BLUE, label="π0.5 DROID (fresh)" if i == 0 else None)
            c = cos.get(f"{r['code']}-base-s{r['slots'][0]}", {}).get("slots")
            if c:
                bar(ax, i + 0.19, sum(c.values()), len(c), NEUTRAL,
                    label="Cosmos3 Edge, same starts (Track A)" if i == 0 else None)
        axes_common(ax, len(fm))
        ax.set_yticks(range(len(fm)))
        ax.set_yticklabels([f"{r['code']}  {r['task'].removesuffix('Task')}" for r in fm], fontsize=8.4)
        ax.set_title("Unedited success, each task's default wording, practice starts 30000–30015",
                     fontsize=10, color=INK, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2, frameon=False, fontsize=8)
        fig.text(0.01, -0.06, "ali-ms RTX 3090, RoboLab default render preset, 8 envs per Isaac process; whiskers: "
                 "Wilson 95%.", fontsize=7.4, color=INK2, va="top")
        save(fig, "fresh_failure_map")

    et = [r for r in M if r["experiment"] == "edit_target"]
    for r in et:
        en = r["edit"]["name"]
        arms = [("unedited", r["unedited_k"], r["n"], NEUTRAL), (f"{en} (edit)", r["edited_k"], r["n"], BLUE)]
        c = r.get("control")
        if c:
            arms.append((f"{c['name']} (random control, {c['slots'][0]}–{c['slots'][1]})", c["k"], c["n"],
                         LIGHT))
        t = r.get("teacher")
        if t:
            arms.append((f"unedited, {'explicit' if r['code'] == 'T1' else 'target-first'} wording "
                         f"(teacher, {t['slots'][0]}–{t['slots'][1]})", t["k"], t["n"], MUTE))
        fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(arms) + 1.0))
        for i, (lab, k, n, col) in enumerate(arms):
            bar(ax, i, k, n, col, h=0.5)
        axes_common(ax, len(arms))
        ax.set_yticks(range(len(arms)))
        ax.set_yticklabels([a[0] for a in arms], fontsize=8.4)
        name = {"T1": "T1 cube then banana", "T8": "T8 mustard right of raisin (RF wording)"}[r["code"]]
        ax.set_title(f"{name}, held-out starts {r['slots'][0]}–{r['slots'][1]}, instruction unchanged",
                     fontsize=10, color=INK, loc="left")
        note = (f"Edit vs unedited: {r['fixed']} fixed / {r['broke']} broke, exact McNemar p = {S.format_p(r['p'])}."
                + (f" Edit vs control: {c['edit_vs_control']['edit_only']} / {c['edit_vs_control']['control_only']}, "
                   f"p = {S.format_p(c['edit_vs_control']['p'])}." if c else "")
                + " Default render preset; whiskers: Wilson 95%.")
        fig.text(0.01, -0.02, note, fontsize=7.4, color=INK2, va="top")
        save(fig, f"fresh_{r['code'].lower()}_heldout")

    se = [r for r in M if r["experiment"] == "side_effect"]
    if se:
        fig, ax = plt.subplots(figsize=(6.4, 0.5 * len(se) + 1.1))
        for i, r in enumerate(se):
            bar(ax, i - 0.19, r["unedited_k"], r["n"], NEUTRAL, label="unedited" if i == 0 else None)
            bar(ax, i + 0.19, r["edited_k"], r["n"], BLUE, label="v2mi-f (the T1 edit)" if i == 0 else None)
        axes_common(ax, len(se))
        ax.set_yticks(range(len(se)))
        ax.set_yticklabels([f"{r['code']}  {r['task'].removesuffix('Task')}\n{r['fixed']} fixed / {r['broke']} broke, "
                            f"p = {S.format_p(r['p'])}" for r in se], fontsize=8.0)
        ax.set_title("Side effects of the T1 edit on other tasks, held-out starts, each task's own wording",
                     fontsize=10, color=INK, loc="left")
        ax.legend(loc="upper center", bbox_to_anchor=(0.4, -0.06), ncol=2, frameon=False, fontsize=8)
        save(fig, "fresh_side_effects")


if __name__ == "__main__":
    main()
