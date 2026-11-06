#!/usr/bin/env python3
"""Forest plot of paired success differences (candidate - base) for every closed-loop comparison in
results/opus/comparisons.json, grouped by failure level. 95% intervals: paired bootstrap over episode pairs."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.transforms import blended_transform_factory
import numpy as np

STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "results/opus/comparisons.json"
OUT_DIRS = [STUDY / "results/figures", STUDY / "blog/figures"]

INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8b8a85", "#e4e3df", "#fcfcfb"
CLAUDE = dict(color="#2a78d6", marker="o", name="Weight edit, Claude Opus (this work)")
ASTRA = dict(color="#eb6834", marker="D", name="Weight edit, GPT-6 Astra (Seraj study)")
NONW = dict(color="#1baf7a", marker="s", name="Not a weight edit (prompt or runtime threshold)")
IDENT = dict(color="#8b8a85", marker="^", name="Unchanged weights (noise control)")


def style(r):
    if r["type"] == "weight":
        return CLAUDE if r["editor"] == "Claude Opus" else ASTRA
    return IDENT if r["type"] == "identity" else NONW


def paired_ci(r, B=20000, seed=0):
    n, fx, br = r["n"], r["fixed"], r["broke"]
    both = r["base"] - br
    neither = n - both - fx - br
    assert min(both, neither) >= 0 and both + fx == r["cand"], r
    p = np.array([both, fx, br, neither], float) / n
    draws = np.random.default_rng(seed).multinomial(n, p, size=B)
    d = (draws[:, 1] - draws[:, 2]) / n
    return (fx - br) / n, np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    rows = json.loads(DATA.read_text())["rows"]
    levels = list(dict.fromkeys(r["level"] for r in rows))
    items, y = [], 0.0          # (kind, payload, y)
    for lv in levels:
        items.append(("header", lv, y)); y += 1
        for r in (r for r in rows if r["level"] == lv):
            items.append(("row", r, y)); y += 1
        y += 0.4
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK})
    fig, ax = plt.subplots(figsize=(9.2, 0.25 * y + 1.4), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.axvline(0, color=INK2, lw=1.0, zorder=1)
    right = blended_transform_factory(ax.transAxes, ax.transData)
    ticks, ticklabels, bold = [], [], []
    for kind, p, yy in items:
        ticks.append(yy)
        if kind == "header":
            ticklabels.append(p); bold.append(True); continue
        r = p
        d, lo, hi = paired_ci(r)
        st = style(r)
        ax.plot([100 * lo, 100 * hi], [yy, yy], color=st["color"], lw=2, solid_capstyle="round", zorder=2)
        ax.scatter([100 * d], [yy], s=40, marker=st["marker"], color=st["color"], edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ticklabels.append(f'{r["task"]}  ·  {r["label"]}'); bold.append(False)
        ax.text(1.015, yy, r.get("display") or f'{r["cand"]}/{r["n"]} vs {r["base"]}/{r["n"]}', transform=right, ha="left", va="center",
                color=INK2, fontsize=7.6, clip_on=False)
    ax.text(1.015, -1.1, "edit vs base", transform=right, ha="left", va="center", color=INK2, fontsize=7.6,
            style="italic", clip_on=False)
    ax.set_yticks(ticks)
    ax.set_yticklabels(ticklabels, fontsize=7.9)
    for t, b in zip(ax.get_yticklabels(), bold):
        if b:
            t.set_fontweight("bold"); t.set_fontsize(8.4)
    ax.tick_params(axis="y", length=0, pad=6)
    ax.set_ylim(y - 0.3, -1.6)
    ax.set_xlim(-45, 80)
    ax.set_xticks([-40, -20, 0, 20, 40, 60, 80])
    ax.set_xticklabels(["−40", "−20", "0", "+20", "+40", "+60", "+80"])
    ax.grid(axis="x", color=GRID, lw=0.6, zorder=0)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Success rate, candidate minus unedited base (percentage points; paired 95% bootstrap interval)", fontsize=8)
    handles = [plt.Line2D([], [], color=v["color"], marker=v["marker"], lw=2, ms=6, label=v["name"])
               for v in (CLAUDE, ASTRA, NONW, IDENT)]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False, fontsize=7.8, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    for d in OUT_DIRS:
        d.mkdir(parents=True, exist_ok=True)
        for ext in ("png", "svg"):
            fig.savefig(d / f"opus_forest.{ext}", dpi=200, bbox_inches="tight", facecolor=SURFACE)
    for kind, r, _ in items:
        if kind == "row":
            d, lo, hi = paired_ci(r)
            print(f'{r["task"][:26]:26s} {r["label"][:44]:44s} {100*d:+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]')


if __name__ == "__main__":
    main()
