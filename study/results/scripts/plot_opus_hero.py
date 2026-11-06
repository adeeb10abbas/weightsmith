#!/usr/bin/env python3
"""Hero figure: success rate without -> with the edit, for the headline comparisons (dumbbell chart)."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STUDY = Path(__file__).resolve().parents[2]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8b8a85", "#e4e3df", "#fcfcfb"
BEFORE, AFTER = "#a3a29c", "#2a78d6"
# (label, sublabel, before k/n, after k/n, worked)
ROWS = [
    ("Misread instruction", "cube then banana · 128 held-out seeds", (27, 128), (106, 128), True),
    ("Same edit, unrelated task", "pick up eyeglasses · 64 held-out seeds", (22, 64), (44, 64), True),
    ("Same edit, spatial instruction", "cube left of the bowl · 64 held-out seeds", (8, 64), (36, 64), True),
    ("Lookalike object", "stack yellow on red · 9 edits, 2 editors, none accepted", (28, 64), (21, 64), False),
    ("Skill never learned", "pick up a flat box · 10 edits", (0, 64), (0, 16), False),
    ("Grasp slips (physics)", "mustard into bin · 1-weight edit", (13, 64), (13, 64), False),
    ("Gripper threshold, 1 weight", "7 tasks pooled", (117, 240), (117, 240), False),
]
plt.rcParams.update({"font.family": "DejaVu Sans"})
fig, ax = plt.subplots(figsize=(8.6, 5.6), facecolor=SURFACE)
ax.set_facecolor(SURFACE)
n = len(ROWS)
for i, (lab, sub, (bk, bn), (ak, an), ok) in enumerate(ROWS):
    y = n - 1 - i
    b, a = 100 * bk / bn, 100 * ak / an
    ax.plot([b, a], [y, y], color=AFTER if ok else MUTED, lw=2.4, solid_capstyle="round", zorder=1)
    ax.scatter([b], [y], s=70, color=BEFORE, edgecolor=SURFACE, linewidth=1.5, zorder=2)
    ax.scatter([a], [y], s=90, color=AFTER if ok else INK2, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(-3, y + 0.13, lab, ha="right", va="center", fontsize=10.5, color=INK, weight="bold")
    ax.text(-3, y - 0.2, sub, ha="right", va="center", fontsize=8, color=INK2)
    lo, hi = sorted([b, a])
    if abs(a - b) < 1:
        ax.text(hi + 3, y, "%d%% → %d%%" % (round(b), round(a)), va="center", fontsize=9.5, color=INK2)
    else:
        ax.text(hi + 3, y, "%d%% → %d%%" % (round(b), round(a)), va="center", fontsize=10.5,
                color=INK if ok else INK2, weight="bold" if ok else "normal")
ax.set_xlim(-2, 112); ax.set_ylim(-0.7, n - 0.3)
ax.set_xticks([0, 25, 50, 75, 100]); ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"], color=INK2, fontsize=9)
ax.set_yticks([])
ax.grid(axis="x", color=GRID, lw=0.7, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(MUTED)
ax.set_xlabel("Task success · light grey: unedited · blue: edit worked · dark grey: edits did not", fontsize=9, color=INK2)
fig.text(0.02, 0.97, "LLM weight editing on a robot policy (π0.5, simulation)", fontsize=13, weight="bold", color=INK, va="top")
fig.text(0.02, 0.915, "It fixed what the robot misread, not what it couldn't do.", fontsize=10.5, color=INK2, va="top")
fig.subplots_adjust(left=0.42, right=0.98, top=0.84, bottom=0.12)
for d in (STUDY / "results/figures", STUDY / "blog/figures"):
    for ext in ("png", "svg"):
        fig.savefig(d / f"opus_hero.{ext}", dpi=220, facecolor=SURFACE)
print("saved")
