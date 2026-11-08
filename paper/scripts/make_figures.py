#!/usr/bin/env python3
"""Regenerate every figure and generated table of the paper from repository data.

Inputs (all inside this repository):
  study/results/opus/comparisons.json        paired closed-loop comparisons (n, base, cand, fixed, broke)
  study/results/opus/heldout/*.json          held-out runs (summary: Wilson 95% CIs, exact McNemar p, per-slot outcomes)
  study/blog/failure-levels.md               unedited 16-task survey (Appendix A table) and the transfer-map table
  study/results/opus/ctb/README.md, EVIDENCE.md, failure-levels.md section 5
                                             offline release counts of the edit's construction steps (Fig. 2, typed
                                             below in CONSTRUCTION with their source)
  study/blog/figures/opus_mustard_slip.jpg   wrist-camera strip (made by study/results/scripts/make_mustard_strip.py)
  paper/figures/src/teaser_90002.json, teaser_90002_frames.png
                                             held-out slot 90002: gripper commands, events and wrist frames, built
                                             from the recorded episodes by scripts/build_teaser_data.py
  paper/figures/src/heldout_v2mif_events.json
                                             per-slot event summary of the held-out cube-then-banana run, built by
                                             scripts/summarize_heldout_events.py (cube drops, Fig. 1 slot rule)

Outputs: paper/figures/*.pdf (embedded in the paper), paper/figures/preview/*.png (for inspection),
         paper/generated/*.tex (tables input by main.tex).

Statistics: paired 95% intervals are bootstrap intervals over the four paired outcome types (both succeed, fixed,
broke, both fail), 20,000 multinomial draws with seed 0, as in study/results/scripts/plot_opus_forest.py; the pooled
one-weight gripper row is resampled within each task; p-values are exact two-sided McNemar tests; minimum detectable
effects use (z_0.975 + z_0.8) * sqrt(q / n) for a discordant fraction q.

Usage: python3 paper/scripts/make_figures.py
"""
import glob
import json
import re
from math import comb
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OPUS = ROOT / "study/results/opus"
REPORT = ROOT / "study/blog/failure-levels.md"
PAPER = ROOT / "paper"
OUT = PAPER / "figures"
SRC = OUT / "src"
PREVIEW = OUT / "preview"
GEN = PAPER / "generated"

# ---- visual identity: neutral greys, one accent for the edit, a muted red only for harm -------------------------
INK, INK2, MUTE = "#1C1C1C", "#4D4D4D", "#858585"
LIGHT, PALE, RULE = "#C6C6C6", "#ECECEC", "#A3A3A3"
TEAL, TEAL_PALE = "#147D92", "#D5E8EC"
RED = "#B1504A"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Arial Unicode MS"],
    "font.size": 6.8,
    "axes.titlesize": 7.2,
    "axes.labelsize": 6.8,
    "axes.edgecolor": RULE,
    "axes.linewidth": 0.5,
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.0,
    "ytick.major.size": 2.0,
    "mathtext.fontset": "custom",
    "mathtext.rm": "Arial",
    "mathtext.it": "Arial:italic",
    "mathtext.bf": "Arial:bold",
    "mathtext.fallback": "stixsans",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
})
TEXTW = 5.5  # \textwidth of the CoRL style, inches: figures are drawn at their printed size
FS, FS_SMALL, FS_TITLE = 6.8, 6.1, 7.2


def save(fig, name, pad=0.02):
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(PREVIEW / f"{name}.png", bbox_inches="tight", pad_inches=pad, dpi=250)
    plt.close(fig)
    print("wrote", name)


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


# ---- data ------------------------------------------------------------------------------------------------------
ROWS = json.loads((OPUS / "comparisons.json").read_text())["rows"]
HELDOUT, HALVES = {}, {}
for f in sorted(glob.glob(str(OPUS / "heldout/*.json"))):
    js = json.loads(Path(f).read_text())
    s = js["summary"]
    per = list(js["per_slot"].values())  # [unedited success, edited success] per held-out slot
    assert len(per) == s["paired_n"] and sum(b for b, _ in per) == s["base_success"]
    assert sum(c for _, c in per) == s["cand_success"]
    assert sum(1 for b, c in per if c and not b) == s["fixed"] and sum(1 for b, c in per if b and not c) == s["broke"]
    HELDOUT[(s["task"], s["cand"])] = s
    mid = s["slots"][0] + s["paired_n"] // 2  # run order alternates by half
    halves = []
    for sel in (lambda k: int(k) < mid, lambda k: int(k) >= mid):
        h = [v for k, v in js["per_slot"].items() if sel(k)]
        halves.append((len(h), sum(b for b, _ in h), sum(c for _, c in h), sum(1 for b, c in h if c and not b),
                       sum(1 for b, c in h if b and not c)))
    HALVES[(s["task"], s["cand"])] = halves


def md_table(heading_prefix, ncols):
    """Rows of the first markdown table after a line that starts with heading_prefix (header row dropped)."""
    lines = REPORT.read_text().splitlines()
    start = next(i for i, l in enumerate(lines) if l.startswith(heading_prefix))
    out, seen = [], False
    for l in lines[start + 1:]:
        if l.startswith("|"):
            seen = True
            cells = [c.strip() for c in l.strip().strip("|").split("|")]
            if len(cells) == ncols and not set(cells[1]) <= set("-: "):
                out.append(cells)
        elif seen:
            break
    return out[1:]


def frac(s):
    m = re.fullmatch(r"\**(\d+)/(\d+)\**", s.replace(" ", ""))
    assert m, s
    return int(m.group(1)), int(m.group(2))


SURVEY = {t: (*frac(s), cls.replace("**", "")) for t, s, cls in md_table("## Appendix A.", 3)}
assert len(SURVEY) == 16 and sum(v[0] for v in SURVEY.values()) == 97, SURVEY
TRANSFER_MAP = {r[0].replace("**", ""): r for r in md_table("**A transfer map.**", 4)}
FLATBOX_TX = TRANSFER_MAP["Pick up the flat green box"]  # distilled edit on the flat box: 0/32 vs 0/32 (not in json)
assert frac(FLATBOX_TX[1]) == (0, 32) and frac(FLATBOX_TX[2]) == (0, 32)


def find(task, label_start):
    """The one row of `task` whose label equals label_start, else the one whose label starts with it."""
    hits = [r for r in ROWS if r["task"] == task and r["label"] == label_start]
    hits = hits or [r for r in ROWS if r["task"] == task and r["label"].startswith(label_start)]
    assert len(hits) == 1, (task, label_start, hits)
    return hits[0]


def cells(r):
    both = r["base"] - r["broke"]
    neither = r["n"] - both - r["fixed"] - r["broke"]
    assert min(both, neither) >= 0 and both + r["fixed"] == r["cand"], r
    return np.array([both, r["fixed"], r["broke"], neither], float)


def paired_ci(r, B=20000, seed=0):
    """Paired bootstrap of the success difference (as in plot_opus_forest.py)."""
    p = cells(r) / r["n"]
    draws = np.random.default_rng(seed).multinomial(r["n"], p, size=B)
    d = (draws[:, 1] - draws[:, 2]) / r["n"]
    return (r["fixed"] - r["broke"]) / r["n"], np.percentile(d, 2.5), np.percentile(d, 97.5)


def pooled_ci(rows, B=20000, seed=0):
    """Paired bootstrap stratified by row (task), for one edit evaluated on several tasks."""
    rng = np.random.default_rng(seed)
    N = sum(r["n"] for r in rows)
    diff = np.zeros(B)
    for r in rows:
        draws = rng.multinomial(r["n"], cells(r) / r["n"], size=B)
        diff += draws[:, 1] - draws[:, 2]
    d = sum(r["fixed"] - r["broke"] for r in rows) / N
    return d, np.percentile(diff / N, 2.5), np.percentile(diff / N, 97.5)


def mcnemar(f, b):
    m = f + b
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(f, b) + 1)) / 2 ** m)


def mde(n, q, z=1.959964 + 0.841621):
    """Normal-approximation minimum detectable paired difference (80% power, two-sided alpha 0.05)."""
    return z * np.sqrt(q / n)


GRIP = [r for r in ROWS if r["label"].startswith("gthr04")]
GRIP_POOLED = dict(n=sum(r["n"] for r in GRIP), base=sum(r["base"] for r in GRIP), cand=sum(r["cand"] for r in GRIP),
                   fixed=sum(r["fixed"] for r in GRIP), broke=sum(r["broke"] for r in GRIP))
assert (GRIP_POOLED["n"], GRIP_POOLED["base"], GRIP_POOLED["cand"]) == (240, 117, 117)
CTB = HELDOUT[("RubiksCubeThenBananaTask", "v2mi-f")]
CTB_MIN = HELDOUT[("RubiksCubeThenBananaTask", "abl-imgl8")]
EYE = HELDOUT[("PickGlassesTask", "ctb-v2mi")]
LEFT = HELDOUT[("RubiksCubeLeftOfBowlTask", "ctb-v2mi")]
for (task, lab), s in [(("Cube then banana", "v2mi-f: binding edit, HELD"), CTB),
                       (("Cube then banana", "minimal edit"), CTB_MIN),
                       (("Pick up eyeglasses", "binding edit, HELD"), EYE),
                       (("Cube left of bowl", "binding edit, HELD"), LEFT)]:
    r = find(task, lab)  # the json rows and the held-out files must agree
    assert (r["n"], r["base"], r["cand"], r["fixed"], r["broke"]) == (
        s["paired_n"], s["base_success"], s["cand_success"], s["fixed"], s["broke"]), (r, s)


def wilson(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


for s in (CTB, CTB_MIN, EYE, LEFT):  # Wilson intervals reproduce the stored ones
    for k, key in ((s["base_success"], "base_wilson95"), (s["cand_success"], "cand_wilson95")):
        assert np.allclose(wilson(k, s["paired_n"]), s[key], atol=6e-4), (s["task"], key)

# Event-log summary of the held-out run: successes must match the held-out file slot by slot.
EVENTS = json.loads((SRC / "heldout_v2mif_events.json").read_text())["slots"]
_per = json.loads((OPUS / "heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json").read_text())
assert {k: [v["unedited"]["success"], v["edited"]["success"]] for k, v in EVENTS.items()} == _per["per_slot"]
FIXED = [k for k, v in EVENTS.items() if not v["unedited"]["success"] and v["edited"]["success"]]
FIXED_UNEDITED_DROP = sum(bool(EVENTS[k]["unedited"]["cube_drops_outside_bowl"]) for k in FIXED)
FIXED_EDITED_DROP = sum(bool(EVENTS[k]["edited"]["cube_drops_outside_bowl"]) for k in FIXED)
assert (len(FIXED), FIXED_UNEDITED_DROP, FIXED_EDITED_DROP) == (81, 75, 15)


def _drops_cube_first(v):
    u = v["unedited"]
    d = u["cube_drops_outside_bowl"]
    return bool(d) and (u["first_banana_grasp"] is None or d[0] < u["first_banana_grasp"])


# Fig. 1 shows the lowest-numbered fixed slot whose unedited run dropped the cube before touching the banana.
assert min(int(k) for k in FIXED if _drops_cube_first(EVENTS[k])) == 90002


# ---- Fig. 1: one held-out start, unedited vs edited -----------------------------------------------------------------
def spans(steps, values, thr=0.5):
    """Contiguous [start, end) step spans where the commanded gripper is closed (> thr)."""
    out, start = [], None
    for s, v in zip(steps, values):
        if v > thr and start is None:
            start = s
        if v <= thr and start is not None:
            out.append((start, s)); start = None
    if start is not None:
        out.append((start, steps[-1] + 1))
    return out


def fig_teaser():
    meta = json.loads((SRC / "teaser_90002.json").read_text())
    sheet = np.asarray(Image.open(SRC / "teaser_90002_frames.png").convert("RGB"))
    h, w = meta["frame_size"]
    gap = meta["frame_gap"]
    moments = meta["storyboard_steps"]
    U, E = meta["arms"]["unedited"], meta["arms"]["edited"]
    assert not U["success"] and E["success"] and E["final_step"] == 257

    def ev(A, name, obj=None, what=None):
        hits = [e["step"] for e in A["events"] if e["name"] == name and (obj is None or obj in e["info"])
                and (what is None or what in e["info"])]
        return min(hits)

    grab_u, drop_u = ev(U, "OBJECT_GRABBED_SUCCESS", "rubiks"), ev(U, "TARGET_OBJECT_DROPPED")
    banana_u = ev(U, "WRONG_OBJECT_GRABBED_FAILURE", "banana")
    banana_off_u = ev(U, "OK", "banana")
    cube_bowl_u = ev(U, "OBJECT_IN_CONTAINER_SUCCESS")
    grab_e, banana_e = ev(E, "OBJECT_GRABBED_SUCCESS", "rubiks"), ev(E, "OBJECT_GRABBED_SUCCESS", "banana")
    bowl_e = sorted(e["step"] for e in E["events"] if e["name"] == "OBJECT_IN_CONTAINER_SUCCESS")
    assert (grab_u, drop_u, banana_u, banana_off_u, cube_bowl_u) == (73, 76, 148, 196, 351)
    assert (grab_e, bowl_e, banana_e) == (78, [119, 255], 198)

    W, H = TEXTW, 3.04
    fig = plt.figure(figsize=(W, H))
    lab_w = 0.52                     # inches reserved for row labels
    fw = (W - lab_w - 3 * 0.05) / 4  # frame width
    fh = fw * h / w
    top = H - 0.34
    notes = {(0, 0): ("cube in the gripper", INK2), (0, 1): ("cube dropped", RED),
             (0, 2): ("reaching for the banana", INK2), (0, 3): ("banana in the bowl", INK2),
             (1, 0): ("cube in the gripper", INK2), (1, 1): ("cube held", TEAL),
             (1, 2): ("cube in the bowl", TEAL), (1, 3): ("banana in: success", TEAL)}
    fig.text(0.0, (H - 0.015) / H, "Held-out start 90002.  Instruction, unchanged: "
             "“Put the cube then the banana in the bowl”", ha="left", va="top", fontsize=FS, color=INK)
    for r, (name, color) in enumerate((("Unedited", MUTE), ("Edited", TEAL))):
        y = top - (r + 1) * fh - r * 0.05
        fig.text(0.0, (y + fh / 2) / H, name, ha="left", va="center", fontsize=FS, weight="bold", color=color)
        for c in range(4):
            x = lab_w + c * (fw + 0.05)
            ax = fig.add_axes([x / W, y / H, fw / W, fh / H])
            ax.imshow(sheet[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w],
                      interpolation="lanczos")
            bare(ax)
            note, ncol = notes[(r, c)]
            ax.text(0.03, 0.06, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_SMALL - 0.3,
                    color="white", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.12", fc=ncol, ec="none",
                                             alpha=0.88))
            if r == 0:
                fig.text(x / W, (top + 0.04) / H, f"{c + 1}", ha="left", va="bottom", fontsize=FS, weight="bold",
                         color=INK)
                fig.text((x + 0.1) / W, (top + 0.04) / H, f"step {moments[c]}", ha="left", va="bottom",
                         fontsize=FS_SMALL, color=INK2)

    # timeline: commanded gripper (closed = filled) and logged events, both rollouts
    t_left, t_right, t_bot, t_h = lab_w, 4.02, 0.3, 1.0
    ax = fig.add_axes([t_left / W, t_bot / H, (t_right - t_left) / W, t_h / H])
    xmax = 300
    yU, yE, bh = 1.0, 0.0, 0.3
    ax.add_patch(Rectangle((75, -0.5), 15, 2.12, fc=TEAL_PALE, ec="none", alpha=0.55, zorder=0))
    for y, A, color in ((yU, U, MUTE), (yE, E, TEAL)):
        ax.add_patch(Rectangle((0, y - bh / 2), xmax, bh, fc=PALE, ec="none", zorder=0.6))
        for a, b in spans(A["cmd_steps"], A["gripper_cmd"]):
            a, b = max(a, 0), min(b, xmax)
            if a < b:
                ax.add_patch(Rectangle((a, y - bh / 2), b - a, bh, fc=color, ec="none", zorder=1))
    for c, s in enumerate(moments):
        ax.plot([s, s], [-0.42, 1.66], color=RULE, lw=0.45, ls=(0, (1, 1.6)), zorder=0.8)
        ax.text(s, 1.7, f"{c + 1}", ha="center", va="bottom", fontsize=FS_SMALL, weight="bold", color=INK)
    inside = dict(ha="center", va="center", fontsize=FS_SMALL - 0.5, color="white", zorder=4)
    ax.text((135 + 195) / 2, yU, "banana", **inside)
    ax.text((75 + 118) / 2, yE, "cube", **inside)
    ax.text((191 + 254) / 2, yE, "banana", **inside)
    # unedited: the cube is released three steps after the grasp; the banana goes in first
    ax.plot([drop_u], [yU], marker="x", ms=4.4, mew=1.2, color=RED, zorder=5)
    ax.text(drop_u + 5, yU + 0.2, f"cube released at step {drop_u}, {drop_u - grab_u} steps after the grasp",
            ha="left", va="bottom", fontsize=FS_SMALL, color=RED, zorder=5,
            bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
    ax.text(136, yU - 0.27, f"banana released at {banana_off_u}; cube placed only at {cube_bowl_u}; fails",
            ha="left", va="top", fontsize=FS_SMALL, color=INK2)
    # edited: holds the cube until it is in the bowl, then the banana
    for s in bowl_e:
        ax.plot([s], [yE], marker="o", ms=3.4, mfc="white", mec=TEAL, mew=0.9, zorder=5)
    ax.text(grab_e - 1, yE - 0.27, f"cube held from {grab_e} until it is in the bowl ({bowl_e[0]})", ha="left",
            va="top", fontsize=FS_SMALL, color=TEAL)
    ax.text(bowl_e[1] + 6, yE, f"success at {E['final_step']}", ha="left", va="center", fontsize=FS_SMALL,
            color=TEAL, weight="bold", zorder=5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.75, 1.95)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("Control step at 15 Hz (filled: gripper commanded closed)", fontsize=FS_SMALL, color=INK2,
                  labelpad=2)
    to_fig = lambda yy: (t_bot + t_h * (yy + 0.75) / 2.7) / H
    fig.text(0.0, to_fig(yU), "Unedited", ha="left", va="center", fontsize=FS_SMALL, color=MUTE, weight="bold")
    fig.text(0.0, to_fig(yE), "Edited", ha="left", va="center", fontsize=FS_SMALL, color=TEAL, weight="bold")

    # inset: held-out success with Wilson 95% intervals
    i_left, i_w = 4.42, 1.0
    ia = fig.add_axes([i_left / W, (t_bot + 0.1) / H, i_w / W, 0.62 / H])
    for y, (k, n, lo, hi, color) in enumerate((
            (CTB["base_success"], CTB["paired_n"], *CTB["base_wilson95"], MUTE),
            (CTB["cand_success"], CTB["paired_n"], *CTB["cand_wilson95"], TEAL))):
        ia.plot([100 * lo, 100 * hi], [y, y], color=color, lw=2.2, solid_capstyle="butt", alpha=0.45)
        ia.plot([100 * k / n], [y], marker="o", ms=3.6, color=color, mec="white", mew=0.5)
        ia.text(100 * k / n, y + 0.28, f"{k}/{n}", ha="center", va="bottom", fontsize=FS_SMALL, color=color,
                weight="bold" if color == TEAL else "normal")
    ia.set_xlim(0, 100)
    ia.set_ylim(-0.45, 1.75)
    ia.set_yticks([])
    ia.set_xticks([0, 50, 100])
    ia.set_xticklabels(["0", "50", "100%"])
    for s in ("top", "right", "left"):
        ia.spines[s].set_visible(False)
    fig.text(i_left / W, (t_bot + 0.86) / H, "Success, 128 held-out starts", ha="left", va="bottom",
             fontsize=FS_SMALL, color=INK, weight="bold")
    fig.text(i_left / W, (t_bot + 0.76) / H, "with Wilson 95% intervals", ha="left", va="bottom",
             fontsize=FS_SMALL - 0.4, color=MUTE)
    save(fig, "fig1_teaser")


# ---- Fig. 2: what it takes to fix the decision ----------------------------------------------------------------------
# Offline replays of the 21 recorded premature-release decisions (cube then banana). Sources:
# study/results/opus/ctb/README.md and EVIDENCE.md (addendum), study/blog/failure-levels.md sections 5.3-5.5.
# (label, size note, still releasing, decisions checked, closed loop note, kind)
REWORDINGS = [  # replayed premature-release decisions that still release, of 21 (failure-levels.md section 5.2)
    ("“Put the cube then the banana in the bowl” (original)", 21, "word", ""),
    ("“Put the rubiks cube then the banana in the bowl”", 21, "word", ""),
    ("“Put the banana then the cube in the bowl”", 3, "word", ""),
    ("“Put the cube in the bowl then the banana”", 3, "word", ""),
    ("“Put the cube in the bowl then put the banana\nin the bowl” (explicit)", 2, "word",
     "22/32 (3/32), unedited weights"),
    ("“Put the cube in the bowl”", 1, "word", ""),
]
CONSTRUCTION = [  # (label, factored size, still releasing, decisions checked, closed loop, kind)
    ("Steer along a correlated cube–banana axis", "", 21, 21, "", "fail"),
    ("Copy the mean shift at the state tokens", "", 6, 6, "", "fail"),
    ("Copy per-position shifts at 40 prompt/state slots", None, None, None, "", "group"),
    ("strength ×0.5", "12.5M", 15, 21, "", "fail"),
    ("strength ×1", "12.5M", 13, 21, "", "fail"),
    ("strength ×2", "12.5M", 8, 21, "", "fail"),
    ("+ 32 image cells: the distilled edit", "22.6M", 0, 21, "25/32 (3/32) · 106/128 (27/128)", "edit"),
]
VARIANTS = [
    ("Image cells only", "10.0M", 0, 21, "", "edit"),
    ("All 72 keys, layers 8–16 only", "11.9M", 0, 21, "", "edit"),
    ("Half strength", "22.6M", 2, 21, "", "edit"),
    ("Image cells, layers 8–16 (0.16%)", "5.3M", 2, 21, "25/32 (3/32) · 102/128 (19/128)", "edit"),
    ("Image cells + preservation keys, 5 tasks", "", 1, 21, "24/32 (3/32)", "edit"),
]


def fig_construction():
    rows = [("head", "a", "Rewording the instruction, weights unchanged")]
    rows += [("row", lab, "", k, 21, closed, kind) for lab, k, kind, closed in REWORDINGS]
    rows += [("head", "b", "Editing the weights, original instruction")]
    rows += [("row", *r) for r in CONSTRUCTION]
    rows += [("head", "", "Variants of the distilled edit")]
    rows += [("row", *r) for r in VARIANTS]
    heights = [1.45 if kind == "row" and "\n" in r[0] else 1.0 for kind, *r in rows]
    pitch = 0.12
    n_lines = sum(heights) + 0.45 * 2 + 0.4
    body = pitch * n_lines
    H = body + 0.36
    W = TEXTW
    fig = plt.figure(figsize=(W, H))
    lab_l, size_r, bar_l, bar_w, cl_l = 0.13, 2.6, 2.66, 1.22, 4.0
    ax = fig.add_axes([bar_l / W, 0.31 / H, bar_w / W, body / H])
    tr_fig = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    y = 0.0
    path, first = [], True
    for (kind, *r), hgt in zip(rows, heights):
        if kind == "head":
            letter, title = r
            y += 0.0 if first else 0.45
            if letter:
                ax.text(0.0, y, letter, ha="left", va="center", fontsize=FS_TITLE, weight="bold", color=INK,
                        transform=tr_fig)
            ax.text(lab_l / W, y, title, ha="left", va="center", fontsize=FS_SMALL, weight="bold", color=INK,
                    transform=tr_fig)
            if first:
                ax.text(0, y, "replayed releases that persist", ha="left", va="center", fontsize=FS_SMALL - 0.3,
                        color=MUTE)
                ax.text(cl_l / W, y, "closed loop: edited (unedited)", ha="left", va="center",
                        fontsize=FS_SMALL - 0.3, color=MUTE, transform=tr_fig)
            if letter == "b":
                ax.text(size_r / W, y, "size", ha="right", va="center", fontsize=FS_SMALL - 0.3, color=MUTE,
                        transform=tr_fig)
            first = False
            y += 1
            continue
        label, size, k, n, closed, cls = r
        y += (hgt - 1) / 2
        indent = 0.13 if label.startswith("strength") else 0.0
        ax.text((lab_l + 0.08 + indent) / W, y, label, ha="left", va="center", fontsize=FS_SMALL,
                color=INK, linespacing=1.05, transform=tr_fig)
        if cls != "group":
            share = 100 * k / n
            ax.add_patch(Rectangle((0, y - 0.3), 100, 0.6, fc=PALE, ec="none", zorder=0))
            if share > 0:
                ax.add_patch(Rectangle((0, y - 0.3), share, 0.6, fc=TEAL_PALE if cls == "edit" else LIGHT,
                                       ec=TEAL if cls == "edit" else "none", lw=0.6, zorder=1))
            txt = f"{k}/{n}" + (" checked" if n != 21 else "")
            inside = share > 70
            ax.text(share - 2 if inside else share + 2.5, y, txt, ha="right" if inside else "left", va="center",
                    fontsize=FS_SMALL, color=TEAL if cls == "edit" else INK2, weight="bold" if k == 0 else "normal",
                    zorder=3)
            if size:
                ax.text(size_r / W, y, size, ha="right", va="center", fontsize=FS_SMALL - 0.3, color=MUTE,
                        transform=tr_fig)
            if closed:
                ax.text(cl_l / W, y, closed, ha="left", va="center", fontsize=FS_SMALL,
                        color=TEAL if cls == "edit" else INK2, transform=tr_fig)
            if cls == "fail" or (cls == "edit" and len(path) and path[-1][2] == "fail"):
                path.append((y, share, cls))
        y += 1 + (hgt - 1) / 2
    for (y0, s0, _), (y1, s1, _) in zip(path[:-1], path[1:]):
        ax.plot([s0, s0, s1], [y0 + 0.3, y1 - 0.3, y1 - 0.3], color=INK2, lw=0.6, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.5, -0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100%"])
    for s_ in ("top", "right", "left"):
        ax.spines[s_].set_visible(False)
    save(fig, "fig2_construction")


# ---- Fig. 3: paired outcomes --------------------------------------------------------------------------------------
def ho_row(s, **kw):
    return dict(n=s["paired_n"], base=s["base_success"], cand=s["cand_success"], fixed=s["fixed"],
                broke=s["broke"], **kw)


def outcome_rows(full=False):
    """Groups of (label, comparison, kind) for the paired-outcome figure."""
    if full:
        groups = []
        for lv in LEVEL_ORDER:
            with_task = lv in ("Binding edit on other tasks", "Gripper commitment (all tasks)",
                               "Unchanged weights (noise controls)")
            items = [(readable(r, with_task), r, "held" if r["type"] == "heldout" else
                      "noedit" if r["type"] in ("identity", "prompt", "runtime") else "") for r in ROWS
                     if r["level"] == lv]
            if lv == "Gripper commitment (all tasks)":
                items.append(("Seven tasks pooled: gripper bias (1 value)", GRIP_POOLED, ""))
            groups.append((LEVEL_NAMES[lv], items))
        return groups
    return [
        ("Distilled edit, cube then banana", [
            ("Explicit wording, no weight change", find("Cube then banana", "Explicit two-step wording (no"), "noedit"),
            ("Explicit wording, held-out seeds", find("Cube then banana", "Explicit two-step wording, HELD-OUT"), "noedit"),
            ("Practice seeds", find("Cube then banana", "v2mi-f: binding edit, practice"), ""),
            ("Held-out seeds", ho_row(CTB), "held"),
            ("0.16% variant, held-out seeds", ho_row(CTB_MIN), "held"),
            ("Random directions, same size, held-out", find("Cube then banana", "random directions"), ""),
        ]),
        ("Same edit, other tasks", [
            ("Eyeglasses, held-out seeds", ho_row(EYE), "held"),
            ("Cube left of bowl, held-out seeds", ho_row(LEFT), "held"),
            ("Cube right of bowl", find("Cube right of bowl", "binding edit"), ""),
            ("Butter on raisin box", find("Butter box onto raisin box", "binding edit (vs first"), ""),
        ]),
        ("Perception: stack yellow on red", [
            ("Claude, attention gain", find("Stack yellow on red", "v1a1"), ""),
            ("Claude, key/value gain", find("Stack yellow on red", "dkv2"), ""),
            ("Claude, key/value + head rewire", find("Stack yellow on red", "combo1: dkv2"), ""),
            ("Astra, 4,412 values", find("Stack yellow on red", "Astra A1 p1"), ""),
            ("Astra, 1,543 values", find("Stack yellow on red", "Astra A2 p1"), ""),
            ("Astra, 791 values", find("Stack yellow on red", "Astra A2 p3"), ""),
            ("Astra, 6,225 values", find("Stack yellow on red", "Astra fixed16 p1"), ""),
            ("Claude, Astra protocol 1", find("Stack yellow on red", "Claude, Astra protocol p1"), ""),
            ("Claude, Astra protocol 2", find("Stack yellow on red", "Claude, Astra protocol p2"), ""),
        ]),
        ("Competence, contact, gripper", [
            ("Mustard, one weight", find("Mustard into left bin", "gthr04"), ""),
            ("Seven tasks, one weight", GRIP_POOLED, ""),
            ("Stacking, runtime threshold", find("Stack yellow on red", "Astra study: runtime"), "noedit"),
        ]),
        ("Unchanged weights, rerun", [
            ("Stacking, rerun 1", find("Stack yellow on red", "Astra study: original identity"), "noedit"),
            ("Stacking, rerun 2", find("Stack yellow on red", "Astra study: extra unchanged"), "noedit"),
            ("Stacking, rerun 3", find("Stack yellow on red", "Astra study: fixed16 identity"), "noedit"),
            ("Butter, rerun", find("Butter box onto raisin box", "this work: unedited"), "noedit"),
        ]),
    ]


SEG = [("broke", RED, "white"), ("both succeed", "#7D7D7D", "white"), ("both fail", "#DCDCDC", INK2),
       ("fixed", TEAL, "white")]


def draw_outcomes(groups, name, pitch=0.092, label_w=1.72):
    """100% bars of the four paired outcome types. Left of the dark/light boundary: pairs the unedited policy won
    (broke, both succeed); right of it: pairs it lost (both fail, fixed)."""
    n_rows = sum(len(g[1]) for g in groups)
    W = TEXTW
    n_lines = n_rows + len(groups) * 1.0 + (len(groups) - 1) * 0.35 + 2.6
    body = pitch * n_lines
    H = body + 0.3
    fig = plt.figure(figsize=(W, H))
    bar_l, bar_r = label_w + 0.06, W - 0.66
    ax = fig.add_axes([bar_l / W, 0.2 / H, (bar_r - bar_l) / W, body / H])
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    tr_fig = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    bw = bar_r - bar_l
    c_pairs, c_fixed, c_broke = 1 + 0.06 / bw, 1 + 0.27 / bw, 1 + 0.47 / bw  # right-hand columns, set in inches
    # key: one bar with the four pair types named in place, and what each side of the split means
    yk = 0.0
    kx = [0.0, 0.17, 0.42, 0.75, 1.0]
    for (seg, col, tcol), a, b in zip(SEG, kx[:-1], kx[1:]):
        ax.add_patch(Rectangle((a, yk - 0.38), b - a, 0.76, fc=col, ec="white", lw=0.5))
        ax.text((a + b) / 2, yk, seg, ha="center", va="center", fontsize=FS_SMALL - 0.3, color=tcol)
    for a, b, txt in ((kx[0], kx[2], "unedited succeeded"), (kx[2], kx[4], "unedited failed")):
        ax.plot([a + 0.004, a + 0.004, b - 0.004, b - 0.004], [yk - 0.55, yk - 0.75, yk - 0.75, yk - 0.55],
                color=MUTE, lw=0.5)
        ax.text((a + b) / 2, yk - 0.82, txt, ha="center", va="bottom", fontsize=FS_SMALL - 0.4, color=MUTE)
    ax.text(-0.012, yk, "Each pair: one start, unedited vs edited", transform=tr, ha="right", va="center",
            fontsize=FS_SMALL - 0.3, color=MUTE)
    for x, txt, col in ((c_pairs, "pairs", MUTE), (c_fixed, "fixed", TEAL), (c_broke, "broke", RED)):
        ax.text(x, yk, txt, ha="left", va="center", fontsize=FS_SMALL - 0.3, color=col, transform=tr)
    y = 1.6
    for gi, (gname, items) in enumerate(groups):
        y += 0.35 if gi else 0.0
        ax.text(0.0, y, gname, transform=tr_fig, ha="left", va="center", fontsize=FS_SMALL, weight="bold",
                color=INK)
        y += 1.0
        for label, r, kind in items:
            both = r["base"] - r["broke"]
            neither = r["n"] - both - r["fixed"] - r["broke"]
            x = 0.0
            for (seg, col, _), k in zip(SEG, [r["broke"], both, neither, r["fixed"]]):
                wdt = k / r["n"]
                if wdt > 0:
                    ax.add_patch(Rectangle((x, y - 0.36), wdt, 0.72, fc=col, ec="white", lw=0.35))
                x += wdt
            held = kind == "held"
            ax.text(-0.012, y, label, ha="right", va="center", fontsize=FS_SMALL, transform=tr,
                    color=TEAL if held else INK, weight="bold" if held else "normal",
                    style="italic" if kind == "noedit" else "normal")
            ax.text(c_pairs, y, f"{r['n']}", ha="left", va="center", fontsize=FS_SMALL, color=MUTE, transform=tr)
            ax.text(c_fixed, y, f"+{r['fixed']}" if r["fixed"] else "0", ha="left", va="center", fontsize=FS_SMALL,
                    color=TEAL if r["fixed"] else MUTE, transform=tr)
            ax.text(c_broke, y, f"−{r['broke']}" if r["broke"] else "0", ha="left", va="center",
                    fontsize=FS_SMALL, color=RED if r["broke"] else MUTE, transform=tr)
            y += 1
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.45, -1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "", "50", "", "100% of pairs"])
    ax.set_yticks([])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    save(fig, name)


def fig_outcomes():
    draw_outcomes(outcome_rows(), "fig3_outcomes")


def fig_outcomes_full():
    draw_outcomes(outcome_rows(full=True), "figA_outcomes_full", pitch=0.108, label_w=2.25)


# ---- Fig. 5: detectability map ----------------------------------------------------------------------------------------
def fig_detectability():
    W, H = TEXTW, 1.8
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([0.075, 0.19, 0.915, 0.78])
    ns = np.geomspace(12, 330, 300)
    lo, mid, hi = (100 * mde(ns, q) for q in (0.2, 0.3, 0.4))
    ax.fill_between(ns, 0, mid, color=PALE, lw=0, zorder=0)
    ax.fill_between(ns, lo, hi, color="#DCDCDC", lw=0, zorder=0)
    ax.plot(ns, mid, color=MUTE, lw=0.7, zorder=1)
    pts = []
    for i, r in enumerate(ROWS):
        if r["label"].startswith("10 edits"):
            continue
        d = 100 * (r["cand"] - r["base"]) / r["n"]
        if r["type"] in ("identity", "prompt", "runtime"):
            kind = "noedit"
        elif r["label"].startswith("random directions"):  # same-size control: an edit, but not the distilled one
            kind = "other"
        elif r["level"] in ("Language binding", "Binding edit on other tasks"):
            kind = "edit"
        else:
            kind = "other"
        pts.append((r, r["n"] * (1 + 0.035 * ((i % 5) - 2)), d, kind))  # small fixed offsets separate markers
    pts.append((GRIP_POOLED, 240.0, 0.0, "other"))
    for r, x, d, kind in pts:
        held = r.get("type") == "heldout"
        mk = "^" if d > 0 else ("v" if d < 0 else "o")
        if kind == "edit":
            fc, ec, ms = (TEAL, TEAL, 5.2) if held else ("white", TEAL, 3.9)
        elif kind == "noedit":
            fc, ec, ms = ("white", MUTE, 3.7)
        else:
            fc, ec, ms = (INK2, INK2, 3.4)
        ax.plot([x], [abs(d)], marker=mk, ms=ms, mfc=fc, mec=ec, mew=0.8, ls="", zorder=3 + held)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.set_xticklabels(["16", "32", "64", "128", "256"])
    ax.minorticks_off()
    ax.set_xlim(12, 330)
    ax.set_ylim(0, 72)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Paired episodes per arm", fontsize=FS_SMALL, color=INK2, labelpad=2)
    ax.set_ylabel("|Difference| (points)", fontsize=FS_SMALL, color=INK2, labelpad=2)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    # the detectability curve, labelled along the upper edge of its band
    xs = 108.0
    ys = 100 * mde(xs, 0.4)
    p0 = ax.transData.transform((xs, ys))
    p1 = ax.transData.transform((xs * 1.3, 100 * mde(xs * 1.3, 0.4)))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(xs, ys + 2.6, "smallest detectable change, 80% power", rotation=ang, rotation_mode="anchor",
            ha="left", va="bottom", fontsize=FS_SMALL - 0.4, color=INK2)
    lab = dict(fontsize=FS_SMALL, va="center")
    ax.text(150, 53.0, "distilled edit, held-out\n(full and 0.16% variant)", ha="left", color=TEAL, weight="bold",
            linespacing=1.05, **lab)
    ax.text(36.5, 62.6, "distilled edit, practice seeds", ha="left", color=TEAL, **lab)
    ax.text(36.5, 55.6, "explicit wording, no edit", ha="left", color=MUTE, **lab)
    ax.text(70, 44.0, "cube left of bowl, held-out", ha="left", color=TEAL, **lab)
    ax.text(70, 34.5, "eyeglasses, held-out and practice", ha="left", color=TEAL, **lab)
    ax.text(70, 24.5, "runtime gripper threshold", ha="left", color=MUTE, **lab)
    ax.text(18.0, 39.5, "unedited rerun, butter", ha="left", color=MUTE, **lab)
    ax.text(72, 6.0, "other edits and unchanged reruns", ha="left", color=INK2, **lab)
    save(fig, "fig5_detectability")


# ---- Fig. 4: where to edit ------------------------------------------------------------------------------------------
# study/results/opus/where/results: one shared unedited run on 64 held-out slots (94016-94079) and 32 paired practice
# slots per transfer task (94016-94047). Values: owsp diff (diff_*.json); releases: offline_49.json.
WHERE = OPUS / "where/results"
WHERE_HO = json.loads((WHERE / "heldout_analysis.json").read_text())
WHERE_TX = {t: json.loads((WHERE / f"transfer_{t}_analysis.json").read_text()) for t in ("glasses", "left", "cube")}
WHERE_STAGES = json.loads((WHERE / "heldout_stages.json").read_text())["arms"]
WHERE_BEHAV = json.loads((WHERE / "heldout_behaviour.json").read_text())
WHERE_ARMS = [  # (arm key, label, values changed, location kind)
    ("v1full", "VLM, full edit", "450M", "vlm"),
    ("v1img816", "VLM, 0.16% variant", "248M", "vlm"),
    ("va", "VLM + action expert", "506M", "both"),
    ("a1", "Action expert", "60M", "action"),
    ("h1", "Action head, least norm", "8,056", "action"),
    ("p1", "Action head, gated (blind editor)", "1,024", "action"),
]


def _where_cmp(an, cand, ref="base"):
    hits = [c for c in an["comparisons"] if c["cand"] == cand and c["ref"] == ref]
    assert len(hits) == 1, (cand, ref)
    return hits[0]


WHERE_BASE = {c["ref_succ"] for c in WHERE_HO["comparisons"] if c["ref"] == "base"}
assert WHERE_BASE == {5} and WHERE_HO["n_slots"] == 64 and WHERE_HO["slot_range"] == [94016, 94079]
for _arm, _k, _f, _b in (("v1full", 55, 50, 0), ("v1img816", 41, 38, 2), ("va", 43, 41, 3), ("a1", 20, 20, 5),
                         ("h1", 0, 0, 5), ("p1", 30, 28, 3)):  # the numbers typed in main.tex, Table where
    _c = _where_cmp(WHERE_HO, _arm)
    assert (_c["n"], _c["cand_succ"], _c["fixed"], _c["broke"]) == (64, _k, _f, _b), _c
    assert WHERE_STAGES[_arm]["success"] == _k
for _t, _ref, _arms in (("glasses", 14, {"v1full": 25, "a1": 11, "p1": 12}), ("left", 4, {"v1full": 20, "a1": 2, "p1": 1}),
                        ("cube", 32, {"v1full": 32, "a1": 31, "p1": 32})):
    for _arm, _k in _arms.items():
        _c = _where_cmp(WHERE_TX[_t], _arm)
        assert (_c["n"], _c["ref_succ"], _c["cand_succ"]) == (32, _ref, _k), _c


def fig_where():
    W = TEXTW
    pitch, top, bot = 0.175, 0.36, 0.25
    n = len(WHERE_ARMS)
    H = top + n * pitch + bot
    size_r, a_l, a_w, b_l, b_w = 1.98, 2.08, 1.45, 3.76, 1.70
    fig_a = plt.figure(figsize=(W, H))
    axa = fig_a.add_axes([a_l / W, bot / H, a_w / W, n * pitch / H])
    axb = fig_a.add_axes([b_l / W, bot / H, b_w / W, n * pitch / H])
    tr_fig = mtransforms.blended_transform_factory(fig_a.transFigure, axa.transData)
    base_k = 5
    yt = -1.5
    axa.text(0.0, yt, "a", transform=tr_fig, ha="left", va="center", fontsize=FS_TITLE, weight="bold", color=INK)
    axa.text(0.13 / W, yt, "Cube then banana, 64 held-out seeds", transform=tr_fig, ha="left", va="center",
             fontsize=FS_SMALL, weight="bold", color=INK)
    axa.text(size_r / W, yt + 0.02, "values", transform=tr_fig, ha="right", va="center", fontsize=FS_SMALL - 0.3,
             color=MUTE)
    axb.text(-0.12, yt, "b", transform=axb.get_yaxis_transform(), ha="left", va="center", fontsize=FS_TITLE,
             weight="bold", color=INK)
    axb.text(0.0, yt, "Transfer, 32 practice seeds", transform=axb.get_yaxis_transform(), ha="left",
             va="center", fontsize=FS_SMALL, weight="bold", color=INK)
    for i, (arm, label, vals, kind) in enumerate(WHERE_ARMS):
        c = _where_cmp(WHERE_HO, arm)
        k = c["cand_succ"]
        share = 100 * k / 64
        vlm = kind == "vlm"
        axa.text(0.13 / W, i, label, transform=tr_fig, ha="left", va="center", fontsize=FS_SMALL, color=INK)
        axa.text(size_r / W, i, vals, transform=tr_fig, ha="right", va="center", fontsize=FS_SMALL - 0.3, color=MUTE)
        axa.add_patch(Rectangle((0, i - 0.3), 100, 0.6, fc=PALE, ec="none", zorder=0))
        if k:
            axa.add_patch(Rectangle((0, i - 0.3), share, 0.6, fc=TEAL_PALE if vlm else LIGHT,
                                    ec=TEAL if vlm else "none", lw=0.6, zorder=1))
        lo, hi = wilson(k, 64)
        xb = 100 * base_k / 64
        axa.plot([100 * lo, 100 * hi], [i, i], color=TEAL if vlm else INK2, lw=0.7, zorder=2, solid_capstyle="butt")
        txt = f"{k}/64"
        axa.text(max(100 * hi, xb) + 2.5, i, txt, ha="left", va="center", fontsize=FS_SMALL,
                 color=TEAL if vlm else INK2, weight="bold" if arm == "v1full" else "normal", zorder=3)
        # transfer: unedited (open) to edited (filled), eyeglasses above, cube left of bowl below
        runs = [(t, dy, mk) for t, dy, mk in (("glasses", -0.17, "o"), ("left", 0.17, "^"))
                if any(cc["cand"] == arm and cc["ref"] == "base" for cc in WHERE_TX[t]["comparisons"])]
        if not runs:
            axb.text(2, i, "not run", ha="left", va="center", fontsize=FS_SMALL - 0.3, color=MUTE)
        for t, dy, mk in runs:
            cc = _where_cmp(WHERE_TX[t], arm)
            x0, x1 = 100 * cc["ref_succ"] / 32, 100 * cc["cand_succ"] / 32
            col = TEAL if vlm else INK2
            axb.plot([x0, x1], [i + dy, i + dy], color=LIGHT, lw=0.8, zorder=1)
            axb.plot([x0], [i + dy], marker=mk, ms=3.6, mfc="white", mec=MUTE, mew=0.7, ls="", zorder=2)
            axb.plot([x1], [i + dy], marker=mk, ms=3.6, mfc=col, mec=col, mew=0.7, ls="", zorder=3)
            right = x1 >= x0
            axb.text(x1 + (3 if right else -3), i + dy, f"{cc['cand_succ']}", ha="left" if right else "right",
                     va="center", fontsize=FS_SMALL - 0.6, color=col)
    # unedited reference: one shared run, 5/64
    xb = 100 * base_k / 64
    axa.plot([xb, xb], [-0.5, n - 0.5], color=INK2, lw=0.6, ls=(0, (2, 1.5)), zorder=1.5)
    axa.text(xb + 1.5, -0.62, f"unedited {base_k}/64", ha="left", va="bottom", fontsize=FS_SMALL - 0.4, color=INK2)
    # key for panel b
    ky = -0.55
    for x, mk, txt, fc in ((0, "o", "eyeglasses", INK2), (31, "^", "cube left of bowl", INK2),
                           (81, "o", "unedited", "white")):
        axb.plot([x + 1.5], [ky - 0.12], marker=mk, ms=3.4, mfc=fc, mec=INK2 if fc != "white" else MUTE, mew=0.7,
                 ls="", clip_on=False)
        axb.text(x + 5, ky - 0.12, txt, ha="left", va="center", fontsize=FS_SMALL - 0.4,
                 color=INK2 if fc != "white" else MUTE)
    for a in (axa, axb):
        a.set_xlim(0, 100)
        a.set_ylim(n - 0.5, -0.5)
        a.set_yticks([])
        a.set_xticks([0, 50, 100])
        a.set_xticklabels(["0", "50", "100%"])
        for s_ in ("top", "right", "left"):
            a.spines[s_].set_visible(False)
    axb.set_xticks([0, 25, 50, 75, 100])
    axb.set_xticklabels(["0", "", "50", "", "100%"])
    save(fig_a, "fig4_where")


# ---- appendix: mustard wrist-camera strip -------------------------------------------------------------------------
def fig_mustard():
    im = np.asarray(Image.open(ROOT / "study/blog/figures/opus_mustard_slip.jpg").convert("RGB"))
    # layout of make_mustard_strip.py: pad 8, header 26, tiles 320x180; drop its header text and relabel
    rows = [im[34:214, 8:1640], im[248:428, 8:1640]]
    titles = ["Slips: cap grasp at step 48, bottle dropped at step 98",
              "Holds: cap grasp at step 60, released over the bin at step 168"]
    w = TEXTW
    hrow = w * rows[0].shape[0] / rows[0].shape[1]
    title_h, gap = 0.16, 0.05
    H = 2 * (hrow + title_h) + gap
    fig = plt.figure(figsize=(w, H))
    y = H
    for p, t in zip(rows, titles):
        y -= title_h
        fig.text(0.0, (y + 0.03) / H, t, ha="left", va="bottom", fontsize=FS, color=INK)
        y -= hrow
        ax = fig.add_axes([0, y / H, 1, hrow / H])
        ax.imshow(p, interpolation="lanczos")
        ax.set_axis_off()
        y -= gap
    fig.savefig(OUT / "figA_mustard.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "figA_mustard.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote figA_mustard")


# ---- labels shared by the full outcome figure and the generated tables ---------------------------------------------
LEVEL_ORDER = ["Language binding", "Binding edit on other tasks", "Perception: wrong target object",
               "Motor: missing competence", "Motor: grasp physics", "Gripper commitment (all tasks)",
               "Unchanged weights (noise controls)"]
LEVEL_NAMES = {
    "Language binding": "Language binding (cube then banana)",
    "Binding edit on other tasks": "Same distilled edit, other tasks",
    "Perception: wrong target object": "Perception: lookalike target (stack yellow on red)",
    "Motor: missing competence": "Missing competence (flat box)",
    "Motor: grasp physics": "Contact physics (mustard)",
    "Gripper commitment (all tasks)": "Gripper commitment",
    "Unchanged weights (noise controls)": "Unchanged weights (noise controls)",
}
assert set(LEVEL_ORDER) == {r["level"] for r in ROWS}
LABELS = {  # readable names for the internal candidate labels of comparisons.json
    "Explicit two-step wording (no weight change)": "Explicit wording (no edit)",
    "v2mi-f: binding edit, practice seeds": "Distilled edit, practice",
    "v2mi-f: binding edit, HELD-OUT seeds": "Distilled edit, held-out",
    "binding edit + preservation keys (v2p-f)": "With preservation keys (v2p-f)",
    "minimal edit (0.16%), HELD-OUT seeds": "0.16% variant, held-out",
    "random directions, same size as v2mi-f, HELD-OUT seeds": "Random directions, same size, held-out",
    "Explicit two-step wording, HELD-OUT seeds (no weight change)": "Explicit wording, held-out (no edit)",
    "binding edit, HELD-OUT seeds": "held-out",
    "binding edit": "practice",
    "binding edit, practice (2 runs)": "practice, two runs",
    "binding edit (vs first base run)": "edit vs first unedited run",
    "binding edit + preservation keys (vs first base run)": "preservation keys vs first run",
    "v1a1: attention gain on identity axis": "Attention gain on identity axis (v1a1)",
    "dkv2: VLM key/value gain": "VLM key/value gain (dkv2)",
    "combo1: dkv2 + head rewire (run 1)": "Key/value gain + head rewire (combo1)",
    "combo1 rerun, same weights (run 2)": "Same weights as combo1, second run",
    "Claude, Astra protocol p1 (64 rows, L17)": "Claude, Astra protocol p1 (64 rows)",
    "10 edits (readout, MLP rows, neuron flip, biases)": "10 edits (readout, rows, flips, biases)",
    "gthr04: gripper bias (1 value)": "gripper bias (1 value)",
    "Astra study: runtime threshold 0.4": "runtime threshold 0.4 (Astra study)",
    "Astra study: original identity": "unchanged rerun (Astra study)",
    "Astra study: extra unchanged repeat": "extra unchanged rerun (Astra study)",
    "Astra study: fixed16 identity": "unchanged rerun, fixed16 (Astra study)",
    "this work: unedited rerun, same seeds": "unedited rerun, same seeds",
}
TASKS = {"Butter box onto raisin box": "Butter", "Stack left bowl on right bowl": "Stack bowls",
         "Stack yellow on red": "Stacking", "Mustard into left bin": "Mustard"}


def readable(r, with_task):
    lab = LABELS.get(r["label"], r["label"])
    if not with_task:
        return lab[0].upper() + lab[1:]
    return f'{TASKS.get(r["task"], r["task"])}: {lab}'


# ---- generated LaTeX tables ------------------------------------------------------------------------------------------
def tex_escape(s):
    return (s.replace("&", "\\&").replace("%", "\\%").replace("_", "\\_").replace("#", "\\#")
            .replace("—", "---").replace("×", "$\\times$").replace("→", "$\\to$"))


def fmt_p(p):
    if p >= 0.995:
        return "1"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.001:
        return f"{float(f'{p:.2g}'):g}"
    m, e = f"{p:.1e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt_ci(d, lo, hi):
    def s(v):
        return "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "$-$")
    return f"{s(100 * d)} [{s(100 * lo)}, {s(100 * hi)}]"


def write_tables():
    GEN.mkdir(parents=True, exist_ok=True)
    lines = [f"{tex_escape(t)} & {k}/{n} & {tex_escape(cls)}\\\\" for t, (k, n, cls) in SURVEY.items()]
    (GEN / "survey_table.tex").write_text("\n".join(lines) + "\n")
    lines, first = [], True
    for lv in LEVEL_ORDER:
        if not first:
            lines.append("\\addlinespace")
        first = False
        lines.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{tex_escape(LEVEL_NAMES[lv])}}}}}\\\\")
        with_task = lv in ("Binding edit on other tasks", "Gripper commitment (all tasks)",
                           "Unchanged weights (noise controls)")
        for r in (r for r in ROWS if r["level"] == lv):
            vals = "---" if r["values_changed"] in (None, 0) else f'{r["values_changed"]:,}'
            if r["label"].startswith("10 edits"):
                counts, fb, ci, p = "0/16 each vs 0/64", "0 / 0", "0.0", "---"
            else:
                counts = f'{r["cand"]}/{r["n"]} vs {r["base"]}/{r["n"]}'
                fb = f'{r["fixed"]} / {r["broke"]}'
                ci = fmt_ci(*paired_ci(r))
                p = fmt_p(mcnemar(r["fixed"], r["broke"]))
            lines.append(f"{tex_escape(readable(r, with_task))} & {vals} & {counts} & {fb} & {ci} & {p}\\\\")
        if lv == "Gripper commitment (all tasks)":
            g = GRIP_POOLED
            lines.append(f"Seven tasks pooled: gripper bias (1 value) & 1 & 117/240 vs 117/240 & "
                         f"{g['fixed']} / {g['broke']} & {fmt_ci(*pooled_ci(GRIP))} & "
                         f"{fmt_p(mcnemar(g['fixed'], g['broke']))}\\\\")
    (GEN / "comparisons_table.tex").write_text("\n".join(lines) + "\n")
    tx = [("Pick up eyeglasses, held-out", find("Pick up eyeglasses", "binding edit, HELD")),
          ("Pick up eyeglasses, two practice runs", find("Pick up eyeglasses", "binding edit, practice")),
          ("Cube left of bowl, held-out", find("Cube left of bowl", "binding edit, HELD")),
          ("Cube left of bowl, practice", find("Cube left of bowl", "binding edit")),
          ("Cube right of bowl", find("Cube right of bowl", "binding edit")),
          ("Stack three cubes", find("Stack three cubes", "binding edit")),
          ("Block tower", find("Block tower", "binding edit")),
          ("Yogurt into bowl", find("Yogurt into bowl", "binding edit")),
          ("Grab a bagel", find("Grab a bagel", "binding edit")),
          ("Marker in mug", find("Marker in mug", "binding edit")),
          ("Pick up flat box", dict(n=32, base=0, cand=0, fixed=0, broke=0)),  # failure-levels.md transfer map
          ("Mustard into bin", find("Mustard into bin", "binding edit")),
          ("Stack yellow on red", find("Stack yellow on red", "binding edit")),
          ("Stack bowls", find("Stack bowls", "binding edit")),
          ("Butter on raisin box, vs first unedited run", find("Butter box onto raisin box", "binding edit (vs first")),
          ("Cube into bowl", find("Cube into bowl", "binding edit")),
          ("Banana onto plate", find("Banana onto plate", "binding edit")),
          ("Butter, unedited rerun vs first unedited run", find("Butter box onto raisin box", "this work: unedited"))]
    lines = []
    for name, r in tx:
        lines.append(f'{tex_escape(name)} & {r["n"]} & {r["base"]} & {r["cand"]} & {r["fixed"]} / {r["broke"]} & '
                     f'{fmt_ci(*paired_ci(r))} & {fmt_p(mcnemar(r["fixed"], r["broke"]))}\\\\')
        if name.startswith("Cube left of bowl, practice") or name.startswith("Banana onto plate"):
            lines.append("\\addlinespace")
    (GEN / "transfer_table.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for r in GRIP:
        name = {"Stack left bowl on right bowl": "Stack bowls", "Butter box onto raisin box": "Butter on raisin box",
                "Mustard into left bin": "Mustard into bin"}.get(r["task"], r["task"])
        lines.append(f'{tex_escape(name)} & {r["n"]} & {r["base"]} & {r["cand"]} & {r["fixed"]} / {r["broke"]}\\\\')
    g = GRIP_POOLED
    lines.append("\\midrule")
    lines.append(f'Total & {g["n"]} & {g["base"]} & {g["cand"]} & {g["fixed"]} / {g["broke"]}\\\\')
    (GEN / "gripper_table.tex").write_text("\n".join(lines) + "\n")
    names = {("RubiksCubeThenBananaTask", "v2mi-f"): "Cube then banana, distilled edit",
             ("RubiksCubeThenBananaTask", "abl-imgl8"): "Cube then banana, 0.16\\% variant",
             ("PickGlassesTask", "ctb-v2mi"): "Pick up eyeglasses, distilled edit",
             ("RubiksCubeLeftOfBowlTask", "ctb-v2mi"): "Cube left of bowl, distilled edit"}
    lines = []
    for key in names:
        s, (h1, h2) = HELDOUT[key], HALVES[key]
        lo, hi = s["slots"]
        mid = lo + s["paired_n"] // 2
        for (n, b, c, fx, br), order, rng in ((h1, "unedited first", f"{lo}--{mid - 1}"),
                                              (h2, "edited first", f"{mid}--{hi}")):
            lines.append(f"{names[key] if order.startswith('unedited') else ''} & {rng} & {order} & "
                         f"{b}/{n} & {c}/{n} & {fx} / {br}\\\\")
    (GEN / "heldout_halves_table.tex").write_text("\n".join(lines) + "\n")
    lines = []
    for arm, label, _, _ in WHERE_ARMS:
        st, bh = WHERE_STAGES[arm], WHERE_BEHAV[arm]
        assert bh["n"] == WHERE_STAGES[arm].get("n", 64) == 64
        conv = "---" if st["banana_conversion"] is None else f'{100 * st["banana_conversion"]:.0f}\\%'
        lines.append(f'{tex_escape(label)} & {st["cube_placed"]}/64 & {st["success"]}/64 & {conv} & '
                     f'{100 * bh["first_chunk_closed"]:.0f}\\% & {bh["wrong_object_grabs_per_episode"]:.2f}\\\\')
    lines.insert(0, f'Unedited & {WHERE_STAGES["base"]["cube_placed"]}/64 & {WHERE_STAGES["base"]["success"]}/64 & '
                    f'{100 * WHERE_STAGES["base"]["banana_conversion"]:.0f}\\% & '
                    f'{100 * WHERE_BEHAV["base"]["first_chunk_closed"]:.0f}\\% & '
                    f'{WHERE_BEHAV["base"]["wrong_object_grabs_per_episode"]:.2f}\\\\')
    (GEN / "where_stages_table.tex").write_text("\n".join(lines) + "\n")
    print("wrote generated/{survey,comparisons,transfer,gripper,heldout_halves,where_stages}_table.tex")


def print_table():
    print("\nPaired differences and bootstrap 95% CIs (pp), exact McNemar p:")
    for r in ROWS:
        d, lo, hi = paired_ci(r)
        print(f'{r["task"][:26]:26s} {r["label"][:46]:46s} {100*d:+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}] '
              f'p={mcnemar(r["fixed"], r["broke"]):.2g}')
    d, lo, hi = pooled_ci(GRIP)
    print(f'{"7 tasks pooled":26s} {"gthr04":46s} {100*d:+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]')
    print("MDE at 30% discordant:", {n: round(100 * mde(n, 0.3), 1) for n in (32, 64, 128, 256)})


if __name__ == "__main__":
    fig_teaser()
    fig_construction()
    fig_outcomes()
    fig_outcomes_full()
    fig_detectability()
    fig_where()
    fig_mustard()
    write_tables()
    print_table()
