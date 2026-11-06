#!/usr/bin/env python3
"""Regenerate every figure and generated table of the paper from repository data.

Inputs (all inside this repository):
  study/results/opus/comparisons.json        paired closed-loop comparisons (n, base, cand, fixed, broke)
  study/results/opus/heldout/*.json          held-out runs (summary: Wilson 95% CIs, exact McNemar p, per-slot outcomes)
  study/blog/failure-levels.md               unedited 16-task survey (Appendix A table) and the transfer-map table
  study/blog/figures/opus_mustard_slip.jpg   wrist-camera strip (made by study/results/scripts/make_mustard_strip.py)
  paper/figures/src/ctb_heldout90000_wrist.png
      wrist-camera views of held-out slot 90000 (cube then banana), unedited above and edited below, cropped from
      frame 210 (t = 7.0 s) of the side-by-side clip cube_then_banana.mp4; see paper/README.md.

Outputs: paper/figures/*.pdf (embedded in the paper), paper/figures/preview/*.png (for inspection),
         paper/generated/*.tex (tables input by main.tex).

Paired 95% intervals are bootstrap intervals over the four paired outcome types (both succeed, fixed, broke,
both fail): 20,000 multinomial draws with seed 0, exactly as in study/results/scripts/plot_opus_forest.py.
The pooled one-weight gripper row is bootstrapped stratified by task. p-values are exact two-sided McNemar tests
on the discordant pairs.

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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OPUS = ROOT / "study/results/opus"
REPORT = ROOT / "study/blog/failure-levels.md"
PAPER = ROOT / "paper"
OUT = PAPER / "figures"
PREVIEW = OUT / "preview"
GEN = PAPER / "generated"

# ---- style (matches the reference paper's figures) -----------------------------------------------------------
TEAL, ORANGE, PURPLE = "#147D92", "#C46635", "#7856A5"
GRAY, GRAY_TXT = "#8E959B", "#5F666D"
INK, INK2, GRID, SEP, NOTRUN = "#1F252B", "#4A5056", "#E6E8EA", "#9AA5A8", "#E5E7E9"
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 6.6,
    "axes.titlesize": 7.2,
    "axes.labelsize": 6.6,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "xtick.labelsize": 6.4,
    "ytick.labelsize": 6.4,
    "xtick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.width": 0.6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 300,
    "mathtext.fontset": "dejavusans",
})
TEXTW = 5.5  # \textwidth of the CoRL style, inches: figures are drawn at their printed size
SEQ = LinearSegmentedColormap.from_list("teal_seq", [
    (0.00, "#F1F8F8"), (0.05, "#E6F2F2"), (0.25, "#AFD8D8"), (0.50, "#52A7AE"), (0.70, "#087B88"), (1.00, "#03505A")])


def save(fig, name, pad=0.02):
    OUT.mkdir(parents=True, exist_ok=True)
    PREVIEW.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight", pad_inches=pad)
    fig.savefig(PREVIEW / f"{name}.png", bbox_inches="tight", pad_inches=pad, dpi=250)
    plt.close(fig)
    print("wrote", name)


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
    # run order alternates by half: first half unedited then edited, second half edited then unedited
    mid = s["slots"][0] + s["paired_n"] // 2
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


def color_of(r):
    """Teal: distilled (language-binding) edit. Orange: GPT-6 Astra. Purple: other Claude Opus edits.
    Gray: no weight change (prompt, runtime threshold, unchanged-weight reruns)."""
    if r["type"] in ("identity", "prompt", "runtime"):
        return GRAY
    if r["editor"] == "GPT-6 Astra":
        return ORANGE
    if r["level"] in ("Language binding", "Binding edit on other tasks"):
        return TEAL
    return PURPLE


def txt_color(c):
    return GRAY_TXT if c == GRAY else c


def signed(v):
    return "0.0" if abs(v) < 0.05 else f"{v:+.1f}".replace("-", "−")


# ---- Table 1 image: held-out start 90000, unedited vs edited -------------------------------------------------
def fig_teaser():
    im = np.asarray(Image.open(OUT / "src/ctb_heldout90000_wrist.png").convert("RGB"))
    h = (im.shape[0] - 6) // 2
    panels = [(im[:h], "Unedited", GRAY_TXT), (im[h + 6:], "Edited", TEAL)]
    w_in = 0.30 * TEXTW
    fig = plt.figure(figsize=(w_in, w_in * (2 * h + 6) / im.shape[1]))
    gap = 6 / (2 * h + 6)
    for i, (p, lab, c) in enumerate(panels):
        ax = fig.add_axes([0, (1 - i) * (0.5 + gap / 2), 1, 0.5 - gap / 2])
        ax.imshow(p, interpolation="lanczos")
        ax.set_axis_off()
        ax.text(0.03, 0.95, lab, transform=ax.transAxes, ha="left", va="top", fontsize=6.4, color="white",
                weight="bold", bbox=dict(boxstyle="round,pad=0.25,rounding_size=0.15", fc=c, ec="none", alpha=0.92))
    fig.savefig(OUT / "teaser_frames.pdf", dpi=400)
    fig.savefig(PREVIEW / "teaser_frames.png", dpi=250)
    plt.close(fig)
    print("wrote teaser_frames")


# ---- effect panels in the style of the reference's wording-effect figure --------------------------------------
XLIM = (-72, 140)


def effect_panel(ax, items, title, label_x, xlabel=True, fs=6.4):
    """Diamonds (paired difference) and bars (paired 95% bootstrap interval); the difference is printed at the right
    in the row colour. items: ('header', text) or ('row', label, comparison | (d, lo, hi), colour).
    label_x: left edge (figure fraction) of the group headers."""
    fig = ax.figure
    y, ys, labels = 0.0, [], []
    head_tf = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    for it in items:
        if it[0] == "header":
            y += 0.3 if ys else 0
            ax.text(label_x, y, it[1], transform=head_tf, ha="left", va="center", fontsize=fs, weight="bold",
                    color=INK, clip_on=False)
            ys.append(y); labels.append("")
            y += 1
            continue
        _, lab, r, c = it
        if isinstance(r, tuple):
            d, lo, hi = r
        elif r.get("pooled"):
            d, lo, hi = pooled_ci(r["parts"])
        else:
            d, lo, hi = paired_ci(r)
        d, lo, hi = 100 * d, 100 * lo, 100 * hi
        if hi > lo:
            ax.add_patch(Rectangle((lo, y - 0.32), hi - lo, 0.64, fc=c, ec="none", alpha=0.16, zorder=1))
            ax.plot([lo, hi], [y, y], color=c, lw=1.0, zorder=2, solid_capstyle="butt")
            for x in (lo, hi):
                ax.plot([x, x], [y - 0.22, y + 0.22], color=c, lw=1.0, zorder=2)
        ax.plot([d], [y], marker="D", ms=3.9, mfc="white", mec=c, mew=1.05, zorder=3, ls="")
        ax.text(XLIM[1] - 1, y, signed(d), ha="right", va="center", fontsize=fs, color=txt_color(c), zorder=4)
        ys.append(y); labels.append(lab)
        y += 1
    ax.set_ylim(y - 0.4, -0.75)
    ax.set_xlim(*XLIM)
    ax.set_xticks([-50, 0, 50, 100])
    ax.set_xticklabels(["−50", "0", "50", "100"])
    ax.axvline(0, color=SEP, lw=0.8, ls=(0, (3, 2)), zorder=0)
    for v in (-50, 50, 100):
        ax.axvline(v, color=GRID, lw=0.6, zorder=0)
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=fs, color=INK)
    ax.tick_params(axis="y", length=0, pad=2.5)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    if title:
        ax.set_title(title, fontsize=7.0, pad=4)
    if xlabel:
        ax.set_xlabel("Success-rate difference\n(percentage points)", fontsize=6.4)
    return y


def ho_row(s):
    return dict(n=s["paired_n"], base=s["base_success"], cand=s["cand_success"], fixed=s["fixed"], broke=s["broke"])


def fig_effects():
    flat = (0.0, 0.0, 0.0)  # ten flat-box edits: 0/16 each against 0/64 unedited; no interval
    a = [
        ("header", "Cube then banana"),
        ("row", "Explicit wording, no edit · 32", find("Cube then banana", "Explicit two-step"), GRAY),
        ("row", "Practice seeds · 32", find("Cube then banana", "v2mi-f: binding edit, practice"), TEAL),
        ("row", "Held-out seeds · 128", ho_row(CTB), TEAL),
        ("row", "0.16% variant, held-out · 128", ho_row(CTB_MIN), TEAL),
        ("row", "Preservation keys · 32", find("Cube then banana", "binding edit + preservation"), TEAL),
        ("header", "Same edit, other tasks"),
        ("row", "Eyeglasses, practice · 64", find("Pick up eyeglasses", "binding edit, practice"), TEAL),
        ("row", "Eyeglasses, held-out · 64", ho_row(EYE), TEAL),
        ("row", "Cube left, practice · 32", find("Cube left of bowl", "binding edit"), TEAL),
        ("row", "Cube left, held-out · 64", ho_row(LEFT), TEAL),
        ("row", "Cube right · 32", find("Cube right of bowl", "binding edit"), TEAL),
        ("header", "Unchanged weights, rerun"),
        ("row", "Stacking, rerun 1 · 64", find("Stack yellow on red", "Astra study: original identity"), GRAY),
        ("row", "Stacking, rerun 2 · 64", find("Stack yellow on red", "Astra study: extra unchanged"), GRAY),
        ("row", "Stacking, rerun 3 · 64", find("Stack yellow on red", "Astra study: fixed16 identity"), GRAY),
        ("row", "Butter, rerun · 16", find("Butter box onto raisin box", "this work: unedited"), GRAY),
    ]
    b = [
        ("header", "Perception: stack yellow on red"),
        ("row", "Claude, attention gain · 64", find("Stack yellow on red", "v1a1"), PURPLE),
        ("row", "Claude, key/value gain · 64", find("Stack yellow on red", "dkv2"), PURPLE),
        ("row", "Claude, key/value + head · 64", find("Stack yellow on red", "combo1: dkv2"), PURPLE),
        ("row", "Same edit, rerun · 64", find("Stack yellow on red", "combo1 rerun"), PURPLE),
        ("row", "Astra, 4,412 values · 32", find("Stack yellow on red", "Astra A1 p1"), ORANGE),
        ("row", "Astra, 1,543 values · 32", find("Stack yellow on red", "Astra A2 p1"), ORANGE),
        ("row", "Astra, 791 values · 32", find("Stack yellow on red", "Astra A2 p3"), ORANGE),
        ("row", "Astra, 6,225 values · 32", find("Stack yellow on red", "Astra fixed16 p1"), ORANGE),
        ("row", "Claude, Astra protocol 1 · 32", find("Stack yellow on red", "Claude, Astra protocol p1"), PURPLE),
        ("row", "Claude, Astra protocol 2 · 32", find("Stack yellow on red", "Claude, Astra protocol p2"), PURPLE),
        ("header", "Competence, contact, gripper"),
        ("row", "Flat box, 10 edits · 16 each", flat, PURPLE),
        ("row", "Mustard, one weight · 64", find("Mustard into left bin", "gthr04"), PURPLE),
        ("row", "Seven tasks, one weight · 240", dict(pooled=True, parts=GRIP), PURPLE),
        ("row", "Runtime threshold · 64", find("Stack yellow on red", "Astra study: runtime"), GRAY),
    ]
    n = max(len(a) + 0.3 * 2, len(b) + 0.3 * 1)
    h = 0.116 * n + 0.58
    fig = plt.figure(figsize=(TEXTW, h))
    bottom, top = 0.48 / h, 1 - 0.2 / h
    ax_a = fig.add_axes([0.258, bottom, 0.218, top - bottom])
    ax_b = fig.add_axes([0.782, bottom, 0.218, top - bottom])
    effect_panel(ax_a, a, "(a) Distilled edit", label_x=0.0)
    effect_panel(ax_b, b, "(b) Edits at other failure levels", label_x=0.505)
    for ax in (ax_a, ax_b):  # same row pitch in both panels
        ax.set_ylim(n - 0.4, -0.75)
    save(fig, "fig_effects")


# ---- heatmap in the style of the reference's goal-response figure -------------------------------------------
HEAT_TASKS = [  # display, survey name, distilled-edit row (task, label), gripper row task, held-out summary
    ("Cube then banana", "Cube then banana into bowl", ("Cube then banana", "v2mi-f: binding edit, practice"), None, CTB),
    ("Pick up eyeglasses", "Pick up eyeglasses", ("Pick up eyeglasses", "binding edit, practice"),
     "Pick up eyeglasses", EYE),
    ("Cube left of bowl", "Cube left of bowl", ("Cube left of bowl", "binding edit"), None, LEFT),
    ("Cube into bowl", "Cube into bowl", ("Cube into bowl", "binding edit"), "Cube into bowl", None),
    ("Banana onto plate", "Banana onto plate", ("Banana onto plate", "binding edit"), "Banana onto plate", None),
    ("Butter on raisin box", "Butter box onto raisin box", ("Butter box onto raisin box", "binding edit (vs first"),
     "Butter box onto raisin box", None),
    ("Stack three cubes", "Stack three cubes", ("Stack three cubes", "binding edit"), None, None),
    ("Block tower", "Block tower, any order", ("Block tower", "binding edit"), None, None),
    ("Stack bowls", "Stack left bowl on right bowl", ("Stack bowls", "binding edit"), "Stack left bowl on right bowl",
     None),
    ("Stack yellow on red", "Stack yellow on red", ("Stack yellow on red", "binding edit"), "Stack yellow on red", None),
    ("Mustard into bin", "Mustard into left bin", ("Mustard into bin", "binding edit"), "Mustard into left bin", None),
    ("Marker in mug", "Marker in mug", ("Marker in mug", "binding edit"), None, None),
    ("Yogurt into bowl", "Yogurt into bowl", ("Yogurt into bowl", "binding edit"), None, None),
    ("Cube right of bowl", "Cube right of bowl", ("Cube right of bowl", "binding edit"), None, None),
    ("Grab a bagel", "Grab a bagel", ("Grab a bagel", "binding edit"), None, None),
    ("Pick up flat box", "Pick up flat green box", None, None, None),
]


def heat_rows():
    out = []
    for disp, sname, brow, gtask, ho in HEAT_TASKS:
        sv = SURVEY[sname][:2]
        if brow is None:  # flat box: transfer-map row of failure-levels.md
            b = (frac(FLATBOX_TX[1]), frac(FLATBOX_TX[2]))
        else:
            r = find(*brow)
            b = ((r["base"], r["n"]), (r["cand"], r["n"]))
        h = None if ho is None else ((ho["base_success"], ho["paired_n"]), (ho["cand_success"], ho["paired_n"]))
        g = None
        if gtask:
            r = [x for x in GRIP if x["task"] == gtask][0]
            g = ((r["base"], r["n"]), (r["cand"], r["n"]))
        out.append((disp, [sv, *b, *(h or (None, None)), *(g or (None, None))]))
    assert len(out) == 16
    return out


def fig_heatmap():
    rows = heat_rows()
    groups = [("Survey", 1), ("Distilled edit,\npractice seeds", 2), ("Distilled edit,\nheld-out seeds", 2),
              ("One-weight\ngripper edit", 2)]
    sub = ["Unedited", "Unedited", "Edited", "Unedited", "Edited", "Unedited", "Edited"]
    ncol, nrow = len(sub), len(rows)
    gapx = 0.3
    xs, x = [], 0.0
    for _, k in groups:
        for _ in range(k):
            xs.append(x); x += 1
        x += gapx
    width_cells = x - gapx
    sep_after = 3  # thin rule after the three tasks with held-out gains
    ys = [i + (0.3 if i >= sep_after else 0) for i in range(nrow)]
    fig_w, fig_h = TEXTW, 0.128 * (nrow + 0.3) + 0.7
    fig = plt.figure(figsize=(fig_w, fig_h))
    left, right = 1.08 / fig_w, 1 - 0.42 / fig_w
    bottom, top = 0.28 / fig_h, 1 - 0.42 / fig_h
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    for i, (disp, vals) in enumerate(rows):
        y = ys[i]
        for j, v in enumerate(vals):
            x0 = xs[j]
            if v is None:
                ax.add_patch(Rectangle((x0 + 0.03, y - 0.46), 0.94, 0.92, fc=NOTRUN, ec="none"))
                continue
            k, n = v
            rate = k / n
            ax.add_patch(Rectangle((x0 + 0.03, y - 0.46), 0.94, 0.92, fc=SEQ(rate), ec="none"))
            label = f"{k}/{n}" + ("†" if disp == "Butter on raisin box" and j == 1 else "")
            ax.text(x0 + 0.5, y, label, ha="center", va="center", fontsize=6.1,
                    color="white" if rate >= 0.58 else INK)
    ax.set_xlim(-0.05, width_cells + 0.05)
    ax.set_ylim(ys[-1] + 0.6, -0.6)
    ax.set_yticks(ys)
    ax.set_yticklabels([r[0] for r in rows], fontsize=6.4)
    for i, t in enumerate(ax.get_yticklabels()):
        if i < sep_after:
            t.set_color(TEAL)
            t.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xticks([xs[j] + 0.5 for j in range(ncol)])
    ax.set_xticklabels(sub, fontsize=6.2)
    ax.tick_params(axis="x", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)
    j = 0
    for gi, (name, k) in enumerate(groups):
        x0, x1 = xs[j], xs[j + k - 1] + 1
        ax.text((x0 + x1) / 2, -0.98, name, ha="center", va="bottom", fontsize=6.4, weight="bold", color=INK,
                linespacing=1.05, clip_on=False)
        ax.plot([x0 + 0.05, x1 - 0.05], [-0.74, -0.74], color=GRAY_TXT, lw=0.6, clip_on=False)
        if gi > 0:
            xm = x0 - gapx / 2
            ax.plot([xm, xm], [-0.55, ys[-1] + 0.55], color=SEP, lw=0.9, clip_on=False)
        j += k
    ax.plot([-0.05, width_cells + 0.05], [(ys[sep_after - 1] + ys[sep_after]) / 2] * 2, color=SEP, lw=0.6)
    cax = fig.add_axes([right + 0.1 / fig_w, bottom + 0.05, 0.08 / fig_w, top - bottom - 0.1])
    cax.imshow(np.linspace(0, 1, 256)[:, None], aspect="auto", cmap=SEQ, origin="lower", extent=(0, 1, 0, 100))
    cax.set_xticks([])
    cax.yaxis.tick_right()
    cax.set_yticks([0, 50, 100])
    cax.tick_params(axis="y", labelsize=6.0, length=2, pad=1.5)
    for s in cax.spines.values():
        s.set_visible(False)
    cax.set_title("%", fontsize=6.0, pad=2)
    save(fig, "fig_heatmap")


# ---- main text: the procedure and where the edit acts ------------------------------------------------------------
def box(ax, x, y, w, h, text="", ec=SEP, fc="white", fs=5.9, color=INK, weight="normal", lw=0.7):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.07", fc=fc, ec=ec, lw=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=color, weight=weight,
                linespacing=1.28)


def arrow(ax, p, q, color=GRAY_TXT, lw=0.7, rad=0.0, ms=6):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=ms, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0.5, shrinkB=0.5))


STEPS = [  # header, description, outcome for cube then banana
    ("1  Diagnose", "Replay failure\ndecisions under\nrewordings; count\nreleases", "original: 21 of 21\nexplicit: 2 of 21"),
    ("2  Known-good", "Same weights,\nexplicit wording,\nclosed loop", "22/32 vs 3/32"),
    ("3  Distil", "Map original keys\nto the explicit\nresidual changes,\nEq. (1)", "17 layers,\n72 keys per layer"),
    ("4  Check offline", "Replay the recorded\ndecisions with the\nedited weights", "releases 21 → 0,\nothers unchanged"),
    ("5  Confirm", "Closed loop: practice\nseeds, then held-out\nseeds", "3 → 25 of 32\n27 → 106 of 128"),
]


def fig_method():
    W, H = 11.0, 4.9  # drawing units; 1 unit = 0.5 in
    fig = plt.figure(figsize=(TEXTW, TEXTW * H / W))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.text(0.03, 4.88, "(a) Closed-form instruction distillation", fontsize=7.0, weight="bold", va="top", color=INK)
    gap = 0.2
    bw = (W - 0.06 - 4 * gap) / 5
    y0, bh = 2.62, 1.94
    for i, (head, desc, res) in enumerate(STEPS):
        x = 0.03 + i * (bw + gap)
        box(ax, x, y0, bw, bh, ec=TEAL if i == 2 else SEP, lw=1.0 if i == 2 else 0.7,
            fc="#EEF6F7" if i == 2 else "white")
        ax.text(x + 0.1, y0 + bh - 0.12, head, ha="left", va="top", fontsize=6.3, weight="bold", color=INK)
        ax.text(x + 0.1, y0 + bh - 0.46, desc, ha="left", va="top", fontsize=5.7, color=INK2, linespacing=1.25)
        ax.plot([x + 0.1, x + bw - 0.1], [y0 + 0.66, y0 + 0.66], color=GRID, lw=0.6)
        ax.text(x + 0.1, y0 + 0.09, res, ha="left", va="bottom", fontsize=5.8, color=TEAL, linespacing=1.25)
        if i < 4:
            arrow(ax, (x + bw + 0.01, y0 + bh / 2), (x + bw + gap - 0.01, y0 + bh / 2), ms=5)
    ax.text(0.03, 2.36, "(b) Where the edit acts in $\\pi_{0.5}$", fontsize=7.0, weight="bold", va="top", color=INK)
    box(ax, 0.03, 1.12, 1.62, 0.86, "Original\ninstruction\n(unchanged)", fs=5.7)
    box(ax, 0.03, 0.06, 1.62, 0.86, "Two camera images\nand joint/gripper\nstate", fs=5.6)
    ax.add_patch(FancyBboxPatch((2.0, 0.03), 4.62, 2.0, boxstyle="round,pad=0,rounding_size=0.08", fc="#F6F7F7",
                                ec=SEP, lw=0.7))
    ax.text(2.1, 1.95, "PaliGemma VLM prefix, 18 layers", fontsize=5.9, color=INK, va="top", weight="bold")
    ty = 1.02
    for i in range(10):
        ax.add_patch(Rectangle((2.18 + i * 0.21, ty), 0.16, 0.26, fc=TEAL, ec="none", alpha=0.95 if i % 2 else 0.7))
    for i in range(7):
        ax.add_patch(Rectangle((4.68 + i * 0.21, ty), 0.16, 0.26, fc=GRAY, ec="none"))
    ax.text(2.18 + 1.02, ty - 0.06, "32 image cells", fontsize=5.5, ha="center", va="top", color=INK2)
    ax.text(4.68 + 0.71, ty - 0.06, "40 prompt + state positions", fontsize=5.5, ha="center", va="top", color=INK2)
    arrow(ax, (5.3, ty + 0.3), (3.3, ty + 0.3), color=INK2, rad=0.14, lw=0.6, ms=4.5)
    ax.text(4.3, 1.66, "image tokens attend to the instruction", fontsize=5.4, ha="center", va="center",
            color=INK2, style="italic")
    box(ax, 2.18, 0.14, 4.26, 0.46, "MLP down-projections, layers 0–16:  $W_\\ell \\leftarrow W_\\ell + "
        "\\Delta W_\\ell$", ec=TEAL, lw=1.0, fs=5.8)
    arrow(ax, (1.65, 1.55), (2.0, 1.4))
    arrow(ax, (1.65, 0.49), (2.0, 0.7))
    box(ax, 7.02, 0.56, 1.75, 1.05, "Gemma action\nexpert, 18 layers,\nflow matching", fs=5.7)
    arrow(ax, (6.62, 1.08), (7.02, 1.08))
    ax.text(7.895, 0.47, "13–44% of its attention\non image tokens", fontsize=5.3, ha="center", va="top",
            color=INK2, style="italic", linespacing=1.2)
    box(ax, 9.17, 0.56, 1.8, 1.05, "15-step action\nchunk: keep\nholding or release", fs=5.7)
    arrow(ax, (8.77, 1.08), (9.17, 1.08))
    save(fig, "fig_method")


# ---- appendix: every comparison in comparisons.json -----------------------------------------------------------
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


def fig_forest_full():
    items = []
    for lv in LEVEL_ORDER:
        items.append(("header", LEVEL_NAMES[lv]))
        with_task = lv in ("Binding edit on other tasks", "Gripper commitment (all tasks)",
                           "Unchanged weights (noise controls)")
        for r in (r for r in ROWS if r["level"] == lv):
            items.append(("row", f'{readable(r, with_task)} · {r["n"]}', r, color_of(r)))
    nh = sum(1 for i in items if i[0] == "header")
    n = len(items) + 0.3 * (nh - 1)
    h = 0.124 * n + 0.6
    fig = plt.figure(figsize=(TEXTW, h))
    bottom, top = 0.48 / h, 1 - 0.14 / h
    ax = fig.add_axes([0.43, bottom, 0.355, top - bottom])
    effect_panel(ax, items, "", label_x=0.0, fs=6.0)
    ax.set_ylim(n - 0.4, -0.75)
    right = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    for yy, it in zip(ax.get_yticks(), items):
        if it[0] == "row":
            r = it[2]
            txt = r.get("display") or f'{r["cand"]}/{r["n"]} vs {r["base"]}/{r["n"]}'
            ax.text(1.03, yy, txt, transform=right, ha="left", va="center", fontsize=5.8, color=INK2, clip_on=False)
    ax.text(1.03, -1.05, "edited vs unedited", transform=right, ha="left", va="center", fontsize=5.8,
            color=INK2, style="italic", clip_on=False)
    save(fig, "figA_forest_full")


# ---- appendix: noise floor ---------------------------------------------------------------------------------------
def mde(n, pd, z=1.959964 + 0.841621):
    """Normal-approximation minimum detectable paired difference (80% power, two-sided alpha 0.05)."""
    return z * np.sqrt(pd / n)


def fig_noise():
    fig = plt.figure(figsize=(TEXTW, 2.45))
    a1 = fig.add_axes([0.235, 0.3, 0.25, 0.6])
    a2 = fig.add_axes([0.61, 0.3, 0.375, 0.6])
    cats = []
    ident = [r for r in ROWS if r["type"] == "identity"]
    cats.append(("Unchanged weights,\nrerun vs. first run", [((r["fixed"] + r["broke"]) / r["n"], GRAY) for r in ident]))
    # two runs of the same edited weights agreed on 46 of 64 paired episodes (study/results/opus/EVIDENCE.md)
    cats.append(("Same edited weights,\ntwo runs", [(18 / 64, PURPLE)]))
    head = [r for r in GRIP if 0 < r["base"] < r["n"] and r["n"] >= 32]
    assert sum(r["fixed"] + r["broke"] for r in head) == 70 and sum(r["n"] for r in head) == 192
    cats.append(("Neutral one-weight edit\n(tasks with headroom)", [(70 / 192, PURPLE)]))
    pe = [r for r in ROWS if r["level"].startswith("Perception") and r["type"] == "weight"]
    cats.append(("Perception edits\n(stacking)", [((r["fixed"] + r["broke"]) / r["n"], color_of(r)) for r in pe]))
    ho = [r for r in ROWS if r["type"] == "heldout"]
    cats.append(("Distilled edit,\nheld-out seeds", [((r["fixed"] + r["broke"]) / r["n"], TEAL) for r in ho]))
    for ci, (lab, pts) in enumerate(cats):
        offs = np.linspace(-0.2, 0.2, len(pts)) if len(pts) > 1 else [0.0]
        for (f, c), o in zip(sorted(pts), offs):
            a1.plot([100 * f], [ci + o], "o", ms=3.4, mfc=c, mec="white", mew=0.4, zorder=3)
    a1.axvspan(28, 36, color=GRID, zorder=0, lw=0)
    a1.text(32, len(cats) - 0.58, "28–36%", ha="center", va="center", fontsize=5.7, color=GRAY_TXT)
    a1.set_yticks(range(len(cats)))
    a1.set_yticklabels([c[0] for c in cats], fontsize=6.0)
    a1.set_ylim(len(cats) - 0.25, -0.5)
    a1.tick_params(axis="y", length=0)
    a1.set_xlim(0, 80)
    a1.set_xticks([0, 20, 40, 60, 80])
    a1.set_xlabel("Paired episodes whose outcome differs (%)", fontsize=6.2)
    a1.set_title("(a) Discordant pairs", fontsize=7.0, pad=4)
    for s in ("top", "right", "left"):
        a1.spines[s].set_visible(False)
    ns = np.geomspace(14, 300, 200)
    a2.fill_between(ns, -100 * mde(ns, 0.3), 100 * mde(ns, 0.3), color=GRID, lw=0, zorder=0)
    for pd, ls in ((0.30, "-"), (0.50, (0, (3, 2)))):
        e = 100 * mde(ns, pd)
        a2.plot(ns, e, color=GRAY_TXT, lw=0.7, ls=ls)
        a2.plot(ns, -e, color=GRAY_TXT, lw=0.7, ls=ls)
    rng = np.random.default_rng(3)
    for r in ROWS:
        if r["label"].startswith("10 edits"):
            continue
        d = 100 * (r["cand"] - r["base"]) / r["n"]
        a2.plot([r["n"] * rng.uniform(0.93, 1.07)], [d], "D" if r["type"] == "heldout" else "o",
                ms=3.6 if r["type"] == "heldout" else 3.1, mfc=color_of(r), mec="white", mew=0.4, zorder=3)
    a2.axhline(0, color=SEP, lw=0.7, ls=(0, (3, 2)))
    a2.set_xscale("log", base=2)
    a2.set_xticks([16, 32, 64, 128, 256])
    a2.set_xticklabels(["16", "32", "64", "128", "256"])
    a2.minorticks_off()
    a2.set_xlim(12, 330)
    a2.set_ylim(-52, 80)
    a2.set_xlabel("Paired episodes (log scale)", fontsize=6.2)
    a2.set_ylabel("Observed difference (pp)", fontsize=6.2)
    a2.set_title("(b) Observed effects and detectable change", fontsize=7.0, pad=4)
    for s in ("top", "right"):
        a2.spines[s].set_visible(False)
    handles = [plt.Line2D([], [], ls="", marker=m, ms=3.4, mfc=c, mec="white", mew=0.4, label=l) for c, m, l in (
        (TEAL, "o", "Distilled edit, practice"), (TEAL, "D", "Distilled edit, held-out"),
        (PURPLE, "o", "Other Claude edits"), (ORANGE, "o", "Astra edits"), (GRAY, "o", "No weight change"))]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=5.8, handletextpad=0.2,
               columnspacing=1.1, bbox_to_anchor=(0.5, 0.0))
    save(fig, "figA_noise")


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
        fig.text(0.0, (y + 0.03) / H, t, ha="left", va="bottom", fontsize=6.6, color=INK)
        y -= hrow
        ax = fig.add_axes([0, y / H, 1, hrow / H])
        ax.imshow(p, interpolation="lanczos")
        ax.set_axis_off()
        y -= gap
    fig.savefig(OUT / "figA_mustard.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(PREVIEW / "figA_mustard.png", dpi=220, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print("wrote figA_mustard")


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
    # 6 columns: comparison, values changed, counts, fixed/broke, difference [CI], p
    (GEN / "comparisons_table.tex").write_text("\n".join(lines) + "\n")
    # Appendix E: the distilled edit on the 15 other surveyed tasks
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
    # Appendix D: the one-weight gripper edit on seven tasks
    lines = []
    for r in GRIP:
        name = {"Stack left bowl on right bowl": "Stack bowls", "Butter box onto raisin box": "Butter on raisin box",
                "Mustard into left bin": "Mustard into bin"}.get(r["task"], r["task"])
        lines.append(f'{tex_escape(name)} & {r["n"]} & {r["base"]} & {r["cand"]} & {r["fixed"]} / {r["broke"]}\\\\')
    g = GRIP_POOLED
    lines.append("\\midrule")
    lines.append(f'Total & {g["n"]} & {g["base"]} & {g["cand"]} & {g["fixed"]} / {g["broke"]}\\\\')
    (GEN / "gripper_table.tex").write_text("\n".join(lines) + "\n")
    # Appendix F: held-out runs split by run order
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
    print("wrote generated/{survey,comparisons,transfer,gripper,heldout_halves}_table.tex")


def print_table():
    print("\nPaired differences and bootstrap 95% CIs (pp), exact McNemar p:")
    for r in ROWS:
        d, lo, hi = paired_ci(r)
        print(f'{r["task"][:26]:26s} {r["label"][:46]:46s} {100*d:+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}] '
              f'p={mcnemar(r["fixed"], r["broke"]):.2g}')
    d, lo, hi = pooled_ci(GRIP)
    print(f'{"7 tasks pooled":26s} {"gthr04":46s} {100*d:+6.1f} [{100*lo:+6.1f}, {100*hi:+6.1f}]')
    # minimum detectable paired differences quoted in the paper (30% discordant pairs)
    print("MDE at 30% discordant:", {n: round(100 * mde(n, 0.3), 1) for n in (32, 64, 128, 256)})


if __name__ == "__main__":
    fig_teaser()
    fig_method()
    fig_effects()
    fig_heatmap()
    fig_forest_full()
    fig_noise()
    fig_mustard()
    write_tables()
    print_table()
