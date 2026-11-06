#!/usr/bin/env python3
"""Render the paper's four main figures for the project site, at screen size, in light and dark.

The data, the statistics and the figure designs come from the paper (paper/scripts/make_figures.py, imported below), so
the site and the paper show the same pictures. The drawing functions are copies of the paper's with three changes:
  * sizes: text is set about 1.45x larger and the layout about 1.35x wider, so a figure shown at its natural size
    (about 700-730 CSS px) has 12-14 px text;
  * colour: the site uses one accent, teal, for the edit; the paper's muted red for harm becomes a dark neutral;
  * a dark variant, drawn on a transparent background with light text.

Outputs, in site/assets/fig/: <name>.svg and <name>-dark.svg (text as paths), plus <name>.png and <name>-dark.png at
2x as a fallback. Names: teaser, construction, outcomes, detectability.

Usage: python3 site/scripts/make_web_figures.py      (needs numpy, matplotlib and pillow, as the paper's script)
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import numpy as np
from matplotlib.patches import Rectangle
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "paper/scripts"))
import make_figures as P  # noqa: E402  (loads the data and runs the paper's consistency checks)

OUT = ROOT / "site/assets/fig"
K = 1.45   # text scale relative to the paper
G = 1.35   # layout scale relative to the paper
W = P.TEXTW * G
FS, FS_SMALL, FS_TITLE = P.FS * K, P.FS_SMALL * K, P.FS_TITLE * K

THEMES = {
    "light": dict(ink="#1A1915", ink2="#4A4944", mute="#7A7873", light="#C9C6BE", pale="#ECEAE4", rule="#A9A6A0",
                  teal="#147D92", teal_pale="#D3E7EB", harm="#2E2D2A", both="#8F8C86", none="#DEDBD3",
                  page="#FAF9F5", on_fill="#FFFFFF"),
    "dark": dict(ink="#ECEAE4", ink2="#C4C1BA", mute="#9C9993", light="#5C5A55", pale="#2F2E2B", rule="#77746E",
                 teal="#2AA5BC", teal_pale="#1D3F47", harm="#E6E3DC", both="#86837D", none="#3B3A36",
                 page="#1A1917", on_fill="#141413"),
}
C = THEMES["light"]


def use_theme(name):
    global C
    C = THEMES[name]
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Arial Unicode MS"],
        "font.size": FS, "axes.labelsize": FS, "xtick.labelsize": FS_SMALL, "ytick.labelsize": FS_SMALL,
        "axes.edgecolor": C["rule"], "axes.linewidth": 0.7, "axes.labelcolor": C["ink2"],
        "xtick.color": C["ink2"], "ytick.color": C["ink2"], "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.major.size": 3.0, "ytick.major.size": 3.0, "text.color": C["ink"],
        "mathtext.fontset": "custom", "mathtext.rm": "Arial", "mathtext.it": "Arial:italic",
        "mathtext.bf": "Arial:bold", "mathtext.fallback": "stixsans",
        "svg.fonttype": "path", "svg.hashsalt": "weightsmith",
        "figure.facecolor": "none", "axes.facecolor": "none", "savefig.facecolor": "none",
    })


def save(fig, name, theme):
    OUT.mkdir(parents=True, exist_ok=True)
    stem = name + ("-dark" if theme == "dark" else "")
    # dpi only sets the resolution of embedded rasters (the teaser's camera frames, 224 px wide at source)
    fig.savefig(OUT / f"{stem}.svg", bbox_inches="tight", pad_inches=0.04, transparent=True, dpi=144,
                metadata={"Date": None})
    fig.savefig(OUT / f"{stem}.png", bbox_inches="tight", pad_inches=0.04, transparent=True, dpi=192)
    plt.close(fig)
    print("wrote", stem)


def bare(ax):
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def plain_axes(ax, keep=("bottom",)):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


# ---- Fig. 1: one held-out start, unedited vs edited (paper: fig_teaser) ------------------------------------------
def fig_teaser(theme):
    meta = P.json.loads((P.SRC / "teaser_90002.json").read_text())
    sheet = np.asarray(Image.open(P.SRC / "teaser_90002_frames.png").convert("RGB"))
    h, w = meta["frame_size"]
    gap = meta["frame_gap"]
    moments = meta["storyboard_steps"]
    U, E = meta["arms"]["unedited"], meta["arms"]["edited"]
    assert not U["success"] and E["success"] and E["final_step"] == 257

    def ev(A, name, obj=None):
        return min(e["step"] for e in A["events"] if e["name"] == name and (obj is None or obj in e["info"]))

    grab_u, drop_u = ev(U, "OBJECT_GRABBED_SUCCESS", "rubiks"), ev(U, "TARGET_OBJECT_DROPPED")
    banana_off_u = ev(U, "OK", "banana")
    cube_bowl_u = ev(U, "OBJECT_IN_CONTAINER_SUCCESS")
    grab_e = ev(E, "OBJECT_GRABBED_SUCCESS", "rubiks")
    bowl_e = sorted(e["step"] for e in E["events"] if e["name"] == "OBJECT_IN_CONTAINER_SUCCESS")
    assert (grab_u, drop_u, banana_off_u, cube_bowl_u, grab_e, bowl_e) == (73, 76, 196, 351, 78, [119, 255])

    lab_w = 0.52 * K
    fgap = 0.05 * G
    fw = (W - lab_w - 3 * fgap) / 4
    fh = fw * h / w
    head = 0.34 * K
    t_bot, t_h = 0.3 * K, 1.0 * K
    H = head + 2 * fh + fgap + 0.08 * K + t_bot + t_h
    fig = plt.figure(figsize=(W, H))
    top = H - head
    # chips sit on the camera frames, not on the page, so they keep the same colours in both themes
    grey, harm, teal = "#4A4944", "#2E2D2A", "#147D92"
    notes = {(0, 0): ("cube in the gripper", grey), (0, 1): ("cube dropped", harm),
             (0, 2): ("reaching for the banana", grey), (0, 3): ("banana in the bowl", grey),
             (1, 0): ("cube in the gripper", grey), (1, 1): ("cube held", teal),
             (1, 2): ("cube in the bowl", teal), (1, 3): ("banana in: success", teal)}
    fig.text(0.0, (H - 0.02 * K) / H, "Held-out start 90002.  Instruction, unchanged: "
             "“Put the cube then the banana in the bowl”", ha="left", va="top", fontsize=FS, color=C["ink"])
    for r, (name, color) in enumerate((("Unedited", C["mute"]), ("Edited", C["teal"]))):
        y = top - (r + 1) * fh - r * fgap
        fig.text(0.0, (y + fh / 2) / H, name, ha="left", va="center", fontsize=FS, weight="bold", color=color)
        for c in range(4):
            x = lab_w + c * (fw + fgap)
            ax = fig.add_axes([x / W, y / H, fw / W, fh / H])
            ax.imshow(sheet[r * (h + gap):r * (h + gap) + h, c * (w + gap):c * (w + gap) + w], interpolation="lanczos")
            bare(ax)
            note, ncol = notes[(r, c)]
            ax.text(0.03, 0.06, note, transform=ax.transAxes, ha="left", va="bottom", fontsize=FS_SMALL - 0.3 * K,
                    color="#FFFFFF", bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.12", fc=ncol, ec="none",
                                             alpha=0.9))
            if r == 0:
                fig.text(x / W, (top + 0.04 * K) / H, f"{c + 1}", ha="left", va="bottom", fontsize=FS, weight="bold",
                         color=C["ink"])
                fig.text((x + 0.1 * K) / W, (top + 0.04 * K) / H, f"step {moments[c]}", ha="left", va="bottom",
                         fontsize=FS_SMALL, color=C["ink2"])

    # timeline: commanded gripper (closed = filled) and logged events, both rollouts
    t_left, t_right = lab_w, 4.02 * G
    ax = fig.add_axes([t_left / W, t_bot / H, (t_right - t_left) / W, t_h / H])
    xmax = 300
    yU, yE, bh = 1.0, 0.0, 0.3
    ax.add_patch(Rectangle((75, -0.5), 15, 2.12, fc=C["teal_pale"], ec="none", alpha=0.55, zorder=0))
    for y, A, color in ((yU, U, C["mute"]), (yE, E, C["teal"])):
        ax.add_patch(Rectangle((0, y - bh / 2), xmax, bh, fc=C["pale"], ec="none", zorder=0.6))
        for a, b in P.spans(A["cmd_steps"], A["gripper_cmd"]):
            a, b = max(a, 0), min(b, xmax)
            if a < b:
                ax.add_patch(Rectangle((a, y - bh / 2), b - a, bh, fc=color, ec="none", zorder=1))
    for c, s in enumerate(moments):
        ax.plot([s, s], [-0.42, 1.66], color=C["rule"], lw=0.6, ls=(0, (1, 1.6)), zorder=0.8)
        ax.text(s, 1.7, f"{c + 1}", ha="center", va="bottom", fontsize=FS_SMALL, weight="bold", color=C["ink"])
    inside = dict(ha="center", va="center", fontsize=FS_SMALL - 0.5 * K, color=C["on_fill"], zorder=4)
    ax.text((135 + 195) / 2, yU, "banana", **inside)
    ax.text((75 + 118) / 2, yE, "cube", **inside)
    ax.text((191 + 254) / 2, yE, "banana", **inside)
    ax.plot([drop_u], [yU], marker="x", ms=4.4 * K, mew=1.2 * K, color=C["harm"], zorder=5)
    ax.text(drop_u + 5, yU + 0.2, f"cube released at step {drop_u}, {drop_u - grab_u} steps after the grasp",
            ha="left", va="bottom", fontsize=FS_SMALL, color=C["ink"], zorder=5,
            bbox=dict(boxstyle="square,pad=0.05", fc=C["page"], ec="none", alpha=0.85))
    ax.text(128, yU - 0.27, f"banana released at {banana_off_u}; cube placed only at {cube_bowl_u}; fails",
            ha="left", va="top", fontsize=FS_SMALL, color=C["ink2"])
    for s in bowl_e:
        ax.plot([s], [yE], marker="o", ms=3.4 * K, mfc=C["page"], mec=C["teal"], mew=0.9 * K, zorder=5)
    ax.text(grab_e - 1, yE - 0.27, f"cube held from {grab_e} until it is in the bowl ({bowl_e[0]})", ha="left",
            va="top", fontsize=FS_SMALL, color=C["teal"])
    ax.text(bowl_e[1] + 6, yE, f"success at {E['final_step']}", ha="left", va="center", fontsize=FS_SMALL,
            color=C["teal"], weight="bold", zorder=5)
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.75, 1.95)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100, 150, 200, 250, 300])
    plain_axes(ax)
    ax.set_xlabel("Control step at 15 Hz (filled: gripper commanded closed)", fontsize=FS_SMALL, color=C["ink2"],
                  labelpad=3)
    to_fig = lambda yy: (t_bot + t_h * (yy + 0.75) / 2.7) / H  # noqa: E731
    fig.text(0.0, to_fig(yU), "Unedited", ha="left", va="center", fontsize=FS_SMALL, color=C["mute"], weight="bold")
    fig.text(0.0, to_fig(yE), "Edited", ha="left", va="center", fontsize=FS_SMALL, color=C["teal"], weight="bold")

    # inset: held-out success with Wilson 95% intervals
    i_left, i_w = 4.42 * G + 0.18 * K, 1.0 * G - 0.18 * K
    ia = fig.add_axes([i_left / W, (t_bot + 0.1 * K) / H, i_w / W, 0.62 * K / H])
    CTB = P.CTB
    for y, (k, n, lo, hi, color) in enumerate((
            (CTB["base_success"], CTB["paired_n"], *CTB["base_wilson95"], C["mute"]),
            (CTB["cand_success"], CTB["paired_n"], *CTB["cand_wilson95"], C["teal"]))):
        ia.plot([100 * lo, 100 * hi], [y, y], color=color, lw=2.2 * K, solid_capstyle="butt", alpha=0.45)
        ia.plot([100 * k / n], [y], marker="o", ms=3.6 * K, color=color, mec=C["page"], mew=0.6)
        ia.text(100 * k / n, y + 0.3, f"{k}/{n}", ha="center", va="bottom", fontsize=FS_SMALL, color=color,
                weight="bold" if color == C["teal"] else "normal")
    ia.set_xlim(0, 100)
    ia.set_ylim(-0.45, 1.8)
    ia.set_yticks([])
    ia.set_xticks([0, 50, 100])
    ia.set_xticklabels(["0", "50", "100%"])
    plain_axes(ia)
    fig.text(i_left / W, (t_bot + 0.88 * K) / H, "Success, 128 held-out starts", ha="left", va="bottom",
             fontsize=FS_SMALL, color=C["ink"], weight="bold")
    fig.text(i_left / W, (t_bot + 0.76 * K) / H, "with Wilson 95% intervals", ha="left", va="bottom",
             fontsize=FS_SMALL - 0.4 * K, color=C["mute"])
    save(fig, "teaser", theme)


# ---- Fig. 2: what it takes to fix the decision (paper: fig_construction) ------------------------------------------
def fig_construction(theme):
    rows = [("head", "a", "Rewording the instruction, weights unchanged")]
    rows += [("row", lab, "", k, 21, closed, kind) for lab, k, kind, closed in P.REWORDINGS]
    rows += [("head", "b", "Editing the weights, original instruction")]
    rows += [("row", *r) for r in P.CONSTRUCTION]
    rows += [("head", "", "Variants of the distilled edit")]
    rows += [("row", *r) for r in P.VARIANTS]
    heights = [1.45 if kind == "row" and "\n" in r[0] else 1.0 for kind, *r in rows]
    pitch = 0.12 * K
    body = pitch * (sum(heights) + 0.45 * 2 + 0.4)
    lab_l, size_r, bar_w = 0.13 * K, 2.6 * K, 1.22 * G
    bar_l = size_r + 0.06 * K
    cl_l = bar_l + bar_w + 0.12 * K
    Wc = cl_l + 1.95 * K * 0.95
    H = body + 0.36 * K
    fig = plt.figure(figsize=(Wc, H))
    ax = fig.add_axes([bar_l / Wc, 0.31 * K / H, bar_w / Wc, body / H])
    tr_fig = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    y, path, first = 0.0, [], True
    small = FS_SMALL - 0.3 * K
    for (kind, *r), hgt in zip(rows, heights):
        if kind == "head":
            letter, title = r
            y += 0.0 if first else 0.45
            if letter:
                ax.text(0.0, y, letter, ha="left", va="center", fontsize=FS_TITLE, weight="bold", color=C["ink"],
                        transform=tr_fig)
            ax.text(lab_l / Wc, y, title, ha="left", va="center", fontsize=FS_SMALL, weight="bold", color=C["ink"],
                    transform=tr_fig)
            if first:
                ax.text(0, y, "replayed releases that persist", ha="left", va="center", fontsize=small,
                        color=C["mute"])
                ax.text(cl_l / Wc, y, "closed loop: edited (unedited)", ha="left", va="center", fontsize=small,
                        color=C["mute"], transform=tr_fig)
            if letter == "b":
                ax.text(size_r / Wc, y, "size", ha="right", va="center", fontsize=small, color=C["mute"],
                        transform=tr_fig)
            first = False
            y += 1
            continue
        label, size, k, n, closed, cls = r
        y += (hgt - 1) / 2
        indent = 0.13 * K if label.startswith("strength") else 0.0
        ax.text((lab_l + 0.08 * K + indent) / Wc, y, label, ha="left", va="center", fontsize=FS_SMALL,
                color=C["ink"], linespacing=1.05, transform=tr_fig)
        if cls != "group":
            share = 100 * k / n
            ax.add_patch(Rectangle((0, y - 0.3), 100, 0.6, fc=C["pale"], ec="none", zorder=0))
            if share > 0:
                ax.add_patch(Rectangle((0, y - 0.3), share, 0.6, fc=C["teal_pale"] if cls == "edit" else C["light"],
                                       ec=C["teal"] if cls == "edit" else "none", lw=0.8, zorder=1))
            txt = f"{k}/{n}" + (" checked" if n != 21 else "")
            inside = share > 70
            ax.text(share - 2 if inside else share + 2.5, y, txt, ha="right" if inside else "left", va="center",
                    fontsize=FS_SMALL, color=C["teal"] if cls == "edit" else C["ink2"],
                    weight="bold" if k == 0 else "normal", zorder=3)
            if size:
                ax.text(size_r / Wc, y, size, ha="right", va="center", fontsize=small, color=C["mute"],
                        transform=tr_fig)
            if closed:
                ax.text(cl_l / Wc, y, closed, ha="left", va="center", fontsize=FS_SMALL,
                        color=C["teal"] if cls == "edit" else C["ink2"], transform=tr_fig)
            if cls == "fail" or (cls == "edit" and len(path) and path[-1][2] == "fail"):
                path.append((y, share, cls))
        y += 1 + (hgt - 1) / 2
    for (y0, s0, _), (y1, s1, _) in zip(path[:-1], path[1:]):
        ax.plot([s0, s0, s1], [y0 + 0.3, y1 - 0.3, y1 - 0.3], color=C["ink2"], lw=0.8, zorder=2)
    ax.set_xlim(0, 100)
    ax.set_ylim(y - 0.5, -0.6)
    ax.set_yticks([])
    ax.set_xticks([0, 50, 100])
    ax.set_xticklabels(["0", "50", "100%"])
    plain_axes(ax)
    save(fig, "construction", theme)


# ---- Fig. 3: paired outcomes (paper: draw_outcomes) ---------------------------------------------------------------
def fig_outcomes(theme, pitch=0.092 * K, label_w=1.72 * K):
    groups = P.outcome_rows()
    seg = [("broke", C["harm"], C["on_fill"]), ("both succeed", C["both"], C["on_fill"]),
           ("both fail", C["none"], C["ink2"]), ("fixed", C["teal"], C["on_fill"])]
    n_rows = sum(len(g[1]) for g in groups)
    body = pitch * (n_rows + len(groups) * 1.0 + (len(groups) - 1) * 0.35 + 2.6)
    H = body + 0.3 * K
    right = 0.66 * K
    Wo = W
    fig = plt.figure(figsize=(Wo, H))
    bar_l, bar_r = label_w + 0.06 * K, Wo - right
    ax = fig.add_axes([bar_l / Wo, 0.2 * K / H, (bar_r - bar_l) / Wo, body / H])
    tr = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
    tr_fig = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    bw = bar_r - bar_l
    c_pairs, c_fixed, c_broke = 1 + 0.06 * K / bw, 1 + 0.27 * K / bw, 1 + 0.47 * K / bw
    small = FS_SMALL - 0.3 * K
    yk, kx = 0.0, [0.0, 0.17, 0.42, 0.75, 1.0]
    for (name, col, tcol), a, b in zip(seg, kx[:-1], kx[1:]):
        ax.add_patch(Rectangle((a, yk - 0.38), b - a, 0.76, fc=col, ec=C["page"], lw=0.8))
        ax.text((a + b) / 2, yk, name, ha="center", va="center", fontsize=small, color=tcol)
    for a, b, txt in ((kx[0], kx[2], "unedited succeeded"), (kx[2], kx[4], "unedited failed")):
        ax.plot([a + 0.004, a + 0.004, b - 0.004, b - 0.004], [yk - 0.55, yk - 0.75, yk - 0.75, yk - 0.55],
                color=C["mute"], lw=0.7)
        ax.text((a + b) / 2, yk - 0.82, txt, ha="center", va="bottom", fontsize=FS_SMALL - 0.4 * K, color=C["mute"])
    ax.text(-0.012, yk, "Each pair: one start, unedited vs edited", transform=tr, ha="right", va="center",
            fontsize=small, color=C["mute"])
    for x, txt, col in ((c_pairs, "pairs", C["mute"]), (c_fixed, "fixed", C["teal"]), (c_broke, "broke", C["ink"])):
        ax.text(x, yk, txt, ha="left", va="center", fontsize=small, color=col, transform=tr)
    y = 1.6
    for gi, (gname, items) in enumerate(groups):
        y += 0.35 if gi else 0.0
        ax.text(0.0, y, gname, transform=tr_fig, ha="left", va="center", fontsize=FS_SMALL, weight="bold",
                color=C["ink"])
        y += 1.0
        for label, r, kind in items:
            both = r["base"] - r["broke"]
            neither = r["n"] - both - r["fixed"] - r["broke"]
            x = 0.0
            for (_, col, _), k in zip(seg, [r["broke"], both, neither, r["fixed"]]):
                wdt = k / r["n"]
                if wdt > 0:
                    ax.add_patch(Rectangle((x, y - 0.36), wdt, 0.72, fc=col, ec=C["page"], lw=0.6))
                x += wdt
            held = kind == "held"
            ax.text(-0.012, y, label, ha="right", va="center", fontsize=FS_SMALL, transform=tr,
                    color=C["teal"] if held else C["ink"], weight="bold" if held else "normal",
                    style="italic" if kind == "noedit" else "normal")
            ax.text(c_pairs, y, f"{r['n']}", ha="left", va="center", fontsize=FS_SMALL, color=C["mute"], transform=tr)
            ax.text(c_fixed, y, f"+{r['fixed']}" if r["fixed"] else "0", ha="left", va="center", fontsize=FS_SMALL,
                    color=C["teal"] if r["fixed"] else C["mute"], transform=tr)
            ax.text(c_broke, y, f"−{r['broke']}" if r["broke"] else "0", ha="left", va="center", fontsize=FS_SMALL,
                    color=C["ink"] if r["broke"] else C["mute"], transform=tr)
            y += 1
    ax.set_xlim(0, 1)
    ax.set_ylim(y - 0.45, -1.0)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
    ax.set_xticklabels(["0", "", "50", "", "100% of pairs"])
    ax.set_yticks([])
    plain_axes(ax)
    save(fig, "outcomes", theme)


# ---- Fig. 4: detectability map (paper: fig_detectability) ---------------------------------------------------------
def fig_detectability(theme):
    Hd = 2.75
    fig = plt.figure(figsize=(W, Hd))
    left, bottom = 0.62, 0.52
    ax = fig.add_axes([left / W, bottom / Hd, (W - left - 0.06) / W, (Hd - bottom - 0.06) / Hd])
    ns = np.geomspace(12, 330, 300)
    lo, mid, hi = (100 * P.mde(ns, q) for q in (0.2, 0.3, 0.4))
    ax.fill_between(ns, 0, mid, color=C["pale"], lw=0, zorder=0)
    ax.fill_between(ns, lo, hi, color=C["none"], lw=0, zorder=0)
    ax.plot(ns, mid, color=C["mute"], lw=0.9, zorder=1)
    pts = []
    for i, r in enumerate(P.ROWS):
        if r["label"].startswith("10 edits"):
            continue
        d = 100 * (r["cand"] - r["base"]) / r["n"]
        if r["type"] in ("identity", "prompt", "runtime"):
            kind = "noedit"
        elif r["level"] in ("Language binding", "Binding edit on other tasks"):
            kind = "edit"
        else:
            kind = "other"
        pts.append((r, r["n"] * (1 + 0.035 * ((i % 5) - 2)), d, kind))
    pts.append((P.GRIP_POOLED, 240.0, 0.0, "other"))
    for r, x, d, kind in pts:
        held = r.get("type") == "heldout"
        mk = "^" if d > 0 else ("v" if d < 0 else "o")
        if kind == "edit":
            fc, ec, ms = (C["teal"], C["teal"], 5.2) if held else (C["page"], C["teal"], 3.9)
        elif kind == "noedit":
            fc, ec, ms = (C["page"], C["mute"], 3.7)
        else:
            fc, ec, ms = (C["ink2"], C["ink2"], 3.4)
        ax.plot([x], [abs(d)], marker=mk, ms=ms * 1.3, mfc=fc, mec=ec, mew=1.0, ls="", zorder=3 + held)
    ax.set_xscale("log", base=2)
    ax.set_xticks([16, 32, 64, 128, 256])
    ax.set_xticklabels(["16", "32", "64", "128", "256"])
    ax.minorticks_off()
    ax.set_xlim(12, 330)
    ax.set_ylim(0, 74)
    ax.set_yticks([0, 20, 40, 60])
    ax.set_xlabel("Paired episodes per arm", fontsize=FS_SMALL, color=C["ink2"], labelpad=3)
    ax.set_ylabel("|Difference| (points)", fontsize=FS_SMALL, color=C["ink2"], labelpad=3)
    plain_axes(ax, keep=("bottom", "left"))
    xs = 108.0
    ys = 100 * P.mde(xs, 0.4)
    p0 = ax.transData.transform((xs, ys))
    p1 = ax.transData.transform((xs * 1.3, 100 * P.mde(xs * 1.3, 0.4)))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(xs, ys + 2.6, "smallest detectable change, 80% power", rotation=ang, rotation_mode="anchor",
            ha="left", va="bottom", fontsize=FS_SMALL - 0.4 * K, color=C["ink2"])
    lab = dict(fontsize=FS_SMALL, va="center")
    ax.text(150, 53.0, "distilled edit, held-out\n(full and 0.16% variant)", ha="left", color=C["teal"],
            weight="bold", linespacing=1.05, **lab)
    ax.text(36.5, 67.5, "distilled edit, practice seeds", ha="left", color=C["teal"], **lab)
    ax.text(36.5, 59.5, "explicit wording, no edit", ha="left", color=C["mute"], **lab)
    ax.text(70, 44.0, "cube left of bowl, held-out", ha="left", color=C["teal"], **lab)
    ax.text(70, 34.5, "eyeglasses, held-out and practice", ha="left", color=C["teal"], **lab)
    ax.text(70, 24.5, "runtime gripper threshold", ha="left", color=C["mute"], **lab)
    ax.text(18.0, 39.5, "unedited rerun, butter", ha="left", color=C["mute"], **lab)
    ax.text(72, 6.0, "other edits and unchanged reruns", ha="left", color=C["ink2"], **lab)
    save(fig, "detectability", theme)


if __name__ == "__main__":
    for theme in ("light", "dark"):
        use_theme(theme)
        fig_teaser(theme)
        fig_construction(theme)
        fig_outcomes(theme)
        fig_detectability(theme)
