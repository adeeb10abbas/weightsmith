#!/usr/bin/env python3
"""Build the draft cross-model page (site/cosmos/) from the result records. One command refreshes everything:

    python3 site/scripts/make_cosmos_metrics.py            # fetch the three cross-model branches, rebuild all outputs
    python3 site/scripts/make_cosmos_metrics.py --no-fetch # use the remote-tracking refs as they are
    python3 site/scripts/make_cosmos_metrics.py --cosmos a.json --out /tmp/site-test  # test records: a copy of site/

Test records (--cosmos / --groot / --pi05 files) need --out DIR: every output then goes to a fresh copy of site/ there,
never to site/cosmos. The "updated" stamp moves only when a source branch moved and a number the page shows changed,
both against the committed metrics.json (HEAD; numbers_hash); --set-updated resets it by hand.

The page asks where weight-space programming fixes failures in a world-action model (NVIDIA Cosmos3 Edge Policy) and
in a VLA on a reasoning backbone (NVIDIA GR00T N1.7 DROID, Cosmos-Reason2-2B), with OpenPI pi0.5 as the control, on
the shared RoboLab DROID tasks T1-T8 and a retention task R1.

Inputs
  Cross-model results, one file per model (any may be missing; its cells then read "running"):
    origin/cross-cosmos/v1:study/results/cross/metrics_cosmos.json
    origin/cross-groot/v1:study/results/cross/metrics_groot.json
    origin/cross-pi05/v1:study/results/cross/metrics_pi05.json     (fresh pi0.5 runs on the shared slots)
  Each holds one row per experiment: model, task, wording pair, n, unedited k, edited k, fixed, broke, p, CI,
  control k, teacher k, recovery %, transfer, retention, edit location, values changed, notes. Key spelling is free
  ("unedited k", "unedited_k", "uneditedK"); unknown keys are printed and kept. Tasks match T1-T8 and R1 by id
  ("T3"), RoboLab class name or plain name. A row without edited k is a failure-map row; teacher k gives the wording
  gap; "edit task" (or an experiment name such as "T1 edit") marks a side-effect row: the edit built for that task,
  run on this row's task. "headline": true picks a task's main edit (else the first listed). A top-level
  "status": "final" turns that model's empty cells from "running" to "—". Rows name their edit in an edit field or as
  "edit NAME" in the notes (offline records in offline.edit). Follow-up builds of an edit, named like it with one more
  letter ("t8undp", "t8undr" after "t8und"; hyphens ignored), stay out of the main tables and fill the subsection
  "Can the side effect be prevented?" (PREVENT), with their offline gates. Stacked edits, named after two tasks
  ("t8t3und"), stay out of the main tables (stacked()); the headline task (PREVENT["target"]) is the main fix.
  pi0.5's earlier records fill its column until the shared-slot runs land, each labelled with its own slots and n:
    study/results/opus/heldout/*.json              held-out runs: per-slot outcomes, Wilson CIs, exact McNemar p
    study/results/opus/comparisons.json            every paired comparison (noise controls, transfer map)
    study/results/opus/controls/ctl1-90000-64.*    random-direction control and the explicit-wording arm
    study/results/opus/where/results/*.json        where to edit: VLM, action expert, action head; transfer
    study/results/ctrl/summary.json, heldout/      blind editors, one per task: unedited rates on T1-T7
    study/results/sgdw/results/*.json, runs/       T8: mustard right of the raisin box, reference-first wording
    study/results/tutorial/*.json                  the packaged tool (`weightsmith distill`) end to end
    study/blog/failure-levels.md                   the transfer-map row for the flat box (not in the JSON records)

Outputs (all under site/cosmos/)
  metrics.json                 every table of the page, with raw rows, display strings and sources
  Labels: TASK_NAME is the one task-name mapping; dename() turns any task code left in a display string (cells,
  captions, figure text, the brief) into its name, so the codes appear only in the records' task legend.
  index.html, brief.html       generated regions between <!-- gen:NAME --> and <!-- /gen:NAME --> markers. index.html
                               reads as nine short sections (TL;DR from tldr_story; prose with numbers from fix_text,
                               left_text, groot_text, control_text; inline-SVG diagrams dg_*); every table, figure and
                               record sits in one collapsed "All numbers and records" section (<!-- allnums -->), and
                               the remaining clips in "More clips" (<!-- gallery -->); neither counts toward the read time
  fig/<name>[-dark].svg|png    six figures, SVG and PNG at 2x, light and dark; fig/brief-*.svg for the PDF
  weightsmith-cosmos-brief.pdf one-page brief printed from brief.html by headless Chrome

Teacher recovery, (edited - unedited) / (clear wording - unedited), is shown as a % only where the clear wording's own
gain is clearly significant (exact McNemar p < 0.05, teacher_gain_p); elsewhere the cell says in words how the edit
compares with the clear wording (guard_recovery, edit_vs_clear).

Statistics follow the paper (paper/scripts/make_figures.py): exact two-sided McNemar on fixed vs broke pairs, Wilson
95% intervals, and a paired bootstrap 95% interval of the difference (20,000 multinomial draws over the four paired
outcome types, seed 0). Every number is read from a record or computed from a record's counts by these methods
(marked "computed here"); the few values that exist only in prose are in TEXT below, each with its source.
"""
import argparse
import datetime as dt
import hashlib
import html
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from math import comb, sqrt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "site/cosmos"
FIG = OUT / "fig"
REPO = "https://github.com/adeeb10abbas/weightsmith"
CHROME = os.environ.get("CHROME", "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
N_PARAMS = 3_353_433_872  # pi05_droid_jointpos parameters and buffers (BENCHMARK.md section 8; owsp diff)
SHARED_SLOTS = "practice 30000–30031, held-out 96000–96063"

SOURCES = [  # (model id, branch, path)
    ("cosmos", "cross-cosmos/v1", "study/results/cross/metrics_cosmos.json"),
    ("groot", "cross-groot/v1", "study/results/cross/metrics_groot.json"),
    ("pi05", "cross-pi05/v1", "study/results/cross/metrics_pi05.json"),
]
ORDER = ["cosmos", "groot", "pi05"]

MODELS = {
    "cosmos": dict(
        name="Cosmos3 Edge Policy", short="world-action model", kind="world-action model",
        status="running", checkpoint="`nvidia/Cosmos3-Edge-Policy-DROID` at `a7c7288`, 3.37B parameters by the track’s "
                                     "count (the model card lists 4B)",
        architecture="One transformer denoises an action chunk together with future video latents, conditioned on the "
                     "instruction: a mixture of transformers with 28 layers of width 2048, each holding a text pathway "
                     "and a generation pathway for video and action tokens. It sees the wrist and both shoulder cameras "
                     "as one composite image and returns 32 joint-position steps per request, run open loop at 15 Hz.",
        entry="Text tokens attend only to text; video and action tokens attend to the text keys and values in every "
              "layer, so the instruction reaches the actions only through those keys and values (read from the code "
              "that served Cosmos3 Nano).",
        where="text pathway / video-action (generation) pathway",
        evaluation="Closed loop on the shared RoboLab DROID tasks and slots; settings of the Same Goal, Different Words "
                   "study (4 sampling steps, guidance 3).",
    ),
    "groot": dict(
        name="GR00T N1.7", short="reasoning-backbone VLA", kind="VLA on a reasoning backbone",
        status="running", checkpoint="`nvidia/GR00T-N1.7-DROID` at `05e7cc9`, 3.14B parameters",
        architecture="A Cosmos-Reason2-2B vision-language backbone (Qwen3-VL), cut to its first 16 language layers, reads an "
                     "exterior and a wrist camera (180 × 320 each) and the instruction. The action head, 4 self-attention "
                     "layers and 32 diffusion-transformer blocks, turns the backbone's tokens and the robot state into "
                     "40-step chunks by flow matching; the first 8 steps run at 15 Hz.",
        entry="As text tokens in the Cosmos-Reason2 backbone, next to the image tokens (lowercased, punctuation removed). "
              "The action head reads the instruction only through the backbone's tokens.",
        where="Cosmos-Reason2 VLM backbone / flow-matching DiT action head",
        evaluation="Closed loop on the shared RoboLab DROID tasks and slots, NVIDIA's validated client and embodiment "
                   "setting.",
    ),
    "pi05": dict(
        name="π0.5", short="OpenPI VLA, control", kind="VLA with an action expert (OpenPI)",
        status="measured", checkpoint="`pi05_droid_jointpos` (OpenPI), about 3.35 billion parameters",
        architecture="An 18-layer PaliGemma vision-language model (VLM) and an 18-layer Gemma action expert. The action "
                     "expert attends to the VLM's keys and values and produces 15-step action chunks, including a "
                     "gripper command, by flow matching.",
        entry="As text tokens in the VLM, together with image tokens from two cameras and robot-state tokens, under "
              "bidirectional attention, so the image tokens read the instruction. The distilled edit goes into the "
              "VLM's MLP weights.",
        where="VLM backbone / action expert / action output projection",
        evaluation="Closed loop on RoboLab in Isaac Sim: on the shared starts with the default render preset, and, for "
                   "the comparisons not rerun there, the paper’s evaluation (RTX 3090, performance render preset), each "
                   "with its own slots and number of starts. Trained on DROID data and none of these tasks.",
    ),
}

# The shared task set; the last field says how pi0.5 fails there (paper; study/results/ctrl; study/results/sgdw).
TASKS = [
    ("T1", "Cube then banana", "RubiksCubeThenBananaTask", ["cubethenbanana", "rubikscubethenbanana", "ctb"],
     "lets go of the cube early: binds “in the bowl” to the banana"),
    ("T2", "Pick up the eyeglasses", "PickGlassesTask", ["eyeglasses", "pickglasses", "glasses"],
     "grasps one temple arm and lifts too little"),
    ("T3", "Cube left of the bowl", "RubiksCubeLeftOfBowlTask",
     ["cubeleftofbowl", "cubeleftofthebowl", "rubikscubeleftofbowl", "leftofbowl", "leftofthebowl"],
     "carries along a diagonal and leaves the cube behind the bowl"),
    ("T4", "Stack yellow on red", "StackYellowOnRedTask", ["stackyellowonred", "yellowonred", "yellowred"],
     "goes first for a lookalike yellow object"),
    ("T5", "Mustard into the left bin", "MustardInLeftBinTask", ["mustardintoleftbin", "mustardinleftbin",
                                                                 "mustardintotheleftbin", "leftbin"],
     "the bottle slips from a cap grasp mid-carry"),
    ("T6", "Stack three cubes", "Stack3RubiksCubeTask", ["stackthreecubes", "stack3", "threecubes"],
     "builds two cubes, then closes on air above the stack"),
    ("T7", "Pick up the flat box", "PickUpGreenObjectTask", ["flatbox", "pickupgreenobject", "flatgreenbox",
                                                             "greenobject", "greenvegetableblock"],
     "never lifts it: the box is wider than the open gripper"),
    ("T8", "Mustard right of the raisin box, reference-first", "MustardRightOfRaisinRFTask",
     ["mustardrightofraisin", "mustardrightoftheraisin", "s4r", "referencefirst"],
     "takes the wrong object or carries it the wrong way"),
    ("R1", "Retention task", "RubiksCubeTask", ["retention", "cubeintobowl", "cubeintothebowl", "rubikscubetask"],
     "already solved"),
]
TASK = {t[0]: t for t in TASKS}
# The one label mapping: every task name the page, the figures and the brief show. The codes (T1-T8, R1) and RoboLab
# class names appear only in the task legend under "All numbers and records"; dename() turns any code that a record
# or an older string still carries into its name, so a refresh cannot bring the codes back.
TASK_NAME = {"T1": "Cube then banana", "T2": "Eyeglasses", "T3": "Cube left of bowl", "T4": "Yellow on red",
             "T5": "Mustard in left bin", "T6": "Three-cube tower", "T7": "Flat box",
             "T8": "Mustard right of raisin box", "R1": "Cube in bowl"}
# Track A's prospective test of the token-count rule (README, pre-registered in e01cbfe) uses its own codes for new
# wording pairs on stock RoboLab tasks; S1L shares T3's RoboLab class, so these codes never fall back to a class name.
PROSPECT = dict(model="cosmos", commit="e01cbfe", gate=85, heldout_n=32, pairs={   # README: 32 unseen starts
    "S1L": dict(plain="Cube left of bowl, worded bowl-first vs cube-first", equal=True, cls="RubiksCubeLeftOfBowlTask"),
    "S1R": dict(plain="Cube right of bowl, worded bowl-first vs cube-first", equal=True, cls="RubiksCubeRightOfBowlTask"),
    "S1F": dict(plain="Cube in front of bowl, worded bowl-first vs cube-first", equal=False,
                cls="RubiksCubeInFrontOfBowlTask")})
PTASK_NAME = {k: v["plain"] for k, v in PROSPECT["pairs"].items()}
OLD_SHORT = {"T1": "Cube then banana", "T2": "Eyeglasses", "T3": "Cube left of bowl", "T4": "Stack yellow on red",
             "T5": "Mustard into left bin", "T6": "Stack three cubes", "T7": "Flat box",
             "T8": "Mustard, reference-first", "R1": "Retention task"}
SHORT = TASK_NAME
# Mustard right of raisin box: the two wordings of its goal
WORDED = {"rf": "worded raisin-box-first", "tf": "worded mustard-first"}


def tn(tid):
    return TASK_NAME.get(tid, tid)


_RANGE = re.compile(r"(?<![-\w/.])T(\d)–T(\d)(?![-\w]|\.\w)")
_CODE = re.compile(r"(?<![-\w/.])([TR]\d)(?![-\w]|\.\w)")
_PAIRS = sorted({(f"{t} {n}", TASK_NAME[t]) for t in TASK_NAME
                 for n in (OLD_SHORT[t], TASK[t][1], TASK_NAME[t], "Cube into bowl (retention)", "Cube into bowl")
                 if t == "R1" or not n.startswith("Cube into")}, key=lambda x: -len(x[0]))


def dename(s):
    """Task codes to task names, and the T8 wordings' shorthand to plain words, in any display string. Record file
    names ("T8-base-s96000.json") and edit names ("t8und") are left alone."""
    if not isinstance(s, str) or not s:
        return s
    for a, b in _PAIRS:
        s = s.replace(a, b)
    s = _RANGE.sub(lambda m: ", ".join(tn(f"T{i}") for i in range(int(m.group(1)), int(m.group(2)) + 1)), s)
    s = _CODE.sub(lambda m: tn(m.group(1)), s)
    s = re.sub(r"(?<![-\w/.])(S1[LRF])(?![-\w]|\.\w)", lambda m: PTASK_NAME[m.group(1)], s)
    s = re.sub(r"\breference-first\b", "raisin-box-first", s)
    s = re.sub(r"\btarget-first\b", "mustard-first", s)
    s = re.sub(r"\bRF\b", "raisin-box-first", s)
    s = re.sub(r"\bTF\b", "mustard-first", s)
    return s


def dename_deep(x, skip=("key", "id", "src", "sources", "cross_rows", "url", "path", "branch", "commit", "kind",
                         "name", "loc", "s", "left", "target", "base", "model", "let", "cams", "oc", "cols", "gaps")):
    if isinstance(x, dict):
        return {k: (v if k in skip else dename_deep(v, skip)) for k, v in x.items()}
    if isinstance(x, list):
        return [dename_deep(v, skip) for v in x]
    return dename(x) if isinstance(x, str) else x

# Values that exist only in prose, with their source.
TEXT = {
    "combo1_agree": (46, 64),  # paper App. B: two runs of one edited policy (combo1) agreed on 46 of 64 pairs
    "t1_keys": 72,             # paper Section 2: 72 keys per layer (40 prompt/state positions + 32 image cells)
    "offline_kept": "all 9 recorded correct carries and all 19 banana-phase decisions unchanged",  # paper Section 2
    "t4_wording": "no rewording moved the arm in the blind editor's replays, so there was nothing to distil",  # App. F
}
FACTOR = 16_384 + 2_048  # rows + columns of one VLM MLP down-projection (paper Table 4: rank x (16,384 + 2,048))


# ---------------------------------------------------------------- statistics (as in paper/scripts/make_figures.py)
def cells4(n, base, cand, fixed, broke):
    both = base - broke
    neither = n - both - fixed - broke
    if min(both, neither) < 0 or both + fixed != cand:
        raise ValueError(f"inconsistent paired counts n={n} base={base} cand={cand} fixed={fixed} broke={broke}")
    return [both, fixed, broke, neither]


def paired_ci(n, base, cand, fixed, broke, B=20000, seed=0):
    """Paired bootstrap of the success difference, in points (the paper's method)."""
    import numpy as np
    c = np.array(cells4(n, base, cand, fixed, broke), float)
    draws = np.random.default_rng(seed).multinomial(n, c / n, size=B)
    d = (draws[:, 1] - draws[:, 2]) / n
    return (round(100 * (fixed - broke) / n, 1), round(float(100 * np.percentile(d, 2.5)), 1),
            round(float(100 * np.percentile(d, 97.5)), 1))


def mcnemar(f, b):
    m = f + b
    if m == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(m, i) for i in range(min(f, b) + 1)) / 2 ** m)


def teacher_gain_p(a, t, n, explicit=None, wording=None):
    """The exact McNemar p of the clearer wording's own gain over the unedited robot (a -> t of n): from a record that
    gives it, else from the wording screen on the same starts, else exactly when every discordant pair must point one
    way (unedited 0, or the clearer wording solving every start). None when the records cannot say."""
    if explicit is not None:
        return explicit
    if wording and wording.get("s", "ok") == "ok" and (wording.get("a"), wording.get("b"), wording.get("n")) == (a, t, n) \
            and wording.get("pv") is not None:
        return wording["pv"]
    if a == 0 and t > 0:
        return mcnemar(t, 0)
    if t == n and a < n:
        return mcnemar(n - a, 0)
    return None


def edit_vs_clear(b, t, n, p_cmp=None):
    """The edit against the clearer wording, in words: 'the edit did at least as well as the clearer wording (12 vs 8
    of 32; not a clear difference)'."""
    if p_cmp is None:   # exact when every discordant pair must point one way
        p_cmp = (mcnemar(max(b, t), 0) if 0 in (b, t) and b != t else
                 mcnemar(n - min(b, t), 0) if n in (b, t) and b != t else None)
    known = p_cmp is not None
    clear = known and p_cmp < SIG
    note = "" if not known or clear else "; not a clear difference"
    if b == t:
        return f"the edit did as well as the clearer wording ({b} vs {t} of {n})"
    if b >= t:
        return (f"the edit did {'clearly better than' if clear and b > t else 'at least as well as'} the clearer "
                f"wording ({b} vs {t} of {n}" + ("" if clear and b > t else note) + ")")
    return f"the edit did {'clearly ' if clear else ''}less well than the clearer wording ({b} vs {t} of {n}{note})"


def guard_recovery(cell, wording=None, explicit=None, p_cmp=None):
    """A recovery % only where the clearer wording's own gain is clearly significant (p < 0.05); elsewhere the ratio
    of two small differences means little, so the cell says in words how the edit compares with the clearer wording."""
    if not cell or cell.get("s") != "ok" or None in (cell.get("a"), cell.get("b"), cell.get("t"), cell.get("n")):
        return cell
    a, b, t, n = cell["a"], cell["b"], cell["t"], cell["n"]
    pt = teacher_gain_p(a, t, n, explicit, wording)
    if b >= t and t > a:   # a ratio of 100% or more reads oddly: say it in words
        words = edit_vs_clear(b, t, n, p_cmp)
        keep = [x for x in (cell.get("d") or "").split("; ") if x and not x.startswith("as reported")
                and " / (" not in x and "computed here" not in x]
        return dict(cell, v=words[0].upper() + words[1:], words=True,
                    d="; ".join(keep + (["practice starts"] if cell.get("prac") else [])) or None)
    if pt is not None and pt < SIG and t > a:
        return cell
    words = edit_vs_clear(b, t, n, p_cmp)
    return dict(cell, v=words[0].upper() + words[1:], words=True,
                d=f"no recovery %: the clearer wording’s own gain ({a} → {t} of {n}) is not a clear difference"
                  + (f" (p = {fp(pt)})" if pt is not None else "") + ("; practice starts" if cell.get("prac") else ""))


def mde(n, q=0.3, z=1.959964 + 0.841621):
    return 100 * z * sqrt(q / n)


# ---------------------------------------------------------------- formatting
SUP = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
MINUS = "−"


def fp(p):
    """p as the paper prints it: two significant digits, or a.b x 10^-k below 0.001."""
    if p is None:
        return None
    if isinstance(p, str):
        return p
    if p >= 0.9995:
        return "1"
    if p >= 0.001:
        s = f"{p:.2g}"
        return s if "e" not in s else f"{p:.4f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m} × 10{str(int(e)).translate(SUP)}"


def sgn(x, plus=True, nd=1):
    s = f"{abs(x):.{nd}f}"
    if x < 0 and float(s) != 0:
        return MINUS + s
    return ("+" if plus and float(s) != 0 else "") + s


def fci(d, lo, hi):
    return f"{sgn(d)} points [{sgn(lo, False)}, {sgn(hi, False)}]"


def rint(x):
    """Round half up (Python's round() rounds 12.5 to 12)."""
    return int(x + 0.5) if x >= 0 else -int(-x + 0.5)


def pct(k, n):
    return f"{rint(100 * k / n)}%"


def dsgn(d):
    return "±0" if d == 0 else sgn(d, nd=0)


def num(x):
    return f"{x:,}"


def C(v, d=None, s="ok", src=None, **kw):
    """One table cell: value, detail, status (ok | running | na), sources, and numbers the figures use."""
    out = {"v": v, "s": s}
    if d:
        out["d"] = d
    if src:
        out["src"] = sorted(set([src] if isinstance(src, str) else src))
    out.update({k: v for k, v in kw.items() if v is not None})
    return out


RUN = {"v": "running", "s": "running"}
NA = {"v": "—", "s": "na"}


# ---------------------------------------------------------------- reading sources
def git(*args):
    r = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


class Sources:
    def __init__(self):
        self.used = {}

    def text(self, rel):
        p = ROOT / rel
        if p.exists():
            self.used[rel] = "working tree"
            return p.read_text()
        txt = git("show", f"origin/main:{rel}")
        if txt is None:
            raise FileNotFoundError(rel)
        self.used[rel] = "origin/main"
        return txt

    def json(self, rel):
        return json.loads(self.text(rel))


def heldout(S, rel):
    """A heldout.py record; its per-slot outcomes must reproduce its summary."""
    d = S.json(rel)
    s, ps = d.get("summary", d), d["per_slot"]
    b = sum(bool(v[0]) for v in ps.values())
    c = sum(bool(v[1]) for v in ps.values())
    fx = sum((not v[0]) and bool(v[1]) for v in ps.values())
    br = sum(bool(v[0]) and (not v[1]) for v in ps.values())
    assert (len(ps), b, c, fx, br) == (s["paired_n"], s["base_success"], s["cand_success"], s["fixed"],
                                       s["broke"]), rel
    assert abs(mcnemar(fx, br) - s["mcnemar_exact_p"]) <= 1e-9 * s["mcnemar_exact_p"] + 1e-300, rel
    r = dict(n=s["paired_n"], base=b, cand=c, fixed=fx, broke=br, p=s["mcnemar_exact_p"], slots=s["slots"],
             fingerprint=s.get("fingerprint"), task=s.get("task"), src=rel)
    r["ci"] = paired_ci(r["n"], r["base"], r["cand"], r["fixed"], r["broke"])
    return r


def by_cand_ref(analysis, cand, ref):
    hits = [c for c in analysis["comparisons"] if c["cand"] == cand and c["ref"] == ref]
    assert len(hits) == 1, (cand, ref)
    return hits[0]


def slots_txt(sl, kind="held-out"):
    return f"{kind} slots {sl[0]}–{sl[1]}"


# ---------------------------------------------------------------- pi0.5: earlier records
def build_pi05(S):
    P = {}
    HO = "study/results/opus/heldout/"
    P["edit"] = heldout(S, HO + "cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json")
    P["v016"] = heldout(S, HO + "cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json")
    P["eye"] = heldout(S, HO + "thr-pickglasses__ctb-v2mi-PickGlassesTask-90000-64.json")
    P["left"] = heldout(S, HO + "tx-rubikscubeleftofbowl__ctb-v2mi-RubiksCubeLeftOfBowlTask-90000-64.json")
    P["tool"] = heldout(S, "study/results/tutorial/tutorial-ctb__ctb-first-95008-64.results.json")
    assert P["tool"]["fingerprint"] == "3047901c3502ffea" and P["tool"]["slots"] == [95008, 95071]

    # blind editors (one per task): unedited rates on 32 development starts, and the held-out score of each submission
    ctrl = S.json("study/results/ctrl/summary.json")
    key_of = {"T1": "ctb", "T2": "glasses", "T3": "left", "T4": "yellowred", "T5": "mustard", "T6": "stack3",
              "T7": "flatbox"}
    P["dev"], P["blind"], P["blind_dev"] = {}, {}, {}
    for tid, k in key_of.items():
        v = ctrl[k]
        assert v["task"] == TASK[tid][2], (tid, v["task"])
        P["dev"][tid] = dict(k=v["dev"]["base"]["success"], n=v["dev"]["base"]["n"], src="study/results/ctrl/summary.json")
        if v.get("heldout"):
            h = v["heldout"][-1]
            rel = f"study/results/ctrl/heldout/ctrl-{k}__{h['cand']}-{v['task']}-{h['slots'][0]}-{h['paired_n']}.json"
            P["blind"][tid] = heldout(S, rel)
            P["blind"][tid]["values"] = v["dev"][h["cand"]]["size"]["changed_elements"] if h["cand"] in v["dev"] else None
        cands = [c for c in v["dev"] if c != "base"]
        if cands:
            d = v["dev"][cands[0]]
            P["blind_dev"][tid] = dict(base=d["paired_base"], cand=d["success"], n=d["paired_n"], fixed=d["fixed"],
                                       broke=d["broke"], values=d["size"]["changed_elements"])
    P["slots_dev"] = (70000, 70031)  # study/results/ctrl/README.md: development slots 70000-70031

    # random-direction control and explicit-wording (teacher) arm on held-out slots 90000-90063
    ctl_rel = "study/results/opus/controls/ctl1-90000-64.analysis.json"
    ctl = S.json(ctl_rel)
    res = S.json("study/results/opus/controls/ctl1-90000-64.results.json")
    for arm in ("rand", "teacher"):
        assert sum(v["success"] for v in res["per_arm"][arm].values() if v.get("ok")) == ctl["success"][arm]
    sc = ctl["success"]
    P["ctl"] = dict(n=ctl["n_slots"], slots=ctl["slot_range"], base=sc["base_A"], base2=sc["base_B"],
                    edit=sc["edit_full"], rand=sc["rand"], teacher=sc["teacher"],
                    rand_vs_base=by_cand_ref(ctl, "rand", "base_A"), teacher_vs_base=by_cand_ref(ctl, "teacher", "base_A"),
                    teacher_vs_edit=by_cand_ref(ctl, "teacher", "edit_full"),
                    base2_vs_base=by_cand_ref(ctl, "base_B", "base_A"), src=ctl_rel,
                    instruction=res["plan"]["default_instruction"], teacher_prompt=res["plan"]["arms"]["teacher"]["prompt"])
    per = S.json(HO + "cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json")["per_slot"]
    first64 = [v for k, v in per.items() if int(k) < 90064]
    assert sum(v[1] for v in first64) == sc["edit_full"] and sum(v[0] for v in first64) == sc["base_A"]

    comps = S.json("study/results/opus/comparisons.json")["rows"]

    def row(task, label):
        hits = [r for r in comps if r["task"] == task and r["label"].startswith(label)]
        assert len(hits) == 1, (task, label)
        return hits[0]

    rnd = row("Cube then banana", "random directions")
    assert (rnd["base"], rnd["cand"], rnd["fixed"], rnd["broke"]) == (
        P["ctl"]["base"], P["ctl"]["rand"], P["ctl"]["rand_vs_base"]["fixed"], P["ctl"]["rand_vs_base"]["broke"])
    P["ctl"]["rand_values"] = rnd["values_changed"]
    P["edit"]["values"] = row("Cube then banana", "v2mi-f: binding edit, HELD-OUT")["values_changed"]
    P["v016"]["values"] = row("Cube then banana", "minimal edit")["values_changed"]
    P["t4_edits"] = [row("Stack yellow on red", lab) for lab in ("v1a1", "dkv2", "combo1: dkv2")]
    P["t7_edits"] = row("Pick up flat green box", "10 edits")
    P["butter_rerun"] = row("Butter box onto raisin box", "this work: unedited rerun")
    P["cube16"] = row("Cube into bowl", "binding edit")
    P["banana16"] = row("Banana onto plate", "binding edit")
    # the T1 edit, unchanged, on the other shared tasks (paper Table 6; the flat box only in failure-levels.md)
    P["tx"] = {"T4": dict(row("Stack yellow on red", "binding edit"), how="compared with an earlier unedited run"),
               "T5": dict(row("Mustard into bin", "binding edit"), how="compared with an earlier unedited run"),
               "T6": dict(row("Stack three cubes", "binding edit"), how="unedited and edited back to back")}
    fl = S.text("study/blog/failure-levels.md")
    m = re.search(r"\|\s*Pick up the flat green box\s*\|\s*(\d+)/(\d+)\s*\|\s*(\d+)/(\d+)\s*\|\s*(\d+)\s*/\s*(\d+)\s*\|", fl)
    assert m, "flat-box row of the transfer map"
    P["tx"]["T7"] = dict(base=int(m[1]), n=int(m[2]), cand=int(m[3]), fixed=int(m[5]), broke=int(m[6]),
                         src="study/blog/failure-levels.md", how="unedited and edited back to back")
    for r in P["tx"].values():
        r.setdefault("src", "study/results/opus/comparisons.json")
    others = {r["task"] for r in comps if r["level"] == "Binding edit on other tasks"}
    P["n_other_tasks"] = len(others) + 1  # 14 tasks in the JSON + the flat box (failure-levels.md) = 15 other tasks
    assert P["n_other_tasks"] == 15
    grip = [r for r in comps if r["label"].startswith("gthr04") and r["n"] >= 32]
    P["grip_flip"] = (sum(r["fixed"] + r["broke"] for r in grip), sum(r["n"] for r in grip))

    # where to edit (held-out slots 94016-94079) and its transfer / retention runs
    W = "study/results/opus/where/results/"
    wa = S.json(W + "heldout_analysis.json")
    wres = S.json(W + "heldout_results.json")["per_arm"]
    for arm, d in wres.items():
        assert sum(bool(v.get("success")) for v in d.values()) == wa["arms"][arm]["success"], arm
    P["where"] = dict(n=wa["n_slots"], slots=wa["slot_range"], arms={k: v["success"] for k, v in wa["arms"].items()},
                      cmp={c["cand"]: c for c in wa["comparisons"] if c["ref"] == "base"},
                      vs_full={c["cand"]: c for c in wa["comparisons"] if c["ref"] == "v1full"},
                      values={k: S.json(W + f"diff_{f}.json")["changed_elements"] for k, f in
                              (("a1", "a1"), ("h1", "h1"), ("p1", "p1-blind"), ("va", "va"), ("v1img816", "v1-img816"))},
                      sizes=S.json(W + "size_summary.json"), src=W + "heldout_analysis.json")
    P["where"]["n_layers_full"] = len(S.json(W + "diff_v1-full.json")["edited_tensors"])
    tx = {}
    for t in ("glasses", "left", "cube"):
        a = S.json(W + f"transfer_{t}_analysis.json")
        tx[t] = dict(n=a["n_slots"], base=a["arms"]["base"]["success"], arms={k: v["success"] for k, v in a["arms"].items()},
                     cmp={c["cand"]: c for c in a["comparisons"] if c["ref"] == "base"}, slots=a["slot_range"],
                     src=W + f"transfer_{t}_analysis.json")
    P["where"]["tx"] = tx

    # T8: "Same Goal, Different Words" goal S4-R, reference-first wording
    G = "study/results/sgdw/results/"
    s4 = S.json(G + "screen_s4r_tf_vs_rf.json")
    pairs = S.json(G + "heldout_pairs_92000-64.json")
    prac_rf = S.json(G + "practice_rf_grp3_vs_base.json")
    prac_tf = S.json(G + "practice_tf_grp3_vs_base.json")
    # in the screens arm a is the target-first (TF) wording and arm b the reference-first (RF) one
    P["s4"] = dict(n=s4["paired_n"], rf=s4["success_b"], tf=s4["success_a"], tf_only=s4["broke"], rf_only=s4["fixed"],
                   p=s4["mcnemar_exact_p"], ci=[-x for x in s4["diff_boot95_pp"][::-1]], diff=-s4["diff_pp"],
                   slots=s4["slots"], tf_prompt=s4["prompt_a"], rf_prompt=s4["prompt_b"], src=G + "screen_s4r_tf_vs_rf.json")
    g3, rd = pairs["pairs"]["g3 vs base"], pairs["pairs"]["rand3 vs base"]
    P["s4ho"] = dict(n=g3["paired_n"], base=g3["success_a"], cand=g3["success_b"], fixed=g3["fixed"], broke=g3["broke"],
                     p=g3["mcnemar_exact_p"], ci=(g3["diff_pp"], *g3["diff_boot95_pp"]), slots=g3["slots"],
                     rand=rd["success_b"], src=G + "heldout_pairs_92000-64.json")
    assert P["s4ho"]["ci"] == paired_ci(P["s4ho"]["n"], P["s4ho"]["base"], P["s4ho"]["cand"], P["s4ho"]["fixed"],
                                        P["s4ho"]["broke"])
    P["s4prac"] = dict(n=prac_rf["paired_n"], base=prac_rf["success_a"], cand=prac_rf["success_b"],
                       tf=prac_tf["success_a"], tf_edit=prac_tf["success_b"], tf_fixed=prac_tf["fixed"],
                       tf_broke=prac_tf["broke"], tf_p=prac_tf["mcnemar_exact_p"], slots=prac_rf["slots"],
                       src=[G + "practice_rf_grp3_vs_base.json", G + "practice_tf_grp3_vs_base.json"])
    assert P["s4prac"]["tf"] == P["s4"]["tf"] and P["s4prac"]["base"] == P["s4"]["rf"]  # same practice slots
    b3 = S.json("study/results/sgdw/runs/s4r/build_grp3.json")
    keys = [int(k) for k in re.findall(r'"keys":\s*(\d+)', S.text("study/results/sgdw/runs/s4r/build_grp3.report.txt"))]
    assert len(keys) == len(b3["layers"]) == 9
    P["s4ho"]["rank"], P["s4ho"]["layers"] = max(keys), (min(b3["layers"]), max(b3["layers"]))
    P["s4ho"]["factors"] = sum(k * FACTOR for k in keys)
    return P


def task_label(tid):
    return tn(tid)


def pi05_cells(P):
    """pi0.5's earlier records: {table id: {row key: cell}} and row labels; every cell says its slots and n."""
    e, c, w, tx, s4, s4ho, sp = P["edit"], P["ctl"], P["where"], P["where"]["tx"], P["s4"], P["s4ho"], P["s4prac"]
    T = {k: {} for k in TABLE_IDS}
    L = {}

    # failure map: unedited success on each shared task
    for tid, name, cls, _, how in TASKS:
        if tid in P["dev"]:
            d = P["dev"][tid]
            T["failure_map"][tid] = C(f"{d['k']} of {d['n']}", f"{pct(d['k'], d['n'])}; {how}; development slots "
                                      f"{P['slots_dev'][0]}–{P['slots_dev'][1]}", src=d["src"], k=d["k"], n=d["n"])
        elif tid == "T8":
            T["failure_map"][tid] = C(f"{s4['rf']} of {s4['n']}", f"{pct(s4['rf'], s4['n'])}; {how}; "
                                      f"{slots_txt(s4['slots'], 'practice')}", src=s4["src"], k=s4["rf"], n=s4["n"])
        else:
            cb = tx["cube"]
            T["failure_map"][tid] = C(f"{cb['base']} of {cb['n']}", f"{pct(cb['base'], cb['n'])}; cube into the bowl, "
                                      f"{how}; slots {cb['slots'][0]}–{cb['slots'][1]}", src=cb["src"], k=cb["base"], n=cb["n"])

    # wording sensitivity: unedited, original vs clear wording, same starts
    tv = c["teacher_vs_base"]
    T["wording"]["T1"] = C(f"{c['base']} → {c['teacher']} of {c['n']}",
                           f"{tv['fixed']} / {tv['broke']}, p = {fp(tv['mcnemar_p'])}; {fci(tv['diff_pp'], *tv['boot95_pp'])}; "
                           f"{slots_txt(c['slots'])}; “{c['instruction']}” → “{c['teacher_prompt']}”", src=c["src"],
                           a=c["base"], b=c["teacher"], n=c["n"], pv=tv["mcnemar_p"], fl=f"{SHORT['T1']} ({c['n']} unseen)")
    T["wording"]["T8"] = C(f"{s4['rf']} → {s4['tf']} of {s4['n']}",
                           f"{s4['tf_only']} / {s4['rf_only']}, p = {fp(s4['p'])}; {fci(s4['diff'], *s4['ci'])}; "
                           f"{slots_txt(s4['slots'], 'practice')}; reference-first → target-first: “{s4['rf_prompt']}” → "
                           f"“{s4['tf_prompt']}”", src=s4["src"], a=s4["rf"], b=s4["tf"], n=s4["n"], pv=s4["p"],
                           fl=f"{SHORT['T8']} ({s4['n']} practice)")
    T["wording"]["T4"] = C("no clear wording found", TEXT["t4_wording"])

    # edits on each task's own failure, unseen starts
    bl = P["blind"]

    def ho_cell(r, what, color="teal", fl=None, fb=None):
        return C(f"{r['base']} → {r['cand']} of {r['n']}",
                 f"{what}; {r['fixed']} / {r['broke']}, p = {fp(r['p'])}; {fci(*r['ci'])}; {slots_txt(r['slots'])}",
                 src=r["src"], a=r["base"], b=r["cand"], n=r["n"], ci=list(r["ci"]), color=color, fl=fl, fb=fb,
                 fx=r["fixed"], br=r["broke"], pv=r["p"])

    t4, t7, bd3 = P["t4_edits"], P["t7_edits"], P["blind_dev"]["T3"]
    G = T["edit_gain"]
    G["T1"] = ho_cell(e, "distilled edit, operator-built", fl="T1 Cube then banana", fb="T1 distilled")
    G["T1b"] = ho_cell(P["tool"], "decisions chosen by rule", fl="T1, packaged tool, end to end", fb="T1 packaged tool")
    G["T1c"] = ho_cell(bl["T1"], "gated action-head change, 1,024 values", "dark", fl="T1, blind editor's own edit",
                       fb="T1 blind editor")
    G["T2"] = ho_cell(bl["T2"], "blind editor's edit; not significant", "dark", fl="T2 Eyeglasses, blind editor",
                      fb="T2 blind editor")
    G["T3"] = C("no edit submitted", f"the blind editor's candidate scored {bd3['base']} → {bd3['cand']} of {bd3['n']} on "
                "practice starts and broke grasping", src="study/results/ctrl/summary.json")
    G["T4"] = C("no gain", f"three edits, {t4[0]['base']} → {t4[0]['cand']} of {t4[0]['n']} each on practice starts; no "
                "clear wording to distil", src="study/results/opus/comparisons.json")
    G["T5"] = ho_cell(bl["T5"], "blind editor's edit; not significant", "dark", fl="T5 Mustard into left bin, blind editor",
                      fb="T5 blind editor")
    G["T6"] = ho_cell(bl["T6"], "blind editor's edit; not significant", "dark", fl="T6 Stack three cubes, blind editor",
                      fb="T6 blind editor")
    n_t7 = re.match(r"(\d+) edits", t7["label"])[1]
    G["T7"] = C("no gain", f"unedited 0 of 64; {n_t7} edits, none succeeded (a skill the policy lacks)",
                src="study/results/opus/comparisons.json")
    G["T8"] = ho_cell(dict(s4ho), "distilled edit from the target-first wording", fl="T8 Mustard, reference-first",
                      fb="T8 distilled")
    L["T1b"] = "T1, built end to end by the packaged tool (`weightsmith distill`)"
    L["T1c"] = "T1, a blind autonomous editor's own edit"

    rv = c["rand_vs_base"]
    T["random_control"]["T1"] = C(
        f"control {c['rand']}, unedited {c['base']}, edit {c['edit']} of {c['n']}",
        f"control vs unedited {rv['fixed']} / {rv['broke']}, p = {fp(rv['mcnemar_p'])}; values changed "
        f"{num(c['rand_values'])} (control) vs {num(e['values'])} (edit); {slots_txt(c['slots'])}",
        src=[c["src"], "study/results/opus/comparisons.json"], u=c["base"], r=c["rand"], e=c["edit"], n=c["n"],
        t=c["teacher"], fl=f"{SHORT['T1']} ({c['n']} unseen)")
    T["random_control"]["T8"] = C(
        f"control {s4ho['rand']}, unedited {s4ho['base']}, edit {s4ho['cand']} of {s4ho['n']}", slots_txt(s4ho["slots"]),
        src=s4ho["src"], u=s4ho["base"], r=s4ho["rand"], e=s4ho["cand"], n=s4ho["n"], fl=f"{SHORT['T8']} ({s4ho['n']} unseen)")

    tve = c["teacher_vs_edit"]
    rec1 = 100 * (c["edit"] - c["base"]) / (c["teacher"] - c["base"])
    rec8 = 100 * (sp["cand"] - sp["base"]) / (sp["tf"] - sp["base"])
    T["recovery"]["T1"] = C(f"{rec1:.0f}%", f"({c['edit']} − {c['base']}) / ({c['teacher']} − {c['base']}) of {c['n']}, "
                            f"computed here; edit vs teacher {tve['broke']} / {tve['fixed']}, p = {fp(tve['mcnemar_p'])}; "
                            f"{slots_txt(c['slots'])}", src=c["src"], a=c["base"], b=c["edit"], t=c["teacher"], n=c["n"])
    T["recovery"]["T8"] = C(f"{rec8:.0f}%", f"({sp['cand']} − {sp['base']}) / ({sp['tf']} − {sp['base']}) of {sp['n']}, "
                            f"{slots_txt(sp['slots'], 'practice')}, which were used to build the edit; the teacher was not "
                            "run on unseen starts", src=sp["src"], a=sp["base"], b=sp["cand"], t=sp["tf"], n=sp["n"])
    T["recovery"]["T1"] = guard_recovery(T["recovery"]["T1"], T["wording"].get("T1"),
                                         p_cmp=tve.get("mcnemar_p"))
    T["recovery"]["T8"] = guard_recovery(T["recovery"]["T8"], T["wording"].get("T8"))

    cm = w["cmp"]
    W_ = T["where"]
    W_["starts"] = C(f"T1, {w['n']} unseen starts", slots_txt(w["slots"]), src=w["src"])
    W_["unedited"] = C(f"{w['arms']['base']} of {w['n']}", src=w["src"])
    W_["backbone"] = C(f"VLM MLPs: {w['arms']['v1full']} of {w['n']}",
                       f"{cm['v1full']['fixed']} / {cm['v1full']['broke']}, p = {fp(cm['v1full']['mcnemar_p'])}; the VLM's "
                       "weights also process the image tokens, where this edit is keyed", src=w["src"])
    W_["action"] = C(f"action expert {w['arms']['a1']}, action output projection {w['arms']['h1']} of {w['n']}",
                     f"expert {cm['a1']['fixed']} / {cm['a1']['broke']}, p = {fp(cm['a1']['mcnemar_p'])}; output projection "
                     f"{cm['h1']['fixed']} / {cm['h1']['broke']}; an output-projection edit with a hand-built gate (blind "
                     f"editor): "
                     f"{w['arms']['p1']}", src=w["src"])
    W_["both"] = C(f"VLM + action expert: {w['arms']['va']} of {w['n']}",
                   f"below the VLM edit alone, p = {fp(w['vs_full']['va']['mcnemar_p'])}", src=w["src"])
    W_["transfers"] = C("only the backbone edit",
                        f"eyeglasses {tx['glasses']['base']} → {tx['glasses']['arms']['v1full']}, cube left of the bowl "
                        f"{tx['left']['base']} → {tx['left']['arms']['v1full']} of {tx['left']['n']}; action expert "
                        f"{tx['glasses']['arms']['a1']} and {tx['left']['arms']['a1']}, gated head {tx['glasses']['arms']['p1']} "
                        f"and {tx['left']['arms']['p1']} (all p ≥ "
                        f"{fp(min(tx[k]['cmp'][a]['mcnemar_p'] for k in ('glasses', 'left') for a in ('a1', 'p1')))}); "
                        f"slots {tx['left']['slots'][0]}–{tx['left']['slots'][1]}", src=[tx["glasses"]["src"], tx["left"]["src"]])
    W_["size"] = C(f"VLM {100 * e['values'] / N_PARAMS:.1f}%, expert {100 * w['values']['a1'] / N_PARAMS:.1f}% of values; "
                   f"head {num(w['values']['h1'])}", f"median change per layer: VLM {100 * w['sizes']['v1-full']['median']:.1f}% "
                   f"of the weight norm, action expert {100 * w['sizes']['a1']['median']:.0f}%",
                   src=["study/results/opus/where/results/size_summary.json", "study/results/opus/comparisons.json"])

    Z = T["size"]
    Z["T1"] = C(f"{num(e['values'])} values ({100 * e['values'] / N_PARAMS:.1f}%)",
                f"VLM MLP down-projections, layers 0–16, rank ≤ {TEXT['t1_keys']} per layer; as factors "
                f"{TEXT['t1_keys'] * FACTOR * w['n_layers_full'] / 1e6:.1f}M "
                f"({100 * TEXT['t1_keys'] * FACTOR * w['n_layers_full'] / N_PARAMS:.2f}%); median change "
                f"{100 * w['sizes']['v1-full']['median']:.1f}% of each layer's norm", src="study/results/opus/comparisons.json")
    Z["T1v"] = C(f"{32 * FACTOR * 9 / 1e6:.1f}M as factors ({100 * 32 * FACTOR * 9 / N_PARAMS:.2f}%)",
                 f"image cells, layers 8–16, rank ≤ 32 per layer; {P['v016']['base']} → {P['v016']['cand']} of "
                 f"{P['v016']['n']} unseen starts; stored values changed {num(P['v016']['values'])}; the packaged tool's "
                 "edit uses this recipe", src=P["v016"]["src"])
    Z["T1c"] = C(f"{num(P['blind']['T1']['values'])} values", "the blind editor's gated action-head change",
                 src="study/results/ctrl/summary.json")
    Z["T8"] = C(f"{s4ho['factors'] / 1e6:.1f}M as factors ({100 * s4ho['factors'] / N_PARAMS:.2f}%)",
                f"VLM layers {s4ho['layers'][0]}–{s4ho['layers'][1]}, rank ≤ {s4ho['rank']} per layer, image cells; three "
                "decision phases as separate key groups", src="study/results/sgdw/runs/s4r/build_grp3.json")
    L["T1v"] = "T1, smaller variant (image cells, layers 8–16)"
    L["T1c_size"] = "T1, smallest edit that passed unseen starts"

    # side-effect map: the T1 edit on every suite task, paired on the same slots
    X = T["side_effects"]

    def fx_cell(r, how, target=False):
        d = r["cand"] - r["base"]
        p = r.get("p") if r.get("p") is not None else mcnemar(r["fixed"], r["broke"])
        return C(f"{dsgn(d)}: {r['base']} → {r['cand']} of {r['n']}", f"{r['fixed']} / {r['broke']}, p = {fp(p)}; {how}",
                 src=r["src"], delta=d, a=r["base"], b=r["cand"], n=r["n"], p=p, target=target or None)

    X["T1"] = fx_cell(e, f"the target task; {slots_txt(e['slots'])}", target=True)
    X["T2"] = fx_cell(P["eye"], slots_txt(P["eye"]["slots"]))
    X["T3"] = fx_cell(P["left"], slots_txt(P["left"]["slots"]))
    for tid in ("T4", "T5", "T6", "T7"):
        r = P["tx"][tid]
        X[tid] = fx_cell(r, f"{r['n']} practice starts, {r['how']}")
    X["T8"] = C("not run", "the T1 edit was not run on T8, which has its own edit")
    cb = tx["cube"]
    X["R1"] = fx_cell(dict(n=cb["n"], base=cb["base"], cand=cb["arms"]["v1full"], fixed=cb["cmp"]["v1full"]["fixed"],
                           broke=cb["cmp"]["v1full"]["broke"], p=cb["cmp"]["v1full"]["mcnemar_p"], src=cb["src"]),
                      f"cube into the bowl; slots {cb['slots'][0]}–{cb['slots'][1]}")
    L["side_effects_edit"] = "the T1 edit (distilled, VLM)"

    T["retention"]["R1"] = C(f"{cb['base']} → {cb['arms']['v1full']} of {cb['n']}",
                             f"cube into the bowl, the T1 edit; slots {cb['slots'][0]}–{cb['slots'][1]}; also "
                             f"{P['cube16']['base']} → {P['cube16']['cand']} of {P['cube16']['n']}, and banana onto plate "
                             f"{P['banana16']['base']} → {P['banana16']['cand']} of {P['banana16']['n']}",
                             src=[cb["src"], "study/results/opus/comparisons.json"])
    T["retention"]["clear"] = C(f"{sp['tf']} → {sp['tf_edit']} of {sp['n']}",
                                f"T8 under the target-first wording, with the T8 edit; {sp['tf_fixed']} / {sp['tf_broke']}, "
                                f"p = {fp(sp['tf_p'])}: not detectable at {sp['n']} pairs, not excluded; "
                                f"{slots_txt(sp['slots'], 'practice')}", src=sp["src"])
    T["retention"]["offline"] = C("unchanged", TEXT["offline_kept"] + " (T1 edit, offline replay)")
    L["clear"] = "The clear wording, after the edit"
    L["offline"] = "Recorded decisions it must not change"

    b2, ag, br = c["base2_vs_base"], TEXT["combo1_agree"], P["butter_rerun"]
    N = T["noise"]
    N["unedited2"] = C(f"{c['base']} vs {c['base2']} of {c['n']}",
                       f"{b2['fixed'] + b2['broke']} of {c['n']} pairs differ ({pct(b2['fixed'] + b2['broke'], c['n'])}); "
                       f"T1, {slots_txt(c['slots'])}", src=c["src"])
    N["edited2"] = C(f"agreed on {ag[0]} of {ag[1]}", f"{pct(ag[1] - ag[0], ag[1])} of pairs differ (paper, Appendix B)")
    N["butter"] = C(f"{br['base']} vs {br['cand']} of {br['n']}", f"{pct(br['fixed'] + br['broke'], br['n'])} of pairs "
                    f"differ; exact p = {fp(mcnemar(br['fixed'], br['broke']))} with nothing changed",
                    src="study/results/opus/comparisons.json")
    N["typical"] = C("about 30%", f"a near-neutral one-weight edit: {P['grip_flip'][0]} of {P['grip_flip'][1]}",
                     src="study/results/opus/comparisons.json")
    N["mde"] = C(" / ".join(f"{mde(n):.0f}" for n in (32, 64, 128, 256)) + " points",
                 "at 32 / 64 / 128 / 256 pairs, 80% power, α = 0.05, 30% of pairs differing")
    L.update(unedited2="Unedited, run twice on the same starts", edited2="Identical edited weights, run twice",
             butter="Unedited rerun, butter on the raisin box", typical="Typical share of pairs that differ",
             mde="Smallest detectable change", starts="Task, starts", unedited="Unedited",
             backbone="Backbone / text pathway", action="Action pathway", both="Both", transfers="Which one transfers",
             offline_where="Offline check: share of the wording’s action change reproduced",
             size="Size of each")
    return T, L


# ---------------------------------------------------------------- cross-model rows: tolerant reading
ALIASES = {
    "model": ["model", "modelname"],
    "checkpoint": ["checkpoint", "ckpt", "revision", "weights"],
    "task": ["task", "taskname", "goal"],
    "code": ["code", "taskcode", "taskid"],
    "edit_task": ["edittask", "editfor", "builtfor", "sourcetask", "edittarget", "targettask", "fromtask", "editof"],
    "wording_pair": ["wordingpair", "wordings", "wording", "prompts", "instructionpair", "instructions"],
    "n": ["n", "pairedn", "nstarts", "starts", "nepisodes", "episodes", "nitems", "items"],
    "unedited_k": ["uneditedk", "unedited", "basek", "base", "uneditedsuccess", "basesuccess", "uneditedsuccesses"],
    "edited_k": ["editedk", "edited", "candk", "cand", "editedsuccess", "candsuccess", "editedsuccesses"],
    "fixed": ["fixed", "nfixed"],
    "broke": ["broke", "broken", "nbroke", "nbroken"],
    "p": ["p", "pvalue", "mcnemarp", "mcnemarexactp", "exactp", "mcnemar"],
    "ci": ["ci", "ci95", "bootci", "boot95", "boot95pp", "diffci", "ci95pp", "bootstrapci", "diffboot95pp"],
    "control_k": ["controlk", "control", "randomk", "randk", "randomcontrolk", "randomcontrol", "controlsuccess"],
    "teacher_k": ["teacherk", "teacher", "clearwordingk", "clearwording", "explicitk", "teachersuccess"],
    "recovery_pct": ["recovery", "recoverypct", "recoverypercent", "teacherrecovery", "recoverypc"],
    "transfer": ["transfer", "transfers", "sideeffects", "sideeffect"],
    "retention": ["retention", "retained"],
    "edit_location": ["editlocation", "location", "where", "editsite", "site"],
    "values_changed": ["valueschanged", "values", "nvalues", "editsize", "size", "changedvalues", "changedelements"],
    "notes": ["notes", "note", "comment", "comments", "caveats"],
    "split": ["split", "seeds", "starttype", "slotset"],
    "slots": ["slots", "slotrange"],
    "headline": ["headline", "primary", "main"],
    "rerun_k": ["rerunk", "rerun", "identityk", "noisek", "unedited2k", "uneditedrerunk"],
    "experiment": ["experiment", "name", "id", "label", "arm", "edit", "candidate", "track"],
}
KEYMAP = {a: k for k, al in ALIASES.items() for a in al}


def nk(s):
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


def as_int(x):
    if isinstance(x, dict):
        return as_int(x.get("k", x.get("success")))
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return int(x) if float(x).is_integer() else None
    m = re.fullmatch(r"\s*(\d+)\s*(?:/\s*(\d+))?\s*", str(x))
    return int(m.group(1)) if m else None


def frac_n(x):
    m = re.fullmatch(r"\s*\d+\s*/\s*(\d+)\s*", str(x)) if isinstance(x, str) else None
    return int(m.group(1)) if m else None


def as_float(x):
    if isinstance(x, bool) or x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).replace("−", "-").replace("×", "x").replace(" ", "").translate(str.maketrans("⁻⁰¹²³⁴⁵⁶⁷⁸⁹", "-0123456789"))
    m = re.fullmatch(r"([-+]?\d*\.?\d+)(?:e([-+]?\d+)|x10\^?([-+]?\d+))?%?", s)
    if not m:
        return None
    v = float(m.group(1))
    ex = m.group(2) or m.group(3)
    return v * 10 ** int(ex) if ex else v


def as_ci(x):
    """A difference CI in points; fractions (|x| <= 1) are scaled by 100. From a dict, only a difference interval."""
    if x is None:
        return None
    if isinstance(x, dict):
        keys = [k for k in x if any(w in nk(k) for w in ("boot", "diffci", "ci95pts", "diff95"))]
        return as_ci(x[keys[0]]) if keys else None
    vals = x if isinstance(x, (list, tuple)) else re.findall(r"[-+−]?\d*\.?\d+", str(x))
    vals = [as_float(str(v).replace("−", "-")) for v in vals]
    vals = [v for v in vals if v is not None]
    if len(vals) != 2:
        return None
    lo, hi = sorted(vals)
    if max(abs(lo), abs(hi)) <= 1.0:
        lo, hi = 100 * lo, 100 * hi
    return [round(lo, 1), round(hi, 1)]


def task_id(x):
    s = nk(describe(x) or "")
    if not s:
        return None
    m = re.match(r"^(t[1-8]|r1)(?![0-9])", s)
    if m:
        return m.group(1).upper()
    for tid, name, cls, aliases, _ in TASKS:
        if s == nk(cls) or nk(cls) in s or nk(name) in s or any(a in s for a in aliases):
            if tid == "T5" and any(a in s for a in TASK["T8"][3]):
                continue
            return tid
    return None


def edit_task_of(r):
    """The task an edit was built for, when this row runs it elsewhere (a side-effect row)."""
    t = task_id(r.get("edit_task"))
    if t:
        return t
    for f in ("experiment", "notes"):
        m = re.search(r"\b([TR][1-8])\b\s*(?:edit|'s edit)", describe(r.get(f)) or "", re.I)
        if m:
            return m.group(1).upper()
    return None


def edit_name(r):
    e = r.get("experiment_edit") if "experiment_edit" in r else r["_extra"].get("edit")
    return (e.get("name") if isinstance(e, dict) else describe(e)) or None


def normalize(row):
    out, extra = {}, {}
    for k, v in row.items():
        key = KEYMAP.get(nk(k))
        if key and key not in out:
            out[key] = v
        else:
            extra[k] = v
    for k in ("unedited_k", "edited_k", "control_k", "teacher_k", "rerun_k"):
        if out.get("n") is None and frac_n(out.get(k)):
            out["n"] = frac_n(out[k])
    return out, extra


def describe(x):
    """Free-form values (transfer, retention, notes, names) as one readable line."""
    if x is None:
        return None
    if isinstance(x, str):
        return x.strip() or None
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, (int, float)):
        return f"{x:g}"
    if isinstance(x, (list, tuple)):
        return "; ".join(p for p in (describe(i) for i in x) if p) or None
    if isinstance(x, dict):
        r, _ = normalize(x)
        u, ed, n = as_int(r.get("unedited_k")), as_int(r.get("edited_k")), as_int(r.get("n"))
        if u is not None and ed is not None:
            s = f"{describe(r.get('task')) or ''}: {u} → {ed}" + (f" of {n}" if n else "")
            fx, br = as_int(r.get("fixed")), as_int(r.get("broke"))
            if fx is not None and br is not None:
                pv = as_float(r.get("p"))
                s += f" ({fx} / {br}" + (f", p = {fp(pv)})" if pv is not None else ")")
            return s.strip(": ")
        return "; ".join(f"{k}: {describe(v)}" for k, v in x.items() if describe(v)) or None
    return str(x)


def read_cross(args, meta_out):
    """Rows per model id, from the branches (or the files given on the command line)."""
    out = {}
    for mid, branch, rel in SOURCES:
        override = getattr(args, mid)
        text, where = None, None
        if override:
            text, where = Path(override).read_text(), dict(kind="file", path=str(override))
        else:
            ref = f"origin/{branch}"
            if git("rev-parse", "--verify", "--quiet", ref):
                text = git("show", f"{ref}:{rel}")
                if text is not None:
                    where = dict(kind="branch", branch=branch, path=rel, url=f"{REPO}/blob/{branch}/{rel}",
                                 commit=(git("rev-parse", "--short=12", ref) or "").strip())
            if text is None and (ROOT / rel).exists():
                text, where = (ROOT / rel).read_text(), dict(kind="checkout", path=rel, url=f"{REPO}/blob/main/{rel}")
        if text is None:
            meta_out[mid] = dict(kind="none", branch=branch, path=rel)
            print(f"  {mid}: no results yet ({branch}:{rel})")
            continue
        data = json.loads(text)
        meta, rows = {}, None
        if isinstance(data, list):
            rows = data
        elif isinstance(data, dict):
            for k in ("experiments", "rows", "records", "results", "metrics", "runs"):
                if isinstance(data.get(k), list):
                    rows = data[k]
                    break
            meta = {nk(k): v for k, v in data.items() if not isinstance(v, list)}
            if rows is None:
                rows = [data]
        norm, unknown = [], set()
        file_status = str(meta.get("status", "")).lower()
        for r in rows or []:
            if not isinstance(r, dict):
                continue
            nr, extra = normalize(r)
            unknown |= set(extra) - {"status", "edit", "hardware", "source", "values_changed_note"}
            ctl = extra.get("control")
            if isinstance(ctl, dict) and as_int(ctl.get("n")) is not None:
                extra.setdefault("control_n", as_int(ctl.get("n")))
                extra.setdefault("control_slots", ctl.get("slots"))
                if as_int(ctl.get("unedited_k_same_slots")) is not None:
                    extra.setdefault("control_unedited_k", as_int(ctl.get("unedited_k_same_slots")))
                evc = ctl.get("edit_vs_control") or {}
                extra.setdefault("edit_vs_control", {"cand_k": ctl.get("edited_k_same_slots"), "ref_k": ctl.get("k"),
                                                     "p": evc.get("p"), "slots": ctl.get("slots")})
            tch = extra.get("teacher")
            if isinstance(tch, dict) and as_int(tch.get("n")) is not None:
                vu, ve = tch.get("vs_unedited") or {}, tch.get("vs_edit") or {}
                extra.setdefault("teacher_n", as_int(tch.get("n")))
                extra.setdefault("teacher_slots", tch.get("slots"))
                for k_, v_ in (("teacher_unedited_k", tch.get("unedited_k_same_slots")),
                               ("teacher_edited_k", tch.get("edited_k_same_slots")),
                               ("teacher_p", vu.get("p")), ("teacher_fixed", vu.get("fixed")),
                               ("teacher_broke", vu.get("broke"))):
                    if v_ is not None:
                        extra.setdefault(k_, v_)
                if ve.get("p") is not None:
                    extra.setdefault("edit_vs_teacher", {"p": ve.get("p"), "ref_k": tch.get("k"),
                                                         "cand_k": tch.get("edited_k_same_slots")})
            nr["_extra"] = extra
            pc = (describe(nr.get("code")) or "").upper()
            nr["_ptask"] = pc if pc in PROSPECT["pairs"] else None
            nr["_task"] = None if nr["_ptask"] else (task_id(nr.get("code")) or task_id(nr.get("task")))
            nr["_edit_task"] = edit_task_of(nr)
            nr["_prov"] = str(extra.get("status") or file_status).lower() == "provisional"
            norm.append(nr)
        # side-effect rows that name the edit (e.g. "v2mi-f") instead of its task: map the name to the task it was built for
        built = {edit_name(r): r["_task"] for r in norm if edit_name(r) and r["_task"]
                 and not re.search(r"side|transfer", nk(describe(r.get("experiment")) or ""))}
        # ... or say "built for T8" in their notes, or share the edit location with exactly one task's own edit rows
        by_loc = {}
        for r in norm:
            if r["_task"] and as_int(r.get("edited_k")) is not None and not re.search(
                    r"side|transfer", nk(describe(r.get("experiment")) or "")):
                by_loc.setdefault(describe(r.get("edit_location")), set()).add(r["_task"])
        for r in norm:
            if not r["_edit_task"] and re.search(r"side|transfer", nk(describe(r.get("experiment")) or "")):
                m = re.search(r"built for\s+([TR][1-8])\b", describe(r.get("notes")) or "", re.I)
                same = by_loc.get(describe(r.get("edit_location"))) or set()
                r["_edit_task"] = (built.get(edit_name(r)) or (m.group(1).upper() if m else None)
                                   or (next(iter(same)) if len(same) == 1 else None))
        final = str(meta.get("status", "")).lower() in ("final", "done", "complete", "finished")
        readme = None if override else git("show", f"origin/{branch}:{str(Path(rel).parent)}/{mid}/README.md")
        m_st = (re.search(r"^Status:\s*\*\*([^*]*\bcomplete\b[^*]*)\*\*", readme or "", re.M | re.I)
                or re.search(r"^Status \(([^)]*\bfinal)\)", readme or "", re.M | re.I)    # "Status (date, final): ..."
                or re.search(r"^\*\*Status \(([^)]*\bruns finished)\)", readme or "", re.M | re.I))   # Track C'
        if m_st:
            final = True
            meta["status_note"] = f"{m_st.group(1).strip()} ({branch} README)"
        out[mid] = dict(rows=norm, meta=meta, unknown=sorted(unknown), final=final,
                        runs=None if override else (f"origin/{branch}", f"{Path(rel).parent}/{mid}/results/runs"))
        meta_out[mid] = where
        unmatched = sorted({describe(r.get("task")) or "?" for r in norm if not r["_task"]})
        print(f"  {mid}: {len(norm)} row(s) from {where.get('branch') or where.get('path')}"
              + (f"; unknown keys kept: {', '.join(sorted(unknown))}" if unknown else "")
              + (f"; tasks not in T1-T8/R1: {', '.join(unmatched)}" if unmatched else ""))
    return out


def loc_class(s):
    s = nk(s or "")
    if not s:
        return None
    back = any(w in s for w in ("text", "language", "vlm", "llm", "backbone", "reason", "prompt", "instruction",
                                "selfattn", "vproj", "eagle", "qwen"))
    act = any(w in s for w in ("generation", "moegen", "video", "action", "head", "expert", "denois", "dit",
                               "diffusion", "flow"))
    if "both" in s or "plus" in s or (back and act):
        return "both"
    return "backbone" if back else ("action" if act else None)


def ints(r):
    """Counts of one row. A control or teacher given as {k, n} on a different number of starts is not paired."""
    out = {k: as_int(r.get(k)) for k in ("n", "unedited_k", "edited_k", "fixed", "broke", "control_k", "teacher_k",
                                         "rerun_k")}
    for k in ("control_k", "teacher_k"):
        v = r.get(k)
        if isinstance(v, dict) and as_int(v.get("n")) not in (None, out["n"]):
            out[k] = None
    return out


def stat(r):
    """Edited vs unedited: counts, p and CI (computed from the counts if not reported), difference."""
    i = ints(r)
    p, p_note = as_float(r.get("p")), None
    if p is None and i["fixed"] is not None and i["broke"] is not None:
        p, p_note = mcnemar(i["fixed"], i["broke"]), "p computed here"
    ci, ci_note = as_ci(r.get("ci")), None
    d = None
    if None not in (i["n"], i["unedited_k"], i["edited_k"]):
        d = round(100 * (i["edited_k"] - i["unedited_k"]) / i["n"], 1)
    if ci is None and None not in (i["n"], i["unedited_k"], i["edited_k"], i["fixed"], i["broke"]):
        try:
            _, lo, hi = paired_ci(i["n"], i["unedited_k"], i["edited_k"], i["fixed"], i["broke"])
            ci, ci_note = [lo, hi], "CI computed here"
        except ValueError:
            pass
    return i, p, p_note, ci, ci_note, d


def where_txt(r):
    sl = r.get("slots")
    sl = f"{sl[0]}–{sl[1]}" if isinstance(sl, (list, tuple)) and len(sl) == 2 else describe(sl)
    return "; ".join(x for x in (describe(r.get("split")), f"slots {sl}" if sl else None) if x)


def slot_kind(r):
    """' practice' / ' unseen' for the shared slot ranges, as the figure labels say it."""
    sl = r.get("slots")
    lo = sl[0] if isinstance(sl, (list, tuple)) and sl and isinstance(sl[0], int) else None
    return " unseen" if lo is not None and lo >= 96000 else " practice" if lo is not None and 30000 <= lo < 30032 else ""


def edit_id(r):
    """The edit's own name ('t8und', 't8undp'): its edit field, else the first "edit NAME" in its notes."""
    n = edit_name(r) if "_extra" in r else None
    if n:
        return n
    m = re.search(r"\bedit ([A-Za-z][\w.-]*\d[\w.-]*|[a-z]+\d\w*(?:-\w+)?)", describe(r.get("notes")) or "")
    return m.group(1) if m else None


def offline_of(r):
    o = r.get("offline") if r.get("offline") is not None else (r.get("_extra") or {}).get("offline")
    return o if isinstance(o, dict) else None


def edit_key(r):
    """An edit's name for comparing rows: case, hyphens and dots dropped, so the records' 't8undp' and the README's
    't8und-p' are one edit. From edit_id, else the "edit" field of the row's offline record."""
    n = edit_id(r) or (offline_of(r) or {}).get("edit")
    return nk(n) if n else None


def stacked(name):
    """True for an edit named after two tasks or more ('t8t3und'), i.e. one task's edit built on another's."""
    return bool(name) and len(re.findall(r"[tr][1-8]", name)) > 1


def followups(rows):
    """Follow-up builds of an edit, named like it with one letter more ('t8undp', 't8undr' after 't8und'):
    {follow-up: base}. They are not other edit locations, so the main tables leave them out."""
    names = {edit_key(r) for r in rows} - {None}
    return {n: n[:-1] for n in names if len(n) > 2 and n[-1].isalpha() and n[:-1] in names}


def loc_short(loc):
    """'text pathway', 'VLM backbone', 'diffusion transformer': the shorter of a location's name and its gloss."""
    m = re.match(r"\s*(.*?)\s*\((.*)\)\s*$", loc or "")
    return min((m.group(1), m.group(2)), key=len) if m else (loc or "")


def is_practice(r):
    """An edit scored on the practice slots (the starts it was built from), or named so by its record."""
    return slot_kind(r) == " practice" or "practice" in nk(describe(r.get("experiment")) or "")


def run_success(src, r, tid, slots):
    """Successes of the row's edit on the slots first-last, summed over the track's per-run records that tile them
    (results/runs/<machine>/<task>-E<edit>-s<first slot>.json), or None when they do not cover exactly those slots."""
    mach = r.get("machine") or (r.get("_extra") or {}).get("machine")
    if not (src.get("runs") and isinstance(slots, (list, tuple)) and len(slots) == 2 and edit_id(r) and mach):
        return None
    ref, d = src["runs"]
    start, total = slots[0], 0
    while start <= slots[1]:
        text = git("show", f"{ref}:{d}/{mach}/{tid}-E{edit_id(r)}-s{start}.json")
        try:
            j = json.loads(text) if text else None
        except ValueError:
            return None
        sl = list((j or {}).get("slots") or [])
        if not j or len(sl) != 2 or sl[0] != start or sl[1] > slots[1] or j.get("completed") != j.get("n") \
                or as_int(j.get("success")) is None:
            return None
        total += as_int(j["success"])
        start = sl[1] + 1
    return total


def row_cells(src):
    """Cells of one model from its cross-model rows, keyed like the tables."""
    T = {k: {} for k in TABLE_IDS}
    if not src or not src["rows"]:
        return T
    fu = followups(src["rows"])   # follow-up builds of an edit go to their own table (prevention), not these
    # stacked edits ('t8t3und': one task's edit built on top of another's) are neither task's own edit: kept out of
    # these tables, so they never stand in for a task's edit, its where-to-edit bars or its side-effect map
    rows = [r for r in src["rows"] if edit_key(r) not in fu and not stacked(edit_key(r))]
    own = [r for r in rows if r["_task"] and not (r["_edit_task"] and r["_edit_task"] != r["_task"])]
    by_task = {}
    for r in own:
        by_task.setdefault(r["_task"], []).append(r)
    for tid, rs in by_task.items():
        rs = sorted(rs, key=lambda r: not (r.get("headline") is True or as_int(r.get("headline")) == 1))
        un = [r for r in rs if None not in (as_int(r.get("unedited_k")), as_int(r.get("n")))]
        un = ([r for r in un if "failure" in nk(describe(r.get("experiment")) or "")]
              or [r for r in un if as_int(r.get("edited_k")) is None and as_int(r.get("teacher_k")) is None] or un)
        if un:
            r = max(un, key=lambda r: as_int(r.get("n")))
            k, n = as_int(r.get("unedited_k")), as_int(r.get("n"))
            T["failure_map"][tid] = C(f"{k} of {n}", "; ".join(x for x in (pct(k, n), where_txt(r)) if x), k=k, n=n)
        tw = [r for r in rs if None not in (ints(r)["unedited_k"], ints(r)["teacher_k"], ints(r)["n"])]
        def tview(r):
            """The row's counts, or, when its clear-wording arm ran on fewer starts, the counts on those starts."""
            i = ints(r)
            tn_ = as_int(xg_(r, "teacher_n"))
            if not tn_ or tn_ == i["n"]:
                return i, where_txt(r), None
            ua = as_int(xg_(r, "teacher_unedited_k"))
            if ua is None:
                return None
            ts = xg_(r, "teacher_slots")
            return (dict(i, n=tn_, unedited_k=ua, fixed=as_int(xg_(r, "teacher_fixed")), broke=as_int(xg_(r, "teacher_broke"))),
                    f"slots {ts[0]}–{ts[1]}" if isinstance(ts, (list, tuple)) and len(ts) == 2 else None,
                    as_float(xg_(r, "teacher_p")))
        tw = [r for r in tw if tview(r)]
        if tw:
            r = tw[0]
            i, wtxt, tp = tview(r)
            wp = r.get("wording_pair")
            if isinstance(wp, (list, tuple)) and len(wp) == 2:
                wp = f"“{wp[0]}” → “{wp[1]}”"
            elif isinstance(wp, dict):
                o = wp.get("original") or wp.get("orig") or wp.get("rf") or wp.get("from")
                cl = wp.get("clear") or wp.get("teacher") or wp.get("tf") or wp.get("to")
                wp = f"“{o}” → “{cl}”" if o and cl else describe(wp)
            wd = [f"{sgn(100 * (i['teacher_k'] - i['unedited_k']) / i['n'])} points"]
            pw = None
            if (as_int(r.get("edited_k")) is None or tp is not None) and None not in (i["fixed"], i["broke"]):
                pw = tp if tp is not None else as_float(r.get("p"))
                wd = [f"{i['fixed']} / {i['broke']}" + (f", p = {fp(pw)}" if pw is not None else "")]
                cw = as_ci(r.get("ci")) if tp is None else None
                d0 = round(100 * (i["teacher_k"] - i["unedited_k"]) / i["n"], 1)
                wd.append(fci(d0, *cw) if cw else f"{sgn(d0)} points")
            T["wording"][tid] = C(f"{i['unedited_k']} → {i['teacher_k']} of {i['n']}",
                                  "; ".join(x for x in (*wd, wtxt, describe(wp)) if x),
                                  a=i["unedited_k"], b=i["teacher_k"], n=i["n"], pv=pw,
                                  fl=f"{SHORT[tid]} ({i['n']}{slot_kind(r)})")
        ed = [r for r in rs if as_int(r.get("edited_k")) is not None and as_int(r.get("unedited_k")) is not None]
        if not ed:
            continue
        ed.sort(key=is_practice)   # stable: unseen starts first, then the practice starts the edit was built from
        r = ed[0]
        i, p, p_note, ci, ci_note, d = stat(r)
        loc = describe(r.get("edit_location"))
        prac = is_practice(r)
        det = (["scored on the practice starts the edit was built from, not on unseen starts"] if prac else []) + [
            loc or describe(r.get("experiment"))]
        if None not in (i["fixed"], i["broke"]):
            det.append(f"{i['fixed']} / {i['broke']}" + (f", p = {fp(p)}" if p is not None else ""))
        if ci is not None and d is not None:
            det.append(fci(d, *ci))
        det += [where_txt(r)] + [x for x in (p_note, ci_note) if x]
        same_edit = [x for x in ed[1:] if describe(x.get("edit_location")) == loc and is_practice(x) and not prac
                     and edit_id(x) == edit_id(r)]
        for x in same_edit:
            ix = ints(x)
            det.append(f"on the practice starts it was built from: {ix['unedited_k']} → {ix['edited_k']}"
                       + (f" of {ix['n']}" if ix["n"] else ""))
        others = len(ed) - 1 - len(same_edit)
        if others > 0:
            det.append(f"{others} more edit(s) on this task in the where-to-edit table")
        short_loc = (loc[:26] + "…") if loc and len(loc) > 27 else loc
        alts = [dict(loc=describe(x.get("edit_location")), a=ints(x)["unedited_k"], b=ints(x)["edited_k"], n=ints(x)["n"],
                     p=as_float(x.get("p")), prac=is_practice(x), ctl=ints(x)["control_k"])
                for x in ed[1:] if describe(x.get("edit_location")) != loc or edit_id(x) != edit_id(r)]
        cell = C(f"{i['unedited_k']} → {i['edited_k']}" + (f" of {i['n']}" if i["n"] else ""),
                 "; ".join(x for x in det if x), a=i["unedited_k"], b=i["edited_k"], n=i["n"], fx=i["fixed"], br=i["broke"],
                 pv=p,
                 ci=[d, *ci] if (ci is not None and d is not None) else None,
                 color="teal" if loc_class(loc) in (None, "backbone", "both") else "dark", prac=prac or None,
                 loc=loc, alts=alts or None,
                 fl=(f"{SHORT[tid]}, practice starts" if prac else
                     f"{SHORT[tid]}" + (f", {short_loc}" if loc else "")),
                 fb=f"{tid}" + (" practice" if prac else (f" {loc}"[:18] if loc else "")))
        if tid == "R1":
            T["retention"]["R1"] = cell
        else:
            T["edit_gain"][tid] = cell
        # random control and teacher recovery: from the first edit row (unseen starts first) that has them,
        # each tagged with the starts it was measured on
        rr = next((x for x in ed if ints(x)["control_k"] is not None), None)
        xg = lambda row, k: row.get(k) if row.get(k) is not None else (row.get("_extra") or {}).get(k)
        cn = as_int(xg(rr, "control_n")) if rr is not None else None
        if rr is not None and cn and ints(rr)["n"] and cn != ints(rr)["n"]:
            ic, pc = ints(rr), is_practice(rr)
            evc, cs = xg(rr, "edit_vs_control"), xg(rr, "control_slots")
            e_sub = as_int(evc.get("cand_k")) if isinstance(evc, dict) else None
            u_sub = (as_int(xg(rr, "control_unedited_k")) if as_int(xg(rr, "control_unedited_k")) is not None else
                     0 if ic["unedited_k"] == 0 else (cn if ic["unedited_k"] == ic["n"] else None))
            sl = (f"slots {cs[0]}–{cs[1]}" if isinstance(cs, (list, tuple)) and len(cs) == 2 else None)
            pcv = as_float(evc.get("p")) if isinstance(evc, dict) else None
            T["random_control"][tid] = C(
                f"control {ic['control_k']}" + (f", unedited {u_sub}" if u_sub is not None else "")
                + (f", edit {e_sub}" if e_sub is not None else "") + f" of {cn}",
                "; ".join(x for x in (f"the control ran on {cn} of the {ic['n']} starts", sl,
                                      f"edit against control p = {fp(pcv)}" if pcv is not None else None,
                                      "practice starts the edit was built from" if pc else None) if x),
                u=u_sub if e_sub is not None else None, r=ic["control_k"] if None not in (u_sub, e_sub) else None,
                e=e_sub if u_sub is not None else None, n=cn, prac=pc or None, sub=True,
                fl=f"{SHORT[tid]} ({cn}{slot_kind(rr) or (' practice' if pc else '')})")
            rr = None
        if rr is not None:
            ic, pc = ints(rr), is_practice(rr)
            T["random_control"][tid] = C(f"control {ic['control_k']}, unedited {ic['unedited_k']}, edit {ic['edited_k']}"
                                         + (f" of {ic['n']}" if ic["n"] else ""),
                                         "; ".join(x for x in ("practice starts the edit was built from" if pc else None,
                                                               where_txt(rr)) if x) or None,
                                         u=ic["unedited_k"], r=ic["control_k"], e=ic["edited_k"], n=ic["n"],
                                         t=ic["teacher_k"], tn=as_int(xg(rr, "teacher_n")) or ic["n"], prac=pc or None, loc=describe(rr.get("edit_location")),
                                         fl=(f"{tid}, {loc_short(describe(rr.get('edit_location')))} edit" if alts
                                             and rr.get("edit_location") else f"{SHORT[tid]}")
                                         + (f" ({ic['n']}{slot_kind(rr) or (' practice' if pc else '')})"
                                            if ic["n"] else ""))
        rt = next((x for x in ed if as_float(x.get("recovery_pct")) is not None
                   or (ints(x)["teacher_k"] is not None and ints(x)["teacher_k"] != ints(x)["unedited_k"])), None)
        if rt is not None:
            it, pt = ints(rt), is_practice(rt)
            rp = as_float(rt.get("recovery_pct"))
            if rp is not None:
                if abs(rp) <= 1.5 and not float(rp).is_integer():
                    rp *= 100
                tn_ = as_int(xg(rt, "teacher_n")) or it["n"]
                same = it["teacher_k"] is not None and tn_ == it["n"] and it["edited_k"] is not None
                te, tu = as_int(xg(rt, "teacher_edited_k")), as_int(xg(rt, "teacher_unedited_k"))
                sub = not same and None not in (te, tu, it["teacher_k"])
                T["recovery"][tid] = C(f"{sgn(rp, False, 0)}%", "as reported" + (
                    f"; ({it['edited_k']} − {it['unedited_k']}) / ({it['teacher_k']} − {it['unedited_k']}) of {it['n']}"
                    if same else f"; ({te} − {tu}) / ({it['teacher_k']} − {tu}) on the {tn_} starts where the clear "
                                 f"wording also ran" if sub else "") + ("; on practice starts" if pt else ""),
                    prac=pt or None,
                    **(dict(a=it["unedited_k"], b=it["edited_k"], t=it["teacher_k"], n=it["n"]) if same else
                       dict(a=tu, b=te, t=it["teacher_k"], n=tn_, sub=True) if sub else {}))
            elif as_int(xg(rt, "teacher_n")) and as_int(xg(rt, "teacher_n")) != it["n"]:
                # the clear wording ran on fewer starts than the edit: compare on those starts only
                tn_, ts = as_int(xg(rt, "teacher_n")), xg(rt, "teacher_slots")
                u_sub = 0 if it["unedited_k"] == 0 else (tn_ if it["unedited_k"] == it["n"] else None)
                e_sub = run_success(src, rt, tid, ts)
                sl = f"slots {ts[0]}–{ts[1]}" if isinstance(ts, (list, tuple)) and len(ts) == 2 else f"{tn_} starts"
                if None not in (u_sub, e_sub) and it["teacher_k"] != u_sub:
                    T["recovery"][tid] = C(
                        f"{100 * (e_sub - u_sub) / (it['teacher_k'] - u_sub):.0f}%",
                        f"({e_sub} − {u_sub}) / ({it['teacher_k']} − {u_sub}) on the {tn_} starts where the clear wording "
                        f"also ran ({sl}), computed here" + ("; practice starts" if pt else ""),
                        prac=pt or None, a=u_sub, b=e_sub, t=it["teacher_k"], n=tn_, sub=True)
            else:
                T["recovery"][tid] = C(
                    f"{100 * (it['edited_k'] - it['unedited_k']) / (it['teacher_k'] - it['unedited_k']):.0f}%",
                    f"({it['edited_k']} − {it['unedited_k']}) / ({it['teacher_k']} − {it['unedited_k']}), computed here"
                    + ("; on practice starts" if pt else ""), prac=pt or None,
                    a=it["unedited_k"], b=it["edited_k"], t=it["teacher_k"], n=it["n"])
            if tid in T["recovery"]:
                tx = xg(rt, "teacher")
                evt = xg(rt, "edit_vs_teacher")
                T["recovery"][tid] = guard_recovery(
                    T["recovery"][tid], T["wording"].get(tid),
                    explicit=(as_float(xg(rt, "teacher_p")) if as_float(xg(rt, "teacher_p")) is not None else
                              as_float(tx.get("p")) if isinstance(tx, dict) else None),
                    p_cmp=as_float(evt.get("p")) if isinstance(evt, dict) else None)
        vc = r.get("values_changed")
        if vc is not None:
            T["size"][tid] = C(num(as_int(vc)) + " values" if as_int(vc) is not None else describe(vc), loc)
        if r.get("retention") is not None:
            T["retention"]["_" + tid] = C(describe(r.get("retention")), f"the {tid} edit")
        if i["rerun_k"] is not None:
            T["noise"]["unedited2"] = C(f"{i['unedited_k']} vs {i['rerun_k']} of {i['n']}", f"{tid}, unedited, run twice")
        if i["n"]:
            T["noise"]["mde"] = C(f"{mde(i['n']):.0f} points at {i['n']} pairs", "if about 30% of pairs differ")
    # side effects: the model's main edit, unchanged, on every task
    edits = sorted([r for r in own if as_int(r.get("edited_k")) is not None], key=is_practice)
    if edits:
        main = next((r for r in edits if r.get("headline") is True or as_int(r.get("headline")) == 1),
                    next((r for r in edits if r["_task"] == "T1"), edits[0]))
        et = main["_task"]
        items = [(et, main, True)]
        items += sorted([(r["_task"], r, False) for r in rows if r["_edit_task"] == et and r["_task"] and r["_task"] != et],
                        key=lambda x: is_practice(x[1]))   # unseen starts first: setdefault keeps the first cell per task
        for field in ("transfer", "retention"):
            tr = main.get(field)
            if isinstance(tr, dict) and tr and all(task_id(k) for k in tr):
                tr = [dict(v, task=k) for k, v in tr.items() if isinstance(v, dict)]
            for it in (tr if isinstance(tr, list) else [tr] if isinstance(tr, dict) else []):
                if isinstance(it, dict):
                    nr, _ = normalize(it)
                    tid = task_id(nr.get("task"))
                    if tid and tid != et:
                        items.append((tid, nr, False))
        for tid, r, target in items:
            i, p, p_note, _, _, _ = stat(r)
            if i["edited_k"] is None or i["unedited_k"] is None:
                continue
            dlt = i["edited_k"] - i["unedited_k"]
            have = T["side_effects"].get(tid)
            if have and is_practice(r) and not have.get("prac") and not target:
                # the unseen-start cell stays; the practice run of the same edit joins its detail
                have["d"] = "; ".join(x for x in (have.get("d"), f"practice starts: {i['unedited_k']} → {i['edited_k']}"
                                                  + (f" of {i['n']}" if i["n"] else "")
                                                  + (f", p = {fp(p)}" if p is not None else "")
                                                  + (f" ({where_txt(r)})" if where_txt(r) else "")) if x)
                continue
            T["side_effects"].setdefault(tid, C(
                f"{dsgn(dlt)}: {i['unedited_k']} → {i['edited_k']}" + (f" of {i['n']}" if i["n"] else ""),
                "; ".join(x for x in ((f"{i['fixed']} / {i['broke']}" if i["fixed"] is not None else None),
                                      (f"p = {fp(p)}" if p is not None else None), "the target task" if target else None,
                                      "practice starts" if is_practice(r) else None, where_txt(r), p_note) if x),
                delta=dlt, a=i["unedited_k"], b=i["edited_k"], n=i["n"], p=p, target=target or None,
                prac=is_practice(r) or None))
        r1 = T["side_effects"].get("R1")
        if r1 and not r1.get("target") and "R1" not in T["retention"]:   # retention is the side effect on R1
            T["retention"]["R1"] = C(f"{r1['a']} → {r1['b']} of {r1['n']}", f"the {et} edit, unchanged; {r1.get('d', '')}",
                                     prac=r1.get("prac"))
        T["side_effects"]["_edit"] = C(f"the {et} edit" + (" on practice starts" if is_practice(main) else "")
                                       + (f" ({describe(main.get('edit_location'))})"
                                                          if main.get("edit_location") else ""))
    # where to edit, offline: how much of the clearer wording's action change each edit reproduces on recorded
    # decisions (no rollouts); the held-aside "check" decisions are preferred over the ones the edit was built from
    offl = []
    for r in own:
        o = r.get("offline") or (r.get("_extra") or {}).get("offline")
        if isinstance(o, dict) and as_float(o.get("gap_closed_pct")) is not None:
            offl.append((r, o))
    if offl:
        chk = [(r, o) for r, o in offl if str(o.get("group")).lower() == "check"] or offl
        t0 = chk[0][0]["_task"]
        chk = [(r, o) for r, o in chk if r["_task"] == t0]
        o0 = chk[0][1]
        held = str(o0.get("group")).lower() == "check"
        T["where"]["offline_where"] = C(
            "; ".join(f"{describe(r.get('edit_location'))}: {sgn(as_float(o['gap_closed_pct']), False, 0)}%"
                      + (f" (random, same size: {sgn(as_float(o['control_gap_closed_pct']), False, 0)}%)"
                         if as_float(o.get("control_gap_closed_pct")) is not None else "") for r, o in chk),
            f"{t0}, {as_int(o0.get('n'))} recorded decisions"
            + (" held aside, never used to build either edit" if held else " the edits were built from")
            + "; share of the clearer wording’s action change reproduced, by end-effector distance at the end of "
              "the action chunk; offline, no rollouts",
            parts=[[describe(r.get("edit_location")), as_float(o["gap_closed_pct"]), as_float(o.get("control_gap_closed_pct"))]
                   for r, o in chk])
    # where to edit, offline, the teacher-projection form (GR00T's record): the share of the clearer wording's action
    # change that the edit reproduces, on failed practice decisions no build used; one edit per location (the base
    # variant, the shortest name) and its same-size random control where one was run
    if "offline_where" not in T["where"]:
        tp = []
        for r in own:
            o = r.get("unseen_failures") or (r.get("_extra") or {}).get("unseen_failures")
            name = str(r.get("edit") or (r.get("_extra") or {}).get("edit") or "")
            if isinstance(o, dict) and as_float(o.get("teacher_projection")) is not None:
                tp.append((r, o, name))
        if tp:
            t0 = tp[0][0]["_task"]
            tp = [x for x in tp if x[0]["_task"] == t0]
            out, nd, parts = [], None, []
            for lc in ("backbone", "action", "both"):
                grp = [x for x in tp if loc_class(describe(x[0].get("edit_location"))) == lc]
                eds = sorted([x for x in grp if "rand" not in x[2]], key=lambda x: len(x[2]))
                ctl = [x for x in grp if "rand" in x[2]]
                if not eds:
                    continue
                r, o, _ = eds[0]
                nd = nd or as_int(o.get("n_decisions"))
                parts.append([describe(r.get("edit_location")), 100 * as_float(o["teacher_projection"]),
                              100 * as_float(ctl[0][1]["teacher_projection"]) if ctl else None])
                out.append(f"{describe(r.get('edit_location'))}: {sgn(100 * as_float(o['teacher_projection']), False, 0)}%"
                           + (f" (random, same size: {sgn(100 * as_float(ctl[0][1]['teacher_projection']), False, 0)}%)"
                              if ctl else ""))
            if out:
                T["where"]["offline_where"] = C(
                    "; ".join(out),
                    f"{t0}, {nd} failed practice decisions that no build used; share of the clearer wording’s action "
                    "change reproduced, as the projection of the edit’s change onto the wording’s over the executed "
                    "actions (joints and gripper); offline, no rollouts", parts=parts)
    # where to edit: every edit with a location, on its task
    locs = [r for r in own if loc_class(describe(r.get("edit_location"))) and as_int(r.get("edited_k")) is not None]
    locs = [r for r in locs if "where" in nk(describe(r.get("experiment")) or "")] or locs
    locs.sort(key=is_practice)
    if locs:
        # compare locations on the same starts: the (task, n, slots) group that tried the most locations, unseen first
        groups = {}
        for r in locs:
            groups.setdefault((r["_task"], as_int(r.get("n")), describe(r.get("slots"))), []).append(r)
        same = max(groups.values(), key=lambda g: (len({loc_class(describe(x.get("edit_location"))) for x in g}),
                                                   not is_practice(g[0])))
        t0 = same[0]["_task"]
        n0 = as_int(same[0].get("n"))
        T["where"]["starts"] = C(f"{t0}, {n0}{slot_kind(same[0]) or (' practice' if is_practice(same[0]) else '')} starts"
                                 if n0 else t0,
                                 where_txt(same[0]) or None)
        if as_int(same[0].get("unedited_k")) is not None:
            T["where"]["unedited"] = C(f"{as_int(same[0].get('unedited_k'))}" + (f" of {n0}" if n0 else ""))
        for cls in ("backbone", "action", "both"):
            hits = [r for r in same if loc_class(describe(r.get("edit_location"))) == cls]
            if hits:
                T["where"][cls] = C("; ".join(f"{describe(r.get('edit_location'))}: {as_int(r.get('edited_k'))}"
                                              + (f" of {as_int(r.get('n'))}" if as_int(r.get("n")) else "") for r in hits))
        T["where"]["_bars"] = dict(task=t0, n=n0, base=as_int(same[0].get("unedited_k")),
                                   kind=slot_kind(same[0]) or (" practice" if is_practice(same[0]) else ""),
                                   bars=[(describe(r.get("edit_location")), as_int(r.get("edited_k")),
                                          "teal" if loc_class(describe(r.get("edit_location"))) == "backbone" else "dark")
                                         for r in same],
                                   sites=[dict(loc=describe(r.get("edit_location")),
                                               cls=loc_class(describe(r.get("edit_location"))), a=stat(r)[0]["unedited_k"],
                                               b=stat(r)[0]["edited_k"], n=stat(r)[0]["n"], p=stat(r)[1],
                                               ctl=ints(r)["control_k"], prac=is_practice(r) or None) for r in same])
    return T


# ---------------------------------------------------------------- rendering sensitivity (validation records)
RENDER = {  # per model: the same starts under RoboLab's performance preset and its default preset
    "groot": dict(branch="cross-groot/v1", task="Banana onto plate", robolab="BananaOnPlateTask",
                  perf="study/results/cross/groot/results/validation/smoke-bop-s0.json",
                  default="study/results/cross/groot/results/validation/smoke-bop-s0-default.json",
                  official="study/results/cross/groot/results/validation/official-bop16.episode_results.jsonl"),
}


def branch_text(branch, rel):
    """A file from origin/<branch>, or from this checkout once merged."""
    if git("rev-parse", "--verify", "--quiet", f"origin/{branch}"):
        t = git("show", f"origin/{branch}:{rel}")
        if t is not None:
            return t, f"{branch}:{rel}"
    if (ROOT / rel).exists():
        return (ROOT / rel).read_text(), rel
    return None, None


def render_cells(cross):
    """{model: cell} for the render-preset pair, from validation records or from rows marked render / preset."""
    out = {}
    for mid, spec in RENDER.items():
        tp, sp = branch_text(spec["branch"], spec["perf"])
        td, sd = branch_text(spec["branch"], spec["default"])
        if not (tp and td):
            continue
        perf, dflt = json.loads(tp), json.loads(td)
        assert perf.get("rendering") in (None, "performance") and dflt.get("rendering") == "default", "render presets"
        a = {r["slot"]: bool(r["success"]) for r in perf["rows"] if r.get("ok", True)}
        b = {r["slot"]: bool(r["success"]) for r in dflt["rows"] if r.get("ok", True)}
        assert sorted(a) == sorted(b) and perf["task"] == dflt["task"] == spec["robolab"]
        n, ka, kb = len(a), sum(a.values()), sum(b.values())
        fx, br = sum((not a[s_]) and b[s_] for s_ in a), sum(a[s_] and not b[s_] for s_ in a)
        srcs = [sp, sd]
        extra = ""
        to, so = branch_text(spec["branch"], spec["official"])
        if to:
            eps = [json.loads(l) for l in to.splitlines() if l.strip()]
            extra = f"; NVIDIA's own serving path, default preset: {sum(bool(e['success']) for e in eps)} of {len(eps)}"
            srcs.append(so)
        out[mid] = C(f"{ka} → {kb} of {n}", f"performance → default; {spec['task']}, the same {n} starts (slots "
                     f"{min(a)}–{max(a)}); {fx} / {br}, p = {fp(mcnemar(fx, br))}{extra}", src=srcs,
                     a=ka, b=kb, n=n, task=spec["task"])
    for mid in ORDER:  # rows a track adds to its metrics file: experiment "render..." with performance / default counts
        if mid in out or not cross.get(mid):
            continue
        rr = [r for r in cross[mid]["rows"] if re.search(r"render|preset", nk(describe(r.get("experiment")) or ""))]
        pk = [r for r in rr if as_int(r["_extra"].get("performance_k")) is not None and as_int(r["_extra"].get("default_k")) is not None]
        if pk:
            r = pk[0]
            ka, kb, n = as_int(r["_extra"]["performance_k"]), as_int(r["_extra"]["default_k"]), as_int(r.get("n"))
            out[mid] = C(f"{ka} → {kb}" + (f" of {n}" if n else ""), "; ".join(x for x in (
                "performance → default", describe(r.get("task")), where_txt(r)) if x), a=ka, b=kb, n=n)
    return out


# ---------------------------------------------------------------- can the side effect be prevented? (follow-up builds)
# Cosmos3 Edge's T8 edit breaks the two other tasks whose instructions say "left" (T3, T5). Track A rebuilds it with
# zero-change preservation keys (rows of K from other instructions with zero rows of V) and scores each build on T8 and
# on T5 and T3, offline first. Rows are found by edit name (edit_key): the records write "edit t8undp" in their notes,
# the README "t8und-p". The descriptions, and the counts in them, are the README's (PREVENT["readme"]).
PREVENT = dict(
    model="cosmos", base="t8und", target="T8", left=("T5", "T3"),
    gate=85,   # README: a rebuild runs on the robot only if, in replays, it reproduces >= 85% of the wording's effect
    # name; plain (the main flow's row name); label (the records' row name); short (the records' sentences); keys (the
    # instructions kept unchanged, when that is the only change); nick (a short handle in the main flow); always (a row
    # even before any record; others appear once a record names them); intro (the records' extra sentence)
    edits=[dict(name="t8und", plain="The edit", label="the edit for T8, as above", always=True),
           dict(name="t8undp", plain="The edit + 65 unrelated instructions kept unchanged",
                label="+ preservation keys from 65 generic RoboLab instructions",
                short="preservation keys from 65 generic instructions", keys="65 unrelated instructions", always=True,
                nick="the 65-instruction version"),
           dict(name="t8undr", plain="The edit + 24 left/right instructions from other tasks kept unchanged",
                label="+ preservation keys from 24 left/right instructions of other RoboLab tasks",
                short="preservation keys from 24 left/right instructions of other tasks",
                keys="24 left/right instructions from other tasks", always=True, nick="the left/right version",
                intro="For `t8undr` a rule fixed before the build leaves out every T3, T5 and T8 instruction and object "
                      "pair, so T3 and T5 stay unseen tests; it has its own random control."),
           dict(name="t8unds", plain="Copying only the words both wordings share (+ the 65 unrelated instructions)",
                label="only the instruction template aligned, + the 65 generic preservation keys",
                short="only the instruction template aligned and the 65 generic preservation keys",
                nick="the shared-words version",
                intro="`t8unds` tests whether the word-for-word alignment causes the break: it aligns only the "
                      "template around the instruction, leaves the differing words unconstrained, and keeps the "
                      "generic keys of `t8undp`.")],
    readme="cross-cosmos/v1:study/results/cross/cosmos/README.md",
    records="cross-cosmos/v1:study/results/cross/metrics_cosmos.json",
)


# "Can a second edit undo the damage?": Cosmos3 Edge's T3 is wording-sensitive too (README, rules 591459d), so the
# track built T3 its own edit (t3und) and the T3 update on top of the T8 edit (t8t3und, stacked; stays out of the
# main tables, see stacked()). Rows: the edits; columns: T8, T3 and T5.
SECOND = dict(
    model="cosmos", target="T8", second="T3", cols=("T8", "T3", "T5"), gate=85,
    edits=[dict(name="t8und", plain="The mustard edit", built="built for Mustard right of raisin box"),
           dict(name="t3und", plain="The cube-left edit", built="built for Cube left of bowl"),
           dict(name="t8t3und", plain="Both edits together", built="the cube-left edit built on top of the mustard edit")],
    col_note={"T8": "the mustard edit’s task", "T3": "the cube-left edit’s task", "T5": "a “left” task"},
    records="cross-cosmos/v1:study/results/cross/metrics_cosmos.json",
    readme="cross-cosmos/v1:study/results/cross/cosmos/README.md",
)
SECOND_LAST = {}   # this run's second-edit rows, for the TL;DR
DONE = {"all": False}   # every track's record final: no "as they land" wording
PROSPECT_LAST = {}  # the prospective test's pairs with records, for the task legend
WHEN_LAST = {}     # whether the records show both sides of "when does the edit work?", for the TL;DR


def pe(name):
    """A build's entry in PREVENT; an unknown follow-up gets a plain label."""
    return next((e for e in PREVENT["edits"] if e["name"] == name),
                dict(name=name, plain=f"Another version ({name})", label=f"a follow-up of {PREVENT['base']}",
                     short=f"the follow-up {name}"))


PCOLS = [("T8", TASK_NAME["T8"], "the task it was built for"), ("T5", TASK_NAME["T5"], "a “left” task"),
         ("T3", TASK_NAME["T3"], "a “left” task"), ("gap", "In replays", "share of the clear wording’s effect"),
         ("dev", "Hand moved", f"{TASK_NAME['T5']}, in replays; lower is better")]


def ofind(o, *keys):
    """The first of these keys (spelled freely) that the offline record gives as a number."""
    by = {nk(k): v for k, v in o.items()}
    for k in keys:
        v = as_float(by.get(nk(k)))
        if v is not None:
            return v, k
    return None, None


def prevent_closed(rs):
    """unedited → edited on held-out starts if run, else on practice starts (tagged); the other kind in the detail."""
    held = [r for r in rs if slot_kind(r) == " unseen" or "heldout" in nk(describe(r.get("experiment")) or "")]
    prac = [r for r in rs if r not in held]
    r = (held or prac or [None])[0]
    if r is None:
        return None
    i, p, p_note, _, _, _ = stat(r)
    pr = r not in held and is_practice(r)
    det = [f"p = {fp(p)}" if p is not None else None, p_note]
    if held and prac:
        ip = ints(prac[0])
        det.append(f"practice {ip['unedited_k']} → {ip['edited_k']} of {ip['n']}")
    if i["control_k"] is not None:
        cn = as_int(r.get("control_n") if r.get("control_n") is not None else r["_extra"].get("control_n")) or i["n"]
        det.append(f"random control {i['control_k']} of {cn}")
    runs = {("unseen" if x in held else "practice"): [ints(x)["unedited_k"], ints(x)["edited_k"], ints(x)["n"],
                                                         describe(x.get("slots"))] for x in reversed(held + prac)}
    return C(f"{i['unedited_k']} → {i['edited_k']} of {i['n']}", "; ".join(x for x in det if x), a=i["unedited_k"],
             b=i["edited_k"], n=i["n"], p=p, delta=i["edited_k"] - i["unedited_k"], prac=pr or None,
             held=bool(held) or None, runs=runs, slots=describe(r.get("slots")))


def prevent_offline(rows, name):
    """Offline cells of one build: the share of the T8 wording gap it closes on held-aside decisions, and how far it
    moves T5's actions (and T3's, where measured; in the detail) from the unedited policy's. Its own record first;
    else the value another record gives for it as the reference ('t8und_gap_closed_pct', 't5_t8und_dev_cm')."""
    gap, dev = None, {}
    own = [o for o in map(offline_of, rows) if o and nk(o.get("edit") or "") == name]
    ref = [o for o in map(offline_of, rows) if o and nk(o.get("edit") or "") not in ("", name)]
    for r, o in ((r, offline_of(r)) for r in rows):
        if not o or nk(o.get("edit") or "") != name:
            continue
        v, _ = ofind(o, "t8_gap_closed_pct", "t8_variant_gap_closed_pct", f"t8_{name}_gap_closed_pct")
        if v is None and r["_task"] == "T8" and str(o.get("group")).lower() == "check":
            v, _ = ofind(o, "gap_closed_pct")
        if v is not None and gap is None:
            n = as_int(o.get("t8_n")) or as_int(o.get("n"))
            cv, _ = ofind(o, "t8_control_gap_closed_pct", "control_gap_closed_pct")
            gap = C(f"{sgn(v, False, 0 if abs(v) >= 10 else 1)}%",
                    "; ".join(x for x in (f"{n} recorded moments" if n else None,
                                          f"random edit: {sgn(cv, False, 1)}%" if cv is not None else None) if x) or None,
                    x=v, ctl=cv)
    for o in ref:   # the build as another record's reference
        v, _ = ofind(o, f"{name}_gap_closed_pct", f"t8_{name}_gap_closed_pct")
        if v is not None:
            if gap is None:
                n = as_int(o.get("t8_n"))
                gap = C(f"{sgn(v, False, 0)}%", f"{n} held-aside decisions" if n else None, x=v)
            elif abs(gap["x"] - v) > 0.05:
                print(f"  CHECK prevention: {name} T8 gap closed {gap['x']} in its own record, {v} in {o.get('edit')}'s")
    for t in ("t5", "t3"):
        for o in own:
            v, _ = ofind(o, f"{t}_variant_dev_cm", f"{t}_{name}_dev_cm", f"{t}_dev_cm", f"{t}_edited_dev_cm")
            if v is not None and t not in dev:
                dev[t] = (v, as_int(o.get(f"{t}_n")))
        for o in ref:
            v, _ = ofind(o, f"{t}_{name}_dev_cm")
            if v is not None and t not in dev:
                dev[t] = (v, as_int(o.get(f"{t}_n")))
            elif v is not None and abs(dev[t][0] - v) > 0.005:
                print(f"  CHECK prevention: {name} {t.upper()} deviation {dev[t][0]} here, {v} in {o.get('edit')}'s record")
    dcell = None
    if "t5" in dev:
        v, n = dev["t5"]
        d3 = dev.get("t3")
        dcell = C(f"{v:.1f} cm", "; ".join(x for x in (f"{n} recorded moments" if n else None,
                                                        f"T3: {d3[0]:.1f} cm" + (f" on {d3[1]}" if d3[1] else "")
                                                        if d3 else None) if x) or None,
                  x=v, x3=d3[0] if d3 else None)
    return gap, dcell


def prevention(cross):
    """The follow-up builds of Cosmos3 Edge's T8 edit, one row each: closed loop on T8, T5 and T3, and the offline
    gates. A missing cell reads running, or "—" once the model's record is final, or "not run" for a follow-up
    whose T5 or T3 run already broke (the track runs further checks only if a build restores T5; README)."""
    src = cross.get(PREVENT["model"])
    rows = [r for r in (src["rows"] if src else []) if not r["_prov"]]
    final = bool(src and src["final"])
    names = [e["name"] for e in PREVENT["edits"]]
    fu = followups(rows)
    extra = sorted(n for n in fu if n not in names and fu[n] in names)
    out = []
    for name in names + extra:
        mine = [r for r in rows if edit_key(r) == name]
        if not mine and not pe(name).get("always"):
            continue
        cells = {}
        for tid in (PREVENT["target"], *PREVENT["left"]):
            rs = [r for r in mine if r["_task"] == tid and None not in (ints(r)["unedited_k"], ints(r)["edited_k"], ints(r)["n"])]
            cells[tid] = prevent_closed(rs)
        cells["gap"], cells["dev"] = prevent_offline(rows, name)
        closed = any(cells[t] for t in (PREVENT["target"], *PREVENT["left"]))
        gate_failed = (name != PREVENT["base"] and not closed and cells["gap"] is not None
                       and cells["gap"]["x"] < PREVENT["gate"])
        broke = [t for t in PREVENT["left"] if cells[t] and cells[t].get("p") is not None and cells[t]["p"] < SIG
                 and cells[t]["delta"] < 0]
        for k, _, _ in PCOLS:
            if cells[k] is None and gate_failed:
                cells[k] = C("—", "not run: in replays it reproduced too little of the clear wording’s effect", s="na")
            if cells[k] is None:   # a known reason first; a final record turns the rest to "—"
                cells[k] = (C("—", f"not run: we went on only if a rebuild saved {tn('T5')}", s="na")
                            if name != PREVENT["base"] and broke and k in (PREVENT["target"], *PREVENT["left"]) else
                            NA if final else RUN)
        m = re.search(r"from (\d+) ", pe(name)["label"])
        for r in mine if m else []:
            m2 = re.search(r"from (\d+) [\w/ -]*instructions", describe(r.get("notes")) or "")
            if m2 and m2.group(1) != m.group(1):
                print(f"  CHECK prevention: {name}: a record says {m2.group(1)} instructions, the label {m.group(1)}")
        out.append(dict(name=name, label=pe(name)["label"], plain=pe(name)["plain"], cells=cells, gate_failed=gate_failed,
                        measured=bool(mine)))
    return dict(model=PREVENT["model"], base=PREVENT["base"], target=PREVENT["target"], left=list(PREVENT["left"]),
                rows=out, sources=[PREVENT["records"], PREVENT["readme"]], text=prevent_finding(out))


FAILED = ("broke", "unrestored")


def kind_of(c):
    return " practice" if c.get("prac") else " unseen"


def _covers(wide, narrow):
    """True when the slot range "first; last" wide contains the range narrow."""
    try:
        (w0, w1), (n0, n1) = ([int(x) for x in str(v).split(";")] for v in (wide, narrow))
    except (ValueError, TypeError):
        return False
    return w0 <= n0 and n1 <= w1


def prevent_parts(r, base, bname):
    """A follow-up build's closed-loop results, each with a verdict, in present and past tense; None while it has none.
    A left task 'broke' if it drops significantly (exact p < 0.05); 'unrestored' if the original edit lowered it on the
    same starts and the build's count is at least as close to the original edit's as to the unedited one; 'near' if
    closer to the unedited one; else 'plain'. bname names the original edit in the text."""
    c8 = r["cells"][PREVENT["target"]]
    left = [(t, r["cells"][t]) for t in PREVENT["left"] if r["cells"][t]["s"] == "ok"]
    if not left and c8["s"] != "ok":
        return None
    t8 = None
    if c8["s"] == "ok":
        a, b, n, k = c8["a"], c8["b"], c8["n"], kind_of(c8)
        if b > a and c8.get("p") is not None and c8["p"] < SIG:
            t8 = ("kept", f"keeps the T8 fix ({a} → {b} of {n}{k} starts)", f"T8 kept its fix ({a} → {b} of {n})")
        elif b > a:
            t8 = ("weak", f"lifts T8 only from {a} to {b} of {n}{k} starts" + (f" (p = {fp(c8['p'])})" if c8.get("p")
                                                                                 is not None else ""),
                  f"T8 rose only from {a} to {b} of {n}")
        else:
            t8 = ("lost", f"loses the T8 fix ({a} → {b} of {n}{k} starts)", f"T8 lost its fix ({a} → {b} of {n})")
    out = []
    for t, c in left:
        a, b, n, k = c["a"], c["b"], c["n"], kind_of(c)
        pv = f"p = {fp(c['p'])}" if c.get("p") is not None else None
        bc = ((base or {}).get("cells", {}).get(t) or {}).get("runs", {}).get(k.strip())
        bb = bc[1] if bc and bc[3] == c.get("slots") and bc[2] == n else None
        if bb is None and bc and bc[1] in (0, bc[2]) and _covers(bc[3], c.get("slots")):
            bb = 0 if bc[1] == 0 else n   # none (or all) of a wider range of starts: the same on these
        if c.get("p") is not None and c["p"] < SIG and b < a:
            out.append((t, "broke", f"{t} still falls from {a} to {b} of {n}{k} starts ({pv})",
                        f"{t} still fell from {a} to {b} of {n}{k} starts", None))
        elif bb is not None and bb < a:
            v = "unrestored" if abs(b - bb) <= abs(b - a) else "near"
            out.append((t, v, (f"{t} goes from {a} to {b} of {n}{k} starts" if v == "unrestored" else
                               f"{t} stays near its unedited rate, {a} → {b} of {n}{k} starts")
                        + f" ({pv + '; ' if pv else ''}{bname}: {bb})",
                        f"{t} went {a} → {b} of {n}{k} starts, against {bb} for the original edit", bb))
        else:
            out.append((t, "plain", f"{t} goes from {a} to {b} of {n}{k} starts" + (f" ({pv})" if pv else ""),
                        f"{t} went {a} → {b} of {n}{k} starts" + (f" ({pv})" if pv else ""), None))
    running = [t for t in (PREVENT["target"], *PREVENT["left"]) if r["cells"][t]["s"] == "running"]
    return dict(t8=t8, left=out, running=running)


def present_clause(subject, parts):
    """'<subject> keeps the T8 fix (...) but does not prevent the break: T5 still falls ...; T3 goes ...'."""
    t8, fails = parts["t8"], [x for x in parts["left"] if x[1] in FAILED]
    others = [x for x in parts["left"] if x[1] not in FAILED]
    s = subject + (f" {t8[1]}" if t8 else "")
    if fails:
        s += (" but" if t8 and t8[0] == "kept" else " and" if t8 else "") + " does not prevent the break: " + and_list(
            x[2] for x in fails)
    if others:
        s += ("; " if fails else ", and " if t8 else ": ") + and_list(x[2] for x in others)
    return s


def prevent_finding(rows):
    """One sentence per follow-up build: what its row shows. '{running}' marks what is not measured yet."""
    base = next((r for r in rows if r["name"] == PREVENT["base"]), None)
    bdev = base["cells"]["dev"] if base else None
    bn = f"`{PREVENT['base']}`"
    parts = []
    for r in rows:
        if r["name"] == PREVENT["base"]:
            continue
        who = f"`{r['name']}` ({pe(r['name'])['short']})"
        res = prevent_parts(r, base, bn)
        g, d = r["cells"]["gap"], r["cells"]["dev"]
        off, smaller = [], set()
        if g["s"] == "ok":
            off.append(f"closes {g['v']} of the T8 gap")
        if d["s"] == "ok":
            mv = []
            for t, x, bx in (("T5", d.get("x"), (bdev or {}).get("x")), ("T3", d.get("x3"), (bdev or {}).get("x3"))):
                if x is None:
                    continue
                mv.append(f"{t}’s {x:.1f} cm" + (f" ({bn}: {bx:.1f} cm)" if bx is not None else ""))
                if bx and x / bx < 0.9:
                    smaller.add(t)
            off.append("moves " + and_list(mv).replace("’s ", "’s actions ", 1) + " from the unedited policy’s")
        if res is None:
            if r.get("gate_failed"):   # failed its offline check: it never ran on the robot
                parts.append(f"{who}: " + (f"offline it {and_list(off)}; " if off else "")
                             + "it failed the offline check, so it did not run on the robot.")
            else:
                parts.append(f"{who}: " + (f"offline it {and_list(off)}; closed loop {RUNNING}." if off else f"{RUNNING}."))
            continue
        s = present_clause(who, res)
        if off:
            s += f"; offline it {and_list(off)}"
        lost = [x[0] for x in res["left"] if x[1] in FAILED and x[0] in smaller]
        if lost:
            s += (f", so the smaller offline deviation on {and_list(lost)} does not carry over to the closed loop")
        if res["running"]:
            s += f"; {and_list(res['running'])}: {RUNNING}"
        parts.append(s + ".")
    return " ".join(parts)


def prevent_html(P, full=True):
    """The subsection's intro, table, records and the sentence on what the rows show; compact (values and tags only,
    a one-line intro) for the main flow."""
    run = '<span class="run">running</span>'
    head = "".join(f'<th scope="col"><span class="mn">{esc(a)}</span><span class="mk">{esc(b)}</span></th>'
                   for _, a, b in PCOLS)
    cols = PCOLS if full else PCOLS[:4]
    if not full:
        head = "".join(f'<th scope="col"><span class="mn">{esc(a)}</span><span class="mk">{esc(b)}</span></th>'
                       for _, a, b in cols)
    body = []
    for r in P["rows"]:
        rh = (f'<code>{esc(r["name"])}</code><span class="rd">{esc(r["label"])}</span>' if full
              else f'{esc(r["plain"])}')
        body.append(f'<tr><th scope="row">{rh}</th>'
                    + "".join(td_html(r["cells"][k] if full or r["cells"][k]["s"] == "na" else
                                      {x: v for x, v in r["cells"][k].items() if x != "d"}, f"{a}, {b}")
                              for k, a, b in cols) + "</tr>")
    table = (f'<table class="ctable ptable{"" if full else " ptable-s"}"><thead><tr><th scope="col"><span class="vh">Edit'
             f'</span></th>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>')
    left = and_list([f"{t} ({TASK[t][1].lower()})" for t in P["left"]])
    note = ("Closed loop: unedited → edited on the same starts, unseen where run, else practice (tagged). Offline, "
            "on recorded decisions, before any closed loop: T8 gap closed is the share of the clearer wording’s action "
            "change a build reproduces on held-aside T8 decisions; T5 deviation is how far the build moves the action "
            "chunk from the unedited policy’s on T5 practice decisions the unedited policy solved (T3 the same way, in the "
            "detail). These decide which builds run closed loop.")
    intro = (f"<p>The T8 edit <code>{esc(P['base'])}</code> breaks {esc(left)}, the two other tasks whose instructions "
             f"say “left”. The Cosmos3 Edge track rebuilds it with zero-change preservation keys: other instructions, "
             f"added as rows of K with zero rows of V, whose layer outputs the solve must keep."
             + "".join(" " + inline(pe(r["name"])["intro"]) for r in P["rows"] if pe(r["name"]).get("intro"))
             + "</p>")
    finding = f'<p>{inline(P["text"]).replace(esc(RUNNING), run)}</p>' if P["text"] else ""
    if not full:
        times = {1: "once", 2: "twice", 3: "three times"}
        n_keep = len([r for r in P["rows"] if pe(r["name"]).get("keys")])
        n_words = len([r for r in P["rows"] if r["gate_failed"]])
        intro = ("<p>We rebuilt the edit " + " and ".join(x for x in (
                     f"{times.get(n_keep, f'{n_keep} times')} telling it to leave other instructions unchanged" if n_keep else "",
                     f"{times.get(n_words, f'{n_words} times')} copying only the words the two wordings share" if n_words else "")
                     if x)
                 + ". Rows without a tag are from unseen starts. “In replays” means replaying recorded moments with the "
                 "edited weights, before any run on the robot: the share of the clear wording’s effect on the robot’s "
                 "planned actions that the edit reproduces.</p>")
        finding = prevent_plain(P)
        finding = f'<p>{esc(finding).replace(esc(RUNNING), run)}</p>' if finding else ""
        return intro + f'<div class="tblock wide">{table}</div>' + finding
    return (intro + f'<div class="tblock wide">{table}{sources_html(P["sources"])}<p class="tnote">{esc(note)}</p></div>'
            + finding)


def prevent_plain(P):
    """The main flow's sentences on the rebuilds, in plain words: what kept the fix, what did not stop the damage,
    what replays predicted, and what copying only the shared words shows."""
    base = next((r for r in P["rows"] if r["name"] == P["base"]), None)
    out, got = [], []
    for r in P["rows"]:
        if r is base:
            continue
        res = prevent_parts(r, base, "")
        if res and res["left"]:
            got.append((r, res))
    if got:
        t8s = [r["cells"][P["target"]] for r, _ in got]
        kept = all(c["s"] == "ok" and c.get("p") is not None and c["p"] < SIG and c["b"] > c["a"] for c in t8s)
        prac = all(r["cells"][t].get("prac") for r, res in got for t, *_ in res["left"])
        fails = all(any(x[1] in FAILED for x in res["left"]) for _, res in got)
        lead = ("Telling the edit to leave other instructions unchanged "
                + (("kept the fix" + (" on practice starts" if all(c.get("prac") for c in t8s) else "") + " ("
                    + and_list(f"{c['b']}" for c in t8s) + f" of {t8s[0]['n']}) but ") if kept else "")
                + ("did not save the “left” tasks: " if fails else "changed the “left” tasks: "))
        per = []
        for r, res in got:
            bits, last = [], None
            for t, v, _, _, bb in res["left"]:
                c = r["cells"][t]
                verb = "fell" if c["b"] < c["a"] else "rose" if c["b"] > c["a"] else "stayed"
                bits.append((f"{tn(t)} {verb} " if verb != last else f"{tn(t)} ")
                            + (f"from {c['a']} to {c['b']} of {c['n']}" if c["b"] != c["a"] else f"at {c['a']} of {c['n']}")
                            + (" practice starts" if c.get("prac") and not prac else "")
                            + (f" (the plain edit: {bb})" if bb is not None else ""))
                last = verb
            k = pe(r["name"]).get("keys")
            per.append((f"with {k} kept unchanged, " if k else f"{pe(r['name'])['plain'].lower()}, ") + and_list(bits))
        out.append(lead + "; ".join(per) + ".")
    off = prevent_offline_note(P)
    if off:
        out.append(off)
    bd = base["cells"]["dev"] if base else None
    for r in P["rows"]:
        if not r.get("gate_failed"):
            continue
        g, d = r["cells"]["gap"], r["cells"]["dev"]
        s = (f"Copying only the words both wordings share reproduced {g['v']} of the clear wording’s effect in replays"
             + (f", no more than a random edit of the same size ({sgn(g['ctl'], False, 1)}%)"
                if g.get("ctl") is not None and g["ctl"] >= g["x"] - 1 else ""))
        if d["s"] == "ok" and d.get("x3") is not None:
            s += (f", and moved the robot’s planned hand position on the “left” tasks only {d['x']:.1f} and "
                  f"{d['x3']:.1f} cm" + (f" (the plain edit: {bd['x']:.1f} and {bd['x3']:.1f})" if bd and bd["s"] == "ok"
                                         and bd.get("x3") is not None else ""))
        s += (". The fix lives in the instruction’s own words, and so does the damage: they are the same change, so "
              "the robot was not run with it.")
        out.append(s)
    return " ".join(out)


def prevent_offline_note(P):
    """A rebuild that failed in closed loop although offline it moved the left tasks' actions far less than the
    original edit: the offline check did not predict the closed loop."""
    base = next((r for r in P["rows"] if r["name"] == P["base"]), None)
    bd = base["cells"]["dev"] if base else None
    for r in P["rows"]:
        if r is base:
            continue
        res = prevent_parts(r, base, "")
        d = r["cells"]["dev"]
        if not res or d["s"] != "ok" or not bd or bd["s"] != "ok":
            continue
        failed = {x[0] for x in res["left"] if x[1] in FAILED}
        pairs = [(t, d.get(k), bd.get(k)) for t, k in (("T5", "x"), ("T3", "x3")) if t in failed]
        pairs = [(t, x, y) for t, x, y in pairs if x is not None and y and x / y < 0.9]
        if pairs:
            who = pe(r["name"]).get("nick") or f"`{r['name']}`"
            return (f"In replays, {who} had moved the robot’s planned hand position on "
                    + and_list(tn(t) for t, _, _ in pairs) + " only " + and_list(f"{x:.1f}" for _, x, _ in pairs)
                    + " cm from the unedited robot’s, against " + and_list(f"{y:.1f}" for _, _, y in pairs)
                    + " cm for the plain edit, yet the tasks still broke: replays did not predict the robot.")
    return ""


def prevent_tldr(P):
    """For the TL;DR's side-effect bullet, past tense: the follow-up builds with a closed-loop T5 or T3 result, in one
    compact sentence; says plainly when they did not prevent the break; nothing while none has such a result."""
    base = next((r for r in P["rows"] if r["name"] == P["base"]), None)
    got, pending = [], False
    for r in P["rows"]:
        if r["name"] == P["base"]:
            continue
        res = prevent_parts(r, base, "the original edit")
        pending |= any(r["cells"][t]["s"] == "running" for t in (P["target"], *P["left"]))
        if res and res["left"]:
            got.append((r, res))
    if not got:
        return ""
    fails = [(r, res) for r, res in got if any(x[1] in FAILED for x in res["left"])]
    cells = [r["cells"][t] for r, res in got for t, *_ in res["left"]]
    prac = all(c.get("prac") for c in cells)
    keyed = all(pe(r["name"]).get("keys") for r, _ in got)

    def left_txt(r, res):
        out, last = [], None
        for t, v, _, _, bb in res["left"]:
            c = r["cells"][t]
            a, b, n = c["a"], c["b"], c["n"]
            verb = "fell" if b < a else "rose" if b > a else "stayed"
            k = "" if prac else kind_of(c) + " starts"
            x = (f"from {a} to {b} of {n}{k}" if b != a else f"at {a} of {n}{k}")
            out.append((f"{t} {verb} " if verb != last else f"{t} ") + x + (f" (the original edit: {bb})"
                                                                          if bb is not None else ""))
            last = verb
        return and_list(out)

    def who(r):
        e = pe(r["name"])
        return f"with keys from {e['keys']}" if keyed else f"with {e['short']}"

    t8s = [r["cells"][P["target"]] for r, _ in got]
    kept = all(res["t8"] and res["t8"][0] == "kept" for _, res in got)
    t8 = (" kept T8 (" + and_list(f"{c['a']} → {c['b']}" for c in t8s) + f" of {t8s[0]['n']}"
          + (" practice starts)" if all(c.get("prac") for c in t8s) else ")")
          if kept and len({c["n"] for c in t8s}) == 1 else "")
    lead = "Rebuilt with preservation keys, the edit" if keyed else "Rebuilt, the edit"
    if fails and len(fails) == len(got):
        verb = "has not prevented the break so far" if pending else "did not prevent the break"
        return (f" {lead}{t8}{' but' if t8 else ''} {verb}{' on practice starts' if prac else ''}: "
                + "; ".join(f"{who(r)}, {left_txt(r, res)}" for r, res in got) + ".")
    return " " + " ".join(f"Rebuilt {who(r)}: " + (res["t8"][2] + "; " if res["t8"] else "") + left_txt(r, res) + "."
                          for r, res in got)


def prevent_conclusion(P):
    """Conclusion clauses, present tense: each follow-up build with a closed-loop result."""
    base = next((r for r in P["rows"] if r["name"] == P["base"]), None)
    out = []
    for r in P["rows"]:
        if r["name"] == P["base"]:
            continue
        res = prevent_parts(r, base, "the original edit")
        if res:
            out.append(present_clause(f"rebuilt with {pe(r['name'])['short']}, the edit", res)
                       .replace("p = ", "p\u00a0=\u00a0"))
    return out


# ---------------------------------------------------------------- tables
TABLES = [
    ("failure_map", "Failure map", "Success of the unedited model on each task, and how it fails where diagnosed."),
    ("render", "Rendering sensitivity",
     "The unedited model on the same starts under RoboLab's performance render preset and its default preset."),
    ("wording", "Wording sensitivity",
     "The unedited model under the original and the clear wording of the same goal, on the same starts."),
    ("edit_gain", "Edits on unseen starts",
     "Unedited → edited on held-out starts never used to build the edit: fixed / broke pairs, exact McNemar p, paired "
     "bootstrap 95% CI of the difference."),
    ("random_control", "Random-direction control",
     "The edit's layers, rank and singular values in random directions, the same size, scored on the same starts."),
    ("recovery", "Teacher recovery",
     "(edited − unedited) / (teacher − unedited), where the teacher is the unedited model given the clear wording."),
    ("where", "Where to edit",
     "The same distillation written into the backbone / text pathway or into the action pathway, on the same starts."),
    ("size", "Edit size", "Values changed, rank per layer, and share of the parameters, also as low-rank factors."),
    ("side_effects", "Side-effect map",
     "Change in successes, edited minus unedited, on every suite task, paired on the same slots; significant at exact "
     "p < 0.05."),
    ("retention", "Retention", "After the edit: a task the model already solves, and the clear wording itself."),
    ("noise", "Noise floor", "Paired outcomes that differ when identical weights are rerun on the same starts; it sets "
                             "the smallest detectable change."),
]
TABLE_IDS = [t[0] for t in TABLES]
TASK_TABLES = ("failure_map", "wording", "edit_gain", "random_control", "recovery", "side_effects")
ROW_ORDER = {
    "edit_gain": ["T1", "T1b", "T1c", "T2", "T3", "T4", "T5", "T6", "T7", "T8"],
    "size": ["T1", "T1v", "T1c", "T8"],
    "where": ["starts", "unedited", "backbone", "action", "both", "offline_where", "transfers", "size"],
    "retention": ["R1", "clear", "offline"],
    "noise": ["unedited2", "edited2", "butter", "typical", "mde"],
}
PI_ONLY = {"T1b", "T1c", "T1v", "clear", "offline", "edited2", "butter", "typical"}

NOTES = {
    "render": "The performance preset renders without the RTX denoiser, which leaves speckle noise on every surface; the "
              "default preset is NVIDIA's validated configuration. All three models are compared under the default "
              "preset. The paper’s evaluation of π0.5 used the performance preset.",
    "failure_map": "π0.5's cells are from the shared slots where it was run there, else from the paper’s evaluation "
                   "(the largest run, on its own slots). Its unedited rate on T1 ranged from 8% to 21% across six start "
                   "sets, so compare models on the shared slots.",
    "wording": "Reference-first describes the placement from the reference object's side, target-first from the moved "
               "object's side. In the Same Goal, Different Words study ⟦abbas2026samegoal⟧ (8 starts per scene, the "
               "same starts for every "
               "wording, not this page's slots), reference-first wording lowered stable placement by 8.1 points for "
               "Cosmos3 Edge across 12 goals; "
               "on this goal Edge placed 4 of 8 under target-first and 0 of 8 under reference-first.",
    "edit_gain": "Distilled edits copy the clear wording's internal effect (teal in the figures); the blind editors "
                 "designed their own edits under the same rules (dark). The packaged tool's 2 × 2 table happens to equal "
                 "the T1 edit's transfer to T3 (8 → 36 of 64, 33 / 5): different tasks, different starts.",
    "random_control": "The control keeps every layer's singular values and replaces the singular vectors with random "
                      "orthonormal bases (seed 0), so its Frobenius norm matches the edit's in every layer.",
    "recovery": "100% would mean the edit does everything the clear wording does while the original wording stays.",
    "where": "In π0.5 the VLM's weights process image and text tokens alike, and the edit is keyed at the image tokens. "
             "In Cosmos3 the video and action tokens have their own weights (the generation pathway), so its closest "
             "counterpart is the video-action pathway; the text pathway is the other candidate. In GR00T the backbone is "
             "Cosmos-Reason2-2B and the action pathway is the flow-matching DiT action head.",
    "size": "A low-rank update is dense in its weight matrix: the 0.16% variant changes 7.4% of stored values; as factors "
            "it is 0.16% of the parameters.",
    "side_effects": "Each model's main edit, unchanged, on every suite task; π0.5's is the T1 edit. In the paper’s "
                    "evaluation it harmed none of fifteen other surveyed tasks detectably.",
    "retention": "At these sample sizes a retention check rules out only large losses (see the noise floor).",
    "noise": "Simulated physics on GPUs is not bit-repeatable, and a policy that executes many actions between looks "
             "amplifies small differences. Every comparison pairs starts and uses exact tests for that reason.",
}


def merge_cell(tid, key, fresh, track_prov, mine):
    """A shared-slot (fresh) cell wins and names the earlier record; otherwise the earlier record, tagged as such.
    The track's own provisional rows fill the failure map and side-effect map; elsewhere the records read here do."""
    earlier = None
    if track_prov and tid in ("failure_map", "side_effects"):
        earlier = dict(track_prov)
        if mine and tid == "failure_map":
            how = TASK.get(key, (None,) * 5)[4]
            earlier["d"] = "; ".join(x for x in (earlier.get("d"), how if key in TASK and how != "already solved" else None) if x)
    earlier = earlier or (dict(mine) if mine else (dict(track_prov) if track_prov else None))
    if fresh and fresh["s"] == "ok":
        c = dict(fresh)
        if earlier and earlier["s"] == "ok":
            c["d"] = "; ".join(x for x in (c.get("d"), f"paper’s evaluation: {earlier['v']}") if x)
        return c
    if earlier:
        if earlier["s"] == "ok":
            earlier["prov"] = True
        return earlier
    return None


def cross_check(prov, rcp):
    """Numbers that both the records read here and the pi0.5 track's provisional file give must agree."""
    bad = []
    for tid in ("edit_gain", "side_effects", "wording"):
        for key, mine in prov[tid].items():
            theirs = rcp["pi05"][tid].get(key)
            if not theirs or mine["s"] != "ok" or theirs["s"] != "ok":
                continue
            fields = [f for f in ("a", "b", "n", "delta") if f in mine and f in theirs]
            if any(mine[f] != theirs[f] for f in fields):
                bad.append(f"{tid}/{key}: {mine['v']} here vs {theirs['v']} in metrics_pi05.json")
    for b in bad:
        print("  CHECK", b)
    return bad


# Experiments a track has dropped, with the reason and where it says so. Their empty cells read "not run", not
# "running"; a measured cell still wins if one lands.
NOT_RUN = {
    ("groot", "side_effects"): ("not run", "dropped: both T1 edits lowered success on T1 itself, so no side-effect map "
                                           "was run for them (cross-groot/v1 639a29b)"),
}


def build_tables(P, cross):
    prov, labels = pi05_cells(P)

    def subset(src, provisional):
        return dict(src, rows=[r for r in src["rows"] if r["_prov"] == provisional]) if src else None

    rc = {mid: row_cells(subset(cross.get(mid), False)) for mid in ORDER}   # shared-slot runs
    rcp = {mid: row_cells(subset(cross.get(mid), True)) for mid in ORDER}   # a track's earlier records
    cross_check(prov, rcp)
    # fresh pi0.5 cells replace the earlier records only where they cover the same comparison: a where-to-edit with
    # fewer places than the earlier one does not, nor does one sample size's detectable effect replace the whole curve
    locs = lambda d: sum(k in d for k in ("backbone", "action", "both"))
    if locs(rc["pi05"]["where"]) < locs(prov["where"]):
        rc["pi05"]["where"] = {}
    if "mde" in prov["noise"]:
        rc["pi05"]["noise"].pop("mde", None)
    tables = []
    order = {t[0]: i for i, t in enumerate(TASKS)}
    rcell = render_cells(cross)
    for tid, title, definition in TABLES:
        if tid == "render":
            row = {"key": "render", "label": "Same starts: performance → default preset", "cells": {}}
            for mid in ORDER:
                src = cross.get(mid)
                row["cells"][mid] = rcell.get(mid) or (NA if (src and src["final"]) else RUN)
            srcs = sorted({x for c in row["cells"].values() for x in c.get("src", [])})
            tables.append(dict(id=tid, title=title, definition=definition, rows=[row], note=NOTES.get(tid), sources=srcs))
            continue
        keys = list(dict.fromkeys(ROW_ORDER.get(tid, []) + list(prov[tid])
                                  + [k for m in ORDER for k in rc[m][tid] if k in TASK]))
        if tid in TASK_TABLES:
            pos = {k: i for i, k in enumerate(ROW_ORDER.get(tid, []))}
            keys.sort(key=lambda k: (order.get(k[:2], 99), pos.get(k, 0), k))
        if tid == "side_effects":
            keys = [t[0] for t in TASKS]
        rows = []
        for key in keys:
            cells = {}
            for mid in ORDER:
                src = cross.get(mid)
                cell = merge_cell(tid, key, rc[mid][tid].get(key), rcp[mid][tid].get(key),
                                  prov[tid].get(key) if mid == "pi05" else None)
                if cell is None and (mid, tid) in NOT_RUN and not key.startswith("_"):
                    cell = C(NOT_RUN[(mid, tid)][0], NOT_RUN[(mid, tid)][1], s="na")
                if cell is None:
                    cell = NA if (key in PI_ONLY or (src and src["final"]) or (mid == "pi05" and key == "offline_where")) else RUN
                cells[mid] = cell
            label = labels.get(key) or (task_label(key) if key in TASK else key)
            if tid == "size" and key == "T1c":
                label = labels["T1c_size"]
            rows.append({"key": key, "label": label, "cells": cells})
        for mid in ORDER:  # free-form rows only one model has (e.g. its retention notes)
            for key, cell in sorted(rc[mid][tid].items()):
                if key.startswith("_") and key not in ("_bars", "_edit"):
                    rows.append({"key": key + mid, "label": f"{MODELS[mid]['name']}: the {key[1:]} edit",
                                 "cells": {m: (cell if m == mid else NA) for m in ORDER}})
        srcs = sorted({s for r in rows for c in r["cells"].values() for s in c.get("src", [])})
        t = dict(id=tid, title=title, definition=definition, rows=rows, note=NOTES.get(tid), sources=srcs)
        if tid == "side_effects":
            t["edit"] = {mid: (rc[mid]["side_effects"].get("_edit") or {}).get("v")
                         or (labels["side_effects_edit"] if mid == "pi05" else None)
                         or (rcp[mid]["side_effects"].get("_edit") or {}).get("v") for mid in ORDER}
        tables.append(t)
    return tables, rc, prov, rcp


def summary_table(P, tables, cross, rc):
    """The key table: for each task, what weight-space programming did on unseen starts, per model."""
    e, c, w, s4ho, bl, tx = P["edit"], P["ctl"], P["where"], P["s4ho"], P["blind"], P["where"]["tx"]
    pi = {
        "T1": C(f"fixed: {e['base']} → {e['cand']} of {e['n']}", f"distilled edit; {e['fixed']} / {e['broke']}, p = {fp(e['p'])}"),
        "T2": C(f"T1 edit: {P['eye']['base']} → {P['eye']['cand']} of {P['eye']['n']}",
                f"own edit {bl['T2']['base']} → {bl['T2']['cand']} of {bl['T2']['n']}, not significant"),
        "T3": C(f"T1 edit: {P['left']['base']} → {P['left']['cand']} of {P['left']['n']}", "no own edit submitted"),
        "T4": C("no gain", "lookalike perception; no clear wording"),
        "T5": C("no gain", f"own edit {bl['T5']['base']} → {bl['T5']['cand']} of {bl['T5']['n']}; the grasp slips"),
        "T6": C("no gain", f"own edit {bl['T6']['base']} → {bl['T6']['cand']} of {bl['T6']['n']}"),
        "T7": C("no gain", "a missing skill; 0 of 64 unedited"),
        "T8": C(f"fixed: {s4ho['base']} → {s4ho['cand']} of {s4ho['n']}", f"distilled edit; {s4ho['fixed']} / {s4ho['broke']}, "
                                                                          f"p = {fp(s4ho['p'])}"),
        "R1": C(f"kept: {tx['cube']['base']} → {tx['cube']['arms']['v1full']} of {tx['cube']['n']}", "cube into the bowl, T1 edit"),
        "control": C(f"{c['rand']} of {c['n']}", f"T1; unedited {c['base']}, edit {c['edit']}"),
        "where": C(f"VLM {w['arms']['v1full']}, action expert {w['arms']['a1']}, output projection {w['arms']['h1']} of "
                   f"{w['n']}",
                   f"T1; unedited {w['arms']['base']}; only the VLM edit transfers"),
    }
    rows = []
    for tid, *_ in TASKS:
        r = {"key": tid, "label": task_label(tid), "cells": {}}
        for mid in ORDER:
            src = cross.get(mid)
            g = rc[mid]["edit_gain" if tid != "R1" else "retention"].get(tid)
            u = rc[mid]["failure_map"].get(tid)
            if g:
                bits = [f"{g['fx']} / {g['br']}" if g.get("fx") is not None and g.get("br") is not None else None,
                        f"p = {fp(g['pv'])}" if g.get("pv") is not None else None]
                cell = C(g["v"], ", ".join(x for x in bits if x) or None, prac=g.get("prac"))
                if mid == "pi05":
                    cell["d"] = "; ".join(x for x in (cell.get("d"), f"paper’s evaluation: {pi[tid]['v']}") if x)
            elif mid == "pi05":
                se = rc[mid]["side_effects"].get(tid) or {}
                fresh_se = se.get("s") == "ok" and not se.get("target")
                m_et = re.match(r"the ([TR]\d) edit", (rc[mid]["side_effects"].get("_edit") or {}).get("v") or "")
                same = fresh_se and m_et and (pi[tid]["v"].startswith(f"{m_et.group(1)} edit")
                                              or (tid == "R1" and f"{m_et.group(1)} edit" in (pi[tid].get("d") or "")))
                if same:   # a fresh run of the same comparison replaces the earlier record, which moves to the detail
                    cell = C(f"{m_et.group(1)} edit: {se['a']} → {se['b']} of {se['n']}" if tid != "R1" else
                             f"{'kept' if se['b'] >= se['a'] else 'lowered'}: {se['a']} → {se['b']} of {se['n']}",
                             "; ".join(x for x in (se.get("d", "").split("; slots")[0].replace("; practice starts", ""),
                                                   f"paper’s evaluation: {pi[tid]['v']}") if x), prac=se.get("prac"))
                else:
                    cell = dict(pi[tid], prov=True)
                    if u:
                        cell["d"] = "; ".join(x for x in (cell.get("d"), f"shared slots, unedited: {u['v']}") if x)
                    if fresh_se and m_et:
                        cell["d"] = "; ".join(x for x in (cell.get("d"), f"shared slots, {m_et.group(1)} edit: "
                                                          f"{se['a']} → {se['b']} of {se['n']}") if x)
            elif (rc[mid]["side_effects"].get(tid) or {}).get("s") == "ok" and not rc[mid]["side_effects"][tid].get("target"):
                se = rc[mid]["side_effects"][tid]
                m_et = re.match(r"the ([TR]\d) edit", (rc[mid]["side_effects"].get("_edit") or {}).get("v") or "")
                cell = C(f"{m_et.group(1) + ' edit' if m_et else 'edit'}: {se['a']} → {se['b']} of {se['n']}",
                         se.get("d", "").split("; slots")[0].replace("; practice starts", "") or None, prac=se.get("prac"))
            elif u:
                cell = C(f"unedited {u['v']}", "not edited" if (src and src["final"]) else "not edited yet")
            else:
                cell = NA if (src and src["final"]) else RUN
            r["cells"][mid] = cell
        rows.append(r)
    rtab = next(t for t in tables if t["id"] == "render")["rows"][0]["cells"]
    rows.append({"key": "render", "label": "Rendering: performance → default preset", "sub": True,
                 "cells": {m: (C(rtab[m]["v"], rtab[m].get("task")) if rtab[m]["s"] == "ok" else rtab[m]) for m in ORDER}})
    for key, label, tid in (("control", "Random control, same size", "random_control"), ("where", "Where to edit", "where")):
        r = {"key": key, "label": label, "cells": {}, "sub": True}
        for mid in ORDER:
            src = cross.get(mid)
            got = rc[mid][tid]
            if key == "control" and got:
                k0 = (PREVENT["target"] if mid == PREVENT["model"] and PREVENT["target"] in got else sorted(got)[0])
                r["cells"][mid] = C(got[k0]["v"], k0, prac=got[k0].get("prac"))
            elif key == "where" and got.get("_bars"):
                wb = got["_bars"]
                r["cells"][mid] = C(", ".join(f"{loc_name(mid, loc)[4:]} {k}" for loc, k, _ in wb["bars"])
                                    + (f" of {wb['n']}" if wb.get("n") else ""),
                                    f"{wb['task']}, {wb['n']}{wb.get('kind') or ''} starts; unedited {wb['base']}")
            elif key == "where" and any(k in got for k in ("backbone", "action", "both")):
                r["cells"][mid] = C("; ".join(got[k]["v"] for k in ("backbone", "action", "both") if k in got))
            elif mid == "pi05":
                r["cells"][mid] = dict(pi[key], prov=True)
            else:
                r["cells"][mid] = NA if (src and src["final"]) else RUN
        rows.append(r)
    return dict(id="summary", title="Where weight-space programming fixed failures", rows=rows)


# ---------------------------------------------------------------- HTML (mirrored by cosmos.js)
def esc(s):
    return html.escape(s or "", quote=True)


def inline(s):
    return re.sub(r"`([^`]+)`", r"<code>\1</code>", esc(s))


def td_html(cell, label):
    """One cell; label names its column on small screens, where each row becomes a block."""
    cls = {"ok": "c", "running": "c st-run", "na": "c st-na"}[cell["s"]] + (" prov" if cell.get("prov") else "")
    tag = (
        ' <span class="tag">practice starts</span>' if cell.get("prac") else "")
    inner = f'<span class="v">{inline(cell["v"])}{tag}</span>'
    if cell.get("d"):
        inner += f'<span class="d">{inline(cell["d"])}</span>'
    return f'<td class="{cls}" data-model="{esc(label)}">{inner}</td>'


def cell_html(cell, mid, models):
    return td_html(cell, models[mid]["name"])


def se_models(t):
    """The side-effect map's columns: the models with at least one measured cell beyond their own task."""
    return [m for m in ORDER if any(r["cells"][m].get("s") == "ok" and "delta" in r["cells"][m]
                                    and not r["cells"][m].get("target") for r in t["rows"])]


def se_left_out(t, models):
    """One sentence per model left out of the side-effect map: no edit passed its own task, so nothing to map."""
    out = []
    for m in ORDER:
        if m in se_models(t):
            continue
        own = next((r["cells"][m] for r in t["rows"] if r["cells"][m].get("target") and r["cells"][m].get("s") == "ok"), None)
        k = next((r["key"] for r in t["rows"] if r["cells"][m].get("target")), None)
        out.append(f"No {models[m]['name']} edit passed its own task"
                   + (f" ({dename(TASK_NAME.get(k, k))}: {own['a']} → {own['b']} of {own['n']}"
                      f"{' practice' if own.get('prac') else ''} starts)" if own and "a" in own else "")
                   + ", so there was nothing to map.")
    return " ".join(out)


def table_html(t, models, mids=None):
    mids = mids or ORDER
    head = "".join(f'<th scope="col"><span class="mn">{esc(models[m]["name"])}</span>'
                   f'<span class="mk">{esc(models[m]["short"])}</span></th>' for m in mids)
    body = []
    for r in t["rows"]:
        cls = ' class="sub"' if r.get("sub") else ""
        body.append(f'<tr{cls}><th scope="row">{inline(r["label"])}</th>'
                    + "".join(cell_html(r["cells"][m], m, models) for m in mids) + "</tr>")
    return (f'<table class="ctable"><thead><tr><th scope="col"><span class="vh">Row</span></th>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table>')


def gh(rel):
    """A record's GitHub URL; 'branch:path' points into a branch that is not merged yet."""
    branch, _, rel = rel.rpartition(":") if ":" in rel else ("main", "", rel)
    return f"{REPO}/blob/{branch}/{rel}" if Path(rel).suffix else f"{REPO}/tree/{branch}/{rel}"


def sources_html(srcs):
    if not srcs:
        return ""
    links = ", ".join(f'<a href="{esc(gh(s))}"><code>{esc(s.split(":")[-1].replace("study/results/", ""))}</code></a>'
                      for s in srcs)
    return f'<p class="src">Records: {links}</p>'


def block_html(t, models):
    mids = se_models(t) if t["id"] == "side_effects" else None
    lo = se_left_out(t, models) if t["id"] == "side_effects" else ""
    return (table_html(t, models, mids) + sources_html(t.get("sources"))
            + (f'<p class="tnote">{inline(t["note"])}</p>' if t.get("note") else "")
            + (f'<p class="tnote">{esc(lo)}</p>' if lo else ""))


def replace_region(text, name, content):
    pat = re.compile(rf"(<!-- gen:{re.escape(name)} -->)(.*?)(<!-- /gen:{re.escape(name)} -->)", re.S)
    if not pat.search(text):
        raise KeyError(f"marker gen:{name} not found")
    return pat.sub(lambda m: m.group(1) + content + m.group(3), text, count=1)


# ---------------------------------------------------------------- figures
THEMES = {
    "light": dict(bg="#ffffff", ink="#1a1915", ink2="#4a4944", mute="#7a7873", rule="#d6d3cb", grid="#efede7",
                  teal="#147d92", grey="#bdbab3", dark="#3b3a36", ring="#ffffff", heat0="#f4f2ed", heat1="#3b3a36",
                  empty="#f7f6f2", on_dark="#ffffff"),
    "dark": dict(bg="#201f1d", ink="#ecebe6", ink2="#c4c1ba", mute="#9c9993", rule="#4a4844", grid="#2d2c29",
                 teal="#2aa5bc", grey="#5c5a55", dark="#d4d2cb", ring="#201f1d", heat0="#2a2927", heat1="#d4d2cb",
                 empty="#252422", on_dark="#141413"),
}
# "page": the web page (shown about 770 px wide, 13 px text); "brief": half a Letter page in the PDF
LAYOUTS = {
    "page": dict(W=8.0, FS=9.8, FS_S=8.8, FS_H=10.2, left=3.2, right=1.35, row=0.25, top=0.44, bottom=0.4,
                 lx=0.12, lx2=0.32, ms=7.5, lw=2.2, bar=0.21, leg_ms=7),
    "brief": dict(W=3.72, FS=6.8, FS_S=6.2, FS_H=7.1, left=1.5, right=0.66, row=0.15, top=0.36, bottom=0.31,
                  lx=0.02, lx2=0.1, ms=4.6, lw=1.5, bar=0.21, leg_ms=4.8),
}
FIGNAME = {"pi05": "π0.5", "cosmos": "Cosmos3 Edge", "groot": "GR00T N1.7"}
WHERE_REF = {}
WHERE_OTHER = {}


def setup_mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                         "svg.fonttype": "path", "svg.hashsalt": "weightsmith-cross", "axes.unicode_minus": True})
    import matplotlib.text as mtext
    if not getattr(mtext.Text.set_text, "_dename", False):   # every figure label through the one task-name mapping
        orig = mtext.Text.set_text

        def set_text(self, s):
            return orig(self, dename(s) if isinstance(s, str) else s)
        set_text._dename = True
        mtext.Text.set_text = set_text
    return plt


def rows_figure(rows, L, row=None, hf=None):
    """One unit of height per row (hf: units per row kind); heads get a quarter unit of space above."""
    plt = setup_mpl()
    row = row or L["row"]
    hf = hf or {}
    units = sum(1.25 if k == "head" else hf.get(k, 1.0) for k, *_ in rows)
    H = L["top"] + L["bottom"] + units * row
    fig = plt.figure(figsize=(L["W"], H))
    ax = fig.add_axes([L["left"] / L["W"], L["bottom"] / H, (L["W"] - L["left"] - L["right"]) / L["W"], units * row / H])
    ys, y = [], 0.0
    for k, *_ in rows:
        u = 1.0 if k == "head" else hf.get(k, 1.0)
        if k == "head":
            y += 0.25
        ys.append(y + u / 2)
        y += u
    ax.set_ylim(units, 0)
    return fig, ax, ys, H


def style_axes(ax, T, L, xmin=0, xmax=100, xticks=(0, 25, 50, 75, 100), xlabel=None, fmt=lambda v: f"{v:g}"):
    ax.set_xlim(xmin, xmax)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color(T["rule"])
    ax.spines["bottom"].set_linewidth(0.8)
    ax.set_yticks([])
    ax.set_xticks(list(xticks))
    ax.set_xticklabels([fmt(v) for v in xticks])
    ax.tick_params(axis="x", colors=T["mute"], labelsize=L["FS_S"], length=3, width=0.7, pad=2)
    for x in xticks:
        ax.axvline(x, color=T["grid"], lw=0.8, zorder=0)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=L["FS_S"], color=T["mute"], labelpad=3)
    ax.patch.set_alpha(0)


def put(fig, ax, L, x_in, y, text, T, color="ink", size=None, weight="normal", ha="left", style="normal"):
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transData)
    ax.text(x_in / L["W"], y, text, transform=tr, ha=ha, va="center", fontsize=size or L["FS"], weight=weight,
            color=T[color], clip_on=False, style=style)


def wrap_in(text, width_in, pt):
    """Line breaks so that text in Arial at pt points fits about width_in inches (average glyph 0.52 em)."""
    import textwrap
    return "\n".join(textwrap.wrap(str(text), max(8, int(width_in / (0.52 * pt / 72))), break_long_words=False))


def col_head(fig, ax, L, T, text):
    """What the right-hand column of numbers is, above its first row."""
    import matplotlib.transforms as mtransforms
    tr = mtransforms.blended_transform_factory(fig.transFigure, ax.transAxes)
    ax.text((L["W"] - L["right"] + 0.12) / L["W"], 1.0, text, transform=tr, ha="left", va="bottom",
            fontsize=L["FS_S"] - 0.6, color=T["mute"], clip_on=False, linespacing=1.15)


def legend(fig, items, T, H, L, ncol=None):
    from matplotlib.lines import Line2D
    ms = L["leg_ms"]
    hs = []
    for kind, lab in items:
        if kind == "ring":
            hs.append(Line2D([], [], marker="o", ls="", ms=ms, mfc=T["ring"], mec=T["dark"], mew=1.4, label=lab))
        elif kind.startswith("ci-"):
            hs.append(Line2D([], [], color=T[kind[3:]], lw=L["lw"] * 0.9, marker="o", ms=ms * 0.85, mfc=T[kind[3:]],
                             mec=T["ring"], mew=1.0, label=lab))
        elif kind.startswith("bar-"):
            hs.append(Line2D([], [], marker="s", ls="", ms=ms * 1.1, mfc=T[kind[4:]], mec="none", label=lab))
        elif kind.startswith("mk-"):
            mk, col = kind[3:].split(":")
            hs.append(Line2D([], [], marker=mk, ls="", ms=ms * 0.85, mfc=T[col], mec=T["ring"], mew=1.0, label=lab))
        else:
            hs.append(Line2D([], [], marker="o", ls="", ms=ms, mfc=T[kind], mec=T["ring"], mew=1.2, label=lab))
    leg = fig.legend(handles=hs, loc="lower left", bbox_to_anchor=(L["left"] / L["W"] - 0.004, 1 - (L["top"] - 0.12) / H),
                     ncol=ncol or len(items), frameon=False, fontsize=L["FS_S"], handletextpad=0.35,
                     columnspacing=1.1, borderaxespad=0, handlelength=1.1)
    for t in leg.get_texts():
        t.set_color(T["ink2"])


CHANGED = set()


def write_if_changed(path, data):
    """Write bytes only when they differ, so unchanged figures keep their mtime and git sees no change."""
    if path.exists() and path.read_bytes() == data:
        return False
    path.write_bytes(data)
    CHANGED.add(path.name)
    return True


def save_fig(fig, stem, T, png=True):
    import io
    import matplotlib.pyplot as plt
    FIG.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(T["bg"])
    buf = io.BytesIO()
    fig.savefig(buf, format="svg", facecolor=T["bg"], metadata={"Date": None})
    write_if_changed(FIG / f"{stem}.svg", buf.getvalue())
    if png:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", facecolor=T["bg"], dpi=192, metadata={"Software": None})
        write_if_changed(FIG / f"{stem}.png", buf.getvalue())
    plt.close(fig)


def grouped(models, cells_by_model, L):
    """Rows of a per-model figure: a head per model, then its rows (or one 'running' row)."""
    rows = []
    for mid in ORDER:
        name = FIGNAME[mid] if L is LAYOUTS["brief"] else models[mid]["name"]
        rows.append(("head", name + (" (control)" if mid == "pi05" else "")))
        items = cells_by_model.get(mid) or []
        rows += [("row", it) for it in items] if items else [("run", "running")]
    return rows


def mix(a, b, t):
    a, b = [int(a[i:i + 2], 16) for i in (1, 3, 5)], [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def lum(hexc):
    r, g, b = [int(hexc[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def model_rows(tb, tid, keys=None):
    out = {m: [] for m in ORDER}
    for r in tb[tid]["rows"]:
        if keys and r["key"] not in keys:
            continue
        for m in ORDER:
            c = r["cells"][m]
            if c["s"] == "ok":
                out[m].append((r, c))
    return out


def fig_failure_map(tb, models, T, L):
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap
    from matplotlib.patches import Rectangle
    plt = setup_mpl()
    brief = L is LAYOUTS["brief"]
    fm = tb["failure_map"]["rows"]
    nrow = len(fm)
    lw = L["left"] - (0.15 if brief else 0.25)
    leg_w = 0.0 if brief else 0.95
    cw = (L["W"] - lw - leg_w - 0.05) / 3
    ch = L["row"] * (1.25 if brief else 1.35)
    top = 0.36 if brief else 0.86
    H = top + nrow * ch + 0.12
    fig = plt.figure(figsize=(L["W"], H))
    ax = fig.add_axes([lw / L["W"], 0.12 / H, 3 * cw / L["W"], nrow * ch / H])
    ax.set_xlim(0, 3)
    ax.set_ylim(nrow, 0)
    ax.axis("off")
    for j, m in enumerate(ORDER):
        if brief:
            ax.text(j + 0.5, -0.2, FIGNAME[m], ha="center", va="bottom", fontsize=L["FS"], weight="bold", color=T["ink"])
        else:
            sub = wrap_in(models[m].get("fig_short") or models[m]["short"], cw - 0.1, L["FS_S"] - 0.6)
            ax.text(j + 0.5, -0.16, sub, ha="center", va="bottom", fontsize=L["FS_S"] - 0.6, color=T["mute"],
                    linespacing=1.1)
            ax.text(j + 0.5, -0.24 - 0.42 * (sub.count("\n") + 1), models[m]["name"], ha="center", va="bottom",
                    fontsize=L["FS"], weight="bold", color=T["ink"])
    for i, r in enumerate(fm):
        lab = SHORT[r["key"]] if r["key"] in SHORT else r["label"]
        put(fig, ax, L, L["lx"], i + 0.5, lab, T)
        for j, m in enumerate(ORDER):
            c = r["cells"][m]
            if c["s"] == "ok" and c.get("n"):
                f = c["k"] / c["n"]
                col = mix(T["heat0"], T["heat1"], f)
                ax.add_patch(Rectangle((j + 0.025, i + 0.06), 0.95, 0.88, fc=col, ec="none"))
                if T["bg"] == "#ffffff":
                    tc = T["on_dark"] if lum(col) < 0.45 else T["ink"]
                else:
                    tc = T["on_dark"] if lum(col) > 0.5 else T["ink"]
                txt = f"{c['k']}/{c['n']} · {rint(100 * f)}%" if brief else f"{c['k']} of {c['n']}  ·  {rint(100 * f)}%"
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=L["FS_S"], color=tc)
            else:
                ax.add_patch(Rectangle((j + 0.025, i + 0.06), 0.95, 0.88, fc=T["empty"], ec="none"))
                ax.text(j + 0.5, i + 0.5, c["v"], ha="center", va="center", fontsize=L["FS_S"], color=T["mute"],
                        style="italic" if c["s"] == "running" else "normal")
    if not brief:
        lx = lw + 3 * cw + 0.22
        cax = fig.add_axes([lx / L["W"], (0.12 + nrow * ch - 1.25) / H, 0.13 / L["W"], 1.25 / H])
        cmap = LinearSegmentedColormap.from_list("g", [T["heat0"], T["heat1"]])
        cax.imshow(np.linspace(1, 0, 64)[:, None], aspect="auto", cmap=cmap, extent=(0, 1, 0, 1))
        cax.set_xticks([])
        cax.yaxis.tick_right()
        cax.set_yticks([0, 0.5, 1])
        cax.set_yticklabels(["0%", "50%", "100%"])
        cax.tick_params(axis="y", colors=T["mute"], labelsize=L["FS_S"] - 0.6, length=2, width=0.6)
        for s in cax.spines.values():
            s.set_visible(False)
        fig.text(lx / L["W"], (0.12 + nrow * ch + 0.1) / H, "share of\nstarts\ncompleted,\nunedited",
                 fontsize=L["FS_S"] - 0.6, color=T["mute"], va="bottom", ha="left", linespacing=1.1)
    return fig


def fig_wording(tb, models, T, L):
    cbm = {m: [(c.get("fl") or r["label"], c["a"], c["b"], c["n"]) for r, c in v if "a" in c]
           for m, v in model_rows(tb, "wording").items()}
    rows = grouped(models, cbm, L)
    fig, ax, ys, H = rows_figure(rows, L)
    style_axes(ax, T, L, xlabel="Task success of the unedited robot: share of starts completed", fmt=lambda v: f"{v:g}%")
    col_head(fig, ax, L, T, "starts completed:\noriginal → clear")
    for (k, *rest), y in zip(rows, ys):
        if k == "head":
            put(fig, ax, L, L["lx"], y, rest[0], T, size=L["FS_H"], weight="bold")
        elif k == "run":
            put(fig, ax, L, L["lx2"], y, rest[0], T, color="mute", style="italic")
        else:
            lab, a, b, n = rest[0]
            put(fig, ax, L, L["lx2"], y, lab, T)
            xa, xb = 100 * a / n, 100 * b / n
            ax.plot([xa, xb], [y, y], color=T["rule"], lw=L["lw"], solid_capstyle="round", zorder=1)
            ax.plot([xa], [y], "o", ms=L["ms"], mfc=T["grey"], mec=T["ring"], mew=1.3, zorder=3)
            ax.plot([xb], [y], "o", ms=L["ms"], mfc=T["ring"], mec=T["dark"], mew=1.6, zorder=4)
            put(fig, ax, L, L["W"] - L["right"] + 0.12, y, f"{a} → {b} of {n}", T, color="ink2", size=L["FS_S"])
    legend(fig, [("grey", "original wording"), ("ring", "clear wording")], T, H, L)
    return fig


def fig_edit_gain(tb, models, T, L):
    brief = L is LAYOUTS["brief"]
    cbm = {m: [((c.get("fb") if brief else None) or c.get("fl") or r["label"], c["ci"], c["a"], c["b"], c["n"],
                c.get("color", "teal"), bool(c.get("prac"))) for r, c in v if c.get("ci")]
           for m, v in model_rows(tb, "edit_gain").items()}
    rows = grouped(models, cbm, L)
    fig, ax, ys, H = rows_figure(rows, L)
    import math
    lows = [x[1][1] if x[1][1] is not None else x[1][0] for v in cbm.values() for x in v]
    k = max(1, math.ceil((-min(lows + [0]) + 2) / 20))       # room on the left for edits that made things worse
    xmin = -20 * k
    ticks = tuple(range(xmin, 101, 20)) if not brief else tuple(range(xmin if xmin < -20 else 0, 101, 25))
    style_axes(ax, T, L, xmin=xmin, xmax=100, xticks=ticks,
               xlabel="change in share of starts completed, edited − unedited, percentage points (95% CI)" if brief
               else "Change in the share of starts completed (edited − unedited), in percentage points, with its 95% "
                    "interval", fmt=lambda v: f"{v:+g}" if v else "0")
    col_head(fig, ax, L, T, "starts completed:\nunedited → edited")
    ax.axvline(0, color=T["mute"], lw=0.9, zorder=1)
    for (k, *rest), y in zip(rows, ys):
        if k == "head":
            put(fig, ax, L, L["lx"], y, rest[0], T, size=L["FS_H"], weight="bold")
        elif k == "run":
            put(fig, ax, L, L["lx2"], y, rest[0], T, color="mute", style="italic")
        else:
            lab, ci, a, b, n, colr, hollow = rest[0]
            d, lo, hi = ci
            put(fig, ax, L, L["lx2"], y, lab, T)
            if lo is not None:
                ax.plot([lo, hi], [y, y], color=T[colr], lw=L["lw"], solid_capstyle="round", zorder=2,
                        dashes=(2.2, 1.6) if hollow else (None, None))
            ax.plot([d], [y], "o", ms=L["ms"], mfc=T["ring"] if hollow else T[colr], mec=T[colr] if hollow else T["ring"],
                    mew=1.6 if hollow else 1.3, zorder=3)
            put(fig, ax, L, L["W"] - L["right"] + (0.06 if brief else 0.12), y, f"{a} → {b} / {n}" if brief else
                f"{a} → {b} of {n}", T, color="ink2", size=L["FS_S"])
    hollow_any = any(len(x) > 6 and x[6] for v in cbm.values() for x in v)
    legend(fig, [("ci-teal", "distilled edit"), ("ci-dark", "other edit")]
           + ([("ring", "on practice starts")] if hollow_any else []), T, H, L)
    return fig


def fig_random_control(tb, models, T, L):
    from matplotlib.patches import Rectangle
    cbm = {m: [(c.get("fl") or r["label"], c["u"], c["r"], c["e"], c["n"], c.get("t"), c.get("tn") or c["n"])
               for r, c in v if "r" in c] for m, v in model_rows(tb, "random_control").items()}
    rows = grouped(models, cbm, L)
    k3 = 2.6
    fig, ax, ys, H = rows_figure(rows, L, hf={"row": k3})
    style_axes(ax, T, L, xlabel="Task success: share of starts completed", fmt=lambda v: f"{v:g}%")
    for (k, *rest), y in zip(rows, ys):
        if k == "head":
            put(fig, ax, L, L["lx"], y, rest[0], T, size=L["FS_H"], weight="bold")
        elif k == "run":
            put(fig, ax, L, L["lx2"], y, rest[0], T, color="mute", style="italic")
        else:
            lab, u, rnd, ed, n, te, tn_ = rest[0]
            put(fig, ax, L, L["lx2"], y, wrap_in(lab, L["left"] - L["lx2"] - 0.15, L["FS"]), T)
            for off, val, col in ((-0.27, u, "grey"), (0.0, rnd, "dark"), (0.27, ed, "teal")):
                x = 100 * val / n
                if x > 0:
                    ax.add_patch(Rectangle((0, y + k3 * (off - L["bar"] / 2)), x, k3 * L["bar"], fc=T[col], ec="none",
                                           zorder=2))
                ax.text(x + 1.2, y + k3 * off, f"{val} of {n}", va="center", ha="left", fontsize=L["FS_S"],
                        color=T["ink2"], zorder=5, bbox=dict(fc=T["bg"], ec="none", pad=0.6))
            if te is not None:
                xt = 100 * te / tn_
                ax.plot([xt, xt], [y - k3 * 0.42, y + k3 * 0.42], color=T["ink"], lw=1.3, zorder=4)
                ax.text(xt + (1.0 if xt < 60 else -1.0), y - k3 * 0.44, f"unedited, clear wording: {te} of {tn_}",
                        fontsize=L["FS_S"] - 0.6, color=T["ink2"], va="bottom", ha="left" if xt < 60 else "right")
    legend(fig, [("bar-grey", "unedited"), ("bar-dark", "random edit, same size"), ("bar-teal", "edit")], T, H, L)
    return fig


def fig_where(tb, models, T, L):
    from matplotlib.patches import Rectangle
    ref = WHERE_REF
    rows = [("head", f"π0.5 (control): T1, {ref['n']} unseen starts"),
            ("bar", "Unedited", ref["base"], ref["n"], "grey")]
    rows += [("bar", lab, k, ref["n"], col) for lab, k, col in ref["bars"]]
    rows.append(("head", f"π0.5: the same edits on other tasks, {ref['tx']['glasses']['n']} starts each"))
    for key, lab in (("glasses", "T2 Eyeglasses"), ("left", "T3 Cube left of bowl"), ("cube", "R1 Cube into bowl (retention)")):
        rows.append(("tx", lab, ref["tx"][key]))
    for mid in ("cosmos", "groot"):
        wb = WHERE_OTHER.get(mid)
        rows.append(("head", models[mid]["name"] + (f": {wb['task']}, {wb['n']}{wb.get('kind') or ''} starts"
                                                    if wb and wb.get("n") else "")))
        if not wb:
            rows.append(("run", "running"))
            continue
        if wb.get("base") is not None:
            rows.append(("bar", "Unedited", wb["base"], wb["n"], "grey"))
        def short(lab):   # "Cosmos-Reason2 backbone", "Text pathway": the model's own words, short enough for the label column
            nm = loc_name(mid, lab)
            return (nm[4:5].upper() + nm[5:]) if nm.startswith("the ") else (lab[:34] + "…" if len(lab) > 35 else lab)
        rows += [("bar", short(lab), k, wb["n"], col) for lab, k, col in wb["bars"]]
    L = dict(L, top=L["top"] + 0.22)
    fig, ax, ys, H = rows_figure(rows, L, row=L["row"] * 1.08)
    style_axes(ax, T, L, xlabel="Task success: share of starts completed", fmt=lambda v: f"{v:g}%")
    for (k, *rest), y in zip(rows, ys):
        if k == "head":
            put(fig, ax, L, L["lx"], y, rest[0], T, size=L["FS_H"], weight="bold")
            if "other tasks" in rest[0]:
                put(fig, ax, L, L["W"] - L["right"] + 0.12, y, "starts completed:\nunedited → VLM edit", T,
                    color="mute", size=L["FS_S"] - 0.6)
        elif k == "run":
            put(fig, ax, L, L["lx2"], y, rest[0], T, color="mute", style="italic")
        elif k == "bar":
            lab, val, n, col = rest
            put(fig, ax, L, L["lx2"], y, lab, T)
            if n:
                x = 100 * val / n
                if x > 0:
                    ax.add_patch(Rectangle((0, y - 0.3), x, 0.6, fc=T[col], ec="none", zorder=2))
                ax.text(x + 1.2, y, f"{val} of {n}", va="center", ha="left", fontsize=L["FS_S"], color=T["ink2"])
        elif k == "tx":
            lab, t = rest
            put(fig, ax, L, L["lx2"], y, lab, T)
            n = t["n"]
            for arm, col, ms, mk in (("base", "grey", 1.0, "o"), ("a1", "dark", 0.85, "s"), ("p1", "dark", 0.85, "D"),
                                     ("v1full", "teal", 1.0, "o")):
                ax.plot([100 * t["arms"][arm] / n], [y], mk, ms=L["ms"] * ms, mfc=T[col], mec=T["ring"], mew=1.2,
                        zorder=3 + (arm == "v1full"))
            put(fig, ax, L, L["W"] - L["right"] + 0.12, y, f"{t['base']} → {t['arms']['v1full']} of {n}", T, color="ink2",
                size=L["FS_S"])
    legend(fig, [("bar-grey", "unedited"), ("bar-teal", "backbone (VLM) edit"), ("bar-dark", "action-side edit"),
                 ("mk-s:dark", "action expert"), ("mk-D:dark", "gated head")], T, H, L, ncol=3)
    return fig


def fig_side_effects(tb, models, T, L):
    """Task x model grid of the edited-minus-unedited change; teal: gain, dark: loss (exact p < 0.05); grey: neither."""
    from matplotlib.patches import Rectangle
    plt = setup_mpl()
    brief = L is LAYOUTS["brief"]
    t = tb["side_effects"]
    rows = t["rows"]
    nrow = len(rows)
    lw = L["left"] - (0.15 if brief else 0.25)
    cols = se_models(t)
    cw = (L["W"] - lw - 0.05) / max(len(cols), 1)
    ch = L["row"] * (1.25 if brief else 1.35)
    top = 0.36 if brief else 0.86
    bottom = 0.44 if brief else 0.98
    H = top + nrow * ch + bottom
    fig = plt.figure(figsize=(L["W"], H))
    ax = fig.add_axes([lw / L["W"], bottom / H, len(cols) * cw / L["W"], nrow * ch / H])
    ax.set_xlim(0, max(len(cols), 1))
    ax.set_ylim(nrow, 0)
    ax.axis("off")
    for j, m in enumerate(cols):
        if brief:
            ax.text(j + 0.5, -0.2, FIGNAME[m], ha="center", va="bottom", fontsize=L["FS"], weight="bold", color=T["ink"])
        else:
            ax.text(j + 0.5, -1.05, models[m]["name"], ha="center", va="bottom", fontsize=L["FS"], weight="bold",
                    color=T["ink"])
            ed = dename((t.get("edit") or {}).get(m) or "")
            hd = re.match(r"the .+? edit(?: on practice starts)?(?= \(|$)", ed) if ed else None
            ed = wrap_in(hd.group(0) if hd else ed or "edit: running", cw - 0.1, L["FS_S"] - 0.6)
            ax.text(j + 0.5, -0.16, ed, ha="center", va="bottom", fontsize=L["FS_S"] - 0.6, color=T["mute"],
                    linespacing=1.1)
    grey = mix(T["empty"], T["grey"], 0.55)
    for i, r in enumerate(rows):
        lab = SHORT[r["key"]] if r["key"] in SHORT else r["label"]
        put(fig, ax, L, L["lx"], i + 0.5, lab, T)
        for j, m in enumerate(cols):
            c = r["cells"][m]
            x0, y0, wd, ht = j + 0.025, i + 0.06, 0.95, 0.88
            if c["s"] == "ok" and "delta" in c:
                sig = c.get("p") is not None and c["p"] < 0.05 and c["delta"] != 0
                col = (T["teal"] if c["delta"] > 0 else T["dark"]) if sig else grey
                ax.add_patch(Rectangle((x0, y0), wd, ht, fc=col, ec="none"))
                if c.get("target"):
                    ax.add_patch(Rectangle((x0 + 0.02, y0 + 0.05), wd - 0.04, ht - 0.1, fc="none", ec=T["bg"], lw=1.1))
                light_fill = lum(col) > 0.5
                tc = T["ink"] if (light_fill and T["bg"] == "#ffffff") else (
                    "#141413" if light_fill else "#ffffff")
                if not sig:
                    tc = T["ink"]
                if brief:
                    txt = f"{dsgn(c['delta'])} / {c['n']}"
                else:
                    txt = f"{dsgn(c['delta'])}: {c['a']} → {c['b']} of {c['n']}" if "a" in c else f"{dsgn(c['delta'])} of {c['n']}"
                txt += "*" if sig else ""
                ax.text(j + 0.5, i + 0.5, txt, ha="center", va="center", fontsize=L["FS_S"], color=tc,
                        weight="bold" if sig else "normal")
            else:
                ax.add_patch(Rectangle((x0, y0), wd, ht, fc=T["empty"], ec="none"))
                ax.text(j + 0.5, i + 0.5, c["v"], ha="center", va="center", fontsize=L["FS_S"], color=T["mute"],
                        style="italic" if c["s"] == "running" else "normal")
    from matplotlib.patches import Patch
    hs = [Patch(fc=T["teal"], ec="none", label="gain, p < 0.05"), Patch(fc=T["dark"], ec="none", label="loss, p < 0.05"),
          Patch(fc=grey, ec="none", label="no detectable change")]
    leg = fig.legend(handles=hs, loc="upper left", bbox_to_anchor=((L["lx"] if brief else lw) / L["W"], (bottom - 0.06) / H), ncol=3,
                     frameon=False, fontsize=L["FS_S"], handlelength=1.0, handleheight=0.9, handletextpad=0.4,
                     columnspacing=1.2, borderaxespad=0)
    for tx_ in leg.get_texts():
        tx_.set_color(T["ink2"])
    fig.text((L["lx"] if brief else lw) / L["W"], (0.1 if brief else 0.28) / H,
             "Each cell: the change in starts completed (edited − unedited): unedited → edited, of the starts tried.\n"
             "* exact McNemar p < 0.05. Outlined: the task the edit was built for."
             + (("\n" + wrap_in(se_left_out(t, models), L["W"] - lw - 0.1, L["FS_S"] - 0.6)) if se_left_out(t, models) else "")
             if not brief else
             "Cell: change in starts completed / starts tried. * p < 0.05; outlined: the edit's own task",
             fontsize=L["FS_S"] - 0.6, color=T["mute"], ha="left", va="center", linespacing=1.3)
    return fig


FIG_FUNCS = {"failure-map": fig_failure_map, "wording-gap": fig_wording, "edit-gain": fig_edit_gain,
             "random-control": fig_random_control, "where-to-edit": fig_where, "side-effects": fig_side_effects}
BRIEF_FIGS = ("failure-map", "side-effects")


def draw_figures(tables, models):
    tb = {t["id"]: t for t in tables}
    for theme in ("light", "dark"):
        T = THEMES[theme]
        for name, fn in FIG_FUNCS.items():
            save_fig(fn(tb, models, T, LAYOUTS["page"]), name + ("-dark" if theme == "dark" else ""), T)
    for name in BRIEF_FIGS:
        save_fig(FIG_FUNCS[name](tb, models, THEMES["light"], LAYOUTS["brief"]), "brief-" + name, THEMES["light"], png=False)

    def say(tid, fmt):
        parts = []
        for m in ORDER:
            its = [x for x in (fmt(r, c) for r, c in model_rows(tb, tid)[m]) if x]
            parts.append(f"{models[m]['name']}: " + ("; ".join(its) if its else "running"))
        return ". ".join(parts) + "."

    ref = WHERE_REF
    return {
        "failure-map": "Unedited success per task and model. " + say("failure_map", lambda r, c: f"{r['label']} {c['v']}"),
        "wording-gap": "Unedited success under the original and the clear wording. " + say(
            "wording", lambda r, c: f"{r['label']}: {c['v']}" if "a" in c else None),
        "edit-gain": "Edited minus unedited success on unseen starts, with 95% intervals. " + say(
            "edit_gain", lambda r, c: f"{c.get('fl') or r['label']}: {c['v']}" if c.get("ci") else None),
        "random-control": "Unedited, random-direction control of the same size, and edit, on the same starts. " + say(
            "random_control", lambda r, c: f"{r['label']}: {c['v']}" if "r" in c else None),
        "where-to-edit": (f"π0.5, T1 on {ref['n']} unseen starts: unedited {ref['base']}, "
                          + ", ".join(f"{lab} {k}" for lab, k, _ in ref["bars"])
                          + f". On other tasks only the VLM edit lifts success: eyeglasses {ref['tx']['glasses']['base']} → "
                          f"{ref['tx']['glasses']['arms']['v1full']}, cube left of the bowl {ref['tx']['left']['base']} → "
                          f"{ref['tx']['left']['arms']['v1full']} of 32; cube into the bowl stays at "
                          f"{ref['tx']['cube']['arms']['v1full']} of 32. "
                          + " ".join(f"{models[m]['name']}: " + ("running." if not WHERE_OTHER.get(m) else ", ".join(
                              f"{lab} {k}" for lab, k, _ in WHERE_OTHER[m]["bars"]) + ".") for m in ("cosmos", "groot"))),
        "side-effects": "Change in successes after each model's main edit, on every task, paired on the same starts. " + say(
            "side_effects", lambda r, c: f"{r['key']} {c['v']}" + (" (significant)" if c.get("p") is not None and
                                                                     c["p"] < 0.05 and c.get("delta") else "")
            if "delta" in c else None),
    }


FIGS = [
    ("failure-map", "Each cell: the starts the unedited robot completed, of the starts tried, and that share; stronger "
                    "shading is a higher share."),
    ("wording-gap", "The share of starts the unedited robot completed under the original wording (grey) and the clear "
                    "wording (ring), from the same starts."),
    ("edit-gain", "How much the share of starts completed changed with each edit, in percentage points (edited minus "
                  "unedited, paired by start), with the paired bootstrap 95% interval. Teal: distilled edits; neutral: "
                  "edits the blind editors designed themselves."),
    ("random-control", "Starts completed by the unedited robot, by a random edit of the same layers, rank and singular "
                       "values, and by the edit, from the same unseen starts; the tick is the unedited robot given the "
                       "clear wording."),
    ("where-to-edit", "Starts completed with the same distillation in different places of each model, from one shared "
                      "set of starts per model; for π0.5 also on other tasks."),
    ("side-effects", "Each model's main edit, unchanged, on every suite task: the change in starts completed, edited minus "
                     "unedited, from the same starts. Teal: a gain; near-black (near-white in dark mode): a loss (exact McNemar p < 0.05); grey: no "
                     "detectable change; the outlined "
                     "cell is the task the edit was built for."),
]


# ---------------------------------------------------------------- clips (MP4s in site/cosmos/video/, <= 2 MB each)
# ---------------------------------------------------------------- diagrams: where the instruction enters, where an edit goes
# Inline SVG drawn here and themed by the page's CSS tokens (classes .dg-*, cosmos.css), so light and dark come from the
# page itself. Every architecture fact is from the records (MODELS above; cross-cosmos/v1 and cross-groot/v1 READMEs;
# study/results/opus/where/README.md; paper Appendix "Where to edit"); every number from the same data as metrics.json.
import math as _math

OUTCOMES = [("fixed", "fixed: clearly better, as good as anywhere we tried"),
            ("partial", "partly fixed: clearly better, but clearly below the best place"),
            ("none", "no effect: no clear change"),
            ("broke", "broke: clearly worse")]


def _t(x, y, s, cls="", anchor=None):
    a = f' text-anchor="{anchor}"' if anchor else ""
    return f'<text x="{x:g}" y="{y:g}" class="{cls}"{a}>{esc(s)}</text>'


def _arrow(pts, cls="dg-ln", head=True):
    """A polyline through pts with a filled head at its last point."""
    d = " ".join(f"{x:g},{y:g}" for x, y in pts)
    out = f'<polyline points="{d}" class="{cls}"/>'
    if head:
        (x1, y1), (x2, y2) = pts[-2], pts[-1]
        L = _math.hypot(x2 - x1, y2 - y1) or 1
        ux, uy = (x2 - x1) / L, (y2 - y1) / L
        bx, by = x2 - 7 * ux, y2 - 7 * uy
        out += (f'<polygon points="{x2:g},{y2:g} {bx - 4 * uy:g},{by + 4 * ux:g} {bx + 4 * uy:g},{by - 4 * ux:g}" '
                f'class="{cls.split()[0]}-h"/>')
    return out


def _mark(cx, cy, kind, r=6.5):
    """Outcome marker: shape and fill differ, so the outcome never rests on colour alone."""
    if kind == "fixed":
        return f'<circle cx="{cx:g}" cy="{cy:g}" r="{r}" class="dg-o-fix"/>'
    if kind == "partial":
        return (f'<circle cx="{cx:g}" cy="{cy:g}" r="{r}" class="dg-o-part"/>'
                f'<path d="M{cx:g},{cy - r:g} A{r},{r} 0 0 0 {cx:g},{cy + r:g} Z" class="dg-o-fill"/>')
    if kind == "broke":
        k = r * 0.5
        return (f'<circle cx="{cx:g}" cy="{cy:g}" r="{r}" class="dg-o-broke"/>'
                f'<path d="M{cx - k:g},{cy - k:g} L{cx + k:g},{cy + k:g} M{cx + k:g},{cy - k:g} L{cx - k:g},{cy + k:g}" '
                f'class="dg-o-x"/>')
    return f'<circle cx="{cx:g}" cy="{cy:g}" r="{r}" class="dg-o-none"/>'


def _letter(cx, cy, s):
    return f'<circle cx="{cx:g}" cy="{cy:g}" r="9.5" class="dg-let"/>' + _t(cx, cy + 4.5, s, "dg-let-t", "middle")


def _svg(vb_w, vb_h, title, desc, body, cls="", ident=""):
    return (f'<svg class="dg {cls}" viewBox="0 0 {vb_w} {vb_h}" role="img" aria-labelledby="{ident}-t {ident}-d">'
            f'<title id="{ident}-t">{esc(title)}</title><desc id="{ident}-d">{esc(desc)}</desc>{body}</svg>')


def dg_schematic():
    """Diagram 1, the edit as a four-step story. Camera frames and the joint state flow into the policy, the
    instruction's words into its VLM backbone or text pathway, which passes its states to the action expert or action
    head, and an action chunk goes to the robot. The still drawing (no script, print) is the last step, with the edit
    in place; cosmos.js plays the steps (DG1_STEPS)."""
    W = []
    # the instruction, in its two wordings (only the words that carry the relation)
    o_words, c_words = ["raisin box", "is to the left of", "mustard bottle"], ["mustard bottle", "is to the right of",
                                                                             "raisin box"]
    co, xo = _chips(o_words, 170, 18, "dg-in")
    cc, xc = _chips(c_words, 170, 18, "dg-chip-c")
    W.append(_t(170, 11, "instruction", "dg-xs dg-mu", "middle"))
    W.append(_st(co, "1=o:1 4=o:1"))
    W.append(_st(cc, "2=o:1 3=o:1", static=False))
    # cameras and joint state, at the left
    cams = ("cosmos_l", "cosmos_r", "cosmos_w")
    for i, key in enumerate(cams):
        W.append(_thumb(10, 70 + 36 * i, 46, 31, key, f"dg1c{i}"))
    W.append(_t(33, 186, "cameras", "dg-xs dg-mu", "middle"))
    W.append(_gauge(10, 198, 46, 30, 0.4) + _t(33, 242, "joint state", "dg-xs dg-mu", "middle"))
    # the policy, its two components
    W.append('<rect x="68" y="52" width="262" height="250" rx="10" class="dg-pol"/>'
             + _t(78, 66, "robot policy", "dg-xs dg-mu"))
    W.append('<rect x="80" y="76" width="238" height="128" rx="8" class="dg-blk"/>')
    W.append(_st('<rect x="81.5" y="77.5" width="235" height="125" rx="7" class="dg-wash"/>',
                 "2=o:1,@1.1,d:0.6 4=o:1,@1.1,d:0.6,re"))
    W.append(_t(92, 96, "VLM backbone / text pathway", "dg-b") + _t(92, 112, "grounds the instruction", "dg-s dg-mu"))
    W.append('<line x1="92" y1="156.5" x2="208" y2="156.5" class="dg-axis"/>')
    W.append(_st(_bars(93, 156, 12, 7, 2.6, 30, 0.3, "dg-sb-o"), "1=o:1,@1.1,d:0.6", static=False))
    W.append(_st(_bars(93, 156, 12, 7, 2.6, 30, 2.1, "dg-sb-c"), "2=o:1,@1.1,d:0.6 3=o:1 4=o:1,@1.1,d:0.6,re"))
    W.append(_t(92, 177, "states", "dg-xs dg-mu"))
    # its weights, and the patch the edit writes into them
    cells = "".join(f'<rect x="{230 + c * 20:g}" y="{124 + r * 13:g}" width="16" height="10" rx="2" class="dg-wc"/>'
                    for r in range(3) for c in range(4))
    patch = "".join(f'<rect x="{230 + c * 20:g}" y="{124 + r * 13:g}" width="16" height="10" rx="2" '
                    f'class="{"dg-wp1" if (r * 3 + c * 5) % 4 else "dg-wp2"}"/>' for r in range(3) for c in range(4))
    W.append(cells + _st(patch, "3=o:1,@1.3,pop 4=o:1"))
    W.append(_st(_t(268, 177, "weights", "dg-xs dg-mu", "middle"), "1=o:1 2=o:1", static=False)
             + _st(_t(268, 177, "edited weights", "dg-xs dg-acc dg-b", "middle"), "3=o:1,@1.5 4=o:1"))
    W.append(_st(_arrow([(211, 141), (225, 141)], "dg-acc-ln") + _t(92, 196, "the difference, written into the weights",
                                                                    "dg-xs dg-acc dg-b"), "3=o:1,@0.3", static=False))
    # the component that produces the actions
    W.append(_arrow([(199, 204), (199, 230)]) + _t(207, 221, "attends to its states", "dg-xs dg-mu"))
    W.append('<rect x="80" y="230" width="238" height="60" rx="8" class="dg-blk"/>'
             + _t(92, 252, "action expert / action head", "dg-b") + _t(92, 268, "generates the action chunk", "dg-s dg-mu"))
    W.append(_arrow([(199, 290), (199, 314)]) + _chunk(199 - _chunk_w(12, 3, 1.6) / 2, 318, 12, 3, 1.6, 14)
             + _t(234, 329, "action chunk", "dg-xs dg-mu") + _arrow([(199, 336), (199, 354)]))
    # the robot
    W.append(_st(_mark(146, 368, "broke", 7) + _t(160, 372.5, "the robot fails", "dg-s dg-b"), "1=o:1,@2.75,pop",
                 static=False, cls="pp"))
    W.append(_st(_mark(140, 368, "fixed", 7) + _t(154, 372.5, "the robot succeeds", "dg-s dg-b"),
                 "2=o:1,@2.75,pop 4=o:1,@2.75,pop", cls="pp"))
    W.append(_st(_t(199, 372.5, "no robot run: computed offline", "dg-s dg-mu", "middle"), "3=o:1,@0.3", static=False))
    # tokens: words into the VLM / text pathway, frames and joint state into the policy, states on, chunks out
    n = len(xo)
    tx = [104 + (294 - 104) * i / (n - 1) for i in range(n)]
    for i, x in enumerate(xo):
        W.append(_tok("word", [(x, 40), (tx[i], 82)], "1 4", 0.35 + 0.05 * i, 0.85, "tk-o"))
    for i, x in enumerate(xc):
        W.append(_tok("word", [(x, 40), (tx[i], 82)], "2", 0.35 + 0.05 * i, 0.85, "tk-c"))
    for i, key in enumerate(cams):
        W.append(_tok("frame", [(42, 85.5 + 36 * i), (80, 85.5 + 36 * i)], "1 2 4", 0.3 + 0.06 * i, 0.75, key=key))
    for i, yy in enumerate((208, 218)):
        W.append(_tok("state", [(52, yy), (80, yy)], "1 2 4", 0.42 + 0.06 * i, 0.7))
    for i, x in enumerate((185, 199, 213)):
        W.append(_tok("dot", [(x, 206), (x, 229)], "1", 1.4 + 0.06 * i, 0.55, "tk-o"))
        W.append(_tok("dot", [(x, 206), (x, 229)], "2 4", 1.4 + 0.06 * i, 0.55, "tk-c"))
    for i in range(2):
        W.append(_tok("chunk", [(199, 325), (199, 356)], "1", 2.0 + 0.4 * i, 0.8, "tk-o"))
        W.append(_tok("chunk", [(199, 325), (199, 356)], "2 4", 2.0 + 0.4 * i, 0.8, "tk-c"))
    W.append(_tok("delta", [(150, 136), (268, 141)], "3", 0.5, 1.0))
    W.append(_tok("ring", [(0, 0)], "3", 1.2, 1.7, label=(225, 119, 86, 46)))
    title = "The edit in four steps"
    desc = (" ".join(f"Step {i}: {s}" for i, s in enumerate(DG1_STEPS, 1))
            + " Camera frames and the joint state enter the policy; the instruction’s words enter its VLM backbone or "
              "text pathway, whose states the action expert or action head attends to; an action chunk goes to the "
              "robot.")
    return _svg(340, 384, title, desc, "".join(W), "dg-schem", "dg1")


def where_sites(P, rc, tables):
    """Per model: the schematic's parts in plain words (the model's own name for each, once, in brackets), and every
    edit tried there: a 2-4 word variant name, a verdict, its success and, where run on the same starts, its random edit."""
    T = {t["id"]: {r["key"]: r["cells"] for r in t["rows"]} for t in tables}
    w = P["where"]
    n, base = w["n"], w["arms"]["base"]
    best = max(w["arms"], key=lambda k: w["arms"][k] if k != "base" else -1)

    def pi(arm, variant, row):
        c, k = w["cmp"].get(arm), w["arms"][arm]
        p = c["mcnemar_p"] if c else None
        vb = w["vs_full"].get(arm) if arm != best else None
        oc = ("none" if p is None or p >= SIG else "broke" if k < base else
              "partial" if vb and vb["mcnemar_p"] < SIG and k < w["arms"][best] else "fixed")
        return dict(variant=variant, row=row, oc=oc, k=k, n=n, a=base, p=p, ctl=None)

    out = [dict(id="pi05", name="π0.5", task=tn("T1"), n=n, base=base, kind="unseen", cams="A",
                blocks=[dict(let="A", detail="vision-language model",
                             sites=[pi("v1full", "whole part", "VLM, all layers"),
                                    pi("v1img816", "image part only", "VLM, image tokens only (0.16% of parameters)")]),
                        dict(let="B", detail="action expert",
                             sites=[pi("a1", "this part alone", "Action expert"),
                                    pi("va", "A and B together", "VLM + action expert")]),
                        dict(let="C", detail="action output projection",
                             sites=[pi("h1", "always on", "Action output projection, unconditional"),
                                    pi("p1", "with an on/off switch", "Action output projection, gated*")])])]
    for mid, name, cams, details in (("cosmos", "Cosmos3 Edge", "B", ("text pathway", "video and action pathway", "")),
                                     ("groot", "GR00T N1.7", "A", ("Cosmos-Reason2 VLM backbone", "flow-matching DiT action head", ""))):
        bars = rc[mid]["where"].get("_bars")
        if not bars:
            continue
        kind = "unseen" if (bars.get("kind") or "").strip() == "unseen" else "practice"
        good = [s for s in bars["sites"] if s["p"] is not None and s["p"] < SIG and s["b"] > s["a"]]
        top = max((s["b"] for s in good), default=None)
        blocks = []
        for let, det, cls in zip("ABC", details, ("backbone", "action", None)):
            sites = []
            for s in [x for x in bars["sites"] if x["cls"] == cls] if cls else []:
                oc = ("none" if s["p"] is None or s["p"] >= SIG else "broke" if s["b"] < s["a"] else
                      "fixed" if s["b"] == top else "partial")
                sites.append(dict(variant="", row={"cosmos": ("Text pathway", "Video/action pathway"), "groot": ("VLM backbone", "Flow-matching DiT action head")}[mid][
                    0 if cls == "backbone" else 1], oc=oc,
                                  k=s["b"], n=s["n"], a=s["a"], p=s["p"], ctl=s.get("ctl")))
            blocks.append(dict(let=let, detail=det, sites=sites))
        out.append(dict(id=mid, name=name, task=tn(bars["task"]), n=bars["n"], base=bars["base"], kind=kind, cams=cams,
                        blocks=blocks))
    return out


BADGE = {"fixed": "fixed it", "partial": "partly fixed it", "none": "no effect", "broke": "broke the task"}
PARTS = {"A": "reads the instruction", "B": "turns it into motion", "C": "final output layer"}


def dg_panel(m, nsteps=3):
    """One architecture in Figure 4: its inputs (real camera frames, the joint state, the instruction) flowing into
    its components, an action chunk flowing out, and one verdict per component the edit went into; the component
    holding the edit in each step is outlined. Components with no edit are not drawn."""
    a = ARCH[m["id"]]
    pid = f"dg2{m['id']}"
    W = [_t(14, 20, a["type"], "dg-h")]
    starts = f"{m['n']} {'unseen' if m['kind'] == 'unseen' else 'practice'} starts"
    lines = [f"{m['name']} · {m['task']}", starts]
    if _tw(" · ".join(lines), 11.5) <= 252:
        lines = [" · ".join(lines)]
    for i, ln in enumerate(lines):
        W.append(_t(14, 37 + 14 * i, ln, "dg-xs dg-mu"))
    y_in = 37 + 14 * (len(lines) - 1) + 14
    blocks = [b for b in m["blocks"] if b["sites"]]
    lets = [b["let"] for b in blocks]
    X0, X1 = 16, 258
    cams = a["cams"]
    ih = 23
    toks = []

    def badge(blk, x, y):
        s = blk["sites"][0]
        k = lets.index(blk["let"]) + 1
        spec = " ".join(f"{j}=o:1" + (",@2.9,pop" if j == k else "") for j in range(k, nsteps + 1))
        return _st(f'<title>{esc(BADGE[s["oc"]])}</title>' + _mark(x, y, s["oc"]) + _t(x + 13, y + 4.5, BADGE[s["oc"]],
                                                                                         "dg-s dg-b"), spec, cls="pp")

    def patch(blk, x, y, w, h, tx=None, ty=None):
        """The component holding the edit in this step: outlined, with an "edit" tag inside it (top right, or at
        tx, ty)."""
        k = lets.index(blk["let"]) + 1
        tag_w = 34
        tx = x + w - tag_w - 8 if tx is None else tx
        ty = y + 8 if ty is None else ty
        return _st(f'<rect x="{x - 2:g}" y="{y - 2:g}" width="{w + 4:g}" height="{h + 4:g}" rx="9" class="dg-patch"/>'
                   f'<rect x="{tx:g}" y="{ty:g}" width="{tag_w}" height="15" rx="7.5" class="dg-tag"/>'
                   + _t(tx + tag_w / 2, ty + 11, "edit", "dg-xs dg-b dg-tag-t", "middle"), f"{k}=o:1", static=False)

    if a["layout"] == "stack":
        # inputs: cameras, joint state, instruction
        xs_ = [X0 + i * 38 for i in range(len(cams))]
        for i, (x, key) in enumerate(zip(xs_, cams)):
            W.append(_thumb(x, y_in, 34, ih, key, f"{pid}c{i}"))
        cam_c = (xs_[0] + xs_[-1] + 34) / 2
        if a["state_to"] == "A":
            gx, ix0, ix1 = xs_[-1] + 40, xs_[-1] + 86, X1
        else:
            gx, ix0, ix1 = X1 - 38, xs_[-1] + 40, X1 - 46
        W.append(_gauge(gx, y_in, 38, ih, 1.1 + len(cams)))
        W.append(f'<rect x="{ix0:g}" y="{y_in:g}" width="{ix1 - ix0:g}" height="{ih}" rx="{ih / 2:g}" class="dg-in"/>'
                 + _t((ix0 + ix1) / 2, y_in + 15.5, "instruction", "dg-xs", "middle"))
        W.append(_t(cam_c, y_in + ih + 13, "cameras", "dg-xs dg-mu", "middle")
                 + _t(gx + 19, y_in + ih + 13, "joint state", "dg-xs dg-mu", "middle"))
        y = y_in + ih + 26
        tops, bh = {}, 66
        for i, blk in enumerate(blocks):
            name, det = a["parts"][blk["let"]]
            tops[blk["let"]] = y
            W.append(f'<rect x="{X0}" y="{y}" width="{X1 - X0}" height="{bh}" rx="8" class="dg-blk"/>'
                     + _t(X0 + 12, y + 20, name, "dg-b") + _t(X0 + 12, y + 36, det, "dg-s dg-mu"))
            W.append(patch(blk, X0, y, X1 - X0, bh) + badge(blk, X0 + 18, y + 51))
            y += bh + (30 if i < len(blocks) - 1 else 0)
        ya = tops["A"]
        cx = (X0 + X1) / 2
        ic = (ix0 + ix1) / 2
        W.append(_arrow([(ic, y_in + ih), (ic, ya)]) + _arrow([(cam_c, y_in + ih + 17), (cam_c, ya)]))
        for i, f in enumerate((0.22, 0.42, 0.62, 0.82)):
            x = ix0 + (ix1 - ix0) * f
            toks.append(_tok("word", [(x, y_in + ih + 3), (ic - 24 + 16 * i, ya + 6)], "1 2 3", 0.35 + 0.05 * i, 0.8,
                             "tk-c"))
        for i, (x, key) in enumerate(zip(xs_, cams)):
            toks.append(_tok("frame", [(cam_c - 12 + 24 * i, y_in + ih + 19), (cam_c - 12 + 24 * i, ya + 6)], "1 2 3",
                             0.3 + 0.06 * i, 0.7, key=key))
        if a["state_to"] == "A":
            W.append(_arrow([(gx + 19, y_in + ih + 17), (gx + 19, ya)]))
            for i in range(2):
                toks.append(_tok("state", [(gx + 15 + 8 * i, y_in + ih + 18), (gx + 15 + 8 * i, ya + 6)], "1 2 3",
                                 0.42 + 0.06 * i, 0.75))
        else:
            yb = tops["B"]
            lx = X1 + 12
            W.append(_arrow([(gx + 19, y_in + ih + 17), (gx + 19, y_in + ih + 21), (lx, y_in + ih + 21), (lx, yb + 33),
                             (X1, yb + 33)]))
            for i in range(2):
                toks.append(_tok("state", [(gx + 19, y_in + ih + 18), (gx + 19, y_in + ih + 21), (lx, y_in + ih + 21),
                                           (lx, yb + 33), (X1 - 6, yb + 33)], "1 2 3", 0.4 + 0.12 * i, 1.3))
        for p, q in zip(lets, lets[1:]):
            y0, y1 = tops[p] + bh, tops[q]
            W.append(_arrow([(cx, y0), (cx, y1)]) + (_t(cx + 8, y0 + 19, a["links"][p], "dg-xs dg-mu")
                                                    if p in a["links"] else ""))
            t0 = 1.3 + 0.45 * lets.index(p)
            for i, dx in enumerate((-10, 0, 10)):
                toks.append(_tok("dot", [(cx + dx, y0 + 2), (cx + dx, y1 + 4)], "1 2 3", t0 + 0.06 * i, 0.5, "tk-c"))
        ye = y
        n, run = a["chunk"]
        cw = 64 / n * 0.62
        gap = (64 - n * cw) / (n - 1)
        W.append(_arrow([(cx, ye), (cx, ye + 20)]) + _chunk(cx - 32, ye + 24, n, round(cw, 2), round(gap, 2), 14,
                                                            run if run < n else None)
                 + _t(cx, ye + 54, a["out"], "dg-xs", "middle") + _t(cx, ye + 69, "to the robot", "dg-xs dg-mu", "middle"))
        for i in range(2):
            toks.append(_tok("chunk", [(cx, ye + 31), (cx, ye + 44)], "1 2 3", 2.1 + 0.35 * i, 0.7, "tk-c"))
        H = ye + 78
    else:
        # the world-action model: two pathways side by side; the video/action pathway reads the text pathway's keys
        # and values in every layer
        AX0, AX1, BX0, BX1 = X0, 112, 150, X1
        W.append(f'<rect x="{AX0}" y="{y_in}" width="{AX1 - AX0}" height="{ih}" rx="{ih / 2:g}" class="dg-in"/>'
                 + _t((AX0 + AX1) / 2, y_in + 15.5, "instruction", "dg-xs", "middle"))
        xs_ = [BX0 - 14 + i * 29 for i in range(len(cams))]
        for i, (x, key) in enumerate(zip(xs_, cams)):
            W.append(_thumb(x, y_in, 26, ih - 4, key, f"{pid}c{i}"))
        gx = BX1 - 34
        W.append(_gauge(gx, y_in, 34, ih - 4, 2.2))
        cam_c = (xs_[0] + xs_[-1] + 26) / 2
        W.append(_t(cam_c, y_in + ih + 9, "cameras", "dg-xs dg-mu", "middle")
                 + _t(gx + 17, y_in + ih + 9, "joint state", "dg-xs dg-mu", "middle"))
        ya = y_in + ih + 26
        bh = 132
        ba, bb = blocks[0], blocks[1] if len(blocks) > 1 else None
        W.append(f'<rect x="{AX0}" y="{ya}" width="{AX1 - AX0}" height="{bh}" rx="8" class="dg-blk"/>'
                 + _t(AX0 + 10, ya + 20, a["parts"]["A"][0], "dg-b") + _t(AX0 + 10, ya + 36, "instruction", "dg-s dg-mu")
                 + _t(AX0 + 10, ya + 51, "tokens", "dg-s dg-mu"))
        W.append(patch(ba, AX0, ya, AX1 - AX0, bh, AX0 + 10, ya + 72) + badge(ba, AX0 + 16, ya + bh - 15))
        W.append(f'<rect x="{BX0}" y="{ya}" width="{BX1 - BX0}" height="{bh}" rx="8" class="dg-blk"/>'
                 + _t(BX0 + 10, ya + 20, "video/action", "dg-b") + _t(BX0 + 10, ya + 35, "pathway", "dg-b")
                 + _t(BX0 + 10, ya + 51, "video and", "dg-s dg-mu") + _t(BX0 + 10, ya + 66, "action tokens", "dg-s dg-mu"))
        if bb:
            W.append(patch(bb, BX0, ya, BX1 - BX0, bh, BX0 + 10, ya + 80) + badge(bb, BX0 + 16, ya + bh - 15))
        ic = (AX0 + AX1) / 2
        W.append(_arrow([(ic, y_in + ih), (ic, ya)]) + _arrow([(cam_c, y_in + ih + 13), (cam_c, ya)])
                 + _arrow([(gx + 17, y_in + ih + 13), (gx + 17, ya)]))
        for i, f in enumerate((0.2, 0.4, 0.6, 0.8)):
            x = AX0 + (AX1 - AX0) * f
            toks.append(_tok("word", [(x, y_in + ih + 3), (ic - 21 + 14 * i, ya + 6)], "1 2 3", 0.35 + 0.05 * i, 0.8,
                             "tk-c"))
        for i, (x, key) in enumerate(zip(xs_, cams)):
            toks.append(_tok("frame", [(cam_c - 20 + 20 * i, y_in + ih + 15), (cam_c - 20 + 20 * i, ya + 6)], "1 2 3",
                             0.3 + 0.06 * i, 0.7, key=key))
        for i in range(2):
            toks.append(_tok("state", [(gx + 13 + 8 * i, y_in + ih + 14), (gx + 13 + 8 * i, ya + 6)], "1 2 3",
                             0.42 + 0.06 * i, 0.7))
        for i, yy in enumerate((ya + 62, ya + 82, ya + 102)):
            W.append(_arrow([(AX1, yy), (BX0, yy)]))
            toks.append(_tok("dot", [(AX1 + 2, yy), (BX0 - 4, yy)], "1 2 3", 1.2 + 0.08 * i, 0.5, "tk-c"))
        W.append(_t(AX0, ya + bh + 16, a["links"]["A"], "dg-xs dg-mu"))
        ye = ya + bh
        ox = (BX0 + BX1) / 2
        n, _ = a["chunk"]
        W.append(_arrow([(ox, ye), (ox, ye + 26)]))
        vid = "".join(f'<rect x="{ox - 52 + 4 * i:g}" y="{ye + 30 + 3 * i:g}" width="22" height="15" rx="2" '
                      f'class="dg-vf"/>' for i in range(3))
        cw = 1.25
        gap = (60 - n * cw) / (n - 1)
        W.append(vid + _chunk(ox - 12, ye + 33, n, cw, round(gap, 2), 14))
        W.append(_t(X1, ye + 66, a["out"], "dg-xs", "end") + _t(X1, ye + 81, "to the robot", "dg-xs dg-mu", "end"))
        for i in range(2):
            toks.append(_tok("chunk", [(ox + 18, ye + 36), (ox + 18, ye + 50)], "1 2 3", 1.9 + 0.35 * i, 0.7, "tk-c"))
            toks.append(_tok("video", [(ox - 38, ye + 38), (ox - 38, ye + 50)], "1 2 3", 1.9 + 0.35 * i, 0.7))
        H = ye + 90
    W += toks
    parts = "; ".join(f"{a['parts'][b['let']][0]}: {BADGE[b['sites'][0]['oc']]}" for b in blocks)
    return _svg(280, H, f"{a['type']} ({m['name']}), {m['task']}: where the edit went",
                f"{m['n']} {'unseen' if m['kind'] == 'unseen' else 'practice'} starts. Inputs: "
                + ("three camera views and the joint state into the video/action pathway, the instruction into the text "
                   "pathway, whose keys and values the video/action pathway reads in every layer" if a["layout"] == "side"
                   else f"{len(cams)} camera views and the instruction into the VLM backbone, the joint state into the "
                        + ("VLM backbone" if a["state_to"] == "A" else a["parts"]["B"][0]))
                + f"; output: {a['out']}. One verdict per component: {parts}.", "".join(W), "dg-panel", pid)


def dg_legend():
    items = "".join(f'<li><svg class="dg dg-key" viewBox="-8 -8 16 16" aria-hidden="true">{_mark(0, 0, k)}</svg>{esc(t)}</li>'
                    for k, t in OUTCOMES)
    return f'<ul class="dg-legend">{items}</ul>'



def dg_groot(tables, cross):
    """Diagram 3, the reasoning-backbone VLA's two wordings word by word: the words before and after the clear
    wording's inserted clause line up one to one, the inserted words have nothing to land on. cosmos.js plays it in
    three steps (GROOT_STEPS); the still drawing is the last."""
    T = {t["id"]: {r["key"]: r["cells"] for r in t["rows"]} for t in tables}
    rows = (cross.get("groot") or {}).get("rows") or []
    wp = next((r.get("wording_pair") for r in rows if r["_task"] == "T1" and isinstance(r.get("wording_pair"), list)
               and len(r["wording_pair"]) == 2 and all(r["wording_pair"])), None)
    wd, eg = T["wording"].get("T1", {}).get("groot") or {}, T["edit_gain"].get("T1", {}).get("groot") or {}
    vs = git("show", "origin/cross-groot/v1:study/results/cross/groot/results/wsp/t1/early/value_similarity.json")
    cos = None
    if vs:
        v = sorted(json.loads(vs).get("early_cos_per_layer") or [])
        cos = (v[len(v) // 2] + v[(len(v) - 1) // 2]) / 2 if v else None
    if not wp or wd.get("s") != "ok" or eg.get("s") != "ok" or cos is None:
        return ""
    o, c = wp[0].split(), wp[1].split()
    i = 0
    while i < min(len(o), len(c)) and o[i].lower() == c[i].lower():
        i += 1
    j = 0
    while j < min(len(o), len(c)) - i and o[-1 - j].lower() == c[-1 - j].lower():
        j += 1
    seg_c = [" ".join(c[:i]), " ".join(c[i:len(c) - j]), " ".join(c[len(c) - j:])]
    mid_o = o[i:len(o) - j]
    Y0, P, CH = 70, 17, 14
    ry = lambda r: Y0 + P * r
    LX, RX = 14, 216
    W = [_t(14, 18, "The two wordings, word by word", "dg-b"), _t(14, 36, f"GR00T N1.7 · {tn('T1')}", "dg-s dg-mu"),
         _t(LX, 60, "original wording", "dg-s dg-mu"), _t(RX, 60, "clear wording", "dg-s dg-acc")]
    chip = lambda x, y, s, cls="dg-tok": (f'<rect x="{x:g}" y="{y:g}" width="{_tw(s, 11.5) + 12:.1f}" height="{CH}" '
                                          f'rx="4" class="{cls}"/>' + _t(x + 6, y + 10.5, s, "dg-xs"))
    # the clear wording's inserted clause: a band behind it, and room it cannot land in on the original side
    ins = range(i, len(c) - j)
    bw = max(_tw(c[r], 11.5) for r in ins) + 22
    W.append(_st(f'<rect x="{RX - 5}" y="{ry(i) - 3}" width="{bw:.1f}" height="{P * len(ins) + 2}" rx="6" '
                 f'class="dg-band"/>', "3=o:1,@0.2"))
    gap_rows = range(i + len(mid_o), len(c) - j)
    W.append(_st("".join(f'<rect x="{LX}" y="{ry(r)}" width="44" height="{CH}" rx="4" class="dg-ghost"/>' for r in gap_rows),
                 "3=o:1,@1.4"))
    my = (ry(i) + ry(len(c) - j - 1) + CH) / 2
    W.append(_st(_t(140, my - 3, "no counterpart:", "dg-xs dg-mu dg-b", "middle")
                 + _t(140, my + 11, "nowhere to land", "dg-xs dg-mu", "middle"), "3=o:1,@1.7"))
    # matched words: connectors, drawn from the left
    matched = list(range(i)) + list(range(len(c) - j, len(c)))
    for q, r in enumerate(matched):
        ow = o[r] if r < i else o[len(o) - (len(c) - r)]
        x1 = LX + _tw(ow, 11.5) + 15
        W.append(_st(f'<line x1="{x1:.1f}" y1="{ry(r) + 7}" x2="{RX - 3}" y2="{ry(r) + 7}" class="dg-acc-ln"/>',
                     f"1=o:0,sx:0 2=o:1,sx:1,@{1.0 + 0.05 * q:.2f},d:0.5 3=o:1", cls="sxo"))
    # the original's words, in order (step 1: they come in one by one): the prefix in place, its middle unmatched, its
    # suffix sliding down into line
    come = lambda r, extra="": f"1=o:1{extra},@{0.2 + 0.055 * r:.2f},d:0.4,re"
    for r in range(i):
        W.append(_st(chip(LX, ry(r), o[r]), come(r) + " 2=o:1 3=o:1"))
    for q, w_ in enumerate(mid_o):
        W.append(_st(chip(LX, ry(i + q), w_), come(i + q), static=False)
                 + _st(chip(LX, ry(i + q), w_, "dg-tok dg-dash-b"), "2=o:1,@0.9 3=o:1"))
    shift = -(len(c) - j - (i + len(mid_o))) * P
    for q in range(j):
        r = len(c) - j + q
        W.append(_st(chip(LX, ry(r), o[len(o) - j + q]),
                     come(i + len(mid_o) + q, f",y:{shift}") + f" 2=y:0,@{0.3 + 0.05 * q:.2f},d:0.75 3=y:0"))
    # the clear wording's words; the inserted ones reach for a counterpart and come back
    for r, w_ in enumerate(c):
        st = come(r + 0.5) + " 2=o:1 3=o:1"
        if r in ins:
            q = r - i
            W.append(f'<g class="nd" data-n="3@{0.55 + 0.08 * q:.2f}">'
                     + _st(chip(RX, ry(r), w_, "dg-tok dg-acc-b"), st) + "</g>")
        else:
            W.append(_st(chip(RX, ry(r), w_), st))
    y = ry(len(c)) + 14
    W.append(f'<line x1="{LX}" y1="{y - 4}" x2="{LX + 22}" y2="{y - 4}" class="dg-acc-ln"/>'
             + _t(LX + 30, y, "matched: the edit copies the change at these words", "dg-xs dg-mu"))
    W.append(f'<rect x="{LX - 2}" y="{y + 6}" width="26" height="15" rx="4" class="dg-band"/>'
             f'<rect x="{LX + 1}" y="{y + 8}" width="20" height="11" rx="3" class="dg-tok dg-acc-b"/>'
             + _t(LX + 30, y + 17, "inserted: no counterpart in the original", "dg-xs dg-mu"))
    y += 46
    W.append(_t(14, y + 4, "The clear wording", "dg-b dg-acc"))
    W.append(_t(14, y + 23, "The action head attends to the inserted words,", "dg-s"))
    W.append(_t(14, y + 40, "so the robot’s first reach stays as it was.", "dg-s"))
    W.append(_t(14, y + 61, f"{wd['b']} of {wd['n']} practice starts completed", "dg-s dg-b"))
    W.append(_t(14, y + 78, f"with the original wording: {wd['a']}"
                + (" (could be chance)" if wd.get("pv") is not None and wd["pv"] >= SIG else ""), "dg-s dg-mu"))
    y += 104
    W.append(_t(14, y + 4, "The edit", "dg-b"))
    W.append(_t(14, y + 23, "It can copy only the change at the matched", "dg-s"))
    W.append(_t(14, y + 40, "words. That change is almost the same at the", "dg-s"))
    W.append(_t(14, y + 57, "first grasp as at the banana, so the edit", "dg-s"))
    W.append(_t(14, y + 74, "moves the first reach.", "dg-s"))
    W.append(_mark(20, y + 91.5, "broke" if eg["b"] < eg["a"] else "none")
             + _t(34, y + 96, f"{eg['b']} of {eg['n']} practice starts completed", "dg-s dg-b"))
    W.append(_t(34, y + 113, f"without the edit: {eg['a']}"
                + (" (clearly worse)" if eg.get("pv") is not None and eg["pv"] < SIG and eg["b"] < eg["a"] else ""),
                "dg-s dg-mu"))
    H = y + 125
    desc = (f"Original wording: “{wp[0]}”. Clear wording: “{wp[1]}”. The words before and after the inserted words are "
            f"matched one to one; “{seg_c[1]}” has no counterpart in the original. Under the clear wording the action "
            f"head attends to those inserted words, which keep the first reach unchanged: {wd['a']} → {wd['b']} of "
            f"{wd['n']}. The edit can copy only the change at the matched words, which is the same at the first grasp as "
            f"at the banana (median similarity {cos:.2f}), so it acts from the first reach: {eg['a']} → {eg['b']}.")
    return _svg(340, H, "Why the edit backfires on the reasoning-backbone VLA", desc, "".join(W), "dg-groot", "dg3"), dict(
        original=wp[0], clear=wp[1], clause=seg_c[1], cos=round(cos, 2), teacher=[wd["a"], wd["b"], wd["n"]],
        edit=[eg["a"], eg["b"], eg["n"]])


def dg_schematic_html():
    return ('<figure class="dgfig dg1 anim" data-dur="' + " ".join(f"{d:g}" for d in DG1_DUR) + '">' + dg_schematic()
            + anim_controls(DG1_STEPS, "The edit in four steps")
            + '<figcaption>The edit copies the clear wording’s effect into the weights of the VLM backbone or text '
              'pathway, where the instruction is grounded. Copied into the action expert or action head instead, it '
              'lands on states that also depend on where the robot is: a symptom, not the cause. Like activation '
              'steering ⟦turner2023steering⟧, such an edit then needs an on/off gate or leaks into other moments.'
              '</figcaption></figure>')


def dg_where_html(sites):
    """Figure 4: the three architectures, one verdict per component the edit went into, inputs and outputs moving; the
    variants tried on the action-expert VLA in one or two sentences below it."""
    fixed = [ARCH[m["id"]]["short"] for m in sites if any(s["oc"] == "fixed" for s in m["blocks"][0]["sites"][:1])
             and not any(b["sites"] and b["sites"][0]["oc"] == "fixed" for b in m["blocks"][1:])]
    broke = [ARCH[m["id"]]["short"] for m in sites if all(b["sites"][0]["oc"] == "broke" for b in m["blocks"] if b["sites"])
             and any(b["sites"] for b in m["blocks"])]
    first = or_list(sorted({ARCH[m["id"]]["parts"]["A"][0] for m in sites}, key=lambda s: ("VLM" not in s, s)))
    cap = ((f"In {and_list('the ' + x for x in fixed)}, only the edit in the {first} fixed the task; " if fixed else "")
           + (f"in {and_list('the ' + x for x in broke)}, every placement broke it." if broke else ""))
    cap = cap[0].upper() + cap[1:] if cap else "Where the edit went."
    cap = cap.rstrip("; ") + ("" if cap.endswith(".") else ".")
    cap += (" One verdict per component: clearly better or worse than the unedited model from the same starts (exact "
            "McNemar p < 0.05), or no clear change. The scores are in the next figure.")
    steps = where_steps(sites)
    # the variants tried on the action-expert VLA, in words below the figure
    pi = next((m for m in sites if m["id"] == "pi05"), None)
    var = []
    if pi:
        for b in pi["blocks"]:
            for s in b["sites"][1:]:
                var.append(s)
    note = ""
    if var:
        what = {"VLM, image tokens only (0.16% of parameters)": "the VLM edit restricted to the image tokens (storable in "
                                                                "0.16% of the parameters)",
                "VLM + action expert": "the VLM backbone and the action expert edited together",
                "Action output projection, gated*": "a gated edit of the action output projection written by a "
                                                    "blind AI editor"}
        bits = [what.get(s["row"], s["row"][0].lower() + s["row"][1:]) for s in var]
        ocs = {s["oc"] for s in var}
        n = pi["n"]
        if len(ocs) == 1:
            res = (f"Each {BADGE[var[0]['oc']].replace('fixed it', 'fixed the task')} "
                   f"({and_list(str(s['k']) for s in var)} of {n})")
        else:
            res = "; ".join(f"{w[0].upper() + w[1:]}: {BADGE[s['oc']]}, {s['k']} of {n}" for w, s in zip(bits, var))
        cnt = {1: "one variant", 2: "two variants", 3: "three variants", 4: "four variants"}.get(len(var), f"{len(var)} variants")
        note = (f'<p class="dg-after">On the {ARCH["pi05"]["short"]} we also tried {cnt} on the same {n} starts: '
                f'{"; ".join(bits[:-1]) + "; and " + bits[-1] if len(bits) > 2 else and_list(bits)}. {res}; the next '
                f'figure scores them, and the records have the details.</p>')
    return (f'<figure class="dgfig dg2 wide anim" data-dur="{" ".join(f"{d:g}" for d in WHERE_DUR[:len(steps)])}">'
            + '<div class="dg2-grid">' + "".join(f'<div class="dg2-p">{dg_panel(m, len(steps))}</div>' for m in sites)
            + "</div>" + anim_controls(steps, "Where the edit went")
            + f'<figcaption>{esc(cap)}</figcaption></figure>' + note)


GROOT_STEPS = ["The two wordings, word by word: the clear one inserts a clause.",
               "The words before and after the clause match one to one, so they line up.",
               "The inserted words have no counterpart in the original: the edit has nowhere to copy their effect."]
GROOT_DUR = (4.2, 5.0, 5.8)


def dg_groot_html(tables, cross):
    got = dg_groot(tables, cross)
    if not got:
        return "", None
    svg, data = got
    return (f'<figure class="dgfig dg3 anim" data-dur="{" ".join(f"{d:g}" for d in GROOT_DUR)}">{svg}'
            + anim_controls(GROOT_STEPS, "The two wordings, word by word")
            + '<figcaption>The words of the two wordings line up only where the wordings match; the clear wording’s '
              'inserted clause has no counterpart in the original, so the edit has nowhere to copy its effect. The '
              'change at the matched words already acts on the first reach, so the edit breaks the task.'
              '</figcaption></figure>'), data


THUMBS = {   # camera views for the block diagrams: (clip in video/, crop of one view in its frame at THUMB_T seconds)
    "cosmos_l": ("cosmos_t8_wording", (45, 208, 270, 358)),
    "cosmos_r": ("cosmos_t8_wording", (352, 208, 577, 358)),
    "cosmos_w": ("cosmos_t8_wording", (690, 208, 915, 358)),
    "pi05_ext": ("pi05_t1_edit_control_teacher", (110, 592, 425, 802)),
    "pi05_wrist": ("pi05_t1_edit_control_teacher", (600, 592, 915, 802)),
    "groot_ext": ("groot_t1_wording", (110, 318, 425, 528)),
    "groot_wrist": ("groot_t1_wording", (600, 318, 915, 528)),
}
THUMB_T, THUMB_PX = 0.4, (132, 88)


# ---------------------------------------------------------------- the story: TL;DR and the main flow's numbered prose
# The page reads top to bottom as nine short sections; everything else sits in one collapsed "All numbers and records"
# section. These functions write the prose that carries numbers, so a refresh updates it with the tables.
def _cells(tables):
    return {t["id"]: {r["key"]: r["cells"] for r in t["rows"]} for t in tables}


def _ok(c, f=("a", "b", "n")):
    return bool(c) and c.get("s") == "ok" and all(isinstance(c.get(x), int) for x in f)


def main_fix(tables, mid="cosmos"):
    """The model's edit on unseen starts with a significant gain, and its random control: (task, cell, control) or None."""
    T = _cells(tables)
    order = [t[0] for t in TASKS]
    if mid == PREVENT["model"]:     # the page's headline task first: a later edit for another task never takes its place
        order = [PREVENT["target"]] + [t for t in order if t != PREVENT["target"]]
    for tid in order:
        c = T["edit_gain"].get(tid, {}).get(mid)
        if _ok(c) and not c.get("prac") and c.get("pv") is not None and c["pv"] < SIG and c["b"] > c["a"]:
            return tid, c, T["random_control"].get(tid, {}).get(mid)
    return None


def left_drops(tables, mid="cosmos"):
    """Off-target tasks the main edit lowered significantly, unseen starts first."""
    T = _cells(tables)
    out = []
    for tid in [t[0] for t in TASKS]:
        c = T["side_effects"].get(tid, {}).get(mid)
        if _ok(c, ("a", "b", "n", "delta")) and not c.get("target") and c.get("p") is not None and c["p"] < SIG and c["delta"] < 0:
            out.append((tid, c))
    return out


def _says_left(tids):
    return bool(tids) and all("left" in TASK[t][1].lower() for t in tids)


def tldr_story(tables, models, prevent, sites):
    """Three short bullets, each 25 words or fewer and at most one number: the fix, when it works, the cost."""
    T = _cells(tables)
    items = []
    fx = main_fix(tables)
    if fx:
        tid, c, rc0 = fx
        rec = T["recovery"].get(tid, {}).get("cosmos") or {}
        same = rec.get("s") == "ok" and rec.get("n") == c["n"] and rec.get("t") == c["b"]
        items.append(dict(id="fix", lead="The fix.",
                          text=f"A closed-form weight edit makes a world-action model (Cosmos3 Edge) succeed from {c['b']} "
                               f"of {c['n']} new starts" + (", as often as rewording the instruction does." if same else ".")))
    else:
        items.append(dict(id="fix", lead="The fix.", text=f"The world-action model’s edit on unseen starts: {RUNNING}."))
    pred = WHEN_LAST.get("pred") or []
    held = any(x["equal"] and x["held"] is True for x in pred) and all(x["held"] is not False for x in pred)
    items.append(dict(id="where", lead="When it works.",
                      text="Only in the VLM or text pathway; except in the action-expert VLA, only when the clear "
                           "wording swaps words in place" + (", as predicted." if held else ".")))
    drops = left_drops(tables)
    if drops:
        what = ("the two other tasks that say “left”" if len(drops) == 2 and _says_left([t for t, _ in drops])
                else "other tasks")
        tried = [r for r in prevent["rows"] if r["name"] != prevent["base"]
                 and any(r["cells"][t]["s"] == "ok" for t in prevent["left"])]
        items.append(dict(id="cost", lead="The cost.", text=f"It breaks {what}"
                          + (", and rebuilding it to protect them didn’t help." if tried else ".")))
    for it in items:
        it.update(kind=None, tag=None)
    return dict(items=items, note="")


def fix_text(tables, cross):
    """The payoff on Cosmos3 Edge: the edit against the unedited robot, its random edit and the clearer wording."""
    T = _cells(tables)
    fx = main_fix(tables)
    if not fx:
        return f'<p>The world-action model’s edit on unseen starts is <span class="run">running</span>.</p>'
    tid, c, rc0 = fx
    r = rc0.get("r") if _ok(rc0, ("r", "n")) and rc0["n"] == c["n"] else None
    s1 = (f"On {c['n']} unseen starts, the edited world-action model succeeds {c['b']} times. Unedited it succeeds "
          + (f"{c['a']} times" if c["a"] else "none of the time")
          + (", and so does the robot with a random edit of the same size." if r is not None and r == c["a"] else
             f"; with a random edit of the same size, {r} times." if r is not None else "."))
    rec = T["recovery"].get(tid, {}).get("cosmos") or {}
    s2 = ""
    if rec.get("s") == "ok" and rec.get("n") == c["n"] and isinstance(rec.get("t"), int):
        s2 = (f"Given the clearer wording instead, the unedited robot succeeds {rec['t']} times. "
              + ("The edit gets all the way there, and the robot still receives the original words."
                 if rec["t"] == c["b"] else "The edit does at least as well, and the robot still receives the original words."
                 if c["b"] > rec["t"] else f"The edit gets {rec['v']} of the way there, with the original words."))
    return f"<p>{esc(s1)}</p>" + (f"<p>{esc(s2)}</p>" if s2 else "")


def dg_fix_chart(tables):
    """Figure 3, the fix as three bars from the same unseen starts: the robot without any edit, with a random edit of
    the same size, with the edit. The title says what the bars count; each value is a count of starts and its share."""
    fx = main_fix(tables)
    if not fx:
        return ""
    tid, c, rc0 = fx
    bars = [("without any edit", c["a"], "dg-bar-base")]
    if _ok(rc0, ("r", "n")) and rc0["n"] == c["n"]:
        bars.append(("with a random edit of the same size", rc0["r"], "dg-bar-base"))
    bars.append(("with the edit", c["b"], "dg-bar-acc"))
    rec = _cells(tables)["recovery"].get(tid, {}).get("cosmos")
    if rec and rec.get("s") == "ok" and not rec.get("prac") and rec.get("n") == c["n"] and isinstance(rec.get("t"), int):
        bars.append(("without any edit, given the clear wording", rec["t"], "dg-bar-ref"))
    VW, x0, L, W = 350, 150, 110, []
    ttl = f"{tn(tid)}: starts the robot completed, of {c['n']} unseen starts"
    y = 6
    top = y - 6
    for lab, k, cls in bars:
        w = L * k / c["n"]
        lab_l = wrap_px(lab, x0 - 14, 12.5)
        h = max(18, 15 * len(lab_l) + 3)
        val = f"{k} of {c['n']}" + (f" ({100 * k / c['n']:.0f}%)" if k else "")
        W.append(f'<g><title>Robot {esc(lab)}: {k} of {c["n"]} starts completed ({100 * k / c["n"]:.0f}%)</title>'
                 + "".join(_t(x0 - 10, y + 13 + 15 * q - 7.5 * (len(lab_l) - 1), ln, "dg-s", "end")
                           for q, ln in enumerate(lab_l))
                 + f'<rect x="{x0}" y="{y + h / 2 - 9:g}" width="{w:.1f}" height="18" rx="3" class="{cls}"/>'
                 + _t(x0 + w + 7, y + h / 2 + 4, val, "dg-s dg-b") + "</g>")
        y += h + 12
    W.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y - 6}" class="dg-axis"/>')
    desc = "; ".join(f"robot {lab}: {k} of {c['n']} starts completed" for lab, k, _ in bars)
    svg = _svg(VW, y, ttl, desc + ".", "".join(W), "dg-fixchart", "dg-fix")
    who = {"without any edit": "without any edit", "with a random edit of the same size":
           "with a random edit of the same size (a placebo)",
           "without any edit, given the clear wording": "without any edit but given the clear wording"}
    cap = (f"Each bar counts the starts the robot completed, from the same {c['n']} unseen starts: {c['b']} with the edit, "
           + and_list(f"{k} {who.get(lab, lab)}" for lab, k, _ in bars if lab != "with the edit") + ". The robot "
           "always received the original wording, except in the last bar.") if bars[-1][2] == "dg-bar-ref" else (
           f"Each bar counts the starts the robot completed, from the same {c['n']} unseen starts: {c['b']} with the edit, "
           + and_list(f"{k} {who.get(lab, lab)}" for lab, k, _ in bars[:-1]) + ".")
    return f'<figure class="dgfig dg-fix"><p class="dg-ttl">{esc(ttl)}</p>{svg}<figcaption>{esc(cap)}</figcaption></figure>'


def dg_side_chart(tables, mid="cosmos"):
    """Figure 6, the main edit, unchanged, on every other task: one bar per task, the change in the share of starts
    completed (edited − unedited) in percentage points, with the counts behind it. Dark and bold: clearly worse; teal
    and bold: clearly better; grey: no clear change."""
    T = _cells(tables)
    rows, target = [], None
    for tid in [t[0] for t in TASKS]:
        c = T["side_effects"].get(tid, {}).get(mid)
        if _ok(c, ("a", "b", "n", "delta")):
            if c.get("target"):
                target = tid
            else:
                rows.append((tid, c))
    if not rows:
        return ""
    VW, x0, half, W = 360, 180, 160, []
    W.append(_t(VW, 12, "starts completed:", "dg-xs dg-mu", "end") + _t(VW, 26, "without → with the edit", "dg-xs dg-mu", "end"))
    y, top = 50, 38
    for tid, c in rows:
        sig = c.get("p") is not None and c["p"] < SIG
        d = 100 * c["delta"] / c["n"]
        cls = "dg-bar-acc" if sig and d > 0 else "dg-bar-loss" if sig else "dg-bar-base"
        w = half * abs(d) / 100
        val = f"{c['a']} → {c['b']} of {c['n']}" + (" practice" if c.get("prac") else "")
        tip = (f"{tn(tid)}: {c['a']} of {c['n']} {'practice ' if c.get('prac') else 'unseen '}starts completed without the "
               f"edit, {c['b']} with it; change {sgn(d, True, 0)} percentage points"
               + ("; clearly worse" if sig and d < 0 else "; clearly better" if sig else "; no clear change"))
        W.append(f'<g><title>{esc(tip)}</title>' + _t(0, y + 4, tn(tid), "dg-s dg-b" if sig else "dg-s")
                 + _t(VW, y + 4, val, "dg-s dg-b" if sig else "dg-s dg-mu", "end")
                 + (f'<rect x="{x0 - w if d < 0 else x0:.1f}" y="{y + 10}" width="{max(w, 0):.1f}" height="12" rx="2" '
                    f'class="{cls}"/>' if w else "") + "</g>")
        y += 40
    grid = "".join(f'<line x1="{x0 + half * v / 100:g}" y1="{top}" x2="{x0 + half * v / 100:g}" y2="{y - 4}" class="dg-grid"/>'
                   for v in (-100, -50, 50, 100))
    W.insert(1, grid)
    W.append(f'<line x1="{x0}" y1="{top}" x2="{x0}" y2="{y - 4}" class="dg-axis"/>'
             + "".join(_t(x0 + half * v / 100, y + 10, ("+" if v > 0 else "−" if v < 0 else "") + str(abs(v)),
                          "dg-xs dg-mu", "middle") for v in (-100, -50, 0, 50, 100)))
    y += 30
    for ln in wrap_px("Change in the share of starts completed (edited − unedited), in percentage points", VW - 20,
                      11.5, True):
        W.append(_t(VW / 2, y, ln, "dg-xs dg-b", "middle"))
        y += 15
    desc = "; ".join(f"{tn(tid)}: {c['a']} → {c['b']} of {c['n']} starts completed" + (" (clear)" if c.get("p") is not None
                                                                                         and c["p"] < SIG else "")
                     for tid, c in rows)
    ttl = (f"The {tn(target)} edit, unchanged, on every other task" if target else
           "The edit, unchanged, on every other task")
    svg = _svg(VW, y - 6, ttl, desc + ".", "".join(W), "dg-side", "dg-side")
    drops = [tn(t) for t, c in rows if c.get("p") is not None and c["p"] < SIG and c["delta"] < 0]
    cap = ("Each bar is how much the share of starts the robot completed changed when the edit ran, unchanged, on a task "
           "it was not built for: with the edit minus without, from the same starts ("
           + ("practice starts where marked, else unseen" if any(c.get("prac") for _, c in rows) else "unseen starts")
           + "). Dark bar and bold label: clearly worse; grey: no clear change. "
           + (f"Only {and_list(drops)} changed." if drops else "No task changed clearly."))
    return (f'<figure class="dgfig dg-sidefig"><p class="dg-ttl">{esc(ttl)}</p>{svg}<figcaption>{esc(cap)}</figcaption>'
            f'</figure>')


def left_text(tables):
    """The cost: the two "left" tasks, the likely reason, and π0.5's edit for contrast; then the bridge."""
    drops = left_drops(tables)
    if not drops:
        return f'<p>Side effects on the other tasks are <span class="run">running</span>.</p>'
    tids = [t for t, _ in drops]
    T = _cells(tables)
    first = drops[0]
    s1 = ("Run unchanged on every other task, the edit breaks "
          + ("the two other tasks whose instructions say “left”." if len(drops) == 2 and _says_left(tids)
             else and_list(tn(t) for t in tids) + ".")
          + f" {tn(first[0])} drops from {first[1]['a']} to {first[1]['b']} of {first[1]['n']}"
          + (" practice" if first[1].get("prac") else " unseen") + " starts"
          + ("; on Mustard in left bin the robot carries the mustard to the other bin." if "T5" in tids else ".")
          + " In model-editing terms this is a locality failure ⟦wang2024kesurvey⟧.")
    s2 = ("The edit was built from one sentence pair that turns “left of” into “right of”. It seems to have taught the "
          "text pathway something like “left means right”, well beyond that sentence: a ripple effect "
          "⟦cohen2024ripple⟧. Directional words are also where the grounding of π0.5 and GR00T N1.7 is most sensitive "
          "⟦wulff2026grounding⟧." if _says_left(tids) else "")
    pi = [(t, T["side_effects"][t]["pi05"]) for t in ("T2", "T3", "T4", "T5", "T6", "T7", "R1")
          if _ok(T["side_effects"].get(t, {}).get("pi05"), ("a", "b", "n"))]
    up = [t for t, c in pi if c.get("p") is not None and c["p"] < SIG and c["b"] > c["a"]]
    down = [t for t, c in pi if c.get("p") is not None and c["p"] < SIG and c["b"] < c["a"]]
    s3 = (f"The action-expert VLA’s cube-then-banana edit, run the same way, lifted {and_list(tn(t) for t in up)} and "
          "broke nothing "
          "clearly." if pi and up and not down else "")
    return "".join(f"<p>{esc(x)}</p>" for x in (s1, s2, s3) if x)


def groot_text(tables):
    """GR00T N1.7: it fails at the motion; the clearer wording barely helps; the copy makes it worse."""
    T = _cells(tables)
    fm = [(t, T["failure_map"].get(t, {}).get("groot")) for t in [x[0] for x in TASKS]]
    fm = [(t, c) for t, c in fm if _ok(c, ("k", "n"))]
    if not fm:
        return f'<p>GR00T N1.7’s runs are <span class="run">running</span>.</p>'
    some = [t for t, c in fm if c["k"] > 0]
    wd = T["wording"].get("T1", {}).get("groot")
    s1 = (f"GR00T N1.7, our reasoning-backbone VLA, mostly fails at the motion itself, not at language grounding. "
          f"Unedited, it completes only "
          f"{len(some)} of the {len(fm)} tasks at all.")
    if _ok(wd):
        clear = wd.get("pv") is not None and wd["pv"] < SIG
        s1 += (f" A clearer wording helps only on {tn('T1')}, and {'' if clear else 'not clearly: '}"
               f"{wd['a']} → {wd['b']} of {wd['n']} practice starts.")
    e = T["edit_gain"].get("T1", {}).get("groot")
    rc0 = T["random_control"].get("T1", {}).get("groot") or {}
    s2 = ""
    if _ok(e) and e["b"] < e["a"]:
        s2 = (f"Copying that wording into the weights made things worse wherever we put it: {e['a']} → {e['b']} of {e['n']}"
              + (", while a random edit of the same size stayed near where it started." if rc0.get("r") is not None
                 and abs(rc0["r"] - e["a"]) <= max(3, e["n"] // 10) else ".")
              + " The clearer wording adds a clause, so the edit can only land on the matched words, and there it "
                "moves the robot’s first reach.")
    return f"<p>{esc(s1)}</p>" + (f"<p>{esc(s2)}</p>" if s2 else "")


def control_text(P, tables):
    """π0.5, the control: this page's own runs first (each edit against its random edit and the clearer wording on
    the same starts, then the edit on the other tasks), one clause tying them to the paper's earlier runs, and the
    earlier finding this page did not rerun, labelled as such."""
    e, s4, c = P["edit"], P["s4ho"], P["ctl"]
    T = _cells(tables)
    fresh = lambda tid, tab: (lambda x: x if _ok(x) and not x.get("prov") else None)(T[tab].get(tid, {}).get("pi05"))

    def run(tid):
        f = fresh(tid, "edit_gain")
        if not f:
            return None
        r = T["random_control"].get(tid, {}).get("pi05") or {}
        r = r if r.get("s") == "ok" and not r.get("prov") else None
        t = T["recovery"].get(tid, {}).get("pi05") or {}
        extra = []
        if r and r.get("r") is not None and r.get("n") == f["n"]:
            extra.append(f"{r['r']} for a random edit of the same size")
        if t.get("s") == "ok" and not t.get("prov") and t.get("t") is not None and t.get("n") == f["n"]:
            extra.append(f"{t['t']} for the clearer wording itself")
        return (f"its {tn(tid)} edit went from {f['a']} to {f['b']} of {f['n']}"
                f"{' practice' if f.get('prac') else ' unseen'} starts" + (f", against {and_list(extra)}" if extra else ""))
    runs = [x for x in (run("T1"), run("T8")) if x]
    s = "π0.5 is where the method comes from, and our control."
    if runs:
        s += (" On this page’s starts, " + ". ".join(x[0].upper() + x[1:] if i else x for i, x in enumerate(runs))
              + ". That matches the paper’s earlier runs "
              f"({tn('T1')} {e['base']} → {e['cand']} of {e['n']}, {tn('T8')} {s4['base']} → {s4['cand']} of {s4['n']}).")
    else:
        s += (f" In earlier runs, from other starts, the edit lifted {tn('T1')} from {e['base']} to {e['cand']} of {e['n']} "
              f"unseen starts and {tn('T8')} from {s4['base']} to {s4['cand']} of {s4['n']}; a random edit of the same "
              f"size scored {c['rand']} of {c['n']} on {tn('T1')}, against {c['base']} unedited.")
    se = [(t, T["side_effects"][t]["pi05"]) for t in ("T2", "T3", "T4", "T5", "T6", "T7", "T8", "R1")
          if _ok(T["side_effects"].get(t, {}).get("pi05"), ("a", "b", "n")) and not T["side_effects"][t]["pi05"].get("prov")
          and not T["side_effects"][t]["pi05"].get("target")]
    if se:
        sig = lambda c: c.get("p") is not None and c["p"] < SIG
        up = [f"{tn(t)} from {c['a']} to {c['b']} of {c['n']}" for t, c in se if sig(c) and c["b"] > c["a"]]
        down = [f"{tn(t)} from {c['a']} to {c['b']} of {c['n']}" for t, c in se if sig(c) and c["b"] < c["a"]]
        flat = [f"{tn(t)} ({c['a']} → {c['b']} of {c['n']})" for t, c in se if not sig(c)]
        bits = ([f"lifted {and_list(up)}"] if up else []) + ([f"lowered {and_list(down)}"] if down else [])
        s += (f" Run unchanged on the other tasks, the {tn('T1')} edit " + "; ".join(bits)
              + (", and " if bits and flat else "") + (f"made no clear change on {and_list(flat)}" if flat else "") + ".")
    else:
        lifted = [t for t, k in (("T2", "eye"), ("T3", "left")) if P.get(k) and P[k]["p"] < SIG and P[k]["cand"] > P[k]["base"]]
        if lifted:
            s += f" In the earlier runs the {tn('T1')} edit also lifted {and_list(tn(t) for t in lifted)}."
    s += (f" In the paper’s earlier runs, edits aimed at {and_list(tn(t) for t in ('T4', 'T5', 'T6', 'T7'))} did not help: "
          "their failures are seeing, grasping or a missing skill, not language grounding.")
    return f"<p>{esc(s)}</p>"


def wrap(s, width):
    """Greedy line breaks at about width characters."""
    out, cur = [], ""
    for w_ in str(s).split():
        if cur and len(cur) + 1 + len(w_) > width:
            out.append(cur)
            cur = w_
        else:
            cur = (cur + " " + w_).strip()
    return out + ([cur] if cur else [])


def dg_where_results(sites):
    """Figure 5: how often the robot completed its task with the edit in each part, as a share of that model's starts
    (dot, with the count), against the same robot without any edit (tick) and with a random edit of the same size run
    from the same starts (ring). Two drawings of the same data, one for wide and one for narrow screens; the first row
    of each model names the tick's and the ring's values in text, every mark names its own in a tooltip."""
    rows = [(m, s) for m in sites for b in m["blocks"] for s in b["sites"]]
    if not rows:
        return ""
    desc = "; ".join(f"{m['name']}, {m['task']}, {s['row'].lower()}: {s['k']} of {s['n']} starts completed "
                     f"({100 * s['k'] / s['n']:.0f}%), without any edit {s['a']}"
                     + (f", with a random edit {s['ctl']}" if s.get("ctl") is not None else "") for m, s in rows) + "."
    svgs = ""
    for wide in (True, False):
        body, vw, h = _f5_layout(sites, wide)
        svgs += _svg(vw, h, F5_TITLE, desc, body, "dg-results " + ("dg5-w" if wide else "dg5-n"),
                     "dg5w" if wide else "dg5n")
    key = "".join(f'<li><svg class="dg dg-key" viewBox="-8 -8 16 16" aria-hidden="true">{_glyph(k, 0, 0)}</svg>'
                  f'{esc(t)}</li>' for k, t in F5_KEY)
    # the takeaway, from the records: each model's first reading-part and motion-part edit
    first = lambda m, let: next((s for b in m["blocks"] if b["let"] == let for s in b["sites"]), None)
    pc = lambda k, n: f"{100 * k / n:.0f}%"
    good, bad = [], []
    for m in sites:
        a, b = first(m, "A"), first(m, "B")
        if a and a["oc"] in ("fixed", "partial"):
            good.append((m, a, b))
        elif a and a["oc"] == "broke" and (not b or b["oc"] == "broke"):
            bad.append((m, a, b))
    take = ""
    tname = lambda m: ARCH[m["id"]]["short"]
    pname = lambda m, let: ARCH[m["id"]]["parts"][let][0]
    if good:
        mo = [(m, b) for m, _, b in good if b]
        take = (" With the edit in the " + or_list(dict.fromkeys(pname(m, "A") for m, _, _ in good)) + ", "
                + and_list(f"the {tname(m)} went from {pc(a['a'], a['n'])} to {pc(a['k'], a['n'])}" for m, a, _ in good)
                + (f"; with it in the {or_list(dict.fromkeys(pname(m, 'B') for m, _ in mo))}, "
                   + (f"they completed {and_list(pc(b['k'], b['n']) for _, b in mo)}" if len(mo) > 1 else
                      f"the {tname(mo[0][0])} completed {pc(mo[0][1]['k'], mo[0][1]['n'])}") if mo else "") + ".")
    for m, a, b in bad:
        take += (f" The {tname(m)} fell from {pc(a['a'], a['n'])} to {pc(a['k'], a['n'])}"
                 + (" with the edit in either component." if b and b["k"] == a["k"] else
                    f" with the edit in the {pname(m, 'A')} and to {pc(b['k'], b['n'])} in the {pname(m, 'B')}." if b
                    else "."))
    cap = ("Each dot is the share of starts the robot completed after the edit was placed in that component. The tick is "
           "the same robot without the edit." + take + " Teal dot: clearly better than without the edit (exact McNemar "
           "p < 0.05); dot with a cross: "
           "clearly worse; grey dot: no clear change. The ring, a random edit of the same size, is a placebo, run "
           "from the same starts.")
    if any(m["id"] == "pi05" for m in sites):
        cap += (" *Written by an AI editor (Claude) working alone, who never saw our edit, by weight-space programming "
                "⟦takehi2026sparks⟧; the other π0.5 rows copy the clear wording.")
    return (f'<figure class="dgfig dg2b"><p class="dg-ttl">{esc(F5_TITLE)}</p><ul class="dg-legend dg5-key">{key}</ul>'
            f'{svgs}<figcaption>{esc(cap)}</figcaption></figure>')


def where_text(tables):
    """Which parts fixed the task, in plain words, then the replays, then the bridge to the wording."""
    T = _cells(tables)
    c = (T["where"].get("offline_where") or {}).get("cosmos")
    s = ""
    if c and c.get("s") == "ok" and c.get("parts"):
        parts = {loc_class(loc): v for loc, v, _ in c["parts"]}
        if "backbone" in parts and "action" in parts:
            s = (f"<p>Interpretability work points the same way: in π0.5, instruction information reaches the action "
                 f"expert through a narrow band of layers ⟦swann2026causal⟧. Open-loop replays of recorded decisions, our "
                 f"offline check before any robot run, agree. In the "
                 f"world-action model, the text-pathway edit reproduces {sgn(parts['backbone'], False, 0)}% of the "
                 f"clearer wording’s effect on the planned actions; the video/action-pathway edit, "
                 f"{sgn(parts['action'], False, 0)}%.</p>")
    return s + "<p>Where the edit goes is one condition. The other is the wording itself.</p>"


def paper_line(tables):
    """The records' one line on π0.5's provenance: which of its numbers come from the paper's evaluation, with their
    slots and render preset; every other π0.5 number is from the shared starts."""
    names = {"failure_map": "unedited success", "wording": "clearer wording", "edit_gain": "edit",
             "random_control": "random edit", "recovery": "recovery", "side_effects": "side effect",
             "where": "where to edit", "size": "edit size", "retention": "retention"}
    got, slots = {}, []
    for t in tables:
        if t["id"] not in names:
            continue
        for r in t["rows"]:
            c = r["cells"].get("pi05") or {}
            if c.get("s") == "ok" and c.get("prov"):
                got.setdefault(names[t["id"]], []).append(dename(TASK_NAME.get(r["key"], "")) if r["key"] in TASK_NAME
                                                          else "")
                slots += [int(x) for x in re.findall(r"(?<![\d])(9\d{4}|[2-7]\d{4})(?![\d])", c.get("d") or "")]
    if not got:
        return "Every π0.5 number is from the shared starts, with RoboLab’s default render preset."
    parts = [f"{k} ({', '.join(x for x in v if x)})" if any(v) else k for k, v in got.items()]
    return (f"π0.5 numbers from the paper’s evaluation, on slots {min(slots)}–{max(slots)} with RoboLab’s performance "
            f"render preset: {'; '.join(parts)}. Every other π0.5 number is from the shared starts, with the default "
            "preset." if slots else f"π0.5 numbers from the paper’s evaluation: {'; '.join(parts)}.")


def legend_html(cross, tables=None):
    """The records' key: each task's name, internal code, RoboLab class and default instruction, then the internal
    terms the tables below use. The only place on the page where the codes appear."""
    rows = (cross.get("cosmos") or {}).get("rows") or []
    instr = {}
    for r in rows:
        wp = r.get("wording_pair")
        if r["_task"] and r["_task"] not in instr and isinstance(wp, list) and wp and wp[0]:
            instr[r["_task"]] = wp[0]
    body = "".join(f'<tr><th scope="row">{esc(TASK_NAME[t])}</th><td class="c" data-model="Code"><span class="v">'
                   f'<code>{t}</code></span></td><td class="c" data-model="RoboLab class"><span class="v"><code>'
                   f'{esc(TASK[t][2])}</code></span></td><td class="c" data-model="Default instruction"><span class="v">'
                   f'{esc(instr.get(t, "—"))}</span></td></tr>' for t in TASK_NAME)
    body += "".join(f'<tr><th scope="row">{esc(x["plain"])}</th><td class="c" data-model="Code"><span class="v"><code>'
                    f'{x["code"]}</code></span></td><td class="c" data-model="RoboLab class"><span class="v"><code>'
                    f'{esc(PROSPECT["pairs"][x["code"]]["cls"])}</code></span></td><td class="c" data-model="Default '
                    f'instruction"><span class="v">{esc((x.get("wording") or ["—"])[0])}</span></td></tr>'
                    for x in (PROSPECT_LAST.get("p") or []))
    table = ('<table class="ctable ltable"><thead><tr><th scope="col">Task</th><th scope="col"><span class="mn">Code'
             '</span></th><th scope="col"><span class="mn">RoboLab class</span></th><th scope="col"><span class="mn">'
             f'Default instruction</span></th></tr></thead><tbody>{body}</tbody></table>')
    terms = [
        ("Slots", "the seed indices of the starting scenes: practice starts 30000–30031, unseen (held-out) starts "
                  "96000–96063, the same for every model. " + (paper_line(tables) if tables else "")),
        ("13 / 0, fixed / broke", "starts where only the edited robot succeeded / starts where only the unedited robot "
                                  "did: the discordant pairs."),
        ("p", "exact McNemar test on those starts ⟦mcnemar1947⟧: the chance of a split at least this lopsided if the "
              "edit did nothing."),
        ("+87.5 points [78.1, 95.3]", "change in success, in percentage points, with its paired bootstrap 95% "
                                      "interval."),
        ("Teacher, clear wording", "the unedited model given the clearer wording. Recovery: (edited − unedited) / "
                                   "(teacher − unedited)."),
        ("Random control", "a random edit with the same layers and the same size, a placebo."),
        ("Offline, gap closed", "in replays of recorded moments, the share of the clear wording’s change to the planned "
                                "actions that an edit reproduces. Deviation: how far an edit moves the planned hand "
                                "position from the unedited robot’s."),
        ("Distilled edit", "an edit that copies the clear wording, this page’s method, a closed-form form of context "
                           "distillation ⟦askell2021context,snell2022distilling⟧. Blind editor: an AI editor "
                           "(Claude or GPT) that worked alone on one task and never saw our edits, a weight-space "
                           "programming agent ⟦takehi2026sparks⟧; its accepted π0.5 "
                           "edit is gated (it has an on/off switch). Least norm: the smallest change that does the job, "
                           "always on."),
        ("Model components", "π0.5 ⟦pi2025pi05⟧: VLM backbone (PaliGemma), action expert (flow matching "
                             "⟦lipman2023flowmatching⟧) and action output "
                             "projection (its last linear layer). GR00T N1.7 ⟦nvidia2025gr00tn1⟧: VLM backbone "
                             "(Cosmos-Reason2) and action head (a flow-matching DiT ⟦peebles2023dit⟧). Cosmos3 Edge, a "
                             "world-action model ⟦shen2026wamsurvey,nvidia2026cosmos3⟧: text pathway (the instruction "
                             "tokens) and "
                             "video/action or generation pathway (video and action tokens, which read the text tokens’ "
                             "keys and values in every layer)."),
        ("Edit names", "t8und: the edit for Mustard right of raisin box, “the mustard edit”; t3und: the edit for Cube "
                       "left of bowl, “the cube-left edit”, built the same way from its more specific wording; t8t3und: "
                       "the cube-left edit built on top of the mustard edit, “both edits together”; t8gen: the same copy written into Cosmos3 "
                       "Edge’s video/action pathway; t8undp: + 65 unrelated instructions kept unchanged (preservation keys: "
                       "zero-change constraints in the solve); "
                       "t8undr: + 24 left/right instructions from other tasks kept unchanged; t8unds: only the words "
                       "both wordings share (template positions) + the 65 unrelated instructions."),
        ("Render preset", "the simulator’s image quality setting; all fresh runs use RoboLab’s default."),
    ]
    dl = "".join(f"<dt>{esc(a)}</dt><dd>{esc(b)}</dd>" for a, b in terms)
    return (f'<h3 id="legend">Task names and terms</h3><p>The tables below use the tasks’ internal codes and a few '
            f'technical terms. The names on the rest of the page map to them as follows.</p>'
            f'<div class="tblock">{table}</div><dl class="defs">{dl}</dl>')


def dg_blog_pi05(sites):
    """The main post's figure (site/index.html, pasted there by hand): π0.5's three components by their technical
    names, one verdict and one count each, all on the same 64 new starts."""
    m = next(x for x in sites if x["id"] == "pi05")
    a = ARCH["pi05"]
    W, x0, x1 = [], 30, 266
    n, base = m["n"], m["base"]
    W.append(_t(14, 20, "π0.5", "dg-h") + _t(14, 38, f"{m['task']} · {n} new starts", "dg-s dg-mu")
             + _t(14, 55, f"the unedited robot: {base} of {n}", "dg-s dg-mu"))
    W.append(f'<rect x="30" y="66" width="80" height="24" rx="12" class="dg-in"/>' + _t(70, 82.5, "cameras", "dg-s", "middle"))
    W.append(f'<rect x="150" y="66" width="96" height="24" rx="12" class="dg-in"/>'
             + _t(198, 82.5, "instruction", "dg-s", "middle"))
    y, tops, desc, h = 116, {}, [], 72
    blocks = [b for b in m["blocks"] if b["sites"]]
    for i, blk in enumerate(blocks):
        s = blk["sites"][0]
        name, det = a["parts"][blk["let"]]
        tops[blk["let"]] = y
        W.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h}" rx="7" class="dg-blk"/>'
                 + _t(48, y + 23, name, "dg-b") + _t(48, y + 40, f"({det})", "dg-s dg-mu"))
        W.append(f'<g><title>{esc(BADGE[s["oc"]])}, {s["k"]} of {s["n"]}</title>'
                 + _mark(54, y + 54.5, s["oc"]) + _t(67, y + 59, BADGE[s["oc"]], "dg-s dg-b")
                 + _t(x1 - 8, y + 59, f"{s['k']} of {s['n']}", "dg-s dg-b", "end") + "</g>")
        desc.append(f"{name} ({det}): {BADGE[s['oc']]}, {s['k']} of {s['n']}")
        y += h + (30 if i < len(blocks) - 1 else 0)
    cx = 148
    lets = [b["let"] for b in blocks]
    W.append(_arrow([(198, 90), (198, tops["A"])]) + _arrow([(70, 90), (70, tops["A"])]))
    for p, q in zip(lets, lets[1:]):
        W.append(_arrow([(cx, tops[p] + h), (cx, tops[q])])
                 + (_t(cx + 8, tops[p] + h + 19, a["links"][p], "dg-xs dg-mu") if p in a["links"] else ""))
    ye = tops[lets[-1]] + h
    W.append(_arrow([(cx, ye), (cx, ye + 22)]) + _t(cx, ye + 40, "robot", "dg-s", "middle"))
    svg = _svg(280, ye + 48, "π0.5: where the edit went and what it did",
               f"{m['task']}, {n} new starts; the unedited robot succeeded {base} times. " + "; ".join(desc) + ".",
               "".join(W), "dg-panel", "dg2-wp")
    return (f'<figure class="fig dg-blog">\n      {svg}\n'
            f'      <figcaption>The same fix, built into three parts of π0.5 and scored on {n} new starts: only in the '
            f'vision-language model, where the instruction is read, did it fully work.</figcaption>\n    </figure>')


def _tw(s, size, bold=False):
    """Approximate rendered width of s in Inter at size px (within a few per cent of the browser's), for layout."""
    em = sum(_TW.get(c, 0.68 if c.isupper() else 0.57) for c in str(s))
    return em * size * (1.03 if bold else 1.0)


def wrap_px(s, width, size, bold=False):
    """Greedy line breaks so that no line is wider than width px at size."""
    out, cur = [], ""
    for w_ in str(s).split():
        t = (cur + " " + w_).strip()
        if cur and _tw(t, size, bold) > width:
            out.append(cur)
            cur = w_
        else:
            cur = t
    return out + ([cur] if cur else [])


_TW = {**{c: 0.61 for c in "0123456789"}, **{c: 0.25 for c in "ijl.,:;'’|!"}, "·": 0.32, **{c: 0.36 for c in "frt()/[]"},
       "m": 0.86, "w": 0.76, " ": 0.27, "%": 0.9, "*": 0.45, "−": 0.6, "→": 0.6, "-": 0.4, "±": 0.6, "+": 0.6}
F5_TITLE = "How often the robot completed the task, with the edit placed in each component"
F5_AXIS = "Task success: share of starts the robot completed"
F5_KEY = (("fix", "robot with the edit"), ("ring", "robot with a random edit of the same size"),
          ("tick", "robot without any edit"))


def _glyph(kind, x, y, r=5.5):
    """The chart's three marks, for the key and the labels: a dot (the edit), a ring (a random edit), a tick (no edit)."""
    if kind == "tick":
        return f'<line x1="{x:g}" y1="{y - 8:g}" x2="{x:g}" y2="{y + 8:g}" class="dg-base"/>'
    if kind == "ring":
        return f'<circle cx="{x:g}" cy="{y:g}" r="{r:g}" class="dg-o-none"/>'
    return f'<circle cx="{x:g}" cy="{y:g}" r="{r:g}" class="dg-o-fix"/>'


def _f5_layout(sites, wide):
    """One drawing of Figure 5. wide: the row labels in a left column beside the tracks (screens 700 px and up);
    narrow: each label above its track, wrapped to the width. Returns (svg body, viewBox width, height)."""
    VW = 680 if wide else 320
    LX, X0, X1 = 0, (292 if wide else 9), VW - 9
    LW = (X0 - 22) if wide else VW - 4           # room for a row label
    xs = lambda pct: X0 + (X1 - X0) * pct / 100
    pc = lambda k, n: 100 * k / n
    S, XS = 12.5, 11.5
    W = []
    # axis: what the positions mean, then the ticks
    axl = wrap_px(F5_AXIS, X1 - (X0 if wide else LX), XS, True)
    y = 12
    for ln in axl:
        W.append(_t(X0 if wide else LX, y, ln, "dg-xs dg-b"))
        y += 15
    y += 2
    for v in (0, 25, 50, 75, 100):
        W.append(_t(xs(v), y, f"{v}%", "dg-xs dg-mu", "start" if v == 0 else "end" if v == 100 else "middle"))
    y += 7
    W.append(f'<line x1="{xs(0):g}" y1="{y}" x2="{xs(100):g}" y2="{y}" class="dg-grid"/>')
    y += 24
    for m in sites:
        rows = [s for b in m["blocks"] for s in b["sites"]]
        if not rows:
            continue
        W.append(_t(LX, y, f"{ARCH[m['id']]['type']} ({m['name']})", "dg-s dg-b"))
        sub = wrap_px(f"{m['task']} · {m['n']} {m['kind']} starts, the same in every row", X1 - LX, XS)
        for q, ln in enumerate(sub):
            W.append(_t(LX, y + 16 + 14 * q, ln, "dg-xs dg-mu"))
        y += 36 + 14 * (len(sub) - 1)
        for i, s in enumerate(rows):
            n = s["n"]
            pe, pa = pc(s["k"], n), pc(s["a"], n)
            pr = pc(s["ctl"], n) if s.get("ctl") is not None else None
            lab = wrap_px(s["row"], LW, S)
            if wide:
                yt = y + 7                                # track centre; label lines start level with it
                ly = [yt + 4 + 15 * j for j in range(len(lab))]
            else:
                ly = [y + 15 * j for j in range(len(lab))]
                yt = ly[-1] + 15
            marks = [(xs(pa), 3)] + ([(xs(pr), 7)] if pr is not None else [])   # other marks' x and half-width
            # the dot's value: beside it if it clears the other marks and the frame, else on a line below
            val = f"{pe:.0f}% · {s['k']} of {n}"
            vw = _tw(val, XS, True) + 4
            x = xs(pe)
            vpos = None
            for a, b, anc in ((x + 10, x + 10 + vw, "start"), (x - 10 - vw, x - 10, "end")):
                if a >= X0 - 8 and b <= VW and not any(a - h - 3 < mx < b + h + 3 for mx, h in marks):
                    vpos = (x + 10 if anc == "start" else x - 10, anc)
                    break
            notes = []                                    # (mark x, glyph, text, bold) on lines under the track
            if vpos is None:
                notes.append((x, "dot", val, True))
            if i == 0:
                notes.append((xs(pa), "tick", f"without any edit: {pa:.0f}% · {s['a']} of {n}", False))
                if pr is not None:
                    notes.append((xs(pr), "ring", f"with a random edit: {pr:.0f}% · {s['ctl']} of {n}", False))
            verdict = {"fixed": "clearly better than without the edit", "partial": "clearly better than without the edit",
                       "broke": "clearly worse than without the edit", "none": "no clear change"}[s["oc"]]
            g = [f'<g><title>{esc(s["row"])}</title>' + "".join(_t(LX, yy, ln, "dg-s") for yy, ln in zip(ly, lab)) + "</g>",
                 f'<line x1="{xs(0):g}" y1="{yt}" x2="{xs(100):g}" y2="{yt}" class="dg-track"/>',
                 f'<g><title>Robot without any edit: {s["a"]} of {n} starts completed ({pa:.0f}%)</title>'
                 f'<line x1="{xs(pa):.1f}" y1="{yt - 8}" x2="{xs(pa):.1f}" y2="{yt + 8}" class="dg-base"/>'
                 f'<rect x="{xs(pa) - 5:.1f}" y="{yt - 9}" width="10" height="18" fill="none" pointer-events="all"/></g>']
            if pr is not None:
                g.append(f'<g><title>Robot with a random edit of the same size: {s["ctl"]} of {n} starts completed '
                         f'({pr:.0f}%)</title><circle cx="{xs(pr):.1f}" cy="{yt}" r="5.5" class="dg-o-none"/>'
                         f'<circle cx="{xs(pr):.1f}" cy="{yt}" r="9" fill="none" pointer-events="all"/></g>')
            cls = "dg-o-fix" if s["oc"] in ("fixed", "partial") else "dg-o-none2"
            dot = _mark(x, yt, "broke", 6.5) if s["oc"] == "broke" else f'<circle cx="{x:.1f}" cy="{yt}" r="6" class="{cls}"/>'
            g.append(f'<g><title>Robot with the edit: {s["k"]} of {n} starts completed ({pe:.0f}%), {verdict}</title>'
                     f'{dot}<circle cx="{x:.1f}" cy="{yt}" r="11" fill="none" pointer-events="all"/>'
                     + (_t(vpos[0], yt + 4, val, "dg-xs dg-b", vpos[1]) if vpos else "") + "</g>")
            yy = yt + 22
            for mx, kind, text, bold in notes:
                tw = _tw(text, XS, bold) + 4
                if mx + 9 + tw <= VW:
                    gx, tx, anc = mx, mx + 9, "start"
                elif mx - 9 - tw >= 0:
                    gx, tx, anc = mx, mx - 9, "end"
                else:                                     # too long either side: keep it in the frame, glyph first
                    gx = max(4, VW - tw - 9)
                    tx, anc = gx + 9, "start"
                gl = (_mark(gx, yy - 4, "broke", 4.5) if kind == "dot" and s["oc"] == "broke" else
                      f'<circle cx="{gx:.1f}" cy="{yy - 4}" r="4" class="{cls}"/>' if kind == "dot" else
                      f'<line x1="{gx:.1f}" y1="{yy - 10}" x2="{gx:.1f}" y2="{yy + 2}" class="dg-base"/>' if kind == "tick"
                      else f'<circle cx="{gx:.1f}" cy="{yy - 4}" r="4" class="dg-o-none"/>')
                g.append(gl + _t(tx, yy, text, "dg-xs dg-b" if bold else "dg-xs", anc))
                yy += 16
            W.append("".join(g))
            bottom = max(yt + 9, (yy - 12) if notes else 0, ly[-1] + 4)
            y = bottom + (16 if wide else 24)
        y += 30 if wide else 22
    return "".join(W), VW, y - (30 if wide else 22) - (16 if wide else 24) + 8


def second_edit(cross, tables):
    """"Can a second edit undo the damage?": the T8 edit, Cosmos3 Edge's own edit for T3 (the second wording-sensitive
    task) and the two applied together (the T3 update built on the T8-edited weights), each on T8, T3 and T5. A cell is
    unedited → edited on unseen starts where run, else practice (tagged); "running" until a record exists; "—" once
    the model's record is final, or when the build failed its replay check (no closed loop, README rules)."""
    src = cross.get(SECOND["model"])
    rows = [r for r in (src["rows"] if src else []) if not r["_prov"]]
    final = bool(src and src["final"])
    out = []
    for e in SECOND["edits"]:
        mine = [r for r in rows if edit_key(r) == e["name"]]
        cells = {}
        for tid in SECOND["cols"]:
            rs = [r for r in mine if r["_task"] == tid and None not in (ints(r)["unedited_k"], ints(r)["edited_k"],
                                                                         ints(r)["n"])]
            cells[tid] = prevent_closed(rs)
        closed = any(cells.values())
        build, gate = edit_records(src, SECOND["model"], e["name"]) if e["name"] != SECOND["edits"][0]["name"] else (None, None)
        gate_failed = gate if gate and not gate["passed"] and not closed else None
        offl = [(r["_task"], ofind(o, "gap_closed_pct")[0]) for r, o in ((r, offline_of(r)) for r in rows)
                if o and nk(o.get("edit") or "") == e["name"] and str(o.get("group") or "check").lower() == "check"]
        failed = [(t, v) for t, v in offl if v is not None and v < SECOND["gate"]]
        for tid in SECOND["cols"]:
            if cells[tid] is None and gate_failed:
                cells[tid] = C("—", "didn’t pass the offline check, not run on the robot", s="na")
            if cells[tid] is None:
                cells[tid] = (C("—", "not run: in replays it reproduced only " + and_list(
                    f"{sgn(v, False, 0)}% of the clear wording’s effect on {tn(t)}" for t, v in failed), s="na")
                              if failed and not closed else NA if final else RUN)
        out.append(dict(name=e["name"], plain=e["plain"], built=e["built"], cells=cells, measured=bool(mine),
                        gate_failed=gate_failed, build=build))
    T = _cells(tables)
    w = T["wording"].get(SECOND["second"], {}).get(SECOND["model"])
    wp = next((r.get("wording_pair") for r in rows if r["_task"] == SECOND["second"]
               and isinstance(r.get("wording_pair"), list) and len(r["wording_pair"]) == 2 and all(r["wording_pair"])),
              None)
    return dict(model=SECOND["model"], cols=list(SECOND["cols"]), rows=out, wording=w if _ok(w) else None,
                wording_pair=wp, text=second_finding(out),
                sources=[SECOND["records"], SECOND["readme"]])


def second_finding(rows):
    """One sentence from the closed-loop records, once they exist: what the second edit did on its own task, and what
    the two edits together did to the three tasks. Empty before any record."""
    pre, told = [], set()
    for i, r in enumerate(x for x in rows if x.get("gate_failed")):
        g, b = r["gate_failed"], r.get("build")
        they = "they" if r["plain"].startswith("Both") else "it"
        parts = []
        for t, v in g["gaps"]:
            ref = g.get("refs", {}).get(t)
            parts.append((f"further from the clearer wording’s on {tn(t)} ({sgn(v, True, 0)}% of the gap)" if v < 0 else
                          f"{v:.0f}% of the way toward the clearer wording’s on {tn(t)}")
                         + (f", down from {ref:.0f}% with the mustard edit alone" if ref is not None and ref > v else ""))
        lead = (f"{r['plain']} didn’t pass the offline check, so {they} never ran on the robot" if i == 0 else
                f"{r['plain']} didn’t pass the offline check either")
        s_ = lead + (": in replays" + (f" of {g['n']} held-aside moments" if g.get("n") else "") + f", {they} moved the "
                     "robot’s planned motion " + and_list(parts)
                     + (f"; a random edit of the same size, {sgn(g['ctl'], True, 1)}%" if g.get("ctl") is not None else "")
                     if parts else "") + "."
        key = (b["orig"], b["good"], b.get("matched")) if b else None
        if b and b["orig"] != b["good"] and b.get("matched") and key not in told:
            s_ += (f" The two wordings are {b['orig']} and {b['good']} tokens long, so the edit could only copy the "
                   f"{b['matched']} template positions they share, as in the rebuild above that copies only the words "
                   "both wordings share.")
            told.add(key)
        pre.append(s_)
    by = {r["name"]: r["cells"] for r in rows}
    ok = lambda c: c and c["s"] == "ok"
    sig = lambda c: c.get("p") is not None and c["p"] < SIG
    kind = lambda c: " practice" if c.get("prac") else " unseen"

    def moved(t, c):
        if sig(c) and c["b"] < c["a"]:
            return f"{tn(t)} fell from {c['a']} to {c['b']} of {c['n']}{kind(c)} starts"
        if sig(c) and c["b"] > c["a"]:
            return f"{tn(t)} rose from {c['a']} to {c['b']} of {c['n']}{kind(c)} starts"
        return f"{tn(t)} went from {c['a']} to {c['b']} of {c['n']}{kind(c)} starts, no clear change"
    bits = []
    two = by.get(SECOND["edits"][1]["name"]) or {}
    c3 = two.get(SECOND["second"])
    if ok(c3):
        bits.append(f"on its own, the cube-left edit {'lifted' if sig(c3) and c3['b'] > c3['a'] else 'took'} "
                    f"{tn(SECOND['second'])} from {c3['a']} to {c3['b']} of {c3['n']}{kind(c3)} starts"
                    + ("" if sig(c3) else ", a change these starts cannot tell from chance"))
    both = by.get(SECOND["edits"][2]["name"]) or {}
    parts = []
    c8 = both.get(SECOND["target"])
    if ok(c8):
        parts.append(("kept the mustard fix" if sig(c8) and c8["b"] > c8["a"] else "lost the mustard fix")
                     + f" ({c8['a']} → {c8['b']} of {c8['n']}{kind(c8)} starts)")
    parts += [moved(t, both[t]) for t in SECOND["cols"] if t != SECOND["target"] and ok(both.get(t))]
    if parts:
        bits.append("together, the two edits " + parts[0] + (": " + "; ".join(parts[1:]) if len(parts) > 1 else ""))
    if not bits:
        return " ".join(pre)
    s = "; ".join(bits)
    return " ".join(pre + [s[0].upper() + s[1:] + "."])


def second_html(S, full=True):
    """The subsection: two or three plain sentences, the table (edits as rows; Mustard right of raisin box, Cube left
    of bowl, Mustard in left bin as columns), and the takeaway once closed-loop records exist. full: the records'
    version, with the edits' codes, details and sources."""
    run = '<span class="run">running</span>'
    cols = [(t, TASK_NAME[t], SECOND["col_note"][t]) for t in S["cols"]]
    head = "".join(f'<th scope="col"><span class="mn">{esc(a)}</span><span class="mk">{esc(b)}</span></th>'
                   for _, a, b in cols)
    body = []
    for r in S["rows"]:
        rh = (f'<code>{esc(r["name"])}</code><span class="rd">{esc(r["plain"])}, {esc(r["built"])}</span>' if full
              else f'{esc(r["plain"])}<span class="rd">{esc(r["built"])}</span>')
        g = r.get("gate_failed")
        if g:
            det = "; ".join(x for x in ([f"{tn(t)}: {sgn(v, True, 1)}% of the clearer wording’s effect in replays"
                                         for t, v in g["gaps"] if t]
                                        + ([f"random edit {sgn(g['ctl'], True, 1)}%"] if g.get("ctl") is not None else [])
                                        + ([f"rule: {g['rule']}"] if g.get("rule") else [])) if x)
            body.append(f'<tr><th scope="row">{rh}</th><td class="c st-na" colspan="{len(cols)}" data-model="On the robot">'
                        f'<span class="v">Didn’t pass the offline check, so not run on the robot</span>'
                        + (f'<span class="d">{esc(det)}</span>' if full and det else "") + "</td></tr>")
            continue
        body.append(f'<tr><th scope="row">{rh}</th>'
                    + "".join(td_html(r["cells"][k] if full or r["cells"][k]["s"] == "na" else
                                      {x: v for x, v in r["cells"][k].items() if x != "d"}, f"{a}, {b}")
                              for k, a, b in cols) + "</tr>")
    table = (f'<table class="ctable ptable ptable-s"><thead><tr><th scope="col"><span class="vh">Edit</span></th>{head}'
             f'</tr></thead><tbody>{"".join(body)}</tbody></table>')
    w, wp = S.get("wording"), S.get("wording_pair")
    t3 = tn(SECOND["second"])
    first = (f"Cosmos3 Edge’s grounding also fails on {t3}: a more specific wording"
             + (f", “{wp[1].rstrip('.')}”," if wp and full else "")
             + (f" lifts it from {w['a']} to {w['b']} of {w['n']} practice starts." if w else " helps it."))
    intro = (f"<p>{esc(first)} So we gave that task its own edit, built the same way as the mustard edit, and then "
             f"applied both edits together, the second built on top of the first. Cells are unedited → edited, from the "
             f"same starts: unseen starts unless tagged practice.</p>")
    finding = f'<p>{esc(S["text"])}</p>' if S.get("text") else ""
    if not full:
        return intro + f'<div class="tblock wide">{table}</div>' + finding
    note = ("Closed loop only. Rules fixed before either build (README): the edits run on the robot only if, in "
            f"replays, they reproduce at least {SECOND['gate']}% of the clearer wording’s effect; unseen starts only if "
            "practice passes.")
    return (intro + f'<div class="tblock wide">{table}{sources_html(S["sources"])}<p class="tnote">{esc(note)}</p></div>'
            + finding)



def track_json(mid, rel):
    """A file of a model's track on its branch (study/results/cross/<mid>/<rel>), parsed; None when absent."""
    br = next((b for m, b, _ in SOURCES if m == mid), None)
    text = git("show", f"origin/{br}:study/results/cross/{mid}/{rel}") if br else None
    try:
        return json.loads(text) if text else None
    except ValueError:
        return None


def edit_records(src, mid, name):
    """An edit's build record (results/edits/<machine>/<name>.json: token counts, positions matched) and its offline
    gate (results/offline/<machine>/<name>_gate.json: pass, gap closed per task, random control), from any machine
    the track's rows name."""
    ms = sorted({r.get("machine") or (r.get("_extra") or {}).get("machine") for r in (src or {}).get("rows", [])} - {None})
    build = next((d for d in (track_json(mid, f"results/edits/{m}/{name}.json") for m in ms) if d), None)
    gate = next((d for d in (track_json(mid, f"results/offline/{m}/{name}_gate.json") for m in ms) if d), None)
    g = None
    if isinstance(gate, dict) and "pass" in gate:
        gaps, refs = [], {}
        for k, v in gate.items():
            m = re.fullmatch(r"([tr][1-8])_(?:(\w+?)_)?gap_closed_pct", k)
            if m and as_float(v) is not None and m.group(1).upper() in TASK_NAME:
                if m.group(2):
                    refs[m.group(1).upper()] = as_float(v)      # the same gap closed by another edit, for comparison
                else:
                    gaps.append((m.group(1).upper(), as_float(v)))
        n = as_int(gate.get("n")) or next((as_int(v) for k, v in gate.items() if re.fullmatch(r"[tr][1-8]_n", k)), None)
        g = dict(passed=bool(gate["pass"]), gaps=gaps, refs=refs,
                 ctl=as_float(gate.get("control_gap_closed_pct")), n=n, rule=gate.get("rule"),
                 path=f"results/offline/{ms[0]}/{name}_gate.json" if ms else None)
    b = None
    if isinstance(build, dict) and as_int(build.get("n_tokens_orig")) is not None:
        b = dict(orig=as_int(build["n_tokens_orig"]), good=as_int(build.get("n_tokens_good")),
                 matched=as_int(build.get("matched")), align=build.get("align"))
    return b, g


def when_html(tables, cross, prevent, second, P=None):
    """'The words have to line up', in short paragraphs: the rule, the failures it explains, π0.5's exception, the
    prediction made before testing a new pair, and a small table of the cases. Details live in the records."""
    T = _cells(tables)
    src = cross.get("cosmos")
    b8, _ = edit_records(src, "cosmos", "t8und")
    b3, g3 = edit_records(src, "cosmos", "t3und")
    c8 = T["edit_gain"].get("T8", {}).get("cosmos")
    gr = dg_groot(tables, cross)
    gd = gr[1] if gr else None
    eg = T["edit_gain"].get("T1", {}).get("groot")
    us = next((r for r in prevent["rows"] if r.get("gate_failed")), None)
    if not (b8 and _ok(c8) and b8["orig"] == b8["good"]):
        return ""
    fails = []
    rows = [(f"{tn('T8')}, world-action model", "Same length, swapped in place",
             f"{c8['a']} → {c8['b']} of {c8['n']} unseen starts")]
    if b3 and g3 and not g3["passed"] and b3["orig"] != b3["good"]:
        rows.append((f"{tn('T3')}, world-action model", "Words added", "Failed the offline check"))
        fails.append(f"the world-action model’s {tn('T3')}")
    if gd and _ok(eg) and eg["b"] < eg["a"]:
        rows.append((f"{tn('T1')}, reasoning-backbone VLA", "Words added",
                     f"Worse: {eg['a']} → {eg['b']} of {eg['n']}{' practice' if eg.get('prac') else ''} starts"))
        fails.append(f"the reasoning-backbone VLA’s {tn('T1')}")
    if us and us["cells"]["gap"].get("x") is not None:
        rows.append((f"{tn('T8')}, only the shared words copied", "—",
                     "No more than a random edit, in replays"))
    WHEN_LAST["rule"] = bool(fails)
    ps = [("The edit copies the clearer wording’s effect onto the matching tokens of the original, position by "
           "position: each token’s input is a key, as in model editing ⟦meng2023memit⟧. On the mustard task the two wordings have the same length and swap words in place, so every token "
           "has a match, and the edit works.")]
    if fails:
        ps.append("When the clearer wording adds words, the new tokens have nothing to land on. That is why the edit "
                  f"failed on {and_list(fails)}." + (" Copying only the words both wordings share does nothing at all."
                                                     if us else ""))
    ps.append("The action-expert VLA is the exception, for a reason: π0.5’s VLM backbone attends over the image tokens "
              "and the whole instruction together, so a clearer wording’s effect also reaches the image tokens, which "
              "both wordings share, and the edit can copy it from there. π0.5 is known to hold spatial information at "
              "its image-token positions ⟦swann2026causal⟧.")
    pred = prospect_main(P)
    if pred:
        ps.append(pred[0])
        rows += pred[1]
    table = ('<div class="tblock wide"><table class="ctable ptable wtable"><thead><tr><th scope="col"><span class="vh">'
             'Case</span></th><th scope="col"><span class="mn">The two wordings</span></th><th scope="col"><span '
             'class="mn">Result</span></th></tr></thead><tbody>'
             + "".join(f'<tr><th scope="row">{esc(a)}</th>{td_html(C(b), "The two wordings")}{td_html(C(c), "Result")}'
                       f"</tr>" for a, b, c in rows) + "</tbody></table></div>")
    return ("".join(f"<p>{esc(p)}</p>" for p in ps) + table
            + "<p>Even where it works, though, the edit is not free.</p>")


def xg_(row, k):
    """A field of a row, from its known keys or the ones kept aside."""
    return row.get(k) if row.get(k) is not None else (row.get("_extra") or {}).get(k)


def prospect(cross, tables):
    """Track A's prospective test of the token-count rule (pre-registered in the Cosmos README, PROSPECT["commit"],
    before any screen or build): per pair, its screen (unedited, original vs clear wording), its build record (token
    counts), its offline gate and any closed-loop runs, by the track's own task codes (S1L, S1R, S1F). None until a
    pair has an offline or closed-loop record, so the page shows nothing before then."""
    src = cross.get(PROSPECT["model"])
    if not src:
        return None
    rows = [r for r in src["rows"] if not r["_prov"]]
    out = []
    for code, e in PROSPECT["pairs"].items():
        mine = [r for r in rows if r.get("_ptask") == code]
        screen = next((r for r in mine if as_int(r.get("teacher_k")) is not None and as_int(r.get("edited_k")) is None),
                      None)
        names = sorted({edit_id(r) for r in mine if edit_id(r)} | {f"{code.lower()}und"})
        build = gate = name = None
        for nm in names:
            b, _ = edit_records(src, PROSPECT["model"], nm)
            g = next((d for d in (track_json(PROSPECT["model"], f"results/offline/{m}/{nm}_gate.json")
                                  for m in sorted({r.get("machine") or r["_extra"].get("machine") for r in rows} - {None}))
                      if d), None)
            if b or g:
                build, gate, name = b, g, nm
                break
        gap = ctl = None
        if isinstance(gate, dict):
            for k, v in gate.items():
                if re.fullmatch(r"(?:[a-z0-9]+_)?gap_closed_pct", k) and not k.startswith("control") and \
                        as_float(v) is not None and gap is None:
                    gap = as_float(v)
            ctl = as_float(gate.get("control_gap_closed_pct"))
        for r in mine:   # an offline row in the records, if the gate file is not there
            o = offline_of(r)
            if gap is None and o and nk(o.get("edit") or "") in {nk(x) for x in names}:
                gap, ctl = as_float(o.get("gap_closed_pct")), as_float(o.get("control_gap_closed_pct"))
        passed = (bool(gate["pass"]) if isinstance(gate, dict) and "pass" in gate else
                  None if gap is None else gap >= PROSPECT["gate"] and (ctl is None or abs(ctl) <= 15))
        closed = {}
        for r in mine:
            i = ints(r)
            if None in (i["edited_k"], i["unedited_k"], i["n"]):
                continue
            kind = "practice" if is_practice(r) else "unseen"
            if kind not in closed:
                _, p, _, _, _, _ = stat(r)
                tn_ = as_int(r.get("teacher_n") if r.get("teacher_n") is not None else r["_extra"].get("teacher_n"))
                cn_ = as_int(r.get("control_n") if r.get("control_n") is not None else r["_extra"].get("control_n"))
                closed[kind] = dict(a=i["unedited_k"], b=i["edited_k"], n=i["n"], p=p,
                                    ctl=i["control_k"] if cn_ in (None, i["n"]) else None,
                                    t=i["teacher_k"] if tn_ in (None, i["n"]) else None,
                                    t_part=(i["teacher_k"], tn_) if tn_ and tn_ < i["n"] else None,
                                    p_cmp=as_float((xg_(r, "edit_vs_teacher") or {}).get("p")),
                                    p_t=as_float(screen.get("p")) if screen and kind == "practice"
                                    and as_int(screen.get("n")) == i["n"] else None)
        if gap is None and not closed:
            continue
        sc = ints(screen) if screen else None
        out.append(dict(code=code, name=name, plain=e["plain"], equal=e["equal"], build=build, gap=gap, ctl=ctl,
                        passed=passed, closed=closed, wording=screen.get("wording_pair") if screen else None,
                        screen=dict(a=sc["unedited_k"], b=sc["teacher_k"], n=sc["n"], p=as_float(screen.get("p")))
                        if sc else None))
    return out or None


def prospect_html(P):
    """'A prediction we tested': the prediction as written before any build, then each pair's result from its records
    (offline check, then practice and unseen starts against the random edit), and whether the prediction held; plus the
    rows for the table of "When does the edit work?"."""
    if not P:
        return "", []
    s = [f"Before building anything, we wrote down a prediction (the Cosmos3 Edge track’s "
         f"README, commit {PROSPECT['commit']}): an edit for a new pair of wordings of equal length would pass the "
         "offline check and fix the task; one for wordings of unequal length would fail the check."]
    rows = []
    for x in P:
        b = x["build"]
        toks = f"{b['orig']} and {b['good']} tokens" if b and b.get("good") is not None else None
        sc = x["screen"]
        modest = (f"unedited, the clearer wording lifts it only from {sc['a']} to {sc['b']} of {sc['n']} practice starts"
                  + (f" (p = {fp(sc['p'])})" if sc.get("p") is not None else "")) if sc else None
        res = []
        if x["gap"] is not None:
            res.append((f"in replays the edit closed {sgn(x['gap'], False, 0)}% of the gap to the clearer wording"
                        if x["gap"] >= 0 else f"in replays the edit moved the planned motion away from the clearer "
                                              f"wording’s ({sgn(x['gap'], True, 0)}% of the gap)")
                       + (f" (a random edit of the same size: {sgn(x['ctl'], True, 1)}%)" if x["ctl"] is not None else "")
                       + (", passing the offline check" if x["passed"] else ", failing the offline check"
                          if x["passed"] is False else ""))
        for kind in ("practice", "unseen"):
            c = x["closed"].get(kind)
            if c:
                part = kind == "unseen" and c["n"] < PROSPECT["heldout_n"]
                res.append((f"on the first {c['n']} of {PROSPECT['heldout_n']} unseen starts" if part else
                            f"on {c['n']} {kind} starts") + f" it went from {c['a']} to {c['b']}"
                           + (f", against {c['ctl']} for the random edit" if c.get("ctl") is not None else "")
                           + (f" and {c['t']} for the clearer wording" if c.get("t") is not None else "")
                           + (f" (p = {fp(c['p'])})" if c.get("p") is not None else "")
                           + (f"; the clearer wording, run on the first {c['t_part'][1]} of them, {c['t_part'][0]}"
                              if c.get("t_part") else "")
                           + (f"; {edit_vs_clear(c['b'], c['t'], c['n'], c.get('p_cmp'))}"
                              if kind == "unseen" and c.get("t") is not None else ""))
        # the verdict follows the README's rules: the offline check first; then, on practice starts, the bar for the
        # unseen-start runs (at least half of the clearer wording's paired gain, and more than the random edit); then
        # the fix itself on unseen starts (exact McNemar p < 0.05, and more than the random edit)
        cp, cu = x["closed"].get("practice"), x["closed"].get("unseen")
        beats = lambda c: c.get("ctl") is None or c["b"] > c["ctl"]
        fixed = lambda c: c.get("p") is not None and c["p"] < SIG and c["b"] > c["a"] and beats(c)
        rec = (lambda c: (c["b"] - c["a"]) / (c["t"] - c["a"]) if c.get("t") is not None and c["t"] > c["a"] else None)
        # a recovery % only where the clearer wording's own gain is clear (as in the tables); else the bar in words
        show = lambda c: (teacher_gain_p(c["a"], c["t"], c["n"], c.get("p_t")) or 1) < SIG
        bar = lambda c: rec(c) is not None and rec(c) >= 0.5 and beats(c)
        verdict = ""
        if not x["equal"]:
            held = None if x["passed"] is None else (x["passed"] is False)
        elif x["passed"] is False:
            held = False
        elif cu and cu["n"] < PROSPECT["heldout_n"]:
            held = None
            verdict = (f"The verdict waits for all {PROSPECT['heldout_n']} unseen starts; the other "
                       f"{PROSPECT['heldout_n'] - cu['n']} are next.")
        elif cu:
            held = fixed(cu)
        elif cp and not bar(cp):
            held = False
            verdict = (f"On practice starts it recovered only {100 * rec(cp):.0f}% of the clearer wording’s gain"
                       if rec(cp) is not None and show(cp) else "On practice starts it fell short of the bar"
                       if rec(cp) is not None else "On practice starts it did no better than the random edit") + \
                      ", short of the bar for the unseen-start runs. "
        elif cp:
            held = None
            verdict = ((f"On practice starts it recovered {100 * rec(cp):.0f}% of the clearer wording’s gain and beat the "
                        "random edit" if show(cp) else "On practice starts it met the bar")
                       + ", the bar set beforehand for running it on unseen starts (at least half of the clearer wording’s "
                       "gain, and more than the random edit); whether it fixes the task is decided there, and those "
                       "runs are next.")
        else:
            held = None
            verdict = "So far the prediction holds; the runs on the robot are next." if x["passed"] else ""
        verdict += {True: "The prediction held.", False: "The prediction did not hold.", None: ""}[held]
        WHEN_LAST.setdefault("pred", []).append(dict(equal=x["equal"], held=held, unseen=cu))
        s.append(f"{x['plain']}" + (f", {toks}" if toks else "") + (f"; {modest}, a modest gap" if modest else "")
                 + ": " + "; ".join(res) + ". " + verdict)
        rows.append((x["plain"], C(("Yes: " if x["equal"] else "No: ") + (toks or ("equal length" if x["equal"]
                                                                                   else "unequal length")),
                                   "predicted before the build: " + ("it works" if x["equal"] else "it fails")),
                     C("; ".join(res)[:1].upper() + "; ".join(res)[1:] if res else "—",
                       (f"{modest[0].upper()}{modest[1:]}. " if modest else "") + verdict or None)))
    return (f"<p><b>A prediction we tested.</b> {esc(' '.join(t.strip() for t in s if t.strip()))}</p>", rows)


def prevent_main(P):
    """Keeping other instructions fixed: what each rebuild did, in two short paragraphs, and a small table of the
    measured cells; then the bridge to the second edit."""
    base = next((r for r in P["rows"] if r["name"] == P["base"]), None)
    tasks = (P["target"], *P["left"])
    ran = [r for r in P["rows"] if r is not base and any(r["cells"][t]["s"] == "ok" for t in tasks)]
    gated = [r for r in P["rows"] if r.get("gate_failed")]
    if not (ran or gated):
        return ""
    t5 = P["left"][0]
    sig = lambda c: c["s"] == "ok" and c.get("p") is not None and c["p"] < SIG
    kept = all(sig(r["cells"][P["target"]]) and r["cells"][P["target"]]["b"] > r["cells"][P["target"]]["a"] for r in ran)
    broke = all(r["cells"][t5]["s"] == "ok" and r["cells"][t5]["b"] < r["cells"][t5]["a"] for r in ran)
    keys = [pe(r["name"]).get("keys") for r in ran if pe(r["name"]).get("keys")]
    ps = []
    if ran:
        ps.append(f"We rebuilt it {'twice' if len(ran) == 2 else f'{len(ran)} times'}, telling it to leave other "
                  f"instructions unchanged (preservation keys: zero-change constraints in the solve ⟦meng2023memit⟧)" + (f": {and_list(keys).replace(' and ', ', then ', 1) if len(keys) == 2 else and_list(keys)}" if keys else "")
                  + ". " + (f"{'Both' if len(ran) == 2 else 'All'} kept the fix" if kept else "The fix did not always hold")
                  + (f", and {'both' if len(ran) == 2 else 'all'} still broke {tn(t5)}." if broke else "."))
    if gated:
        ps.append("A third rebuild copied only the words both wordings share. In replays it barely touched the “left” "
                  "tasks, but it lost the fix too, doing no more than a random edit. The fix and the damage are the same "
                  "change. Methods that enforce preservation exactly, by projecting the update away from the preserved "
                  "keys ⟦fang2025alphaedit⟧, are a different approach that we did not test.")
    rows = ([base] if base else []) + ran
    cols = [(k, a, b) for k, a, b in PCOLS if k in (P["target"], t5) and all(r["cells"][k]["s"] == "ok" for r in rows)]
    head = "".join(f'<th scope="col"><span class="mn">{esc(a)}</span><span class="mk">{esc(b)}</span></th>'
                   for _, a, b in cols)
    body = "".join(f'<tr><th scope="row">{esc(r["plain"])}</th>'
                   + "".join(td_html({x: v for x, v in r["cells"][k].items() if x != "d"}, f"{a}, {b}")
                             for k, a, b in cols) + "</tr>" for r in rows)
    table = (f'<div class="tblock wide"><table class="ctable ptable ptable-s"><thead><tr><th scope="col"><span '
             f'class="vh">Edit</span></th>{head}</tr></thead><tbody>{body}</tbody></table></div>') if len(rows) > 1 else ""
    return (f"<p>{esc(ps[0])}</p>" if ps else "") + table + "".join(f"<p>{esc(x)}</p>" for x in ps[1:]) \
        + "<p>What about giving the damaged task an edit of its own?</p>"


def second_main(S):
    """A second edit for Cube left of bowl: why it was tried, and that it failed the offline check for the reason the
    rule predicts; then the bridge to GR00T N1.7."""
    w, t3 = S.get("wording"), tn(SECOND["second"])
    rows = [r for r in S["rows"] if r["name"] != SECOND["edits"][0]["name"]]
    failed = [r for r in rows if r.get("gate_failed")]
    s = [f"The world-action model’s grounding also fails on {t3}: a more specific wording lifts it "
         + (f"from {w['a']} to {w['b']} of {w['n']} practice starts." if w else "clearly.")]
    if failed and len(failed) == len(rows):
        b = next((r.get("build") for r in failed if r.get("build")), None)
        added = b and b["good"] > b["orig"]
        alone = next((r for r in failed if not stacked(r["name"])), None)
        stack = next((r for r in failed if stacked(r["name"])), None)
        s.append(("But that wording adds words, and " if added else "But ")
                 + ("the cube-left edit failed its offline check, which is one of the cases that showed us the rule."
                    if added and alone else "the cube-left edit failed its offline check." if alone else "")
                 + (" Stacked on top of the mustard edit, it failed the check too, so neither ran on the robot."
                    if stack and alone else " Neither version ran on the robot."))
    else:
        return second_html(S, full=False) + "<p>The reasoning-backbone VLA is a different story.</p>"
    return f"<p>{esc(' '.join(s))}</p><p>The reasoning-backbone VLA is a different story.</p>"


def limit_pi05(tables):
    """The Limits item on π0.5's earlier records: exactly which of the main flow's π0.5 numbers still come from them
    (its where-to-edit comparison, figures 4 and 5, unless fresh runs replace it); nothing once none do."""
    T = _cells(tables)
    early = []
    if any(c.get("prov") for c in (T["where"].get(k, {}).get("pi05") or {} for k in ("backbone", "action", "both"))
           if c.get("s") == "ok"):
        early.append("its where-to-edit comparison (figures 4 and 5)")
    for tid, what in (("T1", f"its {tn('T1')} edit"), ("T8", f"its {tn('T8')} edit")):
        c = T["edit_gain"].get(tid, {}).get("pi05") or {}
        if c.get("s") == "ok" and c.get("prov"):
            early.append(what)
    if not early:
        return ""
    return (f"<li><strong>π0.5 from earlier records:</strong> {esc(and_list(early))} "
            f"{'comes' if len(early) == 1 else 'come'} from earlier runs, on other starts and rendering settings; every "
            "other π0.5 number here is from this page’s starts.</li>")


def problem_html(tables, cross):
    """The hook: one goal in two wordings, and how often each model completes it with each, from the same starts."""
    T = _cells(tables)
    rows = (cross.get("cosmos") or {}).get("rows") or []
    wp = next((r.get("wording_pair") for r in rows if r["_task"] == "T8" and isinstance(r.get("wording_pair"), list)
               and len(r["wording_pair"]) == 2 and all(r["wording_pair"])), None)
    per = [(m, T["wording"].get("T8", {}).get(m)) for m in ("cosmos", "pi05", "groot")]
    per = [(m, w) for m, w in per if _ok(w)]
    if not wp or not per:
        return ""
    works = [m for m, w in per if w["b"] > w["a"] and (w.get("pv") is None or w["pv"] < SIG)]
    neither = [m for m, w in per if w["b"] <= w["a"] and w["b"] == 0]
    block = (f'<div class="wordings"><p><span class="wl">Fails</span> <span class="wt">“{esc(wp[0])}”</span></p>'
             f'<p><span class="wl wl-ok">Works</span> <span class="wt">“{esc(wp[1])}”</span></p></div>')
    def prac(w):
        m_ = re.search(r"slots (\d+)", w.get("d") or "")
        return bool(m_ and int(m_.group(1)) < 90000)
    body = "".join(f'<tr><th scope="row">{type_th(m)}</th>'
                   + td_html(C(f"{w['a']} of {w['n']}"), "Raisin box first")
                   + td_html(C(f"{w['b']} of {w['n']}", prac=prac(w) or None), "Mustard first") + "</tr>" for m, w in per)
    table = ('<div class="tblock"><table class="ctable ptable wtable2"><thead><tr><th scope="col"><span class="vh">'
             'Model</span></th><th scope="col"><span class="mn">Raisin box first</span><span class="mk">starts '
             'completed</span></th><th scope="col"><span class="mn">Mustard first</span><span class="mk">starts '
             f'completed</span></th></tr></thead><tbody>{body}</tbody></table></div>')
    p1 = ("<p>Some robot failures come from weak language grounding ⟦shridhar2021cliport,ahn2022saycan⟧: the policy "
          "has the skill but grounds the instruction to the wrong object or relation ⟦fang2026counterfactual⟧. Robot "
          "policies are known to be sensitive to how an instruction is worded "
          "⟦fei2026liberoplus,kim2026liberopara,abbas2026samegoal⟧. Here is one task from RoboLab, a simulated "
          "benchmark ⟦yang2026robolab⟧, as two wordings of the same goal:</p>")
    p2 = ("<p>Same goal, same starting scenes, same weights; only the order of the words changes: the two wordings "
          "state one spatial relation from either object’s side. "
          + (f"So {and_list('the ' + ARCH[m]['short'] for m in works)} {'have' if len(works) > 1 else 'has'} the "
             f"skill; what fails is how {'they ground' if len(works) > 1 else 'it grounds'} the instruction."
             if works else "")
          + (f" {and_list('the ' + ARCH[m]['short'] for m in neither)[0].upper()}"
             f"{and_list('the ' + ARCH[m]['short'] for m in neither)[1:]} fails both ways, a different problem we come "
             "back to at the end." if neither else "") + "</p>")
    p3 = ("<p>That raises a simple question. If a clearer wording works, could the original wording be made to work "
          "too, without changing a word?</p>")
    return p1 + block + table + p2 + "<!-- clipslot -->" + p3


def pi05_story(P, tables):
    """Where it started: π0.5, this page's own starts where they exist, else the paper's; one short paragraph."""
    T = _cells(tables)
    e = T["edit_gain"].get("T1", {}).get("pi05") or {}
    r = T["random_control"].get("T1", {}).get("pi05") or {}
    e8 = T["edit_gain"].get("T8", {}).get("pi05") or {}
    if not _ok(e):
        return ""
    kind = "practice" if e.get("prac") else "unseen"
    s = (f"We first built this on a VLA with a separate action expert: π0.5 ⟦pi2025pi05,black2024pi0⟧, an open robot "
         f"policy (<a href=\"../\">the weightsmith post</a> ⟦abbas2026distilling⟧). On {esc(tn('T1'))}, where its "
         f"grounding fails the same way, the edit lifts it from "
         f"{e['a']} to {e['b']} of {e['n']} "
         f"{kind} starts: scenes never used to build or choose the edit")
    if r.get("s") == "ok" and r.get("r") is not None and r.get("n") == e["n"]:
        s += f". A random edit of the same size stays at {r['r']}"
    s += "."
    if _ok(e8) and e8["b"] > e8["a"]:
        s += " Its own edit for the mustard task works too."
    s += (" In concurrent work, an LLM editor repairs π0.5’s grounding of instruction words by editing token-embedding "
          "rows, with matched random-direction edits as the control ⟦zhang2026debugrobots⟧.")
    return (f"<p>{s}</p><p>But that is one architecture. Would the same edit work on policies built very "
            "differently?</p>")


def type_th(mid):
    """A table row head: the architecture type, and the model tested as its instance."""
    return f'{esc(ARCH[mid]["type"])}<span class="inst">{esc(FIGNAME[mid])}</span>'


def glance_html(tables, sites):
    """All three models side by side: the fix on each model's own task (unedited → edited, random edit, clearer
    wording), where the edit works, and its side effects; the next sections take each row in turn."""
    T = _cells(tables)
    byid = {m["id"]: m for m in sites}
    own = {"cosmos": ["T8"], "groot": ["T1"], "pi05": ["T1", "T8"]}
    sig = lambda c: c.get("p") is not None and c["p"] < SIG

    def fix(mid):
        out = []
        for tid in own[mid]:
            e = T["edit_gain"].get(tid, {}).get(mid) or {}
            if not _ok(e):
                continue
            kind = " practice" if e.get("prac") else ""
            bits = [f"{tn(tid)}: {e['a']} → {e['b']} of {e['n']}{kind}"]
            same = lambda c_: c_.get("n") == e["n"] and bool(c_.get("prov")) == bool(e.get("prov"))
            r = T["random_control"].get(tid, {}).get(mid) or {}
            if r.get("s") == "ok" and r.get("r") is not None and same(r):
                bits.append(f"random edit {r['r']}")
            rec, w = T["recovery"].get(tid, {}).get(mid) or {}, T["wording"].get(tid, {}).get(mid) or {}
            if rec.get("s") == "ok" and rec.get("t") is not None and same(rec):
                bits.append(f"clearer wording {rec['t']}")
            elif _ok(w) and same(w) and w["a"] == e["a"]:
                bits.append(f"clearer wording {w['b']}")
            out.append("; ".join(bits))
        return " · ".join(out) or "—"

    def where(mid):
        m = byid.get(mid)
        if not m:
            return "—"
        bits, first = [], None
        for blk in m["blocks"]:
            if not blk["sites"]:
                continue
            s = blk["sites"][0]
            part = ARCH[mid]["parts"][blk["let"]][0]
            if first is None:
                first = (part, s)
            else:
                bits.append(f"{part} {s['k']} of {s['n']}")
        tail = " practice starts" if m.get("kind") == "practice" else ""
        if all(s["oc"] == "broke" for b in m["blocks"] for s in b["sites"]):
            return ("Nowhere: every placement made it worse (" + ", ".join([f"{first[0]} {first[1]['k']} of {first[1]['n']}"]
                                                                           + bits) + tail + ")")
        part, s = first
        only = all(x["oc"] in ("none", "broke") for b in m["blocks"][1:] for x in b["sites"][:1])
        return (f"{'Only' if only else 'Best'} in the {part} ({s['k']} of {s['n']}{tail if not bits else ''})"
                + ("; " + ", ".join(bits) + tail if bits else ""))

    def side(mid):
        t = next((t for t in tables if t["id"] == "side_effects"), None)
        if not t or mid not in se_models(t):
            return "Nothing to map: no edit passed its own task"
        cells = [(r["key"], r["cells"][mid]) for r in t["rows"] if r["cells"][mid].get("s") == "ok"
                 and "delta" in r["cells"][mid] and not r["cells"][mid].get("target")]
        down = [f"{tn(k)} {c['a']} → {c['b']}" for k, c in cells if sig(c) and c["delta"] < 0]
        up = [f"{tn(k)} {c['a']} → {c['b']}" for k, c in cells if sig(c) and c["delta"] > 0]
        return ("; ".join(x for x in ((f"breaks {and_list(down)}" if down else ""), (f"lifts {and_list(up)}" if up else ""))
                          if x) + "; no clear change elsewhere") if (down or up) else "No clear change on other tasks"
    rows = [("The fix", fix), ("Where it works", where), ("Side effects", side)]
    head = "".join(f'<th scope="col"><span class="mn">{esc(ARCH[m]["type"])}</span><span class="mk">{esc(FIGNAME[m])}'
                   f'</span></th>' for m in ORDER)
    body = "".join(f'<tr><th scope="row">{esc(lab)}</th>'
                   + "".join(td_html(C(fn(m)[0].upper() + fn(m)[1:]), FIGNAME[m]) for m in ORDER) + "</tr>"
                   for lab, fn in rows)
    return (f'<div class="tblock wide"><table class="ctable gtable"><thead><tr><th scope="col"><span class="vh">Row'
            f'</span></th>{head}</tr></thead><tbody>{body}</tbody></table></div>'
            "<p>Two things decide whether the edit works: where in the architecture it goes, and how the two wordings "
            "line up.</p>")


def prospect_main(P):
    """The prediction in two or three sentences, and its rows for the cases table; the verdicts follow the README's
    rules (prospect_html), the numbers stay in the table and the records."""
    if not P:
        return None
    WHEN_LAST.pop("pred", None)
    prospect_html(P)   # computes each pair's verdict into WHEN_LAST["pred"], the README's rules in order
    pred = WHEN_LAST.get("pred") or []
    eq = next(((x, p) for x, p in zip(P, pred) if x["equal"]), None)
    neg = next(((x, p) for x, p in zip(P, pred) if not x["equal"]), None)
    rows, s = [], ["We wrote this rule down before testing a new pair of wordings."]
    if eq:
        x, p = eq
        cu = x["closed"].get("unseen")
        if cu and p["held"] is True:
            s.append(f"On a pair of equal length, {x['plain'].replace(', worded', ' worded')}, the edit lifted the task "
                     f"from {cu['a']} to {cu['b']} of {cu['n']} unseen starts, far above a random edit.")
        elif p["held"] is False:
            s.append(f"On a pair of equal length, {x['plain'].replace(', worded', ' worded')}, the edit did not fix "
                     "the task, against the prediction.")
        elif x["passed"]:
            s.append(f"On a pair of equal length, {x['plain'].replace(', worded', ' worded')}, the edit passed its "
                     "offline check.")
        rows.append((f"Predicted: {x['plain']}", "Same length, swapped in place",
                     (f"{cu['a']} → {cu['b']} of {cu['n']} unseen starts" + (f"; random edit {cu['ctl']}"
                                                                           if cu.get("ctl") is not None else ""))
                     if cu else ("Passed the offline check" if x["passed"] else "Failed the offline check")))
    if neg:
        x, p = neg
        s.append("On a pair of unequal length, it failed its offline check" + (", as predicted." if p["held"] else "."))
        rows.append((f"Predicted: {x['plain']}", "Words added",
                     "Failed the offline check" if x["passed"] is False else "Passed the offline check"))
    return " ".join(s), rows





def _st(inner, spec, static=True, cls=""):
    """A drawing part with one state per animation step. cosmos.js reads data-s, e.g. "1=o:1 2=o:1,y:-68,@0.3,d:0.7,pop":
    per step its opacity o, translation x/y, scale s and horizontal scale sx, eased in after @ seconds over d; pop
    springs it in, re re-lights it; a step not listed hides it. static: whether the still drawing (no script, print,
    the state before the script runs) shows it."""
    c = " ".join(x for x in ("st", "" if static else "sh", cls) if x)
    return f'<g class="{c}" data-s="{spec}">{inner}</g>'


def _chunk(x, y, n, w, gap, h, run=None, cls="dg-chunk"):
    """An action chunk: n steps as thin bars whose heights follow a smooth joint trajectory; the first `run` steps (the
    ones executed) darker."""
    out = []
    for i in range(n):
        hh = h * (0.4 + 0.6 * (0.5 + 0.5 * _math.sin(i * 6.2832 * 1.15 / n + 0.7)))
        c = cls + (" dg-chunk-run" if run and i < run else "")
        out.append(f'<rect x="{x + i * (w + gap):.2f}" y="{y + h - hh:.2f}" width="{w:g}" height="{hh:.2f}" '
                   f'rx="{min(w / 2, 1):g}" class="{c}"/>')
    return "".join(out)


def _chunk_w(n, w, gap):
    return n * w + (n - 1) * gap


def _gauge(x, y, w, h, seed=0.0):
    """The joint state, seven joints and the gripper, as a small bar readout; cosmos.js makes the bars tick by scaling
    them (data-g: the scale factors it steps through)."""
    n, pad, gap = 8, 4, 1.5
    bw = (w - 2 * pad - (n - 1) * gap) / n
    H = h - 2 * pad
    out = [f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="4" class="dg-in"/>']
    for i in range(n):
        v = [0.3 + 0.62 * (0.5 + 0.5 * _math.sin(seed + i * 1.93 + k * 1.37)) for k in range(6)]
        g = " ".join(f"{a / v[0]:.2f}" for a in v)
        out.append(f'<rect x="{x + pad + i * (bw + gap):.2f}" y="{y + pad + H * (1 - v[0]):.2f}" width="{bw:.2f}" '
                   f'height="{H * v[0]:.2f}" rx="0.8" class="gb{" gb-g" if i == n - 1 else ""}" data-g="{g}"/>')
    return "".join(out)


def _tok(kind, pts, on, t, dur, cls="", key=None, label=""):
    """A token that moves along pts (drawn at the first point) in the steps listed in `on`, starting t seconds into the
    step; cosmos.js animates only its transform and opacity."""
    x0, y0 = pts[0]
    rel = " ".join(f"{x - x0:g},{y - y0:g}" for x, y in pts[1:])
    if kind == "word":
        shape = f'<rect x="{x0 - 6:g}" y="{y0 - 3.5:g}" width="12" height="7" rx="3.5"/>'
    elif kind == "dot":
        shape = f'<circle cx="{x0:g}" cy="{y0:g}" r="2.8"/>'
    elif kind == "state":
        shape = f'<rect x="{x0 - 3:g}" y="{y0 - 3:g}" width="6" height="6" rx="1.2"/>'
    elif kind == "frame":
        shape = (f'<image href="fig/thumb-{key}.webp" x="{x0 - 9:g}" y="{y0 - 6:g}" width="18" height="12" '
                 f'preserveAspectRatio="xMidYMid slice"/><rect x="{x0 - 9:g}" y="{y0 - 6:g}" width="18" height="12" '
                 f'rx="1.5" class="tk-fr"/>')
    elif kind == "chunk":
        shape = _chunk(x0 - 15, y0 - 5, 8, 3, 1.3, 10, cls="tk-cb")
    elif kind == "video":
        shape = "".join(f'<rect x="{x0 - 9 + 3 * i:g}" y="{y0 - 7 + 3 * i:g}" width="14" height="10" rx="1.5" '
                        f'class="tk-vf"/>' for i in range(3))
    elif kind == "delta":
        shape = (f'<rect x="{x0 - 11:g}" y="{y0 - 8:g}" width="22" height="16" rx="8" class="tk-dl"/>'
                 + _t(x0, y0 + 4, "Δ", "dg-xs dg-b tk-dlt", "middle"))
    elif kind == "ring":
        x1, y1, w1, h1 = label
        shape = f'<rect x="{x1:g}" y="{y1:g}" width="{w1:g}" height="{h1:g}" rx="6" class="tk-ring"/>'
    else:
        raise ValueError(kind)
    return (f'<g class="tk tk-{kind}{(" " + cls) if cls else ""}" data-on="{on}" data-t="{t:.2f}" data-dur="{dur:g}" '
            f'data-p="{rel}">{shape}</g>')


def _chips(words, cx, y, cls, size=11.5, h=20):
    """A row of instruction chips centred on cx; returns the drawing and the centre of every word in it."""
    ws = [_tw(w, size) + 16 for w in words]
    gap = 6
    x = cx - (sum(ws) + gap * (len(ws) - 1)) / 2
    out, centres = [], []
    for w_, s in zip(ws, words):
        out.append(f'<rect x="{x:.1f}" y="{y:g}" width="{w_:.1f}" height="{h:g}" rx="{h / 2:g}" class="{cls}"/>'
                   + _t(round(x + w_ / 2, 1), y + h / 2 + 4, s, "dg-xs", "middle"))
        xx = x + 8
        for word in s.split():
            ww = _tw(word, size)
            centres.append(xx + ww / 2)
            xx += ww + _tw(" ", size)
        x += w_ + gap
    return "".join(out), centres


def _bars(x, y, n, w, gap, h, seed, cls):
    """A row of activation bars standing on y: an internal state pattern."""
    out = []
    for i in range(n):
        v = 0.18 + 0.82 * (0.5 + 0.5 * _math.sin(seed + i * 2.31 + _math.sin(i * 0.9 + seed) * 1.7))
        out.append(f'<rect x="{x + i * (w + gap):.2f}" y="{y - h * v:.2f}" width="{w:g}" height="{h * v:.2f}" rx="1" '
                   f'class="{cls}"/>')
    return "".join(out)


DG1_STEPS = ["Original wording: the VLM / text pathway grounds it to the wrong relation, and the robot fails.",
             "Clear wording: the states inside the VLM / text pathway change, and the robot succeeds.",
             "Offline, the difference between the two sets of states is written into its weights: one least-squares "
             "solve per layer, no robot run.",
             "Same words, edited weights: the original wording now produces the clear wording’s states, and the robot "
             "succeeds."]
DG1_DUR = (4.0, 4.0, 4.2, 4.6)


def anim_controls(steps, label):
    """Play/pause, one button per step and the current step's words (cosmos.js shows them); the steps themselves stay
    in the page as a list, for print, screen readers and readers without the script."""
    dots = "".join(f'<button type="button" class="anim-dot" data-k="{i}" aria-label="Step {i} of {len(steps)}: '
                   f'{esc(s)}">{i}</button>' for i, s in enumerate(steps, 1))
    icon = ('<svg class="anim-ico" viewBox="0 0 12 12" aria-hidden="true"><path class="i-pause" d="M3 2h2v8H3zM7 2h2v8H7z"/>'
            '<path class="i-play" d="M3.5 2v8l6.5-4z"/></svg>')
    return (f'<div class="anim-ctl" role="group" aria-label="{esc(label)}: animation controls">'
            f'<button type="button" class="anim-play" aria-pressed="false">{icon}<span>Pause</span></button>'
            f'<span class="anim-dots">{dots}</span><span class="anim-bar" aria-hidden="true"><i></i></span>'
            f'<p class="anim-label">' + "".join(f'<span data-k="{i}">{i}. {esc(x)}</span>' for i, x in enumerate(steps, 1))
            + '</p></div>'
            + '<ol class="anim-steps">' + "".join(f"<li>{esc(s)}</li>" for s in steps) + "</ol>")


ARCH = {   # each architecture type, its instance, and how it is wired (MODELS above; the tracks' READMEs)
    "pi05": dict(type="VLA with an action expert", short="action-expert VLA",
                 cams=("pi05_ext", "pi05_wrist"), layout="stack",
                 parts={"A": ("VLM backbone", "PaliGemma"), "B": ("action expert", "flow matching"),
                        "C": ("action output projection", "its last linear layer")},
                 links={"A": "keys and values"}, state_to="A", chunk=(15, 15),
                 out="15-step action chunk"),
    "groot": dict(type="VLA on a reasoning backbone", short="reasoning-backbone VLA",
                  cams=("groot_ext", "groot_wrist"), layout="stack",
                  parts={"A": ("VLM backbone", "Cosmos-Reason2"), "B": ("action head", "flow-matching DiT"),
                         "C": ("action head", "")},
                  links={"A": "backbone tokens"}, state_to="B", chunk=(40, 8),
                  out="40-step action chunk, first 8 run"),
    "cosmos": dict(type="World-action model", short="world-action model",
                   cams=("cosmos_l", "cosmos_r", "cosmos_w"), layout="side",
                   parts={"A": ("text pathway", "instruction tokens"), "B": ("video/action pathway",
                                                                            "video and action tokens"),
                          "C": ("action head", "")},
                   links={"A": "keys and values, every layer"}, state_to="B", chunk=(32, 32),
                   out="predicted video + 32-step action chunk"),
}
TYPE = {k: v["type"] for k, v in ARCH.items()}
WHERE_DUR = (5.0, 5.0, 5.4)


def or_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " or " + xs[-1]


def where_steps(sites):
    """The steps of Figure 4: the edit in each component in turn, named by the architectures' own terms."""
    names = lambda let: or_list(dict.fromkeys(ARCH[m["id"]]["parts"][let][0] for m in sites
                                              if any(b["let"] == let and b["sites"] for b in m["blocks"])))
    out = [f"The edit in the {names('A')}."]
    if any(b["let"] == "B" and b["sites"] for m in sites for b in m["blocks"]):
        out.append(f"The edit in the {names('B')}.")
    c = [m for m in sites if any(b["let"] == "C" and b["sites"] for b in m["blocks"])]
    if c:
        out.append(f"The edit in the {ARCH[c[0]['id']]['parts']['C'][0]} ({and_list(ARCH[m['id']]['short'] for m in c)} "
                   "only).")
    return out


def ensure_thumbs():
    """The camera-view thumbnails (fig/thumb-*.webp, 2x their largest drawn size, a few KB each): one frame of our own
    clips, cropped to one view. Made once; delete a file to remake it."""
    from PIL import Image
    frames = {}
    for key, (stem, box) in THUMBS.items():
        dst = FIG / f"thumb-{key}.webp"
        if dst.exists():
            continue
        if stem not in frames:
            tmp = Path(tempfile.mkdtemp()) / "frame.png"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(THUMB_T), "-i", str(VID / f"{stem}.mp4"),
                            "-frames:v", "1", str(tmp)], check=True)
            frames[stem] = Image.open(tmp).convert("RGB")
        frames[stem].crop(box).resize(THUMB_PX, Image.LANCZOS).save(dst, "WEBP", quality=84, method=6)
        print(f"  wrote {rel(dst)} ({dst.stat().st_size // 1024 + 1} KB)")


def _thumb(x, y, w, h, key, ident):
    """A real camera frame from one of our clips (ensure_thumbs), clipped to rounded corners, with a hairline frame."""
    return (f'<clipPath id="{ident}"><rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="3"/></clipPath>'
            f'<image href="fig/thumb-{key}.webp" x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" '
            f'preserveAspectRatio="xMidYMid slice" clip-path="url(#{ident})"/>'
            f'<rect x="{x:g}" y="{y:g}" width="{w:g}" height="{h:g}" rx="3" class="dg-frame"/>')


def load_refs():
    """The verified references (site/scripts/cosmos_refs.json, from the research agent's refs.json): core and
    alternates by key. Only these can be cited; an unknown key stops the build."""
    d = json.loads(REFS_FILE.read_text())
    out = {}
    for e in (d.get("references") or []) + (d.get("alternates") or []):
        if e.get("verified") is not False and e.get("key"):
            out[e["key"]] = e
    return out


def _ref_title(e):
    t = e.get("verified_title") or e.get("title") or ""
    t = REF_TITLE.get(e["key"], t)
    return re.sub(r"\$π_\{?([0-9.]+)\}?\$", r"π\1", t)


def _ref_html(n, e):
    """One entry of the reference list: authors, title, venue and year, and a link (arXiv, DOI or the page)."""
    links = []
    if e.get("arxiv"):
        links.append(f'<a href="https://arxiv.org/abs/{esc(e["arxiv"])}">arXiv:{esc(e["arxiv"])}</a>')
    if e.get("doi"):
        links.append(f'<a href="https://doi.org/{esc(e["doi"])}">doi:{esc(e["doi"])}</a>')
    if not links and e.get("url"):
        links.append(f'<a href="{esc(e["url"])}">{esc(REF_LINK.get(e["key"]) or re.sub(r"^https?://", "", e["url"]).rstrip("/"))}</a>')
    venue = REF_VENUE.get(e["key"], e.get("venue_year") or "")
    end = lambda t: t if t[-1:] in "?!." else t + "."
    return (f'<li id="ref-{n}"><span class="ref-au">{esc(end(e.get("authors", "")))}</span> '
            f'<span class="ref-t">{esc(end(_ref_title(e)))}</span> {esc(end(venue))} ' + " · ".join(links) + "</li>")


def number_citations(page, refs):
    """Citations are written ⟦key,key⟧ in the generator's text and the page; each becomes a numbered link [n] to the
    References section, numbered in order of first appearance in the article. Spans from an earlier build are
    renumbered from their data-ref, so the static text keeps its citations across builds. Returns the page and the
    reference list (only references cited, from the verified file)."""
    page = re.sub(r"\s*⟦([a-z0-9_,\s]+)⟧", lambda m: ' <span class="cite" data-ref="' + ",".join(
        k.strip() for k in m.group(1).split(",")) + '">[]</span>', page)
    a, b = article(page)
    order = []
    for m in re.finditer(r'<span class="cite" data-ref="([^"]+)">', page[a:b]):
        for k in m.group(1).split(","):
            if k not in refs:
                raise SystemExit(f"citation {k!r} is not in {REFS_FILE.name}: only verified references can be cited")
            if k not in order:
                order.append(k)
    num = {k: i + 1 for i, k in enumerate(order)}

    def one(m):
        ks = m.group(1).split(",")
        return (f'<span class="cite" data-ref="{m.group(1)}">['
                + ", ".join(f'<a href="#ref-{num[k]}" title="{esc(_ref_title(refs[k]))}">{num[k]}</a>' for k in ks) + "]</span>")
    page = page[:a] + re.sub(r'<span class="cite" data-ref="([^"]+)">.*?</span>', one, page[a:b]) + page[b:]
    lst = ('<ol class="refs">' + "".join(_ref_html(num[k], refs[k]) for k in order) + "</ol>") if order else ""
    return page, lst


def strip_cites(x):
    """Plain text for metrics.json and the brief: citations dropped."""
    if isinstance(x, str):
        return re.sub(r"\s*⟦[^⟧]*⟧", "", x)
    if isinstance(x, dict):
        return {k: strip_cites(v) for k, v in x.items()}
    if isinstance(x, list):
        return [strip_cites(v) for v in x]
    return x


REFS_FILE = Path(__file__).resolve().parent / "cosmos_refs.json"
REF_TITLE = {   # the title to cite where it differs from the arXiv one
    "fei2026liberoplus": "LIBERO-Plus: A Progressive Robustness Benchmark for Visual-Language-Action Models",
}
REF_VENUE = {   # where the verified record's venue note is not a venue
    "abbas2026distilling": "2026, arXiv preprint forthcoming",
    "kim2026cosmospolicy": "arXiv 2026",
    "zhang2026debugrobots": "arXiv 2026",
}
REF_LINK = {"abbas2026distilling": "project page: ali-adeeb.com/weightsmith"}

VID = OUT / "video"
# Each clip: stem (video/<stem>.mp4, or "src" for a clip the main page already serves), group (where it goes on the
# page), title, a caption builder (reads its numbers from the run records, never typed in), and an optional poster.
# A clip whose MP4 is missing renders as a "running" placeholder that cosmos.js turns into a player once it exists.
X_COSMOS = ("cross-cosmos/v1", "study/results/cross/cosmos/results/runs/ali-ms")
X_GROOT = ("cross-groot/v1", "study/results/cross/groot/results/runs")


def run_row(where, name, slot):
    """One start's row from a cross-model run record: success, steps."""
    branch, d = where
    txt = git("show", f"origin/{branch}:{d}/{name}.json")
    try:
        rows = json.loads(txt)["rows"] if txt else []
    except (ValueError, KeyError):
        rows = []
    r = next((r for r in rows if r.get("slot") == slot), None)
    return (r, f"{branch}:{d}/{name}.json") if r else (None, None)


def outcome(r):
    return (f"succeeded in {r['steps']} steps" if r.get("success") else f"did not succeed in {r['steps']} steps")


def table_cell(tables, tid, label_prefix, mid):
    t = next((t for t in tables if t["id"] == tid), None)
    r = next((r for r in (t["rows"] if t else []) if r["label"].startswith(label_prefix)), None)
    c = r["cells"].get(mid) if r else None
    return c if c and c["s"] == "ok" else None


def cap_wording(where, task, slot, base, clear, mid, tid_label, top, bottom):
    def f(tables):
        a, sa = run_row(where, base, slot)
        b, sb = run_row(where, clear, slot)
        if not (a and b):
            return None, []
        s = (f"Practice start {slot}: the same weights and the same initial scene, {top} on top, {bottom} below. "
             f"Original wording: {outcome(a)}; clearer wording: {outcome(b)}.")
        c = table_cell(tables, "wording", tid_label, mid)
        if c:
            s += (f" Over all {c['n']} practice starts: {c['a']} → {c['b']} successes"
                  + (f" (exact McNemar p = {fp(c['pv'])})." if c.get("pv") is not None else "."))
        return s, [sa, sb]
    return f


def cap_edit_practice(where, slot, base, edited, mid, tid):
    def f(tables):
        a, sa = run_row(where, base, slot)
        b, sb = run_row(where, edited, slot)
        if not (a and b):
            return None, []
        s = (f"Practice start {slot}, one of the starts the edit was built from: the unedited policy on top, the edited "
             f"policy below, both given the original wording. Unedited: {outcome(a)}; edited: {outcome(b)}.")
        c = table_cell(tables, "edit_gain", tid, mid)
        if c and c.get("prac"):
            s += (f" Over all {c['n']} practice starts: {c['a']} → {c['b']} successes"
                  + (f" (exact McNemar p = {fp(c['pv'])})" if c.get("pv") is not None else "")
                  + "; held-out starts are running.")
        return s, [sa, sb]
    return f


def cap_edit_heldout(where, slot, base, edited, mid, tid):
    def f(tables):
        a, sa = run_row(where, base, slot)
        b, sb = run_row(where, edited, slot)
        if not (a and b):
            return None, []
        s = (f"Unseen start {slot}, never used to build the edit: the unedited robot on top, the edited robot "
             f"below, both given the original wording. Unedited: {outcome(a)}; edited: {outcome(b)}.")
        c = table_cell(tables, "edit_gain", tid, mid)
        if c and not c.get("prac"):
            s += (f" Over all {c['n']} unseen starts{' so far' if c['n'] < 64 else ''}: {c['a']} → {c['b']} successes"
                  + (f" (exact McNemar p = {fp(c['pv'])})." if c.get("pv") is not None else "."))
        return s, [sa, sb]
    return f


def cap_panels(where, slot, panels, lead, mid, tid):
    """Several arms on one start, top to bottom, each outcome read from its run record."""
    def f(tables):
        got = [(lab,) + run_row(where, name, slot) for name, lab in panels]
        if any(r is None for _, r, _ in got):
            return None, []
        s = (f"{lead} Practice start {slot}, the same scene in every panel and the original wording throughout. "
             "Top to bottom: " + "; ".join(f"{lab}: {outcome(r)}" for lab, r, _ in got) + ".")
        c = table_cell(tables, "edit_gain", tid, mid)
        if c and c.get("prac") and c["b"] < c["a"]:
            s += (f" Over all {c['n']} practice starts each edit took success from {c['a']} to {c['b']}"
                  + (f" (exact McNemar p = {fp(c['pv'])})." if c.get("pv") is not None else "."))
        return s, [x for _, _, x in got]
    return f


def cap_side(where, slot, base, edited, mid, tid, et):
    """One start of a task the edit was not built for: unedited against edited, outcomes from the run records."""
    def f(tables):
        a, sa = run_row(where, base, slot)
        b, sb = run_row(where, edited, slot)
        if not (a and b):
            return None, []
        s = (f"The {et} edit, unchanged, on a task it was not built for. Practice start {slot}, the same scene in both "
             f"panels and the task’s own instruction: unedited (top) {outcome(a)}; edited (bottom) {outcome(b)}.")
        c = table_cell(tables, "side_effects", tid, mid)
        if c and c.get("n"):
            s += (f" Over all {c['n']} {'practice' if c.get('prac') else 'unseen'} starts: {c['a']} → {c['b']}"
                  + (f" (exact McNemar p = {fp(c['p'])})." if c.get("p") is not None else "."))
        return s, [sa, sb]
    return f


def cap_same_start(slot):
    def f(tables):
        a, sa = run_row(X_COSMOS, "T1-base-s30000", slot)
        b, sb = run_row(X_GROOT, "fm-T1", slot)
        if not (a and b):
            return None, []
        return (f"Practice start {slot} of T1, the same initial scene for both models, each unedited and seen through "
                f"its own cameras: Cosmos3 Edge Policy (top: left shoulder, right shoulder, wrist) {outcome(a)}; "
                f"GR00T N1.7 (bottom: exterior, wrist) {outcome(b)}."), [sa, sb]
    return f


PI05_CLIPS = ("cross-pi05/v1", "study/results/cross/pi05/videos/clips.json")


def cap_pi05(file, lead):
    """A π0.5 track clip: its panels and outcomes from videos/clips.json, which records every panel's verdict."""
    def f(tables):
        txt = git("show", f"origin/{PI05_CLIPS[0]}:{PI05_CLIPS[1]}")
        try:
            c = next((x for x in json.loads(txt) if x.get("file") == file), None) if txt else None
        except ValueError:
            c = None
        if not c:
            return None, []
        def lab(x):
            return x["label"].replace("edited: ", "edited, ").replace("unedited + ", "unedited, with the ")
        panels = "; ".join(f"{lab(x)}: {outcome(x)}" for x in c["panels"])
        hw = ", ".join(x for x in (c.get("hardware"), (c.get("render_preset") or "") + " render preset") if x)
        s = (f"{lead} Unseen start {c['slot']} from the paper’s evaluation, the same scene and noise seeds in every "
             f"panel. Top to bottom: {panels}. Recorded for the paper ({hw}), {c.get('speed', 3):g}× speed.")
        return s, [f"{PI05_CLIPS[0]}:{PI05_CLIPS[1]}"]
    return f


def fixed_cap(text):
    return lambda tables: (text, [])


CLIPS = [
    dict(stem="cosmos_t8_wording", group="problem", title="Cosmos3 Edge Policy, T8: the wording gap",
         cap=cap_wording(X_COSMOS, "T8", 30000, "T8-base-s30000", "T8-wtf-s30000", "cosmos", "T8",
                         "the original, reference-first wording", "the clearer, target-first wording")),
    dict(stem="groot_t1_wording", group="gallery", title="GR00T N1.7, T1: the wording gap",
         cap=cap_wording(X_GROOT, "T1", 30003, "fm-T1", "wd-T1-expl", "groot", "T1",
                         "the original wording", "the explicit two-step wording")),
    dict(stem="groot_t1_edits", group="gallery", title="GR00T N1.7, T1: two edits that make things worse",
         cap=cap_panels(X_GROOT, 30004, [("fm-T1", "unedited"), ("ed-bb-T1", "edited in the Cosmos-Reason2 backbone"),
                                         ("ed-dit-T1", "edited in the flow-matching DiT action head")],
                        "Both edits distil the explicit two-step wording; neither picks up the cube.", "groot", "T1")),
    dict(stem="t1_same_start", group="gallery", title="T1, one start, the reasoning-backbone VLA and the world-action "
                                                       "model", cap=cap_same_start(30000)),
    dict(stem="pi05_t1_edit_control_teacher", group="gallery", title="π0.5, T1: the edit against its controls",
         cap=cap_pi05("t1_v2mi-f_vs_control_vs_teacher_90000.mp4",
                      "The unedited policy, a random edit of the same size, the distilled edit and the clearer wording.")),
    dict(stem="pi05_t8_edit_control", group="gallery", title="π0.5, T8: the edit against a random edit",
         cap=cap_pi05("t8_s4r-grp3_vs_control_92001.mp4",
                      "The reference-first wording throughout; the edit distils the target-first wording.")),
    dict(stem="cosmos_before_after", group="fix", title="Cosmos3 Edge Policy, T8",
         cap=cap_edit_heldout(X_COSMOS, 96003, "T8-base-s96000", "T8-Et8und-s96000", "cosmos", "T8")),
    dict(stem="groot_before_after", group="gallery", title="GR00T N1.7", mid="groot",
         cap=fixed_cap("Unedited and edited, on an unseen start where the edit helped.")),
    dict(stem="cosmos_t5_side_effect", group="gallery",
         title="Cosmos3 Edge Policy, T5 Mustard into the left bin: a start the T8 edit breaks",
         cap=cap_side(X_COSMOS, 30001, "T5-base-s30000", "T5-Et8und-s30000", "cosmos", "T5", "T8")),
    dict(stem="pi05_t2_fixed", group="gallery", title="π0.5, T2 Pick up the eyeglasses: a start the T1 edit fixes",
         cap=cap_pi05("t2_v2mi-f_side_effect_fixed_90000.mp4", "The T1 edit, unchanged, on a task it was not built for.")),
    dict(stem="pi05_t3_fixed", group="gallery", title="π0.5, T3 Cube left of the bowl: a start the T1 edit fixes",
         cap=cap_pi05("t3_v2mi-f_side_effect_fixed_90000.mp4", "The T1 edit, unchanged, on a task it was not built for.")),
    dict(stem="pi05_t2_broke", group="gallery", title="π0.5, T2 Pick up the eyeglasses: a start the T1 edit breaks",
         cap=cap_pi05("t2_v2mi-f_side_effect_broke_90001.mp4",
                      "Transfer is not free: the same edit also breaks some T2 starts.")),
]
GIF_ORDER = ["cosmos_t8_wording", "groot_t1_wording", "t1_same_start", "cosmos_before_after", "groot_before_after"]


def ff(*args):
    exe = shutil.which("ffmpeg")
    if not exe:
        return False
    return subprocess.run([exe, "-v", "error", "-y", *args], capture_output=True).returncode == 0


def probe_size(path):
    exe = shutil.which("ffprobe")
    if not exe:
        return None
    r = subprocess.run([exe, "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of",
                        "csv=p=0", str(path)], capture_output=True, text=True)
    try:
        w, h = (int(x) for x in r.stdout.strip().split(",")[:2])
        return w, h
    except ValueError:
        return None


def prepare_clips():
    """Which clips exist; first-frame posters for those without one; the brief's GIF (first frame prints as a still)."""
    info = {}
    for c in CLIPS:
        stem = c["stem"]
        if c.get("src"):
            p = (OUT / c["src"]).resolve()
            info[stem] = dict(exists=p.exists(), src=c["src"], poster=c.get("poster"),
                              size=c.get("size") or (probe_size(p) if p.exists() else None))
            continue
        mp4, jpg = VID / f"{stem}.mp4", VID / f"{stem}.jpg"
        ok = mp4.exists() and mp4.stat().st_size > 0
        if ok and mp4.stat().st_size > 2_000_000:
            print(f"  WARNING: video/{stem}.mp4 is {mp4.stat().st_size / 1e6:.1f} MB; keep clips at 2 MB or less")
        if ok and not jpg.exists():
            ff("-i", str(mp4), "-frames:v", "1", "-q:v", "3", str(jpg))
        info[stem] = dict(exists=ok, src=f"video/{stem}.mp4", poster=f"video/{stem}.jpg" if jpg.exists() else None,
                          size=probe_size(mp4) if ok else None)
    gif, src_note = VID / "brief.gif", VID / "brief.gif.src"
    best = next((s for s in GIF_ORDER if info[s]["exists"]), None)
    if best:
        mp4 = VID / f"{best}.mp4"
        stale = (not gif.exists() or not src_note.exists() or src_note.read_text().strip() != best
                 or gif.stat().st_mtime < mp4.stat().st_mtime)
        if stale and ff("-t", "4", "-i", str(mp4), "-vf", "fps=6,scale=360:-2:flags=lanczos,split[a][b];"
                        "[a]palettegen=max_colors=64[p];[b][p]paletteuse=dither=bayer:bayer_scale=3", "-loop", "0", str(gif)):
            src_note.write_text(best + "\n")
            CHANGED.add(gif.name)
            print(f"  brief GIF from {best}.mp4")
    info["_gif"] = best if gif.exists() else None
    return info


def clip_html(c, info, tables, compact=False):
    i = info[c["stem"]]
    caption, srcs = c["cap"](tables)
    if compact and caption:
        caption, srcs = caption.split(" Over all")[0], []
    title = c["title"]
    rec = ("" if not srcs else " Records: " + ", ".join(
        f'<a href="{esc(gh(s))}"><code>{esc(s.split("/")[-1])}</code></a>' for s in srcs) + ".")
    if i["exists"] and caption:
        w, h = i["size"] or (960, 540)
        poster = f' poster="{esc(i["poster"])}"' if i["poster"] else ""
        cls = "clip tall" if h > w else "clip"
        return (f'<figure class="{cls}"><video controls muted loop playsinline preload="metadata"{poster} width="{w}" '
                f'height="{h}" aria-label="{esc(title + ". " + caption)}"><source src="{esc(i["src"])}" type="video/mp4">'
                f'</video><figcaption><b>{esc(title)}.</b> {esc(caption)}{rec}</figcaption></figure>')
    caption = caption or ""
    return (f'<figure class="clip clip-pending" data-clip="{esc(i["src"])}" data-title="{esc(title)}" '
            f'data-caption="{esc(caption)}"><div class="ph" role="img" aria-label="{esc(title)}: running">'
            f'<span class="ph-t">running</span></div>'
            f'<figcaption><b>{esc(title)}.</b> {esc(caption)}</figcaption></figure>')


def clips_section(group, info, tables, models=None):
    """A finished track's placeholder is dropped: its clip will not come. The fix clip's caption is short; its full
    caption and records lead the gallery."""
    if group == "gallery":
        full = "".join(f'<p class="note"><b>{esc(c["title"])}</b> (above): '
                       + re.sub(r"^.*?</b> ", "", clip_html(c, info, tables).split("<figcaption>")[1]).split("</figcaption>")[0]
                       + "</p>" for c in CLIPS if c["group"] == "fix" and info[c["stem"]]["exists"])
        return full + "".join(clip_html(c, info, tables) for c in CLIPS if c["group"] == group
                              and not (c.get("mid") and models and models[c["mid"]]["status"] == "final"
                                       and not info[c["stem"]]["exists"]))
    return "".join(clip_html(c, info, tables, compact=group in ("fix", "problem")) for c in CLIPS if c["group"] == group
                   and not (c.get("mid") and models and models[c["mid"]]["status"] == "final"
                            and not info[c["stem"]]["exists"]))


def gif_html(info):
    if not info.get("_gif"):
        return ""
    title = next(c["title"] for c in CLIPS if c["stem"] == info["_gif"])
    return (f'<figure class="gif"><img src="video/brief.gif" alt="{esc(title)}: first frame of the clip">'
            f'<figcaption>{esc(title)}; the clip plays on the live page.</figcaption></figure>')


def figure_html(name, alt, caption, pdf=False):
    if pdf:
        return (f'<figure class="bfig"><img src="fig/brief-{name}.svg" alt="{esc(alt)}">'
                f'<figcaption>{inline(caption)}</figcaption></figure>')
    return (f'<figure class="cfig"><a class="fig-link" href="fig/{name}.svg" title="Open the figure at full size">'
            f'<picture><source media="(prefers-color-scheme: dark)" srcset="fig/{name}-dark.svg">'
            f'<img src="fig/{name}.svg" alt="{esc(alt)}" loading="lazy" decoding="async"></picture></a>'
            f'<figcaption>{inline(caption)} <a href="fig/{name}.png">PNG</a></figcaption></figure>')


STATUS = {"measured": "the paper’s evaluation", "running": "running", "partial": "first results", "final": "complete"}


def models_html(models):
    out = []
    for mid in ORDER:
        m = models[mid]
        out.append(
            f'<section class="model" aria-labelledby="m-{mid}"><h4 id="m-{mid}">{esc(m["name"])}</h4>'
            f'<p class="mkind">{esc(m["kind"])} · <span class="mst mst-{esc(m["status"])}">{esc(STATUS[m["status"]])}</span></p>'
            f'<dl><dt>Architecture</dt><dd>{inline(m["architecture"])}</dd>'
            f'<dt>Where the instruction enters</dt><dd>{inline(m["entry"])}</dd>'
            f'<dt>Where an edit can go</dt><dd>{inline(m["where"])}</dd>'
            f'<dt>Checkpoint</dt><dd>{inline(m["checkpoint"])}</dd>'
            f'<dt>How it is scored</dt><dd>{inline(m["evaluation"])}</dd></dl></section>')
    return "".join(out)


def render_note(tables):
    c = next(t for t in tables if t["id"] == "render")["rows"][0]["cells"].get("groot")
    if not c or c["s"] != "ok":
        return ""
    return (f'<p class="finding"><strong>A first cross-model finding.</strong> GR00T N1.7 depends on how the simulator '
            f'renders: on the same {c["n"]} starts of {esc(c["task"].lower())} it succeeded {c["a"]} times under RoboLab’s '
            f'performance preset, which renders without the RTX denoiser, and {c["b"]} times under the default preset, '
            f'NVIDIA’s validated configuration. All three models are therefore compared under the default preset; '
            f'the paper’s evaluation of π0.5 used the performance preset. <a href="#render">Rendering sensitivity</a></p>')


def brief_lines(P, rc, models, cross, tables=None):
    """The brief's three lines, in the page's plain words (task names, not codes)."""
    e, c, s4ho = P["edit"], P["ctl"], P["s4ho"]
    lines = [
        "<li><b>Method.</b> When a clearer wording of the same goal makes a robot succeed, copy that wording’s internal "
        "effect into the weights: one least-squares calculation per layer, no training, and the robot still gets the "
        "original instruction.</li>",
        "",
    ]
    T = _cells(tables) if tables else None
    f1 = (T["edit_gain"].get("T1", {}).get("pi05") or {}) if T else {}
    f8 = (T["edit_gain"].get("T8", {}).get("pi05") or {}) if T else {}
    r1 = (T["random_control"].get("T1", {}).get("pi05") or {}) if T else {}
    if _ok(f1) and _ok(f8):   # the numbers the page shows: the shared starts where run, else the paper's
        lines[1] = (f"<li><b>VLA with an action expert (π0.5).</b> The edit fixed its two grounding failures: {tn('T1')} "
                    f"{f1['a']} → {f1['b']} of {f1['n']} unseen starts and {tn('T8')} {f8['a']} → {f8['b']} of {f8['n']}"
                    + (f"; a random edit of the same size did nothing ({r1['r']} vs {r1['u']} of {r1['n']})"
                       if r1.get("s") == "ok" and None not in (r1.get("r"), r1.get("u")) and r1.get("n") == f1["n"]
                       else "") + ".</li>")
    else:
        lines[1] = (f"<li><b>VLA with an action expert (π0.5).</b> The edit fixed its two grounding failures: {tn('T1')} "
                    f"{e['base']} → {e['cand']} of {e['n']} unseen starts and {tn('T8')} {s4ho['base']} → {s4ho['cand']} "
                    f"of {s4ho['n']}; a random edit of the same size did nothing ({c['rand']} vs {c['base']} of {c['n']}).</li>")
    if tables:
        st = tldr_story(tables, models, PREVENT_LAST.get("p") or {"rows": [], "left": [], "target": "T8", "base": ""},
                        PREVENT_LAST.get("sites") or [])
        first = lambda t: re.split(r"(?<=[.;])\s", t, maxsplit=1)[0].rstrip(";") + ("" if t.split(";")[0].endswith(".") else ".")
        bits = " ".join(f"<b>{esc(it['lead'].rstrip('.:'))}:</b> {esc(first(it['text']) if it['id'] != 'fix' else it['text'])}"
                        for it in st["items"])
        lines.append(f"<li><b>Across architectures.</b> {bits}</li>")
    return "".join(lines)


# ---------------------------------------------------------------- TL;DR and section overview (top of the page)
PREVENT_LAST = {}       # the prevention rows and diagram sites of this run, for the brief
RUNNING = "{running}"   # rendered as the italic "running" marker; never a number
SIG = 0.05              # "lifted" / "worse": exact McNemar p below this, paired by start
TLDR_ORDER = ["pi05", "cosmos", "groot"]   # the action-expert VLA's established result first, then the other two


def task_list(keys):
    """['T4','T5','T6','T7'] -> 'T4–T7'; ['T2','T3'] -> 'T2 and T3'."""
    ts = sorted(int(k[1:]) for k in keys if re.fullmatch(r"T\d", k))
    runs = []
    for i in ts:
        if runs and i == runs[-1][-1] + 1:
            runs[-1].append(i)
        else:
            runs.append([i])
    out = []
    for r in runs:
        out += [f"T{r[0]}–T{r[-1]}"] if len(r) >= 3 else [f"T{i}" for i in r]
    out += [k for k in keys if not re.fullmatch(r"T\d", k)]
    return out[0] if len(out) == 1 else ", ".join(out[:-1]) + " and " + out[-1]


def and_list(xs):
    xs = list(xs)
    return xs[0] if len(xs) == 1 else ", ".join(xs[:-1]) + " and " + xs[-1]


def loc_name(mid, loc):
    """'the Cosmos-Reason2 backbone', 'the diffusion action head': a location in the model's own words."""
    parts = [x.strip() for x in MODELS[mid]["where"].split(" / ")]
    k = loc_class(loc)
    name = parts[0] if k == "backbone" else parts[1] if k == "action" and len(parts) > 1 else None
    return f"the {name}" if name else (loc or "an unnamed location")


def edits_phrase(mid, k, c):
    """'T1 edits in the X and in the Y' when several locations were tried on the same starts with the same result."""
    same = [x for x in (c.get("alts") or []) if (x["a"], x["b"], x["n"], x["prac"]) == (c["a"], c["b"], c["n"], bool(c.get("prac")))]
    if not same:
        return f"The {k} edit", False
    names = [loc_name(mid, c.get("loc"))] + [loc_name(mid, x["loc"]) for x in same]
    return f"{k} edits in " + " and in ".join(names), True


def tldr(tables, models, prevent=None):
    """3-5 bullets for the top of the page, read only from the tables; anything not measured yet says running."""
    T = {t["id"]: {r["key"]: r["cells"] for r in t["rows"]} for t in tables}
    edit_of = next(t for t in tables if t["id"] == "side_effects").get("edit", {})
    ok = lambda c: bool(c) and c.get("s") == "ok"
    nums = lambda c, f=("a", "b", "n"): ok(c) and all(isinstance(c.get(x), int) for x in f)
    cell = lambda tid, key, mid: T[tid].get(key, {}).get(mid)
    tasks = [t[0] for t in TASKS if t[0] != "R1"]

    def side(mid):
        """(edit task, [(task, cell)] off-target with counts, R1 cell) of a model's side-effect map."""
        got = [(k, cell("side_effects", k, mid)) for k in [t[0] for t in TASKS]]
        got = [(k, c) for k, c in got if nums(c, ("a", "b", "n", "delta"))]
        tgt = next((k for k, c in got if c.get("target")), None)
        return tgt, [(k, c) for k, c in got if not c.get("target") and k != "R1"], dict(got).get("R1")

    def lifted(c):
        return c.get("p") is not None and c["p"] < SIG and c["delta"] > 0

    items = []
    for mid in TLDR_ORDER:
        m = models[mid]
        prov = False
        gains, flat = [], []
        practice = []
        for k in tasks:
            c = cell("edit_gain", k, mid)
            if nums(c) and c.get("prac"):
                practice.append((k, c))
                continue
            if nums(c):
                prov |= bool(c.get("prov"))
                (gains if (c.get("pv") is not None and c["pv"] < SIG and c["b"] > c["a"]) else flat).append((k, c))
            elif ok(c) and c["v"].startswith("no gain"):
                prov |= bool(c.get("prov"))
                flat.append((k, c))
        _, off, _ = side(mid)
        transfer = {k for k, c in off if lifted(c)}
        bits = []
        if gains:
            bits.append("Edits lifted success " + and_list(
                f"on {k} from {c['a']} to {c['b']} of {c['n']}{' unseen starts' if i == 0 else ''} (p = {fp(c['pv'])})"
                for i, (k, c) in enumerate(gains)) + ".")
            rcn = next((cell("random_control", k, mid) for k, _ in gains if nums(cell("random_control", k, mid), ("u", "r", "n"))), None)
            if rcn:
                k0 = next(k for k, _ in gains if cell("random_control", k, mid) is rcn)
                if rcn.get("sub") and not rcn.get("prac"):
                    bits.append(f"On {rcn['n']} of those starts a random edit of the same size scored {rcn['r']}, "
                                f"against {rcn['e']} for the edit and {rcn['u']} unedited.")
                else:
                    bits.append(("On the practice starts the edit was built from, a" if rcn.get("prac") else "A")
                                + f" random edit of the same size scored {rcn['r']} of {rcn['n']} on {k0}, against "
                                f"{rcn['u']} unedited" + (f" and {rcn['e']} edited." if rcn.get("prac") else "."))
        ng = [k for k, _ in flat if k not in transfer]
        if ng:
            why = ", failures of perception, contact or missing skill" if (mid == "pi05" and ng == ["T4", "T5", "T6", "T7"]) else ""
            bits.append(f"No detectable gain on {task_list(ng)}{why}.")
        done = {k for k, _ in gains} | {k for k, _ in flat}
        if not done:
            fm = [k for k in [t[0] for t in TASKS] if nums(cell("failure_map", k, mid), ("k", "n"))]
            wd = [k for k in [t[0] for t in TASKS] if nums(cell("wording", k, mid))]
            if 0 < len(fm) <= 4:
                bits.append("Unedited so far: " + and_list(
                    f"{k} {cell('failure_map', k, mid)['k']} of {cell('failure_map', k, mid)['n']}" for k in fm)
                    + " starts.")
            elif len(fm) > 4:
                some = [k for k in fm if cell("failure_map", k, mid)["k"] > 0]
                none = [k for k in fm if cell("failure_map", k, mid)["k"] == 0]
                bits.append((f"Unedited, all {len(fm)} tasks: " if len(fm) == len(TASKS) else
                             f"Unedited, {len(fm)} of {len(TASKS)} tasks: ")
                            + and_list(f"{k} {cell('failure_map', k, mid)['k']} of {cell('failure_map', k, mid)['n']}"
                                       for k in some)
                            + (f"; no success on {task_list(none)}" if none else "") + ".")
            elif fm and len({cell("failure_map", k, mid)["n"] for k in fm}) == 1:
                bits.append(f"Unedited so far, of {cell('failure_map', fm[0], mid)['n']} starts each: "
                            + and_list(f"{k} {cell('failure_map', k, mid)['k']}" for k in fm) + ".")
            elif fm:
                bits.append(f"Unedited success is in for {len(fm)} of {len(TASKS)} tasks.")
            if wd:
                wc = [(k, cell("wording", k, mid)) for k in wd]
                up = [(k, c) for k, c in wc if c["b"] > c["a"]]
                no = [k for k, c in wc if c["b"] <= c["a"]]
                if up:
                    bits.append("Clearer wording, same weights: " + and_list(
                        f"{k} {c['a']}\u00a0→ {c['b']} of {c['n']}" + (f" (p = {fp(c['pv'])})" if c.get("pv") is not None else "")
                        for k, c in up) + (f"; no gain on {task_list(no)}." if no else "."))
                elif no:
                    bits.append(f"A clearer wording, same weights, gained nothing on {task_list(no)}.")
            for k, c in practice:
                rc0 = cell("random_control", k, mid)
                who, many = edits_phrase(mid, k, c)
                size_of = (f"the size of the {loc_name(mid, rc0.get('loc'))[4:]} edit" if many and rc0 and rc0.get("loc")
                           else "of the same size")
                ctl = (f"; a random edit {size_of}, {rc0['r']} of {rc0['n']}" if nums(rc0, ("u", "r", "n"))
                       and rc0.get("prac") else "")
                alt_ctl = [x["ctl"] for x in (c.get("alts") or []) if x.get("ctl") is not None]
                if many and ctl and alt_ctl:
                    ctl = (f"; random edits of the same sizes, {and_list(str(v) for v in [rc0['r']] + alt_ctl)} "
                           f"of {rc0['n']}")
                nxt = ("no held-out runs, since no edit passed practice" if m["status"] == "final"
                       else f"further edits: {RUNNING}")
                if c["b"] < c["a"]:
                    who = who[0].upper() + who[1:]
                    bits.append(f"{who} {'each ' if many else ''}lowered success on the {c['n']} practice starts "
                                f"{'they were' if many else 'it was'} built from, {c['a']} → {c['b']}"
                                + (f" (p = {fp(c['pv'])}{' each' if many else ''})" if c.get("pv") is not None else "")
                                + ctl + f"; {nxt}.")
                    continue
                bits.append(f"The {k} edit, on the {c['n']} practice starts it was built from: {c['a']} → {c['b']}"
                            + (f" (p = {fp(c['pv'])})" if c.get("pv") is not None else "") + ctl
                            + f"; on unseen starts, {RUNNING}.")
            if fm:
                if not (practice and len(fm) == len(TASKS)):
                    bits.append(f"Other tasks{'' if practice else ' and edits'}: {RUNNING}.")
            elif m["status"] == "final":
                bits.append("No edit results.")
            else:
                bits.append(f"Failure map and edits: {RUNNING}.")
        elif m["status"] != "final" and any(cell("edit_gain", k, mid) and cell("edit_gain", k, mid)["s"] == "running"
                                            for k in tasks):
            bits.append(f"Other tasks: {RUNNING}.")
        r = cell("render", "render", mid)
        if nums(r):
            bits.append(f"Render check: {(r.get('task') or 'one task').lower()} succeeded on {r['a']} of {r['n']} starts "
                        f"under RoboLab’s performance preset and {r['b']} of {r['n']} under the default preset, which all "
                        f"three models now use.")
        items.append(dict(id=mid, lead=m["name"], kind=MODELS[mid]["short"], tag=None, prov=prov or None,
                          text=" ".join(bits)))

    # side effects: each model's main edit, unchanged, on the other tasks
    parts, waiting = [], []
    for mid in TLDR_ORDER:
        tgt, off, r1 = side(mid)
        if (not tgt or not off) and (mid, "side_effects") in NOT_RUN:
            parts.append(f"{models[mid]['name']}: not run, because its T1 edits lowered success on T1 itself.")
            continue
        if not tgt or not off:
            waiting.append(models[mid]["name"])
            continue
        up = [(k, c) for k, c in off if lifted(c)]
        down = [(k, c) for k, c in off if c.get("p") is not None and c["p"] < SIG and c["delta"] < 0]
        on = task_list([k for k, _ in off] + (["R1"] if r1 else []))
        cl = [("lifted " + and_list(f"{k} ({c['a']}\u00a0→ {c['b']} of {c['n']})" for k, c in up)) if up else "lifted no task",
              "made none detectably worse" if not down else
              "lowered " + and_list(f"{k} ({c['a']}\u00a0→ {c['b']} of {c['n']})" for k, c in down)]
        if r1:
            cl.append(f"kept the retention task R1 at {r1['b']} of {r1['n']}" if r1["delta"] == 0 else
                      f"moved the retention task R1 from {r1['a']} to {r1['b']} of {r1['n']}")
        t = f"{models[mid]['name']}’s {tgt} edit, run unchanged on {on}, " + ", ".join(cl[:-1]) + ", and " + cl[-1]
        parts.append(t + "." + (prevent_tldr(prevent) if prevent and prevent["model"] == mid and down else ""))
    if waiting:
        parts.append(f"{and_list(waiting)}: {RUNNING}.")
    items.append(dict(id="side-effects", lead="Side effects", kind=None, tag=None, text=" ".join(parts)))

    for it in items:  # keep "p = 0.12" and "9 → 16" on one line
        it["text"] = re.sub(r"\s*→\s*", "\u00a0→\u00a0", it["text"].replace("p = ", "p\u00a0=\u00a0"))
    early = [it["lead"] for it in items if it.get("prov")]
    note = ("Lifted and worse mean an exact McNemar p < 0.05 on the same starts, edited against unedited. "
            + (f"Some of {and_list(x + '’s' for x in early)} numbers are from the paper’s evaluation, each on its own "
               "slots and number of starts." if early else "Every number here is from the shared slots."))
    return dict(items=items, note=note)


def tldr_html(t):
    run = '<span class="run">running</span>'
    out = []
    for it in t["items"]:
        lead = f'<b>{esc(it["lead"])}</b>' + (f' <span class="tk">{esc(it["kind"])}</span>' if it.get("kind") else "")
        tag = f' <span class="tag">{esc(it["tag"])}</span>' if it.get("tag") else ""
        out.append(f'<li data-id="{esc(it["id"])}"><span class="tl">{lead}{tag}</span> '
                   f'{esc(it["text"]).replace(esc(RUNNING), run)}</li>')
    return "".join(out)


# ---------------------------------------------------------------- page structure: contents, read time, figure numbers
def article(page):
    a, b = page.find("<article"), page.find("</article>")
    return a, b


def page_sections(page):
    """Every section (h2) and subsection (h3) of the article in page order, and any h4 marked data-toc (nested under
    its h3): the contents list follows the page. Generated regions are skipped, so headings inside them (the model
    cards) never enter the list."""
    page = re.sub(r"<!-- gen:([a-z0-9:_-]+) -->.*?<!-- /gen:\1 -->", "", page, flags=re.S)
    page = re.sub(r"<!-- (allnums|gallery) -->.*?<!-- /\1 -->", "", page, flags=re.S)
    a, b = article(page)
    secs = []
    pat = re.compile(r'<section id="([a-z0-9-]+)"[^>]*>\s*<h2[^>]*>(.*?)</h2>|<h3 id="([a-z0-9-]+)"[^>]*>(.*?)</h3>'
                     r'|<h4 id="([a-z0-9-]+)"[^>]*\bdata-toc\b[^>]*>(.*?)</h4>', re.S)
    for m in pat.finditer(page[a:b]):
        sid, title, level = ((m.group(1), m.group(2), 2) if m.group(1) else (m.group(3), m.group(4), 3) if m.group(3)
                             else (m.group(5), m.group(6), 4))
        title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        secs.append(dict(id=sid, title=title, level=level))
    return secs


def toc_html(secs, cls):
    """Nested lists, one level per heading level."""
    out, levels, open_li = [f'<ol class="{cls}">'], [secs[0]["level"] if secs else 2], False
    for x in secs:
        lv = x["level"]
        if open_li and lv > levels[-1]:
            out.append("<ol>")
            levels.append(lv)
        else:
            if open_li:
                out.append("</li>")
            while len(levels) > 1 and lv < levels[-1]:
                out.append("</ol></li>")
                levels.pop()
        out.append(f'<li><a href="#{esc(x["id"])}">{esc(x["title"])}</a>')
        open_li = True
    out.append("</li>" if open_li else "")
    while len(levels) > 1:
        out.append("</ol></li>")
        levels.pop()
    out.append("</ol>")
    return "".join(out)


WPM = 230


def read_minutes(page):
    """Words a reader meets in the article (collapsed tables and code excluded), at 230 words a minute."""
    a, b = article(page)
    art = re.sub(r"<!-- (allnums|gallery) -->.*?<!-- /\1 -->", " ", page[a:b], flags=re.S)
    art = re.sub(r"<!-- gen:references -->.*?<!-- /gen:references -->", " ", art, flags=re.S)
    art = re.sub(r'<span class="cite".*?</span>', " ", art, flags=re.S)
    art = re.sub(r'<details class="tdetails.*?</details>', " ", art, flags=re.S)
    art = re.sub(r"<svg\b.*?</svg>", " ", art, flags=re.S)
    art = re.sub(r"<(script|style|table|pre)\b.*?</\1>", " ", art, flags=re.S)
    words = len(re.findall(r"\w+", html.unescape(re.sub(r"<[^>]+>", " ", art))))
    return max(1, round(words / WPM)), words


def number_figures(page):
    """'Figure N.' in front of every figure caption in the article, in page order; placeholders stay unnumbered."""
    page = re.sub(r'<span class="fig-n">Figure \d+\.</span> ', "", page)
    a, b = article(page)
    n = 0

    def one(m):
        nonlocal n
        if "clip-pending" in m.group(1):
            return m.group(0)
        n += 1
        return m.group(0).replace("<figcaption>", f'<figcaption><span class="fig-n">Figure {n}.</span> ', 1)

    art = re.sub(r'<figure class="([^"]*)".*?</figure>', one, page[a:b], flags=re.S)
    return page[:a] + art + page[b:], n


def where_note(tables, models):
    """One sentence on the offline where-to-edit checks of the NVIDIA models, read from the where table."""
    row = next((r for t in tables if t["id"] == "where" for r in t["rows"] if r["key"] == "offline_where"), None)
    bits, ctl = [], []
    for mid in ("cosmos", "groot"):
        c = row["cells"].get(mid) if row else None
        if not c or c.get("s") != "ok" or not c.get("parts"):
            continue
        bits.append(f"{models[mid]['name']}’s " + " and ".join(
            f"{loc_name(mid, loc)[4:]} edit {sgn(v, False, 0)}%" for loc, v, _ in c["parts"]))
        ctl += [cv for _, _, cv in c["parts"] if cv is not None]
    if not bits:
        return ""
    rng = (f" Random edits of the same size: {sgn(min(ctl), False, 0)}% to {sgn(max(ctl), False, 0)}%." if len(ctl) > 1
           else f" A random edit of the same size: {sgn(ctl[0], False, 0)}%." if ctl else "")
    return (f'<p>Offline, on recorded decisions no edit was built from, the share of the clearer wording’s action change '
            f'each edit reproduces: {esc("; ".join(bits))}.{rng} Closed loop is what counts: see the figure below and '
            f'the table.</p>')


def adapter_note(cross, models):
    """Adapter checks: NVIDIA's unmodified server and client against this study's adapter, from the track's records."""
    out = []
    for mid in ("cosmos", "groot"):
        src = cross.get(mid)
        rows = [r for r in (src["rows"] if src else []) if "official" in nk(describe(r.get("experiment")) or "")]
        bits = []
        for r in rows:
            m = re.search(r"this adapter:\s*(\d+)\s*/\s*(\d+)", str(r.get("notes") or ""))
            k, n = as_int(r.get("unedited_k")), as_int(r.get("n"))
            if not m or None in (k, n):
                continue
            task = r["_task"] or re.sub(r"^[-\s]+|Task$", "", str(r.get("task") or "")).strip()
            task = SHORT[task] if task in SHORT else re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower()
            bits.append(f"{task}: {k} of {n} with NVIDIA’s, {m.group(1)} of {m.group(2)} with ours")
        if bits:
            out.append(f"<p class=\"note\"><strong>Adapter check, {esc(models[mid]['name'])}.</strong> NVIDIA’s "
                       f"unmodified server and RoboLab client against this study’s adapter, default render preset: "
                       f"{esc('; '.join(bits))}. The zeros in the failure map are the model’s, not the adapter’s.</p>")
    return "".join(out)


def conclusion_html(tables, models, prevent=None):
    """Where each NVIDIA model stands, one sentence each, read from the wording and edit tables."""
    T = {t["id"]: t for t in tables}
    out = []
    for mid in ("cosmos", "groot"):
        name = models[mid]["name"]
        w = [(r["label"].split(" ")[0], r["cells"][mid]) for r in T["wording"]["rows"]
             if r["cells"].get(mid, {}).get("s") == "ok" and r["cells"][mid].get("n")]
        e = [(r["label"].split(" ")[0], r["cells"][mid]) for r in T["edit_gain"]["rows"]
             if not r.get("sub") and r["cells"].get(mid, {}).get("s") == "ok"]
        pe = [(k, c) for k, c in e if c.get("prac")]
        e = [(k, c) for k, c in e if not c.get("prac")]
        bits = []
        helps = [(tid, c) for tid, c in w if c["b"] > c["a"]]
        flat = [tid for tid, c in w if c["b"] <= c["a"]]
        for tid, c in helps:
            sig = c.get("pv") is not None and c["pv"] < SIG
            bits.append(f"a clearer wording, with the same weights, takes {tid} from {c['a']} to {c['b']} of {c['n']} "
                        f"practice starts (p\u00a0=\u00a0{fp(c['pv'])})"
                        + (", a gap to distil" if sig else ", a gap these starts cannot yet tell from noise"))
        if flat:
            bits.append(f"on {task_list(flat)} the clearer wording tried does not help, so there is nothing to distil")
        RC = {r["label"].split(" ")[0]: r["cells"].get(mid) for r in T["random_control"]["rows"]}
        for tid, c in pe:
            r0 = RC.get(tid)
            verb = "lifts success from" if c["b"] > c["a"] else "lowers success from" if c["b"] < c["a"] else "leaves success at"
            who, many = edits_phrase(mid, tid, c)
            size_of = (f"the size of the {loc_name(mid, r0.get('loc'))[4:]} edit" if many and r0 and r0.get("loc")
                       else "of the same size")
            alt_ctl = [x["ctl"] for x in (c.get("alts") or []) if x.get("ctl") is not None]
            if many:
                verb = "each " + verb.replace("lifts", "lift").replace("lowers", "lower").replace("leaves", "leave")
            bits.append(f"on the {c['n']} practice starts {'they were' if many else 'it was'} built from, "
                        f"{who if many else 'the' + who[3:]} {verb} {c['a']}"
                        + (f" to {c['b']}" if c["b"] != c["a"] else "")
                        + (f" (p\u00a0=\u00a0{fp(c['pv'])})" if c.get("pv") is not None else "")
                        + ((f", where random edits of the same sizes reach {and_list(str(v) for v in [r0['r']] + alt_ctl)}"
                            if many and alt_ctl else f", where a random edit {size_of} reaches {r0['r']}")
                           if r0 and r0.get("s") == "ok" and r0.get("r") is not None else ""))
        for tid, c in e:
            r0 = RC.get(tid)
            verb = "lifts success from" if c["b"] > c["a"] else "lowers success from" if c["b"] < c["a"] else "leaves success at"
            bits.append(f"on {c['n']} unseen starts the {tid} edit {verb} {c['a']}" + (f" to {c['b']}" if c["b"] != c["a"] else "")
                        + (f" (p\u00a0=\u00a0{fp(c['pv'])})" if c.get("pv") is not None else "")
                        + ((", and on the practice starts it was built from a random edit of the same size reaches "
                            f"{r0['r']} of {r0['n']}" if r0.get("prac") else
                            f", and on {r0['n']} of those starts a random edit of the same size reaches {r0['r']}"
                            if r0.get("sub") else
                            f", where a random edit of the same size reaches {r0['r']}")
                           if r0 and r0.get("s") == "ok" and r0.get("r") is not None else ""))
        for tid, c in e:   # the same distillation written into another part of the model
            for x in c.get("alts") or []:
                if x.get("b") is not None and x.get("n"):
                    bits.append(f"written into {loc_name(mid, x['loc'])} instead, the same distillation scores {x['b']} "
                                f"of {x['n']}{' practice' if x.get('prac') else ''} starts (unedited {x['a']})")
        # side effects that cleared the bar: off-target tasks the edit lifted or lowered (exact McNemar p < SIG)
        se = [(r["key"], r["cells"].get(mid)) for r in T["side_effects"]["rows"]]
        se = [(k, c) for k, c in se if c and c.get("s") == "ok" and not c.get("target") and c.get("p") is not None
              and c["p"] < SIG and c.get("delta")]
        if se and e:
            bits.append(f"run unchanged on other tasks, the {e[0][0]} edit " + and_list(
                f"{'lifts' if c['delta'] > 0 else 'lowers'} {k} from {c['a']} to {c['b']} of {c['n']}"
                + (" practice starts" if c.get("prac") else "") + f" (p\u00a0=\u00a0{fp(c['p'])})" for k, c in se))
            if prevent and prevent["model"] == mid and any(c["delta"] < 0 for _, c in se):
                bits += prevent_conclusion(prevent)
        if not bits:
            out.append(f"<p><strong>{esc(name)}.</strong> Wording and edit results are "
                       f'<span class="run">running</span>.</p>')
            continue
        worse = any(c["b"] < c["a"] for _, c in pe)
        done = models[mid]["status"] == "final"
        tail = ("" if e else ' No held-out or side-effect runs were made, since no edit passed practice.' if done and pe
                else ' Further edits are <span class="run">running</span>.' if worse else
                ' On unseen starts the edit is <span class="run">running</span>.' if pe else
                ' The edit itself is <span class="run">running</span>.')
        text = "; ".join(bits)
        out.append(f"<p><strong>{esc(name)}.</strong> {esc(text[0].upper() + text[1:])}.{tail}</p>")
    return "".join(out)


def numbers_hash(data):
    """A hash of every number the page's tables show (cell values and details, by table, row and model), and the
    numbers in the TL;DR and the generated paragraphs (when the edit works, prevention, second edit); wording changes in
    the code do not change it, new or changed numbers do."""
    nums = []

    def cells(prefix, rows, key="key"):
        for r in rows:
            for m, c in (r.get("cells") or {}).items():
                if isinstance(c, dict):
                    for f in ("v", "d"):
                        nums.append((prefix, str(r.get(key)), m, f, re.findall(r"\d+(?:\.\d+)?", str(c.get(f) or ""))))
    for t in data["tables"] + [data["summary"]]:
        cells(t["id"], t["rows"])
    cells("prevention", (data.get("prevention") or {}).get("rows") or [], "name")
    cells("second", (data.get("second_edit") or {}).get("rows") or [], "name")
    for k, x in (("tldr", data.get("tldr")), ("when", data.get("when_text")),
                 ("prevention", (data.get("prevention") or {}).get("text")),
                 ("second", (data.get("second_edit") or {}).get("text"))):
        nums.append((k, re.findall(r"\d+(?:\.\d+)?", json.dumps(x, ensure_ascii=False))))
    return hashlib.sha256(json.dumps(nums, ensure_ascii=False).encode()).hexdigest()[:16]


def rel(path):
    """A path for messages: relative to the repository when inside it (a test copy under --out may not be)."""
    try:
        return Path(path).resolve().relative_to(ROOT)
    except ValueError:
        return path


def print_pdf(src_html, pdf):
    """Print brief.html to one Letter page: step down the zoom (body data-fit 0-3) until it fits."""
    if not Path(CHROME).exists():
        print(f"  skip PDF: Chrome not found at {CHROME} (set CHROME=...)")
        return
    text = src_html.read_text()
    with tempfile.TemporaryDirectory() as td:
        for fit in range(4):
            html_fit = re.sub(r'<body[^>]*>', f'<body data-fit="{fit}">', text, count=1)
            src_html.write_text(html_fit)
            tmp = Path(td) / f"brief{fit}.pdf"
            cmd = [CHROME, "--headless=new", "--disable-gpu", "--no-first-run", "--no-default-browser-check",
                   f"--user-data-dir={td}/profile", "--no-pdf-header-footer", "--virtual-time-budget=10000",
                   f"--print-to-pdf={tmp}", src_html.resolve().as_uri()]
            subprocess.run(cmd, capture_output=True, text=True, timeout=180)
            if not tmp.exists():
                print("  PDF failed: Chrome wrote nothing")
                return
            b = tmp.read_bytes()
            pages = len(re.findall(rb"/Type\s*/Page[^s]", b))
            if pages == 1:
                break
        shutil.copyfile(tmp, pdf)
        print(f"wrote {rel(pdf)} ({pages} page{'s' if pages != 1 else ''}, zoom step {fit}, {len(b) // 1024} KB)")
        if pages != 1:
            print("  WARNING: the brief does not fit one page even at the smallest zoom; shorten brief.html")


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description="Rebuild site/cosmos/ from the result records.")
    ap.add_argument("--no-fetch", action="store_true", help="do not git fetch the cross-model branches first")
    ap.add_argument("--cosmos", help="a local metrics_cosmos.json instead of origin/cross-cosmos/v1")
    ap.add_argument("--groot", help="a local metrics_groot.json instead of origin/cross-groot/v1")
    ap.add_argument("--pi05", help="a local metrics_pi05.json instead of origin/cross-pi05/v1")
    ap.add_argument("--no-figures", action="store_true")
    ap.add_argument("--no-pdf", action="store_true")
    ap.add_argument("--force", action="store_true", help="print the PDF even if nothing changed")
    ap.add_argument("--out", help="write everything into a fresh copy of site/ at this directory (videos linked), never "
                                  "into site/cosmos; required with --cosmos / --groot / --pi05, so test records can never "
                                  "touch the committed outputs or the updated stamp")
    ap.add_argument("--set-updated", help="set the updated stamp (YYYY-MM-DDTHH:MMZ), to reset it to the last real data "
                                          "change")
    args = ap.parse_args()
    global OUT, FIG, VID
    if (args.cosmos or args.groot or args.pi05) and not args.out:
        ap.error("local record files are for tests: pass --out DIR so the committed outputs stay untouched")
    if args.out:
        dst = Path(args.out).resolve()
        site = (ROOT / "site").resolve()
        if dst == site or site in dst.parents or dst in site.parents:
            ap.error("--out must be outside site/")
        shutil.rmtree(dst, ignore_errors=True)
        shutil.copytree(site, dst, ignore=shutil.ignore_patterns("video"))
        (dst / "cosmos" / "video").symlink_to(site / "cosmos" / "video")
        OUT, FIG, VID = dst / "cosmos", dst / "cosmos" / "fig", dst / "cosmos" / "video"
        print(f"  test run: every output goes to {dst}")

    if not args.no_fetch:
        for _, branch, _ in SOURCES:
            r = subprocess.run(["git", "-C", str(ROOT), "fetch", "--quiet", "origin",
                                f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], capture_output=True, text=True)
            print(f"  fetch {branch}: {'ok' if r.returncode == 0 else 'not on origin yet'}")

    ensure_thumbs()
    S = Sources()
    P = build_pi05(S)
    meta = {}
    cross = read_cross(args, meta)
    DONE["all"] = all(cross.get(m) and cross[m]["final"] for m in ORDER)

    models = {}
    for mid in ORDER:
        m = dict(MODELS[mid])
        src = cross.get(mid)
        if src and any(not r["_prov"] for r in src["rows"]):
            m["status"] = "final" if src["final"] else "partial"
        elif src and src["rows"]:
            m["status"] = "measured"
            for k in ("checkpoint", "architecture"):
                if src["meta"].get(k):
                    m[k] = describe(src["meta"][k])
            ck = next((r["checkpoint"] for r in src["rows"] if r.get("checkpoint")), None)
            if ck:
                m["checkpoint"] = describe(ck)
        m["source"] = meta.get(mid)
        if mid == "pi05" and m["status"] == "measured":
            m["short"] = "OpenPI VLA, control"
            m["fig_short"] = "control"   # fits the figure's column head
        models[mid] = m

    tables, rc, prov, rcp = build_tables(P, cross)
    fmrows = next(t for t in tables if t["id"] == "failure_map")["rows"]
    pv = sum(1 for r in fmrows if r["cells"]["pi05"].get("prov"))
    fresh = [r["key"] for r in fmrows if r["cells"]["pi05"].get("s") == "ok" and not r["cells"]["pi05"].get("prov")]
    if 0 < pv < len(fmrows):   # some fresh shared-slot cells, the rest earlier records: name the fresh ones in the head
        models["pi05"]["fig_short"] = (f"control · shared slots: {', '.join(fresh)}" if len(fresh) <= 2
                                       else "control")
    summary = summary_table(P, tables, cross, rc)
    prevent = prevention(cross)
    second = second_edit(cross, tables)
    SECOND_LAST.update(s=second)
    WHEN_LAST.clear()
    pros = prospect(cross, tables)
    PROSPECT_LAST.update(p=pros)
    when = when_html(tables, cross, prevent, second, pros)
    w = P["where"]
    WHERE_REF.update(n=w["n"], base=w["arms"]["base"], tx=w["tx"], bars=[
        ("Backbone: VLM MLPs", w["arms"]["v1full"], "teal"), ("Action expert MLPs", w["arms"]["a1"], "dark"),
        ("Action head, least norm", w["arms"]["h1"], "dark"), ("Action head, hand-built gate", w["arms"]["p1"], "dark")])
    for mid in ("cosmos", "groot"):
        if rc[mid]["where"].get("_bars"):
            WHERE_OTHER[mid] = rc[mid]["where"]["_bars"]
    page = (OUT / "index.html").read_text()
    data = {
        "title": "Weight editing across robot models",
        "question": "A world-action model and a reasoning-backbone VLA: where does weight-space programming fix failures?",
        "status": "draft",
        "shared_slots": SHARED_SLOTS,
        "models": [dict(id=mid, **models[mid]) for mid in ORDER],
        "tasks": [dict(id=t[0], name=t[1], robolab=t[2], pi05_failure=t[4]) for t in TASKS],
        "tldr": None,
        "tldr_models": tldr(tables, models, prevent),
        "sections": page_sections(page),
        "summary": summary,
        "prevention": prevent,
        "second_edit": second,
        "when_text": html.unescape(re.sub(r"<[^>]+>", " ", when)).split() and " ".join(
            html.unescape(re.sub(r"<[^>]+>", " ", when)).split()),
        "diagrams": {"where": where_sites(P, rc, tables)},
        "tables": tables,
        "cross_rows": {mid: [{k: v for k, v in r.items() if k != "_extra"} | ({"extra": r["_extra"]} if r["_extra"] else {})
                             for r in cross[mid]["rows"]] for mid in cross},
        "sources": {"pi05_earlier_records": dict(sorted(S.used.items())), **meta},
        "method": "Exact two-sided McNemar on fixed vs broke pairs; paired bootstrap 95% interval of the difference "
                  "(20,000 multinomial draws over the four paired outcome types, seed 0); Wilson 95% intervals.",
    }
    data["tldr"] = tldr_story(tables, models, prevent, data["diagrams"]["where"])
    PREVENT_LAST.update(p=prevent, sites=data["diagrams"]["where"])
    data = strip_cites(dename_deep(data))   # task names, not codes, in every display string (cross_rows and sources
    # stay raw); citations are the page's, not the data's

    def no_commit(x):  # a branch commit that changes no number does not move the "updated" stamp
        if isinstance(x, dict):
            return {k: no_commit(v) for k, v in x.items() if k != "commit"}
        return [no_commit(v) for v in x] if isinstance(x, list) else x
    blob = json.dumps(no_commit(data), sort_keys=True, ensure_ascii=False)
    h = hashlib.sha256(blob.encode()).hexdigest()[:16]
    mpath = OUT / "metrics.json"
    old = json.loads(mpath.read_text()) if mpath.exists() else {}
    changed = old.get("content_hash") != h
    now = dt.datetime.now(dt.timezone.utc)
    data["content_hash"] = h
    # The "updated" stamp says when the data last changed: it moves only when a source branch moved AND a number the
    # page shows changed, both against the committed outputs (HEAD), so a code change, a rerun or a test cannot move it.
    head = git("show", "HEAD:site/cosmos/metrics.json")
    try:
        ref = json.loads(head) if head else old
    except ValueError:
        ref = old
    nh = numbers_hash(data)
    commits = {m: (data["sources"].get(m) or {}).get("commit") for m in ORDER}
    ref_commits = {m: ((ref.get("sources") or {}).get(m) or {}).get("commit") for m in ORDER}
    moved = ref.get("numbers_hash") is not None and nh != ref.get("numbers_hash") and commits != ref_commits
    data["numbers_hash"] = nh
    data["updated"] = (args.set_updated or (now.strftime("%Y-%m-%dT%H:%MZ") if moved or not ref.get("updated")
                                            else ref["updated"]))
    upd = dt.datetime.strptime(data["updated"], "%Y-%m-%dT%H:%MZ")
    data["updated_text"] = f"{upd.day} {upd:%b %Y}, {upd:%H:%M} UTC"
    data["status_text"] = " · ".join(f"{models[m]['name']}: {STATUS[models[m]['status']]}" for m in ORDER)
    mpath.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {rel(mpath)} ({'changed' if changed else 'unchanged'}, {h})")

    if args.no_figures:
        alts = {n: c for n, c in FIGS}
    else:
        alts = draw_figures(dename_deep(tables), models)
        print(f"checked {len(FIGS) * 4 + len(BRIEF_FIGS)} figure files in {rel(FIG)}/")

    idx = OUT / "index.html"
    page = replace_region(page, "tldr", tldr_html(data["tldr"]))
    page = replace_region(page, "toc", toc_html(data["sections"], "toc-l1"))
    page = replace_region(page, "toc-m", toc_html(data["sections"], "toc-l1"))
    page = replace_region(page, "stamp", esc(f"{data['status_text']} · updated {data['updated_text']}"))
    page = replace_region(page, "models", models_html(models))
    page = replace_region(page, "render-note", render_note(tables))
    page = replace_region(page, "conclusion", conclusion_html(tables, models, data["prevention"]))
    page = replace_region(page, "prevent", prevent_main(data["prevention"]))
    page = replace_region(page, "prevent-full", prevent_html(data["prevention"]))
    page = replace_region(page, "when", when)
    page = replace_region(page, "problem", problem_html(tables, cross).replace(
        "<!-- clipslot -->", '<div class="clipgrid one wide"><!-- gen:clips:problem --><!-- /gen:clips:problem --></div>'))
    page = replace_region(page, "glance", glance_html(tables, data["diagrams"]["where"]))
    page = replace_region(page, "prospect-full", prospect_html(PROSPECT_LAST.get("p"))[0] if PROSPECT_LAST.get("p") else "")
    page = replace_region(page, "second", second_main(data["second_edit"]))
    page = replace_region(page, "second-full", second_html(data["second_edit"]))
    page = replace_region(page, "tldr-models", "<ul>" + tldr_html(data["tldr_models"]) + "</ul>")
    page = replace_region(page, "fix-text", fix_text(tables, cross))
    page = replace_region(page, "dg:fix", dg_fix_chart(tables))
    page = replace_region(page, "left-text", left_text(tables))
    page = replace_region(page, "dg:side", dg_side_chart(tables))
    page = replace_region(page, "groot-text", groot_text(tables))
    page = replace_region(page, "control-text", pi05_story(P, tables))
    page = replace_region(page, "dg:schematic", dg_schematic_html())
    page = replace_region(page, "dg:where", dg_where_html(data["diagrams"]["where"]))
    page = replace_region(page, "dg:where-results", dg_where_results(data["diagrams"]["where"]))
    page = replace_region(page, "where-text", where_text(tables))
    page = replace_region(page, "legend", legend_html(cross, tables))
    page = replace_region(page, "dg:groot", dg_groot_html(tables, cross)[0])
    page = replace_region(page, "where-note", where_note(tables, models))
    page = replace_region(page, "adapter-note", adapter_note(cross, models))
    page = replace_region(page, "table:summary", table_html(summary, models))
    for t in tables:
        page = replace_region(page, f"table:{t['id']}", block_html(t, models))
    prac_edit = any(r["cells"][m].get("prac") for t in tables if t["id"] == "edit_gain" for r in t["rows"] for m in ORDER)
    for name, cap in FIGS:
        if name == "failure-map" and 0 < pv < len(fmrows):
            cap += (f" π0.5’s cells for {and_list(dename(TASK_NAME.get(k, k)) for k in [r['key'] for r in fmrows if r['cells']['pi05'].get('prov')])} "
                    "come from the paper’s evaluation, on its own slots.")
        if name == "edit-gain" and prac_edit:
            cap += " Hollow, dashed: scored on the practice starts the edit was built from, not on unseen starts."
        page = replace_region(page, f"fig:{name}", figure_html(name, alts[name], cap))
    clips = prepare_clips()
    for group in ("problem", "fix", "gallery"):
        page = replace_region(page, f"clips:{group}", clips_section(group, clips, tables, models))
    page, nfig = number_figures(page)
    page, reflist = number_citations(page, load_refs())
    page = replace_region(page, "references", reflist)
    mins, words = read_minutes(page)
    page = replace_region(page, "readtime", f"{mins} min read")
    keep = re.search(r"<!-- legend -->.*?<!-- /legend -->", page, re.S)
    page = (dename(page[:keep.start()]) + keep.group(0) + dename(page[keep.end():])) if keep else dename(page)
    idx.write_text(page)
    print(f"  {nfig} numbered figures and clips; {words:,} words, {mins} min read")
    print(f"wrote {rel(idx)}")

    br = OUT / "brief.html"
    btxt = old_brief = br.read_text()
    btxt = replace_region(btxt, "stamp", esc(f"Draft for discussion · {data['status_text']} · {data['updated_text']}"))
    btxt = replace_region(btxt, "summary-lines", brief_lines(P, rc, models, cross, tables))
    btxt = replace_region(btxt, "gif", gif_html(clips))
    # the brief keeps to one page: the π0.5 cells there leave out the fresh shared-slot unedited counts (on the page)
    sb = json.loads(json.dumps(summary))
    for r in sb["rows"]:
        c = r["cells"].get("pi05") or {}
        if c.get("d"):
            c["d"] = re.sub(r"(?:^|; )shared slots, [^;]*", "", c["d"]) or None
    btxt = replace_region(btxt, "table:summary", table_html(sb, models))
    caps = dict(FIGS)
    for name in BRIEF_FIGS:
        btxt = replace_region(btxt, f"fig:{name}", figure_html(name, alts[name], caps[name], pdf=True))
    btxt = strip_cites(dename(btxt))
    br.write_text(btxt)
    print(f"wrote {rel(br)}")

    pdf = OUT / "weightsmith-cosmos-brief.pdf"
    inputs_changed = btxt != old_brief or any(n.startswith("brief") for n in CHANGED)
    if not args.no_pdf and (inputs_changed or args.force or not pdf.exists()):
        print_pdf(br, pdf)
    print(f"  figures rewritten: {len(CHANGED)}" + (f" ({', '.join(sorted(CHANGED)[:6])}{', ...' if len(CHANGED) > 6 else ''})"
                                                     if CHANGED else ""))


if __name__ == "__main__":
    sys.exit(main())
