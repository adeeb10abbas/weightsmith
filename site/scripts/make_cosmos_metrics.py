#!/usr/bin/env python3
"""Build the draft cross-model page (site/cosmos/) from the result records. One command refreshes everything:

    python3 site/scripts/make_cosmos_metrics.py            # fetch the three cross-model branches, rebuild all outputs
    python3 site/scripts/make_cosmos_metrics.py --no-fetch # use the remote-tracking refs as they are
    python3 site/scripts/make_cosmos_metrics.py --cosmos a.json --groot b.json --pi05 c.json   # local files instead

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
  "status": "final" turns that model's empty cells from "running" to "—".
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
  index.html, brief.html       generated regions between <!-- gen:NAME --> and <!-- /gen:NAME --> markers
  fig/<name>[-dark].svg|png    six figures, SVG and PNG at 2x, light and dark; fig/brief-*.svg for the PDF
  weightsmith-cosmos-brief.pdf one-page brief printed from brief.html by headless Chrome

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
        name="Cosmos3 Edge Policy", short="world-action model", kind="NVIDIA world-action model",
        status="running", checkpoint="`nvidia/Cosmos3-Edge-Policy-DROID` at `a7c7288`, 3.37B parameters",
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
        name="GR00T N1.7", short="reasoning-backbone VLA", kind="NVIDIA vision-language-action policy",
        status="running", checkpoint="`nvidia/GR00T-N1.7-DROID` at `05e7cc9`, 3.14B parameters",
        architecture="A Cosmos-Reason2-2B vision-language backbone (Qwen3-VL), cut to its first 16 language layers, reads an "
                     "exterior and a wrist camera (180 × 320 each) and the instruction. The action head, 4 self-attention "
                     "layers and 32 diffusion-transformer blocks, turns the backbone's tokens and the robot state into "
                     "40-step chunks by flow matching; the first 8 steps run at 15 Hz.",
        entry="As text tokens in the Cosmos-Reason2 backbone, next to the image tokens (lowercased, punctuation removed). "
              "The action head reads the instruction only through the backbone's tokens.",
        where="Cosmos-Reason2 backbone / diffusion action head",
        evaluation="Closed loop on the shared RoboLab DROID tasks and slots, NVIDIA's validated client and embodiment "
                   "setting.",
    ),
    "pi05": dict(
        name="π0.5", short="OpenPI VLA, control", kind="OpenPI vision-language-action policy, the control",
        status="measured", checkpoint="`pi05_droid_jointpos` (OpenPI), about 3.35 billion parameters",
        architecture="An 18-layer PaliGemma vision-language model (VLM) and an 18-layer Gemma action expert. The action "
                     "expert attends to the VLM's keys and values and produces 15-step action chunks, including a "
                     "gripper command, by flow matching.",
        entry="As text tokens in the VLM, together with image tokens from two cameras and robot-state tokens, under "
              "bidirectional attention, so the image tokens read the instruction. The distilled edit goes into the "
              "VLM's MLP weights.",
        where="VLM / action expert / action head",
        evaluation="Closed loop on RoboLab in Isaac Sim. Until its shared-slot runs land, its cells are earlier records "
                   "(RTX 3090, performance render preset), each tagged with its own slots and number of starts. Trained on "
                   "DROID data and none of these tasks.",
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
SHORT = {"T1": "Cube then banana", "T2": "Eyeglasses", "T3": "Cube left of bowl", "T4": "Stack yellow on red",
         "T5": "Mustard into left bin", "T6": "Stack three cubes", "T7": "Flat box",
         "T8": "Mustard, reference-first", "R1": "Retention task"}

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
    return f"{tid} {TASK[tid][1]}"


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
                           a=c["base"], b=c["teacher"], n=c["n"], fl=f"T1 {SHORT['T1']} ({c['n']} unseen)")
    T["wording"]["T8"] = C(f"{s4['rf']} → {s4['tf']} of {s4['n']}",
                           f"{s4['tf_only']} / {s4['rf_only']}, p = {fp(s4['p'])}; {fci(s4['diff'], *s4['ci'])}; "
                           f"{slots_txt(s4['slots'], 'practice')}; reference-first → target-first: “{s4['rf_prompt']}” → "
                           f"“{s4['tf_prompt']}”", src=s4["src"], a=s4["rf"], b=s4["tf"], n=s4["n"],
                           fl=f"T8 {SHORT['T8']} ({s4['n']} practice)")
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
        t=c["teacher"], fl=f"T1 {SHORT['T1']} ({c['n']} unseen)")
    T["random_control"]["T8"] = C(
        f"control {s4ho['rand']}, unedited {s4ho['base']}, edit {s4ho['cand']} of {s4ho['n']}", slots_txt(s4ho["slots"]),
        src=s4ho["src"], u=s4ho["base"], r=s4ho["rand"], e=s4ho["cand"], n=s4ho["n"], fl=f"T8 {SHORT['T8']} ({s4ho['n']} unseen)")

    tve = c["teacher_vs_edit"]
    rec1 = 100 * (c["edit"] - c["base"]) / (c["teacher"] - c["base"])
    rec8 = 100 * (sp["cand"] - sp["base"]) / (sp["tf"] - sp["base"])
    T["recovery"]["T1"] = C(f"{rec1:.0f}%", f"({c['edit']} − {c['base']}) / ({c['teacher']} − {c['base']}) of {c['n']}, "
                            f"computed here; edit vs teacher {tve['broke']} / {tve['fixed']}, p = {fp(tve['mcnemar_p'])}; "
                            f"{slots_txt(c['slots'])}", src=c["src"])
    T["recovery"]["T8"] = C(f"{rec8:.0f}%", f"({sp['cand']} − {sp['base']}) / ({sp['tf']} − {sp['base']}) of {sp['n']}, "
                            f"{slots_txt(sp['slots'], 'practice')}, which were used to build the edit; the teacher was not "
                            "run on unseen starts", src=sp["src"])

    cm = w["cmp"]
    W_ = T["where"]
    W_["starts"] = C(f"T1, {w['n']} unseen starts", slots_txt(w["slots"]), src=w["src"])
    W_["unedited"] = C(f"{w['arms']['base']} of {w['n']}", src=w["src"])
    W_["backbone"] = C(f"VLM MLPs: {w['arms']['v1full']} of {w['n']}",
                       f"{cm['v1full']['fixed']} / {cm['v1full']['broke']}, p = {fp(cm['v1full']['mcnemar_p'])}; the VLM's "
                       "weights also process the image tokens, where this edit is keyed", src=w["src"])
    W_["action"] = C(f"action expert {w['arms']['a1']}, action head {w['arms']['h1']} of {w['n']}",
                     f"expert {cm['a1']['fixed']} / {cm['a1']['broke']}, p = {fp(cm['a1']['mcnemar_p'])}; head "
                     f"{cm['h1']['fixed']} / {cm['h1']['broke']}; a head edit with a hand-built gate (blind editor): "
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
            nr["_extra"] = extra
            nr["_task"] = task_id(nr.get("code")) or task_id(nr.get("task"))
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
        m_st = re.search(r"^Status:\s*\*\*([^*]*\bcomplete\b[^*]*)\*\*", readme or "", re.M | re.I)
        if m_st:
            final = True
            meta["status_note"] = f"{m_st.group(1).strip()} ({branch} README)"
        out[mid] = dict(rows=norm, meta=meta, unknown=sorted(unknown), final=final)
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


def loc_short(loc):
    """'text pathway', 'VLM backbone', 'diffusion transformer': the shorter of a location's name and its gloss."""
    m = re.match(r"\s*(.*?)\s*\((.*)\)\s*$", loc or "")
    return min((m.group(1), m.group(2)), key=len) if m else (loc or "")


def is_practice(r):
    """An edit scored on the practice slots (the starts it was built from), or named so by its record."""
    return slot_kind(r) == " practice" or "practice" in nk(describe(r.get("experiment")) or "")


def row_cells(src):
    """Cells of one model from its cross-model rows, keyed like the tables."""
    T = {k: {} for k in TABLE_IDS}
    if not src or not src["rows"]:
        return T
    rows = src["rows"]
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
        if tw:
            r = tw[0]
            i = ints(r)
            wp = r.get("wording_pair")
            if isinstance(wp, (list, tuple)) and len(wp) == 2:
                wp = f"“{wp[0]}” → “{wp[1]}”"
            elif isinstance(wp, dict):
                o = wp.get("original") or wp.get("orig") or wp.get("rf") or wp.get("from")
                cl = wp.get("clear") or wp.get("teacher") or wp.get("tf") or wp.get("to")
                wp = f"“{o}” → “{cl}”" if o and cl else describe(wp)
            wd = [f"{sgn(100 * (i['teacher_k'] - i['unedited_k']) / i['n'])} points"]
            pw = None
            if as_int(r.get("edited_k")) is None and None not in (i["fixed"], i["broke"]):
                pw = as_float(r.get("p"))
                wd = [f"{i['fixed']} / {i['broke']}" + (f", p = {fp(pw)}" if pw is not None else "")]
                cw = as_ci(r.get("ci"))
                d0 = round(100 * (i["teacher_k"] - i["unedited_k"]) / i["n"], 1)
                wd.append(fci(d0, *cw) if cw else f"{sgn(d0)} points")
            T["wording"][tid] = C(f"{i['unedited_k']} → {i['teacher_k']} of {i['n']}",
                                  "; ".join(x for x in (*wd, where_txt(r), describe(wp)) if x),
                                  a=i["unedited_k"], b=i["teacher_k"], n=i["n"], pv=pw,
                                  fl=f"{tid} {SHORT[tid]} ({i['n']}{slot_kind(r)})")
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
        same_edit = [x for x in ed[1:] if describe(x.get("edit_location")) == loc and is_practice(x) and not prac]
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
                for x in ed[1:] if describe(x.get("edit_location")) != loc]
        cell = C(f"{i['unedited_k']} → {i['edited_k']}" + (f" of {i['n']}" if i["n"] else ""),
                 "; ".join(x for x in det if x), a=i["unedited_k"], b=i["edited_k"], n=i["n"], fx=i["fixed"], br=i["broke"],
                 pv=p,
                 ci=[d, *ci] if (ci is not None and d is not None) else None,
                 color="teal" if loc_class(loc) in (None, "backbone", "both") else "dark", prac=prac or None,
                 loc=loc, alts=alts or None,
                 fl=(f"{tid} {SHORT[tid]}, practice starts" if prac else
                     f"{tid} {SHORT[tid]}" + (f", {short_loc}" if loc else "")),
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
            u_sub = 0 if ic["unedited_k"] == 0 else (cn if ic["unedited_k"] == ic["n"] else None)
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
                fl=f"{tid} {SHORT[tid]} ({cn}{slot_kind(rr) or (' practice' if pc else '')})")
            rr = None
        if rr is not None:
            ic, pc = ints(rr), is_practice(rr)
            T["random_control"][tid] = C(f"control {ic['control_k']}, unedited {ic['unedited_k']}, edit {ic['edited_k']}"
                                         + (f" of {ic['n']}" if ic["n"] else ""),
                                         "; ".join(x for x in ("practice starts the edit was built from" if pc else None,
                                                               where_txt(rr)) if x) or None,
                                         u=ic["unedited_k"], r=ic["control_k"], e=ic["edited_k"], n=ic["n"],
                                         t=ic["teacher_k"], prac=pc or None, loc=describe(rr.get("edit_location")),
                                         fl=(f"{tid}, {loc_short(describe(rr.get('edit_location')))} edit" if alts
                                             and rr.get("edit_location") else f"{tid} {SHORT[tid]}")
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
                T["recovery"][tid] = C(f"{sgn(rp, False, 0)}%", "as reported" + ("; on practice starts" if pt else ""),
                                       prac=pt or None)
            else:
                T["recovery"][tid] = C(
                    f"{100 * (it['edited_k'] - it['unedited_k']) / (it['teacher_k'] - it['unedited_k']):.0f}%",
                    f"({it['edited_k']} − {it['unedited_k']}) / ({it['teacher_k']} − {it['unedited_k']}), computed here"
                    + ("; on practice starts" if pt else ""), prac=pt or None)
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
                                         for r in same])
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
              "preset. π0.5's earlier records, including the paper's, used the performance preset.",
    "failure_map": "Until its shared-slot runs land, π0.5's cells are earlier records: for each task the largest earlier "
                   "run, on its own slots, as the π0.5 track lists them with every other measurement. Its unedited rate "
                   "on T1 ranged from 8% to 21% across six start sets, so compare models on the shared slots.",
    "wording": "Reference-first describes the placement from the reference object's side, target-first from the moved "
               "object's side. In the earlier Same Goal, Different Words study (8 starting layouts per goal, no paired "
               "starts), reference-first wording lowered stable placement by 8.1 points for Cosmos3 Edge across 12 goals; "
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
             "Cosmos-Reason2-2B and the action pathway is the diffusion head.",
    "size": "A low-rank update is dense in its weight matrix: the 0.16% variant changes 7.4% of stored values; as factors "
            "it is 0.16% of the parameters.",
    "side_effects": "Each model's main edit, unchanged, on every suite task. π0.5's earlier records use the T1 edit: T1–T3 "
                    "on unseen starts (128 or 64), T4–T7 on 32 practice starts, R1 on 32 fresh starts; it was not run on "
                    "T8. Across fifteen other surveyed tasks it harmed none detectably.",
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
            c["d"] = "; ".join(x for x in (c.get("d"), f"earlier slots: {earlier['v']}") if x)
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
        "where": C(f"VLM {w['arms']['v1full']}, action expert {w['arms']['a1']}, head {w['arms']['h1']} of {w['n']}",
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
                    cell["d"] = "; ".join(x for x in (cell.get("d"), f"earlier record: {pi[tid]['v']}") if x)
            elif mid == "pi05":
                cell = dict(pi[tid], prov=True)
                if u:
                    cell["d"] = "; ".join(x for x in (cell.get("d"), f"shared slots, unedited: {u['v']}") if x)
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
                k0 = sorted(got)[0]
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


def cell_html(cell, mid, models):
    cls = {"ok": "c", "running": "c st-run", "na": "c st-na"}[cell["s"]] + (" prov" if cell.get("prov") else "")
    tag = (' <span class="tag">earlier slots</span>' if cell.get("prov") else "") + (
        ' <span class="tag">practice starts</span>' if cell.get("prac") else "")
    inner = f'<span class="v">{inline(cell["v"])}{tag}</span>'
    if cell.get("d"):
        inner += f'<span class="d">{inline(cell["d"])}</span>'
    return f'<td class="{cls}" data-model="{esc(models[mid]["name"])}">{inner}</td>'


def table_html(t, models):
    head = "".join(f'<th scope="col"><span class="mn">{esc(models[m]["name"])}</span>'
                   f'<span class="mk">{esc(models[m]["short"])}</span></th>' for m in ORDER)
    body = []
    for r in t["rows"]:
        cls = ' class="sub"' if r.get("sub") else ""
        body.append(f'<tr{cls}><th scope="row">{inline(r["label"])}</th>'
                    + "".join(cell_html(r["cells"][m], m, models) for m in ORDER) + "</tr>")
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
    return (table_html(t, models) + sources_html(t.get("sources"))
            + (f'<p class="tnote">{inline(t["note"])}</p>' if t.get("note") else ""))


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
        rows.append(("head", name + ((" (control, earlier slots)" if models["pi05"]["status"] == "measured" else
                                      " (control)") if mid == "pi05" else "")))
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
    top = 0.36 if brief else 0.62
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
            ax.text(j + 0.5, -0.5, models[m]["name"], ha="center", va="bottom", fontsize=L["FS"], weight="bold",
                    color=T["ink"])
            ax.text(j + 0.5, -0.16, models[m].get("fig_short") or models[m]["short"], ha="center", va="bottom",
                    fontsize=L["FS_S"] - 0.6,
                    color=T["mute"])
    for i, r in enumerate(fm):
        lab = f"{r['key']} {SHORT[r['key']]}" if r["key"] in SHORT else r["label"]
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
        fig.text(lx / L["W"], (0.12 + nrow * ch + 0.1) / H, "unedited\nsuccess", fontsize=L["FS_S"] - 0.6,
                 color=T["mute"], va="bottom", ha="left", linespacing=1.1)
    return fig


def fig_wording(tb, models, T, L):
    cbm = {m: [(c.get("fl") or r["label"], c["a"], c["b"], c["n"]) for r, c in v if "a" in c]
           for m, v in model_rows(tb, "wording").items()}
    rows = grouped(models, cbm, L)
    fig, ax, ys, H = rows_figure(rows, L)
    style_axes(ax, T, L, xlabel="success of the unedited model", fmt=lambda v: f"{v:g}%")
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
               xlabel="edited − unedited, points (95% CI)" if brief else
               "edited minus unedited success, points (95% CI)", fmt=lambda v: f"{v:+g}" if v else "0")
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
    cbm = {m: [(c.get("fl") or r["label"], c["u"], c["r"], c["e"], c["n"], c.get("t")) for r, c in v if "r" in c]
           for m, v in model_rows(tb, "random_control").items()}
    rows = grouped(models, cbm, L)
    k3 = 2.6
    fig, ax, ys, H = rows_figure(rows, L, hf={"row": k3})
    style_axes(ax, T, L, xlabel="success, % of starts", fmt=lambda v: f"{v:g}%")
    for (k, *rest), y in zip(rows, ys):
        if k == "head":
            put(fig, ax, L, L["lx"], y, rest[0], T, size=L["FS_H"], weight="bold")
        elif k == "run":
            put(fig, ax, L, L["lx2"], y, rest[0], T, color="mute", style="italic")
        else:
            lab, u, rnd, ed, n, te = rest[0]
            put(fig, ax, L, L["lx2"], y, lab, T)
            for off, val, col in ((-0.27, u, "grey"), (0.0, rnd, "dark"), (0.27, ed, "teal")):
                x = 100 * val / n
                if x > 0:
                    ax.add_patch(Rectangle((0, y + k3 * (off - L["bar"] / 2)), x, k3 * L["bar"], fc=T[col], ec="none",
                                           zorder=2))
                ax.text(x + 1.2, y + k3 * off, f"{val}", va="center", ha="left", fontsize=L["FS_S"], color=T["ink2"],
                        zorder=3)
            if te is not None:
                xt = 100 * te / n
                ax.plot([xt, xt], [y - k3 * 0.42, y + k3 * 0.42], color=T["ink"], lw=1.3, zorder=4)
                ax.text(xt + 1.0, y - k3 * 0.44, f"clear wording {te}", fontsize=L["FS_S"] - 0.6, color=T["ink2"],
                        va="bottom")
            put(fig, ax, L, L["W"] - L["right"] + 0.12, y, f"of {n} starts", T, color="mute", size=L["FS_S"])
    legend(fig, [("bar-grey", "unedited"), ("bar-dark", "random directions, same size"), ("bar-teal", "edit")], T, H, L)
    return fig


def fig_where(tb, models, T, L):
    from matplotlib.patches import Rectangle
    ref = WHERE_REF
    rows = [("head", f"π0.5 (control, earlier slots): T1, {ref['n']} unseen starts"),
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
    fig, ax, ys, H = rows_figure(rows, L, row=L["row"] * 1.08)
    style_axes(ax, T, L, xlabel="success, % of starts", fmt=lambda v: f"{v:g}%")
    for (k, *rest), y in zip(rows, ys):
        if k == "head":
            put(fig, ax, L, L["lx"], y, rest[0], T, size=L["FS_H"], weight="bold")
        elif k == "run":
            put(fig, ax, L, L["lx2"], y, rest[0], T, color="mute", style="italic")
        elif k == "bar":
            lab, val, n, col = rest
            put(fig, ax, L, L["lx2"], y, lab, T)
            if n:
                x = 100 * val / n
                if x > 0:
                    ax.add_patch(Rectangle((0, y - 0.3), x, 0.6, fc=T[col], ec="none", zorder=2))
                ax.text(x + 1.2, y, f"{val}", va="center", ha="left", fontsize=L["FS_S"], color=T["ink2"])
                put(fig, ax, L, L["W"] - L["right"] + 0.12, y, f"of {n}", T, color="mute", size=L["FS_S"])
        elif k == "tx":
            lab, t = rest
            put(fig, ax, L, L["lx2"], y, lab, T)
            n = t["n"]
            for arm, col, ms, mk in (("base", "grey", 1.0, "o"), ("a1", "dark", 0.85, "s"), ("p1", "dark", 0.85, "D"),
                                     ("v1full", "teal", 1.0, "o")):
                ax.plot([100 * t["arms"][arm] / n], [y], mk, ms=L["ms"] * ms, mfc=T[col], mec=T["ring"], mew=1.2,
                        zorder=3 + (arm == "v1full"))
            put(fig, ax, L, L["W"] - L["right"] + 0.12, y, f"{t['base']} → {t['arms']['v1full']} (VLM)", T, color="ink2",
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
    cw = (L["W"] - lw - 0.05) / 3
    ch = L["row"] * (1.25 if brief else 1.35)
    top = 0.36 if brief else 0.86
    bottom = 0.44 if brief else 0.62
    H = top + nrow * ch + bottom
    fig = plt.figure(figsize=(L["W"], H))
    ax = fig.add_axes([lw / L["W"], bottom / H, 3 * cw / L["W"], nrow * ch / H])
    ax.set_xlim(0, 3)
    ax.set_ylim(nrow, 0)
    ax.axis("off")
    for j, m in enumerate(ORDER):
        if brief:
            ax.text(j + 0.5, -0.2, FIGNAME[m], ha="center", va="bottom", fontsize=L["FS"], weight="bold", color=T["ink"])
        else:
            ax.text(j + 0.5, -0.78, models[m]["name"], ha="center", va="bottom", fontsize=L["FS"], weight="bold",
                    color=T["ink"])
            ed = (t.get("edit") or {}).get(m)
            hd = re.match(r"the [TR]\d edit(?: on practice starts)?", ed) if ed else None
            ed = hd.group(0) if hd else ed
            ax.text(j + 0.5, -0.16, ed or "edit: running", ha="center", va="bottom", fontsize=L["FS_S"] - 0.6,
                    color=T["mute"])
    grey = mix(T["empty"], T["grey"], 0.55)
    for i, r in enumerate(rows):
        lab = f"{r['key']} {SHORT[r['key']]}" if r["key"] in SHORT else r["label"]
        put(fig, ax, L, L["lx"], i + 0.5, lab, T)
        for j, m in enumerate(ORDER):
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
    fig.text((L["lx"] if brief else lw) / L["W"], (0.1 if brief else 0.2) / H, "* exact McNemar p < 0.05. Outlined: the task the edit was built "
             "for." if not brief else "* p < 0.05; outlined: the edit's own task", fontsize=L["FS_S"] - 0.6,
             color=T["mute"], ha="left", va="center")
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
    ("failure-map", "Unedited success on each shared task; stronger shading is higher. Each cell gives successes of starts."),
    ("wording-gap", "The unedited model under the original wording (grey) and the clear wording (ring), on the same "
                    "starts."),
    ("edit-gain", "Edited minus unedited success on held-out starts, paired by start, with the paired bootstrap 95% "
                  "interval. Teal: distilled edits; neutral: edits the blind editors designed themselves."),
    ("random-control", "The edit against a random edit of the same layers, rank and singular values, on the same "
                       "unseen starts."),
    ("where-to-edit", "The same distillation in three places of π0.5, scored on one shared set of unseen starts, then on "
                      "other tasks; the NVIDIA models' locations appear as their runs finish."),
    ("side-effects", "Each model's main edit, unchanged, on every suite task: edited minus unedited successes on the same "
                     "starts. Teal: a gain; near-black (near-white in dark mode): a loss (exact McNemar p < 0.05); grey: no "
                     "detectable change; the outlined "
                     "cell is the task the edit was built for."),
]


# ---------------------------------------------------------------- clips (MP4s in site/cosmos/video/, <= 2 MB each)
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
        s = (f"Held-out start {slot}, never used to build the edit: the unedited policy on top, the edited policy "
             f"below, both given the original wording. Unedited: {outcome(a)}; edited: {outcome(b)}.")
        c = table_cell(tables, "edit_gain", tid, mid)
        if c and not c.get("prac"):
            s += (f" Over all {c['n']} held-out starts{' so far' if c['n'] < 64 else ''}: {c['a']} → {c['b']} successes"
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
            s += (f" Over all {c['n']} {'practice' if c.get('prac') else 'held-out'} starts: {c['a']} → {c['b']}"
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
        s = (f"{lead} Held-out start {c['slot']} on π0.5’s earlier slots, the same scene and noise seeds in every "
             f"panel. Top to bottom: {panels}. An earlier record ({hw}), {c.get('speed', 3):g}× speed.")
        return s, [f"{PI05_CLIPS[0]}:{PI05_CLIPS[1]}"]
    return f


def fixed_cap(text):
    return lambda tables: (text, [])


CLIPS = [
    dict(stem="cosmos_t8_wording", group="demo", title="Cosmos3 Edge Policy, T8: the wording gap",
         cap=cap_wording(X_COSMOS, "T8", 30000, "T8-base-s30000", "T8-wtf-s30000", "cosmos", "T8",
                         "the original, reference-first wording", "the clearer, target-first wording")),
    dict(stem="groot_t1_wording", group="demo", title="GR00T N1.7, T1: the wording gap",
         cap=cap_wording(X_GROOT, "T1", 30003, "fm-T1", "wd-T1-expl", "groot", "T1",
                         "the original wording", "the explicit two-step wording")),
    dict(stem="groot_t1_edits", group="demo", title="GR00T N1.7, T1: two edits that make things worse",
         cap=cap_panels(X_GROOT, 30004, [("fm-T1", "unedited"), ("ed-bb-T1", "edited in the Cosmos-Reason2 backbone"),
                                         ("ed-dit-T1", "edited in the diffusion action head")],
                        "Both edits distil the explicit two-step wording; neither picks up the cube.", "groot", "T1")),
    dict(stem="t1_same_start", group="demo", title="T1, one start, two NVIDIA models", cap=cap_same_start(30000)),
    dict(stem="pi05_t1_edit_control_teacher", group="demo", title="π0.5, T1: the edit against its controls",
         cap=cap_pi05("t1_v2mi-f_vs_control_vs_teacher_90000.mp4",
                      "The unedited policy, a random edit of the same size, the distilled edit and the clearer wording.")),
    dict(stem="pi05_t8_edit_control", group="demo", title="π0.5, T8: the edit against a random edit",
         cap=cap_pi05("t8_s4r-grp3_vs_control_92001.mp4",
                      "The reference-first wording throughout; the edit distils the target-first wording.")),
    dict(stem="cosmos_before_after", group="before", title="Cosmos3 Edge Policy, T8",
         cap=cap_edit_heldout(X_COSMOS, 96003, "T8-base-s96000", "T8-Et8und-s96000", "cosmos", "T8")),
    dict(stem="groot_before_after", group="before", title="GR00T N1.7", mid="groot",
         cap=fixed_cap("Unedited and edited, on a held-out start where the edit helped.")),
    dict(stem="cosmos_t5_side_effect", group="transfer",
         title="Cosmos3 Edge Policy, T5 Mustard into the left bin: a start the T8 edit breaks",
         cap=cap_side(X_COSMOS, 30001, "T5-base-s30000", "T5-Et8und-s30000", "cosmos", "T5", "T8")),
    dict(stem="pi05_t2_fixed", group="transfer", title="π0.5, T2 Pick up the eyeglasses: a start the T1 edit fixes",
         cap=cap_pi05("t2_v2mi-f_side_effect_fixed_90000.mp4", "The T1 edit, unchanged, on a task it was not built for.")),
    dict(stem="pi05_t3_fixed", group="transfer", title="π0.5, T3 Cube left of the bowl: a start the T1 edit fixes",
         cap=cap_pi05("t3_v2mi-f_side_effect_fixed_90000.mp4", "The T1 edit, unchanged, on a task it was not built for.")),
    dict(stem="pi05_t2_broke", group="transfer", title="π0.5, T2 Pick up the eyeglasses: a start the T1 edit breaks",
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


def clip_html(c, info, tables):
    i = info[c["stem"]]
    caption, srcs = c["cap"](tables)
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
    """A finished track's placeholder is dropped: its clip will not come."""
    return "".join(clip_html(c, info, tables) for c in CLIPS if c["group"] == group
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


STATUS = {"measured": "earlier records", "running": "running", "partial": "first results", "final": "complete"}


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
            f'π0.5’s earlier records used the performance preset. <a href="#render">Rendering sensitivity</a></p>')


def brief_lines(P, rc, models, cross, tables=None):
    e, c, s4ho = P["edit"], P["ctl"], P["s4ho"]
    lines = [
        "<li><b>Method.</b> When a clearer wording of the same goal makes a robot model succeed, copy that wording's "
        "internal effect into the weights in closed form, one least-squares solve per layer: no training, and the "
        "instruction stays the same.</li>",
        f"<li><b>π0.5, the control (earlier records).</b> Distillation fixed its two language failures: T1 {e['base']} → "
        f"{e['cand']} of {e['n']} unseen starts ({e['fixed']} fixed, {e['broke']} broke, p = {fp(e['p'])}) and T8 "
        f"{s4ho['base']} → {s4ho['cand']} of {s4ho['n']}; a random edit of the same size did nothing ({c['rand']} vs "
        f"{c['base']} of {c['n']}); the T1 edit also lifted T2 and T3; no edit helped T4–T7.</li>",
    ]
    parts = []
    for mid in ("cosmos", "groot"):
        g, u = rc[mid]["edit_gain"], rc[mid]["failure_map"]
        if g:
            parts.append(f"{esc(models[mid]['name'])}: " + "; ".join(
                f"{tid} edit {esc(g[tid]['v'])}" + ((" practice starts (the ones it was built from)" if g[tid].get("b", 0) < g[tid].get("a", 0)
                                                    else " practice starts (the ones it was built from; unseen starts running)")
                                                   if g[tid].get("prac") else " unseen starts") for tid in sorted(g)))
        elif u:
            parts.append(f"{esc(models[mid]['name'])}: unedited rates in, edits running")
        else:
            parts.append(f"{esc(models[mid]['name'])}: running")
    rn = ""
    c = next((t for t in tables or [] if t["id"] == "render"), {"rows": [{"cells": {}}]})["rows"][0]["cells"].get("groot")
    if c and c["s"] == "ok":
        rn = (f" First finding: GR00T N1.7 succeeded on {c['a']} of {c['n']} starts under RoboLab's performance render "
              f"preset and {c['b']} of {c['n']} under the default preset, so all three are compared under the default.")
    lines.append(f"<li><b>NVIDIA models.</b> {'; '.join(parts)}.{rn} Same tasks, slots, metrics and statistics; this "
                 "brief refreshes as runs finish.</li>")
    return "".join(lines)


# ---------------------------------------------------------------- TL;DR and section overview (top of the page)
RUNNING = "{running}"   # rendered as the italic "running" marker; never a number
SIG = 0.05              # "lifted" / "worse": exact McNemar p below this, paired by start
TLDR_ORDER = ["pi05", "cosmos", "groot"]   # the control's established result first, then the NVIDIA models


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


def tldr(tables, models):
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
        items.append(dict(id=mid, lead=m["name"], kind=MODELS[mid]["short"],
                          tag="earlier records" if prov else None, text=" ".join(bits)))

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
        parts.append(t + ".")
    if waiting:
        parts.append(f"{and_list(waiting)}: {RUNNING}.")
    items.append(dict(id="side-effects", lead="Side effects", kind=None, tag=None, text=" ".join(parts)))

    for it in items:  # keep "p = 0.12" and "9 → 16" on one line
        it["text"] = re.sub(r"\s*→\s*", "\u00a0→\u00a0", it["text"].replace("p = ", "p\u00a0=\u00a0"))
    early = [it["lead"] for it in items if it.get("tag")]
    note = ("Lifted and worse mean an exact McNemar p < 0.05 on the same starts, edited against unedited. "
            + (f"{and_list(x + '’s' for x in early)} numbers are earlier records, each on its own slots and number of "
               "starts; the shared-slot runs "
               "replace them as they land." if early else "Every number here is from the shared slots."))
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
    """Every section (h2) and subsection (h3) of the article in page order: the contents list follows the page.
    Generated regions are skipped, so headings inside them (the model cards) never enter the list."""
    page = re.sub(r"<!-- gen:([a-z0-9:-]+) -->.*?<!-- /gen:\1 -->", "", page, flags=re.S)
    a, b = article(page)
    secs = []
    pat = re.compile(r'<section id="([a-z0-9-]+)"[^>]*>\s*<h2[^>]*>(.*?)</h2>|<h3 id="([a-z0-9-]+)"[^>]*>(.*?)</h3>', re.S)
    for m in pat.finditer(page[a:b]):
        sid, title, level = (m.group(1), m.group(2), 2) if m.group(1) else (m.group(3), m.group(4), 3)
        title = html.unescape(re.sub(r"<[^>]+>", "", title)).strip()
        secs.append(dict(id=sid, title=title, level=level))
    return secs


def toc_html(secs, cls):
    out, open_sub = [f'<ol class="{cls}">'], False
    for i, x in enumerate(secs):
        if x["level"] == 2:
            if open_sub:
                out.append("</ol>")
                open_sub = False
            if i:
                out.append("</li>")
            out.append(f'<li><a href="#{esc(x["id"])}">{esc(x["title"])}</a>')
        else:
            if not open_sub:
                out.append("<ol>")
                open_sub = True
            out.append(f'<li><a href="#{esc(x["id"])}">{esc(x["title"])}</a></li>')
    out.append(("</ol>" if open_sub else "") + "</li></ol>")
    return "".join(out)


WPM = 230


def read_minutes(page):
    """Words a reader meets in the article (collapsed tables and code excluded), at 230 words a minute."""
    a, b = article(page)
    art = re.sub(r'<details class="tdetails.*?</details>', " ", page[a:b], flags=re.S)
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
            task = f"{task} {SHORT[task]}" if task in SHORT else re.sub(r"(?<!^)(?=[A-Z])", " ", task).lower()
            bits.append(f"{task}: {k} of {n} with NVIDIA’s, {m.group(1)} of {m.group(2)} with ours")
        if bits:
            out.append(f"<p class=\"note\"><strong>Adapter check, {esc(models[mid]['name'])}.</strong> NVIDIA’s "
                       f"unmodified server and RoboLab client against this study’s adapter, default render preset: "
                       f"{esc('; '.join(bits))}. The zeros in the failure map are the model’s, not the adapter’s.</p>")
    return "".join(out)


def conclusion_html(tables, models):
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
        print(f"wrote {pdf.relative_to(ROOT)} ({pages} page{'s' if pages != 1 else ''}, zoom step {fit}, {len(b) // 1024} KB)")
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
    args = ap.parse_args()

    if not args.no_fetch:
        for _, branch, _ in SOURCES:
            r = subprocess.run(["git", "-C", str(ROOT), "fetch", "--quiet", "origin",
                                f"+refs/heads/{branch}:refs/remotes/origin/{branch}"], capture_output=True, text=True)
            print(f"  fetch {branch}: {'ok' if r.returncode == 0 else 'not on origin yet'}")

    S = Sources()
    P = build_pi05(S)
    meta = {}
    cross = read_cross(args, meta)

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
            m["short"] = "OpenPI VLA, control · earlier slots"
            m["fig_short"] = "control · earlier slots"   # fits the figure's column head
        models[mid] = m

    tables, rc, prov, rcp = build_tables(P, cross)
    fmrows = next(t for t in tables if t["id"] == "failure_map")["rows"]
    pv = sum(1 for r in fmrows if r["cells"]["pi05"].get("prov"))
    fresh = [r["key"] for r in fmrows if r["cells"]["pi05"].get("s") == "ok" and not r["cells"]["pi05"].get("prov")]
    if 0 < pv < len(fmrows):   # some fresh shared-slot cells, the rest earlier records: name the fresh ones in the head
        models["pi05"]["fig_short"] = (f"control · shared slots: {', '.join(fresh)}" if len(fresh) <= 2
                                       else f"control · {pv}/{len(fmrows)} earlier")
    summary = summary_table(P, tables, cross, rc)
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
        "tldr": tldr(tables, models),
        "sections": page_sections(page),
        "summary": summary,
        "tables": tables,
        "cross_rows": {mid: [{k: v for k, v in r.items() if k != "_extra"} | ({"extra": r["_extra"]} if r["_extra"] else {})
                             for r in cross[mid]["rows"]] for mid in cross},
        "sources": {"pi05_earlier_records": dict(sorted(S.used.items())), **meta},
        "method": "Exact two-sided McNemar on fixed vs broke pairs; paired bootstrap 95% interval of the difference "
                  "(20,000 multinomial draws over the four paired outcome types, seed 0); Wilson 95% intervals.",
    }
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
    data["updated"] = old["updated"] if (not changed and old.get("updated")) else now.strftime("%Y-%m-%dT%H:%MZ")
    upd = dt.datetime.strptime(data["updated"], "%Y-%m-%dT%H:%MZ")
    data["updated_text"] = f"{upd.day} {upd:%b %Y}, {upd:%H:%M} UTC"
    data["status_text"] = " · ".join(f"{models[m]['name']}: {STATUS[models[m]['status']]}" for m in ORDER)
    mpath.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n")
    print(f"wrote {mpath.relative_to(ROOT)} ({'changed' if changed else 'unchanged'}, {h})")

    if args.no_figures:
        alts = {n: c for n, c in FIGS}
    else:
        alts = draw_figures(tables, models)
        print(f"checked {len(FIGS) * 4 + len(BRIEF_FIGS)} figure files in {FIG.relative_to(ROOT)}/")

    idx = OUT / "index.html"
    page = replace_region(page, "tldr", tldr_html(data["tldr"]))
    page = replace_region(page, "tldr-note", esc(data["tldr"]["note"]))
    page = replace_region(page, "toc", toc_html(data["sections"], "toc-l1"))
    page = replace_region(page, "toc-m", toc_html(data["sections"], "toc-l1"))
    page = replace_region(page, "stamp", esc(f"{data['status_text']} · updated {data['updated_text']}"))
    page = replace_region(page, "models", models_html(models))
    page = replace_region(page, "render-note", render_note(tables))
    page = replace_region(page, "conclusion", conclusion_html(tables, models))
    page = replace_region(page, "where-note", where_note(tables, models))
    page = replace_region(page, "adapter-note", adapter_note(cross, models))
    page = replace_region(page, "table:summary", table_html(summary, models))
    for t in tables:
        page = replace_region(page, f"table:{t['id']}", block_html(t, models))
    prac_edit = any(r["cells"][m].get("prac") for t in tables if t["id"] == "edit_gain" for r in t["rows"] for m in ORDER)
    for name, cap in FIGS:
        if name == "failure-map" and 0 < pv < len(fmrows):
            cap += (f" π0.5’s cells are earlier records on its own slots, except {and_list(fresh)}, which "
                    f"{'is' if len(fresh) == 1 else 'are'} on the shared slots.")
        if name == "edit-gain" and prac_edit:
            cap += " Hollow, dashed: scored on the practice starts the edit was built from, not on unseen starts."
        page = replace_region(page, f"fig:{name}", figure_html(name, alts[name], cap))
    clips = prepare_clips()
    for group in ("demo", "before", "transfer"):
        page = replace_region(page, f"clips:{group}", clips_section(group, clips, tables, models))
    page, nfig = number_figures(page)
    mins, words = read_minutes(page)
    page = replace_region(page, "readtime", f"{mins} min read")
    idx.write_text(page)
    print(f"  {nfig} numbered figures and clips; {words:,} words, {mins} min read")
    print(f"wrote {idx.relative_to(ROOT)}")

    br = OUT / "brief.html"
    btxt = old_brief = br.read_text()
    btxt = replace_region(btxt, "stamp", esc(f"Draft for discussion · {data['status_text']} · {data['updated_text']}"))
    btxt = replace_region(btxt, "summary-lines", brief_lines(P, rc, models, cross, tables))
    btxt = replace_region(btxt, "gif", gif_html(clips))
    btxt = replace_region(btxt, "table:summary", table_html(summary, models))
    caps = dict(FIGS)
    for name in BRIEF_FIGS:
        btxt = replace_region(btxt, f"fig:{name}", figure_html(name, alts[name], caps[name], pdf=True))
    br.write_text(btxt)
    print(f"wrote {br.relative_to(ROOT)}")

    pdf = OUT / "weightsmith-cosmos-brief.pdf"
    inputs_changed = btxt != old_brief or any(n.startswith("brief") for n in CHANGED)
    if not args.no_pdf and (inputs_changed or args.force or not pdf.exists()):
        print_pdf(br, pdf)
    print(f"  figures rewritten: {len(CHANGED)}" + (f" ({', '.join(sorted(CHANGED)[:6])}{', ...' if len(CHANGED) > 6 else ''})"
                                                     if CHANGED else ""))


if __name__ == "__main__":
    sys.exit(main())
