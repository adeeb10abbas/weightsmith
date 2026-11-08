#!/usr/bin/env python3
"""Write the owsp acts specs of the where-to-edit ablation (project where-to-edit).

Decisions are the paper's: the 6 premature releases distilled by every arm (c2-multi items 0-5 of
projects/cube-then-banana) and the 49 offline-check decisions of capture ctb-ph there
(0-20 premature releases, 21-29 correct carries, 30-36 phase-2 approach, 37-41 phase-2 banana grasp,
42-48 phase-2 carry). Episodes base.base.s20000-s20063 were copied byte for byte into this project.
"""
import json
from pathlib import Path

ORIG = "Put the cube then the banana in the bowl"
EXPL = "Put the cube in the bowl then put the banana in the bowl"
SIX = [("base.base.s20001", 5), ("base.base.s20007", 5), ("base.base.s20009", 4),
       ("base.base.s20022", 4), ("base.base.s20023", 4), ("base.base.s20040", 4)]
D49 = [("base.base.s20001", 5), ("base.base.s20007", 5), ("base.base.s20009", 4), ("base.base.s20012", 5),
       ("base.base.s20020", 12), ("base.base.s20022", 4), ("base.base.s20023", 4), ("base.base.s20024", 4),
       ("base.base.s20028", 4), ("base.base.s20030", 7), ("base.base.s20032", 7), ("base.base.s20040", 4),
       ("base.base.s20043", 5), ("base.base.s20044", 4), ("base.base.s20046", 5), ("base.base.s20048", 5),
       ("base.base.s20049", 7), ("base.base.s20051", 6), ("base.base.s20055", 5), ("base.base.s20060", 6),
       ("base.base.s20063", 4),                                                   # 0-20 premature releases
       ("base.base.s20011", 4), ("base.base.s20029", 5), ("base.base.s20030", 5), ("base.base.s20038", 8),
       ("base.base.s20039", 6), ("base.base.s20042", 5), ("base.base.s20044", 7), ("base.base.s20050", 9),
       ("base.base.s20061", 5),                                                   # 21-29 correct carries
       ("base.base.s20011", 9), ("base.base.s20030", 26), ("base.base.s20038", 14), ("base.base.s20042", 9),
       ("base.base.s20044", 11), ("base.base.s20050", 13), ("base.base.s20061", 9),  # 30-36 phase-2 approach
       ("base.base.s20011", 12), ("base.base.s20030", 28), ("base.base.s20042", 12), ("base.base.s20050", 15),
       ("base.base.s20061", 12),                                                  # 37-41 phase-2 banana grasp
       ("base.base.s20011", 14), ("base.base.s20030", 30), ("base.base.s20038", 19), ("base.base.s20042", 14),
       ("base.base.s20044", 16), ("base.base.s20050", 17), ("base.base.s20061", 15)]  # 42-48 phase-2 carry
assert len(D49) == 49


def items(lst):
    return [{"episode": e, "k": k} for e, k in lst]


# Full action-expert trace at the 6 distilled decisions: all 18 layers, all 15 action tokens, all 10 denoise steps.
FULL = {"vlm_layers": [], "action_layers": list(range(18)), "action_tokens": list(range(15)),
        "denoise_steps": list(range(10))}
# Offline check: final action chunks only (plus the cheap last-layer trace).
LIGHT = {"vlm_layers": [], "action_layers": [17], "action_tokens": "mean", "denoise_steps": [0]}


def main(out=Path(__file__).resolve().parent / "specs"):
    out.mkdir(exist_ok=True)
    specs = {
        "a-orig": {"cand": "base", "prompt": ORIG, "items": items(SIX), **FULL},
        "a-expl": {"cand": "base", "prompt": EXPL, "items": items(SIX), **FULL},
        "va-orig": {"cand": "v1-full", "prompt": ORIG, "items": items(SIX), **FULL},
        "a1-trace": {"cand": "a1", "prompt": ORIG, "items": items(SIX), **FULL},          # A1's own action-expert trace
        "va-trace": {"cand": "va", "prompt": ORIG, "items": items(SIX), **FULL},
        "chk-base": {"cand": "base", "prompt": ORIG, "items": items(D49), **LIGHT},
        "chk-teacher": {"cand": "base", "prompt": EXPL, "items": items(D49), **LIGHT},
    }
    for c in ["v1-full", "v1-img816", "p1-blind", "a1", "a1-nogate", "a1-late", "h1", "h1-grip", "va"]:
        specs[f"chk-{c}"] = {"cand": c, "prompt": ORIG, "items": items(D49), **LIGHT}
    for k, v in specs.items():
        (out / f"{k}.json").write_text(json.dumps({"id": k, **v}, indent=1) + "\n")
    print("wrote", len(specs), "specs to", out)


if __name__ == "__main__":
    main()
