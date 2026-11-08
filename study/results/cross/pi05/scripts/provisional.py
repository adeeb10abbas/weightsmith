"""PROVISIONAL pi0.5 control (Track C, Step 0): existing pi0.5 results, each with its own slots, n and machine.

Nothing here was run for the cross-model comparison. Every number is read from a result file already in this
repository (or, for the earlier side-effect runs, from prior_pairs_ali-ms.json, recomputed from ali-ms's episode
records by prior_pairs.py), and the statistics are recomputed with weightsmith.stats (exact McNemar, Wilson,
paired bootstrap with 20,000 draws, seed 0). Writes results/provisional.json; analyze.py merges its records into
../metrics_pi05.json next to the fresh ones.

    python3 study/results/cross/pi05/scripts/provisional.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]          # study/results/cross/pi05
REPO = HERE.parents[3]
sys.path.insert(0, str(REPO))
from weightsmith import stats  # noqa: E402

RES = REPO / "study/results"
MODEL = "pi05_droid_jointpos (OpenPI pi0.5 DROID, joint position; gs://openpi-assets-simeval/pi05_droid_jointpos)"
HW_OLD = "ali-ms, RTX 3090 (weightsmith runtime; earlier studies)"
TASKS = {
    "T1": ("RubiksCubeThenBananaTask", "Put the cube then the banana in the bowl"),
    "T2": ("PickGlassesTask", "Pick up the eye glasses"),
    "T3": ("RubiksCubeLeftOfBowlTask", "Put the rubiks cube to the left of the bowl"),
    "T4": ("StackYellowOnRedTask", "Stack the yellow block on the red block"),
    "T5": ("MustardInLeftBinTask", "Put the mustard in the left bin"),
    "T6": ("Stack3RubiksCubeTask", "Stack the rubiks cubes in a tower"),
    "T7": ("PickUpGreenObjectTask", "Pick up the green vegetable block"),
    "T8": ("MustardRightOfRaisinRFTask",
           "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."),
    "R1": ("RubiksCubeTask", "Put the cube in the bowl"),
}
CODE = {t: c for c, (t, _) in TASKS.items()}
EXPLICIT = "Put the cube in the bowl then put the banana in the bowl"
TF = "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
V2MI = {"name": "v2mi-f", "fingerprint": "63db0c52a51306c9",
        "location": "VLM MLP down-projections language.blocks.00-16.mlp.linear (17 tensors), rank <= 72 per layer",
        "values_changed": 450468837, "values_changed_note": "450,468,837 stored values (13.4% of 3,353,433,872)"}
ABL = {"name": "abl-imgl8", "fingerprint": "5bf8c4329c1e416a",
       "location": "VLM MLP down-projections language.blocks.08-16.mlp.linear (9 tensors), image-cell keys only, "
                   "rank <= 32 per layer",
       "values_changed": 248057499,
       "values_changed_note": "248,057,499 stored values (7.4%); 5.3M numbers as low-rank factors (0.16%)"}
GRP3 = {"name": "s4r-grp3", "fingerprint": "9e6f2378e28be6a3",
        "location": "VLM MLP down-projections language.blocks.08-16.mlp.linear (9 tensors), 3 key groups "
                    "(approach, grasp, transport), rank <= 96 per layer",
        "values_changed": None, "values_changed_note": "15.9M numbers as low-rank factors (0.48% of parameters); "
                                                       "dense count not recorded"}


def load(p):
    return json.loads((RES / p).read_text())


def pct(x):
    return round(100 * x, 1)


def wil(k, n):
    lo, hi = stats.wilson(k, n)
    return [pct(lo), pct(hi)]


def paired_from_counts(n, ref, cand, fixed, broke):
    t = stats.PairedTable.from_counts(n, ref, cand, fixed, broke)
    lo, hi = stats.paired_bootstrap(t)
    return {"n": n, "unedited_k": ref, "edited_k": cand, "fixed": fixed, "broke": broke,
            "p": stats.mcnemar_exact(fixed, broke),
            "ci": {"wilson95_unedited_pct": wil(ref, n), "wilson95_edited_pct": wil(cand, n),
                   "diff_pts": pct(t.difference), "bootstrap95_diff_pts": [pct(lo), pct(hi)]}}


def paired_from_slots(per_slot):
    """per_slot: {slot: [ref, cand]} -> stats.compare."""
    ref = {int(s): bool(v[0]) for s, v in per_slot.items()}
    cand = {int(s): bool(v[1]) for s, v in per_slot.items()}
    c = stats.compare(ref, cand)
    t = c.table
    lo, hi = c.difference_ci
    return {"n": t.n, "unedited_k": t.ref_success, "edited_k": t.cand_success, "fixed": t.fixed, "broke": t.broke,
            "p": c.p_value,
            "ci": {"wilson95_unedited_pct": wil(t.ref_success, t.n), "wilson95_edited_pct": wil(t.cand_success, t.n),
                   "diff_pts": pct(t.difference), "bootstrap95_diff_pts": [pct(lo), pct(hi)]},
            "paired_slots": [min(ref), max(ref)]}


def record(experiment, code, *, wording, slots, source, notes, edit=None, control_k=None, teacher_k=None,
           recovery_pct=None, transfer=None, retention=None, hw=HW_OLD, **paired):
    task, _ = TASKS[code] if code in TASKS else (code, None)
    return {"experiment": experiment, "status": "provisional", "model": MODEL, "hardware": hw, "task": task,
            "code": code, "wording_pair": wording, "slots": slots,
            "n": paired.get("n"), "unedited_k": paired.get("unedited_k"), "edited_k": paired.get("edited_k"),
            "fixed": paired.get("fixed"), "broke": paired.get("broke"), "p": paired.get("p"),
            "ci": paired.get("ci"), "control_k": control_k, "teacher_k": teacher_k, "recovery_pct": recovery_pct,
            "transfer": transfer, "retention": retention,
            "edit": None if edit is None else {"name": edit["name"], "fingerprint": edit["fingerprint"]},
            "edit_location": None if edit is None else edit["location"],
            "values_changed": None if edit is None else edit["values_changed"],
            "values_changed_note": None if edit is None else edit["values_changed_note"],
            "source": source, "notes": notes}


def main():
    recs, failure = [], []

    # ------------------------------------------------------------------ unedited success, every source we have
    ctrl = load("ctrl/summary.json")
    key_of = {"ctb": "T1", "glasses": "T2", "left": "T3", "yellowred": "T4", "mustard": "T5", "stack3": "T6",
              "flatbox": "T7"}
    for key, code in key_of.items():
        b = ctrl[key]["dev"]["base"]
        failure.append({"code": code, "k": b["success"], "n": b["n"], "slots": [70000, 70031],
                        "source": "study/results/ctrl/summary.json (controlled study, dev slots)"})
        for h in ctrl[key].get("heldout", []):
            failure.append({"code": code, "k": h["base_success"], "n": h["paired_n"], "slots": h["slots"],
                            "source": f"study/results/ctrl/heldout (base arm of {h['cand']} held-out)"})
    for f, code in (("opus/heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json", "T1"),
                    ("opus/heldout/cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json", "T1"),
                    ("opus/heldout/thr-pickglasses__ctb-v2mi-PickGlassesTask-90000-64.json", "T2"),
                    ("opus/heldout/tx-rubikscubeleftofbowl__ctb-v2mi-RubiksCubeLeftOfBowlTask-90000-64.json", "T3")):
        s = load(f)["summary"]
        failure.append({"code": code, "k": s["base_success"], "n": s["paired_n"], "slots": s["slots"],
                        "source": f"study/results/{f} (base arm)"})
    wh = load("opus/where/results/heldout_analysis.json")
    failure.append({"code": "T1", "k": wh["arms"]["base"]["success"], "n": wh["arms"]["base"]["n"],
                    "slots": wh["slot_range"], "source": "study/results/opus/where/results/heldout_analysis.json"})
    for f, code in (("glasses", "T2"), ("left", "T3"), ("cube", "R1")):
        d = load(f"opus/where/results/transfer_{f}_analysis.json")
        failure.append({"code": code, "k": d["arms"]["base"]["success"], "n": d["arms"]["base"]["n"],
                        "slots": d["slot_range"], "source": f"study/results/opus/where/results/transfer_{f}_analysis.json"})
    sc = load("sgdw/results/screen_s4r_tf_vs_rf.json")
    failure.append({"code": "T8", "k": sc["success_b"], "n": sc["paired_n"], "slots": sc["slots"],
                    "source": "study/results/sgdw/results/screen_s4r_tf_vs_rf.json (RF arm)"})
    hp = load("sgdw/results/heldout_pairs_92000-64.json")
    failure.append({"code": "T8", "k": hp["arms"]["base"]["success"], "n": hp["arms"]["base"]["n"],
                    "slots": [92000, 92063], "source": "study/results/sgdw/results/heldout_pairs_92000-64.json"})
    prior = json.loads((HERE / "results/prior_pairs_ali-ms.json").read_text())["records"]
    seen = set()
    for r in prior:
        code = CODE.get(r["task"])
        key = (r["project"], tuple(r["slots"]))
        if code and key not in seen and r["project"] != "cube-then-banana":
            seen.add(key)
            failure.append({"code": code, "k": r["unedited_k"], "n": r["n"], "slots": r["slots"],
                            "source": f"ali-ms projects/{r['project']} base.base episodes (prior_pairs_ali-ms.json)"})
    for f in failure:
        f["wilson95_pct"] = wil(f["k"], f["n"])
    # one headline unedited rate per task: the largest n (ties: the later-listed held-out source)
    for code in TASKS:
        rows = [f for f in failure if f["code"] == code]
        if not rows:
            continue
        best = max(rows, key=lambda f: f["n"])
        others = "; ".join(f"{f['k']}/{f['n']} on {f['slots'][0]}-{f['slots'][1]}" for f in rows if f is not best)
        recs.append(record("failure_map", code, wording=[TASKS[code][1]], slots=best["slots"],
                           source=best["source"], n=best["n"], unedited_k=best["k"],
                           ci={"wilson95_pct": best["wilson95_pct"]},
                           notes=("unedited policy, task's default instruction. Other unedited measurements: "
                                  + (others or "none")
                                  + ". Different slot blocks and runs differ by more than their Wilson intervals "
                                    "suggest (identical reruns flip ~30% of pairs).")))

    # ------------------------------------------------------------------ T1: the paper's edits, control, teacher
    v = load("opus/heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json")
    a = load("opus/heldout/cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json")
    ctl = load("opus/controls/ctl1-90000-64.analysis.json")
    cs = ctl["success"]
    g = load("opus/heldout/thr-pickglasses__ctb-v2mi-PickGlassesTask-90000-64.json")
    lft = load("opus/heldout/tx-rubikscubeleftofbowl__ctb-v2mi-RubiksCubeLeftOfBowlTask-90000-64.json")
    tcube = load("opus/where/results/transfer_cube_analysis.json")
    tc = {c["cand"]: c for c in tcube["comparisons"]}["v1full"]
    rec_v = (cs["edit_full"] - cs["base_A"]) / (cs["teacher"] - cs["base_A"])
    rec_a = (cs["edit_016"] - cs["base_B"]) / (cs["teacher"] - cs["base_B"])
    pv = paired_from_slots(v["per_slot"])
    recs.append(record(
        "edit_target", "T1", edit=V2MI, wording=[TASKS["T1"][1], EXPLICIT], slots=v["summary"]["slots"],
        source="study/results/opus/heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json; "
               "study/results/opus/controls/ctl1-90000-64.analysis.json",
        control_k={"k": cs["rand"], "n": 64, "slots": [90000, 90063],
                   "what": "random-direction update with v2mi-f's layers, rank and singular values (seed 0)"},
        teacher_k={"k": cs["teacher"], "n": 64, "slots": [90000, 90063], "wording": EXPLICIT},
        recovery_pct=round(100 * rec_v, 1),
        transfer={"T2": {**paired_from_slots(g["per_slot"]), "what": "same file on PickGlassesTask, held-out"},
                  "T3": {**paired_from_slots(lft["per_slot"]), "what": "same file on RubiksCubeLeftOfBowlTask, "
                                                                       "held-out"}},
        retention={"R1": {"n": tc["n"], "unedited_k": tc["ref_succ"], "edited_k": tc["cand_succ"],
                          "fixed": tc["fixed"], "broke": tc["broke"], "p": tc["mcnemar_p"],
                          "slots": tcube["slot_range"]}},
        notes=(f"held-out, weightsmith/heldout.py (order alternating by half). Recovery % on slots 90000-90063, where "
               f"the control and teacher ran: (edit {cs['edit_full']} - unedited {cs['base_A']}) / (teacher "
               f"{cs['teacher']} - unedited {cs['base_A']}). Teacher vs edit there: 11 teacher-only, 4 edit-only, "
               "p = 0.12. Practice slots 20000-20031: 3/32 -> 25/32 (22 fixed, 0 broke)."), **pv))
    pa = paired_from_slots(a["per_slot"])
    recs.append(record(
        "edit_target", "T1", edit=ABL, wording=[TASKS["T1"][1], EXPLICIT], slots=a["summary"]["slots"],
        source="study/results/opus/heldout/cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json; "
               "study/results/opus/controls/ctl1-90000-64.analysis.json",
        control_k={"k": cs["rand"], "n": 64, "slots": [90000, 90063], "what": "random-direction control of v2mi-f"},
        teacher_k={"k": cs["teacher"], "n": 64, "slots": [90000, 90063], "wording": EXPLICIT},
        recovery_pct=round(100 * rec_a, 1),
        notes=(f"held-out, heldout.py, against its own unedited run on the same slots. Recovery % on 90000-90063: "
               f"(edit {cs['edit_016']} - unedited {cs['base_B']}) / (teacher {cs['teacher']} - unedited "
               f"{cs['base_B']}). On slots 94016-94079 (where-to-edit) it scored 41/64 against v2mi-f's 55/64 "
               "(5 / 19, p = 0.0066)."), **pa))

    # where to edit (T1, held-out 94016-94079, one shared unedited run)
    comps = {c["cand"]: c for c in wh["comparisons"] if c["ref"] == "base"}
    where = {"VLM (v2mi-f)": "v1full", "VLM image cells (abl-imgl8)": "v1img816", "action expert (A1)": "a1",
             "action head, least-norm (H1)": "h1", "action head, gated rank-1 (P1, blind editor)": "p1",
             "VLM + action expert (VA)": "va"}
    locs = {"v1full": V2MI["location"], "v1img816": ABL["location"],
            "a1": "action-expert MLP down-projections action.blocks.00-17.mlp_1.linear, solved through the adaRMS gate",
            "h1": "action head action.action_out_proj.kernel, 8 used columns, least-norm",
            "p1": "action head gripper column, gated rank-1 (blind editor's accepted edit)",
            "va": "v2mi-f plus the action-expert update"}
    sizes = {"v1full": 450468837, "v1img816": 248057499, "a1": 60378838, "h1": 8056, "p1": 1024, "va": 505694140}
    for label, arm in where.items():
        c = comps[arm]
        ed = {"name": label, "fingerprint": wh["fingerprints"][arm], "location": locs[arm],
              "values_changed": sizes[arm], "values_changed_note": "stored values changed (where/README.md table)"}
        recs.append(record(
            "where_to_edit", "T1", edit=ed, wording=[TASKS["T1"][1], EXPLICIT], slots=wh["slot_range"],
            source="study/results/opus/where/results/heldout_analysis.json",
            notes=f"{label}; fingerprint {wh['fingerprints'][arm]}; all arms paired with one unedited run; "
                  "12 of 14 chunks ran on one lane, so their second 16 slots started from other scenes "
                  "(sensitivity analysis on the 32 unaffected slots in where/README.md)",
            n=c["n"], unedited_k=c["ref_succ"], edited_k=c["cand_succ"], fixed=c["fixed"], broke=c["broke"],
            p=c["mcnemar_p"], ci={"wilson95_unedited_pct": wil(c["ref_succ"], c["n"]),
                                  "wilson95_edited_pct": wil(c["cand_succ"], c["n"]),
                                  "bootstrap95_diff_pts": c["boot95_pp"]}))

    # ------------------------------------------------------------------ v2mi-f on the other tasks (side effects)
    for code, f, what in (("T2", g, "held-out 90000-90063"), ("T3", lft, "held-out 90000-90063")):
        recs.append(record("side_effect", code, edit=V2MI, wording=[TASKS[code][1]], slots=f["summary"]["slots"],
                           source=f"study/results/opus/heldout/ ({what})",
                           notes="the T1 edit, unchanged, on this task's own instruction; weightsmith/heldout.py",
                           **paired_from_slots(f["per_slot"])))
    for f, code in (("glasses", "T2"), ("left", "T3"), ("cube", "R1")):
        d = load(f"opus/where/results/transfer_{f}_analysis.json")
        c = {x["cand"]: x for x in d["comparisons"] if x["ref"] == "base"}["v1full"]
        recs.append(record("side_effect", code, edit=V2MI, wording=[TASKS[code][1]], slots=d["slot_range"],
                           source=f"study/results/opus/where/results/transfer_{f}_analysis.json",
                           notes="where-to-edit transfer run; every slot the first episode of its Isaac process",
                           n=c["n"], unedited_k=c["ref_succ"], edited_k=c["cand_succ"], fixed=c["fixed"],
                           broke=c["broke"], p=c["mcnemar_p"],
                           ci={"wilson95_unedited_pct": wil(c["ref_succ"], c["n"]),
                               "wilson95_edited_pct": wil(c["cand_succ"], c["n"]),
                               "bootstrap95_diff_pts": c["boot95_pp"]}))
    other = []
    for r in prior:
        if r["project"] == "cube-then-banana":
            continue
        code = CODE.get(r["task"], r["task"])
        p = paired_from_slots(r["per_slot"])
        rec = record("side_effect", code, edit=V2MI, wording=[r["instruction"]], slots=r["slots"],
                     source=f"ali-ms projects/{r['project']} (study/results/cross/pi05/results/prior_pairs_ali-ms.json)",
                     notes=("practice slots, owsp eval, edit run after the unedited arm in the same project; "
                            "multi-group chunks may start later groups from other scenes"), **p)
        if code not in TASKS:
            rec["task"], rec["code"] = r["task"], None
            other.append(rec)
        else:
            recs.append(rec)
    recs += other

    # ------------------------------------------------------------------ T8: wording gap and its own edit
    pw = paired_from_slots({s: [v[1], v[0]] for s, v in sc["per_slot"].items()})
    pw["edited_k"] = None
    recs.append(record(
        "wording", "T8", wording=[TASKS["T8"][1], TF], slots=sc["slots"],
        source="study/results/sgdw/results/screen_s4r_tf_vs_rf.json",
        teacher_k={"k": sc["success_a"], "n": sc["paired_n"], "slots": sc["slots"], "wording": TF},
        notes="unedited weights; RF (reference-first, the task's own wording) = unedited_k, TF (target-first, prompt "
              "override) = teacher_k; fixed = TF-only pairs, broke = RF-only pairs",
        **pw))
    g3 = load("sgdw/results/heldout_ctl_92000-64.json")
    base92 = load("sgdw/results/heldout_grp2_92000-64.json")["per_slot"]
    per = {s: [bool(base92[s][0]), bool(g3["per_arm"]["g3"][s]["success"])] for s in base92}
    rf_g3 = load("sgdw/results/practice_rf_grp3_vs_base.json")
    tf_g3 = load("sgdw/results/practice_tf_grp3_vs_base.json")
    recs.append(record(
        "edit_target", "T8", edit=GRP3, wording=[TASKS["T8"][1], TF], slots=[92000, 92063],
        source="study/results/sgdw/results/heldout_ctl_92000-64.json, heldout_grp2_92000-64.json (unedited arm), "
               "heldout_pairs_92000-64.json",
        control_k={"k": hp["arms"]["rand3"]["success"], "n": 64, "slots": [92000, 92063],
                   "what": "random-direction control with s4r-grp3's layers, rank and singular values"},
        teacher_k={"k": sc["success_a"], "n": 32, "slots": sc["slots"], "wording": TF,
                   "note": "TF was not run on the held-out slots"},
        recovery_pct=round(100 * rf_g3["success_b"] / sc["success_a"], 1),
        retention={"T8 under TF (no harm)": {"n": tf_g3["paired_n"], "unedited_k": tf_g3["success_a"],
                                              "edited_k": tf_g3["success_b"], "fixed": tf_g3["fixed"],
                                              "broke": tf_g3["broke"], "p": tf_g3["mcnemar_exact_p"],
                                              "slots": tf_g3["slots"]}},
        notes=(f"T8's own distilled edit (TF into the weights), not v2mi-f. Recovery % on practice slots 20000-20031, "
               f"the only slots with a teacher arm: RF + s4r-grp3 {rf_g3['success_b']}/32 vs RF {rf_g3['success_a']}/32 "
               f"vs TF {sc['success_a']}/32. s4r-grp2 (2 key groups): 19/64 on the same held-out slots."),
        **paired_from_slots(per)))

    out = {"status": "PROVISIONAL: existing pi0.5 results on their own slots; not the shared 30000-30031 / "
                     "96000-96063 protocol", "model": MODEL, "hardware": HW_OLD,
           "unedited_measurements": failure, "records": recs}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results/provisional.json").write_text(json.dumps(out, indent=1) + "\n")
    for r in recs:
        print(f"{r['experiment']:13s} {str(r['code']):26s} {str(r['edit'] and r['edit']['name']):10s} "
              f"{r['slots']} n={r['n']} {r['unedited_k']} -> {r['edited_k']} fixed {r['fixed']} broke {r['broke']} "
              f"p={r['p'] if r['p'] is None else f'{r[chr(112)]:.2g}'} rec={r['recovery_pct']}")


if __name__ == "__main__":
    main()
