#!/usr/bin/env python3
"""Markdown tables for README.md from the result JSONs in results/ (no number is typed by hand)."""
import json
from pathlib import Path

R = Path(__file__).resolve().parent / "results"
ARMS = [  # (label, candidate, offline capture, held-out arm name, location)
    ("V1-full", "v1-full", "chk-v1-full", "v1full", "VLM MLPs, layers 0-16"),
    ("V1-0.16%", "v1-img816", "chk-v1-img816", "v1img816", "VLM MLPs, image cells, layers 8-16"),
    ("A1", "a1", "chk-a1", "a1", "action-expert MLPs, layers 0-17"),
    ("H1", "h1", "chk-h1", "h1", "action head, 8 used columns"),
    ("P1 (blind editor)", "p1-blind", "chk-p1-blind", "p1", "action head, gripper column, gated"),
    ("VA", "va", "chk-va", "va", "V1-full + action-expert MLPs"),
]
EXTRA = [("A1 without gate (control)", "a1-nogate", "chk-a1-nogate", "action-expert MLPs, Eq. (1) literally"),
         ("A1, layers 13-17", "a1-late", "chk-a1-late", "action-expert MLPs, layers 13-17"),
         ("H1, gripper only", "h1-grip", "chk-h1-grip", "action head, gripper column")]
RANK = {"v1-full": "<=72 x 17 layers", "v1-img816": "<=32 x 9 layers", "a1": "150 x 17 + 128 (layer 17)",
        "a1-nogate": "150 x 18", "a1-late": "150 x 4 + 128", "h1": "8", "h1-grip": "1", "p1-blind": "1",
        "va": "<=72 x 17 (VLM) + 150 x 18 (action)"}


def j(name):
    p = R / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_pct(f):
    x = 100 * f
    return f"{x:.1f}%" if x >= 1 else (f"{x:.2f}%" if x >= 0.01 else f"{x:.2g}%")


def fmt_p(p):
    return f"{p:.2g}" if p >= 1e-3 else f"{p:.1e}"


def offline_table():
    off = j("offline_49.json")
    b, t = off["chk-base"], off["chk-teacher"]
    lines = ["| Arm | Location | Values changed | Rank | Releases left (of 21) | of the 15 not distilled | Carries released (of 9) | Phase-2 decisions flipped (of 19) | Chunk-end distance to teacher, cm |",
             "|---|---|---:|---|---:|---:|---:|---:|---:|"]
    lines.append(f"| Unedited | - | 0 | - | {b['PREM_open']} | {b['PREM_open_other15']} | {b['CARRY_open']} | 0 | {b['PREM_ee_to_teacher_cm']} |")
    lines.append(f"| Teacher wording, unedited | - | 0 | - | {t['PREM_open']} | {t['PREM_open_other15']} | {t['CARRY_open']} | "
                 f"{t['P2app_flips'] + t['P2grasp_flips'] + t['P2carry_flips']} | 0 |")
    for lab, cand, cap, *_rest in ARMS + [(a, b_, c, None, d) for a, b_, c, d in EXTRA]:
        loc = _rest[-1]
        o, d = off[cap], j(f"diff_{cand}.json")
        p2 = o["P2app_flips"] + o["P2grasp_flips"] + o["P2carry_flips"]
        lines.append(f"| {lab} | {loc} | {d['changed_elements']:,} ({fmt_pct(d['changed_fraction'])}) | {RANK[cand]} | {o['PREM_open']} | "
                     f"{o['PREM_open_other15']} | {o['CARRY_open']} | {p2} | {o['PREM_ee_to_teacher_cm']} |")
    return "\n".join(lines)


def heldout_table(name="heldout_analysis.json"):
    a = j(name)
    if not a:
        return "(held-out analysis not available)"
    base = a["arms"]["base"]
    cmp_ = {(c["cand"], c["ref"]): c for c in a["comparisons"]}
    lines = [f"Slots {a['slot_range'][0]}-{a['slot_range'][1]} ({a['n_slots']} paired slots" + (f", filter {a['slot_filter']}" if a.get('slot_filter') else "") + ").",
             "", "| Arm | Successes | Wilson 95% | Fixed / broke vs base | Difference vs base, points [95% CI] | McNemar p | Fixed / broke vs V1-full | p vs V1-full |",
             "|---|---:|---|---:|---|---:|---:|---:|",
             f"| Unedited (base) | {base['success']}/{base['n']} | {base['wilson95'][0]:.2f}-{base['wilson95'][1]:.2f} | - | - | - | - | - |"]
    for lab, cand, cap, arm, loc in ARMS:
        if arm not in a["arms"]:
            continue
        s = a["arms"][arm]
        c = cmp_[(arm, "base")]
        v = cmp_.get((arm, "v1full"))
        vtxt = f"{v['fixed']} / {v['broke']} | {fmt_p(v['mcnemar_p'])}" if v else "- | -"
        lines.append(f"| {lab} | {s['success']}/{s['n']} | {s['wilson95'][0]:.2f}-{s['wilson95'][1]:.2f} | {c['fixed']} / {c['broke']} | "
                     f"{c['diff_pp']:+.1f} [{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}] | {fmt_p(c['mcnemar_p'])} | {vtxt} |")
    return "\n".join(lines)


def transfer_table():
    rows = ["| Task | Arm | Successes (arm vs base) | Fixed / broke | Difference, points [95% CI] | McNemar p |", "|---|---|---:|---:|---|---:|"]
    found = False
    for task in ["glasses", "left", "cube"]:
        a = j(f"transfer_{task}_analysis.json")
        if not a:
            continue
        found = True
        lab = {"glasses": "Pick up eyeglasses", "left": "Cube left of bowl", "cube": "Cube into bowl (retention)"}[task]
        for c in a["comparisons"]:
            if c["ref"] != "base":
                continue
            arm = {x[3]: x[0] for x in ARMS}.get(c["cand"], c["cand"])
            rows.append(f"| {lab} | {arm} | {c['cand_succ']} vs {c['ref_succ']} of {c['n']} | {c['fixed']} / {c['broke']} | "
                        f"{c['diff_pp']:+.1f} [{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}] | {fmt_p(c['mcnemar_p'])} |")
    return "\n".join(rows) if found else "(transfer not available)"


if __name__ == "__main__":
    print("## Offline\n\n" + offline_table() + "\n\n## Held-out\n\n" + heldout_table() + "\n\n## Held-out, sensitivity subset\n\n"
          + heldout_table("heldout_analysis_run0.json") + "\n\n## Transfer\n\n" + transfer_table())
