"""Render the FRESH tables of README.md from ../metrics_pi05.json and results/summary.json.

Replaces the text between <!-- fresh:begin --> and <!-- fresh:end --> in README.md. Every number comes from the
metrics file (which analyze.py computes from the per-slot run records); nothing is typed by hand.
    python3 study/results/cross/pi05/scripts/report.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parents[3]))
from weightsmith import stats  # noqa: E402

ORDER = ["T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "R1"]
LABELS = {"T1": "T1 cube then banana", "T2": "T2 pick glasses", "T3": "T3 cube left of bowl",
          "T4": "T4 stack yellow on red", "T5": "T5 mustard in left bin", "T6": "T6 stack 3 cubes",
          "T7": "T7 pick green block", "T8": "T8 mustard right of raisin (RF)", "R1": "R1 cube in bowl"}


def p_txt(p):
    if p is None:
        return "–"
    s = stats.format_p(p)
    m = re.match(r"([0-9.]+)e-0*([0-9]+)", s)
    return f"{m.group(1)} × 10⁻{''.join('⁰¹²³⁴⁵⁶⁷⁸⁹'[int(d)] for d in m.group(2))}" if m else s


def sl(s):
    return f"{s[0]}–{s[1]}"


def frac(k, n):
    return f"{k}/{n}"


def ci(c, key):
    v = c.get(key) if c else None
    return f"{v[0]:.0f}–{v[1]:.0f}%" if v else "–"


def render(recs, summary):
    F = [r for r in recs if r["status"] == "fresh"]
    by = {}
    for r in F:
        by.setdefault(r["experiment"], []).append(r)
    out = []
    jobs = summary.get("jobs", [])
    out.append(f"Jobs finished: **{len(jobs)}** (16 slots each); pod GPU time in finished jobs: "
               f"**{summary.get('gpu_hours_pod_jobs', 0)} GPU-h**. Every row below pairs arms on the same slots "
               "(same scene, same noise seeds), default render preset unless stated.\n")

    part = by.get("failure_map_partial", [])
    if part:
        out.append("### Fresh but partial: the only π0.5 episodes on the shared slots\n")
        out.append("| Task | Unedited | Wilson 95% | Slots | Machine | Preset |")
        out.append("|---|---|---|---|---|---|")
        for r in part:
            out.append(f"| {LABELS[r['code']]} | **{frac(r['unedited_k'], r['n'])}** | "
                       f"{ci(r['ci'], 'wilson95_pct')} | {sl(r['slots'])} | ali-ms RTX 3090 | `{r['render_preset']}` |")
        out.append("\nOne 16-slot group (first and only run of a fresh Isaac process, environment seed 0), run before "
                   "the π0.5 queue left ali-ms. It used the `performance` preset, not the default preset of Tracks A "
                   "and B, so it is not a like-for-like failure-map entry. Per-slot outcomes are in the record.\n")

    fm = {r["code"]: r for r in by.get("failure_map", [])}
    if fm:
        out.append("### 1. Unedited failure map (slots 30000–30031, default preset)\n")
        out.append("| Task | Unedited | Wilson 95% | Slots | Provisional (other slots, `performance`) |")
        out.append("|---|---|---|---|---|")
        prov = {r["code"]: r for r in recs if r["status"] == "provisional" and r["experiment"] == "failure_map"}
        for c in ORDER:
            if c in fm:
                r = fm[c]
                pv = prov.get(c)
                out.append(f"| {LABELS[c]} | **{frac(r['unedited_k'], r['n'])}** | {ci(r['ci'], 'wilson95_pct')} | "
                           f"{sl(r['slots'])} | {frac(pv['unedited_k'], pv['n']) + ' on ' + sl(pv['slots']) if pv else '–'} |")
        out.append("")

    th = [r for r in by.get("edit_target_heldout", []) if r["code"] == "T1"]
    wh = [r for r in by.get("wording_heldout", []) if r["code"] == "T1"]
    ch = [r for r in by.get("control_heldout", []) if r["code"] == "T1"]
    if th or ch or wh:
        out.append("### 2. T1 held-out (slots 96000–96063): the paper's edit, its random control, the teacher\n")
        out.append("| Arm | Slots | Unedited → arm | Fixed / broke | Exact p | Paired diff, bootstrap 95% |")
        out.append("|---|---|---|---|---|---|")
        for r in th + ch:
            e = r["edit"]["name"]
            out.append(f"| `{e}` ({r['edit']['fingerprint']}) | {sl(r['slots'])} (n = {r['n']}) | "
                       f"{r['unedited_k']}/{r['n']} → **{r['edited_k']}/{r['n']}** | {r['fixed']} / {r['broke']} | "
                       f"{p_txt(r['p'])} | {r['ci']['diff_pts']:+.1f} [{r['ci']['bootstrap95_diff_pts'][0]:+.1f}, "
                       f"{r['ci']['bootstrap95_diff_pts'][1]:+.1f}] pts |")
        for r in wh:
            out.append(f"| teacher wording \"{r['teacher_k']['wording']}\" | {sl(r['slots'])} (n = {r['n']}) | "
                       f"{r['unedited_k']}/{r['n']} → **{r['teacher_k']['k']}/{r['n']}** | {r['fixed']} / "
                       f"{r['broke']} | {p_txt(r['p'])} | {r['ci']['diff_pts']:+.1f} "
                       f"[{r['ci']['bootstrap95_diff_pts'][0]:+.1f}, {r['ci']['bootstrap95_diff_pts'][1]:+.1f}] pts |")
        for r in th:
            extra = []
            if r.get("recovery_pct") is not None:
                t = r["teacher_k"]
                extra.append(f"recovery of the wording gap: **{r['recovery_pct']:.0f}%** "
                             f"(({r['edited_k']} − {r['unedited_k']}) / ({t['k']} − {r['unedited_k']})); "
                             f"edit vs teacher: {t['edit_vs_teacher']['edit_only']} edit-only, "
                             f"{t['edit_vs_teacher']['teacher_only']} teacher-only pairs, "
                             f"p = {p_txt(t['edit_vs_teacher']['p'])}")
            if r.get("control_k"):
                k = r["control_k"]
                extra.append(f"edit vs control on the same slots: {k['edit_vs_control']['fixed']} edit-only, "
                             f"{k['edit_vs_control']['broke']} control-only pairs, "
                             f"p = {p_txt(k['edit_vs_control']['p'])}")
            if extra:
                out.append(f"\n`{r['edit']['name']}`: " + "; ".join(extra) + ".")
        out.append("")

    se = [r for r in recs if r["status"] == "fresh" and r["edit"] and r["edit"]["name"] == "v2mi-f"
          and r["experiment"] in ("edit_target", "side_effect", "retention")]
    if se:
        out.append("### 3. Side-effect map of `v2mi-f` (each task's own wording, slots 30000–30031)\n")
        out.append("| Task | Unedited → edited | Fixed / broke | Exact p | Paired diff, bootstrap 95% | Reading |")
        out.append("|---|---|---|---|---|---|")
        for c in ORDER:
            for r in se:
                if r["code"] != c:
                    continue
                d = r.get("transfer") or r.get("retention") or {}
                reading = ("target task" if r["experiment"] == "edit_target" else
                           {"better": "better", "worse": "**worse**"}.get(d.get("direction"), "no detectable change"))
                out.append(f"| {LABELS[c]} | {r['unedited_k']}/{r['n']} → **{r['edited_k']}/{r['n']}** | "
                           f"{r['fixed']} / {r['broke']} | {p_txt(r['p'])} | {r['ci']['diff_pts']:+.1f} "
                           f"[{r['ci']['bootstrap95_diff_pts'][0]:+.1f}, {r['ci']['bootstrap95_diff_pts'][1]:+.1f}] | "
                           f"{reading} |")
        out.append("")

    t8 = [r for r in F if r["code"] == "T8" and r["experiment"] in
          ("wording", "wording_heldout", "edit_target", "edit_target_heldout", "control_heldout", "side_effect")]
    if t8:
        out.append("### 4. T8: the wording gap and its edit\n")
        out.append("| Comparison | Slots | Result | Fixed / broke | Exact p |")
        out.append("|---|---|---|---|---|")
        for r in t8:
            if r["experiment"].startswith("wording"):
                out.append(f"| unedited, RF vs TF wording | {sl(r['slots'])} | RF {r['unedited_k']}/{r['n']}, "
                           f"TF **{r['teacher_k']['k']}/{r['n']}** | {r['fixed']} TF-only / {r['broke']} RF-only | "
                           f"{p_txt(r['p'])} |")
            else:
                e = r["edit"]["name"]
                out.append(f"| `{e}` vs unedited, RF | {sl(r['slots'])} | {r['unedited_k']}/{r['n']} → "
                           f"**{r['edited_k']}/{r['n']}** | {r['fixed']} / {r['broke']} | {p_txt(r['p'])} |")
        rec = [r for r in t8 if r.get("recovery_pct") is not None]
        for r in rec:
            out.append(f"\nRecovery of the TF–RF gap by `{r['edit']['name']}` on {sl(r['slots'])}: "
                       f"**{r['recovery_pct']:.0f}%** (({r['edited_k']} − {r['unedited_k']}) / "
                       f"({r['teacher_k']['k']} − {r['unedited_k']})).")
        out.append("")

    sec = by.get("preset_sensitivity", []) + by.get("edit_under_performance_preset", []) + \
        by.get("hardware_check", []) + [r for r in by.get("wording", []) if r["code"] == "T3"]
    if sec:
        out.append("### 5. Secondary checks\n")
        out.append("| Check | Task | Slots | Result | Fixed / broke | Exact p |")
        out.append("|---|---|---|---|---|---|")
        for r in sec:
            if r["experiment"] == "preset_sensitivity":
                out.append(f"| unedited, default vs `performance` preset | {r['code']} | {sl(r['slots'])} | "
                           f"default {r['unedited_k']}/{r['n']}, performance {r['edited_k']}/{r['n']} | "
                           f"{r['fixed']} / {r['broke']} | {p_txt(r['p'])} |")
            elif r["experiment"] == "edit_under_performance_preset":
                out.append(f"| `v2mi-f` vs unedited, both `performance` | {r['code']} | {sl(r['slots'])} | "
                           f"{r['unedited_k']}/{r['n']} → {r['edited_k']}/{r['n']} | {r['fixed']} / {r['broke']} | "
                           f"{p_txt(r['p'])} |")
            elif r["experiment"] == "hardware_check":
                out.append(f"| RTX 3090 (ali-ms) vs RTX 4080 SUPER (pod), unedited, `performance` | {r['code']} | "
                           f"{sl(r['slots'])} | 3090 {r['unedited_k']}/{r['n']}, 4080S {r['edited_k']}/{r['n']} | "
                           f"{r['fixed']} / {r['broke']} | {p_txt(r['p'])} |")
            else:
                out.append(f"| default vs \"specific\" wording, unedited | {r['code']} | {sl(r['slots'])} | "
                           f"default {r['unedited_k']}/{r['n']}, specific {r['teacher_k']['k']}/{r['n']} | "
                           f"{r['fixed']} / {r['broke']} | {p_txt(r['p'])} |")
        out.append("")

    ab = [r for r in F if r["edit"] and r["edit"]["name"] == "abl-imgl8" and r["experiment"] in
          ("edit_target", "side_effect", "retention")]
    if ab:
        out.append("### 6. The 0.16% variant `abl-imgl8` (slots 30000–30031)\n")
        out.append("| Task | Unedited → edited | Fixed / broke | Exact p |")
        out.append("|---|---|---|---|")
        for c in ORDER:
            for r in ab:
                if r["code"] == c:
                    out.append(f"| {LABELS[c]} | {r['unedited_k']}/{r['n']} → **{r['edited_k']}/{r['n']}** | "
                               f"{r['fixed']} / {r['broke']} | {p_txt(r['p'])} |")
        out.append("")
    return "\n".join(out)


def main():
    recs = json.loads((HERE.parent / "metrics_pi05.json").read_text())
    summary = json.loads((HERE / "results/summary.json").read_text())
    readme = HERE / "README.md"
    s = readme.read_text()
    body = render(recs, summary)
    a, b = "<!-- fresh:begin -->", "<!-- fresh:end -->"
    if a in s and b in s:
        s = s.split(a)[0] + a + "\n" + body + "\n" + b + s.split(b, 1)[1]
        readme.write_text(s)
        print("README updated")
    else:
        print(body)


if __name__ == "__main__":
    main()
