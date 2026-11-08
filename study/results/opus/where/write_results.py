#!/usr/bin/env python3
"""Print the README's results section from results/*.json (every number in it is read from those files)."""
import json
from pathlib import Path

import make_tables as T

R = Path(__file__).resolve().parent / "results"


def j(name):
    return json.loads((R / name).read_text())


def pct_range(vals):
    return f"{min(vals):.0f}-{max(vals):.0f}%"


INTERP = """\
**Held-out, each arm against the shared base.** V1-full {v1}. V1-0.16% {v116}. A1 {a1}. H1 {h1}. P1 {p1}. VA {va}.

**Against V1-full.** V1-0.16% {v116v}. A1 {a1v}. P1 {p1v}. VA {vav}.

**Transfer and retention (arm vs base on 32 paired slots, fixed/broke, exact p).** Eyeglasses: V1-full {g_v1}; P1 {g_p1};
A1 {g_a1}. Cube left of bowl: V1-full {l_v1}; P1 {l_p1}; A1 {l_a1}. Cube into bowl (retention): V1-full {c_v1};
P1 {c_p1}; A1 {c_a1}.

**The pre-registered predictions, checked.** (1) V1-full and V1-0.16% both gave large gains, but they did differ:
V1-0.16% was lower than V1-full on these slots (see above), unlike the paper's held-out runs on slots 90000-90127, where
the two edits were scored against separate unedited runs (106/128 and 102/128) and the 0.16% edit scored 54/64 against
50/64 for the full edit on the first 64 of those slots. Prediction failed. (2) P1 gained more than predicted (+15 to +25
points) and less than V1-full. (3) A1 gained less than V1-full. Held. (4) H1 closed the gripper where it should stay open
and fixed nothing. Held. (5) VA was predicted to equal V1-full; it was lower. Prediction failed. (6) Transfer: V1-full
raised both transfer tasks; P1 and A1 raised neither (H1 was not run, as planned, because it fixed nothing). Retention
was unchanged for every arm. Held. The unedited policy scored lower on these slots ({base_rate}) than in any earlier
held-out run of this task (14-21%); the comparisons are paired, so this lowers absolute rates, not the differences.

**Interpretation: where the fix lives.** In how the instruction is read, upstream of the action expert. A change of
about {v1med} of the norm of each VLM MLP down-projection (median over layers) makes the action expert compute, at the
distilled decisions, what it computes under the explicit wording ({share_v1} of the teacher's residual change in every
action-expert layer), and it does so only where the instruction matters: the first action chunk closes the gripper no
more often than for the unedited policy ({v1_first} of episodes, as for base), no phase-two decision flips offline,
the banana step succeeds in {v1_conv} of the episodes that place the cube, and the gain carries over to two other tasks
without changing the retention task. The VLM is also where the explicit wording's effect is small (it moves the image
tokens by 2-4% of their norm, paper Section 5); by the time it reaches the action expert it is a large signal tied to the
robot's state, and copying it there costs more and leaks:

- In the action expert the same distillation needs updates as large as the weights (median |dW|/|W| {a1med}): the MLP
  output is gated down by adaRMS, and the decision is already a large change of the residual stream. Offline it flips
  20 of 21 releases, but in closed loop the first action chunk closes the gripper in {a1_first} of the episodes, before
  the arm has reached anything, and the gain is a third of V1-full's. The same leak appears on the transfer tasks
  (first chunk closed in {a1_first_tx} of episodes there).
- In the action head a least-norm copy is a gripper-closing bias of about the same size at the premature releases and at
  the approaches to the banana (offline, above): the first chunk closes the gripper in {h1_first} of the episodes and no
  episode succeeds. The blind editor's edit shows that the head can carry
  a fix when a gate separates the premature release from other states (first chunk closed in {p1_first} of episodes);
  that gate was built from negative examples (initial approach, banana approach and grasp, release over the bowl), which
  the distillation recipe does not use, and even so it recovers about half of V1-full's gain and transfers nowhere.
- Adding the action-expert update on top of the VLM edit (VA) brings the action expert closest to the teacher at the
  distilled decisions, yet in closed loop VA is worse than V1-full: it places the cube about as often ({va_placed} vs
  {v1_placed} of 64) but completes the banana step less often ({va_conv} vs {v1_conv}). Matching the teacher more
  closely at 6 decisions bought nothing; the extra action-side update acts in states it was not built from.

So the answer to "where should the edit go" is: where the known-good input differs from the failing one, which here is
the instruction's reading in the VLM. Action-side edits can be made to work on the target task (A1, P1), but they need
either very large updates or a hand-built gate, they recover less, they change behaviour in states that were never
distilled, and they do not transfer. Limits: one task, one teacher wording, 64 held-out and 32 transfer pairs per
comparison; A1 needed one change to the recipe (solving through the adaRMS gate), and other action-side constructions
(keys from more states, preservation keys, an explicit gate) might do better; 12 of the 14 held-out chunks ran on one
lane, and the pre-specified sensitivity analysis on the 32 slots unaffected by that gives the same ordering and the
same significant differences.
"""


def main():
    size = j("size_summary.json")
    gap = j("gap_closed.json")
    sel = {d["cand"]: d["dv7_step0_mean_min_max"] for d in map(json.loads, (R / "head_selectivity.jsonl").read_text().splitlines())}
    ho = j("heldout_analysis.json")
    ho0 = j("heldout_analysis_run0.json")
    st = j("heldout_stages.json")
    beh = j("heldout_behaviour.json")
    gpu = j("gpu_time.json")
    arms = ho["arms"]
    cmp_ = {(c["cand"], c["ref"]): c for c in ho["comparisons"]}

    def cb(a, ref="base"):
        c = cmp_[(a, ref)]
        return c

    def line(a, ref="base"):
        c = cb(a, ref)
        return (f"{c['cand_succ']}/{c['n']} vs {c['ref_succ']}/{c['n']}, {c['fixed']} fixed and {c['broke']} broken, "
                f"{c['diff_pp']:+.1f} points [{c['boot95_pp'][0]:+.1f}, {c['boot95_pp'][1]:+.1f}], exact p = {T.fmt_p(c['mcnemar_p'])}")

    g = {a: gap["arms"][a] for a in gap["arms"]}
    share = {a: pct_range([100 * r["share"] for r in g[a]["layers"]]) for a in g}
    err = {a: pct_range([100 * r["rel_err"] for r in g[a]["layers"]]) for a in g}
    dv = {a: f"{min(g[a]['dv7_per_step']):.2f} to {max(g[a]['dv7_per_step']):.2f}" for a in g}
    tdv = f"{min(gap['teacher_dv7_per_step']):.2f} to {max(gap['teacher_dv7_per_step']):.2f}"
    out = []
    w = out.append
    w("## Results (added after the runs)\n")
    w("Every number below is read from the JSON files in `results/` (`write_results.py`, `make_tables.py`).\n")
    tx0 = {t: json.loads((R / f"transfer_{t}_analysis.json").read_text()) for t in ["glasses", "left", "cube"]
           if (R / f"transfer_{t}_analysis.json").exists()}

    def tsum(t, a):
        x = tx0[t]["arms"]
        return f"{x[a]['success']} vs {x['base']['success']} of {x[a]['n']}"
    side_p = [c["mcnemar_p"] for t in ("glasses", "left") if t in tx0 for c in tx0[t]["comparisons"]
              if c["ref"] == "base" and c["cand"] in ("p1", "a1")]
    w("**Summary.** On 64 new held-out slots, all arms paired with one unedited run "
      f"({arms['base']['success']}/64): V1-full {arms['v1full']['success']}/64, VA {arms['va']['success']}/64, "
      f"V1-0.16% {arms['v1img816']['success']}/64, P1 {arms['p1']['success']}/64, A1 {arms['a1']['success']}/64, "
      f"H1 {arms['h1']['success']}/64. Every arm except H1 beat the unedited policy, and every arm was below V1-full "
      f"(all p <= {max(c['mcnemar_p'] for c in ho['comparisons'] if c['ref'] == 'v1full'):.2g}). Only the VLM edit "
      f"transferred: eyeglasses {tsum('glasses', 'v1full')}, cube left of bowl {tsum('left', 'v1full')}; the action-side "
      f"edits that fixed the target task (P1, A1) raised neither (all p >= {min(side_p):.2g}), and no arm changed the "
      f"retention task (cube into bowl). The fix lives in the VLM's reading of the instruction; copied into the action "
      f"expert it needs updates as large as the weights and closes the gripper at the start of half the episodes, and "
      f"copied into the action head without a gate it closes the gripper almost always.\n")
    w("### Offline: the paper's 49 recorded decisions\n")
    w("Replays with images, robot state and flow-matching noise fixed (`check49.py`; a chunk counts as a release when most "
      "of its 15 gripper commands are below 0.5). The 21 premature releases include the 6 distilled decisions; the next "
      "column counts only the other 15. Phase two: 7 approaches to the banana (gripper open), 5 banana grasps and 7 "
      "banana carries (closed). Last column: mean chunk-end end-effector distance to the teacher's chunk at the 21 releases.\n")
    w(T.offline_table() + "\n")
    w(f"- Every location flips the recorded releases, but not equally cleanly. V1-full, VA and P1 flip all 21 and change no "
      f"carry or phase-two decision; A1 leaves 1 and closes the gripper at 2 of the 7 phase-two approaches; H1 leaves 2 and "
      f"closes it at 6 of 7.")
    w(f"- In the action expert the MLP output reaches the residual stream only through the adaRMS gate (below 0.05 for "
      f"27-99% of the dimensions, by layer). Eq. (1) applied as if the gate were 1 does nothing (`a1-nogate`, 21 of 21 "
      f"releases remain). Solved through the gate, the minimum-norm update is as large as the weights: median |dW|/|W| per "
      f"layer {size['a1']['median']:.2f} (range {size['a1']['min']:.2f}-{size['a1']['max']:.2f}), against "
      f"{size['v1-full']['median']:.3f} ({size['v1-full']['min']:.3f}-{size['v1-full']['max']:.3f}) for V1-full and "
      f"{size['v1-img816']['median']:.3f} for V1-0.16%. The explicit wording moves the action expert's residual stream by "
      f"about 40% of its norm at layer 16 (and the VLM's image tokens by 2-4%, paper Section 5): by the time the decision "
      f"reaches the action expert it is a large signal. (Correction to the pre-registration text: in layers 4-16 A1's "
      f"|dW|/|W| ranges {min(r['dW_rel_norm'] for r in j('build_a1.json')['layers'][4:17]):.2f}-"
      f"{max(r['dW_rel_norm'] for r in j('build_a1.json')['layers'][4:17]):.2f}, not 0.9-2.2; layer 6 is the exception.)")
    w(f"- The VLM edit makes the action expert compute what it computes under the explicit wording without touching it. At "
      f"the 6 distilled decisions V1-full reproduces {share['v1-full']} of the teacher's change of the action-expert residual "
      f"stream in every layer (projection on the teacher's change; relative error {err['v1-full']}), and its gripper-velocity "
      f"change per denoise step is {dv['v1-full']} against the teacher's {tdv}. A1 overshoots ({share['a1']}, gripper "
      f"{dv['a1']}); VA comes closest ({share['va']}, relative error {err['va']}) (`results/gap_closed.json`).")
    h, p = sel["h1"], sel["p1-blind"]
    w(f"- The action head cannot separate the premature release from the approach to the banana without a gate. At denoise "
      f"step 0 the least-norm head edit changes the gripper velocity by {h['PREM'][0]:.2f} at the premature releases and "
      f"{h['P2app'][0]:.2f} at the phase-two approaches; the blind editor's gated edit gives {p['PREM'][0]:.2f} and "
      f"{p['P2app'][0]:.2f} (`results/head_selectivity.jsonl`).\n")

    w("### Closed loop on the target task: held-out slots 94016-94079\n")
    w("All arms and one shared base on the same 64 new slots (two 32-slot chunks, arm order reversed in the second chunk).\n")
    w(T.heldout_table() + "\n")
    w("Sensitivity analysis (pre-specified): the 32 slots that were the first episode of lane 0 in every chunk.\n")
    w(T.heldout_table("heldout_analysis_run0.json") + "\n")
    lanes = ho.get("lanes_per_chunk", {})
    one = sorted(k for k, v in lanes.items() if len(v) == 1)
    w(f"Chunks that ran on one lane: {', '.join(one) if one else 'none'} (of {len(lanes)}).\n")
    w("Stages and behaviour (RoboLab score and events; `where_stages.py`, `where_behaviour.py`):\n")
    w("| Arm | Cube placed | Success | Banana-step conversion | First chunk closes the gripper | Closed share of chunks before the first grasp | Wrong-object grabs per episode |")
    w("|---|---:|---:|---:|---:|---:|---:|")
    names = {x[3]: x[0] for x in T.ARMS}
    names["base"] = "Unedited (base)"
    for a in ["base"] + [x[3] for x in T.ARMS]:
        if a not in st["arms"]:
            continue
        s, b = st["arms"][a], beh[a]
        conv = f"{100 * s['banana_conversion']:.0f}%" if s["banana_conversion"] is not None else "-"
        w(f"| {names[a]} | {s['cube_placed']}/{st['n_slots']} | {s['success']}/{st['n_slots']} | {conv} | "
          f"{100 * b['first_chunk_closed']:.0f}% | {100 * b['closed_share_before_first_grasp']:.0f}% | {b['wrong_object_grabs_per_episode']:.2f} |")
    w("")
    w("### Transfer and retention (32 fresh slots per task, 94016-94047)\n")
    w("Eyeglasses ran as one 32-slot chunk per arm on both GPUs (`where_run.py --all-lanes`). After the deviation noted "
      "above, cube left of bowl and cube into bowl ran on GPU0 alone in 16-slot chunks with the arm order reversed in "
      "the second chunk (`run_transfer2.sh`), so every slot is the first episode of its process; the queued "
      "`--all-lanes` runs for those two tasks were cancelled before they started. Each task's arms share one unedited "
      "run under the task's own instruction.\n")
    w(T.transfer_table() + "\n")
    for task in ["glasses", "left", "cube"]:
        p_ = R / f"transfer_{task}_behaviour.json"
        if p_.exists():
            bb = json.loads(p_.read_text())
            w(f"- {task}: first chunk closes the gripper in " + ", ".join(
                f"{names.get(k, k)} {100 * v['first_chunk_closed']:.0f}%" for k, v in bb.items()) + " of episodes.")
    w("")
    w("### Paired statistics, predictions and interpretation\n")
    tx = {t: json.loads((R / f"transfer_{t}_analysis.json").read_text()) for t in ["glasses", "left", "cube"]
          if (R / f"transfer_{t}_analysis.json").exists()}

    def txc(t, a):
        if t not in tx:
            return None
        for c in tx[t]["comparisons"]:
            if c["cand"] == a and c["ref"] == "base":
                return c
        return None

    def txs(t, a):
        c = txc(t, a)
        return "not run" if c is None else (f"{c['cand_succ']} vs {c['ref_succ']} of {c['n']}, {c['fixed']}/{c['broke']}, "
                                            f"p = {T.fmt_p(c['mcnemar_p'])}")
    btx = {t: json.loads((R / f"transfer_{t}_behaviour.json").read_text()) for t in ["glasses", "left", "cube"]
           if (R / f"transfer_{t}_behaviour.json").exists()}
    a1tx = [100 * btx[t]["a1"]["first_chunk_closed"] for t in btx if "a1" in btx[t]]
    w(INTERP.format(
        v1=line("v1full"), v116=line("v1img816"), a1=line("a1"), h1=line("h1"), p1=line("p1"), va=line("va"),
        a1v=line("a1", "v1full"), p1v=line("p1", "v1full"), vav=line("va", "v1full"), v116v=line("v1img816", "v1full"),
        a1_first=f"{100 * beh['a1']['first_chunk_closed']:.0f}%", h1_first=f"{100 * beh['h1']['first_chunk_closed']:.0f}%",
        p1_first=f"{100 * beh['p1']['first_chunk_closed']:.0f}%", v1_first=f"{100 * beh['v1full']['first_chunk_closed']:.0f}%",
        a1_first_tx=f"{min(a1tx):.0f}-{max(a1tx):.0f}%",
        va_conv=f"{100 * st['arms']['va']['banana_conversion']:.0f}%", v1_conv=f"{100 * st['arms']['v1full']['banana_conversion']:.0f}%",
        va_placed=st['arms']['va']['cube_placed'], v1_placed=st['arms']['v1full']['cube_placed'],
        base_rate=f"{arms['base']['success']}/{arms['base']['n']}",
        g_v1=txs("glasses", "v1full"), g_p1=txs("glasses", "p1"), g_a1=txs("glasses", "a1"),
        l_v1=txs("left", "v1full"), l_p1=txs("left", "p1"), l_a1=txs("left", "a1"),
        c_v1=txs("cube", "v1full"), c_p1=txs("cube", "p1"), c_a1=txs("cube", "a1"),
        share_v1=share["v1-full"], a1med=f"{size['a1']['median']:.2f}", v1med=f"{100 * size['v1-full']['median']:.1f}%"))
    w("### GPU time\n")
    ho_gpu = sum(c["gpu_s"] for c in gpu["chunks"] if c["project"] == "where-to-edit") / 3600
    tx_gpu = sum(c["gpu_s"] for c in gpu["chunks"] if c["project"] != "where-to-edit") / 3600
    w(f"Closed loop: {ho_gpu:.2f} GPU-hours on the target task and {tx_gpu:.2f} on transfer and retention (wall time of each "
      f"chunk times its lanes); activation captures at most {gpu['acts_gpu_h_upper']:.2f} GPU-hours (capture time plus 60 s "
      f"of model load per call). Total {gpu['total_gpu_h']:.2f} GPU-hours on the two-GPU host, i.e. about "
      f"{gpu['total_gpu_h'] / 2:.1f} hours of the whole host (`results/gpu_time.json`).\n")
    w("### Files\n")
    w("Scripts (run on the GPU host with `owsp` against `projects/where-to-edit`, or locally where noted):\n")
    w("- `make_specs.py`, `specs/`: the `owsp acts` capture specs (6 distilled decisions with full action-expert traces; "
      "49 check decisions); `run_acts.sh` runs them.")
    w("- `build_action.py` (A1, VA, `a1-nogate`, `a1-late`; the CFG line of each is in `builds/`), `build_head.py` (H1, "
      "`h1-grip`): `owsp py` sandbox builds.")
    w("- `check49.py` (offline table), `gap_closed.py`, `head_selectivity.py`, `rel_norms.py`: offline analyses.")
    w("- `ctl_run.py` (copied unchanged from `study/results/opus/controls/`; held-out run), `where_run.py` (its copy with "
      "lane recording and `--all-lanes`; transfer runs), `run_transfer.sh`, `run_transfer2.sh`.")
    w("- `where_analyze.py` (paired statistics, as in `ctl_analyze.py`), `where_stages.py`, `where_behaviour.py`, "
      "`gpu_time.py`, `collect_results.sh` (local: runs them and copies the JSON), `make_tables.py`, `write_results.py`.\n")
    w("Results (`results/`): `offline_49.json`, `diff_*.json` (`owsp diff`), `build_*.json`, `size_summary.json`, "
      "`vlm_rel_norms.jsonl`, `gap_closed.json`, `head_selectivity.jsonl`; `heldout_results.json` (per-slot outcomes), "
      "`heldout_plan.json` (arms, fingerprints, schedule), `heldout_analysis.json`, `heldout_analysis_run0.json`, "
      "`heldout_stages.json`, `heldout_behaviour.json`; `transfer_{glasses,left,cube}_{results,plan,analysis,behaviour}.json`; "
      "`gpu_time.json`.\n")
    w("Candidates and fingerprints (projects `where-to-edit`, `where-glasses`, `where-left`, `where-cube`): `v1-full` "
      "63db0c52a51306c9, `v1-img816` 5bf8c4329c1e416a, `p1-blind` 4ca8023b0592b79f (byte copies of the sources, "
      "fingerprints verified), `a1` 09b7c9f5bfe26912, `h1` d965cd706aed9dec, `va` 5ee25e8d2e78a555.")
    print("\n".join(out))


if __name__ == "__main__":
    main()
