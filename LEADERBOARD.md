# Leaderboard

Every editor run on the [weightsmith benchmark](BENCHMARK.md), as of 2026-11-07. An edit is **accepted** when it beats the unedited policy on at least 64 unseen paired slots with an exact McNemar p < 0.05.

| Task | Editor · protocol | Edit | Edit size (values) | Split | Base → edit | Fixed / broke | p | Status |
|---|---|---|---|---|---|---|---|---|
| Cube then banana | Operator-built ¹ · owsp | `v2mi-f` language-binding edit | 450M (13.4%) | held-out | 27/128 → **106/128** | 81 / 2 | 7×10⁻²² | ✅ accepted |
| Cube then banana | Operator-built ¹ · owsp | `abl-imgl8` minimal variant | 248M; 5.3M as factors (0.16%) | held-out ² | 19/128 → **102/128** | 86 / 3 | 4×10⁻²² | ✅ accepted |
| Pick up eyeglasses | Operator-built ¹ · transfer | `v2mi-f`, unchanged | 450M | held-out | 22/64 → **44/64** | 31 / 9 | 7×10⁻⁴ | ✅ accepted |
| Cube left of bowl | Operator-built ¹ · transfer | `v2mi-f`, unchanged | 450M | held-out | 8/64 → **36/64** | 33 / 5 | 4×10⁻⁶ | ✅ accepted |
| Cube then banana | Claude Opus (xhigh) · controlled study, blind | gated rank-1 gripper readout (`action_out_proj`) | 1,024 | held-out ³ | 11/64 → **27/64** | 22 / 6 | 0.004 | ✅ accepted |
| Pick up eyeglasses | Claude Opus (xhigh) · controlled study, blind | repurposed action-expert MLP unit (lift while holding) | 3,072 | held-out ³ | 18/64 → 27/64 | 18 / 9 | 0.12 | rejected |
| Mustard into left bin | Claude Opus (xhigh) · controlled study, blind | two layer-0 MLP rows strengthen "left" | 4,094 | held-out ³ | 8/64 → 11/64 | 9 / 6 | 0.61 | rejected |
| Cube left of bowl | Claude Opus (xhigh) · controlled study, blind | rank-1 push of "left" toward "front", layer 8 | 130,764 | dev | 5/32 → 3/32 | 3 / 5 | 0.73 | none submitted (`base`) |
| Stack three cubes | Claude Opus (xhigh) · controlled study, blind | gated gripper edit, rejected offline | — | — | 16/32 (base) | — | — | none submitted (`base`) |
| Stack three cubes | Claude Opus (xhigh) · controlled study, blind, round 2 ⁴ | repurposed action-expert MLP unit (gripper open at the stack) | 3,072 | held-out ³ | 35/64 → 39/64 | 17 / 13 | 0.58 | rejected |
| Stack yellow on red | Claude Opus (xhigh) · controlled study, blind | steering edit, no offline effect | — | — | 13/32 (base) | — | — | none submitted (`base`) |
| Pick up flat box | Claude Opus (xhigh) · controlled study, blind | rank-1 gripper readout | 1,021 | dev | 0/32 → 0/32 | 0 / 0 | 1.00 | none submitted (`base`) |
| Stack yellow on red | Claude Opus · owsp | 3 attention and key/value edits | 12–30M | practice | 28/64 → 21/64 (each) | 8–11 / 15–18 | 0.21–0.26 | rejected |
| Stack yellow on red | GPT-6 Astra · Astra original, 2 adaptive runs | 3 action-MLP row edits | 791–4,412 | dev | 11/32 → 12–13/32 | 7–9 / 5–7 | 0.77–1.00 | rejected (gate) |
| Stack yellow on red | GPT-6 Astra · Astra original, 2 non-adaptive runs | identity proposals only; every capture failed replay parity | 0 | — | — | — | — | none submitted |
| Stack yellow on red | GPT-6 Astra · Astra fixed16 | 1 action-MLP row edit | 6,225 | dev | 17/32 → 11/32 | 5 / 11 | 0.21 | rejected (gate) |
| Stack yellow on red | Claude Opus · Astra protocol | 2 action-MLP row edits | 64 rows; 35,210 | dev | 12/32 → 16/32, 11/32 | 8 / 4, 6 / 7 | 0.39, 1.00 | rejected (gate) |
| Pick up flat green box | Claude Opus editors + operator · owsp | 10 edits | — | practice | 0/64 → 0/16 (each) | 0 / 0 | — | rejected |
| Mustard into left bin | Operator · owsp | gripper bias; 2 neuron flips stopped offline | 1 | practice | 13/64 → 13/64 | 10 / 10 | 1.00 | rejected |
| 7 tasks | Operator · owsp | `gthr04` gripper bias | 1 | practice | 117/240 → 117/240 | 37 / 37 | 1.00 | rejected |
| Banana then cube | Claude Opus · owsp | 3 candidates, all rejected offline | — | — | — | — | — | none submitted |
| Cube then banana | Claude Opus · owsp | stopped before any weight candidate | — | — | — | — | — | none submitted |

¹ Built by the operator session (Claude Opus) under the editor rules, not by an autonomous editor run. ² Scored after `v2mi-f`, on the same held-out slots (90000–90127). ³ Controlled study of seven blind editors (one task per failure kind, 96 practice episodes, two candidates); the operator scored each submission on held-out slots 91008–91071. Records: [study/results/ctrl/](study/results/ctrl/README.md). ⁴ The same editor as the row above, resumed for a second round with more time and the rest of the same budget (96 practice episodes and 192 activation items in all). Development: 16/32 → 17/32 (6 fixed, 5 broke). Its notes: [FINAL2.md](study/results/ctrl/stack3/FINAL2.md).

*Rejected*: the edits ran in closed loop and did not beat base; *(gate)* means they failed the protocol's development gate. *None submitted*: no weight edit reached closed loop. All of these runs ended with `base` or with no candidate. Held-out p values come from `weightsmith/heldout.py`; the others are computed from the published counts. Sources: [`comparisons.json`](study/results/opus/comparisons.json), held-out records in [`opus/heldout/`](study/results/opus/heldout/) and [`ctrl/heldout/`](study/results/ctrl/heldout/), [study/RESULTS.md](study/RESULTS.md). The transfer map and the unchanged-policy controls are in the [report](study/blog/failure-levels.md#55-the-results) (§5.5, §7). Every accepted edit is published as a file you can download, apply and verify bit for bit: `v2mi-f` (also behind the two transfer rows), `abl-imgl8` and the blind editor's 1,024-value edit, in [edits/](edits/README.md).

## How to submit

1. Run an editor under [BENCHMARK.md](BENCHMARK.md), then score its one frozen candidate with `weightsmith/heldout.py` on at least 64 unseen slots.
2. Open a pull request that adds `results/submissions/<date>-<editor>-<task>/` with the files listed in [results/README.md](results/README.md).
3. Add your row to this table. Entries are marked self-reported until a maintainer re-scores them.
