# Guo–Xie robotics weight-editing report: evidence and implications

Reviewed November 5, 2026. This is an assessment of **first-person experiment threads**, not an independently reproduced paper. Direct X retrieval returned an access error; the authors' posts and replies were read through a public mirror. Their personal website, public repository search and targeted paper searches did not reveal a matching manuscript, checkpoint patch or evaluation-log release. This is a retrieval limitation, not proof that no artifacts exist.

## What the authors report

| Experiment | Reported outcome | Evidence boundary |
|---|---|---|
| One π0.5 task | Four of roughly 3.6 billion parameters changed; successes 28/92 to 82/92 on held-out episodes. | The four-parameter count belongs to one example. Exact parameter coordinates and raw paired outcomes were not available in the inspected thread. |
| Five RoboTwin 2.0 tasks | Single-edit aggregate 31% to 47%; approximately four edits 27% to 65%. | The different starting rates preclude treating these aggregates as a controlled iteration curve. Task-wise denominators and selection rules need clarification. |
| FastWAM | Reported gain of at least 8.7%. | The inspected summary does not resolve percentage points versus relative percent, or provide a full evaluation protocol. |
| Unitree G1 adaptation | Falls 60/64 to 1/64 with a hidden 10 kg load; 31/64 to 3/64 on a 15-degree slope. | These are separate adaptation examples, not manipulation success rates. The inspected posts do not establish physical-hardware evaluation or broader preservation. |

The first example is from [Guo's original post](https://x.com/Lingxiao234/status/2118048171649806424), retrieved through its [public mirror](https://teapawt.net/Lingxiao234/status/2118048171649806424). The aggregate VLA/FastWAM claims are in the [settings post](https://x.com/Lingxiao234/status/2118048183540662745), also available through a [mirror](https://teapawt.net/Lingxiao234/status/2118048183540662745). The G1 counts come from [Xie's companion report](https://x.com/AngchenXie/status/2118052439576465647), retrieved through its [mirror](https://teapawt.net/AngchenXie/status/2118052439576465647). All numbers above remain author-reported.

## How the proposed edits were found

For VLA editing, the authors describe inspecting saved observations, comparing hidden features and attention against failure behavior, testing candidate changes on recorded inputs, and then using limited simulation rollouts to refine the intervention. Their examples identify two hidden features in one case and an attention head in another. This gives a concrete diagnostic sequence but is not a released causal-mechanism audit. [Method post](https://x.com/Lingxiao234/status/2118048223210402109), [retrieved mirror](https://teapawt.net/Lingxiao234/status/2118048223210402109).

For one humanoid case, they describe repurposing nine weakly contributing units to encode compensation for a hidden load, affecting waist and knee responses. This is closer to inserting a control rule into a small network than applying an activation contrast to a large VLA. The observation is useful for classifying interventions; the account alone does not prove that the explanation uniquely identifies the mechanism. [Humanoid mechanism post](https://x.com/Lingxiao234/status/2118048266378191115), [retrieved mirror](https://teapawt.net/Lingxiao234/status/2118048266378191115).

A crucial author clarification is that each edited π0.5 checkpoint was already fine-tuned for its task. The stated transfer concerns other scenes and layouts **within that task**; cross-task preservation remains future work. Training-free editing therefore does not mean a checkpoint acquired its task competence without training. [Scope clarification](https://x.com/Lingxiao234/status/2118159818317840583), [retrieved mirror](https://teapawt.net/Lingxiao234/status/2118159818317840583).

The author also recommends forward-computation tools, bounded rollout access, and tasks with a recognizable recurring failure. These are experimental suggestions, not guarantees of repair. [Author's practical clarification](https://x.com/Lingxiao234/status/2118105243120476298), [retrieved mirror](https://teapawt.net/Lingxiao234/status/2118105243120476298).

## What strengthens the report, and what remains unresolved

The report is relevant positive evidence: it gives numerical held-out claims, a diagnostic workflow, and different intervention mechanisms across policy types. It also explicitly narrows the generalization claim. These merits should remain visible in the review table.

We cannot yet independently check the exact task/checkpoint pairs, edit tensors, test-set construction, evaluator isolation, selection history, repeatability, per-task regressions, complete compute costs, or claimed parameter counts. We should record these as **unverified**, without assuming the authors omitted the controls. We should not infer statistical significance or broad preservation from aggregate success rates. A small final patch does not establish a cheap discovery process.

## Consequences for our next experiment — proposals, not new jobs

Our study is not a matched replication: it used the DROID joint-position checkpoint on one stacking task, allowed only selected action-MLP down projections, and had replay failures, inaccurate semantic annotations, and noisy nominal pairing. The Guo–Xie workflow suggests the following comparison to plan prospectively:

1. **Match the problem first.** Select a task-trained checkpoint with a recurring failure, or reproduce a released positive case when its artifacts are available. Do not expect editing to supply an entirely missing skill.
2. **Validate the diagnostic interface before search.** Correct annotation metadata; preserve batch context and deployment precision; test exact fixed-input controls and unchanged-policy rollout variation. Use randomized repeated comparisons if deterministic full-episode replay is not attainable.
3. **Separate diagnosis from intervention.** Rank candidate features on one development subset, then test fixed-input interventions on a separate subset. Contrasts are hypotheses, not causal evidence. Preserve negative and matched irrelevant-feature controls.
4. **Test access restrictions explicitly.** Compare the original MLP-only interface with an expanded, audited attention-head interface under the same prospective budgets. Their report motivates this ablation; it does not establish that attention access caused their advantage.
5. **Reserve final evaluation.** Freeze the selected patch, report all attempted changes and compute, and evaluate disjoint held-out scenes plus retention tasks. Keep diagnostic, discovery and final evidence distinct.

This is a reason to improve our test and seek a faithful reproduction. It neither validates all WSP claims nor turns our unsuccessful restricted search into evidence against the whole approach. No additional GPU work was launched for this review.
