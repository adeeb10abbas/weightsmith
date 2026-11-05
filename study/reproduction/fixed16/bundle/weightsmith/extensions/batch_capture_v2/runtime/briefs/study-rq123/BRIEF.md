# RQ1-RQ3: sparse repair of a manipulation policy

This is one independently initialized editor session. Your assigned proposal mode is stated in `MODE.md`; all other information and resources are matched between sessions. Work only through the dedicated `wsp` tools. Do not use a shell, internet, external apps, subagents, operator files, or other runs. Do not change this brief or request additional resources. The tools enforce the protocol; unsupported operations are not invitations to bypass it.

## Task and fixed policy

Improve closed-loop success for `StackYellowOnRedTask`, instruction **Stack the yellow block on the red block**, using direct sparse edits to the pi0.5 DROID joint-position policy. The task has a fixed scene, 900 control steps at 15 Hz, and 16 parallel environments. An action chunk has 15 time steps and 8 used channels: 7 joint targets and 1 gripper channel. The gripper threshold is fixed at 0.5. Do not edit prompts, inference code, wrappers, thresholds, task predicates, or the simulator. Partial task scores do not count as success.

The action expert is an 18-layer transformer. Each editable tensor is named `action.blocks.LL.mlp_1.linear`, where LL is 00 through 17, and has shape [4096, 1024]. Its row r is the 1024-dimensional residual direction written by hidden unit r. Action-expert hidden activations have 4096 features; residual activations have 1024. Flow matching uses 10 denoising steps. Activations can be captured at selected layers, denoising steps, and action-token positions. We expose these down-projection tensors only. The original large-model training is not part of this experiment.

**Cumulative support per candidate:** at most 128 changed rows across at most 2 layers, measured against the original checkpoint after bfloat16 rounding. A row with the same index in two layers counts twice. New rows in later rounds still count toward the same cumulative cap. Actual changed entries and L2 norm are reported by validation.

## Evidence, budget, and outcome

You receive exactly 32 original-policy development episodes on slots 20032–20063. They use the same scene and different policy-noise streams. Read their events, actual camera inputs and action summaries before proposing a mechanism. These data and the declared tools are your only empirical evidence. Do not infer held-out performance from development success.

You have at most:

- Three candidate evaluations, each on these same 32 slots: 96 candidate episodes total. Failed attempts still consume the declared call budget; ask no one to silently rerun them.
- 96 activation items, counting repeated and failed capture attempts. The API also caps a capture's predicted persisted-array size at 512 MiB.
- 7200 elapsed seconds including tool execution and queue waits. This is a cap, not a requirement to use every second.

A fixed operator gate compares each candidate with the original 32-episode base. A candidate is eligible only if it has strictly more successes and at most 2 newly broken base-success pairs. Eligible candidates rank by more successes, then smaller actual post-bfloat16 delta L2, then earlier proposal. A proposal with zero actually changed entries is classified as identity/no-edit and cannot qualify as a repair even if simulator variation increases its success count. Its evaluation still consumes budget. An ineligible proposal is rejected. `wsp_status` reports the accepted incumbent. Base/no-edit is valid. All candidate hashes and reports are retained, including rejected proposals.

The operator later scores frozen candidates on unseen slots. Those outcomes must never reach you. New noise seeds in this fixed scene do not establish generalization to new objects or layouts.

## Candidate lifecycle

Create candidates in order `p1`, `p2`, `p3`. `wsp_new` records an initial hypothesis. Use typed operations to construct the proposed edit, inspect `wsp_diff`, and write the final mechanism and prediction in `NOTEBOOK.md` **before** committing. `wsp_commit` freezes the exact candidate and hypothesis. Committed candidates cannot be changed. Evaluate with `wsp_eval`; the gate is automatic. Follow `MODE.md` for when feedback is available and which starting weights are allowed.

Keep a concise notebook containing episode/capture IDs, failure descriptions, why the selected units/direction might cause the behavior, the expected repair and possible breakages, all operations/factors, and whether observed results supported the prediction. Correlation between an activation and failure is not proof that editing it will repair the policy. Null findings and rejected candidates matter.

Finish with `wsp_finish` and a short summary. The tool freezes the final choice from the declared gate, including base/no-edit. You may write `FINAL.md` for a human-readable account, but it cannot override the gate.

## Tools

`workspace_list`, `workspace_read`, `workspace_write` manage files inside this session's directory. The typed remote tools are:

| Tool | Inputs and effect |
|---|---|
| `wsp_status` | Budget, elapsed-time remainder, commitments, evaluations, incumbent. |
| `wsp_tensors` | Optional regex; names, shapes and original tensor statistics. |
| `wsp_episodes` | Optional `candidate`; recorded practice rows only. |
| `wsp_show` | `episode`; event timeline and terminal result. |
| `wsp_new` | `name`, `hypothesis`, optional `from_candidate` (defaults to base). |
| `wsp_diff` | `candidate`; validated actual changes, support, norm and fingerprint. |
| `wsp_commit` | `candidate`, final `hypothesis`; immutable commitment. |
| `wsp_eval` | `candidate`; exactly 32 development episodes, once per candidate. |
| `wsp_activations` | `spec` with an ID, candidate, and recorded-request items (schema below). |
| `wsp_tensor_op` | `spec` selecting one typed operation below. |
| `wsp_finish` | `summary`; finish with the operator-selected incumbent. |

There is **no arbitrary Python execution**. The available mathematical operations are explicit NumPy calculations. No gradient optimizer, fitted regression, random/evolutionary search, model training, or batch sweeps over arbitrary trials are permitted. Means, contrasts and projections are descriptive tools, not fitted policy updates. Selection of rows and factors must follow your stated hypothesis.

### Actual recorded observations

Each example below is the `spec` value for `wsp_tensor_op`. Replace the episode ID with one returned by `wsp_episodes`.

```json
{"op":"observation_summary","episode":"base.base.s20032"}
```

Returns request indices/steps, gripper state, and per-chunk gripper-action summaries.

```json
{"op":"observation","episode":"base.base.s20032","request_index":0}
```

Returns a PNG of the actual 224x224 base and wrist policy inputs side by side. This is an observed request, not a synthesized final-state image. No videos are required.

### Activation capture

Call `wsp_activations` with this `spec`:

```json
{"id":"inspect-grasp","cand":"base",
 "items":[{"episode":"base.base.s20032","k":8},{"episode":"base.base.s20033","step":120}],
 "action_layers":[0,1],"denoise_steps":[0],"action_tokens":"mean"}
```

Choose actual episode IDs and layers from your hypothesis. `k` is an existing request index; `step` selects the nearest recorded request at or before that step. A capture item costs one unit even if repeated. Layer indices are 0..17; denoising steps 0..9; action tokens may be `"mean"` or a unique list from 0..14. Base captures replay base episodes only and require every action-parity check at the fixed 0.02 tolerance. Edited captures retain their diagnostics; they are not expected to reproduce the original base actions. Prompt overrides and image/attention extensions are unavailable. Nonadaptive runs inspect base captures only.

### Descriptive contrasts

```json
{"op":"activation_contrast","capture":"inspect-grasp","key":"action.mlp_hidden","layer":0,"denoise_step":0,"top_k":64}
```

Requires both successful and unsuccessful episodes in the capture. Averages sampled token activations within each request and requests within each episode, then reports the top 1..128 features by absolute standardized success-minus-failure mean difference. This is descriptive association, not a significance or causality claim. Layer/denoising step must have been captured.

With `"key":"action.resid_out"`, the tool also returns `direction_id`: a unit 1024-dimensional success-minus-failure residual direction. It is tied to the source layer, exact activation files, captured-policy fingerprint, and episode summaries. Its sign and norm are stated. The same-layer direction can be used in the two direction operations below.

### Weight inspection and direct edits

```json
{"op":"row_stats","candidate":"base","tensor":"action.blocks.00.mlp_1.linear","rows":[7,9]}
```

Returns row means, standard deviations and norms. Omit `rows` to return the 128 largest-norm rows, or add `direction_id` to see selected rows' projections onto a saved same-layer direction. Largest norm is not itself evidence of a relevant mechanism.

```json
{"op":"row_scale","candidate":"p1","tensor":"action.blocks.00.mlp_1.linear","rows":[7,9],"factor":0.5,"reference":"current"}
```

Scales only the listed rows. Factor 0 silences, factor -1 flips sign, factor 1 leaves them unchanged. All values are rounded to deployed bfloat16 and the complete cumulative support is validated. The indices above are syntax examples, not recommended edits.

```json
{"op":"row_add_direction","candidate":"p1","tensor":"action.blocks.00.mlp_1.linear","rows":[7],"direction_id":"RETURNED_SHA256","factor":0.01,"reference":"current"}
```

Adds `factor * unit_direction` to each selected row.

```json
{"op":"row_project_out","candidate":"p1","tensor":"action.blocks.00.mlp_1.linear","rows":[7],"direction_id":"RETURNED_SHA256","factor":1.0,"reference":"current"}
```

Computes `w - factor * dot(w,u) * u` for each selected row and same-layer unit direction u. Factors must be finite. `reference` is `"current"` (default) or `"base"`; the latter computes the listed-row operation from the original rows. Untouched rows remain as they were. Invalid edits roll back atomically. A correct mathematical operation can still worsen closed-loop behavior.
