# Completed measurements and their interpretation

All results here are simulated, task-specific and exploratory. The task was stacking the yellow block on the red block using the DROID joint-position π0.5 checkpoint. The original and fixed16 runtimes use separate baselines and must not be pooled.

## Actual weight changes

Each search evaluated three proposals. An identity proposal uses unchanged weights; a nonzero proposal changes at least one deployed bfloat16 scalar. A proposal is not a count of individual weights.

| Search / proposal | Changed rows | Changed scalar values | Successes / 32 | Reference successes / 32 | Nominal fixes / breaks | Gate |
|---|---:|---:|---:|---:|---:|---|
| Original adaptive 1 / first | 7 | 4,412 | 12 | 11 | 7 / 6 | Rejected |
| Original adaptive 2 / first | 2 | 1,543 | 13 | 11 | 9 / 7 | Rejected |
| Original adaptive 2 / third | 1 | 791 | 13 | 11 | 7 / 5 | Rejected |
| Fixed16 adaptive / first | 8 | 6,225 | 11 | 17 | 5 / 11 | Rejected |

All four changed outputs on fixed recorded inputs. None established better closed-loop success. All accepted-incumbent curves remain at the base. Both original non-adaptive searches exhausted their capture budgets on failed probes and submitted identities; this does not isolate an advantage of adaptive reasoning under equally usable evidence.

## Final controls

Each row has 64 nominal paired slots. These are control comparisons, not held-out evaluations of rejected edits.

| Control | Base / candidate successes | Fixes / breaks | Difference, nominal paired 95% interval |
|---|---:|---:|---|
| Fixed16 identity | 24 / 27 | 16 / 13 | +0.046875 [-0.125, 0.203125] |
| Original identity | 26 / 23 | 17 / 20 | -0.046875 [-0.234375, 0.140625] |
| Original gripper threshold 0.4 | 23 / 36 | 21 / 8 | +0.203125 [0.046875, 0.359375] |
| Original extra unchanged repeat | 26 / 35 | 21 / 12 | +0.140625 [-0.03125, 0.3125] |

The extra repeat reuses the original identity reference; comparisons are correlated. Nominal seed pairing did not guarantee equal camera observations or outcomes. The intervals are exploratory, not confirmatory significance claims. Runtime gripper postprocessing is not WSP.

## Limitations that must remain visible

- Original development identity repeats broke 3–10 nominal successful base cases, exceeding the gate's cap of two. Rejection does not isolate edit-induced harm; calibrate future gates prospectively rather than changing this one after observing results.
- Only 24 of 368 original capture items belonged to fully validated calls; 316 failed and 28 were unchecked. The fixed16 follow-up validated all 80 requested items but still yielded no accepted edit.
- Incorrect event metadata produced 62 false wrong-object annotations across 28 original baseline episodes, including all 11 successes. Terminal success is computed separately. Editor reliance on those logs is unknown. The proposed correction was not applied in either frozen runtime.
- The first-input audit found different images and actions in all 64 fixed16 identity pairs despite equal recorded robot state and seeds. It did not identify the cause of variability or classify failures as visual versus motor.
- Only one task and four actual edits were tested through selected MLP down-projections. No cross-task retention, physical-robot transfer, universal editor ranking, or training-cost advantage is established.
- Initial post-freeze diagnostics failed allocation. Separately preserved preallocation-only successors passed 24 and 16 forwards. All four edits changed actions; each had zero binary gripper flips among 480 sampled planned actions. This is numerical evidence, not task-success improvement.
- An approximately 90-minute orchestration gap occurred before the original final comparisons. Continuous overnight GPU utilization is not claimed.

See the included JSON measurements for exact counts, source identities and costs, and the full review bundle for supporting media and detailed audits. New teammate results should occupy a separately identified cohort and protocol.
