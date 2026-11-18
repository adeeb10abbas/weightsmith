# Weightsmith repairs on three additional goals

This additive study tests the same **phase-grouped residual-increment ridge write** used by the successful spatial Weightsmith construction. It does not use the later uniform-trajectory, preservation-constrained variant. Existing studies and runtime sources remain unchanged.

The fixed goals are banana left of plate, butter behind raisin box, and left bowl nested inside right bowl. The inverse and direct wordings name the same mover and intended relation. Relations use the robot frame. The bowl identities are checked after each reset. Bowl scoring requires containment/nesting, and every original, clarified, unseen, and preservation request explicitly specifies that relation. Internal case IDs retain `left_bowl_on_right` for the nested configuration. Existing scenes and historical banana-placement/bowl-stack control exposure informed these choices; they are new independently fitted goal constructions, not blind task discoveries. The two relational goals share an instruction pattern and do not establish two distinct failure mechanisms.

## Frozen budget

- Initial qualification: 16 original and 16 clarified-input trials per goal, four development layouts with four noise streams (96 total).
- Repeat every qualifier on fresh noise, reversing wording order (at most96 more). Both rounds require clarification at least8/16, improvement at least4/16, original at most10/16.
- Every twice-qualified goal receives16 trials on its different preservation request (at most48). At least8 successes establishes baseline competence; otherwise report a baseline-floor stress test.
- Every constructed edit receives160 held-out trials: base/edit × original/three unseen wordings/preservation × eight new layouts × two noise streams. At most480 trials. Exact unseen wording is excluded from fitting.
- No goal replacement, new clarification search, goal-specific layer/rank/strength search, test-based candidate selection, or automatic replay of an incomplete policy batch.

The source and candidate manifests are frozen before evaluation. All qualifications, source-coverage failures, costs, and failed attempts remain in the report. New layouts are small predeclared XY perturbations of existing scenes; this does not claim unseen environments or broad scene generalization.

## Task scoring and phase selection

Primary success requires the specified relation/container geometry, surface contact, detached gripper, object speed below2cm/s for one second, mover displacement of at least2cm, and reference-object maximum displacement no more than2cm. Reverse bowl nesting swaps mover/reference roles. Fields named `stock_*` are instantaneous component diagnostics, not exact benchmark terminal scores.

Before each simulator control step the runtime records primary-mover gripper contact and height above its reset position. These records are aligned with the policy input's recorded `step`. The source-selection rule is fixed:

1. **Precontact:** the decision before the chunk that first establishes contact. A failed original trajectory that never grasps supplies its last noncontact decision; this is a failure-local pre-contact category, not evidence of approach motion. Failed original trajectories have priority, followed by successful clarified trajectories.
2. **Grasp:** the decision whose executed chunk first establishes mover/gripper contact, using successful clarified trajectories. Contact observed exactly at a replanning boundary belongs to the preceding chunk.
3. **Transport:** the first recorded decision at which the mover is in gripper contact and at least25mm above reset height, using successful clarified trajectories.

Select six distinct source episodes per phase from the initial qualification, with earliest slots breaking ties. Select two per phase from the independent repeat as offline validation. Do not duplicate the same observation decision across phases or use a time-fraction fallback. Insufficient coverage is a recorded construction failure. Measured simulator contact/clearance is privileged selection information and is disclosed; it is not a human action demonstration. This deterministic event-based source selector is new, while the phase-grouped target and weight-write construction are reused.

For each phase, average paired original/clarified activations over its six decisions. Keep32 image-cell keys per phase,96 rows total; edit VLM MLP output projections in layers8–16 using frozen residual increments and relative ridge1e-3. Layer7 supplies the preceding gap. There are no preservation rows, no rank truncation, and no strength search. Merge the full solve into BF16 weights. Native exact-pool replay at1e-6 and independent singleton teacher parity at.01 must pass. The six validation decisions never enter the solve.

## Execution

```bash
python3 -m unittest discover -s study/iros2027/v2/rss_weightsmith_goals/tests -v
python3 study/iros2027/v2/rss_weightsmith_goals/run.py \
  --root projects/rss-weightsmith-goals-20261118 --gpu 1
```

The durable supervisor runs six policy-free simulator starts (one per scene/split), two reset cycles each, then qualification, source preparation, activation capture, fitting, candidate sealing, held-out testing, and a receipt/raw-observation audit. No policy or action backend is imported in reset preflight. All12 layouts per goal must pass pose-write, initially-false-goal, settling, and identity checks. One simulator attempt per batch; failure preserves its artifacts and stops the stage. Missing source-phase coverage is reported separately and does not discard other qualified goals.

Current status is in the project `study-status.json`, with `preflight/status.json`, `qualification-status.json`, adaptation statuses, and held-out owner statuses supplying the detailed state. A complete final report is `analysis-v1/report.json`. Unit tests, policy-free resets, and completed policy episodes are separate categories.
