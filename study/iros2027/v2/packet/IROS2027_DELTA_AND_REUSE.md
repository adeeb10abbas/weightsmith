# IROS 2027 — no-wasted-experiments audit (v2)

**Decision:** yes, the IROS simulation plan extends the existing WSP paper. Treat old results as validated historical evidence and make the next GPU jobs answer *different*, predeclared research questions. This file overrides any contrary reading of the original packet.

## 1. Evidence ledger: retain, don't repeat

| Existing completed work (source in weightsmith repo) | Conclusion it already supports | Additive next experiment | Priority |
|---|---|---|---|
| `study/results/opus/comparisons.json`: `v2mi-f` 27/128 → 106/128, fixed 81, broke 2 | Strong single-task held-out repair on fixed RoboLab scene | Apply **same frozen edit** to actual novel object arrangements (with new paired contemporaneous baseline) | HIGH |
| `study/results/opus/comparisons.json`: minimal image-cell edit 19/128 → 102/128; action-location ablations in paper | Where editing works and parameter compression | Keep as-is; only do extra edit-placement sweeps if the independent repairs show a contradictory pattern | LOW |
| `study/results/opus/comparisons.json`: random directions 13/64 → 9/64 and explicit prompt 13/64 → 57/64 | Random perturbation cannot explain improvement; explicit wording reveals latent skill | For **each new repair**, repeat matched controls; for CTB do not need identical conditions again | HIGH (new cases) |
| CTB transfer to eyeglasses 22/64 → 44/64 and cube-left 8/64 → 36/64 | Same edit can transfer to other fixed-scene tasks | Map **new independent edits** on the original 16-task suite and add scene-shift tests for CTB | HIGH |
| 16-task baseline survey and original edit retention (`BENCHMARK.md`, `comparisons.json`) | Failure taxonomy and preliminary side effects | New edits × same 16 tasks, plus targeted relation/counterexample suite; quantify regressions rather than re-run old identical matrix | HIGH |
| `study/results/ctrl/README.md`: seven blind agents, only one passing edit; negative capability cases | Editing succeeds selectively; negative evidence matters | Compare automated reproduction on **new** failures with fixed budget, track both eligible and ineligible tasks | MEDIUM |
| `study/RESULTS.md` and `BENCHMARK.md`: ~30% paired-outcome flips with same weights | Reproducibility/noise caveat | Recalibrate on **new scenes / any modified runtime**; no need to re-derive unchanged reference number for same setup | HIGH if runtime changes |
| Project site's reported additional-model cases | Potential secondary model evidence (not verified by this audit) | Inventory artifacts and run only missing architecture-specific measurements | AUDIT FIRST |

The old 16-task study remains important even if the new paper emphasizes only language repairs. Do not discard it or rewrite the negative cases as though never attempted. Keep original edit SHA/policy SHA and initial scene fixed for the original headline plot.

## 2. How to count the 12 proposed instruction pairs

| IDs | What they represent | Count toward independent-repair claim? |
|---|---|---|
| T01 | Published CTB | **No**, old anchor |
| T04 | New wording of CTB goal | No new task; may establish paraphrase generalization |
| T02, T03 | Two wordings for one *new* sequential cube/bowl + banana/plate goal | At most **one** independent goal/failure unit; require a new edit and diagnosis |
| S01, S02 | Cube-left/right of bowl converse formulations on existing tasks | Count distinct only with separate diagnosed failures; report common spatial family/correlation |
| S03 | Existing-site Cosmos3 case per prior packet; verify artifacts | **No** until provenance audited; likely old or secondary |
| S04 | Proposed banana-left-of-plate task | New goal candidate if physical competence and semantic equivalence validated |
| B01, B02 | Two wordings for one order-free cube/bowl + banana/plate goal | At most **one** new goal/failure unit |
| B03, B04 | Two wordings for one red/blue two-bin goal | At most **one** new goal/failure unit |

Also: T02/T03 and B01/B02 share objects/skills, so report both, but don't claim they represent wholly independent task families. A strong paper has diversity **of failures and semantic goals**, not merely diversity of prompts.

## 3. Strict additive execution order

1. **Archive/audit, not bulk replay:** collect original result paths, conditions, task names, versions, edits, seeds, hashes, score predicates, negative outcomes. Label each proposed new run `new_claim`, `replication`, or `duplicate`. Skip duplicates.
2. **Fill the #1 gap:** find 2–3 new independently diagnosed, wording-sensitive failures on different *task goals*, with a known-good clarified prompt and latent physical skill. Prespecify screens and disclose failures. Use new held-out slots only after freezing edits.
3. **Fill the #2 gap:** augment RoboLab with controlled object-layout scene sampling, then re-test **original frozen CTB edit** and successful new edits on new scenes. Keep a separate *fixed original scene vs varied scenes* figure.
4. **Fill the #3 gap:** interference matrix of **new edits** across original tasks plus matched opposite/same spatial relation. Preserve published 16-task comparisons; add targeted trials only where old effect size is too uncertain or new scene condition is different.
5. **Secondary models/hardware later:** use their already verified results first; never make the IROS deadline depend on deploying another policy/robot.

Every added experiment gets: (i) previously unproven claim, (ii) new cohort ID, (iii) frozen code/checkpoint/edit hashes, (iv) held-out and disclosure policy, (v) success metric and cost estimate.

## 4. Publication-safe synthesis

- **Original result remains:** one notable closed-form repair, rigorously controlled in fixed-scene simulation, plus selective transfer and negative controls.
- **New result (if empirically supported):** frequency with which the method repairs other independently diagnosed instruction failures, how it generalizes to new scenes, and where it interferes.
- **Don't pool:** old fixed-scene + new randomized-scene outcomes into a single `n` or p-value; old Claude/Opus + old Astra studies; screen-selected successes without reporting total screens.
- **Don't call:** new prompt variants 'new repairs' unless a new independent edit/failure diagnosis exists; changed random policy seeds 'OOD scenes'; an unverified site result 'reproduced'.

**Done means:** the paper can cite every original core finding unchanged and add new independent rows/plots. Even if no new repair works, the original study is not invalidated; the new work estimates the boundary of the mechanism and informs a more honest IROS submission.
