# Residual-target ablation

This is a new, additive closed-loop study. It never modifies the original
`rss_repair_comparison` sources, fitted candidates, protocol, or test records.
The earlier study's sealed development captures and fitting recipe are reused;
all evaluations use new layout IDs, new policy-noise slots, and new receipts.
Reused observations and weights are not counted as new independent repairs.

The question is whether the target construction matters when keys, preservation,
ridge, edited layers, rank cap, base policy, and input data are fixed. Four
conditions are compared: unedited base, block-residual increments, local MLP
output differences, and correspondence-shuffled block-residual increments.
The local target is reconstructed from the existing paired hidden activations
and the unchanged output-projection matrix. This is a projected module-output
difference, not a separately measured post-rounding activation. Its natural
magnitude is retained; edit/target norms are reported rather than assumed equal.
The shuffled target uses one predeclared within-camera derangement shared across
layers and cases. It preserves target singular values and Frobenius norm but
breaks correspondence with the original image-cell keys. The resulting weight
norms need not match because the keys are anisotropic.

Both semantic cases use the original request, one fixed paraphrase absent from
weight adaptation, and the competence-qualified competing request. Eight newly
generated layouts and two noise streams give 16 trials per cell and condition,
384 episodes overall. These remain two semantic repairs, not six independent
repairs. The paraphrases were previously registered by the experimenter; they
are not a blind natural-language benchmark. No earlier test outcomes select
the cases, targets, controls, layouts, or candidate weights.

All four conditions use the same original request for a cell. Strict final
success is primary. The final analysis reports every cell, paired fixes/breaks,
and exploratory intervals over eight shared layout clusters. No tight
preservation guarantee, target-direction-only causal claim, or architecture-wide
claim follows from this design.

Only the new folder and `projects/rss-residual-targets-20261117/` may be written.
The existing fixed-16 worker/runtime is reused unchanged and hashed into the new
freeze. GPU work requires coordination with the root agent and uses its existing
GPU locks. Failed attempts are preserved; incomplete batches are not silently
replayed. CPU arithmetic or replay checks are not robot-policy results.

From the repository root, fit each case on CPU (replace `CASE` with `temporal`
or `spatial`; each output is new and refuses overwrite):

```sh
OPENBLAS_NUM_THREADS=8 python3 study/iros2027/v2/rss_residual_targets/pipeline.py fit --adaptation-root projects/rss-repair-comparison-20261117/CASE-adaptation --case CASE --output projects/rss-residual-targets-20261117/CASE-construction-v1
OPENBLAS_NUM_THREADS=2 python3 -m unittest discover -s study/iros2027/v2/rss_residual_targets -p 'test_*.py' -v
python3 study/iros2027/v2/rss_residual_targets/plan.py --run-root projects/rss-residual-targets-20261117 --output projects/rss-residual-targets-20261117/heldout-frozen-v1
```

Once the assigned GPU is available, launch each sealed case once. The wrapper
uses a GPU lock for each batch and refuses any preexisting result directory.
The GPU index is an execution choice; it does not alter the frozen case plan.

```sh
python3 study/iros2027/v2/rss_residual_targets/run_case.py --plan projects/rss-residual-targets-20261117/heldout-frozen-v1/CASE-plan.json --root projects/rss-residual-targets-20261117/CASE-heldout-v1 --gpu 1
```

When both workers report 192 completed episodes, produce the complete report.
The analysis refuses partial outcomes and checks all raw observations, metric
arrays, camera/noise identities, receipt checksums, and frozen candidates.

```sh
python3 study/iros2027/v2/rss_residual_targets/report.py --run-root projects/rss-residual-targets-20261117 --output projects/rss-residual-targets-20261117/completed-analysis-v1
```

All source and candidate choices are sealed before either evaluation starts.
The report includes 24 success rows and all 30 predeclared paired contrasts,
target/update norms, competing predicates, camera discrepancies, and costs.
