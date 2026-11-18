# Matched comparison of the successful Weightsmith repair

This additive study reuses the full temporal Weightsmith edit and its matched
random-direction control exactly. Teacher-action LoRA uses the same six recorded
failure decisions and archived clarified action chunks. It adds no preservation
observations. The original checkpoint, successful edited weights, and historical
outcomes are never modified.

The main estimand is Weightsmith minus matched-source LoRA success. Base,
clarified input, and matched random directions are reference conditions. The
existing strict 60-second temporal task is primary; stock-compatible scores on
those same trajectories are secondary. Historical scoring/runtime differences
remain explicit. The completed 96-episode recovery diagnostic is development
evidence and is not pooled with this test.

The complete budget is 48 development selection trials (three registered
learning rates, 16 shared slots each) and 320 held-out trials (five policies,
64 shared slots each). Held-out testing uses 16 new small-perturbation layouts
with four noise streams per layout. Development uses eight separate layouts.
The existing reset validator checks poses, orientation, stability, and initially
false goals before policy execution; a failed reset stops the batch. No policy
outcomes select test layouts.

LoRA selection maximizes strict development success, with ties choosing the
lower learning rate. All candidates, source identities, selection outcomes,
runtime, prompts, and the complete test schedule are sealed before any test
episode. Arm order rotates and reverses across four 16-slot blocks; with five
arms and four blocks it is not perfectly position-balanced. Each batch allows
one simulator attempt. Partial batches are preserved and halt the worker.

The old random-control metadata identifies `v2mi-f` as its parent. Its construction
report verifies exact BF16 reproduction of that parent at every layer before
randomizing singular directions. The reused parent and control fingerprints are
checked against those original artifacts. Rank/spectrum matching applies before
storage rounding; there is one realized randomized control.

## Commands

Run from `/home/ali/weightsmith-iros2027`. Preparation is CPU-only; the supervisor
must be launched by the experiment controller after the design has been reviewed.
It waits for the complete sealed LoRA registry and then executes the sequence.

```bash
python3 study/iros2027/v2/rss_weightsmith_comparison/prepare.py design
python3 -m unittest discover -s study/iros2027/v2/rss_weightsmith_comparison/tests -v
python3 study/iros2027/v2/rss_weightsmith_comparison/supervise.py --gpu 0
```

The project root is `projects/rss-weightsmith-comparison-20261118`. The design,
selection, test freeze, and final report each refuse overwriting their own leaf
directory; LoRA-owned sibling directories are allowed. The supervisor records
`supervisor-status.json` and stage logs. LoRA preparation/training is documented
in `lora/`; no historical live-batch replay certificate is fabricated.

`analysis-v1/report.json` is written only after all test arms complete and their
receipts, observation hashes, metrics, slots, source identities, and pairing are
verified. It reports every arm, paired fixes/breaks, per-layout successes, and
95% intervals from 20,000 layout-cluster resamples. No significance threshold
gates reporting. Identical nominal state/noise does not guarantee identical
camera rendering; mismatches are reported without outcome-based exclusions.

Historical construction/search and manual decision selection are reused costs,
not zero costs. All three LoRA fits and all selection trials count. This is one
previously studied repair goal, with new layouts in the same scene; it is not an
independent repair discovery, new embodiment test, or broad language evaluation.
