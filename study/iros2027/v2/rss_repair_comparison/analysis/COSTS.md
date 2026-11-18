# Measured cost audit

Run after development selection and the full comparison have completed:

```bash
python3 study/iros2027/v2/rss_repair_comparison/analysis/costs.py --run-root projects/rss-repair-comparison-20261117 --analysis-report /absolute/completed-analysis/report.json --output /absolute/new-cost-audit
```

Final mode requires receipts for all **256** common development, qualification,
and wording-screen episodes; **128** additional LoRA-selection episodes; and
**640** final comparison episodes. It follows each selected-candidate receipt's
authoritative `evaluation_root`. Reused exact-cache evidence, replayed teacher
inputs, and repeated references to existing episodes do not add policy episodes.
Duplicate batch or observation paths are rejected. Missing required receipts
produce an error inventory and a failing exit code. `--allow-partial` explicitly
permits an incomplete cost inventory, labeled as partial.

`costs.json` and `costs.csv` contain measured component costs and source hashes.
Shared research qualification, shared target preparation, closed-form operations,
LoRA fitting/selection, rewrite preparation, and final evaluation have separate
scopes. Each case includes teacher preparation, exact replay validation,
activation capture, solve/export, all three LoRA fits, and measured wrapper
residuals. Nested internal timings are detail columns, not extra additive costs.
The three-fit budget includes both unselected rates.

The old common-collection receipts record only batch wall time. Their queue and
execution components therefore remain **unknown**. New worker receipts supply
separate queue and execution measurements. Execution includes recorded loading
and simulation work; it is not renamed GPU compute time. No estimated server
reload time is removed, and no cold-start latency claim is made.

For original adaptation and additional sweeps, actual start/duration receipts
support both summed runner time and elapsed/overlapping runner intervals across
the two GPUs. These are end-to-end runner intervals, including CPU work and
waiting. Historical policy collection lacks complete interval endpoints; its
parallel elapsed time remains unknown rather than being estimated by dividing
the sum by two.

The two preserved failed startup cohorts and the failed singleton-replay
preparation attempt remain in the audit. Where complete duration was not
recorded it remains unknown; measured completed/partial substage time is labeled
as a lower bound. The referenced replay diagnostic is accounted for separately.
Manual wording and engineering time is explicitly unmeasured, not zero.

Rewrite dispatch is measured once per mode/prompt during plan creation. The
audit reports those calls and their measured CPU durations without presenting
them as a robust microbenchmark or online policy-request latency.

Verification uses synthetic receipts only:

```bash
python3 -m unittest discover -s study/iros2027/v2/rss_repair_comparison/analysis -p 'test_costs.py' -v
```

Six tests cover unknown timing, invalid timing arithmetic, parallel versus summed
time, duplicate counting, receipt corruption, and refusal to label missing
receipts as a completed audit.
