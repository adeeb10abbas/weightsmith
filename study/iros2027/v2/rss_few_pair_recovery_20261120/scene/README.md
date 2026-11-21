# Mustard scene reset recovery

The original preflight reused one simulator across three different groups of
16 source layouts. The first group passed both resets. At the next group's
first reset, two mustard bottles tipped, moving about 5 cm. No policy was
called. The failed summary and complete pose records remain unchanged.

The exact 16 poses containing both failures subsequently passed both resets
in a fresh simulator. Maximum bottle XY displacement was 2.76 mm and height
change was below 0.001 mm. The original thresholds remain 12 mm XY and 25 mm
height displacement, alongside stability, table contact, and initial-goal checks.

The recovery qualifies each group of 16 in a fresh process, matching the
actual policy worker's lifecycle. Every original pose is retained exactly;
this is a process-lifecycle correction, not a new layout distribution. The
precise physics-cache or solver mechanism behind the history sensitivity has
not been established.

`isolated_preflight.py` partitions the original immutable layout file, checks
each partition twice, and emits a certificate only after all original layouts
pass. The certificate binds the original full file, every partition and child
summary, the checking code, and zero policy/action calls. Three CPU tests check
exact pose retention, rejection of incomplete evidence, and tampering detection.

The original preflight failure is never overwritten. A collector using this
recovery must verify the new certificate before launching any mustard trials.

The diagnostic launcher also removes inherited SSH display forwarding from
its headless subprocess. A forwarded `DISPLAY=localhost:10.0` stalled GLFW
initialization; the ordinary systemd experiment environment has no DISPLAY.
Both aborted startup attempts and their stop decisions are retained. This
environment correction does not change the simulator or policy code.

Output: `projects/rss-few-pair-recovery-20261120/scene/`.
The initially prepared 8 mm vertical-clearance proposal was not executed or
adopted; it is retained as an unused diagnostic proposal.

Example:

```bash
python3 study/iros2027/v2/rss_few_pair_recovery_20261120/scene/isolated_preflight.py \
  --layouts projects/rss-few-pair-validation-20261120/mustard_right/design/collection-layouts.json \
  --output projects/rss-few-pair-recovery-20261120/scene/collection-isolated-v1
```

Only an already passed child can be skipped on repeated invocation. A failed
or partial child requires a separately documented recovery decision and fresh
attempt directory.
