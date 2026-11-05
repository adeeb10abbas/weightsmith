# Recorded discovery effort

Snapshot: 2026-11-05T18:15:15.947220+00:00.
Driver wall time includes reasoning, tools, simulator evaluations and waits. Candidate counts are reserved episode capacity, not assumed completions. Activation items are logical replay requests; repeated requests are charged each time. No monetary cost or inference-versus-training comparison is estimated.

| Run | Driver wall min | Reserved / verified candidate episodes | Reserved / logged activation items | Capture calls: validated / failed / unchecked | Logical items in those calls | Unique source episodes / recorded requests |
|---|---:|---:|---:|---:|---:|---:|
| adaptive-r1 | 61.58 | 96 / 96 | 80 / 80 | 2 / 1 / 1 | 20 / 32 / 28 | 36 / 76 |
| nonadaptive-r1 | 50.91 | 96 / 96 | 96 / 96 | 0 / 3 / 0 | 0 / 96 / 0 | 32 / 96 |
| adaptive-r2 | 67.32 | 96 / 96 | 96 / 96 | 1 / 4 / 0 | 4 / 92 / 0 | 28 / 68 |
| nonadaptive-r2 | 47.61 | 96 / 96 | 96 / 96 | 0 / 1 / 0 | 0 / 96 / 0 | 16 / 96 |

A validated capture checks and passes **every item**. A failed capture may contain individually passing items; the complete capture remains failed. “Unchecked” denotes an explicitly skipped diagnostic parity comparison, not a passed validation. Unique source episodes count recorded episode IDs, so repeated replay requests and multiple time steps do not create new simulator episodes. Different policy rollouts at one nominal sampling slot remain different recorded episodes.

| Run | Item checks: passed / failed / skipped / invalid | Unreturned reserved items |
|---|---:|---:|
| adaptive-r1 | 47 / 5 / 28 / 0 | 0 |
| nonadaptive-r1 | 84 / 12 / 0 / 0 | 0 |
| adaptive-r2 | 85 / 11 / 0 / 0 | 0 |
| nonadaptive-r2 | 86 / 10 / 0 / 0 | 0 |

Invalid items are a subset of failed item checks. These counts describe the measurement interface; they are not a measure of editor quality or independent task trials. Wall-time comparisons lack a matched fine-tuning/training baseline.

## Separate fixed16 follow-up

Different runtime and fresh baseline; do not pool these costs or outcomes with the original four runs.

Driver wall time includes reasoning, tools, simulator evaluations and waits. Candidate counts are reserved episode capacity, not assumed completions. Activation items are logical replay requests; repeated requests are charged each time. No monetary cost or inference-versus-training comparison is estimated.

| Run | Driver wall min | Reserved / verified candidate episodes | Reserved / logged activation items | Capture calls: validated / failed / unchecked | Logical items in those calls | Unique source episodes / recorded requests |
|---|---:|---:|---:|---:|---:|---:|
| fixed16-v2-adaptive-r1 | 63.43 | 96 / 96 | 80 / 80 | 4 / 0 / 0 | 80 / 0 / 0 | 72 / 80 |

A validated capture checks and passes **every item**. A failed capture may contain individually passing items; the complete capture remains failed. “Unchecked” denotes an explicitly skipped diagnostic parity comparison, not a passed validation. Unique source episodes count recorded episode IDs, so repeated replay requests and multiple time steps do not create new simulator episodes. Different policy rollouts at one nominal sampling slot remain different recorded episodes.

| Run | Item checks: passed / failed / skipped / invalid | Unreturned reserved items |
|---|---:|---:|
| fixed16-v2-adaptive-r1 | 80 / 0 / 0 / 0 | 0 |

Invalid items are a subset of failed item checks. These counts describe the measurement interface; they are not a measure of editor quality or independent task trials. Wall-time comparisons lack a matched fine-tuning/training baseline.

The fixed16 capture worker reserved/started/completed **192/192/192 padded forward positions**, including native and instrumented passes. Across capture calls it checked 96 real packet positions and 6 distinct native packets. Reused packets remain charged. Maximum three-way raw-action-array difference was 0.00488184; its mixed joint/gripper coordinates are not a common physical unit. These forwards are separate from the 400-position operator readiness probe and simulator episodes.
