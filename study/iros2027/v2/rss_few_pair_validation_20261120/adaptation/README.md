# Three independently constructed edits per qualified goal

Run from the repository root after the collection controller has published a
qualified `source-status.json`:

```bash
OPENBLAS_NUM_THREADS=4 python3 study/iros2027/v2/rss_few_pair_validation_20261120/adaptation/run.py --goal temporal --gpu 0
```

The same command accepts `cube_behind` and `mustard_right`. It never collects,
restarts, retunes, or evaluates a policy. Unqualified goals stop before any GPU
access. Native replay and each activation capture hold the repository's shared
GPU lock. An incomplete stage is retained and stops automatic continuation.

All outputs live in `projects/rss-few-pair-validation-20261120/<goal>/adaptation/`:

- `protocol-addendum.json`: a hash-bound correction to the frozen alignment
  description, preserving the exact archived construction recipe.
- `teacher-parity/`: exact fixed16 replay of recorded original-policy actions,
  fresh singleton clarified-policy actions, normalization checks, and receipts.
- `captures/{original,teacher}/`: nine paired activation states, ordered exactly
  as `correction.json`; each capture includes 96 prompt positions and 17 layers.
- `parity.json`: action equivalence and token ID/mask checks for every source.
- `weightsmith/{g0,g1,g2}/workspace/`: three deployable candidate workspaces.
- `weightsmith/candidates.json`: the supervisor interface, including candidate
  IDs `weightsmith_g0`, `weightsmith_g1`, `weightsmith_g2`, absolute workspace and
  capture paths, source indices/pair IDs, gate paths, and artifact hashes.
- `status.json`: running, stopped, or complete, with the current stage.

The validator accepts exactly the first nine eligible pairs and the frozen
random partition into three disjoint triplets. It verifies collection receipts,
source hashes, matching scene/noise identities, selected failure decisions,
original/clarified prompts, and the full recorded fixed16 inference context.
It does not use either held-out or candidate-development outcomes.

Each edit averages the original MLP keys and clarified-minus-original residual
increments over its own three selected states. The key system has 72 rows: three
opening template positions, a fixed 37-position window beginning at `State`, and
32 image cells (two cameras, 16 grid cells each; camera-mean rows are excluded).
The fixed prompt window deliberately preserves the archived recipe's padding
or tail truncation. The audit found 38, 39, or 40 valid positions among the 40
selected positions across the old 12 states. Paired token IDs and validity flags
must match at every selected position; the window offsets must be consistent
across all nine fresh states. This corrects the frozen protocol's mistaken phrase
"common valid State/Action suffix" without changing collection or its results.

Layers 0–16 use relative ridge 0.001, merge multiplier 1, and BF16 rounding of
base plus update. There is no gradient training, strength search, source subset
search, or outcome selection in this stage.

The CPU compatibility gate rebuilds every deployed tensor of the archived N=3
workspace from its first three captured states and requires exact array equality
at all 17 layers. It writes only a new compatibility receipt:

```bash
OPENBLAS_NUM_THREADS=4 python3 study/iros2027/v2/rss_few_pair_validation_20261120/adaptation/compatibility.py
python3 -m unittest discover -s study/iros2027/v2/rss_few_pair_validation_20261120/tests -p test_adaptation.py -v
```

Native action preparation reuses the old scaling routine verbatim, replacing
only its incompatible 1/3/6/12 nested-prefix schema reader with the explicit new
nine-source validator. No LoRA training function is invoked. The numerical
path, source validators, new adapter, and historical helper files are recorded
in the construction seal. Exact replay is required within 1e-6, and instrumented
versus native clarified singleton actions within 0.01, as in the prior study.
Singleton-versus-recorded fixed16 differences are diagnostic, not an identity
gate. Passing these checks establishes numerical preparation, not improved robot
performance; that requires the separately frozen evaluation controller.

The owned `capture_worker.py` is a copy of the existing activation worker with
only its import-root path and maximum captured prompt-position count changed
(64 to 96). The forward computation and activation extraction are unchanged;
a CPU test enforces that exact source delta. Native/instrumented teacher-action
parity remains mandatory for the 96-position captures.
