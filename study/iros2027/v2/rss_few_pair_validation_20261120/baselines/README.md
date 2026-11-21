# Simple inference corrections

These baselines test whether a fixed activation correction can recover the same
behavior as fitting a persistent weight edit. Both retain the original request
and unmodified checkpoint weights. They do add an inference-time intervention;
that distinction from the merged Weightsmith edit must remain explicit.

## Registered methods

**Mean instruction-embedding offset.** For the original and clarified requests,
obtain the native PaliGemma embeddings of instruction tokens inside the Pi0.5
`Task: ..., State: ...` input. Subtract the original token mean from the clarified
token mean. At execution, add the same vector to each instruction-token embedding
of the current request. The task prefix, state suffix, padding, and image tokens
are unchanged. A native tokenizer delimiter identifies the current instruction
span, including when wording changes. The direction depends only on the fixed
instruction pair and checkpoint: it is **one shared condition per goal**, not an
independent baseline for each source triplet. Qualifying executions establish the
working clarification but are not additional observations used to fit this vector.

**Constant image-cell residual offset.** At the preregistered VLM block 16, average
the clarified-minus-original post-block residual over the three selected source
states, preserving the 32 image cells (two cameras, each a 4×4 grid). At every
execution decision, add each saved cell vector to its corresponding 4×4 native
image-patch tokens. These offsets remain constant across states and requests;
they do not use the current observation to predict a correction. Other prefix
positions and the action expert are not directly modified. The offset is applied
after block 16, so block 17's KV cache incorporates it. A post-final-block offset
would not affect the cached prefix and is rejected by artifact validation.

This residual baseline uses a single fixed layer and image-cell support. It is
not an exhaustive test of steering layers, multi-layer corrections, or every
activation-steering method. The root study protocol fixes this scope before
development and held-out evaluation.

The root controller creates strengths 0.5, 1, and 2, evaluates them on the separate
development split, and includes the unchanged base as a zero-strength fallback.
Selection and held-out scheduling are outside this package. No command here
reads development success rates or held-out outcomes.

## Artifacts and runtime contract

Each artifact directory contains `baseline.json` and `direction.npy`. Metadata
pins the direction, source captures/tokenizer, construction identity, and native
hook code. Loading a malformed artifact, unknown directory, altered direction,
or changed native hook fails. Output directories are never overwritten.

`batched_server.py --port PORT [--workspace DIR]` is the entrypoint expected by
`weightsmith.rollout.start_server`. `server.py` also runs directly. The existing
workspace load command accepts ordinary weight-edit workspaces, baseline
directories, or explicit null for the unedited policy. A baseline load first
restores all edited weights, and switching back to base restores the original
native inference function. Every request uses the fixed-16 kernel shape, and
the first load-report JSON records the actual baseline path/fingerprint.

The residual hook uses the original Gemma Block, Linen scan, and Pi0 sampler.
Temporary integer layer markers are attached only to a merged inference module
to identify the block inside the native scan; they are not fitted parameters and
are never added to the server/checkpoint state. The embedding hook uses the
native Gemma embedding method. No transformer or sampling loop is reimplemented.

## Construction commands

Run from the experiment repository. Residual construction is CPU-only:

```sh
python3 study/iros2027/v2/rss_few_pair_validation_20261120/baselines/prepare.py residual \
  --original-capture ORIGINAL_CAPTURE_DIR --teacher-capture CLARIFIED_CAPTURE_DIR \
  --indices 0 1 2 --layer 16 --source-manifest SOURCE_MANIFEST --output RESIDUAL_DIR
```

Capture indices are explicit, distinct observations from three distinct episodes.
Both capture indices and source identities are saved. The averaged targets are
full residual differences, not residual increments or MLP-local differences.

Use the OpenPI Python for embedding construction; it acquires the shared GPU lock:

```sh
/home/ali/openpi-robolab/.venv/bin/python \
  study/iros2027/v2/rss_few_pair_validation_20261120/baselines/prepare.py embedding \
  --original-prompt 'ORIGINAL REQUEST' --clarified-prompt 'CLARIFIED REQUEST' \
  --gpu 0 --source-manifest SOURCE_MANIFEST --output EMBEDDING_DIR
```

Materialize each predeclared strength into its own immutable directory:

```sh
python3 study/iros2027/v2/rss_few_pair_validation_20261120/baselines/prepare.py strength \
  --source BASELINE_DIR --strength 0.5 --output STRENGTH_DIR
```

## Correctness gates

```sh
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES='' /home/ali/openpi-robolab/.venv/bin/python \
  -m unittest discover \
  -s study/iros2027/v2/rss_few_pair_validation_20261120/baselines -p 'test_*.py' -v

/home/ali/openpi-robolab/.venv/bin/python \
  study/iros2027/v2/rss_few_pair_validation_20261120/baselines/verify.py \
  --source-spec THREE_SOURCE_CORRECTION_SPEC --embedding EMBEDDING_DIR \
  --residual RESIDUAL_DIR --gpu 0 --output NEW_GATE_DIR
```

The native gate compares the new unedited server with the existing fixed-16
server, both baseline hooks at zero strength, and restoration back to base.
The positive control injects the **exact** clarified embeddings and token mask
and requires the native clarified actions to be reproduced. Zero and positive
control tolerances are 1e-6 physical action units; the nonzero smoke check requires
a detectable action change greater than 1e-6. Checkpoint-state arrays must retain
their original immutable identities. CPU tests additionally verify that the
residual intervention changes the next block's cache, not the current cache.

The first GPU gate passed with exactly zero parity error for both zero hooks,
the native wrapper, base restoration, and exact clarified-embedding substitution.
It also observed nonzero action changes from both baselines. Subsequent logging
and launcher integration changes are rechecked in a new receipt rather than
overwriting the first gate. These are technical inference checks, **not robot
episodes or evidence that either baseline succeeds at a repair task**. All
qualification artifacts live under the new study's `baseline-qualification/`.
