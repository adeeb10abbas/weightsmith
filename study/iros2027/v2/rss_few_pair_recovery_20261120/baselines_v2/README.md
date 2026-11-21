# Punctuation-safe baseline serving

The original baseline parser expected a comma token followed by `State:`.
The native tokenizer merges a trailing instruction period with the appended
comma into one `.,` token. This affected cube/mustard construction and the
temporal unfamiliar wording.

Version 2 finds the unique `State:` token anchor and excludes the preceding
separator token, which must be either `,` or `.,` under the pinned tokenizer.
Only instruction-token embeddings receive the offset. The `Task:` prefix,
separator, state/action suffix, padding, image inputs, and checkpoint weights
remain unchanged. Period/comma tokens are excluded as a whole because they
cannot be split at an embedding position. Temporal original/clarified masks
are exactly those used by the original implementation.

All files here are additive; old frozen code and candidates remain unchanged.
New baseline artifacts have schema `weightsmith-few-pair-baseline-v2`.

`qualify.py --gpu 0` creates
`projects/rss-few-pair-recovery-20261120/baseline-v2-qualification/passed.json`.
It compares every one of the twelve frozen temporal development candidates
against the original server at all declared strengths, using three recorded
states. All physical actions must be bit-identical. It also tests unchanged
native serving, zero corrections, and exact embedding replacement for all
twelve registered goal/instruction variants. No robot episodes are generated.

The public `load_artifact(path, verify_sources=True)` also accepts the original
temporal artifacts only after this qualification passes, only for the exact
pinned old implementation, only for artifact hashes listed in its certificate,
and only while the certified V2 implementation remains unchanged. Directions,
strengths, and stored artifacts are preserved. The corrected tokenization
metadata is added to an in-memory copy. Server fingerprints continue to use
the original `baseline.json` hash.

Construction and serving command-line interfaces are unchanged. Existing
temporal development workers continue with their frozen original runtime;
recovered workers use this server after qualification.
