# Native batch packet contract

Protocol ID: `wsp-batch-capture-v2-fixed16`. The authoritative codec is `bundle/weightsmith/extensions/batch_capture_v2/runtime/weightsmith/lab/batch_context_v2.py`.

Every native inference call has 1–16 real ordered requests and a padded size of 16. Each request uses its own sampling seed to generate noise shaped `(1,H,D)`; padding repeats request zero and its existing noise. A single normal draw over `(B,H,D)` is not equivalent. No historical batch membership is inferred from timestamps or recreated by duplicating one observation.

The runtime records `project/data/batch_packets/<32hex-id>/` with `packet.json`, `requests.npz`, and `recorded_actions.npz`. Metadata binds actual request order, input content hashes, seeds, real/padded counts, padding positions, raw output dtype, deployment fingerprint, checkpoint checksum manifest, native server source and runtime-binding hashes. The codec rejects missing files, symlinks, altered metadata, inconsistent membership or mismatched array hashes.

Each episode `obs.npz` links every request using `batch_packet_id`, `batch_packet_sha256`, `batch_position`, `batch_real_count`, and `batch_padded_count`; these supplement the exact images, joints, gripper, prompt, sampling seed, action chunk, and control step. The complete request count must equal `ceil(terminal_steps/15)`, with steps `15*k` and the declared slot-derived sampling seeds. Candidate episodes undergo the same verification as the base.

Activation capture replays the **whole actual packet**, including unselected real positions. The policy must match the recording, for base and edited candidates. Native/recorded, instrumented/native and instrumented/recorded actions must pass the fixed 0.02 maximum-coordinate tolerance at every real position, with zero skipped checks. Only then is the required `capture_validation` marker published:

```json
{"protocol_id":"wsp-batch-capture-v2-fixed16","status":"passed","all_real_positions_checked":true,"parity_tolerance":0.02,"zero_skipped":true}
```

Requested raw traces retain their original dtype/bits; BF16 is encoded as uint16 words with dtype metadata. Compatibility arrays are float32, which exactly represents BF16 before any requested averaging. Gate tolerance concerns output actions, not proof that every internal activation equals a nonexistent uninstrumented trace.

Costs distinguish logical requested items, unique packets, reserved padded-forward capacity, actual forwards started, and synchronized forwards completed. Caps are 96 logical items, 512 MiB per capture and 1 GiB retained project capture arrays, including failed attempts. Failed captures remain ineligible for contrasts/directions. New base imports copy the entire packet tree and binding byte for byte; copying only episode observations is insufficient.

The source package contains no actual packets. Use a newly verified cohort or a separately agreed common **fixed16** evidence transfer. Original single-item replay records from the earlier handoff do not satisfy this contract.
