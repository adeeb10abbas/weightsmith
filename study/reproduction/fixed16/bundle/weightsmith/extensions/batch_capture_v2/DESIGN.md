# Batch-aware activation capture v2 — separate prospective runtime

Protocol identity: `wsp-batch-capture-v2-fixed16`. This extension does not change or retroactively validate the original fifty frozen scientific files. Its runtime, baseline, search, and final cohorts are separately versioned. CPU tests passing does not establish GPU numerical parity.

## What changes

All new native policy requests execute with batch dimension16. A real microbatch has1–16 requests; padding repeats request0 **and its existing noise**. Each real request draws `normal(key(sampling_seed), (1,H,D))` independently. A single normal draw of shape `(B,H,D)` would be a different sampling procedure and is not used. The action chunk remains15×8 after the unchanged output transform. The original server's claim that batching is always equal to unbatched inference is not adopted.

At the actual server inference boundary, the runtime snapshots ordered raw inputs before transforms and records every real output at its original precision. Packets include full camera/proprioception/prompt/seed values, content hashes, real and padded counts, position-to-source mapping, checkpoint identity, current policy fingerprint, runtime-binding hash, and server-source hash. Each client request record points to a packet ID, packet hash, and actual position. There is no inferred grouping from old episode timestamps and no claim that a duplicated historical request reconstructs a heterogeneous historical microbatch.

Packet layout is `project/data/batch_packets/<32hex-id>/{packet.json,requests.npz,recorded_actions.npz}`. Episode links are relative IDs; baseline imports copy packet files and `v2_runtime_binding.json` byte-exact. The CPU `read_packet(directory, expected_sha256)` API returns `(metadata, requests, actions)` and validates hashes, layout, ordering, types, precision, and symlinks. Captures require the episode candidate and actual deployment fingerprint to match the loaded policy, including edited candidates.

For each unique requested packet, the worker runs one native forward and one instrumented forward using the **same complete inputs and explicit per-request noise**. It compares native↔recorded, instrumented↔native, and instrumented↔recorded on **every real position**, whether or not that position was selected by the editor. All checks use the fixed0.02 maximum action-coordinate tolerance. Raw joint/gripper errors and binary gripper flips remain separately interpretable. The0.02 gate checks final actions, not bit-identical internal native activations; numerical parity alone says nothing about closed-loop causal improvement.

Requested layer/denoise/token selection occurs inside JIT before host export. Raw trace arrays preserve original dtype/bits, every selected token, and all16 positions, with padding mapped explicitly. Bfloat16 values are stored as uint16 words with a dtype schema. The typed-tools compatibility features are float32, exactly representing BF16 values before any explicitly requested token mean. This differs from the v1 float16 export, so cross-version differences cannot be attributed solely to batch shape.

The worker publishes `capture_validation` only after complete checks. Required fields are protocol ID, `status=passed`, `all_real_positions_checked=true`, `parity_tolerance=0.02`, and `zero_skipped=true`. Parent-owned v2 contrast/direction tools require this marker and exact source-file hashes. Failed captures remain archived and charged; edited captures have no historical-parity exemption.

## Resources and accounting

The logical activation-item cap is distinct from actual forward work. Every unique packet reserves32 padded forward items (16 native +16 instrumented), before dispatch. Reserved capacity, native/instrumented calls started, and synchronized calls completed are recorded separately; a failed dispatch does not turn all reserved capacity into executed work. A packet may support several selected logical items without rerunning it. Estimated raw+compat output must fit512MiB per call, and persisted captures share an operator-fixed1GiB project cap. Failed and partial capture files count toward that cap; no cleanup silently restores budget. Native packet observations have a separate archival lifecycle and are not miscounted as activation trace bytes.

Full18-layer×10-step default traces exceed the per-call limit even for one packet. Explicitly request a small set of sites. A schema example is `action_layers:[0], denoise_steps:[0], action_tokens:"mean"`; these are illustrative indices, not a suggested mechanism or identified causal layer. Select a few recorded requests sharing packet context where useful. The whole real packet is still checked. A request rejected by resource/provenance checks is a charged attempt, with no hidden GPU forward.

The current study has one16-environment simulator client. The fixed16 runtime rejects actual merged batches larger than16; expanding to concurrent clients requires a separately tested queue-splitting policy rather than truncation or changing the batch shape.

## Executable validation gate (operator only)

`validate_gpu.py` requires `--execute-gpu`, the existing private process-group guard (`WSP_MANAGED_PROCESS_GROUP=1`), an immutable runtime binding already present in the fresh `rq123-v2-validation-vN` project (positive integer version, directly under the original isolated root), and the original environment file. It must be launched with the original isolation root and an external600-second cap. Neither this design note nor CPU tests dispatch any GPU work.

Within that guard, execute using the existing OpenPI interpreter:

```sh
python extensions/batch_capture_v2/validate_gpu.py --execute-gpu \
  --source-project /tmp/weightsmith-codex-20261105/projects/rq123-development-v1 \
  --project /tmp/weightsmith-codex-20261105/projects/rq123-v2-validation-v1
```

The helper takes the first recorded request from the first16 distinct numerically sorted original base slots as **offline diagnostic inputs**. It creates new native fixed16 packets for real counts1,2,4,8,16 with mixed observations and distinct recorded seeds. It verifies exact native repeats and a verified zero-edit workspace load, then invokes the production worker's `capture` function on31 logical items across5 authentic new packets. The helper passes its single loaded server into capture; the worker resets it to the checkpoint before matching packet identity. No second policy is allocated. Production captures use the same function and construct one server when none is passed.

The validation spec records raw traces for layers16/17, denoise step9, all action tokens and all batch positions; compatibility means are separately labeled. This site choice bounds validation output and is not a hypothesis offered to the editor. Native repeats/identity checks plus native/instrumented capture total400 padded forward items if all operations complete. The report is `project/study/capture_validation.json`, with status, immutable runtime-binding hash, checks, source observation hashes, attempted work, failure details, and elapsed time. Diagnostic request-link summaries explicitly contain no simulator outcomes. These inputs, packets, and traces must never enter the new editor baseline.

Only a passing validation permits the separate new32-episode baseline (slots30016–30047). Its exact packet links and source artifacts must be verified and frozen before search. Runtime binding exists before validation; a later search binding adds validation and baseline hashes without a circular dependency. Test slots100000–100063 remain inaccessible to discovery. GPU parity, fresh baseline readiness, and closed-loop outcomes remain independent gates. A failed validation is retained and requires an explicit versioned correction; no silent retry promotes an earlier failure.

## CPU checks

Run with a NumPy-enabled CPU Python:

```sh
python -m unittest discover -s extensions/batch_capture_v2/tests -v
```

Checks cover per-request noise/padding, heterogeneous packet roundtrips and exact output precision, metadata/array/order corruption, symlinks, strict same-policy and runtime matching, legacy recording rejection, all-real-position failures, cast-aware error accounting, raw trace preservation, packet deduplication, project/per-call limits, retained failed-attempt charging, editor tool boundaries, and baseline packet-link validation. Bfloat16 bit preservation has an optional CPU test requiring `ml_dtypes`; absent that package, the test is explicitly skipped rather than counted as validated.

## Candidate attribution gate

Every completed candidate episode now passes the shared CPU packet-link verifier before packaging or selection. It binds the actual candidate/server fingerprint, complete slot-derived sampling sequence and ordered15-step request stream, task/prompt/threshold, exact raw input content hashes, runtime binding, full packet metadata, and native responses at client float32 precision. Missing/corrupt provenance or an incomplete cohort stops the attempt. The full requested episode capacity is charged before dispatch and its attempt/row/error audit is retained; there is no automatic retry or successful outcome index for failed validation.
