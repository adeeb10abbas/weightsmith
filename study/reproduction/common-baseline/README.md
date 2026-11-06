# Shared fixed16 baseline evidence, version 1

This is a **separate, permitted common baseline evidence transfer** for a teammate's model comparison. It contains only the completed original-policy fixed16 development cohort and its readiness metadata. It is not part of the sealed source handoff or the compact publication ZIP. It contains no adaptive proposals, editor conversations, candidate outcomes, activation traces, provider credentials, or held-out evidence.

The source is exactly `runpod2:/tmp/weightsmith-codex-20261105/projects/rq123-v2-development-v1`. Its 32 development slots are 30016–30047, task `StackYellowOnRedTask`, instruction “Stack the yellow block on the red block”, and gripper threshold 0.5. The reference manifest is pinned to SHA-256 `579543173d6fa7f119f6ba6d39e6abb725d403b8e258b64968c12602c695771f`. The original terminal simulator flags report 17 successes; they do not establish stable released placement. Event labels are raw simulator feedback and may contain annotation defects; they must not be treated as a separate ground-truth oracle.

The evidence tar contains 427 original manifest-listed files: 32 observations and summaries, the exact JSONL index, the full source row array, 120 native packet directories (packet metadata, raw request arrays, raw native action arrays), and the native runtime binding. Five additional original files preserve the reference manifest, baseline config/status, identity report, and fixed16 GPU-validation report. The first tar member is a transport inventory with every file's byte count and SHA-256. Original JSON and NPZ bytes are unchanged. The transfer streams directly to a local file and creates no remote temporary file, remote write, or GPU job.

`transfer_receipt.json` records the tar digest and full inventory checks. `packet_validation.json` records local CPU verification of every recorded request against its actual packet, position, sampling seed, inputs and float32 client action recording; it also requires every real position of every included packet to map exactly once to a baseline request. `packaging_receipt.json` binds the archive, helper, tests, documentation and verification reports. Read those receipts before using the archive. A partial transfer or failed verification is not a usable package.

## Verify and extract

Use the helper here for this exact baseline only. These commands are local CPU operations. Extraction requires a **new** destination and fails on existing directories, unsafe paths, symlinks, nonregular members, missing or duplicated entries, or changed bytes.

```sh
python3 common_reference.py verify \
  --archive common-fixed16-baseline-evidence.tar.gz \
  --extract-to /new/operator-only/common-fixed16-baseline

python3 common_reference.py verify-packets \
  --root /new/operator-only/common-fixed16-baseline \
  --codec /source-handoff/bundle/weightsmith/extensions/batch_capture_v2/runtime/weightsmith/lab/batch_context_v2.py
```

The second command requires NumPy and checks the codec's exact sealed SHA before importing it. It loads no model or GPU library. The archive prefix is removed by the checked extractor, so the destination directly contains `config.json`, `data/`, `study/`, and `v2_runtime_binding.json`. Do not replace these checks with an unchecked tar extraction. No command in this evidence package imports into an editor, changes a runtime, or starts an experiment.

## Runtime compatibility and comparison meaning

The native binding SHA is `c982b471408232b1a9d5361b320f90932c83e92cefce15d433df65783254ffa0`; the sealed source manifest SHA is `017e7ab14bdfd330bb11b7addb23f6a04fa4a07374442e41aa253617e4def754`. The included GPU report SHA is `49882deb43e51fec6c4a180ce48073be63c7f64ffafc3eaa7ac4b059c58b5fba`. That report passed on the **original host only**, with 31 real positions in five actual heterogeneous packets (real sizes 1, 2, 4, 8, and 16, all padded to 16). It does not validate a teammate's host, GPU, installation, provider adapter, or new runtime identity.

For a **matched-evidence comparison**, both editors must receive these exact baseline bytes and the same scientifically bound runtime, source, checkpoint, task, editing interface and budget. The coordinating operator must first verify the actual environment, checkpoint and source against the native binding and preserve the runtime's packet identities. Then the sealed fork's `scripts/study_development.py:copy_reference_to_editor(source,destination)` can import this checked extraction into an already configured, unused editor project, copying observations, packets and the native binding unchanged and setting the new editor's episode/capture budget to zero. That helper itself is not a host-readiness test. The destination must have the exact v2 task/config fields and must be fresh; importing cannot reset a used search. The provider receives only its clean scientific brief and broker-mediated permitted evidence, not operator receipts or other editors' artifacts. Freeze the provider's exact model ID, effort, adapter, brief, baseline and runtime hashes before search.

For a **different host or newly bound runtime**, first perform that host's full CPU/source/checkpoint and GPU validation, then collect a fresh fixed16 baseline under its own native identity. Report it as a separate runtime/cohort comparison; it is not identical shared baseline evidence even if task, seeds and code match. The sealed replay checks deliberately reject packets with a different runtime binding. Do not rewrite packet hashes, native bindings, recorded checkpoint paths or readiness results to make a cross-host import pass. If a future cross-host shared-evidence bridge is scientifically needed, it requires a separate prospective protocol and genuine compatibility validation; none is provided here.

The original binding names an original-host checkpoint cache path; baseline rows record its resolved RAM-backed checkpoint location. These are preserved provenance, not portable installation instructions. Never create aliases to somebody else's storage or modify original-host paths to satisfy them.

This package adds no permission to read held-out results or to launch final evaluation. Discovery and final closure remain separate operator-controlled steps. Failed/partial searches and their actual costs must be retained. No teammate provider has been run or certified by this packaging task.
