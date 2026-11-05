# Operator setup and readiness gates

All commands here are future operator actions on a **fresh teammate-owned deployment**. Packaging did not execute them. Do not point them at the original Runpod deployment or another project's environments/storage. This source package deliberately contains no recordings; actual diagnostic inputs and a new baseline are still required before search.

## Install the pinned stack

Use Linux with two compatible NVIDIA devices visible as 0 and 1, an existing functioning driver/Vulkan ICD, Git LFS, `uv`, Python, sufficient host RAM for checkpoint restore, and a private disk budget that retains the 8 GiB reserve. The original installation used two 24 GB GPUs; this package supplies no new hardware benchmark or driver installation. A container cannot replace its host driver through a Python environment.

Create a fresh deployment directory and copy `bundle/weightsmith/` into it without rewriting any sealed source. Keep the relative `extensions/batch_capture_v2/runtime` layout. The root `scripts/` files are installation/resource-guard support; runtime scientific code lives in the extension. The root `operator/rq123-v2/design/` directory must retain the exact original protocol/addenda/brief bytes.

Write a private `wsp.env` from `examples/wsp.env.template`, replacing every placeholder with actual paths. Do not put provider credentials in it. The export belongs in `workspaces/`, because the source verifier rejects exports outside that owned directory. Keep HOME and CODEX_HOME unchanged.

The installation support is the same pinned source installer used by the earlier setup. Run its `sources`, `policy`, and `checkpoint` phases under the private guard. The `sources` phase also installs a legacy RoboLab checkout that is **not** the configured simulator. The actual simulator must be `RoboLab-isaac60` at the commit below with the supplied Isaac 6 patch and environment lock; do not run the older simulator profile.

```sh
# Run in the new deployment; DEPLOY is its absolute path.
python3 scripts/run_isolated.py --root "$DEPLOY" -- python3 scripts/setup_deps.py --phase sources
python3 scripts/run_isolated.py --root "$DEPLOY" -- python3 scripts/setup_deps.py --phase policy
python3 scripts/run_isolated.py --root "$DEPLOY" -- python3 scripts/setup_deps.py --phase checkpoint
```

Install `https://github.com/NVlabs/RoboLab.git` at commit `9cce00cadd5dc575fd2b8ea3836f5bdcde574229` under `third_party/RoboLab-isaac60`, apply `third_party/patches/robolab-isaac60/0001-Port-weightsmith-controls-to-Isaac-6.patch`, and fetch its LFS assets using the private guard. Synchronize `envs/robolab-sim-isaac60` with `uv sync --frozen --python 3.12.3`. The root controller environment needs NumPy 1.26.4; the simulator lock intentionally uses NumPy 2.3.1. Do not merge those environments. OpenPI is pinned to `xuningy/openpi@aa6420561529593114160d05e5ad155792b272f3` plus the supplied request-seed patch. All 26 checkpoint files must match `third_party/pi05_droid_jointpos.sha256`; no checkpoint bytes ship here.

Native Linux libraries may need host-specific setup. The optional root `scripts/setup_native_libs.py` supports its documented Ubuntu 24.04 pins in a private sysroot; it is not a general driver installer. Preserve failed setup attempts, inspect their cause, and explicitly version any required deviation before scientific execution.

## Bind and validate before collecting editor evidence

Run the restricted action-down export using the sealed fork's `scripts/prepare_study_workspace.py` under the root private guard; it invokes OpenPI with GPU visibility disabled and exports about 288 MiB. Do not use the old full-parameter export. Then run this package's CPU-only binding helper:

```sh
python3 /package/tools/bind_runtime.py --root "$DEPLOY" \
  --checkpoint "$DEPLOY/runtime/cache/openpi/openpi-assets-simeval/pi05_droid_jointpos" \
  --workspace "$DEPLOY/workspaces/action-down"
```

It creates a **new** `operator/rq123-v2/v2_runtime_binding.json`, checks all 77 source hashes and all 26 checkpoint hashes, and records the actual private environment/export manifests. The reference binding in this package is not a portable success attestation and must never be substituted for a new machine's actual binding. No source file is rewritten by binding. A failed binding attempt remains recorded and must be reviewed.

The frozen GPU validator requires 16 distinct recorded original-policy requests from a separately designated, operator-only readiness source project named `projects/rq123-development-v1`, with slots 20032–20063. Only its raw first-request inputs/seeds are used to generate **new actual fixed16 packets**. No diagnostic inputs become editor evidence. This package does not contain those recordings or claim they already exist on a teammate host. Obtain genuine own-host readiness recordings or an explicitly agreed outcome-free input transfer before proceeding; do not manufacture successful summaries or reuse another model's discovery evidence. If a new input-collection adapter is needed, version and test it outside the 77 sealed sources before GPU use.

Create a fresh `projects/rq123-v2-validation-v1` containing an unchanged copy of the new native binding. Under the original root's private process-group guard, global `runtime/locks/study-dispatch.lock`, and an external 600-second timeout, invoke the sealed `extensions/batch_capture_v2/validate_gpu.py --execute-gpu --source-project "$DEPLOY/projects/rq123-development-v1" --project "$DEPLOY/projects/rq123-v2-validation-v1"` using the OpenPI interpreter. Guard GPU 0; no competing owned or foreign work may be active. Validation tests real counts 1/2/4/8/16, native repeats, identity loading and the exact production capture worker. Require `study/capture_validation.json` to pass every gate and bind the new native SHA. Preserve all failures; a new version is an explicit correction, not a silent retry.

An actual identity-export report is also required by the baseline helper. Generate it with the sealed identity probe from real same-checkpoint readiness inputs; choose valid recorded request indices prospectively. CPU export correctness is not equivalent to a native GPU identity pass.

## Collect the new baseline and launch a separate editor

After genuine runtime and identity validation, run **the fork's** `scripts/study_development.py` with `--root "$DEPLOY/projects/rq123-v2-development-v1"`, `--identity-report ACTUAL_REPORT`, `--runtime-binding ACTUAL_NATIVE_BINDING`, and `--capture-validation ACTUAL_PASSED_REPORT`. Set `WSP_ROOT` to that new baseline path **before imports**. Use the original root's process-group guard, both-GPU resource guard, global dispatch lock and 1,800-second cap. Its v2 configuration collects only the new 32-slot base at threshold 0.5; there is no threshold search in v2.

Require a complete baseline, its packet-link checks and frozen reference manifest. Create the unused editor project at `projects/rq123-v2-adaptive-r1` with the fork's `configs/v2_editor.json`. Use the fork baseline module's `copy_reference_to_editor(source,destination)` to copy all episodes, actual packets and the native binding byte for byte with zero candidate charges. Copy the resulting exact `data/episodes/index.jsonl`, source `study/reference_base_manifest.json`, native binding and GPU-validation report into the **local operator** directory. Do not put them in the agent directory.

Freeze the new model enrollment and search-runtime binding before the first scientific provider call. Include exact native/source/protocol/addenda/base/index/validation/adapter/brief/mode hashes, actual model ID and effort, host/runtime settings and development slots. [MODEL_ENROLLMENT.md](MODEL_ENROLLMENT.md) describes the provider preparation/probe/search commands and comparison boundaries. A fresh provider probe is allowed before numerical setup because its broker permits only local brief inspection; it does not certify GPU readiness.

The extension intentionally has no automatic launch chain or held-out runner. Stop after discovery, preserve actual partial/completed status and all costs, and let the coordinating operator freeze a separate final plan. The bundled historical `scripts/study_final.py` and `configs/study_rq123/` are retained solely because they are sealed source dependencies; **do not use their older final schedule or slot ranges for this enrollment**.
