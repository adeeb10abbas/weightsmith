# Edit zoo

Published weight edits for OpenPI π0.5 (`pi05_droid_jointpos`), one file each, in the portable [`weightsmith.edit`](../docs/INTEGRATIONS.md#portable-edit-files) format. Each file reproduces, bit for bit, the candidate that was scored on held-out seeds: applied to the same base weights, it gives back that candidate's `owsp` fingerprint, which is printed in its held-out record.

| Edit | Task | What it changes | Values changed | Held-out, unedited → edited | Fingerprint | File |
|---|---|---|---|---|---|---|
| `ctrl-ctb-p1` | Cube then banana | Built blind by an autonomous editor (Claude Opus). A rank-1 change to the gripper column of the action output projection, `action.action_out_proj.kernel[:, 7]`, whose key responds to premature-release states | 1,024 in 1 tensor (0.00003%) | 11/64 → **27/64**: 22 fixed, 6 broke, p = 0.0037 | `4ca8023b0592b79f` | [`ctrl-ctb-p1.safetensors`](ctrl-ctb-p1.safetensors), 5.9 KB, in this folder |
| `v2mi-f` | Cube then banana | Closed-form instruction distillation, built by the operator session. A minimum-norm update to the MLP down-projections of VLM layers 0–16, `language.blocks.00–16.mlp.linear`, rank ≤ 72 per layer | 450,468,837 in 17 tensors (13.4%) | 27/128 → **106/128**: 81 fixed, 2 broke, p = 7×10⁻²². The same file on two other tasks: eyeglasses 22/64 → **44/64**, cube left of bowl 8/64 → **36/64** | `63db0c52a51306c9` | [`v2mi-f.safetensors`](https://github.com/adeeb10abbas/weightsmith/releases/download/edits-v1/v2mi-f.safetensors), 877.6 MB, release asset |
| `abl-imgl8` | Cube then banana | The same recipe with image-region keys only, in VLM layers 8–16 (`language.blocks.08–16.mlp.linear`), rank ≤ 32 per layer: 5.3M values as factors (0.16%) | 248,057,499 in 9 tensors (7.4%) | 19/128 → **102/128**: 86 fixed, 3 broke, p = 4×10⁻²² | `5bf8c4329c1e416a` | [`abl-imgl8.safetensors`](https://github.com/adeeb10abbas/weightsmith/releases/download/edits-v1/abl-imgl8.safetensors), 274.0 MB, release asset |

The task is RoboLab's `RubiksCubeThenBananaTask`, "Put the cube then the banana in the bowl". Held-out results pair the unedited and edited policy on identical simulator seeds that no editor saw; the operator scored them with `weightsmith/heldout.py`, and p is the exact two-sided McNemar test. The slots are 91008–91071 for `ctrl-ctb-p1`, 90000–90127 for `v2mi-f` and `abl-imgl8`, and 90000–90063 for the two other tasks.

Files over 25 MB are assets of the GitHub release [`edits-v1`](https://github.com/adeeb10abbas/weightsmith/releases/tag/edits-v1), which also carries `ctrl-ctb-p1.safetensors`. If a download link returns 404, the release has not been published yet. [`SHA256SUMS`](SHA256SUMS) lists all three files.

## Check a download

```bash
edits/verify.sh path/to/v2mi-f.safetensors                              # checksum, metadata, held-out record
edits/verify.sh --download v2mi-f abl-imgl8                              # fetch into edits/downloads/ first
edits/verify.sh path/to/v2mi-f.safetensors --workspace $BASE_WORKSPACE   # also apply it and compare fingerprints
```

[`verify.sh`](verify.sh) runs from a checkout of this repository and needs Python with numpy. For each file it checks that:

1. its sha256 is listed in `SHA256SUMS`;
2. it is an exact `weightsmith.edit` file, and the fingerprint it records appears in a held-out record under `study/results/*/heldout/` ([format](../results/README.md#file-formats));
3. with `--workspace`, applying it to your base export reproduces that fingerprint, and every tensor matches its recorded sha256. The candidate goes to a temporary directory (it honours `TMPDIR`; `v2mi-f` needs about 2.3 GB) and is deleted afterwards.

It exits non-zero if any check fails. For the file in this folder:

```text
$ edits/verify.sh edits/ctrl-ctb-p1.safetensors
edits/ctrl-ctb-p1.safetensors
  ok    sha256 8ed647a7e03497db... matches the published ctrl-ctb-p1.safetensors (edits/ctrl-ctb-p1.safetensors)
  ok    edit of candidate p1: 1,024 values in 1 tensor(s), exact: yes, fingerprint 4ca8023b0592b79f
  ok    held-out record study/results/ctrl/heldout/ctrl-ctb__p1-RubiksCubeThenBananaTask-91008-64.json: RubiksCubeThenBananaTask, slots 91008-91071, 11/64 -> 27/64 (22 fixed, 6 broke, exact McNemar p = 0.0037)
  skip  apply: pass --workspace $BASE_WORKSPACE to check the applied fingerprint
all 1 passed
```

Without the repository, `sha256sum -c SHA256SUMS` (on macOS, `shasum -a 256 -c SHA256SUMS`) in the folder that holds the files checks the checksums. Add `--ignore-missing` if you downloaded only some of them.

## Apply an edit

You need the public checkpoint (sha256 values in [`third_party/pi05_droid_jointpos.sha256`](../third_party/pi05_droid_jointpos.sha256)) and the editable bf16 export made from it once by `scripts/prepare_workspace.sh` (`$BASE_WORKSPACE`, about 11 GB; see [docs/SETUP.md](../docs/SETUP.md)). Applying an edit needs only numpy (`pip install -e .`). Running the edited policy needs the GPU runtime.

```bash
weightsmith edit info  v2mi-f.safetensors      # what it changes, tensor by tensor
weightsmith edit apply v2mi-f.safetensors --workspace $BASE_WORKSPACE --out $WSP_ROOT/cands/v2mi-f
#   ... fingerprint 63db0c52a51306c9: identical to the source candidate
bin/owsp diff v2mi-f && bin/owsp eval v2mi-f --n 64   # closed loop, paired with base on the same practice seeds
```

The new candidate directory has the same layout as one made by `owsp new`, so `owsp` and `weightsmith/heldout.py` accept it. You can also apply an edit outside the `owsp` layout:

- **To an OpenPI params tree**, in Python:

  ```python
  from pathlib import Path

  import numpy as np
  from openpi.models import model as openpi_model

  from weightsmith import edits

  checkpoint_dir = Path("/path/to/pi05_droid_jointpos")
  params = openpi_model.restore_params(checkpoint_dir / "params", restore_type=np.ndarray)
  edits.load("v2mi-f.safetensors").apply_to_params(params)   # edits in place; returns the edited paths
  ```

- **To arrays keyed by tensor name**, such as `language.blocks.08.mlp.linear`: `weightsmith edit apply FILE --arrays base.npz --out edited.npz`.

Every route checks each edited tensor's base values and result against the sha256 recorded in the file, and stops on a mismatch. `--no-verify` skips those checks, and the result is then not the scored edit.

## What "bit for bit" means

- **Exact values.** A file stores a low-rank (or dense or sparse) delta per tensor, plus exact fix-ups for bf16 rounding. Applying it gives back the scored candidate's bf16 values exactly, and the per-tensor sha256 values of base and result are checked.
- **The fingerprint.** `owsp` fingerprints a candidate by the first 16 hex digits of a sha256 over its edited tensor files. `heldout.py` froze each candidate and wrote that fingerprint into the held-out record, and `weightsmith edit apply --workspace` reproduces it.
- **Checked.** On 2026-11-07 each file was re-applied, on the CPU of the reference GPU host and with the code in this repository, and reproduced its fingerprint. The release assets' sha256 digests, as reported by GitHub, equal `SHA256SUMS`.
- **Not identical episodes.** GPU physics and rendering are not bit-deterministic ([BENCHMARK.md §4](../BENCHMARK.md#4-slots-and-seeds)), so a fresh evaluation of an identical edit gives different counts. The unedited policy itself scored 27/128 and then 19/128 on the same held-out slots.

## The edits

**`ctrl-ctb-p1`**, by an autonomous editor working blind.
- **Who built it.** A fresh Claude Opus session (Claude Code, extra-high effort) in the [controlled study of seven blind editors](../study/results/ctrl/README.md). It had no access to the paper, the repository or the other editors' notes.
- **Diagnosis.** Replayed under "Put the cube in the bowl", 16 of 16 recorded premature releases became holds. The editor concluded that the compound instruction makes the policy treat a grasped cube as a finished step.
- **Edit.** `K[:, 7] += c · a` on `action.action_out_proj.kernel` [1024 × 32], so only the gripper channel changes. The key `a` is a least-norm solution of class-mean constraints on recorded activations: 1 on premature releases, 0 on bowl releases, stage-2 approaches and initial approaches. Built by [`build_p1b.py`](../study/results/ctrl/ctb/scripts/build_p1b.py); the editor's account is in [`FINAL.md`](../study/results/ctrl/ctb/FINAL.md).
- **Results.** Development: 4/32 → 10/32 (8 fixed, 2 broke) on slots 70000–70031. Held-out: 11/64 → 27/64, with Wilson 95% intervals of 10–28% and 31–54%; Holm-adjusted over the seven tasks, p = 0.026. It improved in both halves of the alternating run order (7 → 10 and 4 → 17). [Held-out record](../study/results/ctrl/heldout/ctrl-ctb__p1-RubiksCubeThenBananaTask-91008-64.json).

**`v2mi-f`**, closed-form instruction distillation.
- **Who built it.** The operator session (also Claude Opus), following the editor rules. It was not an autonomous editor run.
- **Edit.** At each of the 17 layers, ΔW = Kᵀ(KKᵀ + λI)⁻¹V, built from 6 recorded premature-release decisions.
  - Keys: 72 mean MLP inputs under the original prompt, at 40 prompt/state positions and 32 image regions.
  - Values: the layer's residual change under the explicit wording ("Put the cube in the bowl then put the banana in the bowl") minus the original's.
  - Script: [`v2mi_edit.py`](../study/results/opus/ctb/v2mi_edit.py). The build step of `weightsmith distill` rebuilds the same candidate bit for bit ([docs/DISTILL.md](../docs/DISTILL.md#status)).
- **Results.**
  - Practice: 25/32 vs 3/32 (22 fixed, 0 broke).
  - Held-out on cube then banana: 106/128 vs 27/128, paired difference +61.7 pp [+52.3, +70.3].
  - Held-out on two other tasks, with the same fingerprint (scored as `ctb-v2mi`): eyeglasses 44/64 vs 22/64 (31 fixed, 9 broke, p = 7×10⁻⁴) and cube left of bowl 36/64 vs 8/64 (33 fixed, 5 broke, p = 4×10⁻⁶).
  - A control with random directions and the same layers, rank and singular values scored 9/64, no gain ([`controls/`](../study/results/opus/controls/)).
  - Held-out records: [cube then banana](../study/results/opus/heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json), [eyeglasses](../study/results/opus/heldout/thr-pickglasses__ctb-v2mi-PickGlassesTask-90000-64.json), [cube left of bowl](../study/results/opus/heldout/tx-rubikscubeleftofbowl__ctb-v2mi-RubiksCubeLeftOfBowlTask-90000-64.json).

**`abl-imgl8`**, the 0.16% variant.
- **Edit.** The `v2mi-f` recipe with only the 32 image-region keys, in layers 8–16. Script: [`v2mi2_edit.py`](../study/results/opus/ctb/v2mi2_edit.py).
- **Size.** "0.16%" counts the low-rank factors: rank 32 × (16,384 + 2,048) × 9 layers = 5.3M values. Applied to the weights, the update is dense and changes 248M stored values (7.4%).
- **Results.** Practice: 25/32 vs 3/32 (22 fixed, 0 broke). Held-out: 102/128 vs 19/128 on the same slots as `v2mi-f`, scored after it. [Held-out record](../study/results/opus/heldout/cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json).

## Caveats

- **One checkpoint, simulation only.** The edits were measured in RoboLab on Isaac Sim with this repository's pinned stack, never on a physical robot.
- **Only for these base weights.** The sha256 checks reject any other base. Each file's metadata also records the base export's digest (`editable_sha256` `7f887d21…`).
- **Built for one task.** `v2mi-f` helped 3 of the 15 tasks it was run on and left the rest within noise. It fixed "left of the bowl" but not "right of the bowl" ([report §5.5](../study/blog/failure-levels.md#55-the-results)). Run your own regression checks before relying on an edit elsewhere.
- **Not sparse.** `v2mi-f` changes 13.4% of the scalar values and `abl-imgl8` 7.4%, both at low rank. `ctrl-ctb-p1` is the sparse one, and it recovers only part of the effect: 42% held-out success against 17% unedited, where `v2mi-f` reached 83% on other slots.
- **Rows are not directly comparable.** `ctrl-ctb-p1` was scored on 64 pairs, the other two on 128 pairs. `abl-imgl8` reused `v2mi-f`'s held-out slots.
- **Noise.** About 30% of paired outcomes flip with unchanged weights ([report §7](../study/blog/failure-levels.md#7-how-not-to-fool-yourself)), so treat a small replication run as a filter, not as evidence.
- **Why the large files are large.** The low-rank factors take 2–10 MB per layer. Most of each file is exact fix-ups for bf16 rounding. A lossy export (`--rank N --lossy`) would be far smaller, but it is not the scored edit, and no lossy version has been evaluated.

## Add an edit

1. On the GPU host, export the frozen candidate with `weightsmith edit export $WSP_ROOT/cands/NAME -o NAME.safetensors --rank auto`. Check that `weightsmith edit apply --workspace` reproduces the fingerprint in its held-out record.
2. Put the held-out record, unmodified, under `study/results/<study>/heldout/`. Add the file's sha256 to `SHA256SUMS` and a row to the table above.
3. Commit files up to about 25 MB here. Upload larger ones to a release, with their sha256.
