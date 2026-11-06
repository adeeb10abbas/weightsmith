# Language-binding edit (cube then banana): scripts

Run with the weightsmith `owsp` tool against project `cube-then-banana`. Every script runs in the `owsp py` CPU sandbox, which provides `acts()`, `load()`, `base()`, `save()` and `bf16()`.

| Script | Purpose | Outcome (offline, 21 premature-release / 9 carry / 19 phase-2 decisions) |
|---|---|---|
| `dirs.py`, `phase_scores.py` | Cube-vs-banana "target tag" axis per VLM layer; scores per decision group | Diagnostic |
| `steer_edit.py` | Rank-1 push along that axis at image tokens (`st11a5`, `st1016a4`) | 21/21 still release; correlated axis is not causal |
| `v2s_edit.py` | Mean explicit-minus-original shift at state tokens, rank-1 per layer (`v2s-b`) | 6/6 still release (6-item check) |
| `v2m_edit.py` | Per-slot shift at 40 prompt/state slots, least-norm multi-key (`v2m-h`, `v2m-f`, `v2m-2`) | 15 / 13 / 8 of 21 release |
| `v2mi_edit.py` | `v2m` plus 32 image cells (**`v2mi-f`**, the submitted edit) | **0/21 release**; carries and phase 2 unchanged |
| `v2mi2_edit.py` | Ablation variants (image-only, layers 8–16, half strength) | See report §5.5 |
| `v2p_edit.py` | Image-cell edit with preservation keys from 5 other tasks (`v2p-f`) | 1/21 release; closed loop 24/32 vs 3/32; see report §5.6 |
| `chk.py`, `vs_v2.py`, `gap*.py` | Offline checks | — |

The edit uses capture `c2-multi`: 6 premature-release decisions, each replayed under 8 instructions, with VLM residuals and MLP activations at all layers. Keys are mean activations under the original instruction. Values are the explicit instruction's per-layer residual increments minus the original's. Each layer gets `dW = K^T (K K^T + lam I)^-1 V` with `lam = 1e-3 * trace(KK^T)/n`, layers 0–16, alpha 1.
