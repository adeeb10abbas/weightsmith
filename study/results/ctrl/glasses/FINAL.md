# FINAL (glasses editor): PickGlassesTask, "Pick up the eye glasses"

## Submission: **p1**

## Exact edit (3072 values in 2 tensors; one MLP unit rewired; rank-1 per tensor)
Unit **2437** of `action.blocks.17.mlp_1` was the least-used unit in block 17: mean |activation| x ||down row|| = 0.0015 vs median 0.23. I repurposed it as a gated "lift while holding" unit:
- `action.blocks.17.mlp_1.gating_einsum[0, :, 2437]` (gate column, 1024 values) = alpha * (g_hat - thr * e_hat / kappa)
  - g_hat: the mean contrast (committed-HOLD minus OPEN) of the block-17 MLP input at the last denoise step (t=0.1). It is reconstructed from captured resid_out, mlp_hidden and the adaRMS modulations, then orthogonalized to e_hat and normalized.
  - e_hat: the mean-input (near-constant) direction, with kappa = mean x . e_hat = 32.4.
  - thr = 1.27, which is the maximum over all 40 captured open items (including 14 about-to-close items) plus a 0.5 margin. alpha = 2.97.
  - The unit is therefore ~ GELU(alpha * (Z - thr)): a one-sided gate that is exactly 0 in open states.
- `action.blocks.17.mlp_1.gating_einsum[1, :, 2437]` (up column, 1024 values) = e_hat / kappa, so the up projection is about 1.
- `action.blocks.17.mlp_1.linear[2437, :]` (down row, 1024 values, norm 1247) is the least-norm solution of 296 linear constraints. These go through the block-17 ffw adaRMS gate (~0.002), the final-norm gain and action_out_proj:
  - **zero velocity change on all 32 output dims at denoise steps 0..8**, so x_t and every later network input stay untouched and there is no denoising feedback;
  - at the last step (t=0.1), -0.1*dv = dq/scale for the 7 joints, and 0 for the gripper.
  - dq = least-norm FK-Jacobian joint change for **+8 cm EE z** with orientation held (Franka modified DH): [-0.017 -0.195 0.029 -0.017 0.123 -0.164 -0.104] rad. It applies at the reference (median committed-hold) unit activation.
  - scale = the normalized-delta-action scale calibrated in closed form from captures.
- After bf16 rounding, the early-step |dv| is at most 7e-4. `owsp diff p1`: 2048 + 1024 changed elements, nothing else.

## Mechanism story and evidence
- **Base failures (20/32):**
  - **Held-not-cleared (9):** the policy grasps the glasses by one temple arm and lifts the EE to only ~0.14–0.25 m. The glasses dangle and still touch the table (frames s11, s25), so `object_above(z_margin=0.05)` never fires. The policy then wanders and releases.
  - **Hesitation (~7):** the arm descends, the gripper command hovers at 0.3–0.8, and the arm retreats.
  - **Closed on nothing (persistent in s08, s19).**
- **Why a gated readout:** a linear "holding" readout on the action head is not selective. OPEN-high states project at -2 and OPEN-low states at -1, so a linear edit would push the arm down whenever the gripper is open. No existing unit separated the states. A GELU unit with a threshold gives a one-sided gate.
- **First design rejected offline:** in validation capture v1 (62 items), adding the push at every denoise step gave chaotic results. Holds moved dz -13..+23 cm, and 4/16 holds lost gripper-close tokens, because the flow model reacts to the shifted x_t. Restricting the push to the final Euler step, where x0 = x_0.1 - 0.1 v holds exactly, removes that feedback.
- **Offline prediction for the final edit (a1 captures):**
  - 0.000 change on all 40 open and closing-like items;
  - +0.7..+13 cm lift on 19/26 committed holds, 0 on 7;
  - gripper untouched.
- **Closed loop:** the mean planned chunk rise at held replans went from 9.6 to 10.8 cm. The held-not-cleared fixes show clear rise increases: s25 +0.9 -> +7.7 cm, s26 0 -> +4.9 cm, s10 +3.4 -> +12.1 cm.

## Development result (slots 70000–70031, n=32)
- **base 12/32 -> p1 18/32; fixed 9, broke 3** (paired on the same slots).
- Fixed: s07 s10 s13 s14 s15 s17 s19 s25 s26. Broke: s05 s12 s20.
- Pre-registered prediction: 14/32 (+3/-1). The direction held; the observed gain is larger than the mechanism alone explains.
- Caveats:
  - Several fixes (s07, s17, s19) were base hesitation or closed-on-nothing episodes, where the gate cannot act before a hold. Part of the +6 is likely closed-loop variance; same-weight reruns flip ~30% of pairs.
  - The sign test on the 12 discordant pairs gives one-sided p ~0.07.
- Confidence that p1 beats base on held-out slots: moderate-low, about 60–65%.
