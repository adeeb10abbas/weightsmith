# FINAL2 (round 2, stack3: Stack3RubiksCubeTask, "Stack the rubiks cubes in a tower")

## Submission: `p1`

## Exact edit
One action-expert MLP unit is repurposed as a state-gated "don't close here" detector: unit **2437 of `action.blocks.17.mlp_1`**. 3,072 values change, nothing else:
- `action.blocks.17.mlp_1.gating_einsum[0][:, 2437]` (1024 values, gate/GELU row) = `kappa * (d - theta * v)`.
  - `d` (unit norm) is the contrast in MLP-17 input space (post pre_ffw adaRMS norm). It is the mean over targets minus the mean over preservation decisions, after projecting out the top-256 eigenvectors of the within-item (token × denoise-step) covariance, which strips per-token flow noise.
    - Targets: 22 "empty gripper, at the 2-stack, z > 0.15, commanding close" decisions plus 8 "closed on air at the stack" decisions.
    - Preservation: 24 table-level grasps, 22 carries, and 16 holding-a-cube-over-the-stack decisions.
  - `v = (C + lam*I)^-1 mu / (mu^T (C + lam*I)^-1 mu)`: the most nearly constant linear feature of the MLP-17 input (x·v = 1 ± 0.01). It acts as the bias channel.
  - `theta = 0.634` = P max + 0.25·(T median − P max) in units of x·d / x·v; `kappa = 27.8`.
- `action.blocks.17.mlp_1.gating_einsum[1][:, 2437]` (1024 values, up row) = `v`, so the up channel is about 1.
- `action.blocks.17.mlp_1.linear[2437, :]` (1024 values, down row) = `0.81 * o`. `o` is the weighted least-norm solution (minimising Σ_s |g17_s ⊙ o|², the residual write) of 100 linear constraints, at each of the 10 denoise steps s:
  - `(1+sF_s) ⊙ g17_s ⊙ W[:,j] · o = 0` for the 7 joint dims j = 0..6 (zero first-order joint change);
  - `= 1` for gripper dim j = 7;
  - `g17_s ⊙ rbar_s · o = 0`, using both the target-mean and the all-mean layer-17 residual (no first-order change to the final-norm RMS).
  - Here `W` = `action_out_proj.kernel`, `g17`/`sF` are the layer-17 FFW gate and final-norm scale computed from the time MLPs, and `rbar_s` is the mean layer-17 residual.
- All values are bf16. Size: 3,072 of 3.35B parameters (one unit). Max |Δ| is 3.2 in gating and 280 in the down row; the large down row is needed because the layer-17 FFW adaRMS gate is about 0.01.
- Built by `scripts/build.py`; the forward model was verified by `scripts/recon.py` (final-norm reconstruction relative error 0.003, MLP-17 hidden 0.005).

## Mechanism story and evidence
- **Diagnosis (base, 16/32).** Every failure completes the first two-cube stack. The main failure mode is the stack re-grasp loop: after placing A on C, the empty gripper hovers at the stack and commands close at stack height. It either closes on air repeatedly (73 such replans across the 32 base episodes) or pulls the top cube back off, and cube B is reached late or never (70001, 70010, 70012, 70018, 70019, 70021, 70030). The secondary mode is a misplaced third cube followed by the arm parking and idling.
- **Offline evidence.**
  - Split-half by episode: the gated threshold fires on 0.0000 of held-out preservation tokens in both splits, and on 100% / 81% of held-out target tokens.
  - First-order simulation through the exact final norm, with bf16 weights and inputs: target close-fraction falls from 0.89 to 0.07, preservation decisions are unchanged (|Δa7| ≤ 0.001, no flips), and max |Δjoint| is 0.058.
  - Exact replay through p1 on 12 decisions not used in fitting:
    - All 8 preservation decisions (4 holding over the stack, 3 table grasps, 1 carry) are unchanged: a7 identical to 2 dp, |Δjoint| ≤ 0.005.
    - Of 4 held-out closed-on-air targets, 2 flip to open (a7 0.97→0.17 and 0.92→-0.04), 1 is weakened (0.97→0.68) and 1 is barely changed.
- **Closed loop (mechanism confirmed).**
  - Empty-gripper close decisions at the stack before picking B: 17 → 2.
  - Closed-on-air replans at the stack: 73 → 5.
  - Cube B reached in 27 episodes vs 24.
  - Loop episodes that already succeeded finish much faster (70004 415 vs 719 steps, 70009 280 vs 658, 70022 256 vs 435).
  - No premature-release signature: the fraction of mostly-open chunks while holding is 0.27 vs 0.30 at the stack and 0.05 vs 0.08 elsewhere.

## Development result
- **p1: 17/32 vs base 16/32 on the same slots. Fixed 6 (70012, 16, 17, 19, 21, 28), broke 5 (70000, 02, 13, 15, 26).**
- The net +1 is within same-weight rerun noise (about 30% of paired episodes flip, i.e. roughly 10 flips expected).
- The broken episodes pick B on time and then fail the third placement (the cube slides off the stack and the arm idles), which also happens in base. Third-placement success given B was picked is about the same (base 16/25, p1 17/28).
- Split by loop involvement: the 13 loop-involved slots go 6 → 7 successes; the 19 other slots go 10 → 10.
- p2 was not created; 32 practice episodes were left unused. All 192 activation items were used.

## Why p1 and not base
p1 removes a clearly wrong behaviour: closing on air or un-stacking at the stack. The removal is confirmed both offline and closed-loop, and the edit leaves grasp, carry and hold decisions unchanged on held-out replays. The expected held-out effect is a small gain concentrated in loop episodes. The ceiling is the third-cube placement reliability (about 60%) and the policy's failure to recover after a misplacement, which this edit does not address.

Confidence that p1 beats base on the 64 held-out slots: low to moderate (about 0.55-0.6). Confidence that it does not hurt materially: moderate-high (about 0.75).
