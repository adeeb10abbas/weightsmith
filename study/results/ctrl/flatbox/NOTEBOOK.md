# Notebook (flatbox: PickUpGreenObjectTask, "Pick up the green vegetable block")

## Log
- 11:05 Started. Read BRIEF + PRIOR_FINDINGS. Launched `eval base --n 32 --offset 0` (background).
- 11:12 Base: **0/32**. Episodes are 450 steps (30 replans). Success = object_above(frozen_vegetable_block, table, z_margin=0.05)
  i.e. grasp + lift >=5cm. 31/32 got OBJECT_GRABBED (contact-type event; score 0.5), none lifted. ("WRONG_OBJECT_GRABBED" events
  are bookkeeping noise after group1 satisfied.)

## Failure classification (base, dev slots 70000-70031)
Scene is essentially identical across slots: flat green "Flash Frozen Mixed Vegetables" box lying flat, long axis horizontal
in wrist cam at step 0 (green bbox 23x15 px, elongation 1.6 in all 32). Gripper = Robotiq-like parallel jaw, closes along the
image-horizontal axis at yaw 0, i.e. ALONG the box's long side.
FK (Panda MDH + 0.257 tip): open-finger tip rests at z~0.04 on box top; at table ~0.02; closed fingers rest at ~0.053.
1. **Wrong yaw on first approach (32/32).** Policy descends at yaw~0, fingers land ON the box (long side ~ > max opening),
   commanded z 0.02-0.03 but stuck at 0.04 (contact). Closing either never happens or closes on air (gobs=1.0) and then
   the policy presses down while closed for 100+ steps, lifts empty, reopens, retries.
2. **Late rotation (13/32 eventually rotate q6 to ~+1.2..1.9 rad, yaw ~-80deg)** after 150-300 steps of failure. Then the
   fingers straddle the box and go down to the table (z~0.02-0.03) - correct pre-grasp.
3. **Gripper dithering / no commitment at the straddled pre-grasp.** Chunks look like grip=[0.01,0.0,0.5,0.7]: close only in
   last ~5 steps, next chunk starts open again. When it does close on the box (gobs~0.2-0.32) and even lifts
   (s70005 k16: close+lift to z 0.068), the next replan outputs open [0.01,...] and drops it.
   Gripper close on box gives gobs~0.2-0.3 (wide object), closing on air gives 1.0 (then policy stays closed!).
   Commit stats: gobs (0.1,0.45] -> g0>0.5 only ~45-50%; gobs>0.95 -> 80% stay closed.
Nearest misses: s70005 (rotated by step 120, close+lift at k16, reopened at k17), s70025 k17.

## Hypotheses
- H1 (yaw): language "block" -> policy treats it like a cube, keeps default yaw. A different noun ("box") might rotate early.
- H2 (commit): with a wide object, gobs~0.25 looks 'nearly open' to the policy, so it reopens. Closing on air -> 1.0 -> commits.

## Session 2 (continuation)
- a1 result: prompt wording ("box", "box of frozen vegetables") changes ~nothing (dq6, gripper, z all within noise).
  H1 (language-driven yaw) rejected for diagnosis purposes.
- Hold scan (24/32 eps reach gobs in (0.1,0.6)). Rotated straddle + gobs 0.12-0.45 ("holding box"): policy commits
  (g0>0.5) only ~6/17; the rest reopen at chunk start. When it commits (s70002 k26, s70019 k27, s70007 k28) the box
  slips (gobs->1.0 next replan). Frames s70019 380-425: fingers at near-left corner of box, catch the corner, box drops.
  Frames s70005 238-268: fingers straddle the near END of the box, box tilts up at k17 (held!), then policy opens.
  => grasp location is also off-center (near end), but s70005 k17 shows an actual hold that was thrown away by reopening.
- a2-hold capture (48 items): 16 hold, 8 unrotated-intermediate, 12 rotated-open pre-grasp, 8 approach, 4 closed-on-air;
  action layers 10-17, all 10 denoise steps, attention, prompt slots 0-47 (to find gripper state token).

- a2 findings: gripper state is the LAST state number token, ~= round(gobs*255) ("0" open, "61"/"81" holding box,
  "255" closed on air). No single action-expert MLP unit (L10-17) cleanly selects holding. head_input mean-over-tokens
  contrast H vs (A|P) separates classes on average, but per-token readout is noisy (spread +-2.5 vs class gap ~1).
  Removing top-40 PCs of within-item (per-token/per-step, label-free) variation from the contrast makes it clean:
  summed-over-steps readout H +6.4+-2.1, U +1.9, C +2.7, P 0+-1.35, A -5.2+-1.6 (token 0: H min +3.3, P max +3.7).

## p1 (pre-registration)
- Mechanism: with the wide box, the "holding" gripper reading (gobs 0.15-0.35) does not make the policy commit; it
  reopens at the next replan (~10/16 hold decisions) and throws away grasps (s70005 k17 had the box tilted up).
- Edit: action.action_out_proj.kernel[:,7] += C*d, bias[7] -= C*r. d = unit hold-vs-(approach|open pre-grasp) contrast
  of head_input with 40 nuisance PCs removed; r = mean projection of the P class (zero mean shift on open pre-grasp);
  C = -1.25 (dv7<0 => gripper output up => close). Rank-1, 1021 values (gripper column + gripper bias), |dK|=1.25
  (col7 norm 0.85, so large on that column). Joint columns untouched.
- Offline no-feedback prediction: gripper output shift H +0.80 (tok0 0.41..0.96), U +0.24, C +0.34, P 0.00, A -0.65.
- Replay check a3-p1 (42 items: 16 H in-sample, 12 new open pre-grasp, 10 new approach, 4 U).
- Closed-loop prediction (if replay passes): 2-4/32 successes (base 0), fixed 2-4, broke 0 (nothing to break).
  Risk: committing to bad (corner) grasps -> slip -> closed-on-air -> time wasted; approach unaffected.

- a3-p1 replay result (real denoising dynamics, same seeds): H (16, in-sample) g0>0.5 6 -> 11, close-frac 0.39 -> 0.70,
  full-chunk commits 4 -> 10. Still opening: s70005 k17 (g0 .16), s70003 k9, s70005 k19, s70021 k27.
  New open pre-grasp (12, out-of-sample): close-frac 0.29 -> 0.29 (2 items lose a late close, 2 gain). New approach (9 true
  approach): all stay open (close-frac slightly lower). U 0.75 -> 0.95. Joint outputs move <= 0.16 rad max, mostly <0.05.
  => passes; launched `eval p1 --n 32 --offset 0`. Prediction unchanged: 2-4/32, fixed 2-4, broke 0.

## p1 result: 0/32 (base 0/32), fixed 0, broke 0. Prediction (2-4) did NOT hold.
- The edit worked at the decision level: holding replans (gobs 0.1-0.45) committed 29/39 (base 20/42); after full-chunk
  commits, base 13 cases -> p1 23 cases. p1 turned rotated-straddle holds into close+lift (s70001 k12, s70004 k23,
  s70010 k24, s70014 k19, s70016 k27, s70017 k16 lift to z 0.10-0.19), but in EVERY lift the box slipped out
  (gobs -> 1.0 at the next replan). Holding with z>0.06 at a replan: base 2, p1 1.
- Conclusion: the reopen decision is not the binding constraint. The grasp itself is bad: fingers straddle the near
  end/corner of the box and only the top ~1 cm of it (fingertip z 0.03-0.045 at close vs table ~0.02), so it slips on lift.
  Fixing that needs position/depth changes (joint outputs), which I could not diagnose within the time budget.
- Decision: no p2 (no mechanism with offline evidence left to test in the time). Submit base.

## Acts budget
- a1-prompts (60 items): 20 recorded decisions (8 approach, 8 rotated pre-grasp, 4 holding-ish) x 3 prompts
  (base / "Pick up the green vegetable box" / "Pick up the box of frozen vegetables"). Diagnosis only.
- a2-hold (48 items), a3-p1 (42 items). Total used 150/192. Episodes used 64/96.
