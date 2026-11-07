# ruff: noqa: F821  (obs, acts: helpers of the owsp py sandbox prelude)
# Gripper timeline of one practice episode: what the policy commanded at each replan.
# Runs in the weight sandbox (it reads recorded episodes, not weights):
#   bin/owsp py base examples/cube_then_banana/gripper_timeline.py
# Set EID to an episode id from `bin/owsp episodes -v` first. The policy replans every 15 sim steps and
# outputs a 15-step chunk of joint targets plus a gripper command (0 = open, 1 = closed).
EID = "base.base.s20001"

o = obs(EID)
command = o["actions"][:, :, 7].mean(1)  # mean gripper command over each chunk
for k in range(len(o["step"])):
    grip = float(o["gripper"][k][0])  # measured opening: 0 = open, ~0.3-0.8 = closed on an object
    closed = command[k] > 0.5
    holding = 0.1 <= grip <= 0.9
    note = "  <- opens while holding something" if holding and not closed and k and command[k - 1] > 0.5 else ""
    print(f"k={k:2d} step={int(o['step'][k]):3d}  measured {grip:.2f}  commanded {command[k]:.2f} "
          f"{'CLOSE' if closed else 'open '}{note}")
