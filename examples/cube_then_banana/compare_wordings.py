# ruff: noqa: F821  (obs, acts: helpers of the owsp py sandbox prelude)
# What the policy decides at the same recorded decision under each wording of a capture.
#   bin/owsp py base examples/cube_then_banana/compare_wordings.py
# Reads the capture made from two_wordings.json (set AID if you named it differently).
AID = "two-wordings"

idx, a = acts(AID)
for i, it in enumerate(idx["items"]):
    command = float(a["actions"][i, :, 7].mean())  # mean gripper command over the 15-step chunk
    live = float(a["recorded_actions"][i, :, 7].mean())  # what the policy commanded in the recorded episode
    print(f"{it['episode']} k={it['k']:2d}  gripper {command:.2f} {'CLOSE' if command > 0.5 else 'open '}"
          f"  (live run: {live:.2f})  {it['prompt']!r}")
