# General lessons from earlier editors (other tasks, same policy and tools)

These are hints, not facts about your task.

## Practical
- **Bash runs only bare `owsp ...` commands**: no pipes, mkdir, cat or python. Write scripts with the Write tool (into `scripts/`), then run them with `owsp py NAME scripts/x.py`. Put specs in `specs/`.
- **Persist analysis results under `/work/notes`** (JSON). Python state does not survive between `owsp py` calls.
- **Forward kinematics turns joints into end-effector xyz.** This lets you segment episodes into approach, grasp, carry and release phases. Earlier editors used Franka Panda modified-DH plus a 0.257 m flange-to-fingertip offset, with the 7 joints from `obs(eid)["joint"]`.
- **Reading the gripper.**
  - Observation `gripper`: 0 when open, about 1 when closed on nothing. Holding an object gives an intermediate value that depends on the object's width.
  - Action dim 7: > 0.5 commands close.
  - The 15-step chunk is executed open-loop, so a decision made at one replan plays out for 15 sim steps.

## Method
- **Validate edits offline before spending episodes.**
  - Replay the exact recorded inputs of the decision points you care about (target items) and of decision points that must not change (preservation items).
  - Count how many target decisions flip and how many preservation decisions break.
  - Earlier editors rejected several candidates this way and saved their episode budget.
  - **Offline validation is not the goal.** The goal is closed-loop success. Once an edit passes, *run it closed-loop*: n=32 or 64, compared with base on the same slots.
- **Single-neuron fixes did not work.**
  - Constant injections into one VLM MLP unit, and silencing VLM units that merely *correlate* with a failure, did nothing.
  - The correlated units were not causal. Test causality (ablate or steer, then replay) before building an edit on a unit.
- **Scene and visual information reaches the action expert through attention to the image tokens.**
  - Use `"image_tokens": true` and `"action_attention": true` in captures to see what the expert attends to, and how, at the failing decisions.
- **Global biases are not selective.**
  - Example: a gripper-only bias on the action head applied during early denoising steps suppressed the target failure but also broke correct behaviour.
  - Edits that depend on the *state*, through the network's own features, are more promising than constant offsets.
