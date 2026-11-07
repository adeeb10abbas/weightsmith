# Concepts: editing a robot policy's weights

A two-page primer for robotics and ML engineers new to weight-space programming. The evidence behind every
number is in the paper, [*Distilling Instructions into Weights*](https://ali-adeeb.com/weightsmith/paper.pdf)
([source](../paper/main.tex)). For hands-on steps, see the [cookbook](COOKBOOK.md).

## What weight-space programming is

A trained network is a program written in numbers. Weight-space programming
([Takehi et al.](https://rikiyat.github.io/blog/weight-space-programming/)) treats those numbers as code you can
read and patch: look at the weights and at the activations where the network goes wrong, find the computation that
causes the failure, and change the weights directly. The change is computed in closed form (means, contrasts,
projections, least-norm solves): no gradients, no training data, no search over random variants. In weightsmith the
editor, an LLM agent or a person, works through one tool (`owsp`) on OpenPI's π0.5 policy in NVIDIA RoboLab
simulation, and writes down a prediction before each edit is tested.

## How π0.5 is put together

π0.5 is two transformers that work as one network.

- **The vision-language model (VLM)** is PaliGemma: a SigLIP image encoder plus an 18-layer Gemma language model.
  It reads one sequence: 256 image tokens from each of two cameras (over the shoulder and on the wrist), then the
  text `Task: <instruction>, State: <robot state as numbers>; Action:`. Attention in this prefix runs both ways,
  so image tokens can attend to the instruction.
- **The action expert** is a smaller 18-layer Gemma. Its 15 action tokens attend to the VLM's keys and values at
  every layer. Starting from noise, it takes 10 denoising steps (flow matching) to produce a chunk of 15 actions,
  each 7 joint targets plus a gripper command (above 0.5 means close). The policy is queried every 15 simulator
  steps (a *replan*), and each chunk runs open loop.

| What | Where it lives |
|---|---|
| The instruction | The tokens after `Task:` (slots 3–11 for "Put the cube then the banana in the bowl"), processed by the VLM's layers (`language.blocks.LL.*`) |
| The images | 256 tokens per camera; captures average them into a 4×4 grid of cells per camera |
| The robot state | Number tokens after `State:`; the last one is the gripper reading |
| The gripper decision | Column 7 of the action head `action.action_out_proj.kernel`, read from the expert's last hidden state |

In the cube-then-banana task, rewording the instruction changes the image tokens by only 2–4% of their norm, yet
copying that small change fixes the failure: the action expert puts 13–44% of its attention on image tokens, and the
instruction's effect lands almost entirely on its gripper output.

## Failure levels

A 16-episode survey of the unedited policy on 16 tasks sorted its failures into levels. They predict what editing
can do.

| Level | What goes wrong | Example | Editing so far |
|---|---|---|---|
| Language binding | The wrong object counts as the current target | Cube then banana | Fixed: 27/128 → 106/128 on held-out seeds |
| Perception | The arm goes for a similar-looking object | Stack yellow on red | No gain (edits 21/64 against 28/64) |
| Missing competence | No working motion is ever produced | Pick up the flat box | 10 edits changed the motion, 0 successes |
| Contact physics | The decisions look right, the grasp slips | Mustard into the bin | No decision to correct |
| Gripper commitment | Open/close decisions sit near the threshold | Seven tasks | One weight: 117/240 against 117/240 |

## The known-good input rule

**Edit when some input makes the network do the right thing; train when nothing does.**

To check, replay recorded failing decisions with the images, robot state and noise fixed, changing only the
instruction (a *prompt swap*). Told to "put the cube then the banana in the bowl", π0.5 grasps the cube and lets go
short of the bowl. Replayed under "put the cube in the bowl then put the banana in the bowl", 19 of 21 such
decisions keep holding, and in closed loop the same weights succeed in 22 of 32 episodes against 3 of 32. That
rewording is a *known-good input*: the network can already compute the right thing, and an edit only has to route
the original instruction there. Where no input works, there is nothing to route: rewording did not move the arm
in the yellow-on-red task, and no motion of the policy ever grasps the flat box. Those need training data.

## What has worked, and what has not

- **A misreading was fixed.** *Closed-form instruction distillation* copies the explicit wording's per-layer
  effect into 17 VLM MLP weight matrices, one least-norm solve per layer. On 128 held-out seeds, success rose from
  27 to 106 (81 fixed, 2 broken, exact p = 7×10⁻²²); an update of the same size in random directions did nothing.
- **It transferred.** Unchanged, the edit lifted picking up eyeglasses (22/64 → 44/64) and putting the cube left
  of the bowl (8/64 → 36/64) on held-out seeds, with no detectable harm on the other 13 tasks. It did not lift
  *right* of the bowl (0/32), so it is not a general switch for spatial words.
- **A blind editor repeated the diagnosis.** In a controlled study of seven Claude Opus editors, one per failure
  level, the cube-then-banana editor independently traced the release to the wording and wrote a 1,024-value
  gated gripper edit: 11/64 → 27/64 held-out (p = 0.0037; 0.026 after Holm correction over seven tasks). None of
  the other six passed.
- **No edit taught a missing motor skill.** On the flat box, the best of ten edits brought the wrist to a
  graspable angle in 16/16 episodes instead of 9/16, with no success; a blind editor's edit made the gripper hold
  and lift the box six times, and it slipped every time. Edits aimed at perception, contact physics and the gripper
  threshold also gave no gain.

## How to tell whether an edit worked

- **Paired starts.** A simulator *slot* fixes one episode's scene and noise seeds. Run the unedited and the edited
  policy on the same slots and count the four outcomes per pair: both succeed, both fail, *fixed* (only the edit
  succeeds) and *broke* (only the unedited policy succeeds). The exact McNemar test compares fixed with broke.
- **Practice, then held-out.** Develop on practice slots (20000–20255), freeze the edit, and score it once on
  held-out slots (80000 and above) that were never used during editing, with at least 64 pairs and the run order
  alternating by half. The benchmark accepts an edit with more successes than the unedited policy at exact McNemar
  p < 0.05.
- **Expect noise.** Paired seeds do not make episodes identical: two runs of identical weights flip about a third
  of paired outcomes (the unedited policy scored 14/16 and then 8/16 on the same butter seeds). With that much
  noise, a paired comparison detects changes of about 27 points with 32 pairs, 19 with 64, 14 with 128 and 10
  with 256 (80% power, α = 0.05). A 16-episode screen is a filter, not evidence.

`weightsmith stats` computes all of this from the per-slot outcomes; the [cookbook](COOKBOOK.md#6-paired-evaluation-and-reading-the-statistics)
shows how to read it.
