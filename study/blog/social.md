# Social drafts (for the failure-levels report)

Post only after the held-out and regression numbers are final. Every number here must match `failure-levels.md`.

Attach the before/after video (`site/assets/video/cube_then_banana.mp4`) to tweet 1 and to the LinkedIn post; `figures/opus_hero.png` works as a second image. Name the method as in the paper: *closed-form instruction distillation*.

## X / Twitter thread

**1/**
Can an LLM fix a robot by editing its weights directly, with no training and no gradients?

We tested it on π0.5 across 16 manipulation tasks, with Claude Opus and GPT-6 Astra as the editors.

One clear win. Many instructive failures. 🧵

**2/**
The win: "Put the cube then the banana in the bowl."

The robot kept dropping the cube. Why? Inside the model, "in the bowl" binds to the nearest noun. So while holding the cube, the robot "thinks" it's holding the wrong object, and lets go.

**3/**
The robot already knew the task: reword it explicitly and the same weights succeed 22/32.

So Claude wrote that reading into the weights, in closed form, keeping the original instruction.

3/32 → 25/32. On 128 unseen seeds: 27/128 → 106/128.

And a random edit of exactly the same size (same layers, rank and spectrum, random directions)? 9/64. Same as doing nothing.

And the explicit rewording itself scores 57/64 on those seeds, so the edit gets most of the prompt's benefit without touching the prompt.

Bonus: the same edit transferred. On held-out seeds, "pick up the eyeglasses" went 22/64 → 44/64 and "put the cube to the left of the bowl" went 8/64 → 36/64. (Right of the bowl: still 0. Mechanisms are humbling.)

And a 0.16%-of-parameters version of the edit does the same job: 19/128 → 102/128 held-out.

**4/**
The failures were just as informative:
❌ Grasp slips (contact physics): no gain
❌ A grasp the policy never learned: 10 edits, 0 successes
❌ Grabbing a lookalike object: 9 edits across 2 editors (Claude & Astra, same protocol), no gain
❌ One-weight gripper tweak on 7 tasks: 117/240 → 117/240

**4b/**
Then we took ourselves out of the loop: 7 blind Claude Opus editors, one task per failure kind, identical tools and budget.

Only one edit survived held-out scoring, and it was on the misread instruction: the editor found the same bug on its own and wrote a 1,024-value fix (11/64 → 27/64). Perception, stacking, and a missing grasp: no edit.

**5/**
The pattern: weight editing *re-routes* what a network already knows. It doesn't *create* what it doesn't.

Edit when you can show the network does the right thing under some input. Train when it doesn't.

**6/**
And a warning: identical weights, identical seeds, and still about 1 in 3 episodes flips outcome (GPU physics). A 16-episode "improvement" means nothing. We paired every comparison and held out the final test.

**7/**
Paper: https://ali-adeeb.com/weightsmith/paper.pdf
Site + videos: https://ali-adeeb.com/weightsmith/
Code & benchmark: https://github.com/adeeb10abbas/weightsmith
With @⟦ESMAEIL_X_HANDLE⟧

## LinkedIn (short, casual; attach the 4:5 video)

New project: weightsmith, helping robots through misunderstandings 🤖

We told a robot policy (π0.5, ~3B parameters) to "put the cube then the banana in the bowl." It kept dropping the cube. Turns out it was attaching "in the bowl" to the banana.

So instead of retraining it, we edited its weights directly. No gradient descent, no new data, same instruction. On starts it had never seen, success went from 21% to 83%.

A few takeaways:

→ If the robot does the right thing under some wording, you can often program that into the weights. If no input works, editing won't save you; train instead.

→ A random edit of the exact same size does nothing. Direction matters, not size.

→ We let 7 LLM editors (Claude Opus) try this blind, one task each. One edit survived held-out testing, and it was on the misreading again.

→ Simulation is noisy: the same robot on the same starts flips about a third of outcomes between runs. Pair your comparisons and hold out your test set.

Blog + paper: https://ali-adeeb.com/weightsmith/
Code + benchmark: https://github.com/adeeb10abbas/weightsmith

With Esmaeil Seraj. Builds on our earlier work on instruction sensitivity (Same Goal, Different Words) and on weight-space programming by Takehi et al.
