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

## LinkedIn (short, casual; attach the video or carousel.pdf)

Been working on a fun one with Esmaeil Seraj.

π0.5 (Physical Intelligence's open robot model) kept dropping the cube when told to "put the cube then the banana in the bowl." It was reading "in the bowl" as being about the banana.

So we fixed it inside the model, no training: we took what the network does with a clearer instruction and wrote it straight into the weights. Same original instruction, and success on setups it had never seen went from 21% to 83%.

The parts I didn't expect:
- The same edit fixed two tasks we never built it for: picking up eyeglasses (22 → 44 of 64) and putting the cube left of the bowl (8 → 36).
- A random edit of the same size did nothing.
- It never taught the robot a new skill. Everything that worked was about how the model read the instruction, not how it moves.
- We let 7 Claude agents try this blind. One found the same bug on its own and fixed it. Two others, on different tasks, found the same two neurons for the word "left".

Blog + paper: https://ali-adeeb.com/weightsmith/
Code: https://github.com/adeeb10abbas/weightsmith
