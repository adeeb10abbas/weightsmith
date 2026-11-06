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

**5/**
The pattern: weight editing *re-routes* what a network already knows. It doesn't *create* what it doesn't.

Edit when you can show the network does the right thing under some input. Train when it doesn't.

**6/**
And a warning: identical weights, identical seeds, and still about 1 in 3 episodes flips outcome (GPU physics). A 16-episode "improvement" means nothing. We paired every comparison and held out the final test.

**7/**
Paper: https://adeeb10abbas.github.io/weightsmith/paper.pdf
Site + videos: https://adeeb10abbas.github.io/weightsmith/
Code & benchmark: https://github.com/adeeb10abbas/weightsmith
With @⟦ESMAEIL_X_HANDLE⟧

## LinkedIn

**Can AI fix a robot by editing its brain directly? Sometimes, and the "sometimes" is the interesting part.**

Over the past week, Esmaeil Seraj and I gave frontier models (Claude Opus and GPT-6 Astra) direct read-write access to the weights of π0.5, a 3.35B-parameter robot policy. There was no training and no new data, only diagnosis and closed-form edits. We tested 16 simulated manipulation tasks.

**What worked.** On a two-step task, the robot kept dropping the first object. Claude traced it to how the model's language component binds "in the bowl" to the wrong noun. The robot already had the skill: with the instruction spelled out, the same weights succeed. So Claude distilled that reading into the weights. Success rose from 3/32 to 25/32, and from 27/128 to 106/128 on 128 held-out trials, with the original instruction unchanged. A random edit of exactly the same size did nothing (9/64, the same as the unedited policy). The same edit also transferred to two other tasks on held-out trials (eyeglasses 22/64 → 44/64; cube left of bowl 8/64 → 36/64), with no detectable harm across 15 tasks. A version touching 0.16% of the parameters (in low-rank form) works just as well.

**What didn't.** Contact physics (grasps that slip), a grasp the policy never learned, and confusion between lookalike objects all resisted every edit. That includes a one-parameter gripper change that was exactly neutral across seven tasks (117/240 vs 117/240).

**The takeaway.** Weight editing re-routes what a network already knows; it doesn't create new skills. That matches the strongest published results, which edited policies already trained on their tasks.

**A note on rigor.** In simulation, about one in three episodes flips outcome between identical runs. Small, unpaired evaluations will happily "confirm" fixes that aren't there.

Paper: https://adeeb10abbas.github.io/weightsmith/paper.pdf
Project page with videos: https://adeeb10abbas.github.io/weightsmith/
Code and benchmark: https://github.com/adeeb10abbas/weightsmith
