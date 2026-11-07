# Social drafts (for the failure-levels report)

Post only after the held-out and regression numbers are final. Every number here must match `failure-levels.md`.

Attach the before/after video (`site/assets/video/cube_then_banana.mp4`) to tweet 1 and to the LinkedIn post; `figures/opus_hero.png` works as a second image. Name the method as in the paper: *closed-form instruction distillation*.

## X / Twitter thread (attach the 16:9 video to 1/, carousel slides to 2/–5/)

1/ [attach: weightsmith_linkedin_16x9.mp4]
We fixed a robot's misunderstanding by editing its weights. No training, no new data, same instruction.

π0.5 went from 21% to 83% success on setups it had never seen. 🧵

2/ [attach: carousel slides/slide_2.png]
We told π0.5 to "put the cube then the banana in the bowl," and it kept dropping the cube on the way. It was reading "in the bowl" as being about the banana.

Spell it out and it mostly works. Same goal, different words: 3/32 vs 22/32.

3/ [attach: carousel slides/slide_3.png]
So we fixed it inside the model: we recorded what the network does with the clear instruction and wrote that straight into its weights, in closed form.

Original instruction, new setups: 21% → 83%. A random edit of the same size did nothing.

4/ [attach: carousel slides/slide_4.png]
The part I didn't expect: the same edit fixed two tasks we never built it for.

Picking up eyeglasses: 22 → 44 of 64.
Putting the cube left of the bowl: 8 → 36.

Nothing else got measurably worse.

5/ [attach: carousel slides/slide_5.png]
What it can't do: teach the robot a new skill. When no wording worked (a lookalike block, a box it can't grip), no edit worked either.

Edit when some wording already gets the robot there. Train when nothing does.

6/
Blog + paper: https://ali-adeeb.com/weightsmith/
Code: https://github.com/adeeb10abbas/weightsmith

Point your agents at the repo and see if their edit beats the robot on starts it's never seen.

With [Esmaeil's handle]. Builds on Weight-Space Programming by Takehi et al.

## LinkedIn (short, casual; attach weightsmith_linkedin_4x5.mp4 or carousel.pdf)

Been working on a fun one with Esmaeil Seraj.

π0.5 (Physical Intelligence's open robot model) kept dropping the cube when told to "put the cube then the banana in the bowl." It was reading "in the bowl" as being about the banana.

So we fixed it inside the model, no training: we took what the network does with a clearer instruction and wrote it straight into the weights. Same original instruction, and success on setups it had never seen went from 21% to 83%.

The parts I didn't expect:
- The same edit fixed two tasks we never built it for: picking up eyeglasses (22 → 44 of 64) and putting the cube left of the bowl (8 → 36).
- A random edit of the same size did nothing.
- It never taught the robot a new skill. Everything that worked was about how the model read the instruction, not how it moves.

Blog + paper: https://ali-adeeb.com/weightsmith/
Code: https://github.com/adeeb10abbas/weightsmith

To give it a go, point your agents at the repo and see if their edit beats the robot on starts it's never seen.
