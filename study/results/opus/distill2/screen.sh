#!/bin/bash
# Part B, step 1: screen RoboLab tasks for a language-sensitive failure.
# Paired closed-loop runs of the UNEDITED policy on practice slots 70000..70031 of each task:
# the task's own instruction vs. alternative wordings (diagnostic prompt overrides, not weight edits).
# Alternative wordings are RoboLab's own "specific"/"vague" variants of each task, plus at most one
# hand-written explicit wording per task, fixed here before any run.
# Run on the GPU host:  nohup setsid bash screen.sh > screen.out 2>&1 &
set -u
cd ~/weightsmith
P=/home/ali/vibes/opus-weight-surgery/projects

mkproj() {  # name task instruction
  local d=$P/$1
  if [ ! -f "$d/config.json" ]; then
    mkdir -p "$d"
    cat > "$d/config.json" <<EOF
{
 "task": "$2",
 "instruction": "$3",
 "practice_seed_base": 70000,
 "limits": {"practice_episodes": 1200, "acts_items": 3000, "acts_items_per_call": 96},
 "rendering_type": "performance",
 "practice_slots": 256,
 "envs": 16,
 "prompt_overrides": true
}
EOF
  fi
}

ev() {  # project [--prompt "..."]
  local p=$1; shift
  echo "=== $p $* | $(date)"
  WSP_ROOT=$P/$p python3 weightsmith/owsp.py eval base --n 32 --offset 0 "$@" 2>&1
}

mkproj d2-rightofbowl RubiksCubeRightOfBowlTask "Put the rubiks cube to the right of the bowl"
mkproj d2-yogurt      YogurtInBowlTask          "Put the small red yogurt in the red bowl"
mkproj d2-glasses     PickGlassesTask           "Pick up the eye glasses"
mkproj d2-behindbowl  RubiksCubeBehindBowlTask  "Put the rubiks cube behind the bowl"
mkproj d2-bagel       GrabABagelTask            "Grab a bagel"

ev d2-rightofbowl
ev d2-rightofbowl --prompt "Pick up the rubiks cube and place it on the table to the right side of the bowl"
ev d2-rightofbowl --prompt "Put the rubiks cube on the table to the right of the bowl"
ev d2-yogurt
ev d2-yogurt --prompt "Pick up the small red yogurt container and place it inside the red bowl"
ev d2-yogurt --prompt "Put the yogurt in the bowl"
ev d2-glasses
ev d2-glasses --prompt "Pick up the pair of black eyeglasses that's sitting on the table"
ev d2-behindbowl
ev d2-behindbowl --prompt "Pick up the rubiks cube and place it on the table directly behind the bowl, further from you"
ev d2-bagel
ev d2-bagel --prompt "Reach for one of the bagels on the table and pick it up"
echo "=== done $(date)"
