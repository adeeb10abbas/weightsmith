## Metric 1: failure map (unedited, task's own wording, slots 30000-30031)

| Task | RoboLab task | Success | Rate | Wilson 95% |
|---|---|---|---|---|
| T1 | RubiksCubeThenBananaTask | 9/32 | 28% | 16-45% |
| T3 | RubiksCubeLeftOfBowlTask | 1/32 | 3% | 1-16% |
| T8 | MustardRightOfRaisinRFTask | 0/32 | 0% | 0-11% |

## Metric 2: wording sensitivity (unedited, paired on slots 30000-30031)

| Task | Original | Alternative wording | Original k/n | Alternative k/n | alt only / orig only | exact p |
|---|---|---|---|---|---|---|
| T1 | Put the cube then the banana in the bowl | Put the cube in the bowl then put the banana in the bowl | 9/32 | 16/32 | 11 / 4 | 0.12 |
| T3 | Put the rubiks cube to the left of the bowl | Pick up the rubiks cube and place it on the table to the left of the bowl | 1/32 | 0/32 | 0 / 1 | 1 |
| T8 | Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle. | Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box. | 0/32 | 0/32 | 0 / 0 | 1 |

## Edit `bb` (Cosmos-Reason2 LLM layers 0-15, mlp.down_proj (VLM backbone)) on T1, practice slots 30000-30031

| Arm | Success | fixed / broke vs unedited | exact p | 95% CI of difference |
|---|---|---|---|---|
| unedited | 9/32 | | | |
| edited | 0/32 | 0 / 9 | 0.0039 | -44 to -12 pts |
| teacher (explicit wording, unedited) | 16/32 | | | |

### Side-effect map of `bb` (edited vs unedited, each task's own wording, slots 30000-30031)

| Task | Unedited | Edited | fixed / broke | exact p |
|---|---|---|---|---|
| T1 | 9/32 | 0/32 | 0 / 9 | 0.0039 |
