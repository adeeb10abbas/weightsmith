## Metric 1: failure map (unedited, task's own wording, practice slots from 30000)

| Task | RoboLab task | Slots | Success | Rate | Wilson 95% |
|---|---|---|---|---|---|
| T1 | RubiksCubeThenBananaTask | 30000-30031 | 9/32 | 28% | 16-45% |
| T2 | PickGlassesTask | 30000-30031 | 0/32 | 0% | 0-11% |
| T3 | RubiksCubeLeftOfBowlTask | 30000-30031 | 1/32 | 3% | 1-16% |
| T4 | StackYellowOnRedTask | 30000-30015 | 0/16 | 0% | 0-19% |
| T5 | MustardInLeftBinTask | 30000-30031 | 0/32 | 0% | 0-11% |
| T6 | Stack3RubiksCubeTask | 30000-30015 | 0/16 | 0% | 0-19% |
| T7 | PickUpGreenObjectTask | 30000-30031 | 0/32 | 0% | 0-11% |
| T8 | MustardRightOfRaisinRFTask | 30000-30031 | 0/32 | 0% | 0-11% |
| R1 | RubiksCubeTask | 30000-30031 | 28/32 | 88% | 72-95% |

## Metric 2: wording sensitivity (unedited, paired on slots 30000-30031)

| Task | Original | Alternative wording | Original k/n | Alternative k/n | alt only / orig only | exact p |
|---|---|---|---|---|---|---|
| T1 | Put the cube then the banana in the bowl | Put the cube in the bowl then put the banana in the bowl | 9/32 | 16/32 | 11 / 4 | 0.12 |
| T3 | Put the rubiks cube to the left of the bowl | Pick up the rubiks cube and place it on the table to the left of the bowl | 1/32 | 0/32 | 0 / 1 | 1 |
| T8 | Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle. | Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box. | 0/32 | 0/32 | 0 / 0 | 1 |

## Adapter validation: NVIDIA's unmodified path (run_gr00t_server.py + RoboLab policies/gr00t/run.py)

| Task | Official path | This adapter | Note |
|---|---|---|---|
| BananaOnPlateTask | 16/16 | 16/16 | same 16 slots 0-15, default preset |
| PickUpGreenObjectTask | 0/16 | 0/32 | official path uses RoboLab's own starts (16 envs, one run), not the paired slots |

## Edit `bb` (Cosmos-Reason2 LLM layers 0-15, mlp.down_proj (VLM backbone)) on T1, practice slots 30000-30031

| Arm | Success | fixed / broke vs unedited | exact p | 95% CI of difference |
|---|---|---|---|---|
| unedited | 9/32 | | | |
| edited | 0/32 | 0 / 9 | 0.0039 | -44 to -12 pts |
| teacher (explicit wording, unedited) | 16/32 | | | |

## Edit `dit` (GR00T action head: 32 AlternateVLDiT blocks, ff.net.2 (diffusion transformer)) on T1, practice slots 30000-30031

| Arm | Success | fixed / broke vs unedited | exact p | 95% CI of difference |
|---|---|---|---|---|
| unedited | 9/32 | | | |
| edited | 0/32 | 0 / 9 | 0.0039 | -44 to -12 pts |
| teacher (explicit wording, unedited) | 16/32 | | | |
