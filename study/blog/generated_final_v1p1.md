| Unique comparison | Base / candidate | Nominal fixed / broken | Difference (95% interval) | Original role labels |
|---|---:|---:|---:|---|
| identity | 26/64 / 23/64 | 17 / 20 | -0.047 [-0.234, 0.141] | adaptive-r1-round1, adaptive-r1-round2, adaptive-r1-round3, adaptive-r2-round1, adaptive-r2-round2, adaptive-r2-round3, nonadaptive-r1-final, nonadaptive-r2-final |
| gripper-control | 23/64 / 36/64 | 21 / 8 | 0.203 [0.047, 0.359] | gripper-control |
| base-repeat | 26/64 / 35/64 | 21 / 12 | 0.141 [-0.031, 0.312] | Independent operator control |

Role labels sharing one comparison reuse its same estimate; they are not independent round outcomes.

| Original role | Unique comparison | Shared estimate |
|---|---|---|
| adaptive-r1-round1 | identity | True |
| adaptive-r1-round2 | identity | True |
| adaptive-r1-round3 | identity | True |
| adaptive-r2-round1 | identity | True |
| adaptive-r2-round2 | identity | True |
| adaptive-r2-round3 | identity | True |
| nonadaptive-r1-final | identity | True |
| nonadaptive-r2-final | identity | True |
| gripper-control | gripper-control | False |
