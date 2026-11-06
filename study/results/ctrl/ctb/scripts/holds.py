import json
import numpy as np
R = json.load(open("/work/notes/replans.json"))
print("== replans holding cube (gobs .25-.45) ==")
for eid in sorted(R):
    r = R[eid]
    for row in r["rows"]:
        if 0.25 <= row["gobs"] <= 0.45:
            print(eid[-6:], "S" if r["success"] else "F", f'k{row["k"]:2d} st{row["step"]:3d} g{row["gobs"]:.2f} a{row["amean"]:.2f} a0{row["a0"]:.2f} aend{row["aend"]:.2f} x{row["x"]:.2f} y{row["y"]:.2f} z{row["z"]:.2f}')
print("== low (z<0.13) open-gripper replans with gobs<0.1 in first 12 ==")
for eid in sorted(R):
    r = R[eid]
    for row in r["rows"][:12]:
        if row["gobs"] < 0.1 and row["z"] < 0.13:
            print(eid[-6:], "S" if r["success"] else "F", f'k{row["k"]:2d} g{row["gobs"]:.2f} a{row["amean"]:.2f} amax{row["amax"]:.2f} x{row["x"]:.2f} y{row["y"]:.2f} z{row["z"]:.2f}')
