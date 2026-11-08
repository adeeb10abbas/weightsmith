"""Pre-registered decision rule of the prospective test (README): RF failures of the unedited policy on 30000-30015 in
slot order; build = 1st failure (replan 1), checks = 2nd/3rd/4th failures (replans 1/2/3); held-aside gate set = the
failure episodes after the first four (make_spec --skip-slots <first four> --episodes 6, replans 1-5).
usage (from ~/cross/cosmos): python3 prosp_decisions.py runs/q/S1L-rf-s30000.json  -> shell assignments"""
import json
import sys

run = sys.argv[1]
d = json.load(open(run))
fails = sorted(r["slot"] for r in d["rows"] if r.get("ok") and not r["success"])
if len(fails) < 5:
    raise SystemExit(f"only {len(fails)} RF failures: the rule needs at least 5")
obs = run.replace(".json", ".obs")


def path(s):
    j = s - 30000
    return f"{obs}/g1875o{0 if j < 8 else 8}/obs_r0_e{j % 8}.npz"


print(f"B={path(fails[0])}")
print(f"C1={path(fails[1])}:1")
print(f"C2={path(fails[2])}:2")
print(f"C3={path(fails[3])}:3")
print(f"SKIP={','.join(map(str, fails[:4]))}")
print(f"HELD={','.join(map(str, fails[4:10]))}")
