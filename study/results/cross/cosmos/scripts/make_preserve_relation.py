import re, glob, json, os
# run on ali-ms: python3 make_preserve_relation.py  (writes /tmp/cosmosx_preserve_relation.json)
os.chdir(os.path.expanduser("~/vibes/weight-space-policy-lab/robolab_checkout"))
rows = []
for f in sorted(glob.glob("robolab/tasks/**/*.py", recursive=True)):
    s = open(f).read()
    for m in re.finditer(r"class (\w+Task)\b.*?instruction\s*=\s*\{(.*?)\}", s, re.S):
        for k, v in re.findall(r"\"(\w+)\"\s*:\s*\"([^\"]+)\"", m.group(2)):
            rows.append((m.group(1), k, v.strip()))
LR = re.compile(r"\b(left|right)\b", re.I)
EXCL_TASKS = {"RubiksCubeLeftOfBowlTask", "MustardInLeftBinTask", "MustardRightOfRaisinRFTask", "RubiksCubeThenBananaTask", "PickGlassesTask", "StackYellowOnRedTask", "Stack3RubiksCubeTask", "PickUpGreenObjectTask", "RubiksCubeTask"}
def objs(t):
    t = t.lower()
    o = set()
    if "mustard" in t: o.add("mustard")
    if "raisin" in t: o.add("raisin")
    if re.search(r"\bbins?\b", t): o.add("bin")
    if re.search(r"rubik|\bcubes?\b", t): o.add("cube")
    if re.search(r"\bbowls?\b", t): o.add("bowl")
    return o
SAME_PAIRS = [{"mustard", "bin"}, {"cube", "bowl"}, {"mustard", "raisin"}]  # T5, T3, T8
keep, drop, seen = [], [], set()
for c, k, i in rows:
    if not LR.search(i):
        continue
    why = None
    if c in EXCL_TASKS: why = "a shared-set task (T1-T8, R1)"
    elif any(p <= objs(i) for p in SAME_PAIRS): why = "same object pair as T3, T5 or T8"
    elif "mustard" in i.lower() or "raisin" in i.lower(): why = "mentions T8/T5 objects (mustard/raisin)"
    if why:
        drop.append([c, k, i, why]); continue
    if i.lower() in seen:
        continue
    seen.add(i.lower()); keep.append([c, i, k])
out = {"kept": [[c, i] for c, i, _ in keep], "kept_with_variant": keep, "dropped": drop,
       "rule": "RELATION-AWARE (t8und-r, coordinator spec 07:55 ET, fixed before the build): every instruction variant (default / vague / specific / ...) of a RoboLab v0.2.1 task that contains the word 'left' or 'right', minus every shared-set task (T1-T8, R1; all of their instruction variants), minus any instruction with T3's (rubiks cube + bowl), T5's (mustard + bin) or T8's (mustard + raisin box) object pair, minus anything mentioning mustard or raisin; duplicates removed. Built by scripts/make_preserve_relation.py"}
json.dump(out, open("/tmp/cosmosx_preserve_relation.json", "w"), indent=1)
print(len(keep), "kept")
for c, i, k in keep: print("KEEP", c, k, "|", i)
for d in drop: print("DROP", d[0], d[1], "|", d[2], "|", d[3])
