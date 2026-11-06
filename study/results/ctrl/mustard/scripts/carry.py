import json
T = json.load(open("/work/notes/timeline.json"))
o = obs("base.base.s70000")
print("joint[3]", o["joint"][3], "\nact[2,0]", o["actions"][2, 0], "\nact[2,14]", o["actions"][2, 14])
for eid, d in sorted(T.items()):
    rows = d["rows"]
    s = eid.split(".")[-1]
    # first grasp segment +-1
    i0 = next((i for i, r in enumerate(rows) if r[1] > 0.1), None)
    if i0 is None:
        print(s, "no grasp"); continue
    j = i0
    while j < len(rows) and rows[j][1] > 0.1 and rows[j][1] < 0.95:
        j += 1
    seg = rows[max(0, i0-2): min(len(rows), j+1)]
    print(s, "S" if d["success"] else "-", " ".join("[%d g%.2f a%.2f y%.2f z%.2f x%.2f]" % (r[0], r[1], r[3], r[5], r[6], r[4]) for r in seg))
