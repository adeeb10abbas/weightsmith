import json
T = json.load(open("/work/notes/timeline.json"))
# rows: step, gobs, amin, amax, x, y, z
allg = []
for eid, d in sorted(T.items()):
    rows = d["rows"]
    s = eid.split(".")[-1]
    segs = []
    i = 0
    n = len(rows)
    while i < n:
        # a grasp attempt: first replan whose gripper obs rises above 0.1 after being open
        if rows[i][1] > 0.1 and (i == 0 or rows[i-1][1] <= 0.1):
            j = i
            vals = []
            while j < n and rows[j][1] > 0.1:
                vals.append(rows[j][1]); j += 1
            pre = rows[i-1] if i > 0 else rows[i]
            segs.append(dict(start=rows[i][0], close_z=pre[6], close_xy=(pre[4], pre[5]), g_first=vals[0], g=vals, ymax=max(r[5] for r in rows[i:j]), zmax=max(r[6] for r in rows[i:j])))
            i = j
        else:
            i += 1
    print(s, "S" if d["success"] else "-", "nsteps", rows[-1][0])
    for g in segs:
        print("    t=%d closez=%.3f xy=(%.3f,%.3f) g=%s ymax=%.3f zmax=%.3f" % (g["start"], g["close_z"], g["close_xy"][0], g["close_xy"][1], g["g"], g["ymax"], g["zmax"]))
        allg.append((s, d["success"], g))
