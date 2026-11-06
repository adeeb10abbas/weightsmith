import json
rows = json.load(open('/work/notes/carry_rows.json'))
D = json.load(open('/work/notes/base_traj.json'))
by = {}
for r in rows:
    by.setdefault(r['eid'], []).append(r)
for eid, d in D.items():
    st = d['step']; xyz = np.array(d['xyz']); g = np.array(d['grip'])
    hold = (g > 0.2) & (g < 0.5)
    # first hold, last hold, max y while holding, x at max y
    if hold.any():
        ih = np.where(hold)[0]
        fy = xyz[ih]
        j = ih[np.argmax(fy[:, 1])]
        print(f"{eid[-6:]} {'S' if d['success'] else 'F'} nsteps={st[-1]:3d} grasp@{st[ih[0]]:3d} p=({xyz[ih[0]][0]:+.2f},{xyz[ih[0]][1]:+.2f}) "
              f"maxy@{st[j]:3d}=({xyz[j][0]:+.2f},{xyz[j][1]:+.2f},{xyz[j][2]:.2f}) lasthold@{st[ih[-1]]:3d}=({xyz[ih[-1]][0]:+.2f},{xyz[ih[-1]][1]:+.2f},{xyz[ih[-1]][2]:.2f}) nhold={len(ih)}")
    else:
        print(eid[-6:], 'S' if d['success'] else 'F', 'never hold', 'maxg', g.max().round(2))
