# Offline check on the paper's 49 recorded decisions (run with `owsp py base`). For every capture in ARMS:
# gripper decisions per group (chunk counted as open when fewer than half of its 15 steps command > 0.5, as in
# study/results/opus/ctb/chk.py), flips against the unedited replay, and chunk-end end-effector shift (Panda FK).
# Groups (capture order): 0-20 premature releases (PREM; 0,1,2,5,6,11 are the 6 distilled decisions), 21-29 correct
# carries, 30-36 phase-2 approach, 37-41 phase-2 banana grasp, 42-48 phase-2 carry.
ARMS = globals().get('ARMS') or ['chk-base', 'chk-teacher', 'chk-v1-full', 'chk-v1-img816', 'chk-p1-blind']
import json


def fk(q):
    dh = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
          (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
    T = np.eye(4)
    for i, (a, d, al) in enumerate(dh):
        th = q[i]; ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca], [0, 0, 0, 1]])
    F = np.eye(4); F[2, 3] = 0.257
    return (T @ F)[:3, 3]


G = {'PREM': list(range(0, 21)), 'CARRY': list(range(21, 30)), 'P2app': list(range(30, 37)),
     'P2grasp': list(range(37, 42)), 'P2carry': list(range(42, 49))}
TRAIN = [0, 1, 2, 5, 6, 11]
OOS = [i for i in G['PREM'] if i not in TRAIN]
ib, B = acts('chk-base')
keys = [(i['episode'], i['k']) for i in ib['items']]
_, TT = acts('chk-teacher')
openb = (B['actions'][:, :, 7] > 0.5).mean(1) < 0.5
eb = np.array([fk(a[-1, :7]) for a in B['actions']]); et = np.array([fk(a[-1, :7]) for a in TT['actions']])
out = {}
for aid in ARMS:
    ia, A = acts(aid)
    assert [(i['episode'], i['k']) for i in ia['items']] == keys, aid
    assert all(i['prompt'] == ib['items'][0]['prompt'] for i in ia['items']) or aid == 'chk-teacher', aid
    op = (A['actions'][:, :, 7] > 0.5).mean(1) < 0.5
    ea = np.array([fk(a[-1, :7]) for a in A['actions']])
    r = {'cand': ia['cand'], 'prompt': ia['items'][0]['prompt']}
    for g, ix in G.items():
        r[g + '_open'] = int(op[ix].sum())
        r[g + '_flips'] = int((op[ix] != openb[ix]).sum())
        r[g + '_ee_shift_cm'] = round(float(np.linalg.norm(ea[ix] - eb[ix], axis=1).mean() * 100), 2)
    r['PREM_open_distilled6'] = int(op[TRAIN].sum())
    r['PREM_open_other15'] = int(op[OOS].sum())
    r['preservation_flips'] = int(sum(r[g + '_flips'] for g in ['CARRY', 'P2app', 'P2grasp', 'P2carry']))
    r['PREM_ee_to_teacher_cm'] = round(float(np.linalg.norm(ea[G['PREM']] - et[G['PREM']], axis=1).mean() * 100), 2)
    r['PREM_gripper_chunk_mean'] = round(float(A['actions'][G['PREM'], :, 7].mean()), 3)
    out[aid] = r
    print('%-16s %-10s PREM release %2d/21 (distilled 6: %d, other 15: %2d) | carries released %d/9 | P2app open %d/7 '
          'P2grasp open %d/5 P2carry open %d/7 | preservation flips %d/28 | EE shift PREM %.1f CARRY %.1f P2 %.1f/%.1f/%.1f cm | PREM EE->teacher %.1f cm' % (
              aid, ia['cand'], r['PREM_open'], r['PREM_open_distilled6'], r['PREM_open_other15'], r['CARRY_open'], r['P2app_open'],
              r['P2grasp_open'], r['P2carry_open'], r['preservation_flips'], r['PREM_ee_shift_cm'], r['CARRY_ee_shift_cm'],
              r['P2app_ee_shift_cm'], r['P2grasp_ee_shift_cm'], r['P2carry_ee_shift_cm'], r['PREM_ee_to_teacher_cm']))
print('CHECK_JSON ' + json.dumps(out))
