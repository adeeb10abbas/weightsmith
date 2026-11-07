# How much of the teacher's action-expert change each arm reproduces at the 6 distilled decisions (run with
# `owsp py base`). For action layer L: g = mean over decisions of (teacher - original) residual after block L,
# m = the same for the edited model under the original wording; reported: m.g/g.g (share of the change reproduced
# along its direction) and |m - g|/|g|, over all 10 denoise steps x 15 action tokens. Also the gripper velocity change
# per denoise step (head unchanged in these arms, so v = h @ W_out of the base head).
ARMS = globals().get('ARMS') or {'v1-full': 'va-orig', 'a1': 'a1-trace', 'va': 'va-trace'}
import json
f64 = lambda a: a.astype(np.float64)
io, O = acts('a-orig'); ix, X = acts('a-expl')
RO, RX = f64(O['action.resid_out']), f64(X['action.resid_out'])
Wout = f64(base('action.action_out_proj.kernel'))
g = (RX - RO).mean(0)                                                    # [10, 18, 15, 1024]
v7 = lambda H: (f64(H['action.head_input']) @ Wout[:, 7]).mean((0, 2))   # [10]
out = {'teacher_dv7_per_step': [round(float(x), 3) for x in v7(X) - v7(O)], 'arms': {}}
for arm, cap in ARMS.items():
    try:
        ia, A = acts(cap)
    except Exception as e:  # capture not available yet
        print(arm, 'missing', cap, e); continue
    assert [(i['episode'], i['k']) for i in ia['items']] == [(i['episode'], i['k']) for i in io['items']]
    m = (f64(A['action.resid_out']) - RO).mean(0)
    rows = []
    for L in range(18):
        gl, ml = g[:, L].ravel(), m[:, L].ravel()
        rows.append({'layer': L, 'share': round(float(ml @ gl / (gl @ gl)), 3), 'rel_err': round(float(np.linalg.norm(ml - gl) / np.linalg.norm(gl)), 3)})
    out['arms'][arm] = {'capture': cap, 'cand': ia['cand'], 'layers': rows,
                        'dv7_per_step': [round(float(x), 3) for x in v7(A) - v7(O)],
                        'final_gripper_chunk_mean': [round(float(x), 3) for x in A['actions'][:, :, 7].mean(1)]}
    print('%-8s share of teacher change by layer: %s' % (arm, ' '.join('%d:%.2f' % (r['layer'], r['share']) for r in rows)))
    print('         rel err by layer: %s' % ' '.join('%d:%.2f' % (r['layer'], r['rel_err']) for r in rows))
    print('         gripper velocity change per step: %s  (teacher %s)' % (out['arms'][arm]['dv7_per_step'], out['teacher_dv7_per_step']))
print('GAP_JSON ' + json.dumps(out))
