# First-order selectivity of a head edit (run with `owsp py CAND` for h1, h1-grip, p1-blind): the change of the gripper
# velocity, h . dW[:, 7], that the candidate's action_out_proj edit adds at denoise step 0 of each of the 49 recorded
# decisions (unedited head inputs of capture chk-base, token mean). A negative value pushes the gripper toward closed.
import json
name = 'action.action_out_proj.kernel'
dW = load(name).astype(np.float64) - base(name).astype(np.float64)
ib, B = acts('chk-base')
h = B['action.head_input'][:, 0].astype(np.float64).mean(1)        # [49, 1024], step 0, mean over the 15 tokens
dv7 = h @ dW[:, 7]
G = {'PREM': range(0, 21), 'CARRY': range(21, 30), 'P2app': range(30, 37), 'P2grasp': range(37, 42), 'P2carry': range(42, 49)}
out = {g: [round(float(dv7[list(ix)].mean()), 3), round(float(dv7[list(ix)].min()), 3), round(float(dv7[list(ix)].max()), 3)] for g, ix in G.items()}
print('gripper velocity change at step 0 (mean, min, max) per group:', json.dumps(out))
print('SEL_JSON ' + json.dumps(out))
