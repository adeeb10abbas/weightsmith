import json
import numpy as np
idx, A = acts("a-diag1")
meta = json.load(open("/work/notes/diag1_meta.json"))
print("index keys:", [k for k in idx.keys()][:40])
print("arrays:", {k: v.shape for k, v in A.items()})
for k in idx:
    if "piece" in k and k.endswith("item0"):
        print(k, idx[k])
for k in ("prompt_token_pieces_item40", "prompt_token_pieces_item66"):
    if k in idx:
        print(k, idx[k])
act = A["actions"]
g = act[:, :, 7].mean(1)
N = len(meta)
orig = {}
for i, m in enumerate(meta):
    if m["prompt"] == "orig":
        orig[(m["slot"], m["k"])] = i
print("\n== gripper chunk mean by item ==")
for i, m in enumerate(meta):
    j = orig.get((m["slot"], m["k"]))
    extra = "" if m["prompt"] == "orig" else f" (orig {g[j]:.2f})"
    print(f'{i:2d} {m["tag"]:5s} {m["slot"]} k{m["k"]:2d} {m["prompt"][:28]:28s} g={g[i]:.2f}{extra}')
# group summaries
def grp(tag, prompt):
    ii = [i for i, m in enumerate(meta) if m["tag"] == tag and m["prompt"] == prompt]
    return ii
for tag in ("rel", "hold", "miss", "close"):
    for p in ("orig", "Put the cube in the bowl", "Put the banana in the bowl"):
        ii = grp(tag, p)
        if ii:
            print(tag, p, "n", len(ii), "mean g", round(float(g[ii].mean()), 3), "n>0.5", int((g[ii] > 0.5).sum()))
# attention mass: action.attn_mass [N,S,La,8,5]
if "action.attn_mass" in A:
    am = A["action.attn_mass"]
    print("attn_mass shape", am.shape)
    for tag in ("rel", "hold", "miss", "close"):
        ii = grp(tag, "orig")
        m_ = am[ii].mean(axis=(0, 1, 3))  # [La,5]
        print(tag, "mass by layer (base,wrist,pad,prompt,act):")
        print(np.round(m_[::3], 3))
np.save("/work/notes/diag1_g.npy", g)
