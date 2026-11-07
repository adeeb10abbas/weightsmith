"""Recipe 6: repurpose an unused MLP unit as a gate that fires only in the target states.

Credit: the blind eyeglasses editor of the controlled study (study/results/ctrl/glasses/, scripts/build_p1b.py), who
turned the least-used unit of the action expert's last block into a "lift while holding" unit (3,072 values;
development 12/32 -> 18/32, held-out 18/64 -> 27/64, p = 0.12). The stack-three-cubes editor proposed the same
design after its linear readout failed the split-half check (study/results/ctrl/stack3/FINAL.md).

When: a linear readout (recipe 5) is not selective enough: its response goes negative or stays large on states that
must not change, so pushing harder breaks them. A GELU unit with a threshold is one-sided: exactly zero below it.
See docs/COOKBOOK.md, section 4.3.

Run:  bin/owsp new cookbook-unit --note "recipe 6"
      bin/owsp py cookbook-unit examples/recipes/06_edit_repurpose_unit.py

How it works. Unit r of ``action.blocks.17.mlp_1`` computes h = gelu(x . g) * (x . u) from the block's MLP input x and
writes h * linear[r] into the residual stream (scaled by the block's adaRMS gate). The recipe

1. rebuilds x from the captured residual and MLP activations (exact adaRMS formulas, at the captured denoising time),
2. picks the least-used unit (mean |h| times the norm of its output row),
3. sets the gate column g to a threshold on the network's own target-vs-silent contrast, so the pre-activation is
   at most -5 (|gelu| < 1e-6) on every silent decision, and the up column u so that x . u is about 1,
4. sets the output row to the least-norm vector that, through the final adaRMS norm and the action head, changes
   the gripper velocity by ``target_dv`` at the median target activation and leaves the other 31 output dims alone.

Check it worked: the rebuilt MLP activations match the captured ones (correlation about 0.99), the unit is zero on
every silent decision, in-sample and in the split-half check, and positive on most targets; the first-order
prediction flips targets only. Then replay the candidate (recipe 9) before closed loop. The eyeglasses editor's
first version of this edit failed that replay (pushing at every denoising step shifts x_t and the flow model
reacts nonlinearly); its final version wrote only at the last step, which needs captures of all 10 steps
(specs/action_readout.json). ctb-ph holds step 0 only, so this run checks the gate at t = 1.0 only.
"""
import json
import math

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "capture": "ctb-ph",   # base replays with action.mlp_hidden and action.resid_out at the block below
    "groups": {"premature release": [0, 21], "cube carry": [21, 30], "banana approach": [30, 37],
               "banana grasp": [37, 42], "banana carry": [42, 49]},
    "target": "premature release",
    "silent_on": ["cube carry", "banana approach", "banana grasp", "banana carry"],  # the unit must be 0 here
    "block": 17,           # action-expert block (its MLP output reaches the head directly)
    "unit": None,          # None = the least-used unit of the block
    "threshold_frac": 0.25,  # threshold between the largest silent and the median target score
    "target_dv": -1.5,     # gripper velocity change at the median target activation (negative = toward closed)
    "out_dim": 7,
    "save": True,
}


def capture(aid, keys):
    """(index, {key: array}) of capture ``aid``, reading only ``keys`` (the prelude's ``acts()`` reads every array)."""
    with open(f"{DATA_ROOT}/acts/{aid}/index.json") as f:
        idx = json.load(f)
    z = np.load(f"{DATA_ROOT}/acts/{aid}/acts.npz")
    return idx, {k: z[k] for k in keys}


def w64(name):
    return base(name).astype(np.float64)


def silu(x):
    return x / (1 + np.exp(-x))


def gelu(x):  # tanh approximation, as flax's nn.gelu used by OpenPI
    return 0.5 * x * (1 + np.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def time_cond(t, width):
    """adaRMS conditioning of the action expert at flow time t (sin-cos embedding, then the time MLP)."""
    frac = np.linspace(0.0, 1.0, width // 2)
    period = 4e-3 * (4.0 / 4e-3) ** frac
    emb = np.concatenate([np.sin(t / period * 2 * np.pi), np.cos(t / period * 2 * np.pi)])
    e = silu(emb @ w64("action.time_mlp_in.kernel") + w64("action.time_mlp_in.bias"))
    return silu(e @ w64("action.time_mlp_out.kernel") + w64("action.time_mlp_out.bias"))


def modulation(prefix, cond):
    """(scale, shift, gate) of an adaRMS norm."""
    m = cond @ w64(prefix + ".Dense_0.kernel") + w64(prefix + ".Dense_0.bias")
    return np.split(m, 3)


def rms_norm(x, scale, shift):
    return x / np.sqrt((x ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + scale) + shift


def main(cfg=SETTINGS):
    blk = f"action.blocks.{cfg['block']:02d}"
    idx, arr = capture(cfg["capture"], ["action.mlp_hidden", "action.resid_out", "actions"])
    if cfg["block"] not in idx["action_layers"]:
        raise SystemExit(f"capture {cfg['capture']} lacks action block {cfg['block']}")
    li = idx["action_layers"].index(cfg["block"])
    M = arr["action.mlp_hidden"][:, 0, li].astype(np.float64).mean(axis=1)  # [N, F] step 0, mean over tokens
    R = arr["action.resid_out"][:, 0, li].astype(np.float64).mean(axis=1)   # [N, D]
    a7 = arr["actions"][:, :, cfg["out_dim"]].astype(np.float64)
    t = float(idx["denoise_t"][0])
    G = w64(blk + ".mlp_1.gating_einsum")       # [2, D, F]: gate and up projections
    L = w64(blk + ".mlp_1.linear")              # [F, D]: row r is what unit r writes
    D = G.shape[1]
    cond = time_cond(t, D)
    scale, shift, gate = modulation(blk + ".pre_ffw_norm_1", cond)
    X = rms_norm(R - gate * (M @ L), scale, shift)  # the block's MLP input (from token means: an approximation)
    rebuilt = gelu(X @ G[0]) * (X @ G[1])
    print(f"capture {cfg['capture']}: {len(R)} decisions at flow time t = {t}; rebuilt MLP activations vs captured: "
          f"correlation {np.corrcoef(rebuilt.ravel(), M.ravel())[0, 1]:.3f}")

    usage = np.abs(M).mean(0) * np.linalg.norm(L, axis=1)
    r = int(np.argmin(usage)) if cfg["unit"] is None else int(cfg["unit"])
    print(f"unit {r}: usage {usage[r]:.4f} (median over units {np.median(usage):.3f}); its old function is replaced")

    groups = {g: np.arange(lo, hi) for g, (lo, hi) in cfg["groups"].items()}
    tgt = groups[cfg["target"]]
    sil = np.concatenate([groups[g] for g in cfg["silent_on"]])

    def build_gate(tg, sl):
        e = X.mean(0) / np.linalg.norm(X.mean(0))
        kappa = float((X @ e).mean())
        d = X[tg].mean(0) - X[sl].mean(0)
        gh = d - (d @ e) * e
        gh /= np.linalg.norm(gh)
        Z = X @ gh
        hi_sil, med_t = Z[sl].max(), np.median(Z[tg])
        if med_t <= hi_sil:
            raise SystemExit("no threshold separates the median target from every silent decision on this "
                             "contrast; choose other groups or another layer")
        thr = hi_sil + cfg["threshold_frac"] * (med_t - hi_sil)
        w = gh - thr * e / kappa
        s = X @ w
        alpha = 5.0 / -s[sl].max()  # silent pre-activations end at -5 or below, where gelu is ~0
        return alpha * w, e / kappa

    g_col, u_col = build_gate(tgt, sil)
    h = gelu(X @ g_col) * (X @ u_col)
    for name, ix in groups.items():
        print(f"  {name:<20} unit activation: median {np.median(h[ix]):6.3f}, largest |h| {np.abs(h[ix]).max():7.4f}, "
              f"above 0.1 on {int((h[ix] > 0.1).sum())}/{len(ix)}")
    parity = np.array([int(it["episode"].rsplit("s", 1)[-1]) % 2 for it in idx["items"]])
    for fit in (0, 1):
        gc, uc = build_gate(tgt[parity[tgt] == fit], sil[parity[sil] == fit])
        hh = gelu(X @ gc) * (X @ uc)
        ts, ss = tgt[parity[tgt] != fit], sil[parity[sil] != fit]
        print(f"  split-half (gate from {'even' if fit == 0 else 'odd'} episodes): held-out targets above 0.1 on "
              f"{int((hh[ts] > 0.1).sum())}/{len(ts)}, held-out silent largest |h| {np.abs(hh[ss]).max():.4f}")

    # output row: least-norm, d(velocity)/d(row) through the block gate, the final norm and the action head
    fscale, _, _ = modulation("action.final_norm", cond)
    Wout = w64("action.action_out_proj.kernel")  # [D, 32]
    href = float(np.median(h[tgt]))
    rms_ref = float(np.median(np.sqrt((R[tgt] ** 2).mean(-1))))
    A = (gate * (1 + fscale) / rms_ref)[:, None] * Wout  # [D, 32]: velocity change per unit of row, per output dim
    want = np.zeros(Wout.shape[1])
    want[cfg["out_dim"]] = cfg["target_dv"] / href
    row = A @ np.linalg.solve(A.T @ A, want)
    rms = np.sqrt((R ** 2).mean(-1))
    dv = (h[:, None] * (gate * (1 + fscale) * row)[None, :] / rms[:, None]) @ Wout  # [N, 32], first order
    new_a7 = a7 - dv[:, [cfg["out_dim"]]] / 2.0  # same change at all 10 steps: x7 moves by -dv, a7 = (x7 + 1) / 2
    closed = lambda a: (a > 0.5).mean(-1) >= 0.5  # noqa: E731
    print(f"output row: |row| {np.linalg.norm(row):.1f} (block {cfg['block']} adaRMS gate mean "
          f"{np.abs(gate).mean():.4f}); first-order velocity change on the other 31 dims "
          f"<= {np.abs(np.delete(dv, cfg['out_dim'], axis=1)).max():.1e}")
    for name, ix in groups.items():
        print(f"  {name:<20} gripper velocity change {dv[ix, cfg['out_dim']].mean():+.2f}   gripper closed "
              f"{int(closed(a7[ix]).sum())} -> {int(closed(new_a7[ix]).sum())} of {len(ix)}")

    if cfg["save"]:
        Gn, Ln = base(blk + ".mlp_1.gating_einsum").copy(), base(blk + ".mlp_1.linear").copy()
        Gn[0, :, r], Gn[1, :, r], Ln[r, :] = g_col, u_col, row
        Gn, Ln = bf16(Gn), bf16(Ln)
        n = int(np.count_nonzero(Gn != base(blk + ".mlp_1.gating_einsum"))
                + np.count_nonzero(Ln != base(blk + ".mlp_1.linear")))
        save(blk + ".mlp_1.gating_einsum", Gn)
        save(blk + ".mlp_1.linear", Ln)
        print(f"saved unit {r}: gate and up columns plus output row, {n} stored values changed. Next: recipe 9.")
    return r, h


if __name__ == "__main__":
    main()
