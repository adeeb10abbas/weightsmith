"""Recipe 8: the random-direction control for any edit: same size and shape, random directions.

Credit: the paper's control for the distilled edit (Appendix C; study/results/opus/controls/ctl_rand.py). On
held-out seeds the control scored 9/64 against 13/64 unedited (p = 0.45) while the edit scored 50/64: the fix
comes from the edit's directions, not from perturbing the weights by that much.

When: before you believe a gain, and always when an edit is large. Score the control exactly like the edit (same
practice slots, then the same held-out slots). See docs/COOKBOOK.md, section 4.5.

Run:  bin/owsp new cookbook-control --from cookbook-distill --note "recipe 8: control for cookbook-distill"
      bin/owsp py cookbook-control examples/recipes/08_random_control.py

``--from`` copies the edit's tensors into the new candidate; this recipe then replaces each edited tensor's change
by a random one of the same size:

* ``svd`` (dense, low-rank edits such as distillation): keep the change's singular values (so the rank, spectrum and
  Frobenius norm match) and replace its singular vectors with random orthonormal ones;
* ``support`` (sparse edits such as a readout column or a few unit rows): keep exactly which values change and
  their total size, with random values.

Check it worked: per tensor, the control's Frobenius norm and kept singular values match the edit's, and the
singular values it drops (bf16 rounding noise) are well below the ones it keeps. Stored values are rounded to bf16,
so the rank is estimated (31-34 for the 32-key edit here); set ``rank`` when you know it, or recompute the float64
update as the paper did. Run it once, on a fresh copy: a second run would randomize the control itself. A real
edit should beat its control in closed loop; if the control "helps" as much, the gain is not from the edit's
directions.
"""
import numpy as np

SETTINGS = {
    "mode": "auto",     # "svd", "support", or "auto": svd when at least half of a tensor's values change (dense)
    "rank_tol": 1e-2,   # svd: keep singular values above rank_tol x the largest (rounding to bf16 adds a noise floor)
    "rank": None,       # svd: keep exactly this many instead (e.g. the number of keys of a distilled edit)
    "seed": 0,
    "save": True,
}


def singular_values(D):
    """Singular values of a 2-D array, largest first, from its Gram matrix on the smaller side."""
    G = D.T @ D if D.shape[0] >= D.shape[1] else D @ D.T
    return np.sqrt(np.clip(np.linalg.eigvalsh(G)[::-1], 0, None))


def random_like(delta, mode, rank_tol, rng, rank=None):
    """A random change with the same singular values (svd) or the same support and norm (support)."""
    D = delta.reshape(-1, delta.shape[-1]) if delta.ndim > 1 else delta.reshape(1, -1)
    info = {}
    if mode == "support":
        R = np.zeros_like(D)
        nz = np.nonzero(D)
        R[nz] = rng.standard_normal(len(nz[0]))
        R *= np.linalg.norm(D) / np.linalg.norm(R)
        info["values"] = int(len(nz[0]))
    else:
        s = singular_values(D)
        r = int(rank) if rank else int((s > rank_tol * s[0]).sum())
        info.update(rank=r, top=float(s[0]), last_kept=float(s[r - 1]),
                    first_dropped=float(s[r]) if r < len(s) else 0.0)
        Ur = np.linalg.qr(rng.standard_normal((D.shape[0], r)))[0]
        Vr = np.linalg.qr(rng.standard_normal((D.shape[1], r)))[0]
        R = (Ur * s[:r]) @ Vr.T
    return R.reshape(delta.shape), info


def main(cfg=SETTINGS):
    names = saved()
    if not names:
        raise SystemExit("this candidate has no edits: create it with `owsp new NAME --from EDIT` first")
    rng = np.random.default_rng(cfg["seed"])
    for name in names:
        W0 = base(name).astype(np.float64)
        delta = load(name).astype(np.float64) - W0
        frac = np.count_nonzero(delta) / delta.size
        mode = cfg["mode"] if cfg["mode"] != "auto" else ("svd" if frac >= 0.5 else "support")
        R, info = random_like(delta, mode, cfg["rank_tol"], rng, cfg.get("rank"))
        new = bf16(W0 + R)
        changed = int(np.count_nonzero(new != base(name)))
        line = (f"{name:<40} {mode:<7} |edit| {np.linalg.norm(delta):8.3f}  |control| {np.linalg.norm(R):8.3f}  "
                f"values changed: edit {np.count_nonzero(delta):,}, control {changed:,}")
        if mode == "svd":
            line += (f"\n{'':<48}rank {info['rank']}: singular values {info['top']:.3g} ... {info['last_kept']:.3g} "
                     f"kept, next {info['first_dropped']:.2g} dropped")
        print(line)
        if cfg["save"]:
            save(name, new)
    if cfg["save"]:
        print(f"saved the control for {len(names)} tensors. Score it exactly like the edit: same slots, same protocol.")


if __name__ == "__main__":
    main()
