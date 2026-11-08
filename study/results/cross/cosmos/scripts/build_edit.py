"""Closed-form instruction distillation for Cosmos3 Edge (numpy, CPU): build the edit and its random control.

From a capture (capture.py) of the same decision(s) under the original wording O and the known-good wording G:

  text pathway ("und"): at each edited layer L, keys K_L = O's input of layers.L.mlp.down_proj (9216-d) at the text
  positions both prompts share (longest common token prefix and suffix, align_slots of weightsmith.distill), values
  V_L = d_L - d_{L-1}, d_L = G's minus O's residual after layer L at the same positions (d_{-1} = 0: aligned
  positions hold the same token). Prefix positions come before the instruction, so their d_L = 0: they are kept as
  zero-value keys and pin the shared prompt template for every other instruction.

  generation pathway ("gen"): keys = O's input of layers.L.mlp_moe_gen.down_proj at the action tokens and pooled video
  cells, per forward pass of the request (denoising step x guidance branch), averaged over the decisions; values =
  G's minus O's residual increment at the same positions and pass.

  dW_L = K^T (K K^T + lam I)^-1 V, lam = lam_rel * tr(K K^T) / n (weightsmith.distill.closed_form_edit), stored as
  low-rank factors A = Z^T [2048, n], B = K [n, 9216] (torch Linear convention, W += A @ B, Z = (KK^T+lam I)^-1 V).
  The control has, per layer, the edit's rank and singular values along random orthonormal directions
  (numpy default_rng(0)), as study/results/opus/controls/ctl_rand.py.

usage: build_edit.py CAPTURE.npz OUT_PREFIX --path und|gen --layers 0-26 [--lam-rel 1e-3] [--alpha 1]
writes OUT_PREFIX.pt (edit), OUT_PREFIX-ctl.pt (random control) and OUT_PREFIX.json (build report)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

UND = "net.language_model.model.layers.{}.mlp.down_proj.weight"
GEN = "net.language_model.model.layers.{}.mlp_moe_gen.down_proj.weight"


def align_slots(a, b):
    """(i_O, i_G) pairs over the longest common prefix and the longest common suffix of two token lists."""
    a, b = [int(x) for x in a], [int(x) for x in b]
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    shift = len(b) - len(a)
    return [(i, i) for i in range(p)] + [(i, i + shift) for i in range(len(a) - s, len(a))], p, s


def closed_form(K, V, lam_rel):
    G = K @ K.T
    lam = lam_rel * np.trace(G) / len(G)
    Z = np.linalg.solve(G + lam * np.eye(len(G)), V)  # [n, d_out]
    return Z, lam


def parse_layers(s):
    out = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("out_prefix")
    ap.add_argument("--path", choices=("und", "gen"), required=True)
    ap.add_argument("--layers", required=True)
    ap.add_argument("--orig", default="O")
    ap.add_argument("--good", default="G")
    ap.add_argument("--lam-rel", type=float, default=1e-3)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--no-prefix", action="store_true", help="drop the zero-value prefix keys")
    ap.add_argument("--align", choices=("ps", "all"), default="ps",
                    help="ps: common prefix + suffix only; all: every position i <-> i (equal token counts only)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    import torch
    X = np.load(a.capture)
    layers = parse_layers(a.layers)
    rep = {"capture": a.capture, "path": a.path, "layers": layers, "lam_rel": a.lam_rel, "alpha": a.alpha,
           "per_layer": []}
    edit, ctl = {}, {}
    rng = np.random.default_rng(a.seed)
    if a.path == "und":
        to, tg = X[f"{a.orig}/token_ids"], X[f"{a.good}/token_ids"]
        pairs, p, s = align_slots(to, tg)
        if a.align == "all":
            if len(to) != len(tg):
                raise SystemExit(f"--align all needs equal token counts, got {len(to)} and {len(tg)}")
            pairs = [(i, i) for i in range(len(to))]
        if a.no_prefix:
            pairs = [(i, j) for i, j in pairs if i >= p]
        io = np.array([i for i, _ in pairs])
        ig = np.array([j for _, j in pairs])
        rep.update(n_tokens_orig=int(len(to)), n_tokens_good=int(len(tg)), prefix=p, suffix=s, keys=len(pairs),
                   align=a.align, no_prefix=a.no_prefix)

        def d(L):
            if L < 0:
                return 0.0
            return X[f"{a.good}/und_res/{L}"].astype(np.float64)[ig] - X[f"{a.orig}/und_res/{L}"].astype(np.float64)[io]

        for L in layers:
            K = X[f"{a.orig}/und_key/{L}"].astype(np.float64)[io]
            V = a.alpha * (d(L) - d(L - 1))
            _finish(L, K, V, UND.format(L), a, rng, edit, ctl, rep)
    else:
        names = sorted({k.split("/")[2] for k in X.files if k.startswith(f"{a.orig}/gen_key/")}, key=int)
        for L in layers:
            Ks, Vs = [], []
            for ci in names:
                ko = X[f"{a.orig}/gen_key/{ci}/{L}"].astype(np.float64)       # [D, P, 9216]
                ro = X[f"{a.orig}/gen_res/{ci}/{L}"].astype(np.float64)       # [D, P, 2048]
                rg = X[f"{a.good}/gen_res/{ci}/{L}"].astype(np.float64)
                if L > 0:
                    ro_ = X[f"{a.orig}/gen_res/{ci}/{L - 1}"].astype(np.float64)
                    rg_ = X[f"{a.good}/gen_res/{ci}/{L - 1}"].astype(np.float64)
                    prev = (rg_ - ro_).mean(0)
                else:
                    prev = 0.0
                Ks.append(ko.mean(0))
                Vs.append(a.alpha * ((rg - ro).mean(0) - prev))
            _finish(L, np.concatenate(Ks), np.concatenate(Vs), GEN.format(L), a, rng, edit, ctl, rep)
    torch.save(edit, a.out_prefix + ".pt")
    torch.save(ctl, a.out_prefix + "-ctl.pt")
    for f in (a.out_prefix + ".pt", a.out_prefix + "-ctl.pt"):
        rep.setdefault("sha256_16", {})[Path(f).name] = hashlib.sha256(Path(f).read_bytes()).hexdigest()[:16]
    rep["stored_values"] = int(sum(v["A"].numel() + v["B"].numel() for v in edit.values()))
    Path(a.out_prefix + ".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "per_layer"}, indent=1))
    for r in rep["per_layer"]:
        print(json.dumps(r))


def _finish(L, K, V, name, a, rng, edit, ctl, rep):
    import torch
    Z, lam = closed_form(K, V, a.lam_rel)
    A = Z.T                                     # [d_out, n]
    B = K                                       # [n, d_in]
    fit = K @ (K.T @ Z)                         # K dW' with dW' = K^T Z  -> [n, d_out]
    rel_fit = float(np.linalg.norm(fit - V) / max(np.linalg.norm(V), 1e-30))
    # exact SVD of dW = A @ B through thin QR of the factors
    Qa, Ra = np.linalg.qr(A)
    Qb, Rb = np.linalg.qr(B.T)
    u, s, vt = np.linalg.svd(Ra @ Rb.T, full_matrices=False)
    r = int((s > s[0] * 1e-6).sum()) if s.size and s[0] > 0 else 0
    Ur = np.linalg.qr(rng.standard_normal((A.shape[0], r)))[0]
    Vr = np.linalg.qr(rng.standard_normal((B.shape[1], r)))[0]
    edit[name] = {"A": torch.from_numpy(A.astype(np.float32)), "B": torch.from_numpy(B.astype(np.float32))}
    ctl[name] = {"A": torch.from_numpy((Ur * s[:r]).astype(np.float32)), "B": torch.from_numpy(Vr.T.astype(np.float32))}
    rep["per_layer"].append({"layer": L, "keys": int(K.shape[0]), "rank": r, "fro": float(np.sqrt((s ** 2).sum())),
                             "top_sv": float(s[0]) if s.size else 0.0, "rel_fit_err": rel_fit, "lam": float(lam),
                             "V_norm": float(np.linalg.norm(V)), "K_norm_mean": float(np.linalg.norm(K, axis=1).mean())})


if __name__ == "__main__":
    main()
