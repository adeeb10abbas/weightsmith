#!/usr/bin/env python3
"""Render the `owsp py` sandbox scripts of the sgdw edit with weightsmith.distill (run from the repo root).

  python3 study/results/sgdw/make_scripts.py build  CFG.json > build.py   # the distilled edit (save into a candidate)
  python3 study/results/sgdw/make_scripts.py ctl    CFG.json > ctl.py     # random-direction control of the same edit
  python3 study/results/sgdw/make_scripts.py groups GROUPS.json > build.py # keys averaged within decision groups

GROUPS.json is CFG.json plus "groups": [[orig_items, expl_items], ...]. Each group contributes its own 32 image-cell
keys (the group's mean MLP input under the original wording) and values (the group's mean explicit-minus-original
residual increments); the groups' keys are stacked into one closed-form solve per layer. One group holding every item
is the standard build; one group per item gives individual keys.

CFG.json holds weightsmith.distill.BuildConfig fields (orig_capture, expl_capture, layers, mode, alpha, lam_rel,
orig_items, expl_items, instruction, explicit_instruction, save_factors, ...).

The control recomputes the distilled update of every layer from the same captures with the same code (the rendered
build script's functions), keeps its singular values (same rank and Frobenius norm per layer), and replaces its left
and right singular vectors with random orthonormal bases (numpy default_rng(0)), as in
study/results/opus/controls/ctl_rand.py for the paper's control.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from weightsmith import distill  # noqa: E402

CTL_BODY = r'''
def _sb_kv_groups(cfg, io):
    # keys/values per layer exactly as the build: one group of all items, or the groups of a grouped build
    layers = [int(L) for L in cfg["layers"]]
    assert cfg["mode"] == "image", "control implemented for image-cell edits"
    keys = ["vlm.img_mlp_hidden", "vlm.img_resid_out"]
    cache = {}

    def cap(aid):
        if aid not in cache:
            cache[aid] = _sb_load_capture(io["data_root"], aid, keys)
        return cache[aid]

    idx_o, _ = cap(cfg["orig_capture"])
    idx_x, _ = cap(cfg["expl_capture"])
    groups = cfg.get("groups") or [[cfg.get("orig_items") or list(range(len(idx_o["items"]))),
                                    cfg.get("expl_items") or list(range(len(idx_x["items"])))]]
    cells = cfg.get("cells") or [[c, e] for c in (0, 1) for e in range(1, 17)]
    Ks, Vs = {L: [] for L in layers}, {L: [] for L in layers}
    for grp in groups:
        if isinstance(grp, dict):
            (idx_o, arr_o), (idx_x, arr_x) = cap(grp["orig_capture"]), cap(grp["expl_capture"])
            oi, xi = grp["orig_items"], grp["expl_items"]
        else:
            (idx_o, arr_o), (idx_x, arr_x) = cap(cfg["orig_capture"]), cap(cfg["expl_capture"])
            oi, xi = grp
        for a, b in zip(oi, xi):
            ia, ib = idx_o["items"][a], idx_x["items"][b]
            assert (ia["episode"], ia["k"]) == (ib["episode"], ib["k"])
        kv = build_keys_values({k: v[oi] for k, v in arr_o.items()}, {k: v[xi] for k, v in arr_x.items()},
                               layers=layers, mode="image", alpha=float(cfg.get("alpha", 1.0)), cells=cells,
                               capture_layers=idx_o["vlm_layers"])
        for L in layers:
            Ks[L].append(kv[L][0])
            Vs[L].append(kv[L][1])
    return layers, {L: (np.concatenate(Ks[L]), np.concatenate(Vs[L])) for L in layers}


def _sb_ctl(cfg, io):
    # Random-direction control: same layers, rank and singular values as the distilled edit, random directions.
    layers, kv = _sb_kv_groups(cfg, io)
    rng = np.random.default_rng(0)
    rep = []
    for L in layers:
        K, V = kv[L]
        name = (cfg.get("tensor") or MLP_DOWN).format(L)
        G = K @ K.T
        lam = float(cfg.get("lam_rel", 1e-3)) * np.trace(G) / len(G)
        X = np.linalg.solve(G + lam * np.eye(len(G)), V)              # dW = K^T X
        Qa, Ra = np.linalg.qr(K.T)
        Qb, Rb = np.linalg.qr(X.T)
        u, s, vt = np.linalg.svd(Ra @ Rb.T)
        r = int((s > s[0] * 1e-6).sum())
        W0 = io["base"](name).astype(np.float64)
        # create the control candidate with `owsp new CTL --from EDIT`: load() then returns the edit's stored tensor
        stored = io["load"](name)
        reproduces = float(np.mean(io["bf16"](W0 + K.T @ X) == stored))
        Ur = np.linalg.qr(rng.standard_normal((W0.shape[0], r)))[0]
        Vr = np.linalg.qr(rng.standard_normal((W0.shape[1], r)))[0]
        dR = (Ur * s[:r]) @ Vr.T
        new = io["bf16"](W0 + dR)
        io["save"](name, new)
        row = {"layer": L, "tensor": name, "keys": int(len(K)), "rank": r, "reproduces_edit": reproduces,
               "fro_edit": float(np.linalg.norm(K.T @ X)),
               "fro_rand": float(np.linalg.norm(dR)), "top_sv": float(s[0]),
               "changed_fraction": float(np.count_nonzero(new != W0) / new.size)}
        rep.append(row)
        print(json.dumps(row), flush=True)
    return {"config": cfg, "layers": rep}
'''


GROUPS_BODY = r'''
def _sb_build_groups(cfg, io):
    """Distilled edit with keys averaged within decision groups, stacked into one solve per layer."""
    import hashlib as _hl
    layers = [int(L) for L in cfg["layers"]]
    assert cfg["mode"] == "image", "groups implemented for image-cell edits"
    keys = ["vlm.img_mlp_hidden", "vlm.img_resid_out"]
    cache = {}

    def cap(aid):
        if aid not in cache:
            cache[aid] = _sb_load_capture(io["data_root"], aid, keys)
        return cache[aid]

    cells = cfg.get("cells") or [[c, e] for c in (0, 1) for e in range(1, 17)]
    Ks, Vs = {L: [] for L in layers}, {L: [] for L in layers}
    for grp in cfg["groups"]:
        if isinstance(grp, dict):  # {"orig_capture", "expl_capture", "orig_items", "expl_items"}
            (idx_o, arr_o), (idx_x, arr_x) = cap(grp["orig_capture"]), cap(grp["expl_capture"])
            oi, xi = grp["orig_items"], grp["expl_items"]
        else:  # [orig_items, expl_items] of the config's captures
            (idx_o, arr_o), (idx_x, arr_x) = cap(cfg["orig_capture"]), cap(cfg["expl_capture"])
            oi, xi = grp
        assert len(oi) == len(xi)
        for a, b in zip(oi, xi):
            ia, ib = idx_o["items"][a], idx_x["items"][b]
            assert (ia["episode"], ia["k"]) == (ib["episode"], ib["k"]), (a, b)
            assert ia["prompt"] == cfg["instruction"] and ib["prompt"] == cfg["explicit_instruction"], (a, b)
        kv = build_keys_values({k: v[oi] for k, v in arr_o.items()}, {k: v[xi] for k, v in arr_x.items()},
                               layers=layers, mode="image", alpha=float(cfg.get("alpha", 1.0)), cells=cells,
                               capture_layers=idx_o["vlm_layers"])
        for L in layers:
            Ks[L].append(kv[L][0])
            Vs[L].append(kv[L][1])
    report = []
    for L in layers:
        K, V = np.concatenate(Ks[L]), np.concatenate(Vs[L])
        dW = closed_form_edit(K, V, float(cfg.get("lam_rel", 1e-3)))
        name = (cfg.get("tensor") or MLP_DOWN).format(L)
        W0 = io["base"](name)
        new = io["bf16"](W0 + dW)
        io["save"](name, new)
        row = {"layer": L, "tensor": name, "keys": int(len(K)),
               "fit_rel_err": float(np.linalg.norm(K @ dW - V) / max(np.linalg.norm(V), 1e-300)),
               "dW_rel_norm": float(np.linalg.norm(dW) / np.linalg.norm(W0)),
               "changed_fraction": float(np.count_nonzero(new != W0) / new.size),
               "sha256": _hl.sha256(np.ascontiguousarray(new, dtype=np.float32).tobytes()).hexdigest()[:16]}
        report.append(row)
        print(f"L{L:2d} {name!s:<34} keys {row['keys']:4d}  fit-rel-err {row['fit_rel_err']:.2e}  "
              f"|dW|/|W| {row['dW_rel_norm']:.4f}  changed {100 * row['changed_fraction']:.1f}%", flush=True)
    summary = {"config": cfg, "layers": report}
    with open(f"{io['scratch']}/distill_manifest.json", "w") as f:
        json.dump(summary, f, indent=1)
    return summary
'''


def main() -> None:
    kind, path = sys.argv[1], sys.argv[2]
    raw = json.loads(Path(path).read_text())
    groups = raw.pop("groups", None)
    cfg = distill.BuildConfig.from_dict(raw)
    if kind == "build":
        sys.stdout.write(distill.render_sandbox_script(cfg))
    elif kind == "ctl":
        # the build script with its last two statements (run the build, print its result) swapped for the control
        text = distill.render_sandbox_script(cfg)
        head, sep, _ = text.rpartition("_RESULT = _sb_build(CFG, _IO)")
        assert sep, "unexpected build script layout"
        text = (head.replace("# distill build:", "# random-direction control of the distilled edit; build:", 1)
                + CTL_BODY + (f"\n\nCFG['groups'] = {json.dumps(groups)}" if groups else "")
                + "\n\n_RESULT = _sb_ctl(CFG, _IO)\n\nprint('DISTILL_JSON ' + json.dumps(_RESULT, default=float))\n")
        sys.stdout.write(text)
    elif kind == "groups":
        assert groups, "groups mode needs a 'groups' list in the config"
        text = distill.render_sandbox_script(cfg)
        head, sep, _ = text.rpartition("_RESULT = _sb_build(CFG, _IO)")
        assert sep, "unexpected build script layout"
        text = (head.replace("# distill build:", f"# distill build with {len(groups)} key groups:", 1) + GROUPS_BODY
                + f"\n\nCFG['groups'] = {json.dumps(groups)}\n_RESULT = _sb_build_groups(CFG, _IO)\n\n"
                + "print('DISTILL_JSON ' + json.dumps(_RESULT, default=float))\n")
        sys.stdout.write(text)
    else:
        raise SystemExit("usage: make_scripts.py build|ctl|groups CFG.json")


if __name__ == "__main__":
    main()
