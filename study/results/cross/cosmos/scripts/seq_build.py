"""Sequential closed-form distillation of a wording into Cosmos3 Edge's text pathway (cosmos venv, one GPU).

Cosmos3 Edge is a mixture of transformers: text tokens run through the text ("und") weights and attend causally to
text only; video and action tokens run through the "_moe_gen" weights and read the text through each layer's text
keys and values. So the instruction reaches the actions only through the text states, and these depend on the prompt
alone (not on the images, the robot state or the noise; checked here on two decisions).

The edit makes the original wording O produce, at every edited layer, the text states the known-good wording G
produces, at matched positions:

  for L in layers (in order), with the edits of the layers before L already applied (MEMIT-style, as planned in
  study/results/sgdw/README.md, Phase C):
    keys   K = the input of layers.L.mlp.down_proj (9216-d) under O at the matched positions of the conditional pass,
               plus every position of the unconditional (empty-caption, guidance) pass;
    values V = G's residual after layer L minus the current edited model's under O, at the matched positions;
               the unedited model's minus the current edited model's on the unconditional pass (keeps the guidance
               branch where it was, normally 0);
    dW_L   = K^T (K K^T + lam I)^-1 V, lam = lam_rel * tr(K K^T) / n (weightsmith.distill.closed_form_edit),
             stored as factors A = Z^T [2048, n], B = K [n, 9216] (W += A @ B, torch Linear convention),
             applied exactly as edge_common.apply_edit does (float64 sum, rounded to bf16), then the next layer.

Matched positions: --align all pairs position i with i (the two prompts must have the same token count; T8's RF and
TF wordings do); --align ps pairs the longest common prefix and suffix only (weightsmith.distill.align_slots).

The control (OUT-ctl.pt) has, per layer, the edit's rank and singular values along random orthonormal directions
(numpy default_rng(--seed)), as study/results/opus/controls/ctl_rand.py.

usage: seq_build.py --obs OBS.npz --k 1 --orig "..." --good "..." --layers 0-26 --out edits/NAME [--align auto]
writes NAME.pt, NAME-ctl.pt, NAME.json (per-layer fit, final match, actions of O / G / O+edit / O+ctl on the
decisions given with --check OBS.npz:K ...)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from capture import decoder_layers, load_decision  # noqa: E402
from edge_common import build_service, canonical, infer_seeded  # noqa: E402

UND = "net.language_model.model.layers.{}.mlp.down_proj.weight"


class _Stop(Exception):
    """Raised from a layer hook once the conditional text pass has reached the last layer needed."""


class TextRec:
    """Hooks: per und call, the input of each layer's mlp.down_proj and the text residual after each layer."""

    def __init__(self, net, layers):
        self.active, self.calls, self.ids, self.want = False, [], [], None  # want: layers to record (None = all)
        self.stop_at, self.min_len = None, 40  # stop after this layer of a text call longer than min_len tokens
        self.hs = [net.language_model.model.embed_tokens.register_forward_hook(self._emb)]
        for L, layer in enumerate(layers):
            self.hs.append(layer.mlp.down_proj.register_forward_hook(self._key(L)))
            self.hs.append(layer.register_forward_hook(self._out(L)))

    def _emb(self, mod, inp, out):
        if self.active:
            self.ids.append(inp[0].detach().reshape(-1).cpu().numpy().astype(np.int64))
            self.calls.append({})

    def _key(self, L):
        def h(mod, inp, out):
            if self.active and self.calls and inp[0].numel() and ("key", L) not in self.calls[-1] and \
                    (self.want is None or L in self.want):
                self.calls[-1][("key", L)] = inp[0].detach().double().cpu().numpy().reshape(-1, inp[0].shape[-1])
        return h

    def _out(self, L):
        def h(mod, inp, out):
            if self.active and self.calls and ("key", L) in self.calls[-1] and ("res", L) not in self.calls[-1]:
                r = out[0]["causal_seq"].detach()
                self.calls[-1][("res", L)] = r.double().cpu().numpy().reshape(-1, r.shape[-1])
            if self.active and self.stop_at == L and self.ids and len(self.ids[-1]) > self.min_len:
                raise _Stop()
        return h

    def run(self, svc, obs, seed, prompt, want=None, stop_at=None):
        """stop_at: abort the request right after the conditional text pass reaches this layer (only the conditional
        call is returned, as (call, None, None)); the denoising passes that follow are skipped."""
        self.active, self.calls, self.ids, self.want, self.stop_at = True, [], [], want, stop_at
        try:
            res = infer_seeded(svc, dict(obs, prompt=prompt), seed)
        except _Stop:
            res = None
        finally:
            self.active, self.want, self.stop_at = False, None, None
        calls = [(i, c) for i, c in zip(self.ids, self.calls) if c]
        if stop_at is not None:
            calls = [t for t in calls if len(t[0]) > self.min_len]
            if res is not None or len(calls) != 1:
                raise SystemExit(f"early stop failed: {[len(i) for i, _ in calls]} text calls, res {res is not None}")
            return calls[0], None, None
        if len(calls) != 2:
            raise SystemExit(f"expected 2 text calls (conditional, unconditional), got {[len(i) for i, _ in calls]}")
        calls.sort(key=lambda t: -len(t[0]))
        return calls[0], calls[1], np.asarray(res["action"], np.float32)


def align(a, b, mode):
    a, b = [int(x) for x in a], [int(x) for x in b]
    if mode == "all" or (mode == "auto" and len(a) == len(b)):
        if len(a) != len(b):
            raise SystemExit(f"--align all needs equal token counts: {len(a)} vs {len(b)}")
        return [(i, i) for i in range(len(a))], "all"
    p = 0
    while p < min(len(a), len(b)) and a[p] == b[p]:
        p += 1
    s = 0
    while s < min(len(a), len(b)) - p and a[len(a) - 1 - s] == b[len(b) - 1 - s]:
        s += 1
    sh = len(b) - len(a)
    return [(i, i) for i in range(p)] + [(i, i + sh) for i in range(len(a) - s, len(a))], "ps"


def parse_layers(s):
    out = []
    for part in s.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def svd_factors(A, B):
    Qa, Ra = np.linalg.qr(A)
    Qb, Rb = np.linalg.qr(B.T)
    u, s, vt = np.linalg.svd(Ra @ Rb.T, full_matrices=False)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--obs", required=True)
    ap.add_argument("--k", type=int, default=1)
    ap.add_argument("--orig", required=True)
    ap.add_argument("--good", required=True)
    ap.add_argument("--layers", default="0-26")
    ap.add_argument("--align", choices=("auto", "all", "ps"), default="auto")
    ap.add_argument("--lam-rel", type=float, default=1e-3)
    ap.add_argument("--no-uncond", action="store_true", help="no preservation keys from the unconditional pass")
    ap.add_argument("--preserve-file", default=None,
                    help="JSON {'kept': [[task, instruction], ...]}: instructions whose text states must not change "
                         "(zero-change keys at every position after the prompt template's common prefix)")
    ap.add_argument("--preserve-stride", type=int, default=1, help="keep every n-th preserved position (and the last)")
    ap.add_argument("--gpu-solve", action="store_true", help="Gram matrix, solve and update in float64 on the GPU")
    ap.add_argument("--early-stop", action="store_true",
                    help="preserved prompts: stop each request after the text pass (states identical, much faster)")
    ap.add_argument("--no-ctl", action="store_true", help="skip the random-direction control (saves the SVDs)")
    ap.add_argument("--probe", nargs="*", default=[], help="NAME=PROMPT: report text-state drift for these after the edit")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--check", nargs="*", default=[], help="OBS.npz:K decisions to replay O, G, O+edit, O+ctl on")
    a = ap.parse_args()
    t0 = time.time()
    native, svc = build_service()
    model = svc.model
    layers_mod = decoder_layers(svc)
    params = {canonical(n): p for n, p in model.named_parameters()}
    rec = TextRec(model.net, layers_mod)
    obs, seed = load_decision(a.obs, a.k)
    layers = parse_layers(a.layers)

    (ig_ids, G), (_, GU), act_g = rec.run(svc, obs, seed, a.good)
    (io_ids, O0), (iu_ids, U0), act_o = rec.run(svc, obs, seed, a.orig)
    pairs, mode = align(io_ids, ig_ids, a.align)
    io = np.array([i for i, _ in pairs])
    ig = np.array([j for _, j in pairs])
    rep = {"orig": a.orig, "good": a.good, "obs": a.obs, "k": a.k, "seed": seed, "layers": layers, "align": mode,
           "lam_rel": a.lam_rel, "n_tokens_orig": len(io_ids), "n_tokens_good": len(ig_ids), "n_tokens_uncond": len(iu_ids),
           "matched": len(pairs), "uncond_keys": 0 if a.no_uncond else len(iu_ids), "per_layer": []}
    # text states depend on the prompt only: a second decision must give the same states
    if a.check:
        p2, k2 = a.check[-1].rsplit(":", 1)
        obs2, seed2 = load_decision(p2, int(k2))
        (_, O2), _, _ = rec.run(svc, obs2, seed2, a.orig)
        rep["text_states_image_independent_maxabs"] = max(float(np.abs(O2[("res", L)] - O0[("res", L)]).max())
                                                         for L in layers)
    def prefix_len(a_ids, b_ids):
        n = 0
        while n < min(len(a_ids), len(b_ids)) and int(a_ids[n]) == int(b_ids[n]):
            n += 1
        return n

    keep = [tuple(x) for x in json.loads(Path(a.preserve_file).read_text())["kept"]] if a.preserve_file else []
    P0 = []  # (instruction, positions, {L: unedited residual at those positions})
    last = max(layers)
    if keep and a.early_stop:  # the early-stopped text pass must give exactly the states of a full request
        (_, sf), _, _ = rec.run(svc, obs, seed, keep[0][1])
        (_, se), _, _ = rec.run(svc, obs, seed, keep[0][1], stop_at=last)
        rep["early_stop_check_maxabs"] = max(float(np.abs(sf[("res", L)] - se[("res", L)]).max()) for L in layers)
        print("early-stop check max abs", rep["early_stop_check_maxabs"], flush=True)
        if rep["early_stop_check_maxabs"] != 0.0:
            raise SystemExit("early-stopped text states differ from a full request")
    for _task, instr in keep:
        (ids, st), _, _ = rec.run(svc, obs, seed, instr, stop_at=last if a.early_stop else None)
        pos = np.arange(prefix_len(ids, io_ids), len(ids))
        pos = np.unique(np.concatenate([pos[::a.preserve_stride], pos[-1:]]))
        P0.append((instr, pos, {L: st[("res", L)][pos].astype(np.float32) for L in layers}))
    rep["preserve"] = {"file": a.preserve_file, "prompts": len(P0), "keys": int(sum(len(x[1]) for x in P0))}
    probes = [tuple(x.split("=", 1)) for x in a.probe]
    Q0 = {}
    for name, instr in probes:
        (ids, st), _, _ = rec.run(svc, obs, seed, instr)
        pos = np.arange(prefix_len(ids, io_ids), len(ids))
        Q0[name] = (instr, pos, {L: st[("res", L)][pos] for L in layers})
    edit, ctl, orig_w = {}, {}, {}
    rng = np.random.default_rng(a.seed)
    for L in layers:
        (_, Oc), (_, Uc), _ = rec.run(svc, obs, seed, a.orig, want={L})  # current edited model, layer L not yet edited
        Kc, Vc = Oc[("key", L)][io], G[("res", L)][ig] - Oc[("res", L)][io]
        if a.no_uncond:
            K, V = Kc, Vc
        else:
            K = np.concatenate([Kc, Uc[("key", L)]])
            V = np.concatenate([Vc, U0[("res", L)] - Uc[("res", L)]])
        pdrift = 0.0
        if P0:
            Ks, Vs = [K], [V]
            for instr, pos, st0 in P0:
                (_, Pc), _, _ = rec.run(svc, obs, seed, instr, want={L}, stop_at=L if a.early_stop else None)
                Ks.append(Pc[("key", L)][pos])
                Vs.append(st0[L].astype(np.float64) - Pc[("res", L)][pos])
                pdrift += float(np.linalg.norm(Vs[-1]) ** 2)
            K, V = np.concatenate(Ks), np.concatenate(Vs)
        name = UND.format(L)
        p = params[name]
        orig_w[name] = p.detach().clone()
        with torch.no_grad():
            if a.gpu_solve:  # same float64 algebra on the GPU (the CPU path takes minutes per layer with ~5k keys)
                dev = p.device
                Kt, Vt = torch.from_numpy(K).to(dev), torch.from_numpy(V).to(dev)
                Gt = Kt @ Kt.T
                lam = float(a.lam_rel * torch.trace(Gt) / len(Gt))
                Zt = torch.linalg.solve(Gt + lam * torch.eye(len(Gt), dtype=Gt.dtype, device=dev), Vt)
                A32, B32 = Zt.T.float().cpu().contiguous(), Kt.float().cpu().contiguous()
                Z = None
                del Gt, Vt, Kt
                D = A32.to(dev).double() @ B32.to(dev).double()
                w0 = p.detach().double()
                new = (w0 + D).to(p.dtype)
                dnorm = float(torch.linalg.norm(D))
                del D
                changed = int((new != p.detach()).sum())
                p.copy_(new)
                w0 = w0.cpu()
            else:
                Gm = K @ K.T
                lam = a.lam_rel * np.trace(Gm) / len(Gm)
                Z = np.linalg.solve(Gm + lam * np.eye(len(Gm)), V)
                A32, B32 = torch.from_numpy(Z.T.astype(np.float32)), torch.from_numpy(K.astype(np.float32))
                w0 = p.detach().double().cpu()
                D = A32.double() @ B32.double()
                dnorm = float(torch.linalg.norm(D))
                new = (w0 + D).to(p.dtype)
                changed = int((new != p.detach().cpu()).sum())
                p.copy_(new.to(p.device))
        edit[name] = {"A": A32, "B": B32}
        if a.no_ctl:
            s, r = np.zeros(0), None
        else:
            s = svd_factors(A32.numpy().astype(np.float64), K)
            r = int((s > s[0] * 1e-6).sum()) if s.size and s[0] > 0 else 0
            Ur = np.linalg.qr(rng.standard_normal((A32.shape[0], r)))[0]
            Vr = np.linalg.qr(rng.standard_normal((K.shape[1], r)))[0]
            ctl[name] = {"A": torch.from_numpy((Ur * s[:r]).astype(np.float32)),
                         "B": torch.from_numpy(Vr.T.astype(np.float32))}
        dfro = dnorm if a.no_ctl else float(np.sqrt((s ** 2).sum()))
        rep["per_layer"].append({"layer": L, "keys": int(len(K)), "rank": r, "fro": dfro,
                                 "rel_fro_w": dfro / float(np.linalg.norm(w0.numpy())),
                                 "lam": float(lam), "gap_before": float(np.linalg.norm(Vc)),
                                 "uncond_drift_before": float(np.linalg.norm(U0[("res", L)] - Uc[("res", L)])),
                                 "preserve_drift_before": float(np.sqrt(pdrift)),
                                 "changed_values": changed})
        print(json.dumps(rep["per_layer"][-1]), flush=True)
    # final match: edited model under O vs G at every layer (matched positions) and unconditional drift
    (_, Oe), (_, Ue), act_e = rec.run(svc, obs, seed, a.orig)
    rep["final"] = [{"layer": L, "rel_gap": float(np.linalg.norm(G[("res", L)][ig] - Oe[("res", L)][io])
                                                  / max(np.linalg.norm(G[("res", L)][ig] - O0[("res", L)][io]), 1e-30)),
                     "uncond_rel_drift": float(np.linalg.norm(U0[("res", L)] - Ue[("res", L)])
                                               / max(np.linalg.norm(U0[("res", L)]), 1e-30))} for L in layers]
    def drift(items):
        res = {}
        for name, (instr, pos, st0) in items:
            (_, Pe), _, _ = rec.run(svc, obs, seed, instr, stop_at=last if a.early_stop else None)
            num = sum(float(np.linalg.norm(Pe[("res", L)][pos] - st0[L]) ** 2) for L in layers)
            den = sum(float(np.linalg.norm(st0[L]) ** 2) for L in layers)
            res[name] = {"prompt": instr, "positions": int(len(pos)), "rel_drift": float(np.sqrt(num / max(den, 1e-30)))}
        return res
    if P0:
        d = drift([(instr, (instr, pos, st)) for instr, pos, st in P0])
        v = [x["rel_drift"] for x in d.values()]
        rep["preserve"].update({"rel_drift_mean": float(np.mean(v)), "rel_drift_max": float(np.max(v))})
    if Q0:
        rep["probe_drift"] = drift(list(Q0.items()))
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(edit, str(out) + ".pt")
    if ctl:
        torch.save(ctl, str(out) + "-ctl.pt")
    rep["sha256_16"] = {f: hashlib.sha256(Path(str(out) + f).read_bytes()).hexdigest()[:16]
                        for f in ((".pt", "-ctl.pt") if ctl else (".pt",))}
    rep["stored_values"] = int(sum(v["A"].numel() + v["B"].numel() for v in edit.values()))
    rep["changed_values"] = int(sum(r["changed_values"] for r in rep["per_layer"]))
    # actions on the check decisions: O, G, O+edit (edited weights live now), then O+ctl
    acts = {"O+edit": [], "O": [], "G": [], "O+ctl": []}
    dec = [(a.obs, a.k)] + [(c.rsplit(":", 1)[0], int(c.rsplit(":", 1)[1])) for c in a.check]
    for pth, k in dec:
        ob, sd = load_decision(pth, k)
        acts["O+edit"].append(np.asarray(infer_seeded(svc, dict(ob, prompt=a.orig), sd)["action"], np.float32))
    with torch.no_grad():  # restore the original weights exactly, then apply the control the way apply_edit does
        for n, w0 in orig_w.items():
            params[n].copy_(w0)
    for pth, k in dec:
        ob, sd = load_decision(pth, k)
        acts["O"].append(np.asarray(infer_seeded(svc, dict(ob, prompt=a.orig), sd)["action"], np.float32))
        acts["G"].append(np.asarray(infer_seeded(svc, dict(ob, prompt=a.good), sd)["action"], np.float32))
    rep["restore_matches_unedited_first_decision"] = bool(np.array_equal(acts["O"][0], act_o))
    with torch.no_grad():
        for n, d in ctl.items():
            w = params[n]
            w.copy_((w.detach().double().cpu() + d["A"].double() @ d["B"].double()).to(w.dtype).to(w.device))
    for pth, k in dec if ctl else []:
        ob, sd = load_decision(pth, k)
        acts["O+ctl"].append(np.asarray(infer_seeded(svc, dict(ob, prompt=a.orig), sd)["action"], np.float32))
    np.savez(str(out) + ".actions.npz", **{k: np.stack(v) for k, v in acts.items() if v},
             decisions=np.array([f"{p}:{k}" for p, k in dec]))
    rep["decisions"] = [f"{p}:{k}" for p, k in dec]
    rep["s"] = round(time.time() - t0, 1)
    Path(str(out) + ".json").write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k not in ("per_layer",)}, indent=1), flush=True)


if __name__ == "__main__":
    main()
