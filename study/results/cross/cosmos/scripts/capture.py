"""Replay recorded Cosmos3 Edge decisions under several variants and capture activations (cosmos venv, GPU1).

A decision is one recorded request (obs_r0_e{env}.npz from cosmos_client.py, index k): conditioning composite,
joints, gripper and the sampling seed it used. Every variant replays the same decisions with the same seed, so only
the wording or the weights differ. Variants: {"name", "prompt", "edit"?: path of an edge_common edit}.

Captured (variants with "capture": true):
- text (und) pathway, layers in "und_layers": the input of layers.L.mlp.down_proj (9216-d) and the residual after
  layer L (2048-d) at every text token of the conditional pass (the prompt's text states depend on the prompt only;
  the first decision's are stored, and later decisions are checked to match);
- generation (video/action) pathway, layers in "gen_layers": the input of layers.L.mlp_moe_gen.down_proj and the
  residual after layer L at the action tokens and at pooled video cells, for every forward pass of the request
  (denoising steps x guidance branches), stored per decision.
Always stored: the returned action chunk [32, 8] of every variant and decision, and the prompt token ids.

usage: capture.py SPEC.json OUT.npz   (run through gpu1.sh; TORCHDYNAMO_DISABLE=1 so hooks see eager modules)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from edge_common import apply_edit, build_service, canonical, infer_seeded  # noqa: E402


def decoder_layers(svc):
    lm = svc.model.net.language_model.model
    return [getattr(l, "_orig_mod", l) for l in lm.layers]


def load_decision(path, k):
    d = np.load(path)
    return ({"observation/image": d["image"][k], "observation/joint_position": d["joint"][k].astype(np.float64),
             "observation/gripper_position": d["gripper"][k].astype(np.float64)}, int(d["seed"][k]))


class Recorder:
    """Forward hooks on the chosen layers; collects per-call tensors for one request at a time."""

    def __init__(self, net, layers, und_layers, gen_layers, pool):
        self.layers, self.und_layers, self.gen_layers, self.pool = layers, set(und_layers), set(gen_layers), pool
        self.handles, self.calls = [], []
        self.active = False
        self.cur = None
        self.layout = []          # per network call: gen token layout
        self.token_ids = []       # per text-embedding call: the token ids
        self.handles.append(net.register_forward_pre_hook(self._net_pre, with_kwargs=True))
        emb = net.language_model.model.embed_tokens
        self.handles.append(emb.register_forward_hook(self._emb))
        for L, layer in enumerate(layers):
            if L in self.und_layers:
                self.handles.append(layer.mlp.down_proj.register_forward_hook(self._mk(L, "und_key")))
            if L in self.gen_layers:
                self.handles.append(layer.mlp_moe_gen.down_proj.register_forward_hook(self._mk(L, "gen_key")))
            if L in self.und_layers or L in self.gen_layers:
                self.handles.append(layer.register_forward_hook(self._mk_out(L)))
            if L == 0:
                self.handles.append(layer.register_forward_pre_hook(self._new_call))

    def _net_pre(self, mod, args, kwargs):
        if not self.active:
            return
        ps = args[0] if args else kwargs.get("packed_seq")
        info = {"temporal_causal": bool(getattr(mod, "video_temporal_causal", False))}
        try:
            if ps.vision is not None:
                info["vision_token_shapes"] = [list(map(int, s)) for s in ps.vision.token_shapes]
                info["n_vision"] = int(ps.vision.sequence_indexes.numel())
            if ps.action is not None and isinstance(ps.action.sequence_indexes, torch.Tensor):
                info["n_action"] = int(ps.action.sequence_indexes.numel())
            info["n_text"] = int(ps.text_indexes.numel()) if isinstance(ps.text_indexes, torch.Tensor) else None
        except Exception as e:  # layout is informative only
            info["error"] = repr(e)
        self.layout.append(info)

    def _emb(self, mod, inp, out):
        if self.active:
            self.token_ids.append(inp[0].detach().reshape(-1).cpu().numpy().astype(np.int64))

    def _new_call(self, mod, args):
        if self.active:
            self.cur = {}
            self.calls.append(self.cur)

    def _mk(self, L, kind):
        def h(mod, inp, out):
            if self.active and self.cur is not None:
                self.cur[(L, kind)] = inp[0].detach()
        return h

    def _mk_out(self, L):
        def h(mod, inp, out):
            if not (self.active and self.cur is not None):
                return
            pack = out[0]
            if L in self.und_layers:
                self.cur[(L, "und_res")] = pack["causal_seq"].detach()
            if L in self.gen_layers:
                self.cur[(L, "gen_res")] = pack["full_only_seq"].detach()
        return h

    def start(self):
        self.calls, self.cur, self.active = [], None, True
        self.layout, self.token_ids = [], []

    def stop(self):
        self.active = False
        return self.calls


def pool_gen(x, n_action, video_shape, grid):
    """x [N_gen, F] = video tokens (T, H, W row-major) then n_action action tokens -> [T*gh*gw + n_action, F]."""
    T, H, W = video_shape
    nv = T * H * W
    v = x[:nv].float().reshape(T, H, W, -1)
    gh, gw = grid
    hs = np.array_split(np.arange(H), gh)
    ws = np.array_split(np.arange(W), gw)
    cells = [v[:, h0[0]:h0[-1] + 1, w0[0]:w0[-1] + 1].mean((1, 2)) for h0 in hs for w0 in ws]  # each [T, F]
    cells = torch.stack(cells, 1).reshape(T * gh * gw, -1)
    return torch.cat([cells, x[nv:nv + n_action].float()], 0)


def main():
    spec = json.loads(Path(sys.argv[1]).read_text())
    out_path = sys.argv[2]
    t0 = time.time()
    native, svc = build_service()
    layers = decoder_layers(svc)
    nl = len(layers)
    und_layers = list(range(nl)) if spec.get("und_layers") == "all" else spec.get("und_layers", [])
    gen_layers = list(range(nl)) if spec.get("gen_layers") == "all" else spec.get("gen_layers", [])
    rec = Recorder(svc.model.net, layers, und_layers, gen_layers, spec.get("gen_grid", [2, 2]))
    params = {canonical(n): p for n, p in svc.model.named_parameters()}
    decisions = [load_decision(d["obs"], d["k"]) for d in spec["decisions"]]
    out: dict[str, np.ndarray] = {}
    report = {"load_s": round(time.time() - t0, 1), "n_layers": nl, "variants": [], "spec": spec}
    for var in spec["variants"]:
        saved = {}
        if var.get("edit"):
            delta = torch.load(var["edit"], map_location="cpu", weights_only=True)
            saved = {canonical(n): params[canonical(n)].detach().clone() for n in delta}
            report.setdefault("edit_reports", {})[var["name"]] = apply_edit(svc.model, var["edit"])
        cap = bool(var.get("capture"))
        acts, und_ref = [], None
        gen_store = {}
        tv = time.time()
        for i, (obs, seed) in enumerate(decisions):
            if cap:
                rec.start()
            res = infer_seeded(svc, dict(obs, prompt=var["prompt"]), spec.get("seed_override", seed))
            acts.append(np.asarray(res["action"], np.float32))
            if not cap:
                continue
            calls = rec.stop()
            if i == 0:
                ids = max(rec.token_ids, key=len) if rec.token_ids else np.zeros(0, np.int64)
                out[f"{var['name']}/token_ids"] = ids
                report.setdefault("layout", {})[var["name"]] = rec.layout
                report.setdefault("n_text_ids", {})[var["name"]] = [int(len(t)) for t in rec.token_ids]
            # text states: the conditional pass is the first call whose text holds the prompt (most tokens)
            und_calls = [c for c in calls if c and any(k[1] == "und_key" for k in c)]
            if und_layers and und_calls:
                main_call = max(und_calls, key=lambda c: c[(und_layers[0], "und_key")].shape[0])
                cur = {L: (main_call[(L, "und_key")].float().cpu().numpy(), main_call[(L, "und_res")].float().cpu().numpy())
                       for L in und_layers}
                if und_ref is None:
                    und_ref = cur
                    for L, (k_, r_) in cur.items():
                        out[f"{var['name']}/und_key/{L}"] = k_.astype(np.float32)
                        out[f"{var['name']}/und_res/{L}"] = r_.astype(np.float32)
                else:
                    dev = max(float(np.abs(cur[L][1] - und_ref[L][1]).max()) for L in und_layers)
                    report.setdefault("und_consistency_maxabs", {}).setdefault(var["name"], []).append(dev)
                report.setdefault("calls", {})[var["name"]] = [
                    {f"{k[0]}:{k[1]}": int(v.shape[0]) for k, v in c.items() if k[0] in (und_layers + gen_layers)[:1]}
                    for c in calls]
            if gen_layers:
                lay = rec.layout[0]
                if lay.get("temporal_causal"):
                    raise SystemExit("temporal-causal gen layout: pooling not implemented")
                vs = lay["vision_token_shapes"][0]  # [T, H, W] video latent tokens, then the action tokens
                na = lay.get("n_action", 0)
                for ci, c in enumerate(calls):
                    for L in gen_layers:
                        if (L, "gen_key") not in c:
                            continue
                        gen_store.setdefault((ci, L, "key"), []).append(
                            pool_gen(c[(L, "gen_key")], na, vs, rec.pool).cpu().numpy().astype(np.float16))
                        gen_store.setdefault((ci, L, "res"), []).append(
                            pool_gen(c[(L, "gen_res")], na, vs, rec.pool).cpu().numpy().astype(np.float16))
        out[f"{var['name']}/actions"] = np.stack(acts)
        for (ci, L, kind), arrs in gen_store.items():
            out[f"{var['name']}/gen_{kind}/{ci}/{L}"] = np.stack(arrs)
        report["variants"].append({"name": var["name"], "prompt": var["prompt"], "edit": var.get("edit"),
                                   "decisions": len(decisions), "s": round(time.time() - tv, 1)})
        if saved:
            with torch.no_grad():
                for n, w in saved.items():
                    params[n].copy_(w)
        print(json.dumps(report["variants"][-1]), flush=True)
    np.savez(out_path, **out)
    Path(out_path).with_suffix(".json").write_text(json.dumps(report, indent=1, default=str))
    print("done", out_path, round(time.time() - t0, 1), "s", flush=True)


if __name__ == "__main__":
    main()
