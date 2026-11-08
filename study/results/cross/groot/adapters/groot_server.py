"""GR00T N1.7 DROID policy server for RoboLab, with paired-seed noise and closed-form weight edits.

Runs in the Isaac-GR00T venv (python 3.12). It is the validated RoboLab path (Gr00tPolicy wrapped in
Gr00tSimPolicyWrapper, embodiment OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT, flat ``video.*``/``state.*`` keys) with
three additions:

1. Paired-seed noise. A request may carry ``options = {"seeds": [s_0, ..., s_{B-1}]}``. The flow-matching
   initial noise of sample i is then drawn from ``torch.Generator('cpu').manual_seed(s_i)`` in float32 and cast to
   the model dtype, so a slot always receives the same noise whichever other envs share the batch (as in
   weightsmith's pi0.5 runtime). Without seeds the stock ``torch.randn`` is used.
2. Weight edits. ``--edit FILE`` (a ``torch.save`` dict) is applied once at load:
   ``{"format": "wsp-groot-edit-v1", "layers": {param_name: {"U": [d_in, r], "V": [r, d_out]}}}``
   adds the low-rank update ``(U @ V)`` in float32 to the weight in the ``y = h @ W`` convention
   (``nn.Linear`` stores ``W.T``) and rounds once to the stored dtype (bf16):
   ``W_new = bf16(W0 + dW)``. ``{"replace": {param_name: tensor}}`` writes exact values instead.
   The sha256 of the file and the number of stored values that changed are printed and served on ``info``.
3. ``info`` endpoint: checkpoint, edit fingerprint, values changed; ``get_action`` logs batch size and latency.
"""
from __future__ import annotations

import argparse
import contextvars
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import torch

_SEEDS: contextvars.ContextVar = contextvars.ContextVar("wsp_seeds", default=None)

DEFAULT_CKPT = str(Path.home() / ".cache/huggingface/hub/models--nvidia--GR00T-N1.7-DROID/snapshots/"
                   "05e7cc97e40dbd33b0890c35cc0214fcb0547ab5")
EMBODIMENT = "OXE_DROID_RELATIVE_EEF_RELATIVE_JOINT"


class _SeededTorch(types.ModuleType):
    """Stand-in for the ``torch`` global of the N1.7 action-head module: ``randn`` honours per-sample seeds."""

    def __init__(self, real):
        super().__init__("torch")
        self.__dict__["_real"] = real

    def __getattr__(self, name):
        return getattr(self.__dict__["_real"], name)

    def randn(self, *args, **kw):
        real = self.__dict__["_real"]
        seeds = _SEEDS.get()
        if seeds is None:
            return real.randn(*args, **kw)
        size = kw.pop("size", None)
        if size is None:
            size = args[0] if len(args) == 1 and isinstance(args[0], (tuple, list, torch.Size)) else args
        size = tuple(int(x) for x in size)
        dtype, device = kw.pop("dtype", None), kw.pop("device", None)
        if size[0] != len(seeds):
            raise ValueError(f"seeded randn: batch {size[0]} != {len(seeds)} seeds")
        out = []
        for s in seeds:
            g = torch.Generator(device="cpu")
            g.manual_seed(int(s))
            out.append(real.randn(size[1:], generator=g, dtype=torch.float32))
        return torch.stack(out).to(device=device, dtype=dtype or torch.get_default_dtype())


def install_seeded_noise():
    import gr00t.model.gr00t_n1d7.gr00t_n1d7 as m
    if not isinstance(m.torch, _SeededTorch):
        m.torch = _SeededTorch(torch)


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@torch.no_grad()
def apply_edit(model: torch.nn.Module, path: str | Path) -> dict:
    """Apply a wsp-groot edit file to ``model`` in place; returns a report."""
    ed = torch.load(path, map_location="cpu", weights_only=False)
    params = dict(model.named_parameters())
    report = {"edit": str(path), "sha256": sha256_file(path), "fingerprint": sha256_file(path)[:16],
              "tensors": 0, "values_changed": 0, "values_total": 0, "per_tensor": {}}
    for name, spec in (ed.get("layers") or {}).items():
        p = params[name]
        U = spec["U"].to(device=p.device, dtype=torch.float32)
        V = spec["V"].to(device=p.device, dtype=torch.float32)
        dW = U @ V  # [d_in, d_out], y = h @ W convention
        if p.ndim != 2 or tuple(p.shape) != (dW.shape[1], dW.shape[0]):
            raise ValueError(f"{name}: weight {tuple(p.shape)} vs update {tuple(dW.shape)} (expects nn.Linear [out, in])")
        old = p.data.clone()
        p.data.copy_((p.data.float() + dW.T).to(p.dtype))
        ch = int((p.data != old).sum().item())
        report["per_tensor"][name] = {"rank": int(U.shape[1]), "changed": ch, "numel": p.numel(),
                                      "rel_fro": float(dW.norm() / old.float().norm())}
        report["values_changed"] += ch
        report["values_total"] += p.numel()
        report["tensors"] += 1
    for name, val in (ed.get("replace") or {}).items():
        p = params[name]
        old = p.data.clone()
        p.data.copy_(val.to(device=p.device, dtype=p.dtype))
        ch = int((p.data != old).sum().item())
        report["per_tensor"][name] = {"replace": True, "changed": ch, "numel": p.numel()}
        report["values_changed"] += ch
        report["values_total"] += p.numel()
        report["tensors"] += 1
    report["meta"] = ed.get("meta", {})
    return report


def load_policy(ckpt: str = DEFAULT_CKPT, device: str = "cuda", edit: str | None = None):
    from gr00t.policy.gr00t_policy import Gr00tPolicy, Gr00tSimPolicyWrapper
    install_seeded_noise()
    base = Gr00tPolicy(embodiment_tag=EMBODIMENT, model_path=ckpt, device=device, strict=True)
    report = None
    if edit:
        report = apply_edit(base.model, edit)
    return base, Gr00tSimPolicyWrapper(base), report


class SeededPolicy:
    """Wraps the sim policy: pops ``seeds`` from options into the noise context, logs calls."""

    def __init__(self, sim_policy, log_path: str | None):
        self.p = sim_policy
        self.log = open(log_path, "a") if log_path else None

    def get_action(self, observation, options=None):
        seeds = (options or {}).get("seeds")
        t0 = time.time()
        tok = _SEEDS.set(list(seeds) if seeds is not None else None)
        try:
            act, info = self.p.get_action(observation, None)
        finally:
            _SEEDS.reset(tok)
        dt = time.time() - t0
        if self.log:
            b = len(observation["annotation.language.language_instruction"])
            self.log.write(json.dumps({"t": round(t0, 3), "b": b, "s": round(dt, 4),
                                       "seed0": (seeds[0] if seeds else None)}) + "\n")
            self.log.flush()
        return act, info

    def reset(self, options=None):
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--port", type=int, default=8600)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--edit", default=None)
    ap.add_argument("--calls-log", default=None)
    a = ap.parse_args()
    from gr00t.policy.server_client import PolicyServer
    base, sim, report = load_policy(a.ckpt, "cuda", a.edit)
    n_params = sum(p.numel() for p in base.model.parameters())
    info = {"ckpt": a.ckpt, "embodiment": EMBODIMENT, "params": n_params, "edit": report and {
        k: v for k, v in report.items() if k != "per_tensor"}}
    print("loaded " + json.dumps(info), flush=True)
    if report:
        print("edit " + json.dumps(report), flush=True)
    pol = SeededPolicy(sim, a.calls_log)
    server = PolicyServer(policy=pol, host=a.host, port=a.port)
    server.register_endpoint("info", lambda: info, requires_input=False)
    print(f"Server is ready and listening on tcp://{a.host}:{a.port}", flush=True)
    try:
        server.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    sys.exit(main())
