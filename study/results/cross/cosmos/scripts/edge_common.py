"""Cosmos3 Edge Policy DROID: loader, seeded inference and weight edits (runs in the cosmos-framework venv).

Builds cosmos-framework's RobolabPolicyService (commit cf5d68c, the source the "Same Goal, Different Words" study
pinned for its E3 arm) with the study's E3 sampling settings (experiments/robolab_workshop/policy_server.py of
adeeb10abbas/world-action-model-steerability@9a6fcde): UniPC, 4 steps, shift 5, guidance 3 inside the interval
(960, 1001), 540x640 composite, joint positions + gripper (8 dims), 32-action chunks, history 1, prompt as JSON.
The one change is decode_video=False: the decoded future video is not used, and the actions do not depend on it.
Per-request sampling seeds: cfg.seed := seed, deterministic_seed := True (as the study's server).
"""
from __future__ import annotations

import hashlib
from dataclasses import replace
from pathlib import Path

EDGE_REPO = "nvidia/Cosmos3-Edge-Policy-DROID"
EDGE_REV = "a7c7288f9b6ac1684e993007b0f9703dd26e58ef"
SEED = 6100
E3_ARGS = dict(
    checkpoint_path=EDGE_REPO, hf_revision=EDGE_REV, host="127.0.0.1", port=0,
    domain_name="droid_lerobot", decode_video=False, seed=SEED, deterministic_seed=True,
    guidance=3.0, num_steps=4, shift=5.0, resolution="480", conditioning_fps=15.0,
    action_chunk_size=32, action_dim=8, history_length=1,
    sampler="unipc", guidance_interval=(960, 1001), image_height=540, image_width=640,
    action_space="joint_pos", use_state=True, format_prompt_as_json=True,
    cfg_parallel=False, allow_dcp_checkpoint=False, output_dir=None,
)


def build_service(**overrides):
    from cosmos_framework.scripts import action_policy_server_robolab as native
    args = native.RobolabServerArgs(**{**E3_ARGS, **overrides})
    return native, native.RobolabPolicyService(args)


def infer_seeded(service, obs: dict, seed: int) -> dict:
    service.cfg = replace(service.cfg, seed=int(seed), deterministic_seed=True)
    return service.infer(obs)


def tensor_digest(t) -> str:
    import torch
    a = t.detach().to("cpu").contiguous()
    if a.dtype == torch.bfloat16:
        a = a.view(torch.int16)
    return hashlib.sha256(a.numpy().tobytes()).hexdigest()[:16]


def canonical(name: str) -> str:
    """Parameter name without torch.compile's `_orig_mod.` segment."""
    return name.replace("._orig_mod", "")


def file_digest(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def apply_edit(model, path: str | Path) -> dict:
    """Add a stored weight update to the loaded model in place.

    The file is a torch .pt dict {param_name: {"A": [out, r], "B": [r, in]}} (low-rank factors) or
    {param_name: dense [out, in]}; W_new = bf16(W.double() + A @ B) in torch's Linear convention (y = x W^T).
    Returns a report with per-tensor digests before/after, rank, stored and changed values.
    """
    import torch
    delta = torch.load(str(path), map_location="cpu", weights_only=True)
    params = {canonical(n): p for n, p in model.named_parameters()}
    report = {"file": str(path), "sha256_16": file_digest(path), "params": {}}
    with torch.no_grad():
        for name, d in delta.items():
            name = canonical(name)
            if name not in params:
                raise KeyError(f"edit names unknown parameter {name}")
            p = params[name]
            if isinstance(d, dict):
                dw = d["A"].double() @ d["B"].double()
                stored, rank = d["A"].numel() + d["B"].numel(), int(d["A"].shape[1])
            else:
                dw, stored, rank = d.double(), d.numel(), None
            if tuple(dw.shape) != tuple(p.shape):
                raise ValueError(f"{name}: delta {tuple(dw.shape)} vs param {tuple(p.shape)}")
            before = tensor_digest(p)
            w0 = p.detach().double().cpu()
            new = (w0 + dw).to(p.dtype)
            changed = int((new != p.detach().cpu()).sum())
            p.copy_(new.to(p.device))
            report["params"][name] = {"before": before, "after": tensor_digest(p), "rank": rank,
                                      "stored_values": stored, "changed_values": changed,
                                      "rel_fro": float(dw.norm() / w0.norm())}
    report["stored_values"] = sum(v["stored_values"] for v in report["params"].values())
    report["changed_values"] = sum(v["changed_values"] for v in report["params"].values())
    return report


def _find_vae_decoders(obj, path="model", depth=0, seen=None):
    """(path, module) of every `decoder` that sits beside an `encoder` (the video VAE), searching registered
    submodules and plain attributes (the VAE wrappers hang off the model as plain attributes)."""
    import torch
    seen = set() if seen is None else seen
    if id(obj) in seen or depth > 6:
        return []
    seen.add(id(obj))
    if isinstance(obj, torch.nn.Module) and isinstance(getattr(obj, "decoder", None), torch.nn.Module) \
            and hasattr(obj, "encoder"):
        return [(path + ".decoder", obj.decoder)]
    kids = []
    if isinstance(obj, torch.nn.Module):
        if path.endswith(".net"):  # the transformer holds no VAE; skip its thousands of submodules
            return []
        kids += [(k, v) for k, v in obj._modules.items() if v is not None]
    if hasattr(obj, "__dict__"):
        kids += [(k, v) for k, v in vars(obj).items() if not k.startswith("_")
                 and (isinstance(v, torch.nn.Module) or type(v).__module__.startswith("cosmos"))]
    out = []
    for k, v in kids:
        out += _find_vae_decoders(v, f"{path}.{k}", depth + 1, seen)
    return out


class CPUEmbedding:
    """Wraps the token embedding so its table stays on the CPU (a gather: values are bit-identical)."""

    def __new__(cls, emb):
        import torch

        class _E(torch.nn.Module):
            def __init__(self, e):
                super().__init__()
                self.cpu_emb = e.to("cpu")

            @property
            def weight(self):
                return self.cpu_emb.weight

            def forward(self, ids):
                return self.cpu_emb(ids.to("cpu")).to(ids.device)

        return _E(emb)


def slim(service) -> dict:
    """Free GPU memory that action generation never touches (actions stay bit-identical; mem_probe.py checks it).

    - the language-model head (131072 x 2048, 0.5 GB): only text prediction reads it;
    - the token-embedding table (0.5 GB): kept on the CPU, looked up there (a gather, so values are identical);
    - the video VAE decoder: only decoding the predicted future video uses it (decode_video=False here);
    - the cached allocator blocks left over from loading.
    """
    import torch
    rep = {}
    lm = service.model.net.language_model
    head = getattr(lm, "lm_head", None)
    if head is not None and head.weight.is_cuda:
        rep["lm_head_gb"] = round(head.weight.numel() * head.weight.element_size() / 2**30, 3)
        head.to("cpu")
    emb = lm.model.embed_tokens
    if emb.weight.is_cuda:
        rep["embed_gb"] = round(emb.weight.numel() * emb.weight.element_size() / 2**30, 3)
        lm.model.embed_tokens = CPUEmbedding(emb)
    rep["vae_decoders"] = {}
    for path, d in _find_vae_decoders(service.model):
        b = sum(t.numel() * t.element_size() for t in list(d.parameters()) + list(d.buffers()) if t.is_cuda)
        rep["vae_decoders"][path] = round(b / 2**30, 3)
        d.to("cpu")
    torch.cuda.empty_cache()
    rep["alloc_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
    rep["reserved_gb"] = round(torch.cuda.memory_reserved() / 2**30, 3)
    return rep
