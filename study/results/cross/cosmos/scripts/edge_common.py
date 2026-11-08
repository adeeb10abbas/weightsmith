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
