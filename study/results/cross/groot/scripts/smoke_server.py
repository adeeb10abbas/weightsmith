"""Server-side smoke test (Isaac-GR00T venv): load GR00T N1.7 DROID, check seeded noise and batching, time B=16."""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "adapters"))
import groot_server as gs  # noqa: E402

base, sim, _ = gs.load_policy()
m = base.model
print("model class", type(m).__name__)
tot = sum(p.numel() for p in m.parameters())
groups = {}
for n, p in m.named_parameters():
    k = ".".join(n.split(".")[:3])
    groups[k] = groups.get(k, 0) + p.numel()
print("params", tot)
for k, v in sorted(groups.items(), key=lambda kv: -kv[1])[:25]:
    print(f"  {k:60s} {v/1e6:9.2f}M")
for n, p in m.named_parameters():
    if "language_model.layers.0." in n or "action_head.model.transformer_blocks.0." in n:
        print("  ", n, tuple(p.shape), p.dtype)

rng = np.random.default_rng(0)


def obs(B, text="put the cube in the bowl"):
    return {
        "video.exterior_image_1_left": rng.integers(0, 255, (B, 1, 180, 320, 3), dtype=np.uint8),
        "video.wrist_image_left": rng.integers(0, 255, (B, 1, 180, 320, 3), dtype=np.uint8),
        "state.eef_9d": rng.normal(size=(B, 1, 9)).astype(np.float32) * 0.1,
        "state.joint_position": rng.normal(size=(B, 1, 7)).astype(np.float32) * 0.1,
        "state.gripper_position": np.zeros((B, 1, 1), np.float32),
        "annotation.language.language_instruction": [text] * B,
    }


pol = gs.SeededPolicy(sim, None)
o3 = obs(3)
a1, _ = pol.get_action(o3, {"seeds": [11, 22, 33]})
a2, _ = pol.get_action(o3, {"seeds": [11, 22, 33]})
print({k: v.shape for k, v in a1.items()})
print("same seeds same out:", all(np.array_equal(a1[k], a2[k]) for k in a1))
a3, _ = pol.get_action(o3, {"seeds": [11, 22, 34]})
print("diff seed changes only that sample:",
      float(np.abs(a1["action.joint_position"][:2] - a3["action.joint_position"][:2]).max()),
      float(np.abs(a1["action.joint_position"][2] - a3["action.joint_position"][2]).max()))
o1 = {k: (v[:1] if not isinstance(v, list) else v[:1]) for k, v in o3.items()}
b1, _ = pol.get_action(o1, {"seeds": [11]})
print("batch-1 vs batch-3 same sample, max abs diff:", float(np.abs(b1["action.joint_position"][0] - a1["action.joint_position"][0]).max()))
print("joint action range", float(a1["action.joint_position"].min()), float(a1["action.joint_position"].max()),
      "state", o3["state.joint_position"][0, 0][:3], "first row", a1["action.joint_position"][0, 0][:3])
print("gripper", a1["action.gripper_position"][0, :5, 0])
o16 = obs(16)
for i in range(4):
    torch.cuda.synchronize(); t = time.time()
    pol.get_action(o16, {"seeds": list(range(16))})
    torch.cuda.synchronize(); print(f"B=16 latency {time.time()-t:.3f}s")
print("max mem GB", torch.cuda.max_memory_allocated() / 1e9)
