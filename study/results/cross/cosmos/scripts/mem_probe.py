"""Where the policy server's GPU memory goes, and what can be dropped without changing any action (cosmos venv).

Loads the service, lists the GPU bytes held by each top-level module of the model (parameters and buffers), then
applies edge_common.slim() and reports memory again, and checks that a seeded request returns bit-identical actions
before and after slimming.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from edge_common import build_service, infer_seeded, slim  # noqa: E402


def gb(x):
    return round(x / 2**30, 3)


def module_bytes(m):
    tot = 0
    for t in list(m.parameters(recurse=True)) + list(m.buffers(recurse=True)):
        if t.is_cuda:
            tot += t.numel() * t.element_size()
    return tot


def main():
    t0 = time.time()
    native, svc = build_service()
    torch.cuda.synchronize()
    print("LOAD_S", round(time.time() - t0, 1), "alloc", gb(torch.cuda.memory_allocated()), "reserved",
          gb(torch.cuda.memory_reserved()), flush=True)
    rows = []
    for name, child in svc.model.named_children():
        rows.append((name, type(child).__name__, gb(module_bytes(child))))
        if name == "net":
            for n2, c2 in child.named_children():
                rows.append(("net." + n2, type(c2).__name__, gb(module_bytes(c2))))
                if n2 == "language_model":
                    for n3, c3 in c2.named_children():
                        rows.append(("net.language_model." + n3, type(c3).__name__, gb(module_bytes(c3))))
    for attr in dir(svc.model):
        if attr.startswith("__"):
            continue
        try:
            v = getattr(svc.model, attr)
        except Exception:
            continue
        if isinstance(v, torch.nn.Module) and not any(r[0] == attr for r in rows):
            rows.append(("attr:" + attr, type(v).__name__, gb(module_bytes(v))))
    for r in rows:
        print("MOD", *r)
    rng = np.random.default_rng(0)
    obs = {"observation/image": rng.integers(0, 255, (540, 640, 3), dtype=np.uint8),
           "observation/joint_position": np.array([0, -0.6, 0, -2.2, 0, 1.6, 0.8]),
           "observation/gripper_position": np.zeros(1), "prompt": "Put the cube in the bowl"}
    a0 = infer_seeded(svc, obs, 777)["action"]
    torch.cuda.synchronize()
    print("BEFORE_SLIM alloc", gb(torch.cuda.memory_allocated()), "reserved", gb(torch.cuda.memory_reserved()),
          "peak", gb(torch.cuda.max_memory_allocated()), flush=True)
    rep = slim(svc)
    print("SLIM", json.dumps(rep))
    torch.cuda.reset_peak_memory_stats()
    a1 = infer_seeded(svc, obs, 777)["action"]
    torch.cuda.synchronize()
    print("AFTER_SLIM alloc", gb(torch.cuda.memory_allocated()), "reserved", gb(torch.cuda.memory_reserved()),
          "peak", gb(torch.cuda.max_memory_allocated()), flush=True)
    print("IDENTICAL_ACTIONS", bool(np.array_equal(a0, a1)))
    import subprocess
    print(subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout)


if __name__ == "__main__":
    main()
