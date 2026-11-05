# Vendored from the author's earlier project `weight-space-policy-lab`
# (src/weight_space_policy_lab/sandbox.py), trimmed to the functions this harness uses.
# Logic is unchanged. See NOTICE.
"""Helpers injected in front of every `owsp py` script inside the bwrap sandbox."""

PRELUDE = r'''
import json
import os
import numpy as np

np.set_printoptions(precision=5, suppress=True, linewidth=160)
WORK = "/work"
CONFIG = json.load(open(WORK + "/workspace.json"))
TENSORS = CONFIG["tensors"]
READABLE_NAMES = sorted(TENSORS)
NAMES = sorted(name for name, entry in TENSORS.items() if entry.get("writable", True))
REFERENCES = sorted(set(READABLE_NAMES) - set(NAMES))

def _path(folder, name):
    if name not in TENSORS:
        raise KeyError(f"Unknown tensor {name!r}. Readable names: {READABLE_NAMES}")
    return f"{WORK}/{folder}/{TENSORS[name]['file']}"

def load(name):
    edit_path = _path("edits", name)
    path = edit_path if os.path.exists(edit_path) else _path("base", name)
    return np.load(path, allow_pickle=False).copy()

def base(name):
    return np.load(_path("base", name), allow_pickle=False).copy()

load_original = base

def save(name, value, _base_loader=base):
    if name not in NAMES:
        raise KeyError(f"Tensor {name!r} is read-only. Writable names: {NAMES}")
    value = np.asarray(value)
    original = _base_loader(name)
    if value.shape != original.shape:
        raise ValueError((name, value.shape, original.shape))
    if not np.isfinite(value).all():
        raise ValueError(f"Non-finite values in {name}")
    delta = np.max(np.abs(value.astype(np.float64) - original.astype(np.float64))) if value.size else 0.0
    max_abs_delta = CONFIG.get("max_abs_delta")
    if max_abs_delta is not None and delta > max_abs_delta:
        raise ValueError(f"max abs delta {delta} exceeds {max_abs_delta}")
    os.makedirs(WORK + "/edits", exist_ok=True)
    np.save(_path("edits", name), value.astype(original.dtype), allow_pickle=False)

save_edit = save

def drop_edit(name):
    if name not in NAMES:
        raise KeyError(f"Tensor {name!r} is not writable")
    path = _path("edits", name)
    if os.path.exists(path):
        os.remove(path)

def stats(name, _loader=load, _base_loader=base):
    value = _loader(name).astype(np.float64)
    original = _base_loader(name).astype(np.float64)
    delta = value - original
    return {
        "shape": list(value.shape),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "l2": float(np.linalg.norm(value.ravel())),
        "changed": int(np.count_nonzero(delta)),
        "max_abs_delta": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "l2_delta": float(np.linalg.norm(delta.ravel())),
    }

def saved():
    return [name for name in NAMES if os.path.exists(_path("edits", name))]

edited = saved

def bf16(value):
    """Return the float32 values produced by round-to-nearest-even bfloat16."""
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)
'''
