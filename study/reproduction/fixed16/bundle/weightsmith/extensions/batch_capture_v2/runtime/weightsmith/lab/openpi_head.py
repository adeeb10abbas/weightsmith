# Vendored from the author's earlier project `weight-space-policy-lab`
# (src/weight_space_policy_lab/openpi_head.py), trimmed to the functions this harness uses.
# Logic is unchanged. See NOTICE.
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def _file_name(name: str) -> str:
    return hashlib.sha256(name.encode()).hexdigest()[:16] + ".npy"


def _parameter_count(checkpoint: Path) -> int:
    metadata = json.loads((checkpoint / "params" / "_METADATA").read_text())
    total = 0
    for entry in metadata["tree_metadata"].values():
        shape = entry["value_metadata"]["write_shape"]
        total += int(np.prod(shape, dtype=np.int64))
    return total


def _effective_bfloat16(array: np.ndarray) -> np.ndarray:
    # The public checkpoint is float32, but OpenPI explicitly restores it as
    # bfloat16 for inference. Export the effective values that the policy sees.
    value = np.asarray(array, dtype=np.float32)
    bits = value.view(np.uint32)
    rounding_bias = np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return ((bits + rounding_bias) & np.uint32(0xFFFF0000)).view(np.float32)
