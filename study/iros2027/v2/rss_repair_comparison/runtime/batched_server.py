"""Held-out cohort server: every inference uses the same 16-row kernel shape.

Development used the historical server's dynamic power-of-two buckets. BF16
outputs can change when a shrinking active set selects a different kernel shape.
This separately recorded inference control pads the request list to 16, keeping
real requests first and their sampling seeds intact. Development and held-out
results must retain their distinct inference-regime labels.
"""
from __future__ import annotations

import copy
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
from weightsmith import batched_server as base

KERNEL_BATCH = 16


class Server(base.Server):
    def infer_batch(self, obs_list):
        requested_n = len(obs_list)
        if not 1 <= requested_n <= KERNEL_BATCH:
            raise ValueError(f"Fixed-16 inference requires 1..16 requests; received {requested_n}")
        # Isolate callers and padding rows even if an input transform mutates data.
        padded = [copy.deepcopy(obs) for obs in obs_list]
        padded.extend(copy.deepcopy(obs_list[0]) for _ in range(KERNEL_BATCH - requested_n))
        outputs = super().infer_batch(padded)
        if len(outputs) != KERNEL_BATCH:
            raise RuntimeError(f"Fixed-16 parent returned {len(outputs)} outputs instead of 16")
        kept = []
        for output in outputs[:requested_n]:
            result = dict(output)
            result["policy_timing"] = dict(output.get("policy_timing", {}))
            result["policy_timing"].update(batch=requested_n, requested_n=requested_n,
                                           kernel_batch=KERNEL_BATCH)
            kept.append(result)
        return kept


def main():
    # Process-local substitution: reuse loading, serving and warmup implementation.
    base.Server = Server
    base.main()


if __name__ == "__main__":
    main()
