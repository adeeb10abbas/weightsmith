"""Bounded two-configuration replay audit of the first sealed source group."""
import argparse
import json
import os
from pathlib import Path
import time

import numpy as np
import fresh_lora as F


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True)
    p.add_argument("--mode", choices=("preparation", "serving"), required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, required=True)
    args = p.parse_args()
    from weightsmith import config as C
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.mode == "serving":
        os.environ.update(JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE), JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
                          XLA_PYTHON_CLIENT_PREALLOCATE="true", XLA_PYTHON_CLIENT_MEM_FRACTION=C.SERVER_MEM_FRACTION)
    else:
        os.environ.pop("JAX_COMPILATION_CACHE_DIR", None)
        os.environ.pop("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", None)
        os.environ.pop("XLA_PYTHON_CLIENT_PREALLOCATE", None)
        os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".90"
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    spec, items, _, _ = F.read_spec(args.spec, "full_trajectory")
    first = items[0]
    members = first["live_batch_members"]
    data = [F.read_observation(m["obs_path"], m["decision_index"]) for m in members]
    requests = [{**F.raw_request(raw), "prompt": first["recorded_prompt"], "sampling_seed": seed} for raw, _, seed, _ in data]
    from weightsmith.locks import GpuLock
    start = time.monotonic()
    with GpuLock(args.gpu):
        server = F.module("diagnostic_fixed16_server", F.SERVER).Server()
        if args.mode == "serving":
            server.load(None)
            server.warmup()
        outputs = server.infer_batch(requests)
        repeated = server.infer_batch([requests[0]] * 16)[0]["actions"]
    replay = np.stack([np.asarray(x["actions"], dtype=np.float32) for x in outputs])
    recorded = np.stack([x[1] for x in data])
    errors = [F.N.physical_errors(a, b) for a, b in zip(replay, recorded, strict=True)]
    np.savez_compressed(out / "actions.npz", replay=replay, recorded=recorded, first_repeated=repeated)
    result = {"mode": args.mode, "source_spec": str(Path(args.spec).resolve()), "source_spec_sha256": F.sha(args.spec),
              "source_manifest_sha256": spec["source_manifest_sha256"], "prompt": first["recorded_prompt"],
              "source_step": first["source_step"], "kernel_batch": 16, "items": len(members),
              "maximum_error": max(e["max_abs_error"] for e in errors),
              "per_item": [{"env": m["env_id"], "seed": d[2], "step": d[3], **e,
                            "raw_dtypes": {k: str(v.dtype) for k, v in d[0].items()},
                            "input_digest": F.U.input_digest(d[0], d[2], first["recorded_prompt"])}
                           for m, d, e in zip(members, data, errors, strict=True)],
              "repeated_first": F.N.physical_errors(repeated, recorded[0]),
              "repeat_versus_full_group": F.N.physical_errors(repeated, replay[0]),
              "environment": {k: os.environ.get(k) for k in ("JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS",
                                "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS")},
              "checkpoint": str(C.CHECKPOINT), "wall_seconds": time.monotonic() - start,
              "actions_sha256": F.sha(out / "actions.npz"), "code_sha256": F.sha(__file__),
              "interpretation": "Diagnostic only; no training label accepted and no tolerance changed"}
    F.dump(out / "diagnostic.json", result)
    print(json.dumps({k: result[k] for k in ("mode", "maximum_error", "repeated_first", "repeat_versus_full_group", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
