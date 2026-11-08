"""Replay recorded requests through this host's server code path and compare with the actions recorded on the host
that served them (cosmos venv; run with CUDA_VISIBLE_DEVICES set, compiled as the server unless TORCHDYNAMO_DISABLE=1).

usage: replay_check.py PROMPT OUT.json OBS.npz:K [OBS.npz:K ...]
Reports per request the max |difference| of the 32x8 chunk, the chunk-end end-effector distance (cm), gripper
agreement, and whether the chunk is bit-identical.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from edge_common import build_service, infer_seeded, slim  # noqa: E402
from offline_check import panda_fk  # noqa: E402


def main():
    prompt, out = sys.argv[1], sys.argv[2]
    t0 = time.time()
    native, svc = build_service()
    rep = {"slim": slim(svc), "load_s": round(time.time() - t0, 1), "prompt": prompt, "requests": []}
    for spec in sys.argv[3:]:
        path, k = spec.rsplit(":", 1)
        d = np.load(path)
        k = int(k)
        obs = {"observation/image": d["image"][k], "observation/joint_position": d["joint"][k].astype(np.float64),
               "observation/gripper_position": d["gripper"][k].astype(np.float64), "prompt": prompt}
        a = np.asarray(infer_seeded(svc, obs, int(d["seed"][k]))["action"], np.float32)
        b = d["actions"][k].astype(np.float32)
        r = {"request": spec, "seed": int(d["seed"][k]), "bit_identical": bool(np.array_equal(a, b)),
             "max_abs": float(np.abs(a - b).max()),
             "ee_end_cm": float(100 * np.linalg.norm(panda_fk(a[-1, :7]) - panda_fk(b[-1, :7]))),
             "grip_agree": float(((a[:, 7] > 0.5) == (b[:, 7] > 0.5)).mean())}
        rep["requests"].append(r)
        print(json.dumps(r), flush=True)
    rep["all_bit_identical"] = all(r["bit_identical"] for r in rep["requests"])
    Path(out).write_text(json.dumps(rep, indent=1))
    print(json.dumps({k: v for k, v in rep.items() if k != "requests"}), flush=True)


if __name__ == "__main__":
    main()
