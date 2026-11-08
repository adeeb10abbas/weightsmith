"""OpenPI-protocol websocket server for Cosmos3 Edge Policy DROID (cosmos-framework venv).

A request is one observation dict in RoboLab's Cosmos3Client wire format, or {"__batch__": [obs, ...]}. Each obs may
carry "sampling_seed"; the requests of a batch run one after another, so every answer is bit-identical to a single
request with the same inputs and seed (a slot's actions never depend on what runs beside it).
--edit adds a stored weight update (edge_common.apply_edit) before serving; the load report (digests of the edited
tensors) is printed on the line starting with "loaded ".
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
from edge_common import SEED, apply_edit, build_service, infer_seeded  # noqa: E402


class EdgePolicy:
    def __init__(self, svc, calls_log=None):
        self.svc, self.calls_log, self.served = svc, calls_log, 0

    def _one(self, obs):
        seed = obs.get("sampling_seed")
        seed = SEED if seed is None else int(seed)
        t = time.perf_counter()
        out = infer_seeded(self.svc, obs, seed)
        act = np.asarray(out["action"], dtype=np.float32)
        self.served += 1
        if self.calls_log:
            with open(self.calls_log, "a") as f:
                f.write(json.dumps({"t": round(time.time(), 3), "seed": seed,
                                    "infer_s": round(time.perf_counter() - t, 3), "prompt": obs.get("prompt"),
                                    "grip_mean": round(float(act[:, -1].mean()), 4)}) + "\n")
        return {"action": act, "sampling_seed": seed}

    def infer(self, obs):
        if "__batch__" in obs:
            return {"__batch__": [self._one(o) for o in obs["__batch__"]]}
        return self._one(obs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--calls-log", default=None)
    a = ap.parse_args()
    t0 = time.time()
    native, svc = build_service()
    report = {"edit": None}
    if a.edit:
        report["edit"] = apply_edit(svc.model, a.edit)
    report["load_s"] = round(time.time() - t0, 1)
    print("loaded " + json.dumps(report), flush=True)
    server_cls = native._load_openpi_websocket_policy_server()
    server_cls(policy=EdgePolicy(svc, a.calls_log), host="127.0.0.1", port=a.port, metadata={}).serve_forever()


if __name__ == "__main__":
    main()
