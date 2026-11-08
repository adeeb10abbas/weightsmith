"""Probe Cosmos3 Edge internals for the capture design (cosmos venv, one GPU, TORCHDYNAMO_DISABLE=1).

Prints: the network's layer count and module names of one decoder layer; the keys and token ids of the prepared
sample for two wordings (to align positions); and, for one seeded request, every call of each layer's und (text)
and gen (video/action) MLP down-projection with its token count, so the capture knows how many forward passes a
request makes (denoising steps x guidance branches) and which carry text.
usage: probe.py OBS_NPZ PROMPT_A PROMPT_B
"""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from edge_common import build_service, infer_seeded  # noqa: E402


def layers_of(svc):
    lm = svc.model.net.language_model.model
    return [getattr(l, "_orig_mod", l) for l in lm.layers]


def main():
    obs_path, pa, pb = sys.argv[1], sys.argv[2], sys.argv[3]
    t0 = time.time()
    native, svc = build_service()
    print("LOAD_S", round(time.time() - t0, 1), flush=True)
    layers = layers_of(svc)
    print("N_LAYERS", len(layers))
    print("LAYER0_MODULES", [n for n, _ in layers[0].named_modules()][:60])
    if obs_path == "synthetic":
        rng = np.random.default_rng(0)
        obs = {"observation/image": rng.integers(0, 255, (540, 640, 3), dtype=np.uint8),
               "observation/joint_position": np.array([0, -0.6, 0, -2.2, 0, 1.6, 0.8]),
               "observation/gripper_position": np.zeros(1)}
    else:
        d = np.load(obs_path)
        obs = {"observation/image": d["image"][0], "observation/joint_position": d["joint"][0].astype(np.float64),
               "observation/gripper_position": d["gripper"][0].astype(np.float64)}
    for name, p in (("A", pa), ("B", pb)):
        s = svc._build_sample(dict(obs, prompt=p))
        print("SAMPLE", name, {k: (tuple(v.shape) if hasattr(v, "shape") else type(v).__name__) for k, v in s.items()})
        for k, v in s.items():
            if isinstance(v, torch.Tensor) and v.dtype in (torch.int32, torch.int64) and v.numel() > 8:
                print("IDS", name, k, v.reshape(-1).tolist())
        print("CAPTION", name, json.dumps(s.get("ai_caption"))[:2000])
    tok = None
    for attr in ("text_tokenizer", "tokenizer", "_tokenizer"):
        for obj in (svc.model, getattr(svc.model, "net", None)):
            if obj is not None and hasattr(obj, attr):
                tok = getattr(obj, attr)
                print("TOKENIZER_ATTR", type(obj).__name__, attr, type(tok).__name__)
                break
        if tok is not None:
            break
    calls = defaultdict(list)
    hooks = []
    for L, layer in enumerate(layers):
        def mk(L, kind):
            def h(mod, inp, out):
                calls[(L, kind)].append(int(inp[0].shape[0]))
            return h
        hooks.append(layer.mlp.down_proj.register_forward_hook(mk(L, "und")))
        hooks.append(layer.mlp_moe_gen.down_proj.register_forward_hook(mk(L, "gen")))
    for name, p in (("A", pa), ("B", pb)):
        calls.clear()
        t = time.time()
        out = infer_seeded(svc, dict(obs, prompt=p), 12345)
        print("REQ", name, round(time.time() - t, 2), "s; action[0]", np.round(out["action"][0], 4).tolist())
        print("CALLS", name, json.dumps({f"{L}:{k}": v for (L, k), v in sorted(calls.items()) if L in (0, 1, len(layers) - 1)}))
        out2 = infer_seeded(svc, dict(obs, prompt=p), 12345)
        print("DETERMINISTIC", name, bool(np.array_equal(out["action"], out2["action"])))
    for h in hooks:
        h.remove()
    print("PEAK_GB", round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == "__main__":
    main()
