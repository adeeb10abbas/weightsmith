#!/usr/bin/env python3
"""Write `owsp acts` specs for the sgdw distillation (stdlib only).

  make_specs.py diverge PAIRS.json OUTDIR --rf RF --tf TF [--episodes N] [--kmax K] [--kstep S]
      action-only replays of the first N RF failures whose TF twin succeeded, replans 0..K step S,
      once under each wording (ids div-rf / div-tf)
  make_specs.py full ITEMS.json OUTDIR --rf RF --tf TF --name NAME [--layers 7-16]
      image-token captures of the chosen items under each wording (ids NAME-o / NAME-x), the inputs of
      weightsmith.distill's build step (VLM MLP hidden + residual at 2 cams x 17 cells)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def light(items, prompt, aid):
    return {"id": aid, "cand": "base", "prompt": prompt, "items": items, "vlm_layers": [], "action_layers": [17],
            "action_tokens": [0], "denoise_steps": [0]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["diverge", "full"])
    ap.add_argument("src")
    ap.add_argument("outdir")
    ap.add_argument("--rf", required=True)
    ap.add_argument("--tf", required=True)
    ap.add_argument("--episodes", type=int, default=4)
    ap.add_argument("--kmax", type=int, default=40)
    ap.add_argument("--kstep", type=int, default=2)
    ap.add_argument("--name", default="cap")
    ap.add_argument("--layers", default="7-16")
    a = ap.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    src = json.loads(Path(a.src).read_text())
    if a.mode == "diverge":
        pairs = [p for p in src["pairs"] if (not p["rf_ok"]) and p["tf_ok"]][: a.episodes]
        items = [{"episode": p["rf"], "k": k} for p in pairs for k in range(0, min(a.kmax, p["rf_K"] - 1) + 1, a.kstep)]
        for aid, prompt in (("div-rf", a.rf), ("div-tf", a.tf)):
            (out / f"{aid}.json").write_text(json.dumps(light(items, prompt, aid), indent=1))
        print(f"{len(items)} items from {len(pairs)} episodes: {[p['rf'] for p in pairs]}")
    else:
        items = src["items"] if isinstance(src, dict) else src
        lo, hi = (int(x) for x in a.layers.split("-"))
        common = {"items": items, "vlm_layers": list(range(lo, hi + 1)), "image_tokens": True, "prompt_tokens": [0],
                  "action_layers": [17], "action_tokens": [0], "denoise_steps": [0]}
        for suffix, prompt in (("o", a.rf), ("x", a.tf)):
            spec = {"id": f"{a.name}-{suffix}", "cand": "base", "prompt": prompt, **common}
            (out / f"{a.name}-{suffix}.json").write_text(json.dumps(spec, indent=1))
        print(f"{len(items)} items, VLM layers {lo}-{hi}")


if __name__ == "__main__":
    main()
