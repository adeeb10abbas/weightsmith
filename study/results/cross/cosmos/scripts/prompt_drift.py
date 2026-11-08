"""How much does an edit move the text states of other prompts? (cosmos venv, one GPU, eager)

Text states depend on the prompt only (seq_build.py checks it), so one recorded decision serves every prompt. For each
prompt: the conditional-pass text residual after every layer, unedited and edited; reports the relative change
||edited - unedited|| / ||unedited|| per layer (all positions), its mean over layers, and the change of the
32x8 action chunk on that decision (an action change on a foreign image is only indicative).

usage: prompt_drift.py EDIT.pt OBS.npz:K OUT.json NAME=PROMPT [NAME=PROMPT ...]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np  # noqa: E402
import torch  # noqa: E402
from capture import decoder_layers, load_decision  # noqa: E402
from edge_common import apply_edit, build_service, canonical  # noqa: E402
from seq_build import TextRec  # noqa: E402


def main():
    edit_path, dec, out = sys.argv[1], sys.argv[2], sys.argv[3]
    prompts = [a.split("=", 1) for a in sys.argv[4:]]
    t0 = time.time()
    native, svc = build_service()
    rec = TextRec(svc.model.net, decoder_layers(svc))
    p, k = dec.rsplit(":", 1)
    obs, seed = load_decision(p, int(k))
    nl = len(decoder_layers(svc))
    base = {}
    for name, pr in prompts:
        (ids, c), _, act = rec.run(svc, obs, seed, pr)
        base[name] = (len(ids), {L: c[("res", L)] for L in range(nl) if ("res", L) in c}, act)
    params = {canonical(n): q for n, q in svc.model.named_parameters()}
    delta = torch.load(edit_path, map_location="cpu", weights_only=True)
    keep = {canonical(n): params[canonical(n)].detach().clone() for n in delta}
    rep = {"edit": edit_path, "apply": apply_edit(svc.model, edit_path), "decision": dec, "prompts": {}}
    rep["apply"].pop("params", None)
    for name, pr in prompts:
        (ids, c), _, act = rec.run(svc, obs, seed, pr)
        n, r0, a0 = base[name]
        rel = [float(np.linalg.norm(c[("res", L)] - r0[L]) / max(np.linalg.norm(r0[L]), 1e-30)) for L in sorted(r0)]
        rep["prompts"][name] = {"prompt": pr, "tokens": n, "rel_change_per_layer": rel,
                                "rel_change_mean": float(np.mean(rel)), "rel_change_max": float(np.max(rel)),
                                "action_max_abs_change": float(np.abs(act - a0).max())}
        print(name, n, round(float(np.mean(rel)), 4), round(float(np.max(rel)), 4),
              round(float(np.abs(act - a0).max()), 4), flush=True)
    with torch.no_grad():
        for n_, w in keep.items():
            params[n_].copy_(w)
    rep["s"] = round(time.time() - t0, 1)
    Path(out).write_text(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
