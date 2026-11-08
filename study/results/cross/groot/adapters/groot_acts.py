"""Activation capture for GR00T N1.7 DROID on recorded decisions (Isaac-GR00T venv, GPU).

Replays recorded requests (``obs_r0_e{env}.npz`` written by groot_batched_client.py: exact 180x320 images, state and
noise seed of every replan) under one or more wordings, with the noise seed fixed, and records:

* ``input_ids``: the backbone token ids (Qwen3-VL chat template: images first, then the lowercased instruction);
* per Cosmos-Reason2 LLM layer L = 0..15: the input of ``mlp.down_proj`` (6144-d, the keys of a closed-form edit)
  and the layer's residual increment (output - input, 2048-d; deepstack injections excluded);
* the action head's view (``--head``): per DiT block, the input of ``ff.net.2`` (the MLP down-projection) and the
  block's residual increment at every action/state token for each of the 4 denoising steps, and the
  vl_self_attention blocks' MLP inputs/increments over the VL tokens;
* the decoded action chunk (absolute joints, gripper, eef) and the normalised prediction.

Items: JSON list of {"obs": path, "k": replan index, "tag": optional}. Output: one ``.npz`` per (item, wording)
in ``--out`` plus ``index.json``. ``--edit`` applies a wsp-groot edit first (offline checks).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import groot_server as gs  # noqa: E402


class Recorder:
    def __init__(self, model, head: bool = False, layers=None):
        self.model = model
        self.h = []
        self.rec = {}
        lm = model.backbone.model.model.language_model
        self.n_layers = len(lm.layers)
        self.layers = list(range(self.n_layers)) if layers is None else list(layers)
        for L in self.layers:
            layer = lm.layers[L]
            self.h.append(layer.register_forward_pre_hook(self._pre(f"lm_in.{L}"), with_kwargs=True))
            self.h.append(layer.register_forward_hook(self._post(f"lm_out.{L}")))
            self.h.append(layer.mlp.down_proj.register_forward_pre_hook(self._inp(f"lm_mlp.{L}")))
        self.h.append(model.backbone.register_forward_pre_hook(self._ids))
        if head:
            ah = model.action_head
            for i, blk in enumerate(ah.model.transformer_blocks):
                self.h.append(blk.register_forward_pre_hook(self._pre(f"dit_in.{i}", step=True), with_kwargs=True))
                self.h.append(blk.register_forward_hook(self._post(f"dit_out.{i}", step=True)))
                self.h.append(blk.ff.net[2].register_forward_pre_hook(self._inp(f"dit_mlp.{i}", step=True)))
            vsa = ah.vl_self_attention
            for i, blk in enumerate(getattr(vsa, "transformer_blocks", [])):
                self.h.append(blk.register_forward_pre_hook(self._pre(f"vsa_in.{i}"), with_kwargs=True))
                self.h.append(blk.register_forward_hook(self._post(f"vsa_out.{i}")))
                self.h.append(blk.ff.net[2].register_forward_pre_hook(self._inp(f"vsa_mlp.{i}")))

    def _put(self, key, x, step):
        x = x.detach().float().cpu()
        if step:
            self.rec.setdefault(key, []).append(x)
        else:
            self.rec[key] = x

    def _pre(self, key, step=False):
        def f(mod, args, kwargs):
            x = args[0] if args else kwargs.get("hidden_states")
            self._put(key, x, step)
        return f

    def _post(self, key, step=False):
        def f(mod, args, out):
            x = out[0] if isinstance(out, (tuple, list)) else out
            self._put(key, x, step)
        return f

    def _inp(self, key, step=False):
        def f(mod, args):
            self._put(key, args[0], step)
        return f

    def _ids(self, mod, args):
        vl = args[0]
        self.rec["input_ids"] = vl["input_ids"].detach().cpu()
        self.rec["attention_mask"] = vl["attention_mask"].detach().cpu()

    def clear(self):
        self.rec = {}

    def remove(self):
        for h in self.h:
            h.remove()


def request(npz, k, text):
    d = np.load(npz)
    return {
        "video.exterior_image_1_left": d["ext_img"][k][None, None],
        "video.wrist_image_left": d["wrist_img"][k][None, None],
        "state.eef_9d": d["eef_9d"][k][None, None].astype(np.float32),
        "state.joint_position": d["joint"][k][None, None].astype(np.float32),
        "state.gripper_position": d["gripper"][k].reshape(1, 1, -1).astype(np.float32),
        "annotation.language.language_instruction": [text],
    }, int(d["seed"][k])


def pack(rec, act, head):
    o = {"input_ids": rec["input_ids"][0].numpy().astype(np.int64)}
    Ls = sorted(int(k.split(".")[1]) for k in rec if k.startswith("lm_mlp."))
    o["lm_layers"] = np.array(Ls)
    o["lm_mlp"] = np.stack([rec[f"lm_mlp.{L}"][0].numpy() for L in Ls]).astype(np.float16)            # [L, T, 6144]
    o["lm_inc"] = np.stack([(rec[f"lm_out.{L}"][0] - rec[f"lm_in.{L}"][0]).numpy() for L in Ls]).astype(np.float32)
    o["lm_out"] = np.stack([rec[f"lm_out.{L}"][0].numpy() for L in Ls]).astype(np.float16)
    if head:
        nb = len([k for k in rec if k.startswith("dit_mlp.")])
        o["dit_mlp"] = np.stack([torch.stack(rec[f"dit_mlp.{i}"])[:, 0].numpy() for i in range(nb)]).astype(np.float16)  # [B, S, T, 6144]
        o["dit_inc"] = np.stack([(torch.stack(rec[f"dit_out.{i}"]) - torch.stack(rec[f"dit_in.{i}"]))[:, 0].numpy()
                                 for i in range(nb)]).astype(np.float32)
        nv = len([k for k in rec if k.startswith("vsa_mlp.")])
        if nv:
            o["vsa_mlp"] = np.stack([rec[f"vsa_mlp.{i}"][0].numpy() for i in range(nv)]).astype(np.float16)
            o["vsa_inc"] = np.stack([(rec[f"vsa_out.{i}"][0] - rec[f"vsa_in.{i}"][0]).numpy() for i in range(nv)]).astype(np.float32)
    for k, v in act.items():
        o["act_" + k.replace("action.", "")] = np.asarray(v)[0].astype(np.float32)
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--items", required=True, help="JSON list of {obs, k, tag}")
    ap.add_argument("--wordings", required=True, help="JSON {name: text}")
    ap.add_argument("--out", required=True)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--head", action="store_true", help="also record the action head (DiT, VL self-attention)")
    ap.add_argument("--actions-only", action="store_true", help="store only actions (offline checks), batched")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()
    items = json.loads(Path(a.items).read_text())
    wordings = json.loads(Path(a.wordings).read_text()) if a.wordings.endswith(".json") else json.loads(a.wordings)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base, sim, report = gs.load_policy(edit=a.edit)
    pol = gs.SeededPolicy(sim, None)
    rec = None if a.actions_only else Recorder(base.model, head=a.head)
    index = {"items": items, "wordings": wordings, "edit": report and {k: v for k, v in report.items() if k != "per_tensor"},
             "files": []}
    if a.actions_only:  # batched replays (same chunking for every arm, so arms stay comparable)
        bs = max(1, a.batch)
        for wn, text in wordings.items():
            for i0 in range(0, len(items), bs):
                chunk = list(range(i0, min(i0 + bs, len(items))))
                reqs = [request(items[i]["obs"], int(items[i]["k"]), text) for i in chunk]
                obs = {k: (np.concatenate([r[0][k] for r in reqs]) if k != "annotation.language.language_instruction"
                           else [text] * len(reqs)) for k in reqs[0][0]}
                act, _ = pol.get_action(obs, {"seeds": [r[1] for r in reqs]})
                for j, i in enumerate(chunk):
                    o = {k2.replace("action.", "act_"): np.asarray(v)[j].astype(np.float32) for k2, v in act.items()}
                    f = out / f"i{i:03d}_{wn}.npz"
                    np.savez(f, **o)
                    index["files"].append({"item": i, "wording": wn, "file": str(f), "seed": reqs[j][1]})
            print(f"wording {wn}: {len(items)} items", flush=True)
    else:
        for i, it in enumerate(items):
            for wn, text in wordings.items():
                obs, seed = request(it["obs"], int(it["k"]), text)
                rec.clear()
                act, _ = pol.get_action(obs, {"seeds": [seed]})
                o = pack(rec.rec, act, a.head)
                f = out / f"i{i:03d}_{wn}.npz"
                np.savez(f, **o)
                index["files"].append({"item": i, "wording": wn, "file": str(f), "seed": seed})
            print(f"item {i+1}/{len(items)}", flush=True)
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print("done", flush=True)


if __name__ == "__main__":
    main()
