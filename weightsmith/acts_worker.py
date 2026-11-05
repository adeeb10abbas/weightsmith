"""GPU activation capture on exact recorded policy inputs (openpi venv).

spec (stdin JSON):
  cand: "base" | candidate name (workspace passed via --workspace)
  items: [{"episode": eid, "k": replan_index} | {"episode": eid, "step": sim_step}]   (nearest replan <= step)
  prompt: optional instruction override (default: task instruction)
  action_layers: list of action-expert layers 0..17 (default all)
  vlm_layers: list of VLM layers 0..17 (default [])
  denoise_steps: which of the 10 flow steps to keep (default all) ; t = 1.0 - 0.1*i
  action_tokens: which of 15 action tokens to keep (default "mean" -> mean over tokens) or list of ints
Saved (float16) per item, stacked over items in acts.npz:
  action.mlp_hidden    [N, S, La, T, 4096]  gated GELU hidden = input of action.blocks.LL.mlp_1.linear
  action.resid_out     [N, S, La, T, 1024]  residual stream after block
  action.head_input    [N, S, 15, 1024]     input of action_out_proj
  vlm.mlp_hidden       [N, Lv, P, 16384]    VLM MLP hidden at prompt tokens (P = prompt token positions)
  vlm.resid_out        [N, Lv, P, 2048]
  actions              [N, 15, 8]           final unnormalized action chunk (joint vel/pos + gripper)
  recorded_actions     [N, 15, 8]           what the policy produced live (parity check for base)
"""
from __future__ import annotations

import argparse, dataclasses, json, sys, time, types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402

CHECKPOINT = C.CHECKPOINT


@dataclasses.dataclass(frozen=True)
class Spec:
    layers: tuple
    action_tokens: tuple
    prompt_tokens: tuple


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--workspace", default="")
    a = ap.parse_args()
    spec = json.loads(sys.stdin.read())
    out = Path(a.output)
    out.mkdir(parents=True)

    import jax, jax.numpy as jnp
    from openpi.models import model as openpi_model
    from openpi.policies import policy_config
    from openpi.shared import nnx_utils
    from openpi.training import config as training_config
    from weightsmith.lab.openpi_probe import make_probe_method
    from weightsmith.lab.openpi_vlm_full import _apply_edits

    policy = policy_config.create_trained_policy(training_config.get_config("pi05_droid_jointpos"), CHECKPOINT)
    if a.workspace:
        _apply_edits(policy._model, Path(a.workspace))
    a_layers = spec.get("action_layers") or list(range(18))
    v_layers = spec.get("vlm_layers") or []
    layers = sorted(set(a_layers) | set(v_layers))
    tok = spec.get("action_tokens", "mean")
    toks = tuple(range(15)) if tok == "mean" else tuple(tok)
    dsteps = spec.get("denoise_steps") or list(range(10))
    prompt = spec.get("prompt") or spec["task_instruction"]
    # prompt token positions: probe indexes x.shape[1]-200+i, i over the 200-slot prompt region
    ptoks = tuple(spec.get("prompt_tokens") or range(0, 24))
    if len(ptoks) > 64 or min(ptoks) < 0 or max(ptoks) >= 200:
        raise SystemExit("prompt_tokens must be 0..199, at most 64")
    want_img = bool(spec.get("image_tokens")) and bool(v_layers)
    img_off = tuple(range(-768, -768 + 512))  # base cam tokens 0..255, wrist cam 256..511 (absolute prefix positions)
    probe_toks = ptoks + (img_off if want_img else ())
    want_attn = bool(spec.get("action_attention"))
    probe = nnx_utils.module_jit(types.MethodType(make_probe_method(Spec(tuple(layers), toks, probe_toks)), policy._model))
    NP = len(ptoks)

    def pool_img(x):  # x [Lv, 512, F] -> [Lv, 2 cams, 17 (mean + 4x4 grid), F]
        out = []
        for c in range(2):
            g = x[:, c * 256:(c + 1) * 256].reshape(x.shape[0], 16, 16, -1)
            cells = g.reshape(x.shape[0], 4, 4, 4, 4, -1).mean(axis=(2, 4)).reshape(x.shape[0], 16, -1)
            out.append(np.concatenate([g.mean(axis=(1, 2))[:, None], cells], axis=1))
        return np.stack(out, axis=1)
    li_a = [layers.index(l) for l in a_layers]
    li_v = [layers.index(l) for l in v_layers]

    def red_tok(x):  # [..., T, F]
        return x.mean(axis=-2, keepdims=True) if tok == "mean" else x

    store = {k: [] for k in ["action.mlp_hidden", "action.resid_out", "action.head_input", "vlm.mlp_hidden", "vlm.resid_out",
                             "vlm.img_mlp_hidden", "vlm.img_resid_out", "action.attn_mass", "action.attn_map",
                             "actions", "recorded_actions"]}
    rows = []
    t0 = time.time()
    epi = Path(spec["episodes_dir"])
    cache = {}
    ep_prompt = {}
    for it in spec["items"]:
        sp = epi / it["episode"] / "summary.json"
        if sp.exists():
            pr = json.loads(sp.read_text()).get("prompt")
            if pr:
                ep_prompt[it["episode"]] = pr
    max_err = 0.0
    for n, it in enumerate(spec["items"]):
        eid = it["episode"]
        if eid not in cache:
            z = np.load(epi / eid / "obs.npz")
            cache[eid] = {k: z[k] for k in z.files}
        o = cache[eid]
        if "k" in it:
            k = int(it["k"])
        else:
            k = int(np.searchsorted(o["step"], int(it["step"]), side="right") - 1)
        k = max(0, min(k, len(o["step"]) - 1))
        request = {
            "observation/exterior_image_1_left": o["base_img"][k],
            "observation/wrist_image_left": o["wrist_img"][k],
            "observation/joint_position": o["joint"][k],
            "observation/gripper_position": o["gripper"][k],
            "prompt": it.get("prompt") or spec.get("prompt") or ep_prompt.get(eid) or prompt,
        }
        seed = int(o["seed"][k]) if int(o["seed"][k]) >= 0 else 0
        inputs = policy._input_transform(dict(request))
        inputs = jax.tree.map(lambda v: jnp.asarray(v)[None], inputs)
        x, tr = probe(jax.random.key(seed), openpi_model.Observation.from_dict(inputs))
        actions = policy._output_transform({"state": np.asarray(inputs["state"][0]), "actions": np.asarray(x[0])})["actions"]
        rec = o["actions"][k]
        if not a.workspace and request["prompt"] == ep_prompt.get(eid, prompt):
            max_err = max(max_err, float(np.max(np.abs(actions - rec))))
        act = tr["action"]
        mh = np.asarray(act["mlp_down_input"])[dsteps][:, li_a]          # [S, La, T, 4096]
        ro = np.asarray(act["residual_output"])[dsteps][:, li_a]
        store["action.mlp_hidden"].append(red_tok(mh[:, :, 0] if mh.ndim == 5 else mh).astype(np.float16))
        store["action.resid_out"].append(red_tok(ro[:, :, 0] if ro.ndim == 5 else ro).astype(np.float16))
        store["action.head_input"].append(np.asarray(act["head_input"])[dsteps][:, 0, -15:].astype(np.float16))
        if li_v:
            vm = np.asarray(tr["vlm"]["mlp_down_input"])[li_v][:, 0]   # [Lv, P(+512), 16384]
            vr = np.asarray(tr["vlm"]["residual_output"])[li_v][:, 0]
            store["vlm.mlp_hidden"].append(vm[:, :NP].astype(np.float16))
            store["vlm.resid_out"].append(vr[:, :NP].astype(np.float16))
            if want_img:
                store["vlm.img_mlp_hidden"].append(pool_img(vm[:, NP:].astype(np.float32)).astype(np.float16))
                store["vlm.img_resid_out"].append(pool_img(vr[:, NP:].astype(np.float32)).astype(np.float16))
        if want_attn:
            ap = np.asarray(act["attention_probs"])[dsteps][:, li_a][:, :, 0].astype(np.float32)  # [S, La, H, T, K]
            segs = [(0, 256), (256, 512), (512, 768), (768, 968), (968, ap.shape[-1])]
            mass = np.stack([ap[..., a:b].sum(-1) for a, b in segs], -1).mean(axis=3)  # [S, La, H, 5]
            hm = ap.mean(axis=(2, 3))  # [S, La, K]
            amap = np.stack([hm[..., 0:256].reshape(hm.shape[0], hm.shape[1], 16, 16),
                             hm[..., 256:512].reshape(hm.shape[0], hm.shape[1], 16, 16)], axis=2)  # [S, La, 2, 16, 16]
            store["action.attn_mass"].append(mass.astype(np.float16))
            store["action.attn_map"].append(amap.astype(np.float16))
        store["actions"].append(np.asarray(actions, np.float32))
        store["recorded_actions"].append(np.asarray(rec, np.float32))
        tids = np.asarray(tr["token_ids"])[0]
        tmask = np.asarray(tr["token_mask"])[0]
        rows.append({"i": n, "episode": eid, "k": k, "sim_step": int(o["step"][k]), "seed": seed, "prompt": request["prompt"],
                     "gripper_state": float(o["gripper"][k][0]),
                     "prompt_token_ids_at_slots": [int(tids[j]) for j in ptoks if j < len(tids)],
                     "prompt_token_valid_at_slots": [bool(tmask[j]) for j in ptoks if j < len(tmask)]})
    np.savez(out / "acts.npz", **{k: np.stack(v) for k, v in store.items() if v})
    shapes = {k: list(np.shape(np.stack(v))) for k, v in store.items() if v}
    try:
        import sentencepiece, glob as _g
        spm_path = (_g.glob(str(C.OPENPI_DATA_HOME / "big_vision/paligemma_tokenizer.model")) or _g.glob(str(C.OPENPI_DATA_HOME / "**/paligemma_tokenizer.model"), recursive=True))[0]
        sp = sentencepiece.SentencePieceProcessor(model_file=spm_path)
        pieces0 = [sp.id_to_piece(t) for t in rows[0]["prompt_token_ids_at_slots"]]
    except Exception as e:  # noqa: BLE001
        pieces0 = f"unavailable: {e}"
    index = {"cand": spec.get("cand", "base"), "prompt_token_pieces_item0": pieces0, "action_layers": a_layers, "vlm_layers": v_layers, "denoise_steps": dsteps,
             "denoise_t": [round(1.0 - 0.1 * i, 2) for i in dsteps], "action_tokens": tok, "prompt_token_slots": list(ptoks),
             "items": rows, "shapes": shapes, "base_parity_max_abs_action_error": max_err,
             "image_and_attention_layout": "vlm.img_*: [N, Lv, cam(0=base over-shoulder, 1=wrist), 17, F]; index 0 = mean over the cam's 256 tokens, 1..16 = 4x4 spatial grid cells row-major (row 0 = top of the 224x224 image; each cell = 4x4 patches). action.attn_mass: [N, S, La, head(8), segment] attention mass of the action tokens (mean over kept action tokens) on segments [base_cam, wrist_cam, masked_3rd_cam(padding image), prompt+state, action_tokens]. action.attn_map: [N, S, La, cam, 16, 16] head- and token-averaged attention over each camera's 16x16 patch grid.", "notes": "parity_err compares this replay (acts GPU, instrumented) with what the live batched policy produced; ~0.01 is bf16 noise. action.mlp_hidden[:, s, j] is the input to action.blocks.{a_layers[j]}.mlp_1.linear (shape [4096,1024]); "
                      "row r of that matrix is written to the residual stream scaled by hidden unit r."}
    (out / "index.json").write_text(json.dumps(index, indent=1))
    print(json.dumps({"id": out.name, "items": len(rows), "shapes": shapes, "parity_err": max_err, "sec": round(time.time() - t0, 1)}))


if __name__ == "__main__":
    main()
