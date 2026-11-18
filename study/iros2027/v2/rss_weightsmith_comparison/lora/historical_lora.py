"""Action-target LoRA on the six observations used by the successful temporal edit.

This adapter consumes archived instrumented teacher actions. It does not invent
modern live-batch certificates for historical observations. Selection is owned by
the separate development-rollout controller, never by these fitting diagnostics.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
sys.path.insert(0, str(REPO))
UTILITY = REPO / "study/iros2027/v2/rss_repair_comparison/lora/train_lora.py"
spec = importlib.util.spec_from_file_location("original_lora_utilities", UTILITY)
U = importlib.util.module_from_spec(spec)
spec.loader.exec_module(U)
sha, dump = U.sha, U.dump
SOURCE = Path("/home/ali/vibes/opus-weight-surgery/projects/cube-then-banana")
COHORT = "rss-weightsmith-comparison-20261118"
ORIGINAL = "Put the cube then the banana in the bowl"
TEACHER = "Put the cube in the bowl then put the banana in the bowl"
DECISIONS = [(20001, 5), (20007, 5), (20009, 4), (20022, 4), (20023, 4), (20040, 4)]
LAYERS = tuple(range(17))
RANK, STEPS, SEED = 72, 200, 17
LEARNING_RATES = (0.00003, 0.0001, 0.001)
ROUNDTRIP_TOLERANCE = 1e-5


def verify_artifacts(artifacts):
    for name, expected in artifacts.items():
        if sha(name) != expected:
            raise ValueError(f"Source changed: {name}")


def source_items(index):
    if index["cand"] != "base":
        raise ValueError("Teacher capture must use the unedited checkpoint")
    items = []
    for i, (seed, k) in enumerate(DECISIONS):
        original, teacher = index["items"][i], index["items"][12+i]
        for row, prompt in ((original, ORIGINAL), (teacher, TEACHER)):
            if row["episode"] != f"base.base.s{seed}" or row["k"] != k or row["prompt"] != prompt:
                raise ValueError("Historical source observation/prompt identity differs")
        for field in ("episode", "k", "seed", "sim_step", "gripper_state"):
            if original[field] != teacher[field]:
                raise ValueError(f"Paired capture differs at {field}")
        items.append({"item_id": f"historical-{seed}-k{k}", "episode": original["episode"],
                      "decision_index": k, "policy_seed": original["seed"],
                      "sim_step": original["sim_step"], "gripper_state": original["gripper_state"],
                      "original_row": i, "teacher_row": 12+i,
                      "obs_path": str(SOURCE / "data/episodes" / original["episode"] / "obs.npz")})
    return items


def seal(args):
    from weightsmith import config as C
    out = Path(args.output)
    if out.exists():
        raise FileExistsError(out)
    index_path = SOURCE / "data/acts/c2-multi/index.json"
    index = json.loads(index_path.read_text())
    items = source_items(index)
    files = [Path(__file__), UTILITY, index_path, SOURCE / "data/acts/c2-multi/acts.npz",
             C.BASE_WS / "workspace.json", C.CHECKPOINT / "assets/droid/norm_stats.json",
             REPO / "third_party/pi05_droid_jointpos.sha256",
             C.OPENPI_DIR / "src/openpi/transforms.py",
             C.OPENPI_DIR / "src/openpi/training/config.py",
             C.OPENPI_DIR / "src/openpi/policies/policy_config.py"]
    for item in items:
        files.extend([Path(item["obs_path"]), Path(item["obs_path"]).with_name("summary.json")])
    out.parent.mkdir(parents=True, exist_ok=True)
    dump(out, {"cohort_id": COHORT, "split": "development", "target_source": "archived c2-multi instrumented teacher action chunks",
               "original_prompt": ORIGINAL, "teacher_prompt": TEACHER, "items": items,
               "historical_instrumented_live_max_abs_action_error": index["base_parity_max_abs_action_error"],
               "base_workspace": str(C.BASE_WS), "base_checkpoint": str(C.CHECKPOINT),
               "base_workspace_manifest_sha256": sha(C.BASE_WS / "workspace.json"),
               "config": {"layers": list(LAYERS), "rank": RANK, "steps": STEPS, "seed": SEED,
                          "learning_rates": list(LEARNING_RATES), "batch_size": 1, "weight_decay": 0,
                          "clip_global_norm": 1, "roundtrip_tolerance": ROUNDTRIP_TOLERANCE},
               "artifacts": {str(f.resolve()): sha(f) for f in files},
               "selection": "Separate fresh development original-request rollouts; no fitting-state selection or held-out access"})
    print(json.dumps({"source_seal": str(out), "sha256": sha(out), "items": len(items)}), flush=True)


def raw_request(raw):
    return {"observation/exterior_image_1_left": raw["base_img"].copy(),
            "observation/wrist_image_left": raw["wrist_img"].copy(),
            "observation/joint_position": raw["joint"].copy(),
            "observation/gripper_position": raw["gripper"].copy()}


def normalized_teacher(policy, raw, physical_actions):
    # pi05_droid_jointpos's inference input pipeline expects delta action labels;
    # its output pipeline adds the original joint positions to the first 7 dims.
    delta = np.asarray(physical_actions, dtype=np.float32).copy()
    delta[:, :7] -= raw["joint"][None, :]
    inputs = policy._input_transform({**raw_request(raw), "prompt": ORIGINAL, "actions": delta})
    target = np.asarray(inputs.pop("actions"), dtype=np.float32)
    if target.shape != (15, 32) or not np.isfinite(target).all():
        raise ValueError("Unexpected normalized action shape or nonfinite value")
    if not np.array_equal(target[:, 8:], np.zeros((15, 24), dtype=np.float32)):
        raise ValueError("Unobserved action dimensions must use standard zero padding")
    reconstructed = np.asarray(policy._output_transform({"state": inputs["state"].copy(), "actions": target.copy()})["actions"], dtype=np.float32)
    error = float(np.max(np.abs(reconstructed.astype(np.float64) - physical_actions)))
    if error > ROUNDTRIP_TOLERANCE:
        raise ValueError(f"Teacher normalization roundtrip {error} exceeds {ROUNDTRIP_TOLERANCE}")
    return inputs, target, np.asarray(reconstructed, dtype=np.float32), error


def prepare(args):
    source = json.loads(Path(args.source_seal).read_text())
    verify_artifacts(source["artifacts"])
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    dump(out / "source-seal.json", source)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as om
    start = time.time()
    policy = U.load_policy(); graph, state = nnx.split(policy._model)
    @jax.jit
    def sample(state, observation, seed):
        return nnx.merge(graph, state).sample_actions(jax.random.key(seed), observation)
    capture = SOURCE / "data/acts/c2-multi/acts.npz"
    with np.load(capture, allow_pickle=False) as z:
        archived_actions = z["actions"].copy()
    records = []
    for i, item in enumerate(source["items"]):
        k = item["decision_index"]
        with np.load(item["obs_path"], allow_pickle=False) as z:
            raw = {key: z[key][k].copy() for key in ("base_img", "wrist_img", "joint", "gripper")}
            if int(z["seed"][k]) != item["policy_seed"] or int(z["step"][k]) != item["sim_step"]:
                raise ValueError("Recorded observation and paired capture have different seed/step")
            if float(raw["gripper"][0]) != item["gripper_state"]:
                raise ValueError("Recorded observation and paired capture have different gripper state")
            live_recorded = z["actions"][k].copy()
        teacher_actions = archived_actions[item["teacher_row"]]
        inputs, target, reconstructed, error = normalized_teacher(policy, raw, teacher_actions)
        current, differences = {}, {}
        for label, prompt, archive in (("original", ORIGINAL, archived_actions[item["original_row"]]),
                                       ("teacher", TEACHER, teacher_actions)):
            inp = policy._input_transform({**raw_request(raw), "prompt": prompt})
            obs = om.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], inp))
            predicted = np.asarray(sample(state, obs, item["policy_seed"])[0], dtype=np.float32)
            current[label] = np.asarray(policy._output_transform({"state": inp["state"].copy(), "actions": predicted.copy()})["actions"], dtype=np.float32)
            if not np.isfinite(current[label]).all():
                raise ValueError("Nonfinite current native replay")
            differences[label] = U.physical_errors(current[label], archive)
        name = f"item-{i:02d}.npz"
        np.savez_compressed(out / name, **{"input/"+key: val for key, val in U.flatten(inputs).items()},
                            teacher_actions=target, teacher_actions_physical=teacher_actions,
                            reconstructed_teacher_actions_physical=reconstructed,
                            archived_original_actions=archived_actions[item["original_row"]],
                            recorded_live_actions=live_recorded,
                            current_original_actions=current["original"], current_teacher_actions=current["teacher"])
        records.append({**item, "file": name, "sha256": sha(out / name), "normalization_roundtrip_max_abs": error,
                        "current_native_vs_archived": differences,
                        "current_native_vs_recorded_live": U.physical_errors(current["original"], live_recorded),
                        "original_input_sha256": U.input_digest(raw, item["policy_seed"], ORIGINAL),
                        "teacher_input_sha256": U.input_digest(raw, item["policy_seed"], TEACHER)})
        dump(out / "progress.json", {"stage": "preparing", "done": i+1, "total": 6, "wall_seconds": time.time()-start})
        print(json.dumps({"item": i, "roundtrip_error": error, "native_archive_error": {key: value["max_abs_error"] for key, value in differences.items()}}), flush=True)
    verify_artifacts(source["artifacts"])
    meta = {"cohort_id": COHORT, "split": "development", "source_seal_sha256": sha(out / "source-seal.json"),
            "target_source": source["target_source"], "base_workspace_manifest_sha256": source["base_workspace_manifest_sha256"],
            "records": records, "train_inputs": 6, "validation_inputs": 0,
            "target_padding": "Eight observed action coordinates; remaining 24 are standard zero-padded normalized labels",
            "normalization_gate": {"tolerance": ROUNDTRIP_TOLERANCE, "max_abs_error": max(r["normalization_roundtrip_max_abs"] for r in records), "status": "pass"},
            "replay_interpretation": "Current native versus archived instrumented and recorded live actions are diagnostics, not an exact live-batch certificate. No threshold relaxed and no replay output substitutes for the archived labels.",
            "historical_instrumented_live_max_abs_action_error": source["historical_instrumented_live_max_abs_action_error"],
            "wall_seconds": time.time()-start, "code_sha256": sha(__file__)}
    dump(out / "dataset.json", meta)
    dump(out / "progress.json", {"stage": "complete", "items": 6, "wall_seconds": meta["wall_seconds"]})
    print(json.dumps({"dataset": str(out), "sha256": sha(out / "dataset.json"), "normalization_gate": meta["normalization_gate"]}), flush=True)


def load_dataset(path):
    path = Path(path); meta = json.loads((path / "dataset.json").read_text())
    source = json.loads((path / "source-seal.json").read_text())
    if meta["split"] != "development" or meta["cohort_id"] != COHORT or meta["train_inputs"] != 6 or meta["validation_inputs"] != 0:
        raise ValueError("Wrong study/data split")
    if sha(path / "source-seal.json") != meta["source_seal_sha256"]:
        raise ValueError("Source seal changed")
    verify_artifacts(source["artifacts"])
    if len(meta["records"]) != 6 or len(source["items"]) != 6:
        raise ValueError("All six source observations are required")
    examples = []
    for original, rec in zip(source["items"], meta["records"], strict=True):
        if any(original[k] != rec[k] for k in original):
            raise ValueError("Prepared item differs from original frozen source")
        file = path / rec["file"]
        if sha(file) != rec["sha256"]:
            raise ValueError(f"Prepared data changed: {file}")
        with np.load(file, allow_pickle=False) as z:
            inp = U.unflatten({key[6:]: z[key].copy() for key in z.files if key.startswith("input/")})
            target, physical = z["teacher_actions"].copy(), z["teacher_actions_physical"].copy()
            error = float(np.max(np.abs(z["reconstructed_teacher_actions_physical"].astype(np.float64)-physical)))
        if error > ROUNDTRIP_TOLERANCE or error != rec["normalization_roundtrip_max_abs"] or target.shape != (15, 32):
            raise ValueError("Invalid teacher normalization certificate")
        examples.append((inp, target, physical, rec))
    return meta, examples


def fit(args):
    meta, examples = load_dataset(args.dataset)
    out = Path(args.output); out.mkdir(parents=True, exist_ok=False)
    steps = 1 if args.command == "smoke" else STEPS
    settings = {"layers": list(LAYERS), "rank": RANK, "steps": steps, "learning_rate": args.lr,
                "seed": SEED, "batch_size": 1, "weight_decay": 0, "gradient_clip": 1,
                "objective": "native flow matching on normalized archived teacher actions", "image_augmentation": False,
                "selection": "Separate development rollouts; no checkpoint or fitting-metric selection"}
    dump(out / "settings.json", settings)
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from openpi.models import model as om
    from weightsmith import config as C
    if sha(C.BASE_WS / "workspace.json") != meta["base_workspace_manifest_sha256"]:
        raise ValueError("Base workspace differs from target preparation")
    start = time.time(); policy = U.load_policy(); graph, state = nnx.split(policy._model)
    packed = state["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].value
    rng = np.random.default_rng(SEED)
    factors = {"a": jnp.asarray(rng.normal(0, .01, (len(LAYERS), packed.shape[1], RANK)).astype(np.float32)),
               "b": jnp.zeros((len(LAYERS), RANK, packed.shape[2]), dtype=jnp.float32)}
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(args.lr, weight_decay=0))
    opt_state = optimizer.init(factors)
    zero = U.injected_state(state, factors, LAYERS, jax, jnp)["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].value
    if not bool(jnp.array_equal(zero, packed)):
        raise ValueError("Zero-factor model differs from the base")
    del zero
    def loss(a, st, obs, target, key):
        return jnp.mean(nnx.merge(graph, U.injected_state(st, a, LAYERS, jax, jnp)).compute_loss(key, obs, target, train=False))
    grad = jax.value_and_grad(loss)
    @jax.jit
    def step(a, opt, st, obs, target, key):
        value, g = grad(a, st, obs, target, key)
        updates, opt = optimizer.update(g, opt, a)
        return optax.apply_updates(a, updates), opt, value, optax.global_norm(g)
    @jax.jit
    def sample(a, st, obs, seed):
        return nnx.merge(graph, U.injected_state(st, a, LAYERS, jax, jnp)).sample_actions(jax.random.key(seed), obs)
    def batch(ex):
        return om.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], ex[0])), jnp.asarray(ex[1])[None]
    def diagnostics(a):
        rows = []
        for ex in examples:
            obs, _ = batch(ex)
            pred = np.asarray(sample(a, state, obs, ex[3]["policy_seed"])[0], dtype=np.float32)
            physical = policy._output_transform({"state": ex[0]["state"].copy(), "actions": pred.copy()})["actions"]
            rows.append({"item_id": ex[3]["item_id"], "physical_mse": float(np.mean((physical.astype(np.float64)-ex[2])**2))})
        return {"items": rows, "mean_physical_mse": float(np.mean([r["physical_mse"] for r in rows])),
                "interpretation": "Fitting-state action discrepancy only; not validation, selection, or closed-loop success"}
    history = []
    for i in range(steps):
        obs, target = batch(examples[int(rng.integers(len(examples)))])
        factors, opt_state, value, norm = step(factors, opt_state, state, obs, target, jax.random.key(SEED+i))
        value, norm = float(value), float(norm)
        if not np.isfinite(value+norm) or (i == 0 and norm <= 0):
            raise ValueError("Nonfinite or zero initial LoRA gradient")
        if i == 0 or (i+1) % 10 == 0 or i+1 == steps:
            row = {"step": i+1, "flow_loss": value, "gradient_norm": norm, "wall_seconds": time.time()-start}
            history.append(row); dump(out / "progress.json", {"stage": "fitting", **row, "steps": steps})
            print(json.dumps(row), flush=True)
    report = {**settings, "cohort_id": COHORT, "split": "development", "dataset_sha256": sha(Path(args.dataset) / "dataset.json"),
              "train_inputs": 6, "validation_inputs": 0, "history": history, "zero_start_exact": True,
              "trainable_parameters": sum(x.size for x in factors.values()), "wall_seconds": time.time()-start,
              "memory_stats": jax.devices()[0].memory_stats(), "device": str(jax.devices()[0]),
              "source_code_sha256": sha(__file__), "utility_sha256": sha(UTILITY), "closed_loop_evaluation": "not performed"}
    if args.command == "smoke":
        dump(out / "smoke.json", report)
    else:
        report["fitting_diagnostics"] = diagnostics(factors)
        np.savez_compressed(out / "adapters.npz", **jax.tree.map(np.asarray, factors), layers=np.asarray(LAYERS))
        merged = U.injected_state(state, factors, LAYERS, jax, jnp)
        report["workspace_validation"] = U.export_workspace(out, LAYERS, C.BASE_WS, packed,
            merged["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].value)
        verify_artifacts(json.loads((Path(args.dataset) / "source-seal.json").read_text())["artifacts"])
        report["wall_seconds"] = time.time()-start
        dump(out / "training.json", report)
        files = [out / "settings.json", out / "training.json", out / "adapters.npz", out / "workspace/workspace.json",
                 *sorted((out / "workspace/edits").glob("*.npy"))]
        dump(out / "seal.json", {"candidate_id": f"lora-lr-{args.lr:.5f}".rstrip("0"), "learning_rate": args.lr,
                                "workspace": str((out / "workspace").resolve()), "artifacts": {str(p.resolve()): sha(p) for p in files}})
    dump(out / "progress.json", {"stage": "complete", "wall_seconds": time.time()-start, "steps": steps})
    print(json.dumps({"output": str(out), "steps": steps, "wall_seconds": time.time()-start,
                      "trainable_parameters": report["trainable_parameters"], "memory_stats": report["memory_stats"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("seal"); s.add_argument("--output", required=True); s.set_defaults(func=seal)
    p = sub.add_parser("prepare"); p.add_argument("--source-seal", required=True); p.add_argument("--output", required=True)
    p.add_argument("--gpu", type=int, required=True); p.set_defaults(func=prepare)
    for command in ("smoke", "train"):
        t = sub.add_parser(command); t.add_argument("--dataset", required=True); t.add_argument("--output", required=True)
        t.add_argument("--gpu", type=int, required=True); t.add_argument("--lr", type=float, choices=LEARNING_RATES, default=0.0001)
        t.set_defaults(func=fit)
    args = parser.parse_args()
    if args.command == "seal":
        args.func(args); return
    if os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu)) != str(args.gpu):
        parser.error("--gpu must match the sole physical CUDA_VISIBLE_DEVICES device")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")
    from weightsmith.locks import GpuLock
    with GpuLock(args.gpu):
        args.func(args)


if __name__ == "__main__":
    main()
