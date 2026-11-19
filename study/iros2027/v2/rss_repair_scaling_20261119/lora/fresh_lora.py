"""Fresh self-demonstration LoRA for the modest repair-scaling study.

No historical training label is reused. The primary arm uses every recorded
native action chunk along each selected successful clarified-policy episode.
Student observations carry the original request. Exact fixed-16 source replay
and physical action normalization are separate mandatory acceptance gates.
"""
from __future__ import annotations

import argparse
from collections import OrderedDict
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
NATIVE = REPO / "study/iros2027/v2/rss_weightsmith_goals/adaptation/native_targets.py"
SERVER = REPO / "study/iros2027/v2/rss_repair_comparison/runtime/batched_server.py"
LAYERS = tuple(range(17))
RANK, SEED = 72, 17
COUNTS = (1, 3, 6, 12)
LEARNING_RATES = (0.0001, 0.001)
REPLAY_TOLERANCE, ROUNDTRIP_TOLERANCE = 1e-6, 1e-5


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


U = module("repair_scaling_lora_utility", UTILITY)
N = module("repair_scaling_native_validation", NATIVE)
sha, dump = U.sha, U.dump


def raw_request(raw):
    return {"observation/exterior_image_1_left": raw["base_img"].copy(),
            "observation/wrist_image_left": raw["wrist_img"].copy(),
            "observation/joint_position": raw["joint"].copy(),
            "observation/gripper_position": raw["gripper"].copy()}


def read_observation(path, k, cache=None):
    path = str(Path(path).resolve())
    if cache is not None and path in cache:
        arrays = cache.pop(path)
    else:
        with np.load(path, allow_pickle=False) as z:
            arrays = {key: z[key].copy() for key in ("base_img", "wrist_img", "joint", "gripper", "actions", "seed", "step")}
    if cache is not None:
        cache[path] = arrays
        while len(cache) > 16:
            cache.popitem(last=False)
    if not 0 <= k < len(arrays["seed"]):
        raise ValueError("Decision outside source episode")
    raw = {key: arrays[key][k].copy() for key in ("base_img", "wrist_img", "joint", "gripper")}
    actions = np.asarray(arrays["actions"][k], dtype=np.float32).copy()
    seed, step = int(arrays["seed"][k]), int(arrays["step"][k])
    if seed < 0 or actions.shape != (15, 8) or not np.isfinite(actions).all():
        raise ValueError("Invalid source action chunk or sampling seed")
    return raw, actions, seed, step


def normalize_physical(policy, raw, actions, original_prompt):
    physical = np.asarray(actions, dtype=np.float32)
    if physical.shape != (15, 8) or not np.isfinite(physical).all():
        raise ValueError("Expected finite native physical action chunk (15, 8)")
    relative = physical.copy()
    relative[:, :7] -= raw["joint"][None, :]
    inputs = policy._input_transform({**raw_request(raw), "prompt": original_prompt, "actions": relative})
    target = np.asarray(inputs.pop("actions"), dtype=np.float32)
    if target.shape != (15, 32) or not np.isfinite(target).all() or np.any(target[:, 8:] != 0):
        raise ValueError("Invalid normalized action target or nonzero padded coordinates")
    reconstructed = np.asarray(policy._output_transform({"state": inputs["state"].copy(),
                                                         "actions": target.copy()})["actions"], dtype=np.float32)
    error = N.physical_errors(reconstructed, physical)["max_abs_error"]
    if error > ROUNDTRIP_TOLERANCE:
        raise ValueError(f"Physical normalization roundtrip {error:g} exceeds {ROUNDTRIP_TOLERANCE:g}")
    return inputs, target, reconstructed, error


def read_spec(path, arm):
    spec = json.loads(Path(path).read_text())
    if spec.get("split") != "development" or not spec.get("cohort_id") or not spec.get("items"):
        raise ValueError("An explicit nonempty development source spec is required")
    if arm not in {"full_trajectory", "matched_decision"}:
        raise ValueError("Unknown supervision arm")
    spec_arm = "matched_decision" if spec.get("arm") == "correction" else spec.get("arm", arm)
    if spec_arm != arm:
        raise ValueError("Requested supervision differs from the source spec")
    records, identities, ids, episodes = [], set(), set(), {}
    cache = {"hashes": {}}
    for i, source in enumerate(spec["items"]):
        item = dict(source)
        if item.get("split") != "train":
            raise ValueError("Only training episodes belong in this dataset; selection uses separate rollouts")
        path = Path(item["obs_path"]).resolve()
        if any("heldout" in p.lower() or "held-out" in p.lower() for p in path.parts):
            raise ValueError("Held-out observations cannot train LoRA")
        k = item.get("decision_index")
        if type(k) is not int or k < 0 or (str(path), k) in identities:
            raise ValueError("Invalid or duplicate source decision")
        identities.add((str(path), k))
        item.update(obs_path=str(path), item_id=item.get("item_id", f"item-{i:05d}"),
                    role=item.get("role", "repair"))
        if item["item_id"] in ids or item["role"] != "repair":
            raise ValueError("Duplicate item ID or unsupported source role")
        ids.add(item["item_id"])
        for key in ("original_prompt", "teacher_prompt", "recorded_prompt"):
            item[key] = item.get(key, spec.get(key))
            if not isinstance(item[key], str) or not item[key].strip():
                raise ValueError(f"Missing {key}")
        if arm == "full_trajectory" and item["recorded_prompt"] != item["teacher_prompt"]:
            raise ValueError("Full-trajectory labels require a recorded clarified-policy rollout")
        if arm == "matched_decision" and item["recorded_prompt"] != item["original_prompt"]:
            raise ValueError("Matched decisions require original-request source observations")
        N.validate_live_context(item, spec, cache)
        source_rows = json.loads(Path(item["live_context"]["source_episodes_path"]).read_text())
        source_row = next(row for row in source_rows if Path(row["obs"]).resolve() == path)
        if arm == "full_trajectory" and source_row.get("strict_final", {}).get("cube_then_banana") is not True:
            raise ValueError("A full-trajectory training source did not succeed under the strict temporal predicate")
        episodes.setdefault(str(path), []).append(k)
        records.append(item)
    count = len(episodes)
    if count not in COUNTS or spec.get("correction_count", count) != count:
        raise ValueError("Expected 1, 3, 6, or 12 distinct source episodes")
    for path, indices in episodes.items():
        with np.load(path, allow_pickle=False) as z:
            expected = list(range(len(z["seed"])))
        if arm == "full_trajectory" and sorted(indices) != expected:
            raise ValueError("Full-trajectory supervision must include every recorded decision exactly once")
        if arm == "matched_decision" and len(indices) != 1:
            raise ValueError("Matched supervision requires exactly one chosen decision per source episode")
    if not spec.get("source_manifest") or not spec.get("source_manifest_sha256"):
        raise ValueError("A frozen fresh source-pair manifest is required")
    if "source_manifest" in spec:
        manifest_path = Path(spec["source_manifest"])
        if sha(manifest_path) != spec.get("source_manifest_sha256"):
            raise ValueError("Source pair manifest changed")
        manifest = json.loads(manifest_path.read_text())
        if (manifest.get("cohort_id") != spec["cohort_id"] or manifest.get("split") != "development"
                or count not in manifest.get("correction_budgets_available", [])):
            raise ValueError("Source pair manifest cohort or budget differs")
        expected_pairs = manifest["pairs"][:count]
        if len(expected_pairs) != count or any(pair.get("eligible") is not True for pair in expected_pairs):
            raise ValueError("Source prefix contains missing or ineligible correction pairs")
        if spec.get("distinct_pair_ids") != [pair["pair_id"] for pair in expected_pairs]:
            raise ValueError("Training pairs differ from the predeclared nested prefix")
        source_key = "teacher" if arm == "full_trajectory" else "original"
        expected_episodes = {str(Path(pair[source_key]["obs"]).resolve()): pair["pair_id"] for pair in expected_pairs}
        if set(expected_episodes) != set(episodes) or any(item.get("pair_id") != expected_episodes[item["obs_path"]] for item in records):
            raise ValueError("Training episodes differ from selected correction pairs")
        pairs_by_id = {pair["pair_id"]: pair for pair in expected_pairs}
        for item in records:
            pair = pairs_by_id[item["pair_id"]]
            if (item["original_prompt"] != pair["original"]["effective_prompt"]
                    or item["teacher_prompt"] != pair["teacher"]["effective_prompt"]):
                raise ValueError("Student or teacher instruction differs from the selected source pair")
        if arm == "matched_decision":
            selected = {pair["pair_id"]: pair["selected_decision"] for pair in expected_pairs}
            if any(item["decision_index"] != selected[item["pair_id"]] for item in records):
                raise ValueError("Correction decisions differ from the frozen selector")
        cache["hashes"][str(manifest_path.resolve())] = sha(manifest_path)
        for pair in expected_pairs:
            for cell in ("original", "teacher"):
                row = pair[cell]
                for path_key, hash_key in (("obs", "obs_sha256"), ("metrics", "metrics_sha256")):
                    if path_key in row:
                        if sha(row[path_key]) != row[hash_key]:
                            raise ValueError("Selected pair observation or success metrics changed")
                        cache["hashes"][str(Path(row[path_key]).resolve())] = row[hash_key]
        for name, expected in manifest.get("artifacts", {}).items():
            if sha(name) != expected:
                raise ValueError(f"Source collection artifact changed: {name}")
            cache["hashes"][str(Path(name).resolve())] = expected
    return spec, records, cache, count


def prepare(args):
    start = time.monotonic()
    spec, items, cache, count = read_spec(args.spec, args.arm)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / "input-spec.json", spec)
    # Read-only validation and source hashes above happen before loading CUDA.
    from weightsmith import config as C
    server = module("repair_scaling_fixed16_server", SERVER).Server()
    # Match the serving process, including its compiled-kernel cache and warmup.
    # Fresh compilation under another allocation budget changed BF16 actions in
    # the preserved first-block diagnostic, despite identical model inputs.
    server.load(None)
    server.warmup()
    policy = server.policy
    singleton = None
    if args.arm == "matched_decision":
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from openpi.models import model as om

        @jax.jit
        def singleton(state, observation, seed):
            return nnx.merge(server.graphdef, state).sample_actions(jax.random.key(seed), observation)

    grouped = OrderedDict()
    for index, item in enumerate(items):
        grouped.setdefault(item["live_context_sha256"], []).append((index, item))
    records = [None] * len(items)
    physical_targets = [None] * len(items)
    normalized_targets = [None] * len(items)
    replay_calls, replay_examples, teacher_calls = 0, 0, 0
    observation_cache = OrderedDict()
    for selected in grouped.values():
        first = selected[0][1]
        prompt = first["recorded_prompt"]
        if any(item["recorded_prompt"] != prompt for _, item in selected):
            raise ValueError("One live inference group cannot have conflicting recorded prompts")
        members = first["live_batch_members"]
        native_data = [read_observation(m["obs_path"], m["decision_index"], observation_cache) for m in members]
        requests = [{**raw_request(raw), "prompt": prompt, "sampling_seed": seed}
                    for raw, _, seed, _ in native_data]
        outputs = server.infer_batch(requests)
        replay_calls += 1
        replay_examples += 16
        errors = [N.physical_errors(output["actions"], data[1]) for output, data in zip(outputs, native_data, strict=True)]
        if any(error["max_abs_error"] > REPLAY_TOLERANCE for error in errors):
            np.savez_compressed(out / "replay-failure-actions.npz",
                replay=np.stack([np.asarray(x["actions"], dtype=np.float32) for x in outputs]),
                recorded=np.stack([x[1] for x in native_data]))
            dump(out / "replay-failure.json", {"source_step": first["source_step"], "prompt": prompt,
                 "members": members, "errors": errors, "maximum_error": max(e["max_abs_error"] for e in errors),
                 "actions_sha256": sha(out / "replay-failure-actions.npz")})
            raise ValueError("Exact fixed16 native source replay failed; target preparation stopped")
        member_indices = {(m["obs_path"], m["decision_index"]): i for i, m in enumerate(members)}
        for index, item in selected:
            j = member_indices[(item["obs_path"], item["decision_index"])]
            raw, recorded, seed, step = native_data[j]
            if step != item["source_step"]:
                raise ValueError("Source decision step changed")
            physical = recorded.copy()
            if args.arm == "matched_decision":
                teacher_input = policy._input_transform({**raw_request(raw), "prompt": item["teacher_prompt"]})
                batched = jax.tree.map(lambda x: jnp.asarray(x)[None], teacher_input)
                prediction = np.asarray(singleton(server.state, om.Observation.from_dict(batched), seed)[0])
                physical = policy._output_transform({"state": teacher_input["state"].copy(), "actions": prediction})["actions"]
                teacher_calls += 1
            inputs, target, reconstructed, roundtrip = normalize_physical(policy, raw, physical, item["original_prompt"])
            name = f"item-{index:05d}.npz"
            np.savez_compressed(out / name, **{"input/" + key: value for key, value in U.flatten(inputs).items()},
                                teacher_actions=target, teacher_actions_physical=physical,
                                reconstructed_teacher_actions_physical=reconstructed,
                                live_replay_actions_physical=np.asarray(outputs[j]["actions"], dtype=np.float32),
                                recorded_actions_physical=recorded)
            record = {**item, "policy_seed": seed, "sim_step": step,
                      "source_observation_sha256": cache["hashes"][item["obs_path"]],
                      "original_input_sha256": U.input_digest(raw, seed, item["original_prompt"]),
                      "teacher_input_sha256": U.input_digest(raw, seed, item["teacher_prompt"]),
                      "recorded_input_sha256": U.input_digest(raw, seed, item["recorded_prompt"]),
                      "live_replay_max_abs_error": errors[j]["max_abs_error"],
                      "live_replay_bit_exact": errors[j]["bit_exact"],
                      "live_replay_per_coordinate_max_abs_error": errors[j]["per_coordinate_max_abs_error"],
                      "normalization_roundtrip_max_abs": roundtrip, "file": name, "sha256": sha(out / name)}
            records[index], physical_targets[index], normalized_targets[index] = record, physical, target
        dump(out / "progress.json", {"stage": "preparing", "done": sum(r is not None for r in records),
                                     "total": len(items), "wall_seconds": time.monotonic() - start})
        print(json.dumps({"prepared": sum(r is not None for r in records), "total": len(items),
                          "source_replay_calls": replay_calls}), flush=True)
    np.savez_compressed(out / "teacher-actions.npz", actions=np.stack(physical_targets),
                        normalized_actions=np.stack(normalized_targets),
                        item_ids=np.asarray([r["item_id"] for r in records]),
                        policy_seeds=np.asarray([r["policy_seed"] for r in records]))
    artifacts = {str(path.resolve()): sha(path) for path in (Path(__file__), UTILITY, NATIVE, SERVER,
                 C.BASE_WS / "workspace.json", C.CHECKPOINT / "assets/droid/norm_stats.json")}
    meta = {"cohort_id": spec["cohort_id"], "split": "development", "arm": args.arm,
            "correction_count": count, "train_inputs": len(records), "validation_inputs": 0,
            "target_source": "recorded native clarified-policy physical action chunks along successful full episodes"
                             if args.arm == "full_trajectory" else "fresh native singleton clarified-policy physical action chunks",
            "source_information": "Model self-demonstrations; no human action labels. Full trajectories supply more decisions than the closed-form edit.",
            "base_workspace_manifest_sha256": sha(C.BASE_WS / "workspace.json"),
            "spec_sha256": sha(args.spec), "stored_spec_sha256": sha(out / "input-spec.json"),
            "records": records, "source_observation_sha256": {r["obs_path"]: r["source_observation_sha256"] for r in records},
            "live_replay_context_hashes": cache["hashes"], "source_artifacts": artifacts,
            "teacher_actions_file": "teacher-actions.npz", "teacher_actions_sha256": sha(out / "teacher-actions.npz"),
            "live_replay_gate": {"mode": "exact_pool_bucket", "status": "pass", "tolerance": REPLAY_TOLERANCE,
                "maximum_error": max(r["live_replay_max_abs_error"] for r in records), "items": len(records),
                "forward_calls": replay_calls, "padded_policy_examples": replay_examples, "kernel_batch": 16},
            "normalization_gate": {"tolerance": ROUNDTRIP_TOLERANCE, "status": "pass",
                "maximum_error": max(r["normalization_roundtrip_max_abs"] for r in records)},
            "additional_teacher_forward_calls": teacher_calls, "wall_seconds": time.monotonic() - start,
            "serving_warmup_forward_calls": 5, "serving_warmup_padded_examples": 80,
            "numerical_environment": {key: os.environ.get(key) for key in ("JAX_COMPILATION_CACHE_DIR",
                "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_FLAGS")},
            "source_collection_cost": "Owned by source collection receipts; include all attempted original and clarified episodes, not only selected successes"}
    N.validate_live_certificate(meta)
    dump(out / "dataset.json", meta)
    dump(out / "progress.json", {"stage": "complete", "items": len(records), "wall_seconds": meta["wall_seconds"]})


def load_dataset(path):
    root = Path(path).resolve()
    meta = json.loads((root / "dataset.json").read_text())
    N.validate_live_certificate(meta)
    if meta.get("split") != "development" or meta.get("correction_count") not in COUNTS or meta.get("validation_inputs") != 0:
        raise ValueError("Wrong training cohort count or split")
    if meta["train_inputs"] != len(meta["records"]) or sha(root / "input-spec.json") != meta["stored_spec_sha256"]:
        raise ValueError("Dataset manifest or input spec changed")
    for name, expected in {**meta["source_artifacts"], **meta["live_replay_context_hashes"]}.items():
        if sha(name) != expected:
            raise ValueError(f"Source changed after target preparation: {name}")
    if sha(root / meta["teacher_actions_file"]) != meta["teacher_actions_sha256"]:
        raise ValueError("Teacher action export changed")
    for record in meta["records"]:
        if record["split"] != "train" or sha(root / record["file"]) != record["sha256"]:
            raise ValueError("Training item changed")
        load_example(root, record)
    return root, meta


def load_example(root, record):
    with np.load(root / record["file"], allow_pickle=False) as z:
        inp = U.unflatten({key[6:]: z[key].copy() for key in z.files if key.startswith("input/")})
        target, physical = z["teacher_actions"].copy(), z["teacher_actions_physical"].copy()
        error = N.physical_errors(z["reconstructed_teacher_actions_physical"], physical)["max_abs_error"]
    if (target.shape != (15, 32) or not np.isfinite(target).all() or np.any(target[:, 8:] != 0)
            or error > ROUNDTRIP_TOLERANCE or error != record["normalization_roundtrip_max_abs"]):
        raise ValueError("Invalid normalized training label")
    return inp, target


def planned_steps(n):
    if type(n) is not int or n < 1:
        raise ValueError("At least one training chunk is required")
    return max(200, 2 * n)


def training_order(n, steps, seed=SEED):
    """Complete shuffled passes ensure full-trajectory coverage, even for long runs."""
    if n < 1 or steps < 1:
        raise ValueError("Positive item and step counts are required")
    rng = np.random.default_rng(seed)
    result = []
    while len(result) < steps:
        result.extend(rng.permutation(n).tolist())
    return result[:steps]


def fit(args):
    root, meta = load_dataset(args.dataset)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    n = len(meta["records"])
    steps = 1 if args.command == "smoke" else planned_steps(n)
    settings = {"arm": meta["arm"], "correction_count": meta["correction_count"], "layers": list(LAYERS),
                "rank": RANK, "steps": steps, "learning_rate": args.lr, "seed": SEED, "batch_size": 1,
                "weight_decay": 0, "gradient_clip": 1, "image_augmentation": False,
                "objective": "native flow matching on normalized native physical teacher action chunks",
                "selection": "Separate development closed-loop rollouts; no fitting-metric checkpoint selection",
                "scope": "Selected VLM MLP output projections only; all other weights including action expert frozen"}
    dump(out / "settings.json", settings)
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from openpi.models import model as om
    from weightsmith import config as C
    if sha(C.BASE_WS / "workspace.json") != meta["base_workspace_manifest_sha256"]:
        raise ValueError("Base workspace changed")
    started = time.monotonic()
    policy = U.load_policy()
    graph, state = nnx.split(policy._model)
    packed = state["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].value
    rng = np.random.default_rng(SEED)
    factors = {"a": jnp.asarray(rng.normal(0, .01, (len(LAYERS), packed.shape[1], RANK)).astype(np.float32)),
               "b": jnp.zeros((len(LAYERS), RANK, packed.shape[2]), dtype=jnp.float32)}
    if not bool(jnp.array_equal(U.injected_state(state, factors, LAYERS, jax, jnp)["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].value, packed)):
        raise ValueError("Zero-output LoRA does not reproduce the unchanged weights")
    optimizer = optax.chain(optax.clip_by_global_norm(1), optax.adamw(args.lr, weight_decay=0))
    opt_state = optimizer.init(factors)

    def loss(a, st, obs, target, key):
        return jnp.mean(nnx.merge(graph, U.injected_state(st, a, LAYERS, jax, jnp)).compute_loss(key, obs, target, train=False))

    gradient = jax.value_and_grad(loss)

    @jax.jit
    def step(a, opt, st, obs, target, key):
        value, grad = gradient(a, st, obs, target, key)
        updates, opt = optimizer.update(grad, opt, a)
        return optax.apply_updates(a, updates), opt, value, optax.global_norm(grad)

    order = training_order(n, steps)
    np.save(out / "training-order.npy", np.asarray(order, dtype=np.int32), allow_pickle=False)
    history, seen = [], set()
    for i, index in enumerate(order):
        inp, target = load_example(root, meta["records"][index])
        obs = om.Observation.from_dict(jax.tree.map(lambda x: jnp.asarray(x)[None], inp))
        factors, opt_state, value, norm = step(factors, opt_state, state, obs, jnp.asarray(target)[None], jax.random.key(SEED + i))
        value, norm = float(value), float(norm)
        if not np.isfinite(value + norm) or (i == 0 and norm <= 0):
            raise ValueError("Nonfinite flow loss/gradient or zero initial gradient")
        seen.add(index)
        if i == 0 or (i + 1) % 10 == 0 or i + 1 == steps:
            row = {"step": i + 1, "flow_loss": value, "gradient_norm": norm,
                   "unique_chunks_seen": len(seen), "wall_seconds": time.monotonic() - started}
            history.append(row)
            dump(out / "progress.json", {"stage": "fitting", **row, "planned_steps": steps})
            print(json.dumps(row), flush=True)
    report = {**settings, "cohort_id": meta["cohort_id"], "split": "development", "train_inputs": n,
              "validation_inputs": 0, "unique_chunks_seen": len(seen), "complete_passes": steps // n,
              "history": history, "zero_start_exact": True, "dataset_sha256": sha(root / "dataset.json"),
              "training_order_sha256": sha(out / "training-order.npy"),
              "trainable_parameters": sum(x.size for x in factors.values()), "memory_stats": jax.devices()[0].memory_stats(),
              "source_code_sha256": sha(__file__), "utility_sha256": sha(UTILITY),
              "wall_seconds": time.monotonic() - started, "closed_loop_evaluation": "not performed"}
    if args.command == "smoke":
        dump(out / "smoke.json", report)
    else:
        if len(seen) != n or steps // n < 2:
            raise ValueError("Full-dataset training coverage requirement failed")
        np.savez_compressed(out / "adapters.npz", **jax.tree.map(np.asarray, factors), layers=np.asarray(LAYERS))
        merged = U.injected_state(state, factors, LAYERS, jax, jnp)
        U.export_workspace(out, LAYERS, C.BASE_WS, packed,
            merged["PaliGemma"]["llm"]["layers"]["mlp"]["linear"].value)
        # The base export includes action modules, but this adapter never edits
        # them. Correct this new workspace's scope without changing old helpers.
        workspace_path = out / "workspace/workspace.json"
        workspace = json.loads(workspace_path.read_text())
        workspace["include_action"] = False
        workspace["editable_parameters"] = sum(entry["numel"] for entry in workspace["tensors"].values())
        dump(workspace_path, workspace)
        from weightsmith.lab.openpi_vlm_full import validate_workspace
        report["workspace_validation"] = validate_workspace(out / "workspace")
        report["wall_seconds"] = time.monotonic() - started
        dump(out / "training.json", report)
        files = [out / name for name in ("settings.json", "training.json", "training-order.npy", "adapters.npz", "workspace/workspace.json")]
        files.extend(sorted((out / "workspace/edits").glob("*.npy")))
        dump(out / "seal.json", {"candidate_id": f"lora-trajectory-n{meta['correction_count']:03d}-lr-{args.lr:g}",
             "learning_rate": args.lr, "correction_count": meta["correction_count"], "arm": meta["arm"],
             "workspace": str(out / "workspace"), "artifacts": {str(p): sha(p) for p in files}})
    dump(out / "progress.json", {"stage": "complete", "wall_seconds": report["wall_seconds"], "steps": steps})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--spec", required=True)
    prepare_parser.add_argument("--arm", choices=("full_trajectory", "matched_decision"), default="full_trajectory")
    prepare_parser.set_defaults(func=prepare)
    for command in ("smoke", "train"):
        fit_parser = sub.add_parser(command)
        fit_parser.add_argument("--dataset", required=True)
        fit_parser.add_argument("--lr", type=float, choices=LEARNING_RATES, default=0.0001)
        fit_parser.set_defaults(func=fit)
    for child in (prepare_parser, *[sub.choices[name] for name in ("smoke", "train")]):
        child.add_argument("--output", required=True)
        child.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES", str(args.gpu)) != str(args.gpu):
        parser.error("--gpu must identify the sole physical CUDA_VISIBLE_DEVICES device")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.command == "prepare":
        from weightsmith import config as C
        os.environ.update(JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE), JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS="1",
                          XLA_PYTHON_CLIENT_PREALLOCATE="true", XLA_PYTHON_CLIENT_MEM_FRACTION=C.SERVER_MEM_FRACTION)
    else:
        os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", ".90")
    from weightsmith.locks import GpuLock
    with GpuLock(args.gpu):
        args.func(args)


if __name__ == "__main__":
    main()
