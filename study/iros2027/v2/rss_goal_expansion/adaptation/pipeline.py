"""Additive closed-form repair from the same development inputs as action LoRA.

prepare and solve are CPU-only. capture explicitly launches the existing GPU
activation worker. No historical observation, capture, or weight file is modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
from weightsmith import distill

LAYERS = tuple(range(8, 17))
CAPTURE_LAYERS = tuple(range(7, 17))
ARRAY_KEYS = ("vlm.img_mlp_hidden", "vlm.img_resid_out")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while block := f.read(8 << 20):
            h.update(block)
    return h.hexdigest()


def dump(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def input_digest(request, seed, prompt):
    """Hash exact sensor/state arrays, sampling seed, and instruction (no paths)."""
    h = hashlib.sha256()
    for key in sorted(request):
        value = np.ascontiguousarray(request[key])
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.tobytes())
    h.update(json.dumps({"seed": seed, "prompt": prompt}, sort_keys=True).encode())
    return h.hexdigest()


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get("split") != "development" or not spec.get("cohort_id"):
        raise ValueError("Explicit development split and cohort_id are required")
    if not spec.get("items"):
        raise ValueError("At least one item is required")
    identities, ids = set(), set()
    paths = {"train": set(), "validation": set()}
    records = []
    for i, raw in enumerate(spec["items"]):
        it = dict(raw)
        split = it.get("split")
        if split not in paths:
            raise ValueError("Every item must be train or validation")
        p = Path(it["obs_path"]).resolve()
        if any(x in {"heldout", "held-out"} for x in p.parts):
            raise ValueError(f"Held-out data refused: {p}")
        k = it.get("decision_index")
        if type(k) is not int or k < 0:
            raise ValueError("decision_index must be a nonnegative integer")
        identity = (str(p), k)
        if identity in identities:
            raise ValueError(f"Duplicate observation decision: {identity}")
        identities.add(identity)
        paths[split].add(str(p))
        it.update(obs_path=str(p), item_id=it.get("item_id", f"item-{i:04d}"),
                  role=it.get("role", "repair"))
        if it["item_id"] in ids:
            raise ValueError("Duplicate item_id")
        ids.add(it["item_id"])
        if it["role"] not in {"repair", "preservation"}:
            raise ValueError("role must be repair or preservation")
        for field in ("original_prompt", "teacher_prompt"):
            it[field] = it.get(field, spec.get(field))
            if not isinstance(it[field], str) or not it[field].strip():
                raise ValueError(f"{field} required globally or per item")
        it["recorded_prompt"] = it.get("recorded_prompt", spec.get("recorded_prompt"))
        if it["role"] == "preservation" and it["original_prompt"] != it["teacher_prompt"]:
            raise ValueError("Preservation must have teacher_prompt == original_prompt for both methods")
        records.append(it)
    if not all(paths.values()) or paths["train"] & paths["validation"]:
        raise ValueError("Train and validation require disjoint source episodes")
    if not any(r["role"] == "repair" and r["split"] == "train" for r in records):
        raise ValueError("No training repair decisions")
    return spec, records


def prepare(args):
    started = time.time()
    spec, records = read_spec(args.spec)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "episodes").mkdir()
    dump(out / "input-spec.json", spec)
    hashes = {}
    for i, it in enumerate(records):
        p = Path(it["obs_path"])
        if str(p) not in hashes:
            hashes[str(p)] = sha(p)
        k = it["decision_index"]
        with np.load(p, allow_pickle=False) as z:
            if k >= len(z["seed"]):
                raise IndexError((str(p), k))
            seed = int(z["seed"][k])
            if seed < 0:
                raise ValueError("A recorded deterministic sampling seed is required")
            request = {key: z[key][k] for key in ("base_img", "wrist_img", "joint", "gripper")}
            if not np.isfinite(z["actions"][k]).all():
                raise ValueError("Nonfinite recorded action")
            sim_step = int(z["step"][k])
        # Each decision gets its own summary; mixed-prompt traces cannot silently
        # attribute parity to the wrong recorded instruction.
        eid = f"item-{i:04d}"
        ep = out / "episodes" / eid
        ep.mkdir()
        (ep / "obs.npz").symlink_to(p)
        if it["recorded_prompt"]:
            dump(ep / "summary.json", {"prompt": it["recorded_prompt"]})
        it.update(episode=eid, policy_seed=seed, sim_step=sim_step,
                  source_observation_sha256=hashes[str(p)],
                  original_input_sha256=input_digest(request, seed, it["original_prompt"]),
                  teacher_input_sha256=input_digest(request, seed, it["teacher_prompt"]),
                  recorded_input_sha256=input_digest(request, seed, it["recorded_prompt"]) if it["recorded_prompt"] else None)
    common = {"cand": "base", "episodes_dir": str(out / "episodes"),
              "vlm_layers": list(CAPTURE_LAYERS), "image_tokens": True,
              "prompt_tokens": [0], "action_layers": [0], "denoise_steps": [0],
              "action_tokens": "mean", "action_attention": False}
    for which in ("original", "teacher"):
        capture_spec = {**common, "task_instruction": records[0][f"{which}_prompt"],
                        "prompt": records[0][f"{which}_prompt"],
                        "items": [{"episode": r["episode"], "k": r["decision_index"],
                                   "prompt": r[f"{which}_prompt"]} for r in records]}
        dump(out / f"{which}-capture-spec.json", capture_spec)
    report = {"cohort_id": spec["cohort_id"], "split": "development", "records": records,
              "spec_sha256": sha(args.spec), "stored_spec_sha256": sha(out / "input-spec.json"),
              "capture_spec_sha256": {w: sha(out / f"{w}-capture-spec.json") for w in ("original", "teacher")},
              "source_observation_sha256": hashes,
              "capture_layers": list(CAPTURE_LAYERS), "code_sha256": sha(__file__),
              "wall_seconds": time.time() - started}
    dump(out / "prepared.json", report)
    print(json.dumps({"prepared": str(out), "items": len(records), "gpu_used": False}))


def load_prepared(path):
    root = Path(path).resolve()
    meta = json.loads((root / "prepared.json").read_text())
    if meta["split"] != "development" or sha(root / "input-spec.json") != meta["stored_spec_sha256"]:
        raise ValueError("Invalid prepared dataset")
    # The copied JSON has canonical formatting; compare semantic content rather
    # than requiring its byte hash to equal the original source manifest.
    _, records = read_spec(root / "input-spec.json")
    if len(records) != len(meta["records"]):
        raise ValueError("Prepared metadata changed")
    for old, new in zip(meta["records"], records, strict=True):
        if any(old.get(k) != v for k, v in new.items()):
            raise ValueError("Prepared item content differs from frozen spec")
    for w, expected in meta["capture_spec_sha256"].items():
        if sha(root / f"{w}-capture-spec.json") != expected:
            raise ValueError("Capture request changed after freeze")
    for p, expected in meta["source_observation_sha256"].items():
        if sha(p) != expected:
            raise ValueError(f"Observation changed after freeze: {p}")
    return root, meta


def capture(args):
    from weightsmith import config as C
    from weightsmith.locks import GpuLock
    root, meta = load_prepared(args.prepared)
    certified_live_replay(root, meta, args.lora_dataset)
    dest = root / "captures"
    dest.mkdir(exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(args.gpu),
               JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE),
               XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=C.ACTS_MEM_FRACTION)
    with GpuLock(args.gpu):
        for which in ("original", "teacher"):
            result = dest / which
            receipt = dest / f"{which}-receipt.json"
            specpath = root / f"{which}-capture-spec.json"
            if receipt.exists():
                old = json.loads(receipt.read_text())
                if old["spec_sha256"] != sha(specpath) or old["acts_sha256"] != sha(result / "acts.npz"):
                    raise ValueError("Existing capture identity mismatch")
                continue
            if result.exists():
                raise FileExistsError(f"Incomplete capture preserved for inspection: {result}")
            start = time.time()
            cmd = [args.python or C.OPENPI_PY, "-m", "weightsmith.acts_worker", "--output", str(result)]
            with (dest / f"{which}.log").open("w") as log:
                run = subprocess.run(cmd, input=specpath.read_text(), text=True, cwd=REPO,
                                     env=env, stdout=log, stderr=subprocess.STDOUT)
            if run.returncode:
                raise RuntimeError(f"Capture failed ({run.returncode}); see {dest / (which + '.log')}")
            index = json.loads((result / "index.json").read_text())
            if len(index["items"]) != len(meta["records"]):
                raise ValueError("Capture dropped items")
            dump(receipt, {"spec_sha256": sha(specpath), "acts_sha256": sha(result / "acts.npz"),
                           "index_sha256": sha(result / "index.json"), "wall_seconds": time.time() - start,
                           "worker_sha256": sha(REPO / "weightsmith/acts_worker.py"),
                           "checkpoint": str(C.CHECKPOINT), "base_workspace": str(C.BASE_WS),
                           "base_manifest_sha256": sha(C.BASE_WS / "workspace.json"),
                           "items": len(index["items"])})
            print(f"Captured {which}: {len(index['items'])} development inputs", flush=True)
    singleton_drift_report(root, meta)
    teacher_parity(root, meta, args.lora_dataset, args.parity_tolerance)


def read_capture(root, meta, which):
    path = root / "captures" / which
    receipt = root / "captures" / f"{which}-receipt.json"
    if receipt.exists():
        got = json.loads(receipt.read_text())
        if got["acts_sha256"] != sha(path / "acts.npz") or got["index_sha256"] != sha(path / "index.json"):
            raise ValueError("Capture changed after receipt")
    index = json.loads((path / "index.json").read_text())
    if index["vlm_layers"] != list(CAPTURE_LAYERS) or len(index["items"]) != len(meta["records"]):
        raise ValueError("Capture axes differ from frozen spec")
    for got, want in zip(index["items"], meta["records"], strict=True):
        if (got["episode"], got["k"], got["seed"], got["prompt"]) != (
                want["episode"], want["decision_index"], want["policy_seed"], want[f"{which}_prompt"]):
            raise ValueError("Capture input order or identity differs")
    with np.load(path / "acts.npz", allow_pickle=False) as z:
        arrays = {key: z[key].copy() for key in (*ARRAY_KEYS, "actions", "recorded_actions")}
    if not all(np.isfinite(v).all() for v in arrays.values()):
        raise ValueError("Nonfinite activation or action")
    return arrays


def singleton_drift_report(root, meta):
    rows = []
    for which in ("original", "teacher"):
        arrays = read_capture(root, meta, which)
        for i, r in enumerate(meta["records"]):
            if r.get("recorded_prompt") != r[f"{which}_prompt"]:
                continue
            a, b = arrays["actions"][i], arrays["recorded_actions"][i]
            rows.append({"item_id": r["item_id"], "prompt_kind": which,
                         "bit_exact": bool(np.array_equal(a, b)), "max_abs_error": float(np.max(np.abs(a - b))),
                         "per_coordinate_max_abs_error": np.max(np.abs(a - b), axis=0).tolist()})
    report = {"comparisons": rows,
              "max_abs_error": max((r["max_abs_error"] for r in rows), default=None),
              "status": "diagnostic_only", "used_as_acceptance_gate": False,
              "note": "Singleton instrumented versus live-bucket recorded actions. BF16 batch/graph kernels differ; exact pool replay is the separate acceptance gate."}
    dump(root / "singleton-drift.json", report)
    return report


def shared_lora_dataset(meta, dataset_path):
    data = Path(dataset_path).resolve()
    other = json.loads((data / "dataset.json").read_text())
    if other["cohort_id"] != meta["cohort_id"] or other["spec_sha256"] != meta["spec_sha256"]:
        raise ValueError("LoRA and closed-form development manifests differ")
    if other["source_observation_sha256"] != meta["source_observation_sha256"]:
        raise ValueError("LoRA and closed-form source observations differ")
    for ours, theirs in zip(meta["records"], other["records"], strict=True):
        for key in ("item_id", "obs_path", "decision_index", "policy_seed", "original_prompt", "teacher_prompt",
                    "recorded_prompt", "split", "role", "source_observation_sha256", "original_input_sha256",
                    "teacher_input_sha256", "recorded_input_sha256", "source_env_id", "source_run_id", "source_step",
                    "live_batch_size", "live_batch_bucket", "live_batch_members", "live_context", "live_context_sha256"):
            if ours[key] != theirs[key]:
                raise ValueError(f"Teacher input mismatch in {key}")
        file = data / theirs["file"]
        if sha(file) != theirs["sha256"]:
            raise ValueError("LoRA teacher file changed")
    return data, other


def certified_live_replay(root, meta, dataset_path):
    """Require the independent exact-pool replay certificate for these inputs."""
    data, other = shared_lora_dataset(meta, dataset_path)
    gate = other.get("live_replay_gate", {})
    if (gate.get("mode") != "exact_pool_bucket" or gate.get("status") != "pass"
            or gate.get("tolerance") != 1e-6 or gate.get("items") != len(meta["records"])):
        raise ValueError("An exact live-pool replay certificate at 1e-6 is required")
    rows = []
    for ours, theirs in zip(meta["records"], other["records"], strict=True):
        context_hash = hashlib.sha256(json.dumps(ours["live_context"], sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if context_hash != ours["live_context_sha256"]:
            raise ValueError("Live replay context checksum mismatch")
        error = theirs.get("live_replay_max_abs_error")
        if error is None or not np.isfinite(error) or error > 1e-6 or error < 0:
            raise ValueError("Live pool replay did not pass for every selected input")
        with np.load(data / theirs["file"], allow_pickle=False) as z:
            live = z["live_replay_actions_physical"].astype(np.float32)
            recorded = z["recorded_actions_physical"].astype(np.float32)
        with np.load(ours["obs_path"], allow_pickle=False) as z:
            source_actions = z["actions"][ours["decision_index"]].astype(np.float32)
        if (live.shape != source_actions.shape or recorded.shape != source_actions.shape
                or not np.isfinite(live).all() or not np.array_equal(recorded, source_actions)):
            raise ValueError("Live replay certificate actions differ from the recorded source")
        computed_error = float(np.max(np.abs(live - source_actions)))
        exact = bool(np.array_equal(live, source_actions))
        if computed_error != error or exact != theirs["live_replay_bit_exact"]:
            raise ValueError("Live replay action arrays differ from their certificate")
        rows.append({"item_id": ours["item_id"], "max_abs_error": error,
                     "bit_exact": bool(theirs["live_replay_bit_exact"]),
                     "recorded_input_sha256": ours["recorded_input_sha256"],
                     "live_context_sha256": ours["live_context_sha256"],
                     "live_batch_size": ours["live_batch_size"], "live_batch_bucket": ours["live_batch_bucket"]})
    maximum = max(r["max_abs_error"] for r in rows)
    if not np.isfinite(gate.get("maximum_error", float("nan"))) or gate["maximum_error"] != maximum:
        raise ValueError("Live replay gate summary differs from its individual certificates")
    report = {"status": "pass", "mode": "exact_pool_bucket", "tolerance": 1e-6,
              "dataset": str(data), "dataset_sha256": sha(data / "dataset.json"),
              "max_abs_error": maximum, "all_bit_exact": all(r["bit_exact"] for r in rows),
              "comparisons": rows,
              "note": "Exact live-bucket server replay on identical recorded inputs and seeds, compared in recorded action dtype. Singleton drift is diagnostic only; no acceptance tolerance was inflated."}
    dump(root / "replay-parity.json", report)
    return report


def teacher_parity(root, meta, dataset_path, tolerance):
    if tolerance != 0.01:
        raise ValueError("Singleton teacher parity tolerance is fixed at 0.01; no threshold inflation")
    data, other = shared_lora_dataset(meta, dataset_path)
    acts = read_capture(root, meta, "teacher")["actions"]
    rows = []
    for i, (ours, theirs) in enumerate(zip(meta["records"], other["records"], strict=True)):
        file = data / theirs["file"]
        with np.load(file, allow_pickle=False) as z:
            target = z["teacher_actions_physical"]
        if target.shape != acts[i].shape or not np.isfinite(target).all():
            raise ValueError("Teacher action shape or finiteness mismatch")
        rows.append({"item_id": ours["item_id"], "bit_exact": bool(np.array_equal(acts[i], target)),
                     "max_abs_error": float(np.max(np.abs(acts[i] - target))),
                     "per_coordinate_max_abs_error": np.max(np.abs(acts[i] - target), axis=0).tolist()})
    report = {"dataset": str(data), "dataset_sha256": sha(data / "dataset.json"),
              "comparisons": rows, "tolerance": tolerance,
              "all_bit_exact": all(r["bit_exact"] for r in rows),
              "max_abs_error": max(r["max_abs_error"] for r in rows),
              "status": "pass" if all(r["max_abs_error"] <= tolerance for r in rows) else "fail",
              "note": "Same seed, physical action units, native LoRA teacher versus instrumented activation worker."}
    dump(root / "teacher-parity.json", report)
    if report["status"] != "pass":
        raise ValueError("LoRA teacher versus activation replay exceeds tolerance")
    return report


def key_values(original, teacher, records, pooling):
    repair = [i for i, r in enumerate(records) if r["split"] == "train" and r["role"] == "repair"]
    groups = {}
    for i in repair:
        r = records[i]
        # Never average different instruction pairs into one target.
        group = (r["original_prompt"], r["teacher_prompt"], r.get("phase", "all") if pooling == "phase" else "all")
        if pooling == "phase" and not r.get("phase"):
            raise ValueError("Phase pooling requires phase on each training repair item")
        groups.setdefault(group, []).append(i)
    pairs = []
    for indices in groups.values():
        o = {k: original[k][indices] for k in ARRAY_KEYS}
        t = {k: teacher[k][indices] for k in ARRAY_KEYS}
        pairs.append(distill.build_keys_values(o, t, layers=LAYERS, mode="image", alpha=1.0,
                                              cells=distill.IMAGE_CELLS, capture_layers=CAPTURE_LAYERS))
    hold = [i for i, r in enumerate(records) if r["split"] == "train" and r["role"] == "preservation"]
    out = {}
    for layer in LAYERS:
        K = np.concatenate([p[layer][0] for p in pairs])
        V = np.concatenate([p[layer][1] for p in pairs])
        if hold:
            li = CAPTURE_LAYERS.index(layer)
            H = original[ARRAY_KEYS[0]][hold, li, :, 1:, :].astype(np.float64).reshape(-1, K.shape[1])
        else:
            H = np.empty((0, K.shape[1]), dtype=np.float64)
        out[layer] = K, V, H
    return out, [{"original_prompt": g[0], "teacher_prompt": g[1], "phase": g[2],
                  "item_ids": [records[i]["item_id"] for i in indices]} for g, indices in groups.items()]


def fit(K, V, H, hold_weight=1.0, rank=32, lam_rel=1e-3):
    """Ridge solve with zero preservation targets; cap total rank across phases."""
    if hold_weight <= 0 or rank <= 0 or lam_rel <= 0:
        raise ValueError("Positive hold weight, rank, and ridge required")
    if len(H) % len(distill.IMAGE_CELLS):
        raise ValueError("Preservation keys must contain all 32 cells per decision")
    hold_items = len(H) // len(distill.IMAGE_CELLS)
    # A repair key averages observations. Preserve each decision separately,
    # while averaging its squared penalty across decisions: no cancellation and
    # no implicit multiplier from the number of preservation observations.
    hold_scale = np.sqrt(hold_weight / max(1, hold_items))
    A = np.concatenate([K, hold_scale * H])
    B = np.concatenate([V, np.zeros((len(H), V.shape[1]))])
    gram = A @ A.T
    # Fix the ridge from repair keys only, as in the prior SGDW hold-key solve.
    # Adding or duplicating preservation examples must not change regularization.
    lam = lam_rel * np.sum(K * K) / len(K)
    if not lam > 0:
        raise ValueError("All-zero edit keys")
    coefficient = np.linalg.solve(gram + lam * np.eye(len(gram)), B)
    factors = distill.low_rank((A.T, coefficient), tol=1e-6, max_rank=rank)
    delta = factors.dense()
    return delta, {"repair_keys": len(K), "preservation_keys": len(H), "rank": factors.rank,
                   "rank_cap": rank, "rank_truncation_relative_error": factors.rel_error,
                   "preservation_items": hold_items, "preservation_key_scale": float(hold_scale),
                   "preservation_loss_normalization": "mean over preservation decisions; all 32 individual cell keys retained",
                   "ridge_reference": "mean squared norm of repair keys only; invariant to preservation count",
                   "ridge_absolute": float(lam), "ridge_relative": lam_rel,
                   "repair_fit_relative_error": float(np.linalg.norm(K @ delta - V) / max(np.linalg.norm(V), 1e-30)),
                   "preservation_output_shift_l2": float(np.linalg.norm(H @ delta)),
                   "preservation_output_shift_l2_per_item_rms": float(np.linalg.norm(H @ delta) / np.sqrt(max(1, hold_items)))}


def solve(args):
    started = time.time()
    from weightsmith import config as C
    from weightsmith.lab.openpi_vlm_full import validate_workspace
    root, meta = load_prepared(args.prepared)
    original = read_capture(root, meta, "original")
    teacher = read_capture(root, meta, "teacher")
    parity = certified_live_replay(root, meta, args.lora_dataset)
    drift = singleton_drift_report(root, meta)
    comparison = teacher_parity(root, meta, args.lora_dataset, args.parity_tolerance)
    kv, groups = key_values(original, teacher, meta["records"], args.pooling)
    base = Path(args.base_workspace or C.BASE_WS).resolve()
    for which in ("original", "teacher"):
        receipt_path = root / "captures" / f"{which}-receipt.json"
        if receipt_path.exists():
            receipt = json.loads(receipt_path.read_text())
            if receipt["base_manifest_sha256"] != sha(base / "workspace.json"):
                raise ValueError("Capture and export base-workspace identity differ")
    cfg = json.loads((base / "workspace.json").read_text())
    selected = {distill.MLP_DOWN.format(l): cfg["tensors"][distill.MLP_DOWN.format(l)] for l in LAYERS}
    cfg.update(tensors=selected, max_changed_fraction=1.0,
               max_changed_elements=cfg["total_parameters_and_buffers"])
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    workspace = out / "workspace"
    workspace.mkdir()
    (workspace / "edits").mkdir()
    (workspace / "scratch").mkdir()
    (workspace / "base").symlink_to(base / "base", target_is_directory=True)
    dump(workspace / "workspace.json", cfg)
    start, rows = time.time(), []
    for layer in LAYERS:
        delta, stats = fit(*kv[layer], hold_weight=args.hold_weight)
        name = distill.MLP_DOWN.format(layer)
        entry = selected[name]
        source = base / "base" / entry["file"]
        W = np.load(source, allow_pickle=False)
        if W.shape != delta.shape:
            raise ValueError(f"Weight/activation shape mismatch at layer {layer}")
        edited = distill.bf16_round(W.astype(np.float64) + delta).astype(np.float32)
        target = workspace / "edits" / entry["file"]
        np.save(target, edited, allow_pickle=False)
        rows.append({**stats, "layer": layer, "tensor": name,
                     "base_sha256": sha(source), "edited_sha256": sha(target),
                     "changed_values": int(np.count_nonzero(edited != W)),
                     "delta_relative_l2": float(np.linalg.norm(delta) / max(np.linalg.norm(W), 1e-30))})
        print(json.dumps({"layer": layer, "rank": stats["rank"], "repair_keys": stats["repair_keys"],
                          "preservation_keys": stats["preservation_keys"]}), flush=True)
    report = {"cohort_id": meta["cohort_id"], "method": "closed-form image-cell instruction distillation",
              "training_split": "train", "validation_used_in_fit": False,
              "spec_sha256": meta["spec_sha256"], "source_observation_sha256": meta["source_observation_sha256"],
              "base_workspace": str(base), "base_manifest_sha256": sha(base / "workspace.json"),
              "prepared_sha256": sha(root / "prepared.json"), "code_sha256": sha(__file__),
              "groups": groups, "pooling": args.pooling, "hold_weight": args.hold_weight,
              "rank_cap": 32, "layers": rows, "workspace": str(workspace),
              "workspace_validation": validate_workspace(workspace), "replay_parity": parity,
              "singleton_recorded_drift": drift,
              "teacher_action_parity": comparison, "solve_export_wall_seconds": time.time() - start,
              "wall_seconds": time.time() - started,
              "capture_sha256": {w: sha(root / "captures" / w / "acts.npz") for w in ("original", "teacher")},
              "note": "bf16 merged tensors. Rank cap applies before bf16 rounding; rounding can increase algebraic rank."}
    dump(out / "solve.json", report)
    print(json.dumps({"workspace": str(workspace), "solve": str(out / "solve.json")}))


def parity_command(args):
    root, meta = load_prepared(args.prepared)
    print(json.dumps(teacher_parity(root, meta, args.lora_dataset, args.parity_tolerance)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="CPU: freeze/hash shared development inputs and build worker specs")
    p.add_argument("--spec", required=True); p.add_argument("--output", required=True); p.set_defaults(func=prepare)
    p = sub.add_parser("capture", help="GPU: original and teacher activation capture on the exact frozen inputs")
    p.add_argument("--prepared", required=True); p.add_argument("--gpu", type=int, required=True)
    p.add_argument("--lora-dataset", required=True)
    p.add_argument("--python"); p.add_argument("--parity-tolerance", type=float, default=0.01); p.set_defaults(func=capture)
    p = sub.add_parser("solve", help="CPU: fit train only and export a bf16 sparse merged workspace")
    p.add_argument("--prepared", required=True); p.add_argument("--output", required=True)
    p.add_argument("--base-workspace"); p.add_argument("--lora-dataset", required=True)
    p.add_argument("--pooling", choices=("global", "phase"), default="global")
    p.add_argument("--hold-weight", type=float, default=1.0)
    p.add_argument("--parity-tolerance", type=float, default=0.01); p.set_defaults(func=solve)
    p = sub.add_parser("parity", help="CPU: compare instrumented teacher actions with native LoRA targets")
    p.add_argument("--prepared", required=True); p.add_argument("--lora-dataset", required=True)
    p.add_argument("--parity-tolerance", type=float, default=0.01); p.set_defaults(func=parity_command)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
