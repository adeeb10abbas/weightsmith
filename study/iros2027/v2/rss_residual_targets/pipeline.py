"""CPU construction and frozen plans for an additive residual-target ablation.

No command in this module launches policy inference or writes to reused sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
PRIOR = HERE.parent / "rss_repair_comparison"
sys.path.insert(0, str(REPO))
from weightsmith import distill

LAYERS = tuple(range(8, 17))
METHODS = ("base", "residual_increment", "local_mlp", "shuffled_increment")
SEED = 2026111804


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(8 << 20):
            digest.update(block)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def verify_map(mapping):
    require(bool(mapping), "An empty identity is not a source seal")
    for path, expected in mapping.items():
        require(sha(path) == expected, f"Sealed source changed: {path}")


def within_camera_permutation(seed=SEED, cameras=2, cells=16):
    """First random derangement per camera; never chosen using policy outcomes."""
    require(cells >= 2, "A derangement requires at least two cells")
    rng = np.random.default_rng(seed)
    result = []
    for camera in range(cameras):
        for _ in range(10000):
            order = rng.permutation(cells)
            if np.all(order != np.arange(cells)):
                break
        else:
            raise RuntimeError("Could not construct the predeclared derangement")
        result.extend((order + camera * cells).tolist())
    return np.asarray(result, dtype=int)


def local_mlp_target(original_hidden, teacher_hidden, weights):
    """Project paired mean hidden differences through the unchanged linear map.

    Inputs are [repair observation, image cell, hidden feature]. Keeping the
    cell axis intact is essential: averaging it would erase correspondence.
    """
    a = np.asarray(original_hidden, dtype=np.float64)
    b = np.asarray(teacher_hidden, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    require(a.ndim == b.ndim == 3 and a.shape == b.shape and len(a) > 0,
            "Paired repair hidden arrays must have identical nonempty axes")
    require(w.ndim == 2 and a.shape[-1] == w.shape[0], "MLP weight/hidden mismatch")
    require(all(np.isfinite(x).all() for x in (a, b, w)), "Nonfinite target inputs")
    return (b - a).mean(axis=0) @ w


def target_variants(K, residual, local, permutation):
    require(K.ndim == residual.ndim == local.ndim == 2, "Targets and keys must be matrices")
    require(len(K) == len(residual) == len(local) == len(permutation), "Cell axes differ")
    require(residual.shape == local.shape, "Target output widths differ")
    require(sorted(permutation.tolist()) == list(range(len(K))), "Not a cell permutation")
    return {"residual_increment": residual, "local_mlp": local,
            "shuffled_increment": residual[permutation]}


def source_bundle(adaptation, case):
    """Validate development-only provenance; no earlier heldout file is read."""
    adaptation = Path(adaptation).resolve()
    require(adaptation.name == f"{case}-adaptation", "Expected the fixed development adaptation directory")
    fit = read(adaptation / "edit-fit/solve.json")
    prepared = adaptation / "edit-prepared"
    meta = read(prepared / "prepared.json")
    spec = read(adaptation / "spec.json")
    require(meta["split"] == spec["split"] == "development", "Heldout data cannot enter target construction")
    require(spec["case"] == case and fit["training_split"] == "train"
            and fit["validation_used_in_fit"] is False, "Unexpected fit scope")
    counts = Counter((r["split"], r["role"]) for r in meta["records"])
    require(counts == Counter({("train", "repair"): 12, ("train", "preservation"): 12,
                              ("validation", "repair"): 4, ("validation", "preservation"): 4}),
            "Require exactly the original 24 training and 8 validation observations")
    require(fit["pooling"] == "global" and fit["hold_weight"] == 1
            and fit["rank_cap"] == 32, "Source fit differs from the fixed ablation recipe")
    require(fit["replay_parity"]["status"] == fit["teacher_action_parity"]["status"] == "pass",
            "Source numerical identity checks did not pass")
    require(sha(adaptation / "spec.json") == fit["spec_sha256"], "Shared specification changed")
    require(sha(prepared / "prepared.json") == fit["prepared_sha256"], "Prepared inputs changed")
    require(sha(PRIOR / "edit/pipeline.py") == fit["code_sha256"], "The reused fit implementation changed")
    paths = [adaptation / "spec.json", adaptation / "candidate-seals.json",
             adaptation / "edit-fit/solve.json", prepared / "prepared.json",
             prepared / "input-spec.json", PRIOR / "protocol.json", PRIOR / "edit/pipeline.py",
             REPO / "weightsmith/distill.py", REPO / "weightsmith/acts_worker.py",
             REPO / "weightsmith/lab/openpi_probe.py"]
    for which in ("original", "teacher"):
        receipt = prepared / "captures" / f"{which}-receipt.json"
        require(receipt.exists(), "A capture without its immediate receipt is not reusable")
        paths += [receipt, prepared / f"{which}-capture-spec.json",
                  prepared / "captures" / which / "index.json",
                  prepared / "captures" / which / "acts.npz"]
        require(sha(paths[-1]) == fit["capture_sha256"][which], "Capture differs from original fitted candidate")
    base = Path(fit["base_workspace"])
    paths.append(base / "workspace.json")
    require(sha(base / "workspace.json") == fit["base_manifest_sha256"], "Base workspace changed")
    manifest = read(base / "workspace.json")
    expected = {distill.MLP_DOWN.format(layer) for layer in LAYERS}
    old_workspace = adaptation / "edit-fit/workspace"
    old_manifest = read(old_workspace / "workspace.json")
    require(set(old_manifest["tensors"]) == expected, "Source candidate has a different module footprint")
    seals = read(adaptation / "candidate-seals.json")["closed_form"]["artifacts"]
    verify_map(seals)
    for name in sorted(expected):
        paths += [base / "base" / manifest["tensors"][name]["file"],
                  old_workspace / "edits" / old_manifest["tensors"][name]["file"]]
    paths.append(old_workspace / "workspace.json")
    identities = {str(path.resolve()): sha(path) for path in paths}
    core = module("residual_targets_reused_fit", PRIOR / "edit/pipeline.py")
    checked_root, checked_meta = core.load_prepared(prepared)
    require(checked_meta == meta and checked_root == prepared, "Prepared-data validation disagrees")
    original = core.read_capture(prepared, meta, "original")
    teacher = core.read_capture(prepared, meta, "teacher")
    kv, groups = core.key_values(original, teacher, meta["records"], "global")
    require(len(groups) == 1, "This controlled study uses one instruction pair per case")
    return core, meta, original, teacher, kv, base, manifest, identities, fit


def build_case(adaptation, case, output):
    output = Path(output).resolve()
    require(not output.exists(), "Candidate output already exists; preserve previous attempts")
    started = time.time()
    output.mkdir(parents=True)
    state = {"case": case, "status": "preparing", "started_at": started,
             "new_policy_episodes": 0, "reuse_or_new": "reused sealed development captures; new CPU fits"}
    dump(output / "status.json", state)
    try:
        core, meta, original, teacher, kv, base, manifest, sources, old_fit = source_bundle(adaptation, case)
        sources[str(Path(__file__).resolve())] = sha(__file__)
        protocol_path = HERE / "protocol.json"
        sources[str(protocol_path)] = sha(protocol_path)
        protocol = read(protocol_path)
        permutation = np.asarray(protocol["target_recipe"]["cell_permutation"], dtype=int)
        require(np.array_equal(permutation, within_camera_permutation()), "Permutation differs from the fixed recipe")
        repair = [i for i, r in enumerate(meta["records"]) if r["split"] == "train" and r["role"] == "repair"]
        train_ids = [meta["records"][i]["item_id"] for i in repair]
        workspaces, reports = {}, {method: [] for method in METHODS if method != "base"}
        selected = {distill.MLP_DOWN.format(l): manifest["tensors"][distill.MLP_DOWN.format(l)] for l in LAYERS}
        for method in reports:
            workspace = output / method / "workspace"
            (workspace / "edits").mkdir(parents=True)
            (workspace / "scratch").mkdir()
            (workspace / "base").symlink_to(base / "base", target_is_directory=True)
            cfg = {**manifest, "tensors": selected, "max_changed_fraction": 1.0,
                   "max_changed_elements": manifest["total_parameters_and_buffers"]}
            dump(workspace / "workspace.json", cfg)
            workspaces[method] = workspace
        state["status"] = "fitting"
        dump(output / "status.json", state)
        for layer in LAYERS:
            name = distill.MLP_DOWN.format(layer)
            entry = selected[name]
            W = np.load(base / "base" / entry["file"], allow_pickle=False)
            li = core.CAPTURE_LAYERS.index(layer)
            hidden = [arr[core.ARRAY_KEYS[0]][repair, li, :, 1:, :].reshape(len(repair), 32, -1)
                      for arr in (original, teacher)]
            local = local_mlp_target(*hidden, W)
            K, V, H = kv[layer]
            variants = target_variants(K, V, local, permutation)
            for method, target in variants.items():
                fit_started = time.time()
                delta, stats = core.fit(K, target, H, hold_weight=1, rank=32, lam_rel=1e-3)
                edited = distill.bf16_round(W.astype(np.float64) + delta).astype(np.float32)
                destination = workspaces[method] / "edits" / entry["file"]
                np.save(destination, edited, allow_pickle=False)
                if method == "residual_increment":
                    prior = Path(adaptation) / "edit-fit/workspace/edits" / entry["file"]
                    require(np.array_equal(edited, np.load(prior, allow_pickle=False)),
                            f"Residual reconstruction differs from the sealed original at layer {layer}")
                denom = max(float(np.linalg.norm(V)), 1e-30)
                reports[method].append({**stats, "layer": layer, "tensor": name,
                    "target_frobenius": float(np.linalg.norm(target)), "residual_target_frobenius": float(np.linalg.norm(V)),
                    "target_difference_from_residual_relative": float(np.linalg.norm(target - V) / denom),
                    "target_cosine_to_residual": float(np.vdot(target, V) / max(np.linalg.norm(target) * np.linalg.norm(V), 1e-30)),
                    "update_frobenius_before_rounding": float(np.linalg.norm(delta)),
                    "merged_difference_frobenius": float(np.linalg.norm(edited.astype(np.float64) - W)),
                    "changed_bf16_values": int(np.count_nonzero(edited != W)),
                    "base_sha256": sha(base / "base" / entry["file"]), "edited_sha256": sha(destination),
                    "wall_seconds": time.time() - fit_started})
                print(json.dumps({"case": case, "layer": layer, "method": method, "rank": stats["rank"]}), flush=True)
            del hidden, W
        verify_map(sources)
        artifacts = {}
        for method, workspace in workspaces.items():
            report = {"case": case, "method": method, "split": "development", "validation_used_in_fit": False,
                      "reuse_or_new": "reused observations and frozen base; new CPU construction",
                      "workspace": str(workspace), "layers": reports[method], "training_repair_item_ids": train_ids,
                      "source_sha256": sources, "cell_permutation": permutation.tolist() if method == "shuffled_increment" else None,
                      "rank_note": "Rank32 before BF16 merging; rounded dense differences can have larger algebraic rank.",
                      "target_note": protocol["target_recipe"]["definitions"][method]}
            dump(output / method / "fit.json", report)
            artifacts[method] = {str(p): sha(p) for p in
                                [output / method / "fit.json", workspace / "workspace.json", *sorted((workspace / "edits").glob("*.npy"))]}
        dump(output / "candidate-seals.json", artifacts)
        state.update(status="complete", wall_seconds=time.time() - started,
                     candidate_seals_sha256=sha(output / "candidate-seals.json"),
                     original_residual_candidate_byte_identical=True)
    except BaseException as exc:
        state.update(status="stopped", reason=repr(exc), wall_seconds=time.time() - started)
        raise
    finally:
        dump(output / "status.json", state)
    return state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fit = sub.add_parser("fit")
    fit.add_argument("--adaptation-root", type=Path, required=True)
    fit.add_argument("--case", choices=("temporal", "spatial"), required=True)
    fit.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "fit":
        print(json.dumps(build_case(args.adaptation_root, args.case, args.output), indent=2))


if __name__ == "__main__":
    main()
