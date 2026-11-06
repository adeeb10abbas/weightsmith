# Vendored from the author's earlier project `weight-space-policy-lab`
# (src/weight_space_policy_lab/openpi_vlm_full.py), trimmed to the functions this harness uses.
# Logic is unchanged. See NOTICE.
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .openpi_head import _effective_bfloat16, _file_name, _parameter_count

LOGGER = logging.getLogger(__name__)
IMAGE_PREFIX = ("PaliGemma", "img", "Transformer", "encoderblock")
LANGUAGE_PREFIX = ("PaliGemma", "llm", "layers")
DEFAULT_MAX_CHANGED_FRACTION = 2e-4
DEFAULT_SANDBOX_MEMORY_BYTES = 96 << 30
DEFAULT_SANDBOX_TIMEOUT_SECONDS = 900
ACTION_DOWN_PATH = (*LANGUAGE_PREFIX, "mlp_1", "linear")
ACTION_DOWN_NAME = re.compile(r"action\.blocks\.(\d{2})\.mlp_1\.linear\Z")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _walk_arrays(tree: Any, path: tuple[str, ...] = ()):
    if isinstance(tree, Mapping):
        for key, value in tree.items():
            yield from _walk_arrays(value, (*path, str(key)))
        return
    yield path, np.asarray(tree)


def _exposed_depth(path: tuple[str, ...]) -> int | None:
    """Return the scan depth for a paper-style internal VLM tensor, else None."""
    if path[: len(IMAGE_PREFIX)] == IMAGE_PREFIX:
        return 27
    if path[: len(LANGUAGE_PREFIX)] != LANGUAGE_PREFIX:
        return None
    # Every component with the `_1` suffix belongs to the separate action expert.
    if any(component.endswith("_1") for component in path[len(LANGUAGE_PREFIX) :]):
        return None
    return 18


def _logical_name(path: tuple[str, ...], layer: int) -> str:
    depth = _exposed_depth(path)
    if depth is None:
        raise ValueError(f"Path is outside the editable VLM internals: {'/'.join(path)}")
    if not 0 <= layer < depth:
        raise IndexError(layer)
    if path[: len(IMAGE_PREFIX)] == IMAGE_PREFIX:
        suffix = path[len(IMAGE_PREFIX) :]
        return f"vision.blocks.{layer:02d}." + ".".join(suffix)
    suffix = path[len(LANGUAGE_PREFIX) :]
    return f"language.blocks.{layer:02d}." + ".".join(suffix)


def _editable_layout(path: tuple[str, ...], include_action: bool = False):
    """Return logical names and scan indices; None means a whole unscanned array."""
    depth = _exposed_depth(path)
    if depth is not None:
        return [(_logical_name(path, layer), layer) for layer in range(depth)]
    if not include_action:
        return []
    if path[: len(LANGUAGE_PREFIX)] == LANGUAGE_PREFIX:
        suffix = ".".join(path[len(LANGUAGE_PREFIX) :])
        return [(f"action.blocks.{layer:02d}.{suffix}", layer) for layer in range(18)]
    if path[:3] == ("PaliGemma", "llm", "final_norm_1"):
        return [("action.final_norm." + ".".join(path[3:]), None)]
    if path[0] in {"action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"}:
        return [("action." + ".".join(path), None)]
    return []


def _compact_stats(array: np.ndarray) -> dict[str, Any]:
    value = np.asarray(array, dtype=np.float32)
    return {
        "shape": list(value.shape),
        "dtype": "float32-effective-bfloat16",
        "numel": int(value.size),
        "min": float(value.min()),
        "max": float(value.max()),
        "mean": float(value.mean(dtype=np.float64)),
        "std": float(value.std(dtype=np.float64)),
    }


def _update_array_digest(digest: Any, name: str, array: np.ndarray) -> None:
    value = np.ascontiguousarray(array)
    digest.update(name.encode())
    digest.update(str(value.dtype).encode())
    digest.update(json.dumps(list(value.shape)).encode())
    digest.update(value.tobytes())


def prepare_workspace(
    checkpoint: Path,
    workspace: Path,
    max_changed_fraction: float = DEFAULT_MAX_CHANGED_FRACTION,
    include_action: bool = False,
    *,
    action_down_only: bool = False,
    max_changed_rows: int = 128,
    max_changed_layers: int = 2,
) -> dict[str, Any]:
    """Export every transformer-block VLM tensor, split into human-readable layers."""
    if workspace.exists() and any(workspace.iterdir()):
        raise FileExistsError(f"Refusing to reuse non-empty workspace: {workspace}")
    if not 0 < max_changed_fraction <= 1:
        raise ValueError("max_changed_fraction must be in (0, 1]")
    if action_down_only:
        if not 1 <= max_changed_rows <= 128 or not 1 <= max_changed_layers <= 2:
            raise ValueError("Study limits are 1..128 changed rows and 1..2 changed layers")
        include_action = True
    workspace.mkdir(parents=True, exist_ok=True)
    base_dir = workspace / "base"
    base_dir.mkdir()
    (workspace / "edits").mkdir()
    (workspace / "scratch").mkdir()

    from openpi.models import model as openpi_model

    LOGGER.info("Restoring checkpoint for full internal-VLM export")
    params = openpi_model.restore_params(checkpoint / "params", restore_type=np.ndarray)
    tensors: dict[str, dict[str, Any]] = {}
    editable_parameters = 0
    digest = hashlib.sha256()
    packed_tensor_count = 0
    for path, packed in _walk_arrays(params):
        if action_down_only and path != ACTION_DOWN_PATH:
            continue
        if action_down_only and packed.shape != (18, 4096, 1024):
            raise ValueError(f"Expected action down-projection shape (18,4096,1024), got {packed.shape}")
        layout = _editable_layout(path, include_action)
        if not layout:
            continue
        depth = len(layout)
        if layout[0][1] is not None and (packed.ndim == 0 or packed.shape[0] != depth):
            raise ValueError(f"Expected scan depth {depth} at {'/'.join(path)}, got {packed.shape}")
        packed_tensor_count += 1
        for name, layer in layout:
            value = _effective_bfloat16(packed if layer is None else packed[layer])
            file_name = _file_name(name)
            np.save(base_dir / file_name, value, allow_pickle=False)
            _update_array_digest(digest, name, value)
            tensors[name] = {
                **_compact_stats(value),
                "file": file_name,
                "model_path": list(path),
                "scan_index": layer,
                "packed_shape": list(packed.shape),
                "writable": True,
            }
            if action_down_only:
                tensors[name]["base_file_sha256"] = _file_sha256(base_dir / file_name)
            editable_parameters += int(value.size)
    del params

    if not tensors:
        raise ValueError("No internal PaliGemma transformer tensors were found")
    total_parameters = _parameter_count(checkpoint)
    config = {
        "format": "openpi-paligemma-internals-v1",
        "checkpoint": str(checkpoint.resolve()),
        "exposure": {
            "included": [
                "PaliGemma image-transformer blocks 0..26",
                "PaliGemma language-transformer blocks 0..17",
            ],
            "excluded": [
                "input, patch, and position embeddings",
                "VLM and vision final norms / output projector",
                "all _1 action-expert tensors",
                "action_in_proj and action_out_proj",
                "time MLP",
            ],
        },
        "base_vlm_sha256": digest.hexdigest(),
        "packed_tensor_count": packed_tensor_count,
        "tensor_count": len(tensors),
        "editable_parameters": editable_parameters,
        "total_parameters_and_buffers": total_parameters,
        "max_changed_fraction": max_changed_fraction,
        "max_changed_elements": int(total_parameters * max_changed_fraction),
        "sandbox_memory_bytes": DEFAULT_SANDBOX_MEMORY_BYTES,
        "sandbox_timeout_seconds": DEFAULT_SANDBOX_TIMEOUT_SECONDS,
        "frozen": [
            "embeddings and final norms",
            "action expert",
            "action input and output projections",
            "time MLP",
        ],
        "tensors": tensors,
    }
    if include_action:
        config.update(
            {
                "format": "openpi-vlm-action-internals-v1",
                "include_action": True,
                "edit_scope": "VLM blocks, action expert, action projections, and time MLP",
                "base_editable_sha256": digest.hexdigest(),
                "exposure": {
                    "included": [
                        "VLM transformer blocks",
                        "all action expert blocks and final norm",
                        "action input/output projections",
                        "time MLP",
                    ],
                    "excluded": ["VLM embeddings, final norms and vision output projector"],
                },
                "frozen": ["VLM embeddings, final norms and vision output projector"],
            }
        )
    if action_down_only:
        config.update({
            "format": "openpi-action-down-rows-v1",
            "edit_scope": "action-expert MLP down-projection rows only",
            "exposure": {"included": ["18 action-expert MLP down-projection matrices"],
                         "excluded": ["all other checkpoint tensors"]},
            "frozen": ["all checkpoint tensors outside action-expert MLP down projections"],
            "edit_constraints": {"scope": "action_mlp_down_rows", "row_axis": 0,
                                 "max_changed_rows": max_changed_rows,
                                 "max_changed_layers": max_changed_layers},
            "max_changed_elements": max_changed_rows * 1024,
            "max_changed_fraction": min(max_changed_fraction, max_changed_rows * 1024 / total_parameters),
            "zero_edit_allowed": True,
        })
    (workspace / "workspace.json").write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    LOGGER.info("Exported %d layer tensors (%d editable values)", len(tensors), editable_parameters)
    return config


def _difference_stats(base: np.ndarray, edit: np.ndarray) -> dict[str, float | int]:
    base_flat = base.reshape(-1)
    edit_flat = edit.reshape(-1)
    changed = 0
    maximum = 0.0
    squared = 0.0
    base_squared = 0.0
    edit_squared = 0.0
    cross = 0.0
    chunk_size = 1_000_000
    for start in range(0, base_flat.size, chunk_size):
        stop = min(start + chunk_size, base_flat.size)
        before = base_flat[start:stop].astype(np.float64)
        after = edit_flat[start:stop].astype(np.float64)
        delta = after - before
        changed += int(np.count_nonzero(delta))
        maximum = max(maximum, float(np.max(np.abs(delta), initial=0.0)))
        squared += float(np.dot(delta, delta))
        base_squared += float(np.dot(before, before))
        edit_squared += float(np.dot(after, after))
        cross += float(np.dot(before, after))
    scale = cross / base_squared if base_squared else 0.0
    residual_squared = max(edit_squared - 2 * scale * cross + scale * scale * base_squared, 0.0)
    scale_residual = np.sqrt(residual_squared / max(edit_squared, np.finfo(float).tiny))
    return {
        "changed": changed,
        "max_abs_delta": maximum,
        "l2_delta": float(np.sqrt(squared)),
        "relative_l2_delta": float(np.sqrt(squared) / np.sqrt(max(base_squared, np.finfo(float).tiny))),
        "fitted_scale": float(scale),
        "scale_residual": float(scale_residual),
    }


def validate_workspace(workspace: Path) -> dict[str, Any]:
    config = json.loads((workspace / "workspace.json").read_text())
    tensors = config["tensors"]
    constraints = config.get("edit_constraints")
    if constraints:
        if constraints.get("scope") != "action_mlp_down_rows" or constraints.get("row_axis") != 0:
            raise ValueError("Unknown edit-constraint scope or row axis")
        if not (isinstance(constraints.get("max_changed_rows"), int)
                and 1 <= constraints["max_changed_rows"] <= 128
                and isinstance(constraints.get("max_changed_layers"), int)
                and 1 <= constraints["max_changed_layers"] <= 2):
            raise ValueError("Study limits exceed 128 rows or two layers")
        for name, entry in tensors.items():
            match = ACTION_DOWN_NAME.fullmatch(name)
            if (not match or not 0 <= int(match[1]) < 18
                    or entry["model_path"] != list(ACTION_DOWN_PATH)
                    or entry["scan_index"] != int(match[1])
                    or entry["file"] != _file_name(name)):
                raise ValueError(f"Tensor outside the action-down-projection allowlist: {name}")
            base_path = workspace / "base" / entry["file"]
            if _file_sha256(base_path) != entry.get("base_file_sha256"):
                raise ValueError(f"Base export checksum mismatch for {name}")
    known_files = {entry["file"] for entry in tensors.values()}
    unknown = sorted(
        path.name for path in (workspace / "edits").glob("*.npy") if path.name not in known_files
    )
    if unknown:
        raise ValueError(f"Edit directory contains non-allowlisted tensors: {unknown}")

    changed_total = 0
    changed_rows_total = 0
    changed_layers = set()
    edited: dict[str, dict[str, Any]] = {}
    for name, entry in tensors.items():
        edit_path = workspace / "edits" / entry["file"]
        if not edit_path.exists():
            continue
        if not entry.get("writable", True):
            raise ValueError(f"Edit supplied for read-only tensor {name}")
        base = np.load(workspace / "base" / entry["file"], allow_pickle=False, mmap_mode="r")
        edit = np.load(edit_path, allow_pickle=False, mmap_mode="r")
        if edit.shape != base.shape:
            raise ValueError(f"Shape changed for {name}: {edit.shape} != {base.shape}")
        if not np.isfinite(edit).all():
            raise ValueError(f"Non-finite edit for {name}")
        effective = _effective_bfloat16(edit)
        if not np.array_equal(edit, effective):
            raise ValueError(
                f"Edit for {name} is not exactly bfloat16-representable; use bf16(...)"
            )
        stats = _difference_stats(base, edit)
        changed = int(stats["changed"])
        if not changed:
            continue
        if (
            changed == int(base.size)
            and abs(float(stats["fitted_scale"]) - 1.0) > 1e-7
            and float(stats["scale_residual"]) < 5e-3
        ):
            raise ValueError(f"Refusing whole-tensor scalar multiplication for {name}")
        changed_total += changed
        edited[name] = {
            "changed": changed,
            "numel": int(base.size),
            "changed_fraction": changed / int(base.size),
            "max_abs_delta": stats["max_abs_delta"],
            "l2_delta": stats["l2_delta"],
            "relative_l2_delta": stats["relative_l2_delta"],
        }
        if constraints:
            if base.ndim != 2 or base.shape != (4096, 1024):
                raise ValueError(f"Invalid action down-projection shape for {name}: {base.shape}")
            rows = np.flatnonzero(np.any(base != edit, axis=1)).tolist()
            changed_rows_total += len(rows)
            changed_layers.add(entry["scan_index"])
            edited[name]["changed_rows"] = rows

    if constraints:
        if changed_rows_total > constraints["max_changed_rows"]:
            raise ValueError(f"Cumulative changed-row budget exceeded: {changed_rows_total} > {constraints['max_changed_rows']}")
        if len(changed_layers) > constraints["max_changed_layers"]:
            raise ValueError(f"Changed-layer budget exceeded: {len(changed_layers)} > {constraints['max_changed_layers']}")

    total = int(config["total_parameters_and_buffers"])
    fraction = changed_total / total
    if fraction > float(config["max_changed_fraction"]):
        raise ValueError(
            f"Global change budget exceeded: {fraction:.8%} > "
            f"{float(config['max_changed_fraction']):.8%}"
        )
    report = {
        "edit_scope": config.get("edit_scope", "internal PaliGemma transformer blocks only"),
        "edited_tensors": edited,
        "changed_elements": changed_total,
        "editable_parameters": int(config["editable_parameters"]),
        "total_parameters_and_buffers": total,
        "changed_fraction": fraction,
        "max_changed_fraction": float(config["max_changed_fraction"]),
        "base_vlm_sha256": config["base_vlm_sha256"],
        "action_expert_and_head_frozen": not config.get("include_action", False),
        "vlm_embeddings_and_final_norms_frozen": True,
    }
    if constraints:
        report.update(edit_constraints=constraints, changed_rows=changed_rows_total,
                      changed_layers=sorted(changed_layers),
                      l2_delta=float(np.sqrt(sum(e["l2_delta"] ** 2 for e in edited.values()))),
                      addressed_row_elements=changed_rows_total * 1024)
    return report


def _state_at(state: Any, path: list[str]) -> Any:
    node = state
    for component in path:
        node = node[component]
    return node


def _action_side_hash(state: Any) -> str:
    digest = hashlib.sha256()

    def visit(node: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, Mapping):
            for key, value in node.items():
                visit(value, (*path, str(key)))
            return
        is_action_module = bool(path) and path[0] != "PaliGemma"
        is_action_expert = path[:3] == ("PaliGemma", "llm", "final_norm_1") or any(
            component.endswith("_1") for component in path[3:]
        )
        if not (is_action_module or is_action_expert):
            return
        value = np.asarray(node.value)
        _update_array_digest(digest, "/".join(path), value)

    visit(state)
    return digest.hexdigest()


def _untouched_hash(state: Any, edited_entries: list[dict]) -> str:
    """Hash every model value except the exact slices that are about to be edited."""
    omitted: dict[tuple[str, ...], set[int | None]] = {}
    for entry in edited_entries:
        omitted.setdefault(tuple(entry["model_path"]), set()).add(entry["scan_index"])
    digest = hashlib.sha256()

    def visit(node: Any, path: tuple[str, ...] = ()) -> None:
        if isinstance(node, Mapping):
            for key in sorted(node):
                visit(node[key], (*path, str(key)))
            return
        skip = omitted.get(path, set())
        if None in skip:
            return
        value = np.asarray(node.value)
        if skip:
            for index in range(value.shape[0]):
                if index not in skip:
                    _update_array_digest(digest, f"{'/'.join(path)}[{index}]", value[index])
        else:
            _update_array_digest(digest, "/".join(path), value)

    visit(state)
    return digest.hexdigest()


def _apply_edits(model: Any, workspace: Path) -> dict[str, Any]:
    import jax.numpy as jnp
    from flax import nnx

    config = json.loads((workspace / "workspace.json").read_text())
    report = validate_workspace(workspace)
    if not report["edited_tensors"]:
        if not (config.get("zero_edit_allowed") and config.get("edit_constraints")):
            raise ValueError("No non-empty edit is present")
        # An explicit identity candidate exercises the export/load path without
        # modifying the checkpoint. Validate every exposed slice against the
        # actual loaded inference dtype before permitting parity evaluation.
        state = nnx.state(model)
        for name, entry in config["tensors"].items():
            variable = _state_at(state, entry["model_path"])
            loaded = np.asarray(variable.value[entry["scan_index"]]).astype(np.float32)
            base = np.load(workspace / "base" / entry["file"], allow_pickle=False, mmap_mode="r")
            if not np.array_equal(loaded, base):
                raise ValueError(f"Loaded checkpoint does not match workspace base for {name}")
        report["identity_candidate"] = True
        report["verified_base_slices"] = len(config["tensors"])
        return report
    state = nnx.state(model)
    edited_entries = [config["tensors"][name] for name in report["edited_tensors"]]
    frozen_before = _untouched_hash(state, edited_entries)
    for name in report["edited_tensors"]:
        entry = config["tensors"][name]
        variable = _state_at(state, entry["model_path"])
        index = entry["scan_index"]
        base = np.load(workspace / "base" / entry["file"], allow_pickle=False)
        loaded = np.asarray(variable.value if index is None else variable.value[index]).astype(
            np.float32
        )
        if not np.array_equal(loaded, base):
            raise ValueError(f"Loaded checkpoint does not match workspace base for {name}")
        edit = np.load(workspace / "edits" / entry["file"], allow_pickle=False)
        value = jnp.asarray(edit, dtype=variable.value.dtype)
        variable.value = value if index is None else variable.value.at[index].set(value)

    nnx.update(model, state)
    applied_state = nnx.state(model)
    for name in report["edited_tensors"]:
        entry = config["tensors"][name]
        variable = _state_at(applied_state, entry["model_path"])
        index = entry["scan_index"]
        applied = np.asarray(variable.value if index is None else variable.value[index]).astype(
            np.float32
        )
        expected = np.load(workspace / "edits" / entry["file"], allow_pickle=False)
        if not np.array_equal(applied, expected):
            raise ValueError(f"Applied model value does not match edit for {name}")
    frozen_after = _untouched_hash(applied_state, edited_entries)
    if frozen_before != frozen_after:
        raise ValueError("A model value outside the selected edit slices changed")
    report["untouched_values_sha256"] = frozen_after
    if not config.get("include_action", False):
        report["action_side_sha256"] = _action_side_hash(applied_state)
    return report


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Layer-split bf16 export of pi0.5 internals (VLM + action expert)")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--checkpoint", type=Path, required=True)
    prepare.add_argument("--workspace", type=Path, required=True)
    prepare.add_argument("--include-action", action="store_true")
    prepare.add_argument("--action-down-only", action="store_true",
                         help="Study export: 18 action MLP down matrices (~288 MiB), <=128 rows / two layers")
    prepare.add_argument("--max-changed-rows", type=int, default=128)
    prepare.add_argument("--max-changed-layers", type=int, default=2)
    prepare.add_argument("--max-changed-fraction", type=float, default=DEFAULT_MAX_CHANGED_FRACTION)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--workspace", type=Path, required=True)
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    args = _build_parser().parse_args()
    if args.command == "prepare":
        result = prepare_workspace(
            args.checkpoint, args.workspace, args.max_changed_fraction, args.include_action,
            action_down_only=args.action_down_only, max_changed_rows=args.max_changed_rows,
            max_changed_layers=args.max_changed_layers,
        )
    else:
        result = validate_workspace(args.workspace)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
