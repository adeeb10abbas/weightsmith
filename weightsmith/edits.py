"""Portable weight edits: one self-describing file per candidate.

A candidate made with ``owsp`` is a directory of full edited tensors (``cands/<name>/edits/*.npy``)
that only works next to the base export it was made from. This module turns it into one file that
anyone with the public checkpoint can inspect, verify and apply, and back::

    weightsmith edit export $WSP_ROOT/cands/v2mi-f -o v2mi-f.safetensors --rank auto
    weightsmith edit info   v2mi-f.safetensors
    weightsmith edit apply  v2mi-f.safetensors --workspace $BASE_WORKSPACE --out $WSP_ROOT/cands/v2mi-f-copy

Format ``weightsmith.edit``, version 1
--------------------------------------
A safetensors file (or an ``.npz`` with the same keys). For every changed tensor ``T`` it stores an
encoding of ``target - base``, where both are bf16 values held in float32:

* ``dense``    ``T/delta``          float32, the tensor's shape
* ``sparse``   ``T/index``, ``T/delta``   flat C-order indices and float32 deltas of the changed values
* ``lowrank``  ``T/u``, ``T/v``     float32 factors of the delta matricized as ``(prod(shape[:-1]), shape[-1])``;
  ``u = U_r * S_r`` and ``v = V_r^T`` from an exact SVD truncated to rank ``r``

plus optional exact fix-ups ``T/fix_index`` and ``T/fix_value`` (absolute target values). Applying an
edit computes ``bf16(base + delta)`` (in float64 for low-rank factors), then writes the fix-ups. The
exporter checks the result bit for bit, so an exact edit reproduces the candidate exactly.
``--lossy`` drops the low-rank fix-ups and reports how many values are then off.

The JSON metadata (safetensors header key ``weightsmith.edit``, or the ``__weightsmith_edit__``
entry of an npz) records the format version, the source candidate and its ``owsp`` fingerprint, the
base export (checkpoint name, export format, editable-values digest), and per tensor: shape, encoding,
changed values, sha256 of the base and target values, low-rank error report, and the OpenPI parameter
path (``model_path``, ``scan_index``) so the edit can be applied straight to a restored params tree.
"""
from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections.abc import Iterator, Mapping, MutableMapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from weightsmith import _safetensors

FORMAT = "weightsmith.edit"
FORMAT_VERSION = 1
METADATA_KEY = "weightsmith.edit"  # safetensors __metadata__ key holding the JSON metadata
NPZ_METADATA_KEY = "__weightsmith_edit__"
ENCODINGS = ("dense", "sparse", "lowrank")
_PARTS = {"dense": ("delta",), "sparse": ("index", "delta"), "lowrank": ("u", "v")}


class EditError(ValueError):
    """An edit cannot be built, read or applied as asked."""


# ---------------------------------------------------------------------------------------------- helpers
def bf16_round(x: Any) -> np.ndarray:
    """Round to the nearest bfloat16 (ties to even), returned as float32.

    The same rule as ``bf16()`` in the ``owsp py`` sandbox and the base export.
    """
    value = np.asarray(x, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return rounded.view(np.float32)


def tensor_sha256(x: Any) -> str:
    """sha256 of the float32, little-endian, C-order bytes of ``x``."""
    value = np.ascontiguousarray(np.asarray(x, dtype="<f4"))
    return hashlib.sha256(value.data).hexdigest()


def fingerprint(edits_dir: str | Path) -> str:
    """The ``owsp`` fingerprint of a candidate's ``edits/`` directory (same algorithm as owsp and heldout)."""
    h = hashlib.sha256()
    for f in sorted(Path(edits_dir).glob("*.npy")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def _bits(x: np.ndarray) -> np.ndarray:
    return np.asarray(x, dtype=np.float32).view(np.uint32)


def _index_dtype(numel: int) -> type:
    return np.int32 if numel < 2**31 else np.int64


def _producer() -> str:
    try:
        from importlib.metadata import version

        return f"weightsmith {version('weightsmith')}"
    except Exception:  # noqa: BLE001 - running from a source checkout
        return "weightsmith (source)"


def _human_bytes(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000
    return f"{n:.1f} GB"


def _decode(info: Mapping[str, Any], arrays: Mapping[str, np.ndarray], base: np.ndarray) -> np.ndarray:
    """Edited values for one tensor; ``base`` must already be bf16-rounded float32."""
    encoding = info["encoding"]
    if encoding == "dense":
        out = bf16_round(base + np.asarray(arrays["delta"], dtype=np.float32).reshape(base.shape))
    elif encoding == "sparse":
        out = np.array(base, dtype=np.float32, order="C")
        index = np.asarray(arrays["index"])
        np.put(out, index, bf16_round(np.take(base, index) + np.asarray(arrays["delta"], dtype=np.float32)))
    elif encoding == "lowrank":
        m, n = info["matrix_shape"]
        x = base.reshape(m, n).astype(np.float64)
        x += np.asarray(arrays["u"], dtype=np.float64) @ np.asarray(arrays["v"], dtype=np.float64)
        out = bf16_round(x.astype(np.float32)).reshape(base.shape)
    else:
        raise EditError(f"unknown encoding {encoding!r}")
    out = np.array(out, dtype=np.float32, order="C")
    if "fix_index" in arrays:
        np.put(out, np.asarray(arrays["fix_index"]), np.asarray(arrays["fix_value"], dtype=np.float32))
    return out


# ---------------------------------------------------------------------------------------------- one tensor
@dataclass
class TensorEdit:
    """The edit of one tensor: JSON-serializable ``info`` plus the payload ``arrays``."""

    name: str
    info: dict[str, Any]
    arrays: dict[str, np.ndarray]

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.info["shape"])

    @property
    def encoding(self) -> str:
        return self.info["encoding"]

    @property
    def nbytes(self) -> int:
        return int(sum(np.asarray(a).nbytes for a in self.arrays.values()))

    def reconstruct(self, base: Any, *, verify: bool = True) -> np.ndarray:
        """Return the edited values of this tensor given its base values.

        ``base`` may be raw float32 checkpoint values or the bf16 export; it is rounded to bf16 first,
        as OpenPI does at inference. With ``verify``, the base must match the edit's recorded base and,
        for an exact edit, the result must match the recorded target, both by sha256.
        """
        b = bf16_round(base)
        if b.shape != self.shape:
            raise EditError(f"{self.name}: base shape {b.shape} != edit shape {self.shape}")
        if verify and tensor_sha256(b) != self.info["base_sha256"]:
            raise EditError(f"{self.name}: base values differ from the base this edit was made on "
                            "(sha256 mismatch); pass verify=False / --no-verify to apply anyway")
        out = _decode(self.info, self.arrays, b)
        if verify and self.info.get("exact") and tensor_sha256(out) != self.info["target_sha256"]:
            raise EditError(f"{self.name}: reconstruction does not match the recorded target (sha256 mismatch)")
        return out


def encode_tensor(name: str, base: Any, target: Any, *, rank: int | str | None = None, lossy: bool = False,
                  extra: Mapping[str, Any] | None = None) -> TensorEdit | None:
    """Encode ``target - base`` for one tensor; returns None when nothing changed.

    ``rank=None`` picks the smaller exact encoding of dense and sparse. ``rank=r`` stores a rank-r
    factorization (tensors with 2+ dims) plus exact fix-ups. ``rank="auto"`` tries ranks 1, 2, 4, ...
    and keeps whichever exact encoding is smallest. ``lossy=True`` (explicit rank only) drops the
    fix-ups and reports the values that are then off.
    """
    b = np.array(bf16_round(base), order="C")
    t = np.array(target, dtype=np.float32, order="C")
    if t.shape != b.shape:
        raise EditError(f"{name}: edited shape {t.shape} != base shape {b.shape}")
    if not np.isfinite(t).all():
        raise EditError(f"{name}: non-finite edited values")
    if not np.array_equal(_bits(bf16_round(t)), _bits(t)):
        raise EditError(f"{name}: edited values are not exactly bfloat16-representable; round them with bf16()")
    if lossy and (rank is None or rank == "auto"):
        raise EditError("lossy export needs an explicit low-rank rank (e.g. rank=72)")
    differ = np.flatnonzero(_bits(t) != _bits(b))
    if differ.size == 0:
        return None
    d64 = t.astype(np.float64) - b.astype(np.float64)
    info: dict[str, Any] = {
        "shape": list(t.shape), "dtype": "bfloat16", "numel": int(t.size),
        "changed": int(np.count_nonzero(d64)), "max_abs_delta": float(np.max(np.abs(d64))),
        "l2_delta": float(np.linalg.norm(d64.ravel())),
        "base_sha256": tensor_sha256(b), "target_sha256": tensor_sha256(t), **(extra or {}),
    }
    index_dtype = _index_dtype(t.size)
    index_size = np.dtype(index_dtype).itemsize

    def with_fixups(enc: dict[str, Any], arrays: dict[str, np.ndarray], keep_fixups: bool = True):
        off = np.flatnonzero(_bits(_decode(enc, arrays, b)) != _bits(t))
        enc = {**enc, "values_off_before_fixups": int(off.size)}
        if keep_fixups and off.size:
            arrays = {**arrays, "fix_index": off.astype(index_dtype), "fix_value": np.take(t, off)}
        enc["fixups"] = int(off.size) if keep_fixups else 0
        enc["values_off"] = 0 if keep_fixups else int(off.size)
        enc["exact"] = keep_fixups or off.size == 0
        return enc, arrays, int(sum(a.nbytes for a in arrays.values()))

    # exact dense or sparse: choose by size, then add fix-ups for the rare inexact float32 deltas
    if differ.size * (index_size + 4) < t.size * 4:
        exact = with_fixups({"encoding": "sparse"}, {"index": differ.astype(index_dtype),
                                                     "delta": np.take(t, differ) - np.take(b, differ)})
    else:
        exact = with_fixups({"encoding": "dense"}, {"delta": t - b})
    best = exact

    if rank is not None and t.ndim >= 2:
        m, n = int(np.prod(t.shape[:-1])), int(t.shape[-1])
        u, s, vt = np.linalg.svd(d64.reshape(m, n), full_matrices=False)
        energy = np.cumsum(s.astype(np.float64) ** 2)
        total = float(energy[-1]) if energy.size else 0.0

        def factor(r: int, keep_fixups: bool = True):
            kept = float(energy[r - 1]) if r else 0.0
            enc = {"encoding": "lowrank", "rank": r, "matrix_shape": [m, n],
                   "rel_fro_error": math.sqrt(max(total - kept, 0.0) / total) if total else 0.0}
            arrays = {"u": (u[:, :r] * s[:r]).astype(np.float32), "v": vt[:r].astype(np.float32)}
            return with_fixups(enc, arrays, keep_fixups)

        if rank == "auto":
            r = 1
            while r <= s.size and r * (m + n) * 4 < best[2]:  # stop once the factors alone are larger
                trial = factor(r)
                if trial[2] < best[2]:
                    best = trial
                r *= 2
        else:
            r = int(rank)
            if r < 1:
                raise EditError("rank must be >= 1")
            best = factor(min(r, s.size), keep_fixups=not lossy)

    enc, arrays, nbytes = best
    info.update(enc)
    info["bytes"] = nbytes
    te = TensorEdit(name, info, arrays)
    if te.info["exact"] and not np.array_equal(_bits(_decode(info, arrays, b)), _bits(t)):
        raise AssertionError(f"{name}: internal error, exact encoding does not round-trip")  # pragma: no cover
    if not te.info["exact"]:
        rec = _decode(info, arrays, b).astype(np.float64)
        err = rec - t.astype(np.float64)
        info["max_abs_error"] = float(np.max(np.abs(err)))
        info["rel_l2_error"] = float(np.linalg.norm(err.ravel()) / max(info["l2_delta"], np.finfo(float).tiny))
    return te


# ---------------------------------------------------------------------------------------------- an edit
@dataclass
class Edit:
    """A set of tensor edits plus provenance (``meta``: source, base, producer, created)."""

    tensors: dict[str, TensorEdit]
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[TensorEdit]:
        return iter(self.tensors.values())

    @property
    def exact(self) -> bool:
        return all(t.info.get("exact", False) for t in self)

    @property
    def changed_values(self) -> int:
        return sum(int(t.info["changed"]) for t in self)

    @property
    def nbytes(self) -> int:
        return sum(t.nbytes for t in self)

    def summary(self) -> dict[str, Any]:
        total = (self.meta.get("base") or {}).get("total_parameters")
        return {
            "tensors": len(self.tensors),
            "changed_values": self.changed_values,
            "changed_fraction": self.changed_values / total if total else None,
            "exact": self.exact,
            "values_off": sum(int(t.info.get("values_off", 0)) for t in self),
            "payload_bytes": self.nbytes,
            "dense_bytes": sum(int(t.info["numel"]) * 4 for t in self),
        }

    def metadata(self) -> dict[str, Any]:
        """The JSON metadata written into the file."""
        meta = {k: v for k, v in self.meta.items() if k not in ("format", "format_version", "summary", "tensors")}
        meta.setdefault("producer", _producer())
        meta.setdefault("created", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        return {"format": FORMAT, "format_version": FORMAT_VERSION, **meta, "summary": self.summary(),
                "tensors": {name: t.info for name, t in sorted(self.tensors.items())}}

    def save(self, path: str | Path, *, container: str | None = None) -> Path:
        """Write the edit to ``path`` as safetensors (default) or npz (``.npz`` suffix or container='npz')."""
        path = Path(path)
        container = container or ("npz" if path.suffix == ".npz" else "safetensors")
        payload = {f"{name}/{part}": array for name, t in self.tensors.items() for part, array in t.arrays.items()}
        text = json.dumps(self.metadata(), sort_keys=True)
        if container == "safetensors":
            _safetensors.save_file(payload, path, {METADATA_KEY: text})
        elif container == "npz":
            with open(path, "wb") as f:  # a file object: np.savez must not append ".npz"
                np.savez(f, **payload, **{NPZ_METADATA_KEY: np.array(text)})
        else:
            raise EditError(f"unknown container {container!r} (use 'safetensors' or 'npz')")
        return path

    def apply(self, arrays: Mapping[str, Any], *, verify: bool = True) -> dict[str, np.ndarray]:
        """Return a copy of ``arrays`` (logical tensor name -> values) with the edit applied.

        Edited tensors come back as float32 arrays holding bf16 values; others pass through unchanged.
        """
        missing = sorted(set(self.tensors) - set(arrays))
        if missing:
            raise EditError(f"{len(missing)} edited tensor(s) missing from the input, e.g. {missing[:3]}")
        out = dict(arrays)
        for name, t in self.tensors.items():
            out[name] = t.reconstruct(arrays[name], verify=verify)
        return out

    def apply_to_params(self, params: MutableMapping[str, Any], *, verify: bool = True) -> list[str]:
        """Apply the edit in place to a nested OpenPI params tree of numpy arrays.

        For example ``openpi.models.model.restore_params(path, restore_type=np.ndarray)``. Each tensor
        is located by its recorded ``model_path`` and ``scan_index`` (scanned layers are stacked on
        axis 0). Returns the edited parameter paths.
        """
        done = []
        for name, t in self.tensors.items():
            path, index = t.info.get("model_path"), t.info.get("scan_index")
            if not path:
                raise EditError(f"{name}: the edit records no OpenPI model_path")
            node = params
            for key in path[:-1]:
                if key not in node:
                    raise EditError(f"{name}: params tree has no {'/'.join(path)}")
                node = node[key]
            if path[-1] not in node:
                raise EditError(f"{name}: params tree has no {'/'.join(path)}")
            leaf = np.array(node[path[-1]], copy=True)
            value = t.reconstruct(leaf if index is None else leaf[index], verify=verify)
            if index is None:
                leaf = value.astype(leaf.dtype)
            else:
                leaf[index] = value.astype(leaf.dtype)
            node[path[-1]] = leaf
            done.append("/".join(path) + ("" if index is None else f"[{index}]"))
        return done


# ---------------------------------------------------------------------------------------------- build
def from_arrays(base: Mapping[str, Any], edited: Mapping[str, Any], *, rank: int | str | None = None,
                lossy: bool = False, tensor_info: Mapping[str, Mapping[str, Any]] | None = None,
                meta: Mapping[str, Any] | None = None) -> Edit:
    """Build an edit from base and edited arrays keyed by logical tensor name (unchanged ones are skipped)."""
    tensors = {}
    for name in sorted(edited):
        if name not in base:
            raise EditError(f"{name}: no base values")
        t = encode_tensor(name, base[name], edited[name], rank=rank, lossy=lossy,
                          extra=(tensor_info or {}).get(name))
        if t is not None:
            tensors[name] = t
    return Edit(tensors, dict(meta or {}))


def from_candidate(candidate: str | Path, *, rank: int | str | None = None, lossy: bool = False,
                   progress: Any = None) -> Edit:
    """Build an edit from an owsp candidate directory (``workspace.json``, ``base/``, ``edits/``, ``meta.json``).

    Run it where the candidate's base export is (the GPU host). ``progress`` is called with each
    tensor name.
    """
    cand = Path(candidate)
    if not (cand / "workspace.json").is_file() or not (cand / "edits").is_dir():
        raise EditError(f"{cand} is not an owsp candidate (needs workspace.json and edits/)")
    cfg = json.loads((cand / "workspace.json").read_text())
    by_file = {entry["file"]: name for name, entry in cfg["tensors"].items()}
    files = sorted((cand / "edits").glob("*.npy"))
    unknown = [f.name for f in files if f.name not in by_file]
    if unknown:
        raise EditError(f"edits/ holds files that are not exported tensors: {unknown[:5]}")
    tensors = {}
    for f in files:
        name = by_file[f.name]
        if progress:
            progress(name)
        entry = cfg["tensors"][name]
        extra = {"model_path": entry.get("model_path"), "scan_index": entry.get("scan_index")}
        t = encode_tensor(name, np.load(cand / "base" / f.name, allow_pickle=False),
                          np.load(f, allow_pickle=False), rank=rank, lossy=lossy, extra=extra)
        if t is not None:
            tensors[name] = t
    source: dict[str, Any] = {"candidate": cand.name, "fingerprint": fingerprint(cand / "edits")}
    if (cand / "meta.json").is_file():
        m = json.loads((cand / "meta.json").read_text())
        source.update({k: m[k] for k in ("from", "note", "created") if k in m})
    return Edit(tensors, {"source": source, "base": _base_identity(cfg)})


def _base_identity(cfg: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "checkpoint": Path(cfg["checkpoint"]).name if cfg.get("checkpoint") else None,
        "workspace_format": cfg.get("format"),
        "editable_sha256": cfg.get("base_editable_sha256") or cfg.get("base_vlm_sha256"),
        "total_parameters": cfg.get("total_parameters_and_buffers"),
        "tensor_count": cfg.get("tensor_count"),
    }


# ---------------------------------------------------------------------------------------------- read
def load(path: str | Path) -> Edit:
    """Read an edit written by :meth:`Edit.save` (safetensors or npz)."""
    path = Path(path)
    with open(path, "rb") as f:
        magic = f.read(2)
    if magic == b"PK":
        with np.load(path, allow_pickle=False) as z:
            if NPZ_METADATA_KEY not in z.files:
                raise EditError(f"{path}: not a weightsmith edit (no {NPZ_METADATA_KEY})")
            meta = json.loads(str(z[NPZ_METADATA_KEY][()]))
            payload = {k: z[k] for k in z.files if k != NPZ_METADATA_KEY}
    else:
        payload, header = _safetensors.load_file(path)
        if METADATA_KEY not in header:
            raise EditError(f"{path}: not a weightsmith edit (no '{METADATA_KEY}' metadata)")
        meta = json.loads(header[METADATA_KEY])
    if meta.get("format") != FORMAT:
        raise EditError(f"{path}: format {meta.get('format')!r} is not {FORMAT!r}")
    if int(meta.get("format_version", 0)) > FORMAT_VERSION:
        raise EditError(f"{path}: format version {meta['format_version']} is newer than this reader "
                        f"({FORMAT_VERSION}); upgrade weightsmith")
    parts: dict[str, dict[str, np.ndarray]] = {}
    for key, array in payload.items():
        name, part = key.rsplit("/", 1)
        parts.setdefault(name, {})[part] = array
    tensors = {}
    for name, info in meta["tensors"].items():
        arrays = parts.get(name, {})
        need = _PARTS.get(info.get("encoding"))
        if need is None or any(p not in arrays for p in need):
            raise EditError(f"{path}: tensor {name} is missing payload for encoding {info.get('encoding')!r}")
        tensors[name] = TensorEdit(name, dict(info), arrays)
    return Edit(tensors, {k: v for k, v in meta.items() if k not in ("tensors", "summary")})


# ---------------------------------------------------------------------------------------------- apply
def apply_to_workspace(edit: Edit, workspace: str | Path, out: str | Path, *, verify: bool = True,
                       note: str = "") -> dict[str, Any]:
    """Materialize ``edit`` as an owsp candidate directory ``out`` on top of a base export.

    ``workspace`` is the layer-split bf16 export made by ``scripts/prepare_workspace.sh``
    (``$BASE_WORKSPACE``), or any candidate made from it. The new candidate has the same layout as
    one made by ``owsp new``, so ``owsp diff/eval`` and ``weightsmith/heldout.py`` accept it when it
    sits in ``$WSP_ROOT/cands/``. Returns a report including the reproduced fingerprint.
    """
    ws, out = Path(workspace), Path(out)
    if not (ws / "workspace.json").is_file() or not (ws / "base").is_dir():
        raise EditError(f"{ws} is not a base export (needs workspace.json and base/)")
    if out.exists():
        raise EditError(f"{out} exists; choose a new candidate directory")
    cfg = json.loads((ws / "workspace.json").read_text())
    missing = sorted(set(edit.tensors) - set(cfg["tensors"]))
    if missing:
        raise EditError(f"{len(missing)} edited tensor(s) are not in this export, e.g. {missing[:3]}")
    base_dir = (ws / "base").resolve()
    out.mkdir(parents=True)
    try:
        (out / "edits").mkdir()
        (out / "scratch").mkdir()
        (out / "base").symlink_to(base_dir)
        cand_cfg = dict(cfg, max_changed_fraction=1.0, sandbox_memory_bytes=100 << 30, sandbox_timeout_seconds=1800)
        (out / "workspace.json").write_text(json.dumps(cand_cfg))  # as `owsp new` writes it
        for name, t in edit.tensors.items():
            entry = cfg["tensors"][name]
            value = t.reconstruct(np.load(base_dir / entry["file"], allow_pickle=False), verify=verify)
            np.save(out / "edits" / entry["file"], value, allow_pickle=False)
        source = edit.meta.get("source") or {}
        (out / "meta.json").write_text(json.dumps({
            "created": time.time(), "from": None,
            "note": note or f"weightsmith edit apply (source {source.get('candidate', '?')}, "
                            f"fingerprint {source.get('fingerprint', '?')})",
            "edit": {"source": source, "base": edit.meta.get("base"), "exact": edit.exact},
        }))
    except BaseException:
        shutil.rmtree(out, ignore_errors=True)
        raise
    fp = fingerprint(out / "edits")
    expected = (edit.meta.get("source") or {}).get("fingerprint")
    return {"candidate": str(out), "fingerprint": fp, "source_fingerprint": expected,
            "fingerprint_matches": bool(expected) and fp == expected, "tensors": len(edit.tensors),
            "changed_values": edit.changed_values, "exact": edit.exact}


# ---------------------------------------------------------------------------------------------- describe
def describe(edit: Edit) -> str:
    """A human-readable summary (what ``weightsmith edit info`` prints)."""
    s = edit.summary()
    src, base = edit.meta.get("source") or {}, edit.meta.get("base") or {}
    lines = [f"{FORMAT} v{edit.meta.get('format_version', FORMAT_VERSION)}"
             + (f"  ·  {edit.meta['producer']}" if edit.meta.get("producer") else "")]
    if src:
        lines.append(f"  source    {src.get('candidate', '?')}  ·  fingerprint {src.get('fingerprint', '?')}"
                     + (f"  ·  from {src['from']}" if src.get("from") else ""))
    if base:
        digest = base.get("editable_sha256") or ""
        lines.append(f"  base      {base.get('checkpoint') or '?'}  ·  {base.get('workspace_format') or '?'}"
                     + (f"  ·  editable sha256 {digest[:12]}…" if digest else ""))
    frac = f" ({100 * s['changed_fraction']:.3f}% of {base['total_parameters']:,} parameters)" \
        if s["changed_fraction"] is not None else ""
    lines.append(f"  changed   {s['changed_values']:,} values in {s['tensors']} tensors{frac}")
    lines.append("  exact     yes: applying it reproduces the edited values bit for bit" if s["exact"] else
                 f"  exact     no: {s['values_off']:,} values differ from the source after bf16 rounding (lossy)")
    lines.append(f"  payload   {_human_bytes(s['payload_bytes'])} (dense float32: {_human_bytes(s['dense_bytes'])})")
    lines.append("")
    width = max([len("tensor")] + [len(n) for n in edit.tensors])
    lines.append(f"  {'tensor':{width}s}  {'shape':>14s}  {'changed':>12s}  {'encoding':12s}  {'fix-ups':>9s}  "
                 f"{'trunc. error':>12s}")
    for name, t in sorted(edit.tensors.items()):
        i = t.info
        enc = f"lowrank r={i['rank']}" if i["encoding"] == "lowrank" else i["encoding"]
        err = f"{i['rel_fro_error']:.2e}" if "rel_fro_error" in i else "-"
        shape = "x".join(str(d) for d in i["shape"])
        lines.append(f"  {name:{width}s}  {shape:>14s}  {i['changed']:>12,}  {enc:12s}  {i.get('fixups', 0):>9,}  "
                     f"{err:>12s}")
    return "\n".join(lines)
