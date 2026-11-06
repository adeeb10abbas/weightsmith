"""Minimal safetensors reader and writer that needs only numpy.

Implements the published format (https://github.com/huggingface/safetensors#format): an 8-byte
little-endian header length, a JSON header, then one contiguous little-endian byte buffer. Files
written here load with the official ``safetensors`` library in any framework, and this module reads
files the official library writes; ``tests/test_edits.py`` checks both directions when it is installed.
"""
from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from pathlib import Path

import numpy as np

_DTYPES = {
    "F64": "<f8", "F32": "<f4", "F16": "<f2",
    "I64": "<i8", "I32": "<i4", "I16": "<i2", "I8": "i1",
    "U64": "<u8", "U32": "<u4", "U16": "<u2", "U8": "u1",
    "BOOL": "?",
}
_CODES = {np.dtype(v): k for k, v in _DTYPES.items()}
_MAX_HEADER = 100 << 20  # the reference implementation's limit


def save_file(tensors: Mapping[str, np.ndarray], path: str | Path, metadata: Mapping[str, str] | None = None) -> None:
    """Write ``tensors`` (name -> array) and optional string ``metadata`` to ``path``."""
    header: dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    prepared = {}
    for name, value in tensors.items():
        if name == "__metadata__":
            raise ValueError("'__metadata__' is reserved")
        array = np.asarray(value)
        if array.dtype.byteorder == ">":
            array = array.astype(array.dtype.newbyteorder("<"))
        if not array.flags.c_contiguous:
            array = array.copy(order="C")
        if np.dtype(array.dtype) not in _CODES:
            raise TypeError(f"{name}: dtype {array.dtype} is not supported")
        prepared[name] = array
    arrays = []
    offset = 0
    # Widest dtypes first (as the reference writer does), so every tensor starts aligned.
    for name in sorted(prepared, key=lambda k: (-prepared[k].dtype.itemsize, k)):
        array = prepared[name]
        code = _CODES[np.dtype(array.dtype)]
        header[name] = {"dtype": code, "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        arrays.append(array)
        offset += array.nbytes
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)  # align the byte buffer to 8 bytes
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(raw)))
        f.write(raw)
        for array in arrays:
            if array.nbytes:
                f.write(array.reshape(-1).view(np.uint8).data)


def load_file(path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, str]]:
    """Return ``(tensors, metadata)``. Tensors are read-only views of a memory map."""
    path = Path(path)
    size = path.stat().st_size
    with open(path, "rb") as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path}: not a safetensors file (too short)")
        (n,) = struct.unpack("<Q", prefix)
        if n > _MAX_HEADER or 8 + n > size:
            raise ValueError(f"{path}: not a safetensors file (bad header length)")
        header = json.loads(f.read(n))
    metadata = header.pop("__metadata__", None) or {}
    start = 8 + n
    buffer = (np.memmap(path, dtype=np.uint8, mode="r", offset=start) if size > start
              else np.zeros(0, dtype=np.uint8))
    tensors = {}
    for name, info in header.items():
        if info["dtype"] not in _DTYPES:
            raise TypeError(f"{name}: dtype {info['dtype']} is not supported")
        dtype = np.dtype(_DTYPES[info["dtype"]])
        shape = tuple(int(d) for d in info["shape"])
        begin, end = (int(x) for x in info["data_offsets"])
        if not 0 <= begin <= end <= buffer.size or end - begin != int(np.prod(shape, dtype=np.int64)) * dtype.itemsize:
            raise ValueError(f"{path}: bad data offsets for {name}")
        tensors[name] = buffer[begin:end].view(dtype).reshape(shape)
    return tensors, {str(k): str(v) for k, v in metadata.items()}
