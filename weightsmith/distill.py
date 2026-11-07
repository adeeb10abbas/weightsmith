"""Closed-form instruction distillation: turn a prompt fix into a weight fix.

When a policy fails under its own instruction but succeeds when the same task is worded explicitly, the
network already computes the right thing for *some* input. This module copies the explicit wording's effect
into the weights, in closed form, from a handful of recorded decisions. There is no optimiser and no
training data: one linear solve per edited layer.

The math (per edited VLM layer L; ``language.blocks.LL.mlp.linear`` is the MLP down-projection, stored
``[16384, 2048]`` and applied as ``y = h @ W``):

* aligned positions: prompt-template/state slots (``mode="state"``), the 2 cameras x 16 image cells
  (``mode="image"``), or both. An aligned position holds the same token (or image region) under both wordings.
* ``d_L[p]`` = mean over the recorded decisions of (residual under the explicit wording - residual under the
  original wording) at the output of layer L, position p.
* keys ``K[p]`` = mean MLP hidden activation (the down-projection's input) of the *original* wording at p.
* values ``V[p] = alpha * (d_L[p] - d_{L-1}[p])``: the part of the gap that layer L itself adds (``d_{-1} = 0``).
* update ``dW = K^T (K K^T + lam I)^-1 V`` with ``lam = lam_rel * trace(K K^T) / n``: the minimum-norm change
  that maps every key to its value (exactly, up to the small ridge). It is dense but has rank <= n.

Only VLM layers 0..16 are edited: the action expert reads each VLM layer's keys/values, so the MLP output of
the last layer (17) never reaches the actions.

Layout of this module

1. numeric core (numpy only): :func:`closed_form_edit`, :func:`layer_increments`, :func:`align_slots`,
   :func:`build_keys_values`, :func:`low_rank`, :class:`LowRank`, :class:`Edit`, :func:`build_edit`.
2. decision helpers: :func:`gripper_closed`, :func:`panda_fk`, :func:`select_early_releases`,
   :func:`select_grasps`, :func:`decision_summary`.
3. sandbox scripts for ``owsp py`` (which cannot import this package; the needed functions are inlined with
   :func:`inspect.getsource`): :func:`render_sandbox_script` (build), :func:`render_check_script`,
   :func:`render_select_script`, :func:`render_compare_script`.
4. pipeline (stdlib only; drives ``bin/owsp``): :func:`run_pipeline`, :func:`main`.
   ``python -m weightsmith.distill run CONFIG.json [--dry-run]``.

The method is Section 2 of the paper; examples/cube_then_banana/distill.json is a worked config, and
study/results/opus/ctb/v2mi_edit.py is the original script that this module reproduces bit for bit.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import inspect
import json
import math
import os
import shlex
import subprocess
import sys
import textwrap
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "MLP_DOWN", "EDITABLE_LAYERS", "IMAGE_CELLS", "PI05_FRAME_PREFIX", "PI05_STATE_TOKEN", "MODES",
    "closed_form_edit", "layer_increments", "align_slots", "pi05_slot_map", "build_keys_values",
    "bf16_round", "LowRank", "low_rank", "Edit", "build_edit",
    "gripper_closed", "panda_fk", "select_early_releases", "select_grasps", "decision_summary",
    "BuildConfig", "render_sandbox_script", "render_check_script", "render_select_script",
    "render_compare_script", "parse_result", "capture_specs", "check_specs", "check_groups", "run_pipeline", "main",
]

VERSION = "1"
MLP_DOWN = "language.blocks.{:02d}.mlp.linear"  # VLM MLP down-projection, [16384, 2048], y = h @ W
EDITABLE_LAYERS = tuple(range(17))  # VLM layers whose MLP output the action expert can read
IMAGE_CELLS = tuple((cam, cell) for cam in (0, 1) for cell in range(1, 17))  # 2 cameras x 4x4 grid (0 = mean)
PI05_FRAME_PREFIX = 3  # "<bos>", "Task", ":" open every pi0.5 prompt
PI05_STATE_TOKEN = 3040  # "▁State" (PaliGemma tokenizer): start of ", State: ...;\nAction: "
MODES = ("state", "image", "both")


# =============================================================================== 1. numeric core
def closed_form_edit(K: np.ndarray, V: np.ndarray, lam_rel: float = 1e-3) -> np.ndarray:
    """Minimum-norm weight update mapping each key row of ``K`` to the matching row of ``V``.

    Returns ``dW = K^T (K K^T + lam I)^-1 V`` with ``lam = lam_rel * trace(K K^T) / n`` (n = number of keys),
    computed in float64. Then ``K @ dW = G (G + lam I)^-1 V`` with ``G = K K^T``: equal to ``V`` up to the ridge,
    and exactly ``V`` as ``lam_rel -> 0`` when ``K`` has full row rank. ``dW`` lies in the row space of ``K``,
    so it is the smallest (Frobenius) update achieving that fit and has rank <= n.

    Args:
        K: ``[n, d_in]`` keys (inputs of the linear map, e.g. MLP hidden activations).
        V: ``[n, d_out]`` values (desired change of the map's output at each key).
        lam_rel: ridge strength relative to the mean squared key norm; must be > 0 if keys are collinear.

    Returns:
        ``[d_in, d_out]`` float64 update for a weight applied as ``y = h @ W``.
    """
    K = np.asarray(K, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    if K.ndim != 2 or V.ndim != 2 or K.shape[0] != V.shape[0] or K.shape[0] == 0:
        raise ValueError(f"need K [n, d_in] and V [n, d_out] with the same n > 0, got {K.shape} and {V.shape}")
    if not lam_rel >= 0:
        raise ValueError(f"lam_rel must be >= 0, got {lam_rel}")
    G = K @ K.T
    lam = lam_rel * np.trace(G) / len(G)
    return K.T @ np.linalg.solve(G + lam * np.eye(len(G)), V)


def layer_increments(gaps: Mapping[int, np.ndarray], layers: Sequence[int],
                     alpha: float = 1.0) -> dict[int, np.ndarray]:
    """Per-layer targets ``V_L = alpha * (d_L - d_{L-1})`` with ``d_{-1} = 0``.

    ``gaps[L]`` is ``d_L``: the mean (explicit - original) residual difference at the output of layer L, one row
    per aligned position. Writing ``V_L`` into layer L reproduces the share of the gap that layer L adds, so the
    increments of consecutive layers telescope: ``sum_{L<=M} V_L = alpha * d_M``. Layer ``L-1`` must be present
    in ``gaps`` for every ``L > 0`` even when it is not edited.
    """
    out: dict[int, np.ndarray] = {}
    for L in layers:
        if L not in gaps or (L > 0 and (L - 1) not in gaps):
            raise KeyError(f"layer {L} needs gaps for layers {L} and {L - 1}")
        out[L] = alpha * (gaps[L] - (gaps[L - 1] if L > 0 else 0))
    return out


def align_slots(orig_tokens: Sequence[int], expl_tokens: Sequence[int], *, orig_valid: Sequence[bool] | None = None,
                expl_valid: Sequence[bool] | None = None, max_prefix: int | None = None,
                suffix_from: int | None = None) -> list[tuple[int, int]]:
    """Pair prompt positions that hold the same token in the same frame under two wordings.

    The prefix part pairs positions ``i <-> i`` over the longest common prefix of the valid tokens (capped at
    ``max_prefix``). The suffix part pairs the longest common suffix (``i <-> i + len(expl) - len(orig)``); with
    ``suffix_from`` (a token id) it starts at that token's first occurrence inside the common suffix.

    Args:
        orig_tokens, expl_tokens: token ids per prompt position (padding allowed at the end).
        orig_valid, expl_valid: validity masks (True = real token); valid tokens must form a prefix.
        max_prefix: at most this many prefix positions.
        suffix_from: token id where the suffix part starts (e.g. :data:`PI05_STATE_TOKEN`).

    Returns:
        ``(orig_position, expl_position)`` pairs, prefix first, in increasing order.
    """
    a = [int(t) for t in orig_tokens]
    b = [int(t) for t in expl_tokens]

    def n_valid(tokens, valid):
        if valid is None:
            return len(tokens)
        v = [bool(x) for x in valid]
        n = sum(v)
        if v[:n] != [True] * n:
            raise ValueError("valid tokens must form a prefix (padding only at the end)")
        return n

    na, nb = n_valid(a, orig_valid), n_valid(b, expl_valid)
    p = 0
    while p < min(na, nb) and a[p] == b[p]:
        p += 1
    if max_prefix is not None:
        p = min(p, max_prefix)
    s = 0
    while s < min(na, nb) - p and a[na - 1 - s] == b[nb - 1 - s]:
        s += 1
    start = na - s
    if suffix_from is not None:
        hits = [i for i in range(start, na) if a[i] == suffix_from]
        if not hits:
            raise ValueError(f"token {suffix_from} does not occur in the common suffix")
        start = hits[0]
    shift = nb - na
    return [(i, i) for i in range(p)] + [(i, i + shift) for i in range(start, na)]


def pi05_slot_map(orig_tokens: Sequence[int], expl_tokens: Sequence[int], orig_valid: Sequence[bool] | None = None,
                  expl_valid: Sequence[bool] | None = None) -> list[tuple[int, int]]:
    """Template-frame alignment for pi0.5 prompts ``"<bos>Task: {instruction}, State: {digits};\\nAction: "``.

    Pairs the 3 opening tokens and everything from ``State`` to the last valid token, i.e. the positions outside
    the instruction (40 pairs for the cube-then-banana prompts; the comma before ``State`` is not used).
    """
    return align_slots(orig_tokens, expl_tokens, orig_valid=orig_valid, expl_valid=expl_valid,
                       max_prefix=PI05_FRAME_PREFIX, suffix_from=PI05_STATE_TOKEN)


def build_keys_values(orig: Mapping[str, np.ndarray], expl: Mapping[str, np.ndarray], *, layers: Sequence[int],
                      mode: str = "image", alpha: float = 1.0, slot_map: Sequence[Sequence[int]] | None = None,
                      cells: Sequence[Sequence[int]] = IMAGE_CELLS,
                      capture_layers: Sequence[int] | None = None) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Keys and values for each edited layer, from paired captures of the same decisions.

    Args:
        orig, expl: capture arrays (as saved by ``owsp acts``) restricted to the same decisions in the same order
            (item axis first): ``vlm.mlp_hidden`` ``[N, Lc, P, F]`` and ``vlm.resid_out`` ``[N, Lc, P, D]`` for
            ``state``; ``vlm.img_mlp_hidden`` ``[N, Lc, 2, 17, F]`` and ``vlm.img_resid_out`` ``[N, Lc, 2, 17, D]``
            for ``image``. Keys use only the original capture's MLP hidden arrays.
        layers: model layers to edit.
        mode: ``"state"`` (aligned prompt/state slots), ``"image"`` (camera cells) or ``"both"``.
        alpha: value scale (1 = reproduce the explicit wording's increments).
        slot_map: ``(orig_index, expl_index)`` pairs on the P axis of the two captures (state/both modes).
        cells: ``(camera, cell)`` pairs; cell 0 is the camera mean, 1..16 the 4x4 grid (row-major from the top).
        capture_layers: model layer held at each index of the Lc axis (default ``0..Lc-1``).

    Returns:
        ``{L: (K [n, F], V [n, D])}`` in float64, rows ordered slots first, then cells.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    use_state, use_image = mode in ("state", "both"), mode in ("image", "both")
    if use_state and not slot_map:
        raise ValueError("state/both modes need a slot_map")
    slot_map = [(int(x), int(y)) for x, y in (slot_map or [])] if use_state else []
    cells = [(int(c), int(e)) for c, e in cells] if use_image else []
    ref = orig["vlm.resid_out" if use_state else "vlm.img_resid_out"]
    n_items, n_cap = ref.shape[0], ref.shape[1]
    for k in ((["vlm.mlp_hidden", "vlm.resid_out"] if use_state else [])
              + (["vlm.img_mlp_hidden", "vlm.img_resid_out"] if use_image else [])):
        src = (orig,) if "hidden" in k else (orig, expl)
        for arrs in src:
            if k not in arrs:
                raise KeyError(f"capture lacks {k!r} (needed for mode {mode!r})")
            if arrs[k].shape[0] != n_items:
                raise ValueError(f"{k}: {arrs[k].shape[0]} items vs {n_items}; captures must hold the same decisions")
    pos = {int(L): i for i, L in enumerate(capture_layers if capture_layers is not None else range(n_cap))}
    need = sorted(set(int(L) for L in layers) | {int(L) - 1 for L in layers if int(L) > 0})
    missing = [L for L in need if L not in pos]
    if missing:
        raise KeyError(f"capture lacks VLM layers {missing} (edited layers and the layer before each are needed)")

    def gap(L):
        li, rows = pos[L], []
        if use_state:
            Ro, Re = orig["vlm.resid_out"], expl["vlm.resid_out"]
            rows += [(Re[:, li, b].astype(np.float64) - Ro[:, li, a].astype(np.float64)).mean(0) for a, b in slot_map]
        if use_image:
            Io, Ie = orig["vlm.img_resid_out"], expl["vlm.img_resid_out"]
            rows += [(Ie[:, li, c, e].astype(np.float64) - Io[:, li, c, e].astype(np.float64)).mean(0)
                     for c, e in cells]
        return np.stack(rows)

    def keys(L):
        li, rows = pos[L], []
        if use_state:
            Ho = orig["vlm.mlp_hidden"]
            rows += [Ho[:, li, a].astype(np.float64).mean(0) for a, _ in slot_map]
        if use_image:
            Hi = orig["vlm.img_mlp_hidden"]
            rows += [Hi[:, li, c, e].astype(np.float64).mean(0) for c, e in cells]
        return np.stack(rows)

    V = layer_increments({L: gap(L) for L in need}, [int(L) for L in layers], alpha)
    return {int(L): (keys(int(L)), V[int(L)]) for L in layers}


def bf16_round(value: np.ndarray) -> np.ndarray:
    """float32 values after round-to-nearest-even bfloat16 rounding (the policy's storage precision).

    Identical to the ``bf16`` helper of the ``owsp py`` sandbox: float64 input is first rounded to float32.
    """
    value = np.asarray(value, dtype=np.float32)
    bits = value.view(np.uint32)
    rounded = bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))
    return (rounded & np.uint32(0xFFFF0000)).view(np.float32)


@dataclasses.dataclass
class LowRank:
    """A matrix stored as ``left @ right`` (``left [m, r]``, ``right [r, n]``).

    ``rel_error`` is the relative Frobenius error of this factorisation against the matrix it came from
    (0 for an exact factorisation such as ``K^T @ X`` from :func:`build_edit`).
    """

    left: np.ndarray
    right: np.ndarray
    rel_error: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return (self.left.shape[0], self.right.shape[1])

    @property
    def rank(self) -> int:
        return int(self.left.shape[1])

    @property
    def num_params(self) -> int:
        return int(self.left.size + self.right.size)

    def dense(self) -> np.ndarray:
        return self.left @ self.right


def low_rank(dW: np.ndarray | tuple[np.ndarray, np.ndarray], tol: float = 1e-6, max_rank: int | None = None) -> LowRank:
    """Truncated-SVD factorisation ``dW ~= left @ right`` with its exact reconstruction error.

    Keeps singular values ``> tol * s_max`` (and at most ``max_rank``). ``rel_error`` is
    ``||dW - left @ right||_F / ||dW||_F = sqrt(sum of discarded s^2 / sum of all s^2)``.

    Args:
        dW: a dense ``[m, n]`` matrix, or a factor pair ``(A [m, k], B [k, n])`` with ``dW = A @ B``; the pair is
            factorised through two thin QRs and a ``k x k`` SVD without ever forming ``dW``.
        tol: relative singular-value cutoff (0 keeps every nonzero direction).
        max_rank: optional cap on the rank.
    """
    if isinstance(dW, tuple):
        A, B = (np.asarray(x, dtype=np.float64) for x in dW)
        if A.ndim != 2 or B.ndim != 2 or A.shape[1] != B.shape[0]:
            raise ValueError(f"factor shapes {A.shape} @ {B.shape} do not chain")
        Qa, Ra = np.linalg.qr(A)
        Qb, Rb = np.linalg.qr(B.T)
        U, s, Vt = np.linalg.svd(Ra @ Rb.T)
        U, Vt = Qa @ U, Vt @ Qb.T
    else:
        U, s, Vt = np.linalg.svd(np.asarray(dW, dtype=np.float64), full_matrices=False)
    m, n = U.shape[0], Vt.shape[1]
    if s.size == 0 or s[0] == 0:
        return LowRank(np.zeros((m, 0)), np.zeros((0, n)), 0.0)
    r = int(np.sum(s > tol * s[0]))
    if max_rank is not None:
        r = min(r, int(max_rank))
    err = math.sqrt(float(np.sum(s[r:] ** 2)) / float(np.sum(s ** 2)))
    return LowRank(U[:, :r] * s[:r], Vt[:r], err)


@dataclasses.dataclass
class Edit:
    """Weight deltas keyed by tensor name (dense arrays or :class:`LowRank` factors) plus JSON metadata."""

    deltas: dict[str, np.ndarray | LowRank] = dataclasses.field(default_factory=dict)
    meta: dict[str, Any] = dataclasses.field(default_factory=dict)

    def names(self) -> list[str]:
        return sorted(self.deltas)

    def dense(self, name: str) -> np.ndarray:
        d = self.deltas[name]
        return d.dense() if isinstance(d, LowRank) else np.asarray(d)

    def apply(self, weights: Mapping[str, np.ndarray], *, round_bf16: bool = True) -> dict[str, np.ndarray]:
        """Edited copies ``W + dW`` (float32, rounded to bfloat16 by default) of the tensors this edit touches."""
        out = {}
        for name in self.names():
            W = np.asarray(weights[name])
            new = W + self.dense(name)
            out[name] = bf16_round(new) if round_bf16 else new.astype(np.float32)
        return out

    def num_params(self) -> int:
        """Stored values: factor entries for low-rank deltas, all entries for dense ones."""
        return int(sum(d.num_params if isinstance(d, LowRank) else np.asarray(d).size for d in self.deltas.values()))

    def stats(self) -> dict[str, dict[str, Any]]:
        out = {}
        for name in self.names():
            d = self.deltas[name]
            out[name] = {"shape": list(d.shape), "rank": d.rank if isinstance(d, LowRank) else None,
                         "stored": d.num_params if isinstance(d, LowRank) else int(np.asarray(d).size),
                         "rel_error": d.rel_error if isinstance(d, LowRank) else 0.0,
                         "fro": float(np.linalg.norm(self.dense(name)))}
        return out

    def to_low_rank(self, tol: float = 1e-6) -> Edit:
        """Same edit with every delta factorised by :func:`low_rank` (factor pairs re-truncated)."""
        out = {}
        for name, d in self.deltas.items():
            out[name] = low_rank((d.left, d.right) if isinstance(d, LowRank) else d, tol)
        return Edit(out, dict(self.meta))

    def save(self, path: str | Path) -> None:
        """Write an ``.npz`` (no pickles): ``dense/<name>``, ``left/<name>``, ``right/<name>`` and ``meta.json``."""
        arrays: dict[str, np.ndarray] = {}
        errs = {}
        for name, d in self.deltas.items():
            if isinstance(d, LowRank):
                arrays[f"left/{name}"], arrays[f"right/{name}"] = d.left, d.right
                errs[name] = d.rel_error
            else:
                arrays[f"dense/{name}"] = np.asarray(d)
        meta = {"meta": self.meta, "rel_error": errs, "format": f"weightsmith.distill.Edit/{VERSION}"}
        arrays["meta.json"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        with open(path, "wb") as f:
            np.savez(f, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> Edit:
        z = np.load(path, allow_pickle=False)
        meta = json.loads(bytes(z["meta.json"]).decode())
        deltas: dict[str, np.ndarray | LowRank] = {}
        for key in z.files:
            kind, _, name = key.partition("/")
            if kind == "dense":
                deltas[name] = z[key]
            elif kind == "left":
                deltas[name] = LowRank(z[key], z[f"right/{name}"], float(meta["rel_error"].get(name, 0.0)))
        return cls(deltas, meta["meta"])


def build_edit(layer_kv: Mapping[int, tuple[np.ndarray, np.ndarray]], lam_rel: float = 1e-3, *,
               tensor: str = MLP_DOWN, factored: bool = True, meta: Mapping[str, Any] | None = None) -> Edit:
    """Closed-form edit for every layer of ``layer_kv`` (``{L: (K, V)}`` from :func:`build_keys_values`).

    With ``factored=True`` each delta is kept as the exact factorisation ``K^T @ X`` with
    ``X = (K K^T + lam I)^-1 V`` (rank <= number of keys; ``dense()`` equals :func:`closed_form_edit`).
    """
    deltas: dict[str, np.ndarray | LowRank] = {}
    info = {}
    for L, (K, V) in sorted(layer_kv.items()):
        K = np.asarray(K, dtype=np.float64)
        V = np.asarray(V, dtype=np.float64)
        G = K @ K.T
        lam = lam_rel * np.trace(G) / len(G)
        X = np.linalg.solve(G + lam * np.eye(len(G)), V)
        name = tensor.format(int(L))
        deltas[name] = LowRank(K.T.copy(), X, 0.0) if factored else K.T @ X
        info[name] = {"layer": int(L), "keys": int(len(K)),
                      "fit_rel_err": float(np.linalg.norm(K @ (K.T @ X) - V) / max(np.linalg.norm(V), 1e-300))}
    return Edit(deltas, {"lam_rel": lam_rel, "layers": info, **dict(meta or {})})


# =============================================================================== 2. decision helpers
def gripper_closed(actions: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """True where an action chunk commands the gripper closed (most of its steps have gripper > threshold).

    ``actions``: ``[..., T, 8]`` chunks (7 joint targets, then the gripper command).
    """
    a = np.asarray(actions)
    return (a[..., 7] > threshold).mean(-1) >= 0.5


def panda_fk(q: Sequence[float]) -> np.ndarray:
    """End-effector position (m, robot base frame) of a Franka Panda for 7 joint angles (DH, flange + 0.257 m)."""
    dh = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
          (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
    T = np.eye(4)
    for i, (a, d, al) in enumerate(dh):
        th = float(q[i])
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
    F = np.eye(4)
    F[2, 3] = 0.257
    return (T @ F)[:3, 3]


def select_early_releases(actions: np.ndarray, gripper: np.ndarray, *, max_hold: int = 2, min_grip: float = 0.1,
                          max_grip: float = 0.9, first_only: bool = True) -> list[int]:
    """Replans where the policy opens the gripper on a held object shortly after grasping it.

    A release at replan ``k`` (chunk ``k`` open, chunk ``k-1`` closed) counts when the preceding run of closed
    chunks lasted at most ``max_hold`` replans and the measured gripper position at ``k`` lies in
    ``[min_grip, max_grip]`` (0 = open; ~1 = closed on nothing), i.e. an object is in the hand.

    Args:
        actions: ``[K, T, 8]`` chunks of one episode (``obs(eid)["actions"]``).
        gripper: ``[K]`` or ``[K, 1]`` measured gripper positions at each replan.
    """
    closed = gripper_closed(actions)
    g = np.asarray(gripper, dtype=np.float64).reshape(len(closed), -1)[:, 0]
    out, run = [], 0
    for k in range(len(closed)):
        if closed[k]:
            run += 1
            continue
        if 0 < run <= max_hold and min_grip <= g[k] <= max_grip:
            out.append(k)
            if first_only:
                break
        run = 0
    return out


def select_grasps(actions: np.ndarray, gripper: np.ndarray, *, max_open_grip: float = 0.05,
                  first_only: bool = True) -> list[int]:
    """Replans where the policy decides to grasp: chunk ``k`` closes after an open chunk, hand still open."""
    closed = gripper_closed(actions)
    g = np.asarray(gripper, dtype=np.float64).reshape(len(closed), -1)[:, 0]
    out = []
    for k in range(1, len(closed)):
        if closed[k] and not closed[k - 1] and g[k] <= max_open_grip:
            out.append(k)
            if first_only:
                break
    return out


def decision_summary(base: np.ndarray, expl: np.ndarray | None, cand: np.ndarray) -> dict[str, Any]:
    """Compare action chunks on the same decisions: base policy, explicit-wording replay, edited policy.

    Args:
        base, expl, cand: ``[N, T, 8]`` chunks (``expl`` may be None, e.g. for preservation items).

    Returns counts of gripper-closed decisions, flips vs base, agreement with the explicit replay, and mean
    chunk-end end-effector distances (cm) between the three.
    """
    cb, cc = gripper_closed(base), gripper_closed(cand)
    eb = np.array([panda_fk(a[-1, :7]) for a in np.asarray(base)])
    ec = np.array([panda_fk(a[-1, :7]) for a in np.asarray(cand)])
    out: dict[str, Any] = {"n": int(len(cb)), "closed_base": int(cb.sum()), "closed_cand": int(cc.sum()),
                           "flips_vs_base": int((cb != cc).sum()),
                           "ee_shift_cand_vs_base_cm": round(float(np.linalg.norm(ec - eb, axis=1).mean() * 100), 2)}
    if expl is not None:
        cx = gripper_closed(expl)
        ex = np.array([panda_fk(a[-1, :7]) for a in np.asarray(expl)])
        db = float(np.linalg.norm(eb - ex, axis=1).mean() * 100)
        dc = float(np.linalg.norm(ec - ex, axis=1).mean() * 100)
        out.update({"closed_expl": int(cx.sum()), "base_agrees_expl": int((cb == cx).sum()),
                    "cand_agrees_expl": int((cc == cx).sum()), "ee_base_to_expl_cm": round(db, 2),
                    "ee_cand_to_expl_cm": round(dc, 2)})
    return out


# =============================================================================== 3. sandbox scripts
@dataclasses.dataclass
class BuildConfig:
    """What to distil: two captures of the same decisions (original and explicit wording) and the edit's shape.

    ``orig_items`` / ``expl_items`` index the captures' item lists (None = all) and must name the same decisions
    pairwise. ``slot_map`` is ``"pi05"`` (template-frame alignment from the captured token ids) or explicit
    ``[orig_position, expl_position]`` pairs (prompt positions, not capture indices).
    """

    orig_capture: str
    expl_capture: str
    layers: list[int]
    mode: str = "image"
    alpha: float = 1.0
    lam_rel: float = 1e-3
    orig_items: list[int] | None = None
    expl_items: list[int] | None = None
    slot_map: Any = "pi05"
    cells: list[list[int]] | None = None
    instruction: str | None = None
    explicit_instruction: str | None = None
    tensor: str = MLP_DOWN
    compose: bool = False  # True: add to the candidate's current edits instead of the base weights
    save_factors: bool = False  # also write the exact low-rank factors to /work/scratch/distill_factors.npz
    compare: bool = False  # True: do not save; compare the result with the candidate's stored tensors

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.layers = [int(L) for L in self.layers]
        bad = [L for L in self.layers if L not in EDITABLE_LAYERS]
        if not self.layers or bad or len(set(self.layers)) != len(self.layers):
            raise ValueError(f"layers must be distinct values in 0..16, got {self.layers}")
        if not (math.isfinite(self.alpha) and self.lam_rel > 0):
            raise ValueError("alpha must be finite and lam_rel > 0")
        if self.orig_items is not None and self.expl_items is not None and len(self.orig_items) != len(self.expl_items):
            raise ValueError("orig_items and expl_items must have equal length")
        if self.mode != "image" and not (self.slot_map == "pi05" or isinstance(self.slot_map, (list, tuple))):
            raise ValueError("slot_map must be 'pi05' or a list of [orig_position, expl_position] pairs")

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> BuildConfig:
        names = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - names
        if unknown:
            raise ValueError(f"unknown build config keys: {sorted(unknown)}")
        return cls(**dict(d))

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def _sb_load_capture(data_root, aid, keys):
    """(index, {key: array}) of capture ``aid``; only the requested arrays are read from the npz."""
    idx = json.load(open(f"{data_root}/acts/{aid}/index.json"))
    z = np.load(f"{data_root}/acts/{aid}/acts.npz")
    return idx, {k: z[k] for k in keys if k in z.files}


def _sb_build(cfg, io):
    """Sandbox body of the build script: captures -> keys/values -> dW per layer -> save (or compare)."""
    import hashlib as _hl
    mode, layers = cfg["mode"], [int(L) for L in cfg["layers"]]
    use_state, use_image = mode in ("state", "both"), mode in ("image", "both")
    keys = (["vlm.mlp_hidden", "vlm.resid_out"] if use_state else []) + \
           (["vlm.img_mlp_hidden", "vlm.img_resid_out"] if use_image else [])
    idx_o, arr_o = _sb_load_capture(io["data_root"], cfg["orig_capture"], keys)
    idx_x, arr_x = _sb_load_capture(io["data_root"], cfg["expl_capture"], keys)
    oi = cfg.get("orig_items") or list(range(len(idx_o["items"])))
    xi = cfg.get("expl_items") or list(range(len(idx_x["items"])))
    if len(oi) != len(xi):
        raise ValueError(f"{len(oi)} original vs {len(xi)} explicit items")
    for a, b in zip(oi, xi, strict=True):
        ia, ib = idx_o["items"][a], idx_x["items"][b]
        if (ia["episode"], ia["k"]) != (ib["episode"], ib["k"]):
            raise ValueError(f"item {a} vs {b}: different decisions {ia['episode']}/{ia['k']} and "
                             f"{ib['episode']}/{ib['k']}")
        if cfg.get("instruction") and ia["prompt"] != cfg["instruction"]:
            raise ValueError(f"original item {a} was captured with {ia['prompt']!r}, not {cfg['instruction']!r}")
        if cfg.get("explicit_instruction") and ib["prompt"] != cfg["explicit_instruction"]:
            raise ValueError(f"explicit item {b} was captured with {ib['prompt']!r}, "
                             f"not {cfg['explicit_instruction']!r}")
    if idx_o["vlm_layers"] != idx_x["vlm_layers"]:
        raise ValueError("the two captures hold different VLM layers")
    for k in keys:
        if k not in arr_o or k not in arr_x:
            raise KeyError(f"capture lacks {k} (capture with image_tokens for image modes, "
                           "prompt_tokens for state modes)")
    orig = {k: v[oi] for k, v in arr_o.items()}
    expl = {k: v[xi] for k, v in arr_x.items()}
    slot_map, slot_info = None, {}
    if use_state:
        so, sx = idx_o["prompt_token_slots"], idx_x["prompt_token_slots"]
        if cfg.get("slot_map", "pi05") == "pi05":
            t0, x0 = idx_o["items"][oi[0]], idx_x["items"][xi[0]]
            if so != list(range(len(so))) or sx != list(range(len(sx))) or t0["prompt_token_valid_at_slots"][-1] \
                    or x0["prompt_token_valid_at_slots"][-1]:
                raise ValueError("slot_map 'pi05' needs captures of prompt slots 0..P-1 covering the whole prompt")
            pos = pi05_slot_map(t0["prompt_token_ids_at_slots"], x0["prompt_token_ids_at_slots"],
                                t0["prompt_token_valid_at_slots"], x0["prompt_token_valid_at_slots"])
        else:
            pos = [(int(a), int(b)) for a, b in cfg["slot_map"]]
        slot_map = [(so.index(a), sx.index(b)) for a, b in pos]
        same = []
        for a, b in slot_map:  # per decision: same token at both positions and both real tokens
            ok = [io_["prompt_token_ids_at_slots"][a] == ix_["prompt_token_ids_at_slots"][b]
                  and io_["prompt_token_valid_at_slots"][a] and ix_["prompt_token_valid_at_slots"][b]
                  for io_, ix_ in ((idx_o["items"][i], idx_x["items"][j]) for i, j in zip(oi, xi, strict=True))]
            same.append(sum(ok) / len(ok))
        slot_info = {"slot_positions": pos, "min_slot_consistency": min(same) if same else None,
                     "slots_not_fully_consistent": int(sum(s < 1 for s in same))}
    cells = cfg.get("cells") or [[c, e] for c in (0, 1) for e in range(1, 17)]
    kv = build_keys_values(orig, expl, layers=layers, mode=mode, alpha=float(cfg.get("alpha", 1.0)),
                           slot_map=slot_map, cells=cells, capture_layers=idx_o["vlm_layers"])
    tensor = cfg.get("tensor") or MLP_DOWN
    report, factors, n_factor = [], {}, 0
    for L in layers:
        K, V = kv[L]
        dW = closed_form_edit(K, V, float(cfg.get("lam_rel", 1e-3)))
        name = tensor.format(L)
        W0 = io["load"](name) if cfg.get("compose") else io["base"](name)
        new = io["bf16"](W0 + dW)
        row = {"layer": L, "tensor": name, "keys": int(len(K)),
               "fit_rel_err": float(np.linalg.norm(K @ dW - V) / max(np.linalg.norm(V), 1e-300)),
               "dW_rel_norm": float(np.linalg.norm(dW) / np.linalg.norm(W0)),
               "changed_fraction": float(np.count_nonzero(new != W0) / new.size),
               "sha256": _hl.sha256(np.ascontiguousarray(new, dtype=np.float32).tobytes()).hexdigest()[:16]}
        if cfg.get("compare"):
            stored = io["load"](name).astype(np.float32)
            diff = np.abs(new.astype(np.float64) - stored)
            row.update({"stored_sha256": _hl.sha256(np.ascontiguousarray(stored).tobytes()).hexdigest()[:16],
                        "max_abs_diff_vs_stored": float(diff.max()), "n_diff_vs_stored": int(np.count_nonzero(diff)),
                        "stored_max_abs_delta": float(np.abs(stored.astype(np.float64) - W0).max())})
        else:
            io["save"](name, new)
        if cfg.get("save_factors"):
            G = K @ K.T
            X = np.linalg.solve(G + float(cfg.get("lam_rel", 1e-3)) * np.trace(G) / len(G) * np.eye(len(G)), V)
            factors[f"left/{name}"], factors[f"right/{name}"] = K.T.astype(np.float32), X.astype(np.float32)
        report.append(row)
        n_factor += K.shape[0] * (K.shape[1] + V.shape[1])
        vs_stored = (f"  vs stored: max|diff| {row['max_abs_diff_vs_stored']:.3g}, {row['n_diff_vs_stored']:d} differ"
                     if cfg.get("compare") else "")
        print(f"L{L:2d} {name!s:<34} keys {row['keys']:3d}  fit-rel-err {row['fit_rel_err']:.2e}  "
              f"|dW|/|W| {row['dW_rel_norm']:.4f}  changed {100 * row['changed_fraction']:.1f}%{vs_stored}", flush=True)
    summary = {"config": cfg, "layers": report, "n_items": len(oi), **slot_info,
               "factor_params": int(n_factor)}
    if io.get("saved") and not cfg.get("compare"):
        written = {r["tensor"] for r in report}
        kept = sorted(n for n in io["saved"]() if n not in written)
        summary["other_edits_kept"] = kept
        if kept:
            print(f"note: the candidate also keeps {len(kept)} earlier edits that this build did not write: "
                  f"{kept[:6]}{' ...' if len(kept) > 6 else ''}", flush=True)
    if factors and io.get("scratch"):
        np.savez(f"{io['scratch']}/distill_factors.npz", **factors)
    if io.get("scratch") and not cfg.get("compare"):
        with open(f"{io['scratch']}/distill_manifest.json", "w") as f:
            json.dump(summary, f, indent=1)
    return summary


def _sb_check(cfg, io):
    """Sandbox body of the offline check: gripper decisions and end-effector targets, base vs explicit vs edit."""
    def actions(aid, items):
        idx, arr = _sb_load_capture(io["data_root"], aid, ["actions"])
        items = items if items is not None else list(range(len(idx["items"])))
        return [(idx["items"][i]["episode"], idx["items"][i]["k"]) for i in items], arr["actions"][items]
    out = {}
    ko, base = actions(cfg["orig_capture"], cfg.get("orig_items"))
    kx, expl = actions(cfg["expl_capture"], cfg.get("expl_items"))
    kc, cand = actions(cfg["cand_capture"], cfg.get("cand_items"))
    if not (ko == kx == kc):
        raise ValueError("decision items differ between the original, explicit and candidate captures")
    out["decisions"] = decision_summary(base, expl, cand)
    if cfg.get("check_base_capture"):
        kb, pb = actions(cfg["check_base_capture"], cfg.get("check_base_items"))
        kc2, pc = actions(cfg["cand_capture"], cfg.get("check_cand_items"))
        if kb != kc2:
            raise ValueError("check items differ between the base and candidate captures")
        groups = cfg.get("check_groups") or {"check": [0, len(kb)]}
        out["check"] = {g: decision_summary(pb[lo:hi], None, pc[lo:hi]) for g, (lo, hi) in groups.items()}
    d = out["decisions"]
    print(f"decisions (n={d['n']:d}): gripper closed  base {d['closed_base']:d} | explicit {d['closed_expl']:d} | "
          f"edit {d['closed_cand']:d}   flips vs base {d['flips_vs_base']:d}   edit agrees with "
          f"explicit {d['cand_agrees_expl']:d}/{d['n']:d} (base {d['base_agrees_expl']:d}/{d['n']:d})")
    print(f"  chunk-end end effector: base->explicit {d['ee_base_to_expl_cm']:.1f} cm, "
          f"edit->explicit {d['ee_cand_to_expl_cm']:.1f} cm, edit moved {d['ee_shift_cand_vs_base_cm']:.1f} cm")
    for g, c in out.get("check", {}).items():
        print(f"check {g!s:<26} (n={c['n']:2d}): gripper closed base {c['closed_base']:2d} | edit {c['closed_cand']:2d}"
              f"   flips {c['flips_vs_base']:2d}   edit moved {c['ee_shift_cand_vs_base_cm']:.1f} cm")
    return out


def _sb_select(rule, io):
    """Sandbox body of item selection: scan practice episodes and pick decision replans by rule."""
    rows = io["episodes"](cand=rule.get("cand", "base"), success=rule.get("success", False))
    rows = sorted(rows, key=lambda r: r["eid"])
    exclude = {(str(e), int(k)) for e, k in rule.get("exclude") or []}
    items, scanned = [], 0
    for r in rows:
        if rule.get("episodes") and r["eid"] not in rule["episodes"]:
            continue
        o = io["obs"](r["eid"])
        scanned += 1
        if rule["rule"] == "early_release":
            ks = select_early_releases(o["actions"], o["gripper"], max_hold=int(rule.get("max_hold", 2)),
                                       min_grip=float(rule.get("min_grip", 0.1)),
                                       max_grip=float(rule.get("max_grip", 0.9)),
                                       first_only=bool(rule.get("first_only", True)))
        elif rule["rule"] == "grasp":
            ks = select_grasps(o["actions"], o["gripper"], max_open_grip=float(rule.get("max_open_grip", 0.05)),
                               first_only=bool(rule.get("first_only", True)))
        else:
            raise ValueError(f"unknown selection rule {rule['rule']!r}")
        for k in ks:
            if (r["eid"], int(k)) in exclude:
                continue
            items.append({"episode": r["eid"], "k": int(k), "step": int(o["step"][k]),
                          "gripper": round(float(np.asarray(o["gripper"][k]).ravel()[0]), 3)})
    lim = rule.get("limit")
    if lim:
        items = items[:int(lim)]
    print(f"scanned {scanned:d} episodes, selected {len(items):d} decisions")
    for it in items:
        print(f"  {it['episode']!s} k={it['k']:d} step={it['step']:d} gripper={it['gripper']:.3f}")
    return {"items": [{"episode": it["episode"], "k": it["k"]} for it in items], "details": items, "scanned": scanned}


def _sb_responsive(cfg, io):
    """Sandbox body: the decisions at which the two wordings disagree on the gripper (closed vs open)."""
    idx_o, arr_o = _sb_load_capture(io["data_root"], cfg["orig_capture"], ["actions"])
    idx_x, arr_x = _sb_load_capture(io["data_root"], cfg["expl_capture"], ["actions"])
    co, cx = gripper_closed(arr_o["actions"]), gripper_closed(arr_x["actions"])
    keep = []
    print(f"{'decision':<28} original  explicit")
    for i, (a, b) in enumerate(zip(idx_o["items"], idx_x["items"], strict=True)):
        if (a["episode"], a["k"]) != (b["episode"], b["k"]):
            raise ValueError(f"item {i}: the two captures hold different decisions")
        if co[i] != cx[i]:
            keep.append(i)
        print(f"{a['episode'] + ' k=' + str(a['k']):<28} {'CLOSE' if co[i] else 'open':<9} "
              f"{'CLOSE' if cx[i] else 'open':<9}{'<- the wording changes the decision' if co[i] != cx[i] else ''}")
    print(f"{len(keep)} of {len(co)} decisions change with the wording")
    return {"responsive": keep, "n": int(len(co))}


def _sb_fingerprint(cfg, io):
    """Sandbox body: the candidate's ``owsp`` fingerprint (sha256 over its edit files, as ``owsp eval`` prints)."""
    import glob as _gl
    import hashlib as _hl
    h = _hl.sha256()
    for f in sorted(_gl.glob(f"{cfg['edits']}/*.npy")):
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            h.update(fh.read())
    return {"fingerprint": h.hexdigest()[:16]}


def _sb_capture_items(cfg, io):
    """Sandbox body: the decisions ``[episode, k, prompt]`` held by existing captures (read from index.json)."""
    out = {}
    for aid in cfg["captures"]:
        with open(f"{io['data_root']}/acts/{aid}/index.json") as f:
            out[aid] = [[it["episode"], int(it["k"]), it["prompt"]] for it in json.load(f)["items"]]
    return out


_CORE_FUNCS: list[Callable[..., Any]] = [closed_form_edit, layer_increments, align_slots, pi05_slot_map,
                                         build_keys_values, gripper_closed, panda_fk, select_early_releases,
                                         select_grasps, decision_summary, _sb_load_capture]
_MARK = "DISTILL_JSON "


def _render(body: Callable[..., Any], cfg: Mapping[str, Any], *, data_root: str, scratch: str, title: str) -> str:
    cfg_json = json.dumps(cfg, sort_keys=True)
    digest = hashlib.sha256(cfg_json.encode()).hexdigest()[:12]
    parts = [
        f"# {title}",
        f"# Generated by weightsmith.distill v{VERSION} (config sha256 {digest}). Runs inside `owsp py`, whose",
        "# prelude provides np, json, os, load, base, save, bf16 (and episodes, obs, acts).",
        "from typing import Any, Callable, Mapping, Sequence",
        "import math",
        f"MLP_DOWN = {MLP_DOWN!r}",
        f"MODES = {MODES!r}",
        f"PI05_FRAME_PREFIX = {PI05_FRAME_PREFIX!r}",
        f"PI05_STATE_TOKEN = {PI05_STATE_TOKEN!r}",
        f"IMAGE_CELLS = {IMAGE_CELLS!r}",
    ]
    parts += [inspect.getsource(f) for f in _CORE_FUNCS + [body]]
    parts += [
        f"CFG = json.loads({cfg_json!r})",
        "_IO = dict(load=globals().get('load'), base=globals().get('base'), save=globals().get('save'),",
        "           bf16=globals().get('bf16'), episodes=globals().get('episodes'), obs=globals().get('obs'),",
        "           saved=globals().get('saved'),",
        f"           data_root={data_root!r}, scratch={scratch!r})",
        f"_RESULT = {body.__name__}(CFG, _IO)",
        f"print({_MARK!r} + json.dumps(_RESULT, default=float))",
    ]
    return "\n\n".join(parts) + "\n"


def render_sandbox_script(cfg: BuildConfig | Mapping[str, Any], *, data_root: str = "/work/data",
                          scratch: str = "/work/scratch") -> str:
    """Standalone ``owsp py CAND`` script that builds the distilled edit from two captures and saves it.

    The script inlines the needed functions of this module (the sandbox cannot import the repo), prints one line
    per layer, writes ``/work/scratch/distill_manifest.json`` and ends with ``DISTILL_JSON {...}``.
    """
    c = cfg if isinstance(cfg, BuildConfig) else BuildConfig.from_dict(cfg)
    return _render(_sb_build, c.to_dict(), data_root=data_root, scratch=scratch,
                   title=f"distill build: mode={c.mode} layers={c.layers} alpha={c.alpha} lam_rel={c.lam_rel}")


def render_compare_script(cfg: BuildConfig | Mapping[str, Any], *, data_root: str = "/work/data",
                          scratch: str = "/work/scratch") -> str:
    """Like :func:`render_sandbox_script` but saves nothing: run it in a candidate's sandbox to compare the
    re-derived tensors with that candidate's stored ones (max |diff|, number of differing values, digests)."""
    c = cfg if isinstance(cfg, BuildConfig) else BuildConfig.from_dict(cfg)
    d = {**c.to_dict(), "compare": True, "save_factors": False}
    return _render(_sb_build, d, data_root=data_root, scratch=scratch,
                   title=f"distill compare: mode={c.mode} layers={c.layers}")


def render_check_script(cfg: Mapping[str, Any], *, data_root: str = "/work/data") -> str:
    """``owsp py`` script for the offline check (see :func:`_sb_check` for the config keys)."""
    return _render(_sb_check, dict(cfg), data_root=data_root, scratch="", title="distill offline check")


def render_select_script(rule: Mapping[str, Any], *, data_root: str = "/work/data") -> str:
    """``owsp py base`` script that selects decision items from practice episodes.

    Rules: ``{"rule": "early_release", "max_hold": 2, "min_grip": 0.1, "max_grip": 0.9}`` (grasp then drop) or
    ``{"rule": "grasp", "max_open_grip": 0.05}``; common keys ``cand`` (default "base"), ``success`` (default
    false: failed episodes; null: all episodes), ``first_only`` (default true), ``limit``, ``episodes`` (restrict
    to these ids), ``exclude`` (``[episode, k]`` pairs never to select, e.g. the decisions an edit is built from).
    """
    return _render(_sb_select, dict(rule), data_root=data_root, scratch="", title=f"distill select: {rule.get('rule')}")


def parse_result(output: str) -> dict[str, Any]:
    """The ``DISTILL_JSON`` payload printed by a rendered script (raises with the output tail if absent)."""
    for line in reversed(output.splitlines()):
        if line.startswith(_MARK):
            return json.loads(line[len(_MARK):])
    raise RuntimeError("sandbox script produced no result:\n" + output[-3000:])


# =============================================================================== 4. pipeline
def capture_specs(cfg: Mapping[str, Any], items: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The two ``owsp acts`` specs of a distillation: original wording and explicit wording on the same items."""
    e = cfg["edit"]
    mode, layers = e.get("mode", "image"), [int(L) for L in e["layers"]]
    vlm = sorted(set(layers) | {L - 1 for L in layers if L > 0})
    common = {"items": [dict(it) for it in items], "vlm_layers": vlm, "image_tokens": mode != "state",
              "prompt_tokens": list(range(64)) if mode != "image" else [0],
              "action_layers": [17], "action_tokens": [0], "denoise_steps": [0]}
    return [{"id": f"{cfg['name']}-o", "cand": "base", "prompt": cfg["instruction"], **common},
            {"id": f"{cfg['name']}-x", "cand": "base", "prompt": cfg["explicit_instruction"], **common}]


def _check_item_groups(cfg: Mapping[str, Any]) -> dict[str, Any]:
    ci = cfg.get("check_items") or []
    return dict(ci) if isinstance(ci, Mapping) else ({"check": ci} if ci else {})


def check_groups(cfg: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, list[int]]]:
    """Flatten ``cfg['check_items']`` (a list, or ``{group: [items]}``) into items plus ``{group: [lo, hi]}``.

    Groups given as a selection rule (a dict) must be resolved to items first (:func:`run_pipeline` does that).
    """
    flat: list[dict[str, Any]] = []
    ranges: dict[str, list[int]] = {}
    for g, its in _check_item_groups(cfg).items():
        if isinstance(its, Mapping):
            raise ValueError(f"check group {g!r} is a selection rule; resolve it to items first")
        ranges[g] = [len(flat), len(flat) + len(its)]
        flat += [dict(it) for it in its]
    return flat, ranges


def check_specs(cfg: Mapping[str, Any], items: Sequence[Mapping[str, Any]], cand: str,
                fingerprint: str | None = None) -> list[dict[str, Any]]:
    """Action-only captures for the offline check: the candidate on decision + check items, base on check items.

    With the candidate's ``fingerprint`` the candidate capture id is ``<name>-c-<fp[:6]>``, so a rebuilt edit is
    never checked against the replay of an earlier version.
    """
    light = {"prompt": cfg["instruction"], "vlm_layers": [], "action_layers": [17], "action_tokens": [0],
             "denoise_steps": [0]}
    check, _ = check_groups(cfg)
    cid = f"{cfg['name']}-c" + (f"-{fingerprint[:6]}" if fingerprint else "")
    specs = [{"id": cid, "cand": cand, "items": [dict(it) for it in items] + check, **light}]
    if check:
        specs.append({"id": f"{cfg['name']}-cb", "cand": "base", "items": check, **light})
    return specs


def _q(path: str) -> str:
    """Shell-quoted path; the dry-run placeholder ``$WSP_ROOT`` stays expandable."""
    return '"$WSP_ROOT"' if path == "$WSP_ROOT" else shlex.quote(path)


class Owsp:
    """Runs ``bin/owsp`` for one project (``WSP_ROOT``); in dry-run mode only prints what it would run."""

    def __init__(self, root: str, owsp_bin: str | Path, workdir: Path, dry_run: bool = False):
        self.root, self.bin, self.workdir, self.dry_run = root, str(owsp_bin), workdir, dry_run

    def run(self, *args: str, file_text: str | None = None, file_name: str | None = None) -> str | None:
        argv = [self.bin, *args]
        if file_text is not None:
            path = self.workdir / (file_name or "input.txt")
            path.write_text(file_text)
            argv.append(str(path))
        shown = f"WSP_ROOT={_q(self.root)} " + " ".join(shlex.quote(a) for a in argv)
        if self.dry_run:
            print(f"[dry-run] {shown}")
            return None
        print(f"$ {shown}", flush=True)
        p = subprocess.run(argv, env=dict(os.environ, WSP_ROOT=self.root), capture_output=True, text=True)
        out = p.stdout + p.stderr
        if p.returncode:
            raise RuntimeError(f"owsp {args[0]} failed (exit {p.returncode}):\n{out[-4000:]}")
        return out


def _setting(key: str, repo: Path) -> str:
    """A bin/owsp setting: the environment wins, then the first ``KEY=VALUE`` line of wsp.env (or $WSP_ENV_FILE)."""
    if key in os.environ:
        return os.environ[key]
    path = Path(os.environ.get("WSP_ENV_FILE") or repo / "wsp.env")
    if path.is_file():
        for line in path.read_text().splitlines():
            k, sep, v = line.strip().partition("=")
            if sep and k.strip().removeprefix("export ").strip() == key:
                return v.strip().strip("\"'")
    return ""


def _load_config(path: str | Path) -> dict[str, Any]:
    text = Path(path).read_text()
    if str(path).endswith((".yaml", ".yml")):
        import yaml  # optional dependency, only for YAML configs
        return yaml.safe_load(text)
    return json.loads(text)


def _reusable_captures(ow: Owsp, specs: Sequence[Mapping[str, Any]], existing: Sequence[str],
                       project: str) -> set[str]:
    """Ids of ``specs`` that already exist and hold exactly the decisions and wordings of the spec.

    Raises if an existing capture holds other decisions: reusing it would silently build or check the edit on
    stale data (e.g. after the selection rule changed).
    """
    old = [s for s in specs if s["id"] in existing]
    if not old:
        return set()
    script = _render(_sb_capture_items, {"captures": [s["id"] for s in old]}, data_root="/work/data", scratch="",
                     title="distill: decisions held by existing captures")
    out = ow.run("py", "base", file_text=script, file_name="existing-captures.py")
    if out is None:
        return {s["id"] for s in old}
    have = parse_result(out)
    for s in old:
        want = [(it["episode"], it.get("k"), it.get("prompt") or s.get("prompt")) for it in s["items"]]
        got = have[s["id"]]
        same = len(want) == len(got) and all(
            e == g[0] and (k is None or int(k) == g[1]) and (p is None or p == g[2])
            for (e, k, p), g in zip(want, got, strict=True))
        if not same:
            raise RuntimeError(
                f"capture {s['id']!r} already exists but holds other decisions than this run selected. Give the "
                f"config a new 'name' (capture ids are '<name>-o', '<name>-x', '<name>-c', '<name>-cb'), or move "
                f"{project}/data/acts/{s['id']} away.")
        print(f"capture {s['id']} exists with the same {len(want)} decisions: reusing it")
    return {s["id"] for s in old}


def _resolve_check_items(cfg: Mapping[str, Any], items: Sequence[Mapping[str, Any]], ow: Owsp, wd: Path, *,
                         fresh: bool) -> dict[str, list[dict[str, Any]]]:
    """``cfg['check_items']`` with every rule group (a dict) replaced by the items its rule selects.

    Rule groups run the select script in the base sandbox and never pick the build decisions ``items``. The
    result is cached in ``wd/check_items.json`` and reused unless ``fresh`` (the select step ran again).
    """
    groups = _check_item_groups(cfg)
    rules = {g: r for g, r in groups.items() if isinstance(r, Mapping)}
    cache = wd / "check_items.json"
    found: dict[str, Any] = {}
    if rules and not fresh and cache.exists():
        found = json.loads(cache.read_text())
    exclude = [[it["episode"], int(it["k"])] for it in items]
    for g, rule in rules.items():
        if g in found:
            continue
        out = ow.run("py", "base", file_text=render_select_script({**rule, "exclude": exclude}),
                     file_name=f"select-check-{g}.py")
        if out is None:
            found[g] = [{"episode": f"<{g}-episode>", "k": 0}]
            continue
        found[g] = parse_result(out)["items"]
        print(f"check group {g}: {len(found[g])} decisions selected by rule {rule.get('rule')!r}")
    if rules and not ow.dry_run:
        cache.write_text(json.dumps({g: found[g] for g in rules}, indent=1))
    return {g: [dict(it) for it in (found[g] if g in rules else v)] for g, v in groups.items()}


def _build_config(cfg: Mapping[str, Any]) -> BuildConfig:
    e = dict(cfg["edit"])
    e.setdefault("instruction", cfg["instruction"])
    e.setdefault("explicit_instruction", cfg["explicit_instruction"])
    return BuildConfig(orig_capture=f"{cfg['name']}-o", expl_capture=f"{cfg['name']}-x", **e)


def run_pipeline(cfg: Mapping[str, Any], *, steps: Sequence[str] = ("select", "capture", "build", "check"),
                 dry_run: bool = False, workdir: str | Path | None = None,
                 owsp_bin: str | Path | None = None) -> dict[str, Any]:
    """Distil ``cfg['explicit_instruction']`` into a candidate of project ``cfg['project']``.

    Steps: ``select`` (pick decision replans by rule, unless ``items`` are given), ``capture`` (two ``owsp acts``
    replays of the decisions: original and explicit wording), ``build`` (``owsp new`` + the rendered build script),
    ``check`` (action-only replays of the edited policy, compared offline). Ends by printing the closed-loop and
    held-out commands. Existing captures with the same ids are reused. Returns what each step produced.
    """
    for k in ("name", "instruction", "explicit_instruction", "edit"):
        if k not in cfg:
            raise KeyError(f"config needs {k!r}")
    project = cfg.get("project") or os.environ.get("WSP_ROOT")
    if not project:
        if not dry_run:
            raise KeyError("set WSP_ROOT to the project root on the GPU host (or give 'project' in the config)")
        project = "$WSP_ROOT"
    name, cand = cfg["name"], cfg.get("candidate") or cfg["name"]
    wd = Path(workdir or Path("distill-runs") / name)
    wd.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[1]
    ow = Owsp(project, owsp_bin or repo / "bin" / "owsp", wd, dry_run)
    bcfg = _build_config(cfg)
    print(f"project {project} ({'config' if cfg.get('project') else 'WSP_ROOT'}), candidate {cand}, work dir {wd}")
    result: dict[str, Any] = {"candidate": cand, "workdir": str(wd), "project": project}
    status = {"acts": [], "candidates": [], "instruction": cfg["instruction"]}
    out = ow.run("status")
    if out is not None:
        status = json.loads(out[out.index("{"):])
        if status["instruction"].strip().lower() != cfg["instruction"].strip().lower():
            raise ValueError(f"project instruction {status['instruction']!r} != config {cfg['instruction']!r}")

    items = cfg.get("items")
    if not items and "select" in steps:
        if "select" not in cfg:
            raise KeyError("config needs 'items' or a 'select' rule")
        out = ow.run("py", "base", file_text=render_select_script(cfg["select"]), file_name="select.py")
        if out is None:
            print(f"[dry-run] decision items would come from {wd / 'select.py'}; a placeholder item is used below")
            items = [{"episode": "<selected-episode>", "k": 0}]
        else:
            print(out[:3000])
            items = parse_result(out)["items"]
            (wd / "items.json").write_text(json.dumps(items, indent=1))
    elif not items:
        if not (wd / "items.json").exists():
            raise KeyError("no 'items' in the config and no items.json from an earlier select step")
        items = json.loads((wd / "items.json").read_text())
    result["items"] = items
    print(f"{len(items)} decision items")
    if "check" in steps or ("select" in steps and not cfg.get("items")):
        cfg = {**cfg, "check_items": _resolve_check_items(cfg, items, ow, wd, fresh="select" in steps)}
        result["check_items"] = {g: len(v) for g, v in cfg["check_items"].items()}

    if "capture" in steps:
        specs = capture_specs(cfg, items)
        reuse = _reusable_captures(ow, specs, status["acts"], project)
        for spec in specs:
            if spec["id"] in reuse:
                continue
            if dry_run:
                print(json.dumps({k: v for k, v in spec.items() if k != "items"}), f"+ {len(spec['items'])} items")
            out = ow.run("acts", file_text=json.dumps(spec, indent=1), file_name=f"{spec['id']}.json")
            if out:
                print(out.strip()[-600:])

    if "build" in steps:
        if cfg.get("responsive_only") and bcfg.orig_items is None and bcfg.expl_items is None:
            out = ow.run("py", "base", file_text=_render(
                _sb_responsive, {"orig_capture": bcfg.orig_capture, "expl_capture": bcfg.expl_capture},
                data_root="/work/data", scratch="", title="distill: decisions that change with the wording"),
                file_name="responsive.py")
            if out is not None:
                print(out[:3000].split(_MARK)[0].rstrip())
                keep = parse_result(out)["responsive"]
                if not keep:
                    raise RuntimeError("the explicit wording changes none of the selected decisions, so there is "
                                       "nothing to distil: try other decisions, another rewording, or another edit")
                bcfg = dataclasses.replace(bcfg, orig_items=keep, expl_items=keep)
                result["responsive"] = keep
                print(f"the edit is built from the {len(keep)} decisions that change with the wording")
        script = render_sandbox_script(bcfg)
        if cand not in status["candidates"]:
            ow.run("new", cand, "--note", f"distill {name}: {json.dumps(cfg['edit'], sort_keys=True)}")
        else:
            print(f"candidate {cand} exists; its edits will be overwritten tensor by tensor")
        out = ow.run("py", cand, file_text=script, file_name="build.py")
        if out is not None:
            print(out[:6000])
            result["build"] = parse_result(out)
            (wd / "build.json").write_text(json.dumps(result["build"], indent=1, default=float))
        else:
            print(f"[dry-run] build script: {wd / 'build.py'} ({len(script.splitlines())} lines)")

    if "check" in steps:
        fp = None
        out = ow.run("py", cand, file_text=_render(_sb_fingerprint, {"edits": "/work/edits"}, data_root="/work/data",
                                                    scratch="", title="distill: candidate fingerprint"),
                     file_name="fingerprint.py")
        if out is not None:
            fp = parse_result(out)["fingerprint"]
            result["fingerprint"] = fp
            print(f"candidate {cand} fingerprint {fp}")
        specs = check_specs(cfg, items, cand, fp)
        reuse = _reusable_captures(ow, specs, status["acts"], project)
        for spec in specs:
            if spec["id"] in reuse:
                continue
            ow.run("acts", file_text=json.dumps(spec, indent=1), file_name=f"{spec['id']}.json")
        n = len(items)
        ccfg = {"orig_capture": f"{name}-o", "expl_capture": f"{name}-x", "cand_capture": specs[0]["id"],
                "cand_items": list(range(n))}
        flat, ranges = check_groups(cfg)
        if flat:
            ccfg.update({"check_base_capture": f"{name}-cb", "check_cand_items": list(range(n, n + len(flat))),
                         "check_groups": ranges})
        out = ow.run("py", "base", file_text=render_check_script(ccfg), file_name="check.py")
        if out is not None:
            print(out[:3000])
            result["check"] = parse_result(out)
            (wd / "check.json").write_text(json.dumps(result["check"], indent=1))

    ev = cfg.get("eval", {})
    ho = cfg.get("heldout", {})
    root, n, off = project, ev.get("n", 32), ev.get("offset", 0)
    ho_start, ho_n = ho.get("slot_start", 90000), ho.get("n", 64)
    log = f"heldout-{cand}-{ho_start}.log"
    heldout = (f"WSP_ROOT={_q(root)} nohup setsid python3 weightsmith/heldout.py --cand {cand} "
               f"--slot-start {ho_start} --n {ho_n} > {log} 2>&1 < /dev/null &")
    host = _setting("WSP_HOST", repo)
    if host:
        remote = (_setting("WSP_REMOTE_REPO", repo) or "weightsmith").removeprefix("~/")
        heldout = f"ssh {host} {shlex.quote(f'cd {remote} && {heldout}')}"
    else:
        heldout = f"cd {shlex.quote(str(repo))} && {heldout}"
    print(textwrap.dedent(f"""
        next steps
          1. closed loop on practice slots, ORIGINAL instruction, paired with base on the same slots:
               WSP_ROOT={_q(root)} bin/owsp eval {cand} --n {n} --offset {off}
             (run `bin/owsp eval base --n {n} --offset {off}` first if base has no episodes on those slots)
          2. if it clearly helps: the operator's held-out score on unseen slots (slot start: a multiple of 16,
             >= 80000, never used before; runs base and candidate, {2 * ho_n} episodes; poll the log):
               {heldout}
          3. side effects: copy the candidate's edits into other task projects and run step 1 there.
        """))
    return result


STATUS = """\
Experimental. Proven: the build step. Rebuilt from their recorded capture, the paper's two published edits
(v2mi-f, abl-imgl8) come out bit for bit identical: same owsp fingerprints. Only dry-run so far: the select,
capture and check steps of `run`. Closed-loop and held-out evaluation are never started by this command; it
prints those commands for you. Docs: docs/DISTILL.md."""


def main(argv: Sequence[str] | None = None, *, prog: str = "python -m weightsmith.distill") -> int:
    """CLI: ``run CONFIG [--dry-run] [--steps ...]`` or ``render CONFIG --script build|compare|select``."""
    ap = argparse.ArgumentParser(prog=prog, description="Experimental. " + __doc__.split("\n\n")[0], epilog=STATUS,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run", help="select -> capture -> build -> check on the GPU host via bin/owsp")
    r.add_argument("config")
    r.add_argument("--dry-run", action="store_true", help="print commands, specs and scripts; touch nothing")
    r.add_argument("--steps", default="select,capture,build,check")
    r.add_argument("--workdir", default=None,
                   help="where specs/scripts/results are written (default distill-runs/NAME)")
    r.add_argument("--owsp", default=None, help="path to bin/owsp (default: this repo's)")
    s = sp.add_parser("render", help="print a rendered sandbox script")
    s.add_argument("config")
    s.add_argument("--script", choices=["build", "compare", "select"], default="build")
    a = ap.parse_args(argv)
    cfg = _load_config(a.config)
    if a.cmd == "render":
        if a.script == "select":
            sys.stdout.write(render_select_script(cfg["select"]))
        else:
            f = render_sandbox_script if a.script == "build" else render_compare_script
            sys.stdout.write(f(_build_config(cfg)))
        return 0
    steps = [x.strip() for x in a.steps.split(",") if x.strip()]
    bad = set(steps) - {"select", "capture", "build", "check"}
    if bad:
        ap.error(f"unknown steps {sorted(bad)}")
    run_pipeline(cfg, steps=steps, dry_run=a.dry_run, workdir=a.workdir, owsp_bin=a.owsp)
    return 0


if __name__ == "__main__":
    sys.exit(main())
