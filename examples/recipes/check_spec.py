#!/usr/bin/env python3
"""Check ``owsp acts`` capture specs before spending activation budget on them (runs locally, stdlib only).

    python3 examples/recipes/check_spec.py examples/recipes/specs/*.json [--max-items 96]

``owsp acts`` and the capture worker (weightsmith/acts_worker.py) silently ignore keys they do not know, so a typo
such as ``"vlm_layer"`` captures nothing and still charges the items. This checks every key and range the worker
reads, warns about unknown keys, and estimates how much the capture will write. Item existence (is that episode
recorded? does that replan exist?) needs the project: use recipe 03_capture_info.py (``check_items``).
Exit status 1 if any spec has an error. See docs/COOKBOOK.md, section 3.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")  # owsp's rule for capture ids and candidate names
KNOWN = {"id", "cand", "items", "prompt", "action_layers", "vlm_layers", "denoise_steps", "action_tokens",
         "prompt_tokens", "image_tokens", "action_attention"}
ITEM_KEYS = {"episode", "k", "step", "prompt"}
# float16 values written per item, by array (see weightsmith/acts_worker.py)
VLM_MLP, VLM_RES, ACT_MLP, ACT_RES, HEAD = 16384, 2048, 4096, 1024, 1024


def _ints(spec: dict, key: str, lo: int, hi: int, errors: list[str]) -> list[int] | None:
    v = spec.get(key)
    if v is None:
        return None
    if not isinstance(v, list) or not all(isinstance(x, int) and not isinstance(x, bool) for x in v):
        errors.append(f"{key} must be a list of integers")
        return None
    bad = [x for x in v if not lo <= x <= hi]
    if bad:
        errors.append(f"{key} has values outside {lo}..{hi}: {bad}")
    if len(set(v)) != len(v):
        errors.append(f"{key} has duplicates")
    return v


def check_spec(spec: dict, max_items: int = 96) -> tuple[list[str], list[str], dict]:
    """Return (errors, warnings, info) for one spec; info holds counts and the estimated output size."""
    errors: list[str] = []
    warnings: list[str] = []
    if not isinstance(spec, dict):
        return ["a spec is a JSON object"], [], {}
    unknown = sorted(set(spec) - KNOWN)
    if unknown:
        warnings.append(f"unknown keys {unknown} are ignored by owsp acts (typo?)")
    if "id" in spec and not (isinstance(spec["id"], str) and NAME_RE.match(spec["id"])):
        errors.append("id must match [a-z0-9][a-z0-9_-]{0,40}")
    if "id" not in spec:
        warnings.append("no id: owsp names the capture by the time (aMMDD-HHMMSS)")
    cand = spec.get("cand", "base")
    if not (isinstance(cand, str) and (cand == "base" or NAME_RE.match(cand))):
        errors.append("cand must be 'base' or a candidate name")
    items = spec.get("items")
    if not isinstance(items, list) or not items:
        errors.append("items must be a non-empty list")
        items = []
    if len(items) > max_items:
        errors.append(f"{len(items)} items; owsp allows at most {max_items} per call (acts_items_per_call)")
    prompts = 0
    for i, it in enumerate(items):
        if not isinstance(it, dict):
            errors.append(f"item {i} is not an object")
            continue
        if set(it) - ITEM_KEYS:
            warnings.append(f"item {i}: unknown keys {sorted(set(it) - ITEM_KEYS)}")
        if not isinstance(it.get("episode"), str) or not it.get("episode"):
            errors.append(f"item {i}: needs an episode id (e.g. base.base.s20001)")
        has_k, has_step = "k" in it, "step" in it
        if has_k == has_step:
            errors.append(f"item {i}: give exactly one of k (replan index) or step (sim step)")
        for key in ("k", "step"):
            if key in it and not (isinstance(it[key], int) and not isinstance(it[key], bool) and it[key] >= 0):
                errors.append(f"item {i}: {key} must be a non-negative integer")
        if "prompt" in it:
            prompts += 1
            if not isinstance(it["prompt"], str) or not it["prompt"].strip():
                errors.append(f"item {i}: prompt must be a non-empty string")
    if spec.get("prompt") is not None and not (isinstance(spec["prompt"], str) and spec["prompt"].strip()):
        errors.append("prompt must be a non-empty string")
    if prompts or spec.get("prompt"):
        warnings.append("instruction overrides are for diagnosis only, and projects with prompt_overrides "
                        "disabled refuse the whole spec")
    a_layers = _ints(spec, "action_layers", 0, 17, errors) or list(range(18))
    v_layers = _ints(spec, "vlm_layers", 0, 17, errors) or []
    steps = _ints(spec, "denoise_steps", 0, 9, errors) or list(range(10))
    p_toks = _ints(spec, "prompt_tokens", 0, 199, errors)
    if p_toks is not None and len(p_toks) > 64:
        errors.append("prompt_tokens: at most 64 slots")
    if p_toks is not None and not p_toks:
        errors.append("prompt_tokens must not be empty (omit it for the default slots 0..23)")
    p_toks = p_toks or list(range(24))
    tok = spec.get("action_tokens", "mean")
    if tok == "mean":
        n_tok = 1
    elif isinstance(tok, list) and tok and all(isinstance(x, int) and 0 <= x <= 14 for x in tok):
        n_tok = len(tok)
    else:
        errors.append("action_tokens must be \"mean\" or a list of token indices 0..14")
        n_tok = 1
    for key in ("image_tokens", "action_attention"):
        if key in spec and not isinstance(spec[key], bool):
            errors.append(f"{key} must be true or false")
    img = bool(spec.get("image_tokens"))
    if img and not v_layers:
        warnings.append("image_tokens needs vlm_layers; without them no image arrays are written")
        img = False
    S, La, Lv, P = len(steps), len(a_layers), len(v_layers), len(p_toks)
    per_item = 2 * (S * La * n_tok * (ACT_MLP + ACT_RES) + S * 15 * HEAD)
    per_item += 2 * Lv * P * (VLM_MLP + VLM_RES)
    if img:
        per_item += 2 * Lv * 2 * 17 * (VLM_MLP + VLM_RES)
    if spec.get("action_attention"):
        per_item += 2 * S * La * (8 * 5 + 2 * 16 * 16)
    per_item += 2 * 4 * 15 * 8  # actions and recorded_actions (float32)
    total = per_item * len(items)
    if total > 8e9:
        warnings.append(f"about {total / 1e9:.1f} GB on disk, and loading it needs as much RAM in `owsp py`")
    info = {"items": len(items), "prompt_overrides": prompts + bool(spec.get("prompt")), "cand": cand,
            "mb_per_item": per_item / 1e6, "mb_total": total / 1e6}
    return errors, warnings, info


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Check owsp acts capture specs (keys, ranges, budget, size).")
    ap.add_argument("specs", nargs="+", type=Path)
    ap.add_argument("--max-items", type=int, default=96, help="the project's acts_items_per_call (default 96)")
    a = ap.parse_args(argv)
    failed = 0
    for path in a.specs:
        try:
            spec = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as e:
            print(f"{path}: cannot read: {e}")
            failed += 1
            continue
        errors, warnings, info = check_spec(spec, a.max_items)
        status = "ERROR" if errors else "ok"
        extra = f", {info['prompt_overrides']} with a prompt override" if info.get("prompt_overrides") else ""
        size = f", about {info['mb_total']:,.0f} MB ({info['mb_per_item']:.2f} MB per item)" if info else ""
        print(f"{path}: {status}: {info.get('items', 0)} items on {info.get('cand')!r}{extra}{size}")
        for e in errors:
            print(f"  error: {e}")
        for w in warnings:
            print(f"  warning: {w}")
        failed += bool(errors)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
