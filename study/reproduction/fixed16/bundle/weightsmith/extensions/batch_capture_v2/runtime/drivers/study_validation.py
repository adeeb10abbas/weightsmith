"""Pure allowlist validation for the restricted RQ1-RQ3 editor interface.

No filesystem or model access. The broker must apply these checks before budget
reservation/dispatch; the remote typed tools can reuse episode-ID validation.
"""
from __future__ import annotations

import math
import re

DEVELOPMENT_START = 30016
DEVELOPMENT_STOP = 30048
MAX_CAPTURE_BYTES = 512 * 1024 * 1024
LABELS = frozenset({"base", "p1", "p2", "p3"})
EPISODE = re.compile(r"^(base|p[123])(?:\.([a-z0-9]+))?\.s(\d+)$")
CAPTURE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")
ALLOWED_SPEC_FIELDS = frozenset({"id", "cand", "items", "action_layers", "vlm_layers",
                                 "denoise_steps", "action_tokens", "prompt_tokens",
                                 "parity_tolerance", "require_parity"})


def validate_episode_id(eid):
    """Return an approved development episode ID or fail closed.

    Recognizes the current owsp form ``p1.<hash>.s30016`` and a plain
    ``p1.s30016`` form. Repeated evaluation suffixes (.r2), prompt overrides,
    foreign candidates, path components, and non-development slots are rejected.
    Actual record fingerprint/task/prompt validation is a separate binding check.
    """
    if not isinstance(eid, str):
        raise ValueError("episode ID must be a string")
    match = EPISODE.fullmatch(eid)
    if match is None:
        raise ValueError("episode must name base/p1/p2/p3 on one approved development slot")
    candidate, fingerprint, slot = match.groups()
    if not DEVELOPMENT_START <= int(slot) < DEVELOPMENT_STOP:
        raise ValueError("episode is outside the approved development slots 30016..30047")
    if slot != str(int(slot)):
        raise ValueError("episode slot must use canonical decimal notation")
    if fingerprint is not None:
        if candidate == "base" and fingerprint != "base":
            raise ValueError("base episode fingerprint must be 'base'")
        if candidate != "base" and not re.fullmatch(r"[0-9a-f]{6,64}", fingerprint):
            raise ValueError("candidate episode fingerprint must be 6..64 hexadecimal characters")
    return eid


def _integers(value, name, low, high, max_count, *, allow_empty=False):
    if not isinstance(value, list) or len(value) > max_count or (not value and not allow_empty):
        raise ValueError(f"{name} must be a {'possibly empty ' if allow_empty else 'nonempty '}list of at most {max_count} integers")
    if any(type(item) is not int or not low <= item <= high for item in value):
        raise ValueError(f"{name} values must be integers in {low}..{high}")
    if len(set(value)) != len(value):
        raise ValueError(f"{name} cannot contain duplicates")
    return list(value)


def predicted_capture_bytes(spec):
    """Upper estimate for persisted arrays, including one MiB for metadata.

    This is an output/storage guard, not a bound on JAX compilation/device peak
    memory. GPU memory/resource checks remain the operator's responsibility.
    """
    n = len(spec["items"])
    steps = len(spec["denoise_steps"])
    action_layers = len(spec["action_layers"])
    tokens = 1 if spec["action_tokens"] == "mean" else len(spec["action_tokens"])
    action = n * steps * action_layers * tokens * (4096 + 1024) * 2
    head = n * steps * 15 * 1024 * 2
    vlm = n * len(spec["vlm_layers"]) * len(spec["prompt_tokens"]) * (16384 + 2048) * 2
    action_outputs = n * 15 * 8 * 4 * 2
    return action + head + vlm + action_outputs + 1024 * 1024


def normalize_activation_spec(spec, allowed_candidates, mode):
    """Validate and normalize a capture without accepting code or arbitrary paths.

    Base captures use original base episodes and the fixed 0.02 parity check.
    Edited captures must use their own recorded-policy packets and pass the same
    three-way batch-matched parity checks as base captures. Non-adaptive runs can capture only
    the original base. All lists are bounded and duplicate-free.
    """
    if mode not in ("adaptive", "nonadaptive"):
        raise ValueError("unknown study mode")
    if not isinstance(spec, dict):
        raise ValueError("activation specification must be an object")
    if not {'action_layers', 'denoise_steps'} <= set(spec):
        raise ValueError('v2 requires explicit action_layers and denoise_steps; full default traces exceed the storage cap')
    unknown = set(spec) - ALLOWED_SPEC_FIELDS
    if unknown:
        raise ValueError(f"unsupported activation fields: {sorted(unknown)}")
    name = spec.get("cand", "base")
    allowed = set(allowed_candidates) | {"base"}
    if not isinstance(name, str) or name not in LABELS or name not in allowed:
        raise ValueError("capture policy must be base or a candidate created in this run")
    if mode == "nonadaptive" and name != "base":
        raise ValueError("non-adaptive captures may use only the original base")
    items = spec.get("items")
    if not isinstance(items, list) or not 1 <= len(items) <= 96:
        raise ValueError("activation capture requires 1..96 items")
    normalized_items = []
    for item in items:
        if not isinstance(item, dict) or set(item) not in ({"episode", "k"}, {"episode", "step"}):
            raise ValueError("each item must contain exactly episode plus k or step")
        eid = validate_episode_id(item["episode"])
        episode_candidate = eid.split(".", 1)[0]
        if episode_candidate not in allowed:
            raise ValueError("capture episode belongs to a candidate not created in this run")
        if episode_candidate != name:
            raise ValueError("v2 parity captures require episodes from the exact requested deployed policy")
        key = "k" if "k" in item else "step"
        value = item[key]
        upper = 59 if key == "k" else 900
        if type(value) is not int or not 0 <= value <= upper:
            raise ValueError(f"{key} must be an integer in 0..{upper}")
        normalized_items.append({"episode": eid, key: value})
    result = {"cand": name, "items": normalized_items,
              "action_layers": _integers(spec.get("action_layers", list(range(18))), "action_layers", 0, 17, 18),
              "vlm_layers": _integers(spec.get("vlm_layers", []), "vlm_layers", 0, 17, 18, allow_empty=True),
              "denoise_steps": _integers(spec.get("denoise_steps", list(range(10))), "denoise_steps", 0, 9, 10),
              "prompt_tokens": _integers(spec.get("prompt_tokens", list(range(24))), "prompt_tokens", 0, 199, 64)}
    tokens = spec.get("action_tokens", "mean")
    result["action_tokens"] = "mean" if tokens == "mean" else _integers(tokens, "action_tokens", 0, 14, 15)
    if "id" in spec:
        if not isinstance(spec["id"], str) or not CAPTURE.fullmatch(spec["id"]):
            raise ValueError("capture id must be a short lowercase identifier")
        result["id"] = spec["id"]
    tolerance = spec.get("parity_tolerance", 0.02)
    if isinstance(tolerance, bool) or not isinstance(tolerance, (int, float)) or not math.isfinite(tolerance) or tolerance != 0.02:
        raise ValueError("all-policy replay tolerance is fixed at 0.02")
    required = True
    if "require_parity" in spec and (type(spec["require_parity"]) is not bool or spec["require_parity"] != required):
        raise ValueError(f"require_parity must be {required} for this capture policy")
    result["parity_tolerance"] = 0.02
    result["require_parity"] = required
    result["strict_request_indices"] = True
    predicted = predicted_capture_bytes(result)
    if predicted > MAX_CAPTURE_BYTES:
        raise ValueError(f"predicted capture output {predicted} bytes exceeds 512 MiB; select fewer layers, tokens, steps or items")
    return result
