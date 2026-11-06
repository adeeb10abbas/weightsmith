"""Replay provenance and numerical checks, independent of the GPU runtime."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np


def parity_skip_reason(*, workspace, summary, recorded_prompt, prompt, seed, checkpoint):
    if workspace:
        return "candidate replay (base parity only)"
    if summary.get("fingerprint") != "base":
        return "recorded policy is not verified as base"
    if not summary.get("checkpoint") or Path(summary["checkpoint"]).resolve() != Path(checkpoint).resolve():
        return "checkpoint provenance missing or different"
    if recorded_prompt is None:
        return "recorded request prompt missing"
    if prompt != recorded_prompt:
        return "diagnostic prompt differs from recorded request"
    if seed < 0:
        return "deterministic sampling seed missing"
    return None


class ParityCheck:
    def __init__(self, tolerance=0.02):
        self.tolerance = float(tolerance)
        if not math.isfinite(self.tolerance) or self.tolerance < 0:
            raise ValueError("parity_tolerance must be finite and nonnegative")
        self.checked = 0
        self.skipped = []
        self.errors = []
        self.invalid = 0

    def add(self, actions, recorded, reason=None):
        if reason is not None:
            self.skipped.append(reason)
            return {"checked": False, "reason": reason, "max_abs_action_error": None}
        self.checked += 1
        a, b = np.asarray(actions, dtype=np.float64), np.asarray(recorded, dtype=np.float64)
        if a.shape != b.shape or not a.size or not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
            self.invalid += 1
            return {"checked": True, "passed": False, "reason": "invalid action shape or nonfinite values", "max_abs_action_error": None}
        error = float(np.max(np.abs(a - b)))
        self.errors.append(error)
        return {"checked": True, "passed": error <= self.tolerance, "max_abs_action_error": error}

    def report(self):
        maximum = max(self.errors) if self.errors else None
        passed = None if not self.checked else not self.invalid and maximum is not None and maximum <= self.tolerance
        return {"checked_count": self.checked, "skipped_count": len(self.skipped),
                "skip_reasons": sorted(set(self.skipped)), "invalid_count": self.invalid,
                "max_abs_action_error": maximum, "tolerance": self.tolerance, "passed": passed}

    def require_all(self, count):
        report = self.report()
        return count > 0 and report["checked_count"] == count and report["skipped_count"] == 0 and report["passed"] is True
