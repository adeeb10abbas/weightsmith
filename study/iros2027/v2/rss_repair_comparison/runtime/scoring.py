"""CPU-testable predicates for the additive RSS repair comparison.

Every goal is observed on every trajectory. ``success`` is the selected goal;
cross-goal scores diagnose indiscriminate repair and do not add independent trials.
"""
from __future__ import annotations

import math
import numpy as np

VERSION = "rss-repair-v1"
GOALS = {
    "temporal": ("cube_then_banana", "banana_then_cube", "cube_only"),
    "spatial": ("mustard_right", "mustard_left", "mustard_on_raisin"),
}


class Scorer:
    def __init__(self, case, initial, dt, *, dwell_s=1.0, speed_mps=.02,
                 mover_min_m=.02, reference_max_m=.02):
        if case not in GOALS or dt <= 0:
            raise ValueError((case, dt))
        self.case, self.dt = case, float(dt)
        self.initial = {k: np.asarray(v, dtype=float).copy() for k, v in initial.items()}
        self.n = len(next(iter(self.initial.values())))
        self.previous = {k: v.copy() for k, v in self.initial.items()}
        self.maximum = {k: np.zeros(self.n) for k in initial}
        self.speed = {k: np.zeros(self.n) for k in initial}
        self.first = {k: np.full(self.n, -1, dtype=int) for k in ("rubiks_cube", "banana")}
        self.count = {k: np.zeros(self.n, dtype=int) for k in GOALS[case]}
        self.stock_count = {k: np.zeros(self.n, dtype=int) for k in GOALS[case]}
        self.current = {k: np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.ever = {k: np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.stock = {k: np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.stock_ever = {k: np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.need = math.ceil(dwell_s / self.dt - 1e-6) + 1
        self.speed_mps, self.mover_min_m, self.reference_max_m = speed_mps, mover_min_m, reference_max_m
        self.last_tick = None
        self.parts = {}

    def update(self, tick, positions, predicates, active=None):
        if self.last_tick == tick:
            return self.current
        if self.last_tick is not None and tick < self.last_tick:
            raise RuntimeError("Episode tick moved backward without a new scorer/reset")
        active = np.ones(self.n, dtype=bool) if active is None else np.asarray(active, dtype=bool)
        for name, raw in positions.items():
            p = np.asarray(raw, dtype=float)
            if p.shape != (self.n, 3) or not np.isfinite(p).all():
                raise RuntimeError(f"Invalid physical pose for {name}")
            self.speed[name][active] = np.linalg.norm(p - self.previous[name], axis=1)[active] / self.dt
            self.maximum[name][active] = np.maximum(self.maximum[name], np.linalg.norm(p - self.initial[name], axis=1))[active]
            self.previous[name][active] = p[active]
        pred = {k: np.broadcast_to(np.asarray(v, dtype=bool), (self.n,)) for k, v in predicates.items()}
        previous_parts = self.parts
        if self.case == "temporal":
            for name in ("rubiks_cube", "banana"):
                new = active & (self.first[name] < 0) & pred[name + "_placed"]
                self.first[name][new] = tick
            cube, banana = self.first["rubiks_cube"], self.first["banana"]
            both = pred["rubiks_cube_placed"] & pred["banana_placed"]
            fixed = self.maximum["bowl"] <= self.reference_max_m
            stable_cube = self.speed["rubiks_cube"] < self.speed_mps
            stable_banana = self.speed["banana"] < self.speed_mps
            unchanged_banana = self.maximum["banana"] <= self.reference_max_m
            raw = {
                "cube_then_banana": both & (cube >= 0) & (cube < banana) & fixed & stable_cube & stable_banana,
                "banana_then_cube": both & (banana >= 0) & (banana < cube) & fixed & stable_cube & stable_banana,
                "cube_only": pred["rubiks_cube_placed"] & ~pred["banana_placed"] & unchanged_banana & fixed & stable_cube,
            }
            secondary = {
                "cube_then_banana": both & (cube >= 0) & (cube <= banana),
                "banana_then_cube": both & (banana >= 0) & (banana <= cube),
                # No stock cube-only task; this is explicitly a component diagnostic.
                "cube_only": pred["rubiks_cube_placed"] & ~pred["banana_placed"],
            }
            self.parts = {**pred, "reference_fixed": fixed, "banana_unchanged": unchanged_banana,
                          "cube_stationary": stable_cube, "banana_stationary": stable_banana}
        else:
            fixed = self.maximum["raisin_box"] <= self.reference_max_m
            moved = self.maximum["mustard_bottle"] >= self.mover_min_m
            stationary = self.speed["mustard_bottle"] < self.speed_mps
            base = pred["mustard_detached"] & stationary
            secondary = {
                "mustard_right": pred["right_of"] & pred["mustard_table_contact"] & base,
                "mustard_left": pred["left_of"] & pred["mustard_table_contact"] & base,
                "mustard_on_raisin": pred["on_raisin"] & pred["mustard_raisin_contact"] & base,
            }
            raw = {k: v & fixed & moved for k, v in secondary.items()}
            self.parts = {**pred, "reference_fixed": fixed, "mover_displaced": moved, "mover_stationary": stationary}
        if previous_parts:
            self.parts = {k: np.where(active, v, previous_parts[k]) for k, v in self.parts.items()}
        for key, valid in raw.items():
            self.count[key][active] = np.where(valid, self.count[key] + 1, 0)[active]
            self.current[key][active] = (self.count[key] >= self.need)[active]
            self.ever[key] |= self.current[key]
            if self.case == "spatial":
                self.stock_count[key][active] = np.where(secondary[key], self.stock_count[key] + 1, 0)[active]
                self.stock[key][active] = (self.stock_count[key] >= self.need)[active]
            else:
                self.stock[key][active] = secondary[key][active]
            self.stock_ever[key] |= self.stock[key]
        self.last_tick = tick
        return self.current

    def record(self):
        conv = lambda d: {k: v.tolist() for k, v in d.items()}
        return {"predicate_version": VERSION, "case": self.case, "last_tick": self.last_tick,
                "control_dt": self.dt, "required_consecutive_samples": self.need,
                "initial_positions": conv(self.initial), "final_positions": conv(self.previous),
                "max_displacement_m": conv(self.maximum), "final_speed_mps": conv(self.speed),
                "first_placement_tick": conv(self.first) if self.case == "temporal" else {},
                "strict_final": conv(self.current), "strict_ever": conv(self.ever),
                "stock_final": conv(self.stock), "stock_ever": conv(self.stock_ever),
                "final_components": conv(self.parts)}
