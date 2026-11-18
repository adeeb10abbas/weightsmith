"""Pure-numpy goal expansion scoring; physical inputs are logged independently.

Placement requires one second of detached stable contact, at least 2cm mover
motion, and no reference displacement beyond 2cm anywhere in the trajectory.
Pickup retains the stock grabbed-and-5cm-clear predicate plus one-second dwell.
"""
from __future__ import annotations
import math
import numpy as np

VERSION = 'rss-goal-expansion-v1'
GOALS = {
    'cube_behind': ('cube_behind', 'cube_in_bowl'),
    'glasses_pickup': ('glasses_pickup', 'cube_pickup'),
    'mustard_left_bin': ('mustard_left_bin', 'mustard_right_bin'),
}


class Scorer:
    def __init__(self, case, initial, dt, *, dwell_s=1., speed_mps=.02,
                 mover_min_m=.02, reference_max_m=.02):
        if case not in GOALS or dt <= 0:
            raise ValueError((case, dt))
        self.case, self.dt = case, float(dt)
        self.initial = {k: np.asarray(v, dtype=float).copy() for k,v in initial.items()}
        self.n = len(next(iter(self.initial.values())))
        self.previous = {k:v.copy() for k,v in self.initial.items()}
        self.maximum = {k:np.zeros(self.n) for k in initial}
        self.speed = {k:np.zeros(self.n) for k in initial}
        self.count = {k:np.zeros(self.n, dtype=int) for k in GOALS[case]}
        self.current = {k:np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.ever = {k:np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.stock = {k:np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.stock_ever = {k:np.zeros(self.n, dtype=bool) for k in GOALS[case]}
        self.need = math.ceil(dwell_s / self.dt - 1e-6) + 1
        self.speed_mps, self.mover_min_m = speed_mps, mover_min_m
        self.reference_max_m = reference_max_m
        self.last_tick, self.parts = None, {}

    def update(self, tick, positions, predicates, active=None):
        if self.last_tick == tick:
            return self.current
        if self.last_tick is not None and tick < self.last_tick:
            raise RuntimeError('Episode tick moved backward without reset')
        active = np.ones(self.n,dtype=bool) if active is None else np.asarray(active,dtype=bool)
        for name,raw in positions.items():
            p = np.asarray(raw,dtype=float)
            if p.shape != (self.n,3) or not np.isfinite(p).all():
                raise RuntimeError(f'Invalid physical pose for {name}')
            self.speed[name][active] = np.linalg.norm(p-self.previous[name],axis=1)[active]/self.dt
            self.maximum[name][active] = np.maximum(self.maximum[name],np.linalg.norm(p-self.initial[name],axis=1))[active]
            self.previous[name][active] = p[active]
        pred = {k:np.broadcast_to(np.asarray(v,dtype=bool),(self.n,)) for k,v in predicates.items()}
        prior = self.parts
        if self.case == 'cube_behind':
            fixed = self.maximum['bowl'] <= self.reference_max_m
            moved = self.maximum['rubiks_cube'] >= self.mover_min_m
            stable = self.speed['rubiks_cube'] < self.speed_mps
            secondary = {
                'cube_behind': pred['behind'] & pred['cube_table_contact'] & pred['cube_detached'],
                'cube_in_bowl': pred['cube_in_bowl'] & pred['cube_detached'],
            }
            raw = {
                'cube_behind': secondary['cube_behind'] & fixed & moved & stable,
                'cube_in_bowl': secondary['cube_in_bowl'] & pred['cube_bowl_contact'] & fixed & moved & stable,
            }
            self.parts = {**pred,'reference_fixed':fixed,'mover_displaced':moved,'mover_stationary':stable}
        elif self.case == 'mustard_left_bin':
            fixed = (self.maximum['grey_bin_left'] <= self.reference_max_m) & (self.maximum['grey_bin_right'] <= self.reference_max_m)
            moved = self.maximum['mustard'] >= self.mover_min_m
            stable = self.speed['mustard'] < self.speed_mps
            secondary = {
                'mustard_left_bin': pred['mustard_left'] & pred['mustard_detached'],
                'mustard_right_bin': pred['mustard_right'] & pred['mustard_detached'],
            }
            raw = {goal:secondary[goal] & pred[f'mustard_{side}_contact'] & fixed & moved & stable
                   for goal,side in [('mustard_left_bin','left'),('mustard_right_bin','right')]}
            self.parts = {**pred,'references_fixed':fixed,'mover_displaced':moved,'mover_stationary':stable}
        else:
            secondary = {'glasses_pickup':pred['glasses_picked_up'],'cube_pickup':pred['cube_picked_up']}
            raw = dict(secondary)
            self.parts = pred
        if prior:
            self.parts = {k:np.where(active,v,prior[k]) for k,v in self.parts.items()}
        for key,valid in raw.items():
            self.count[key][active] = np.where(valid,self.count[key]+1,0)[active]
            self.current[key][active] = (self.count[key]>=self.need)[active]
            self.ever[key] |= self.current[key]
            self.stock[key][active] = secondary[key][active]
            self.stock_ever[key] |= self.stock[key]
        self.last_tick = tick
        return self.current

    def record(self):
        conv = lambda d:{k:v.tolist() for k,v in d.items()}
        return {'predicate_version':VERSION,'case':self.case,'last_tick':self.last_tick,
                'control_dt':self.dt,'required_consecutive_samples':self.need,
                'initial_positions':conv(self.initial),'final_positions':conv(self.previous),
                'max_displacement_m':conv(self.maximum),'final_speed_mps':conv(self.speed),
                'strict_final':conv(self.current),'strict_ever':conv(self.ever),
                'stock_final':conv(self.stock),'stock_ever':conv(self.stock_ever),
                'final_components':conv(self.parts)}
