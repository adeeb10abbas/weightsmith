"""Stable placement with intended movement and reference-object preservation."""
from __future__ import annotations
import math
import numpy as np

VERSION = 'weightsmith-new-goals-20261118-v1'
CASES = {
    'banana_left_plate': {'mover':'banana','reference':'plate_large','goals':{
        'banana_left_plate':('banana',('plate_large',),'target','mover_table'),
        'banana_on_plate':('banana',('plate_large',),'preservation','mover_reference')}},
    'butter_behind_raisin': {'mover':'butter','reference':'raisin_box','goals':{
        'butter_behind_raisin':('butter',('raisin_box',),'target','mover_table'),
        'butter_on_raisin':('butter',('raisin_box',),'preservation','mover_reference')}},
    'left_bowl_on_right': {'mover':'bowl_2','reference':'bowl_1','goals':{
        'left_bowl_on_right':('bowl_2',('bowl_1',),'target','mover_reference'),
        'right_bowl_on_left':('bowl_1',('bowl_2',),'preservation','mover_reference')}},
}
GOALS={case:tuple(cfg['goals']) for case,cfg in CASES.items()}


class Scorer:
    def __init__(self,case,initial,dt,*,dwell_s=1.,speed_mps=.02,mover_min_m=.02,reference_max_m=.02):
        if case not in CASES or dt<=0:raise ValueError((case,dt))
        self.case,self.dt=case,float(dt)
        self.initial={k:np.asarray(v,dtype=float).copy() for k,v in initial.items()}
        self.n=len(next(iter(self.initial.values())))
        self.previous={k:v.copy() for k,v in self.initial.items()}
        self.maximum={k:np.zeros(self.n) for k in initial}
        self.speed={k:np.zeros(self.n) for k in initial}
        self.count={k:np.zeros(self.n,dtype=int) for k in GOALS[case]}
        self.current={k:np.zeros(self.n,dtype=bool) for k in GOALS[case]}
        self.ever={k:np.zeros(self.n,dtype=bool) for k in GOALS[case]}
        self.stock={k:np.zeros(self.n,dtype=bool) for k in GOALS[case]}
        self.stock_ever={k:np.zeros(self.n,dtype=bool) for k in GOALS[case]}
        self.need=math.ceil(dwell_s/self.dt-1e-6)+1
        self.speed_mps,self.mover_min_m,self.reference_max_m=speed_mps,mover_min_m,reference_max_m
        self.last_tick,self.parts=None,{}

    def update(self,tick,positions,predicates,active=None):
        if self.last_tick==tick:return self.current
        if self.last_tick is not None and tick<self.last_tick:raise RuntimeError('Backward episode tick')
        active=np.ones(self.n,dtype=bool) if active is None else np.asarray(active,dtype=bool)
        for name,raw in positions.items():
            p=np.asarray(raw,dtype=float)
            if p.shape!=(self.n,3) or not np.isfinite(p).all():raise RuntimeError('Invalid pose '+name)
            self.speed[name][active]=(np.linalg.norm(p-self.previous[name],axis=1)/self.dt)[active]
            self.maximum[name][active]=np.maximum(self.maximum[name],np.linalg.norm(p-self.initial[name],axis=1))[active]
            self.previous[name][active]=p[active]
        pred={k:np.broadcast_to(np.asarray(v,dtype=bool),(self.n,)) for k,v in predicates.items()}
        prior=self.parts
        self.parts={k:np.where(active,v,prior[k]) for k,v in pred.items()} if prior else pred
        for goal,(mover,refs,relation,contact) in CASES[self.case]['goals'].items():
            fixed=np.logical_and.reduce([self.maximum[r]<=self.reference_max_m for r in refs])
            moved=self.maximum[mover]>=self.mover_min_m
            stable=self.speed[mover]<self.speed_mps
            secondary=pred[relation]&pred['detached_'+mover]
            raw=secondary&pred[contact]&fixed&moved&stable
            self.count[goal][active]=np.where(raw,self.count[goal]+1,0)[active]
            self.current[goal][active]=(self.count[goal]>=self.need)[active]
            self.ever[goal]|=self.current[goal]
            self.stock[goal][active]=secondary[active]
            self.stock_ever[goal]|=self.stock[goal]
        self.last_tick=tick
        return self.current

    def record(self):
        conv=lambda d:{k:v.tolist() for k,v in d.items()}
        return {'predicate_version':VERSION,'case':self.case,'last_tick':self.last_tick,
                'control_dt':self.dt,'required_consecutive_samples':self.need,
                'initial_positions':conv(self.initial),'final_positions':conv(self.previous),
                'max_displacement_m':conv(self.maximum),'final_speed_mps':conv(self.speed),
                'strict_final':conv(self.current),'strict_ever':conv(self.ever),
                'stock_final':conv(self.stock),'stock_ever':conv(self.stock_ever),
                'final_components':conv(self.parts)}
