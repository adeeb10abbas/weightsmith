import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1]/'runtime/scoring.py'
spec = importlib.util.spec_from_file_location('goal_expansion_scorer',PATH)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
Scorer = mod.Scorer


def setup(case='cube_behind',n=1):
    names = {'cube_behind':['rubiks_cube','bowl','banana'],
             'glasses_pickup':['glasses','rubiks_cube'],
             'mustard_left_bin':['mustard','grey_bin_left','grey_bin_right']}[case]
    positions = {k:np.zeros((n,3)) for k in names}
    scorer = Scorer(case,positions,.1)
    pred = {'cube_behind':{'behind':True,'cube_table_contact':True,'cube_bowl_contact':False,'cube_detached':True,'cube_in_bowl':False},
            'glasses_pickup':{'glasses_picked_up':True,'cube_picked_up':False},
            'mustard_left_bin':{'mustard_left':True,'mustard_right':False,'mustard_left_contact':True,'mustard_right_contact':False,'mustard_detached':True}}[case]
    if case!='glasses_pickup':
        positions['rubiks_cube' if case=='cube_behind' else 'mustard'][:,0]=.1
    return scorer,positions,pred


def advance(s,p,q,begin=0,end=14,active=None):
    for t in range(begin,end): s.update(t,p,q,active)


def test_dwell_spans_a_full_second_and_requires_stability():
    s,p,q=setup();advance(s,p,q,end=11)
    assert not s.current['cube_behind'][0]
    s.update(11,p,q)
    assert s.current['cube_behind'][0]


@pytest.mark.parametrize('key',['cube_table_contact','cube_detached','behind'])
def test_behind_contact_release_relation_required(key):
    s,p,q=setup();q[key]=False;advance(s,p,q)
    assert not s.ever['cube_behind'][0]


def test_reference_excursion_cannot_be_hidden_by_return():
    s,p,q=setup();p['bowl'][0,1]=.021;s.update(0,p,q)
    p['bowl'][0,1]=0;advance(s,p,q,begin=1,end=30)
    assert not s.ever['cube_behind'][0]


def test_cube_in_bowl_competitor_needs_container_contact():
    s,p,q=setup();q.update(behind=False,cube_in_bowl=True)
    advance(s,p,q)
    assert s.stock['cube_in_bowl'][0] and not s.current['cube_in_bowl'][0]
    q['cube_bowl_contact']=True;advance(s,p,q,begin=14,end=27)
    assert s.current['cube_in_bowl'][0] and not s.current['cube_behind'][0]


@pytest.mark.parametrize('reference',['grey_bin_left','grey_bin_right'])
def test_both_bins_must_stay_fixed(reference):
    s,p,q=setup('mustard_left_bin');p[reference][0,1]=.021;advance(s,p,q)
    assert not s.ever['mustard_left_bin'][0]


def test_named_bin_goals_are_separate():
    s,p,q=setup('mustard_left_bin');advance(s,p,q)
    assert s.current['mustard_left_bin'][0] and not s.current['mustard_right_bin'][0]


def test_pickup_dwell_resets_on_failed_stock_predicate():
    s,p,q=setup('glasses_pickup');advance(s,p,q,end=10)
    assert not s.current['glasses_pickup'][0]
    q['glasses_picked_up']=False;s.update(10,p,q)
    q['glasses_picked_up']=True;advance(s,p,q,begin=11,end=21)
    assert not s.current['glasses_pickup'][0]
    s.update(21,p,q);assert s.current['glasses_pickup'][0]


def test_dedup_ticks_and_frozen_env_keep_final_score():
    s,p,q=setup('glasses_pickup',2);advance(s,p,q,end=11)
    assert s.current['glasses_pickup'].all()
    q['glasses_picked_up']=False;s.update(10,p,q)
    assert s.current['glasses_pickup'].all()
    s.update(11,p,q,active=[False,True])
    assert s.current['glasses_pickup'].tolist()==[True,False]
    assert s.ever['glasses_pickup'].all()


def test_nan_pose_and_backward_time_rejected():
    s,p,q=setup();s.update(1,p,q)
    with pytest.raises(RuntimeError): s.update(0,p,q)
    p['rubiks_cube'][0,0]=np.nan
    with pytest.raises(RuntimeError): s.update(2,p,q)
