"""Historically informed goals, reusing existing scored tasks without modifying them.

Importing this module never imports a simulator or policy. A fresh qualification
gate is required for every goal; a historical success is not qualification.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import importlib.util
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[4]
V2 = HERE.parent.parent
ROBOLAB = Path('/home/ali/vibes/weight-space-policy-lab/robolab_checkout')
GOAL_IDS = ('temporal', 'cube_behind', 'mustard_right')

_GOALS = {
    'temporal': {
        'runtime_case': 'temporal', 'runtime_family': 'rss_repair_comparison',
        'task': 'RSSCubeThenBananaTask', 'goal': 'cube_then_banana',
        'competing_task': 'RSSBananaThenCubeTask', 'competing_goal': 'banana_then_cube',
        'original': 'Put the cube then the banana in the bowl',
        'teacher': 'Put the cube in the bowl then put the banana in the bowl',
        'unseen': 'Place the cube in the bowl, followed by the banana.',
        'competing': 'Put the banana in the bowl then put the cube in the bowl',
        'scene': 'rubiks_cube_banana_bowl.usda',
        'objects': ['rubiks_cube', 'banana', 'bowl'], 'horizon_seconds': 60,
        'source_selector': {'rule': 'first_early_release', 'max_hold': 2,
                            'min_grip': 0.1, 'max_grip': 0.9},
        'historical_context': 'Same known instruction pair as the completed correction-budget study; all source and evaluation episodes in this new study are fresh.',
        'predicate_summary': 'Strict first-placement order; both objects supported in the bowl and detached; speed below 2cm/s; bowl displacement at most 2cm; one-second dwell.',
    },
    'cube_behind': {
        'runtime_case': 'cube_behind', 'runtime_family': 'rss_goal_expansion_v2',
        'task': 'RSSGCubeBehindTask', 'goal': 'cube_behind',
        'competing_task': 'RSSGCubeInBowlTask', 'competing_goal': 'cube_in_bowl',
        'original': "Place the Rubik's cube so that the bowl is in front of the Rubik's cube.",
        'teacher': "Place the Rubik's cube so that the Rubik's cube is behind the bowl.",
        'unseen': "Position the Rubik's cube on the far side of the bowl, so the bowl is closer to you than the cube.",
        'competing': 'Put the rubiks cube in the bowl',
        'scene': 'rubiks_cube_banana_bowl.usda',
        'objects': ['rubiks_cube', 'banana', 'bowl'], 'horizon_seconds': 30,
        'source_selector': {'rule': 'midpoint', 'fraction': 0.5},
        'historical_context': 'Known successful clarification from the prior cube-behind study; this is an independently fitted goal, not blind task discovery. The runtime and strict scoring are reused, not its policy outcomes.',
        'predicate_summary': 'Robot-frame, nonmirrored 45-degree behind cone; cube table contact and detachment; speed below 2cm/s; cube displacement at least 2cm; bowl maximum displacement at most 2cm; one-second dwell. Competing containment requires cube-bowl contact under the same motion/locality constraints.',
    },
    'mustard_right': {
        'runtime_case': 'spatial', 'runtime_family': 'rss_repair_comparison',
        'task': 'RSSMustardRightTask', 'goal': 'mustard_right',
        'competing_task': 'RSSMustardLeftTask', 'competing_goal': 'mustard_left',
        'original': 'Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle.',
        'teacher': 'Move the mustard bottle to the right of the raisin box on the table. Leave the raisin box where it is.',
        'unseen': 'Place the mustard bottle on the table to the right of the raisin box, keeping the raisin box in place.',
        'competing': 'Move the mustard bottle to the left of the raisin box. Leave the raisin box where it is.',
        'scene': 'mustard_raisin_box.usda',
        'objects': ['mustard_bottle', 'raisin_box'], 'horizon_seconds': 40,
        'source_selector': {'rule': 'midpoint', 'fraction': 0.5},
        'historical_context': 'Known inverse/direct relational instruction pattern, evaluated previously with stricter scoring. A new source gate determines whether this particular clarification supplies usable supervision. The two spatial goals are different targets, not proof of different failure mechanisms.',
        'predicate_summary': 'Robot-frame, nonmirrored 45-degree right/left cone; mustard table contact and detachment; speed below 2cm/s; mustard displacement at least 2cm; raisin-box maximum displacement at most 2cm; one-second dwell.',
    },
}


def get_goal(goal_id: str) -> dict:
    if goal_id not in _GOALS:
        raise ValueError(f'Unknown goal: {goal_id!r}')
    result = deepcopy(_GOALS[goal_id])
    result.update(goal_id=goal_id,
                  runtime_dir=str(V2 / result['runtime_family'] / 'runtime'),
                  scene_path=str(ROBOLAB / 'assets/scenes' / result['scene']),
                  scoring_policy='Use the same strict scorer for collection, development and held-out evaluation. Score all native goals on every trajectory.')
    return result


def load_runtime_module(goal_id: str, filename: str):
    """Load a CPU-only legacy helper under a nonconflicting module name."""
    if filename not in ('scoring.py', 'generate_layouts.py', 'layouts.py'):
        raise ValueError('Only CPU-only helper modules may be loaded here')
    path = Path(get_goal(goal_id)['runtime_dir']) / filename
    key = f'fewpair_goal_{goal_id}_{path.stem}'
    if key not in sys.modules:
        spec = importlib.util.spec_from_file_location(key, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[key] = module
        spec.loader.exec_module(module)
    return sys.modules[key]


def runtime_sources(goal_id: str) -> list[Path]:
    goal = get_goal(goal_id)
    return sorted({*Path(goal['runtime_dir']).glob('*.py'),
                   Path(goal['scene_path']), *HERE.glob('*.py')})


def scoring_identity(goal_id: str) -> dict:
    goal = get_goal(goal_id)
    path = Path(goal['runtime_dir']) / 'scoring.py'
    module = load_runtime_module(goal_id, 'scoring.py')
    return {'path': str(path), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'version': module.VERSION, 'case': goal['runtime_case'],
            'goals': list(module.GOALS[goal['runtime_case']]),
            'summary': goal['predicate_summary']}


def new_scorer(goal_id: str, initial: dict, dt: float):
    """Construct exactly the deployed runtime's strict scorer for CPU tests."""
    return load_runtime_module(goal_id, 'scoring.py').Scorer(
        get_goal(goal_id)['runtime_case'], initial, dt)
