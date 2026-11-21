"""CPU tests for goal routing, frozen splits, and critical success predicates."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from goals import GOAL_IDS, get_goal, build_layouts, select_failure_decision
from goals.layouts import STAGES, write_layouts
from goals.registry import new_scorer, scoring_identity


class RegistryTests(unittest.TestCase):
    def test_registered_tasks_match_scoring_goals(self):
        for goal_id in GOAL_IDS:
            goal = get_goal(goal_id)
            for key in ('task', 'competing_task'):
                self.assertTrue((Path(goal['runtime_dir']) / (goal[key] + '.py')).is_file())
            identity = scoring_identity(goal_id)
            self.assertIn(goal['goal'], identity['goals'])
            self.assertIn(goal['competing_goal'], identity['goals'])
            self.assertNotEqual(goal['goal'], goal['competing_goal'])
            self.assertNotEqual(goal['original'], goal['teacher'])

    def test_registry_is_not_mutable_through_callers(self):
        goal = get_goal('temporal')
        goal['objects'].clear()
        self.assertEqual(len(get_goal('temporal')['objects']), 3)


class LayoutTests(unittest.TestCase):
    def test_reproducible_split_disjoint_layouts(self):
        for goal_id in GOAL_IDS:
            ids, positions, seeds = set(), set(), set()
            for split in STAGES:
                spec = build_layouts(goal_id, split)
                self.assertEqual(spec, build_layouts(goal_id, split))
                self.assertEqual(len(spec['layouts']), 48 if split == 'collection' else 16)
                self.assertNotIn(spec['generator_seed'], seeds)
                seeds.add(spec['generator_seed'])
                for layout in spec['layouts']:
                    self.assertNotIn(layout['id'], ids)
                    ids.add(layout['id'])
                    pose_key = json.dumps(layout['poses'], sort_keys=True)
                    self.assertNotIn(pose_key, positions)
                    positions.add(pose_key)

    def test_changed_arrangements_are_not_only_new_seeds(self):
        for goal_id in ('temporal', 'cube_behind'):
            spec = build_layouts(goal_id, 'heldout')
            signs = []
            for row in spec['layouts']:
                p = row['poses']
                signs.append(np.sign(p['banana']['pos'][0] - p['rubiks_cube']['pos'][0]))
            self.assertEqual(signs.count(-1), 8)
            self.assertEqual(signs.count(1), 8)
        spec = build_layouts('mustard_right', 'heldout')
        signs = []
        for row in spec['layouts']:
            p = row['poses']
            delta = np.array(p['mustard_bottle']['pos']) - p['raisin_box']['pos']
            self.assertGreater(abs(delta[0]), .1)
            self.assertLess(abs(delta[1]), .015)
            signs.append(np.sign(delta[0]))
        self.assertEqual(signs.count(-1), 8)
        self.assertEqual(signs.count(1), 8)

    def test_generated_layouts_are_valid_legacy_runtime_inputs(self):
        from goals.registry import load_runtime_module
        with tempfile.TemporaryDirectory() as tmp:
            for goal_id in GOAL_IDS:
                spec = build_layouts(goal_id, 'development')
                path = Path(tmp) / (goal_id + '.json')
                write_layouts(path, spec)
                loaded = load_runtime_module(goal_id, 'layouts.py').load(path)
                self.assertEqual(loaded['case'], get_goal(goal_id)['runtime_case'])
                self.assertEqual(len(loaded['layouts']), 16)

    def test_freeze_never_replaces_existing_layouts(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'layouts.json'
            spec = build_layouts('temporal', 'heldout')
            write_layouts(path, spec)
            write_layouts(path, spec)
            changed = deepcopy(spec)
            changed['layouts'][0]['poses']['bowl']['pos'][0] += .001
            with self.assertRaises(FileExistsError):
                write_layouts(path, changed)


class SelectionTests(unittest.TestCase):
    def test_spatial_midpoint_is_fixed_without_target_actions(self):
        for n, expected in ((1, 0), (2, 0), (9, 4), (10, 4)):
            actions = np.zeros((n, 8, 8))
            grip = np.zeros(n)
            for goal_id in ('cube_behind', 'mustard_right'):
                self.assertEqual(select_failure_decision(goal_id, actions, grip), expected)
                actions[:, :, 7] = 1
                self.assertEqual(select_failure_decision(goal_id, actions, grip), expected)

    def test_temporal_release_is_not_generic_midpoint(self):
        actions = np.zeros((9, 8, 8))
        actions[1, :, 7] = 1
        grip = np.full((9, 1), .5)
        self.assertEqual(select_failure_decision('temporal', actions, grip), 2)
        grip[2] = 1.
        self.assertIsNone(select_failure_decision('temporal', actions, grip))

    def test_malformed_source_is_rejected(self):
        with self.assertRaises(ValueError):
            select_failure_decision('temporal', np.zeros((2, 8, 7)), np.zeros(2))
        with self.assertRaises(ValueError):
            select_failure_decision('cube_behind', np.zeros((2, 8, 8)), np.zeros(1))
        actions = np.zeros((2, 8, 8)); actions[0, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            select_failure_decision('mustard_right', actions, np.zeros(2))
        self.assertIsNone(select_failure_decision('cube_behind', np.zeros((0, 8, 8)), np.zeros(0)))


class ScoringTests(unittest.TestCase):
    def test_simultaneous_placements_do_not_pass_either_order(self):
        initial = {name: [[0., 0., 0.]] for name in ('rubiks_cube', 'banana', 'bowl')}
        scorer = new_scorer('temporal', initial, .2)
        for tick in range(10):
            scorer.update(tick, initial, {'rubiks_cube_placed': [True], 'banana_placed': [True]})
        self.assertFalse(scorer.current['cube_then_banana'][0])
        self.assertFalse(scorer.current['banana_then_cube'][0])

    def test_reverse_order_can_pass_without_primary_goal(self):
        initial = {name: [[0., 0., 0.]] for name in ('rubiks_cube', 'banana', 'bowl')}
        scorer = new_scorer('temporal', initial, .2)
        for tick in range(10):
            scorer.update(tick, initial, {'rubiks_cube_placed': [tick > 0], 'banana_placed': [True]})
        self.assertFalse(scorer.current['cube_then_banana'][0])
        self.assertTrue(scorer.current['banana_then_cube'][0])

    def test_mustard_reference_motion_is_not_forgotten(self):
        initial = {'mustard_bottle': [[0., 0., 0.]], 'raisin_box': [[1., 0., 0.]]}
        scorer = new_scorer('mustard_right', initial, .2)
        pos = {'mustard_bottle': [[0., -.1, 0.]], 'raisin_box': [[1., 0., 0.]]}
        pred = {'right_of': [True], 'left_of': [False], 'on_raisin': [False],
                'mustard_table_contact': [True], 'mustard_raisin_contact': [False],
                'mustard_detached': [True]}
        for tick in range(10):
            scorer.update(tick, pos, pred)
        self.assertTrue(scorer.current['mustard_right'][0])
        pos['raisin_box'] = [[1.03, 0., 0.]]
        scorer.update(10, pos, pred)
        pos['raisin_box'] = [[1., 0., 0.]]
        for tick in range(11, 22):
            scorer.update(tick, pos, pred)
        self.assertFalse(scorer.current['mustard_right'][0])
        self.assertTrue(scorer.stock['mustard_right'][0])

    def test_cube_behind_requires_detachment_and_dwell(self):
        initial = {'rubiks_cube': [[0., 0., 0.]], 'banana': [[0., 1., 0.]], 'bowl': [[1., 0., 0.]]}
        pos = {**initial, 'rubiks_cube': [[.05, 0., 0.]]}
        pred = {'behind': [True], 'cube_table_contact': [True], 'cube_bowl_contact': [False],
                'cube_detached': [True], 'cube_in_bowl': [False]}
        scorer = new_scorer('cube_behind', initial, .2)
        scorer.update(0, pos, pred)
        self.assertFalse(scorer.current['cube_behind'][0])
        for tick in range(1, 8):
            scorer.update(tick, pos, pred)
        self.assertTrue(scorer.current['cube_behind'][0])
        pred['cube_detached'] = [False]
        scorer.update(8, pos, pred)
        self.assertFalse(scorer.current['cube_behind'][0])


if __name__ == '__main__':
    unittest.main()
