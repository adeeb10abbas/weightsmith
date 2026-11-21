"""Scientific invariants of the independent few-pair constructions (CPU only)."""
from __future__ import annotations
import copy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[3]))
from adaptation import recipe, sources
from weightsmith import distill


def token_pair(suffix_length=36):
    def row(instruction):
        values = [2, 77, 90]+instruction+[distill.PI05_STATE_TOKEN]+list(range(101, 100+suffix_length))
        n = len(values)
        return {'prompt_token_ids_at_slots': values+[0]*(96-n),
                'prompt_token_valid_at_slots': [True]*n+[False]*(96-n),
                'episode': 'item-0000', 'k': 3, 'seed': 5, 'sim_step': 8}
    return row([11, 12, 13]), row([21, 22, 23, 24, 25])


def arrays(seed):
    rng = np.random.default_rng(seed)
    return {'vlm.mlp_hidden': rng.normal(size=(9, 17, 40, 5)),
            'vlm.resid_out': rng.normal(size=(9, 17, 40, 3)),
            'vlm.img_mlp_hidden': rng.normal(size=(9, 17, 2, 17, 5)),
            'vlm.img_resid_out': rng.normal(size=(9, 17, 2, 17, 3))}


class AlignmentTests(unittest.TestCase):
    def test_capture_adapter_only_extends_instrumentation_limit(self):
        original = (HERE.parents[3]/'weightsmith/acts_worker.py').read_text()
        expected = original.replace('str(Path(__file__).resolve().parents[1])',
                                    'str(Path(__file__).resolve().parents[5])')
        expected = expected.replace('if len(ptoks) > 64 or', 'if len(ptoks) > 96 or')
        expected = expected.replace('at most 64")', 'at most 96")')
        owned = (HERE/'adaptation/capture_worker.py').read_text().splitlines(keepends=True)
        self.assertEqual(''.join(owned[3:]), expected)

    def test_fixed_window_retains_variable_padding(self):
        for suffix_length, valid_count in ((35, 38), (36, 39), (37, 40), (39, 40)):
            a, b = token_pair(suffix_length)
            pairs = recipe.fixed_window_slots(a, b)
            self.assertEqual(pairs, [(0, 0), (1, 1), (2, 2)]+[(6+i, 8+i) for i in range(37)])
            self.assertEqual(sum(a['prompt_token_valid_at_slots'][x] for x, _ in pairs), valid_count)

    def test_padding_mismatch_is_rejected(self):
        a, b = token_pair(36)
        b['prompt_token_ids_at_slots'][44] = 999
        with self.assertRaises(ValueError):
            recipe.fixed_window_slots(a, b)

    def test_short_capture_is_rejected(self):
        a, b = token_pair()
        a['prompt_token_ids_at_slots'] = a['prompt_token_ids_at_slots'][:42]
        a['prompt_token_valid_at_slots'] = a['prompt_token_valid_at_slots'][:42]
        with self.assertRaises(ValueError):
            recipe.fixed_window_slots(a, b)

    def test_capture_identity_and_offsets_must_match(self):
        a, b = token_pair()
        oi = {'vlm_layers': list(range(17)), 'prompt_token_slots': list(range(96)), 'items': [a]}
        ti = {**oi, 'items': [b]}
        slots, records = recipe.capture_alignment(oi, ti, expected_count=1)
        self.assertEqual(len(slots), 40)
        self.assertEqual(records[0]['padding_positions'], 1)
        ti = copy.deepcopy(ti)
        ti['items'][0]['seed'] += 1
        with self.assertRaisesRegex(ValueError, 'identity'):
            recipe.capture_alignment(oi, ti, expected_count=1)


class ConstructionTests(unittest.TestCase):
    def test_arbitrary_triplet_isolated_from_other_six_sources(self):
        original, teacher = arrays(10), arrays(11)
        selected = [8, 0, 5]
        slots = [(i, i) for i in range(40)]
        first = recipe.pooled_key_values(original, teacher, selected, slots)
        altered = copy.deepcopy(teacher)
        altered['vlm.resid_out'][[1, 2, 3, 4, 6, 7]] += 10000
        altered['vlm.img_resid_out'][[1, 2, 3, 4, 6, 7]] -= 10000
        second = recipe.pooled_key_values(original, altered, selected, slots)
        for layer in range(17):
            np.testing.assert_array_equal(first[layer][0], second[layer][0])
            np.testing.assert_array_equal(first[layer][1], second[layer][1])
        keys, values = first[4]
        self.assertEqual(keys.shape[0], 72)
        np.testing.assert_allclose(keys[40], original['vlm.img_mlp_hidden'][selected, 4, 0, 1].mean(0))
        gap_now = (teacher['vlm.resid_out'][selected, 4, 0]-original['vlm.resid_out'][selected, 4, 0]).mean(0)
        gap_prev = (teacher['vlm.resid_out'][selected, 3, 0]-original['vlm.resid_out'][selected, 3, 0]).mean(0)
        np.testing.assert_allclose(values[0], gap_now-gap_prev)

    def test_duplicate_or_nontriplet_sources_fail(self):
        original, teacher = arrays(10), arrays(11)
        for selected in ([0, 0, 1], [0, 1], [0, 1, 9], [0, 1, True]):
            with self.assertRaises(ValueError):
                recipe.pooled_key_values(original, teacher, selected, [(i, i) for i in range(40)])

    def test_ridge_solution_satisfies_regularized_normal_equation(self):
        rng = np.random.default_rng(3)
        keys, values = rng.normal(size=(72, 8)), rng.normal(size=(72, 4))
        delta, report = recipe.fit(keys, values)
        np.testing.assert_allclose((keys.T@keys+report['ridge_absolute']*np.eye(8))@delta,
                                   keys.T@values, atol=1e-9, rtol=1e-9)
        self.assertEqual(report['keys'], 72)
        for bad in (np.zeros_like(keys), np.full_like(keys, np.nan)):
            with self.assertRaises(ValueError):
                recipe.fit(bad, values)

    def test_merged_updates_are_exactly_bf16_representable(self):
        base = np.array([[1., -3., .003]], dtype=np.float32)
        delta = np.array([[.00390625, .008, .000032]], dtype=np.float64)
        result = recipe.merge(base, delta)
        np.testing.assert_array_equal(result.view(np.uint32) & 0xFFFF, np.zeros(result.shape, dtype=np.uint32))
        np.testing.assert_array_equal(result, distill.bf16_round(base.astype(np.float64)+delta))


class SourceQualificationTests(unittest.TestCase):
    def test_triplets_partition_first_nine_and_cannot_be_reselected(self):
        ids = [f'pair-{i}' for i in range(9)]
        seed = 2026112091
        order = np.random.default_rng(seed).permutation(9).tolist()
        groups = [{'group_id': f'g{i}', 'indices': order[3*i:3*i+3],
                   'pair_ids': [ids[k] for k in order[3*i:3*i+3]]} for i in range(3)]
        self.assertEqual(sources.validate_groups(groups, ids, seed), groups)
        bad = copy.deepcopy(groups)
        bad[0]['indices'][0], bad[1]['indices'][0] = bad[1]['indices'][0], bad[0]['indices'][0]
        with self.assertRaises(ValueError):
            sources.validate_groups(bad, ids, seed)
        with self.assertRaises(ValueError):
            sources.validate_groups(groups, ids[:-1]+[ids[0]], seed)

    def test_unqualified_goal_stops_before_native_loader(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(sources, 'goal_root', return_value=Path(directory)), \
                 patch.object(sources, 'read', return_value={'status': 'collecting', 'qualified': False}), \
                 patch.object(sources, 'load') as loader:
                with self.assertRaisesRegex(ValueError, 'no GPU'):
                    sources.qualified('temporal')
                loader.assert_not_called()


if __name__ == '__main__':
    unittest.main()
