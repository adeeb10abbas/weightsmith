"""CPU checks for the scientific contrasts and immutable evaluation boundary."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import pipeline
import plan
import report


class TargetTests(unittest.TestCase):
    def test_local_target_matches_mean_of_projected_module_outputs(self):
        rng = np.random.default_rng(8)
        original = rng.normal(size=(3, 32, 7))
        teacher = rng.normal(size=(3, 32, 7))
        weights = rng.normal(size=(7, 5))
        target = pipeline.local_mlp_target(original, teacher, weights)
        expected = np.mean([b @ weights - a @ weights for a, b in zip(original, teacher)], axis=0)
        np.testing.assert_allclose(target, expected, rtol=1e-12, atol=1e-12)
        self.assertEqual(target.shape, (32, 5))
        with self.assertRaises(ValueError):
            pipeline.local_mlp_target(original, teacher[:-1], weights)

    def test_shuffle_breaks_cell_correspondence_but_preserves_target_spectrum(self):
        rng = np.random.default_rng(3)
        target = rng.normal(size=(32, 9))
        keys = rng.normal(size=(32, 6))
        order = pipeline.within_camera_permutation()
        self.assertTrue(np.all(order != np.arange(32)))
        self.assertEqual(set(order[:16]), set(range(16)))
        self.assertEqual(set(order[16:]), set(range(16, 32)))
        variant = pipeline.target_variants(keys, target, target * .2, order)
        np.testing.assert_array_equal(variant['residual_increment'], target)
        np.testing.assert_array_equal(variant['local_mlp'], target * .2)
        np.testing.assert_allclose(np.linalg.svd(variant['shuffled_increment'], compute_uv=False),
                                   np.linalg.svd(target, compute_uv=False), atol=1e-12)
        self.assertGreater(np.linalg.norm(variant['shuffled_increment'] - target), 0)
        np.testing.assert_array_equal(order, pipeline.within_camera_permutation())

    def test_shared_fit_matches_independent_primal_mean_preservation_objective(self):
        core = pipeline.module('ablation_test_prior_fit', pipeline.PRIOR / 'edit/pipeline.py')
        rng = np.random.default_rng(6)
        K, V, H = rng.normal(size=(32, 7)), rng.normal(size=(32, 4)), rng.normal(size=(64, 7))
        delta, _ = core.fit(K, V, H)
        ridge = .001 * np.sum(K * K) / 32
        expected = np.linalg.solve(K.T @ K + H.T @ H / 2 + ridge * np.eye(7), K.T @ V)
        np.testing.assert_allclose(delta, expected, rtol=1e-9, atol=1e-10)
        duplicate, _ = core.fit(K, V, np.concatenate([H, H]))
        np.testing.assert_allclose(delta, duplicate, rtol=1e-9, atol=1e-10)

    def test_source_seal_refuses_changed_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'source.json'
            path.write_text('{}')
            seal = {str(path): pipeline.sha(path)}
            pipeline.verify_map(seal)
            path.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'changed'):
                pipeline.verify_map(seal)

    def test_development_boundary_rejects_heldout_before_reading_captures(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / 'temporal-adaptation'
            pipeline.dump(root / 'edit-fit/solve.json', {})
            pipeline.dump(root / 'edit-prepared/prepared.json', {'split': 'heldout'})
            pipeline.dump(root / 'spec.json', {'split': 'development'})
            with self.assertRaisesRegex(ValueError, 'Heldout data'):
                pipeline.source_bundle(root, 'temporal')

    def test_fresh_layouts_and_complete_plan_identity(self):
        protocol = pipeline.read(pipeline.HERE / 'protocol.json')
        for case in ('temporal', 'spatial'):
            layout = plan.layouts(case, protocol)
            self.assertEqual(len(layout['layouts']), 8)
            self.assertEqual(layout, plan.layouts(case, protocol))
            batches = [{'cell': cell['id'], 'condition': method, 'task': cell['task'], 'goal': cell['goal'],
                        'input_prompt': cell['prompt'], 'effective_prompt': cell['prompt'],
                        'workspace': None if method == 'base' else '/synthetic/workspace',
                        'n': 16, 'slot_start': protocol['heldout']['slot_start'][case]}
                       for cell in protocol['cases'][case]['cells'] for method in pipeline.METHODS]
            value = {'case': case, 'cohort_id': protocol['cohort'], 'split': 'heldout',
                     'inference_batching': 'fixed16', 'batches': batches}
            plan.check_plan(value, protocol)
            changed = copy.deepcopy(value)
            changed['batches'][0]['effective_prompt'] = 'a different task'
            with self.assertRaises(ValueError):
                plan.check_plan(changed, protocol)
            self.assertNotEqual(plan.payload_sha(value), plan.payload_sha(changed))
            changed = copy.deepcopy(value)
            changed['batches'][-1] = changed['batches'][0]
            with self.assertRaises(ValueError):
                plan.check_plan(changed, protocol)

    def test_original_worker_binds_all_new_conditions_to_global_freeze(self):
        worker = pipeline.module('ablation_test_frozen_worker', pipeline.PRIOR / 'worker.py')
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / 'candidate'
            artifact.write_text('sealed')
            value = {'case': 'temporal', 'split': 'heldout', 'batches': [{'condition': 'local_mlp'}]}
            freeze = root / 'freeze.json'
            pipeline.dump(freeze, {'artifacts': {str(artifact): pipeline.sha(artifact)},
                                   'plan_sha256': {'temporal': plan.payload_sha(value)}})
            value['global_freeze'] = {'path': str(freeze), 'sha256': pipeline.sha(freeze)}
            worker.verify_global_freeze(value)
            value['batches'][0]['condition'] = 'shuffled_increment'
            with self.assertRaises(ValueError):
                worker.verify_global_freeze(value)

    def test_complete_report_keeps_all_384_trials_and_30_paired_contrasts(self):
        protocol = pipeline.read(pipeline.HERE / 'protocol.json')
        rows = []
        for case, definition in protocol['cases'].items():
            for cell in definition['cells']:
                for method in pipeline.METHODS:
                    for i in range(16):
                        success = i % 2 == 0 if method == 'base' else i % 4 != 0
                        goals = {c['goal']: False for c in definition['cells']}
                        goals[cell['goal']] = success
                        rows.append({'case': case, 'cell': cell['id'], 'condition': method,
                                     'slot': i, 'policy_seed': i * 100, 'layout_id': f'L{i % 8}',
                                     'success': success, 'strict_final': goals, 'strict_ever': goals,
                                     'stock_final': goals, 'first_camera_sha256': {'base_img': 'a', 'wrist_img': 'b'}})
        self.assertEqual(len(rows), 384)
        result = report.summarize(protocol, rows)
        self.assertEqual(len(result['primary']), 24)
        self.assertEqual(len(result['paired']), 30)
        contrast = result['paired'][0]
        self.assertEqual((contrast['fixed'], contrast['broken'], contrast['both_success'], contrast['both_failure']),
                         (8, 4, 4, 0))
        self.assertEqual(contrast['success_rate_difference'], .25)
        self.assertTrue(all(r['layout_clusters'] == 8 for r in result['paired']))
        changed = copy.deepcopy(rows)
        changed[16]['policy_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'Paired identity'):
            report.summarize(protocol, changed)

    def test_cluster_interval_does_not_count_noise_repeats_as_new_layouts(self):
        values, layouts = [0, 1] * 8, [f'L{i}' for i in range(8) for _ in range(2)]
        actual = report.interval(values, layouts)
        self.assertEqual((actual['ci_low'], actual['ci_high'], actual['layout_clusters']), (.5, .5, 8))
        self.assertTrue(actual['degenerate_ci'])
        with self.assertRaises(ValueError):
            report.interval(values, [f'L{i}' for i in range(16)])

    def test_raw_metric_goal_and_camera_checks(self):
        import hashlib
        protocol = pipeline.read(pipeline.HERE / 'protocol.json')
        case, condition = 'temporal', 'base'
        cell = protocol['cases'][case]['cells'][0]
        layout = plan.layouts(case, protocol)
        slot = protocol['heldout']['slot_start'][case]
        seed = (slot // 16 * 64) * 100
        obs = {'seed': np.array([seed]), 'base_img': np.zeros((1, 2, 2, 3), dtype=np.uint8),
               'wrist_img': np.ones((1, 2, 2, 3), dtype=np.uint8)}
        metrics = {'selected_goal': cell['goal'],
                   'layout': {'layout_ids': [layout['layouts'][i % 8]['id'] for i in range(16)]}}
        row = {'case': case, 'cell': cell['id'], 'condition': condition, 'cohort_id': protocol['cohort'],
               'split': 'heldout', 'inference_batching': 'fixed16', 'ok': True,
               'input_prompt': cell['prompt'], 'effective_prompt': cell['prompt'], 'task_name': cell['task'],
               'selected_goal': cell['goal'], 'slot': slot, 'env': 0, 'group': slot // 16,
               'layout_id': layout['layouts'][0]['id'], 'success': True, 'policy_seed': seed, 'decision_count': 1,
               'first_camera_sha256': {k: hashlib.sha256(obs[k][0].tobytes()).hexdigest() for k in ('base_img', 'wrist_img')}}
        for kind in ('strict_final', 'strict_ever', 'stock_final', 'stock_ever'):
            row[kind] = {cell['goal']: True}
            metrics[kind] = {cell['goal']: [True] * 16}
        report.validate_row(row, cell, case, condition, protocol, layout, metrics, obs)
        changed = copy.deepcopy(row)
        changed['selected_goal'] = 'banana_then_cube'
        with self.assertRaisesRegex(ValueError, 'Selected goal'):
            report.validate_row(changed, cell, case, condition, protocol, layout, metrics, obs)
        changed = copy.deepcopy(row)
        changed['first_camera_sha256']['base_img'] = 'wrong'
        with self.assertRaisesRegex(ValueError, 'First-camera'):
            report.validate_row(changed, cell, case, condition, protocol, layout, metrics, obs)

    def test_analysis_does_not_open_partial_outcomes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pipeline.dump(root / 'temporal-heldout-v1/status.json', {'status': 'running'})
            with self.assertRaisesRegex(ValueError, 'outcomes remain unopened'):
                report.collect(root)


if __name__ == '__main__':
    unittest.main()
