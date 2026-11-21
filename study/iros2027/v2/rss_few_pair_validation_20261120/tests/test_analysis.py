"""Receipt-integrity and clustered inference tests; no simulator or policy calls."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import common
import analyze
import evaluate


def rows(outcomes, repeats=1, camera='same'):
    return [{'slot': 16 * k + j, 'layout_id': f'layout-{j}', 'policy_seed': 100 * (16 * k + j),
             'success': bool(value), 'first_camera_sha256': camera}
            for k in range(repeats) for j, value in enumerate(outcomes)]


class AnalysisTests(unittest.TestCase):
    def fixture(self, root, stage='heldout'):
        """Make an entire valid protocol-shaped stage with tiny evidence files."""
        gid = 'temporal'
        gr = root / gid
        layout = gr / ('design/development-layouts.json' if stage == 'selection' else 'design/heldout-layouts.json')
        common.save(layout, {'layouts': [{'id': f'layout-{j}'} for j in range(16)]})
        if stage == 'selection':
            candidates = []
            for family in ('embedding', 'residual_g0', 'residual_g1', 'residual_g2'):
                for strength in (.5, 1., 2.):
                    candidates.append({'candidate_id': f'{family}-{strength}', 'family': family,
                        'strength': strength, 'fingerprint': f'{family}-{strength}',
                        'workspace': str(gr / 'adaptation' / f'{family}-{strength}')})
            batches = [evaluate.selection_cell(gid, 'base', None)]
            batches += [evaluate.selection_cell(gid, c['candidate_id'], c['workspace']) for c in candidates]
        else:
            candidates = [{'candidate_id': name, 'fingerprint': name,
                           'workspace': str(gr / 'adaptation' / name)} for name in analyze.METHODS if name != 'base']
            batches = evaluate.test_cells(gid, 'base', None) + evaluate.test_cells(gid, 'clarified', None)
            for c in candidates:
                batches += evaluate.test_cells(gid, c['candidate_id'], c['workspace'])
        assignment = [j % 2 for j in range(len(batches))]
        schedule = {'cohort_id': common.protocol()['cohort_id'], 'goal_id': gid, 'stage': stage,
                    'batches': batches, 'gpu_assignment': assignment, 'candidates': candidates}
        frozen = gr / f'{stage}-frozen'
        common.save(frozen / 'schedule.json', schedule)
        g = common.goal(gid)
        for gpu in (0, 1):
            assigned = [b for b, owner in zip(batches, assignment) if owner == gpu]
            plan = {'cohort_id': common.protocol()['cohort_id'], 'goal_id': gid, 'case': g['runtime_case'],
                    'runtime_dir': g['runtime_dir'], 'stage': stage, 'split': 'development' if stage == 'selection' else 'heldout',
                    'inference_batching': 'fixed16', 'protocol_sha256': common.sha(HERE / 'protocol.json'),
                    'schedule_sha256': common.canonical(schedule), 'batches': assigned}
            plan_path = common.freeze_plan(frozen / f'plan-g{gpu}.json', plan, {str(layout): common.sha(layout)})
            plan = common.read(plan_path)
            wr = gr / stage / f'g{gpu}'
            common.save(wr / 'plan.json', plan)
            common.save(wr / 'identity.json', {str(layout): common.sha(layout),
                plan['global_freeze']['path']: plan['global_freeze']['sha256']})
            common.save(wr / 'status.json', {'status': 'complete', 'cohort_id': plan['cohort_id'], 'split': plan['split'],
                'gpu': gpu, 'completed_batches': len(assigned), 'completed_episodes': 16 * len(assigned),
                'planned_episodes': 16 * len(assigned)})
            for number, b in enumerate(assigned):
                batch = wr / f"batch-{number:02d}-{b['cell']}-{b['condition']}"
                batch.mkdir()
                metric_path = batch / 'metrics_r0.json'
                outcomes = [j % 3 == 0 for j in range(16)]
                common.save(metric_path, {'reset_calls': 2, 'layout': {'sha256': common.sha(layout),
                    'layout_ids': [f'layout-{j}' for j in range(16)]}, 'selected_goal': b['goal'],
                    'strict_final': {b['goal']: outcomes}})
                episodes = []
                for j, success in enumerate(outcomes):
                    obs = batch / f'obs-{j}.npz'
                    obs.write_bytes(f'fixture-{j}'.encode())
                    slot = b['slot_start'] + j
                    episodes.append({'slot': slot, 'env': j, 'layout_id': f'layout-{j}', 'policy_seed': ((slot // 16) * 64 + j) * 100,
                        'success': success, 'strict_final': {b['goal']: success}, 'first_camera_sha256': {'base_img': 'a', 'wrist_img': 'b'},
                        'cohort_id': plan['cohort_id'], 'goal_id': gid, 'case': plan['case'], 'split': plan['split'],
                        'condition': b['condition'], 'cell': b['cell'], 'input_prompt': b['input_prompt'],
                        'effective_prompt': b['effective_prompt'], 'selected_goal': b['goal'], 'task_name': b['task'],
                        'inference_batching': 'fixed16', 'ok': True, 'obs': str(obs), 'obs_sha256': common.sha(obs),
                        'metrics': str(metric_path), 'metrics_sha256': common.sha(metric_path)})
                common.save(batch / 'episodes.json', episodes)
                log = batch / 'sim-l0a0.log'
                log.write_text('single attempt fixture\n')
                server = {'workspace': b['workspace']}
                if b['workspace']:
                    server['fingerprint'] = b['condition']
                common.save(batch / 'completed.json', {'batch': number, 'cell': b['cell'], 'condition': b['condition'],
                    'n': 16, 'successes': sum(outcomes), 'server': json.dumps(server), 'sim_attempt_logs': [str(log)],
                    'wall_seconds': 2., 'execution_seconds': 1.5, 'queue_seconds': .5,
                    'episodes_sha256': common.sha(batch / 'episodes.json')})
        return schedule

    def test_complete_matrix_has_one_shared_base_and_each_source_group(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            self.fixture(Path(tmp))
            groups, receipts, artifacts = analyze.collect_stage('temporal', 'heldout')
            self.assertEqual(sum(r['n'] for r in receipts), 544)
            self.assertEqual(sum(len(v) for (condition, _), v in groups.items() if condition == 'base'), 64)
            self.assertEqual(len(groups[('weightsmith_g2', 'original')]), 32)
            self.assertEqual(len(groups[('embedding', 'competing')]), 16)
            self.assertTrue(artifacts)
            result = analyze.summarize(groups)
            self.assertEqual(sum(r['n'] for r in result['primary'] if r['condition'] == 'base'), 64)
            self.assertEqual(len(result['source_group_summary']), 6)
            for group in analyze.GROUPS:
                comparisons = [r for r in result['paired'] if r['condition'] == f'weightsmith_{group}' and r['cell'] == 'original']
                self.assertEqual({r['reference'] for r in comparisons}, {'base', 'embedding', f'residual_{group}', 'clarified'})

    def test_selection_requires_all_208_trials(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            self.fixture(Path(tmp), 'selection')
            groups, receipts, _ = analyze.collect_stage('temporal', 'selection')
            self.assertEqual(len(groups), 13)
            self.assertEqual(sum(r['n'] for r in receipts), 208)
            path = Path(tmp) / 'temporal/selection/g1/status.json'
            value = common.read(path)
            common.save(path, {**value, 'status': 'running'})
            with self.assertRaisesRegex(ValueError, 'all planned trials'):
                analyze.collect_stage('temporal', 'selection')

    def test_tampered_observations_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            self.fixture(Path(tmp), 'selection')
            obs = next((Path(tmp) / 'temporal/selection/g0').glob('batch-*/obs-0.npz'))
            obs.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'observations changed'):
                analyze.collect_stage('temporal', 'selection')

    def test_forged_score_rejected_against_physical_metrics(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            self.fixture(Path(tmp), 'selection')
            batch = next((Path(tmp) / 'temporal/selection/g0').glob('batch-*'))
            episodes = common.read(batch / 'episodes.json')
            episodes[0]['success'] = not episodes[0]['success']
            episodes[0]['strict_final'][episodes[0]['selected_goal']] = episodes[0]['success']
            common.save(batch / 'episodes.json', episodes)
            receipt = common.read(batch / 'completed.json')
            common.save(batch / 'completed.json', {**receipt, 'episodes_sha256': common.sha(batch / 'episodes.json'),
                                                   'successes': sum(r['success'] for r in episodes)})
            with self.assertRaisesRegex(ValueError, 'Physical scoring'):
                analyze.collect_stage('temporal', 'selection')

    def test_schedule_cannot_hide_a_source_group(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            schedule = self.fixture(Path(tmp))
            schedule['batches'] = schedule['batches'][:-1]
            schedule['gpu_assignment'] = schedule['gpu_assignment'][:-1]
            with self.assertRaisesRegex(ValueError, 'omits or adds'):
                analyze._check_schedule(schedule, 'temporal', 'heldout')

    def test_wrong_candidate_load_rejected(self):
        with tempfile.TemporaryDirectory() as tmp, patch.object(common, 'ROOT', Path(tmp)):
            self.fixture(Path(tmp), 'selection')
            path = next((Path(tmp) / 'temporal/selection/g1').glob('batch-*/completed.json'))
            value = common.read(path)
            server = json.loads(value['server'])
            server['fingerprint'] = 'another-model'
            common.save(path, {**value, 'server': json.dumps(server)})
            with self.assertRaisesRegex(ValueError, 'fingerprint'):
                analyze.collect_stage('temporal', 'selection')

    def test_layout_bootstrap_keeps_noise_replicates_together(self):
        # Every layout has one success and one failure: resampling whole layouts
        # must preserve .5 exactly, unlike an erroneous trial-level bootstrap.
        result = analyze.interval([0] * 16 + [1] * 16, list(range(16)) * 2, draws=1000)
        self.assertEqual((result['ci_low'], result['ci_high']), (.5, .5))
        self.assertEqual(result['layout_clusters'], 16)
        with self.assertRaisesRegex(ValueError, 'balanced'):
            analyze.interval([1, 0, 1], ['a', 'a', 'b'])

    def test_pairing_rejects_wrong_seed_and_counts_camera_mismatches(self):
        base = rows([False] * 16, repeats=2)
        edit = rows([True] * 16, repeats=2, camera='different')
        result = analyze.contrast(edit, base, 'weightsmith_g0', 'base', 'original')
        self.assertEqual((result['difference'], result['fixed'], result['broken']), (1., 32, 0))
        self.assertEqual(result['first_camera_mismatches'], 32)
        edit[0]['policy_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'identities differ'):
            analyze.contrast(edit, base, 'weightsmith_g0', 'base', 'original')

    def test_aggregate_keeps_failed_goals_and_source_costs(self):
        reports = []
        for i, gid in enumerate(common.protocol()['goal_ids']):
            qualified = i == 0
            reports.append({'goal_id': gid, 'status': 'complete_and_verified' if qualified else 'qualification_failed_verified',
                'source_collection': {'qualified': qualified, 'source_policy_episodes': 32 if qualified else 96},
                'new_selection_policy_episodes': 208 if qualified else 0, 'new_heldout_policy_episodes': 544 if qualified else 0})
        result = analyze.aggregate_reports(reports)
        self.assertEqual((result['registered_goals'], result['qualified_goals']), (3, 1))
        self.assertEqual(result['new_source_policy_episodes'], 224)
        self.assertEqual(result['new_heldout_policy_episodes'], 544)
        with self.assertRaisesRegex(ValueError, 'every registered goal'):
            analyze.aggregate_reports(reports[:1])


if __name__ == '__main__':
    unittest.main()
