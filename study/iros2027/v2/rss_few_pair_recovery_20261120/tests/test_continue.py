import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
spec = importlib.util.spec_from_file_location('fewpair_recovery_continue', HERE / 'continue.py')
C = importlib.util.module_from_spec(spec)
spec.loader.exec_module(C)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class ContinuationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.old = self.root / 'cohort'
        self.recovery = self.root / 'recovery'
        self.amendment = self.recovery / 'design/amendment.json'
        write(self.amendment, {'status': 'authorized_recovery'})
        self.patches = [patch.object(C, 'RECROOT', self.recovery), patch.object(C, 'OLDROOT', self.old),
                        patch.object(C, 'AMENDMENT', self.amendment),
                        patch.object(C, 'goal_root', lambda gid: self.old / gid)]
        for item in self.patches:
            item.start()

    def tearDown(self):
        for item in reversed(self.patches):
            item.stop()
        self.tmp.cleanup()

    def ready(self, goal_id='temporal'):
        manifest = self.old / goal_id / 'sources/block-02/source_manifest.json'
        write(manifest, {'qualified': True})
        source = {'status': 'qualified', 'manifest': str(manifest), 'manifest_sha256': C.sha(manifest)}
        write(self.old / goal_id / 'source-status.json', source)
        ready = {'status': 'ready', 'goal_id': goal_id, 'heldout_policy_episodes': 0,
                 'development_episodes': 208, 'source_manifest': str(manifest),
                 'source_manifest_sha256': C.sha(manifest),
                 'artifacts': {str(manifest): C.sha(manifest), str(self.amendment): C.sha(self.amendment)}}
        path = self.recovery / goal_id / 'ready-for-heldout.json'
        write(path, ready)
        return path, ready

    def test_readiness_binds_source_pool_and_recovery(self):
        path, _ = self.ready()
        result = C.validate_ready_record('temporal', path)
        self.assertEqual(result['status'], 'ready')
        write(self.amendment, {'status': 'different_recovery'})
        with self.assertRaisesRegex(ValueError, 'amendment'):
            C.validate_ready_record('temporal', path)

    def test_partial_selection_or_prior_heldout_cannot_release_tests(self):
        path, ready = self.ready()
        for field, value in [('development_episodes', 192), ('heldout_policy_episodes', 16)]:
            broken = {**ready, field: value}
            write(path, broken)
            with self.assertRaises(ValueError):
                C.validate_ready_record('temporal', path)

    def test_source_substitution_is_rejected(self):
        path, ready = self.ready()
        ready['source_manifest_sha256'] = '0' * 64
        write(path, ready)
        with self.assertRaisesRegex(ValueError, 'Source pool'):
            C.validate_ready_record('temporal', path)

    def test_wait_does_not_drop_missing_third_goal(self):
        self.ready('temporal')
        self.ready('cube_behind')
        with patch.object(C, 'verify_full_ready') as verify_ready:
            with self.assertRaisesRegex(TimeoutError, 'no goal was dropped'):
                C.wait_for_readiness(0)
            verify_ready.assert_not_called()

    def test_preparation_stop_terminates_wait_immediately(self):
        write(self.recovery / 'mustard_right/prepare-status.json', {'status': 'stopped', 'reason': 'test'})
        with self.assertRaisesRegex(RuntimeError, 'mustard_right stopped'):
            C.wait_for_readiness(100)

    def test_global_freeze_must_match_all_registered_dispositions(self):
        goals = C.protocol()['goal_ids']
        readiness = {gid: {'status': 'ready'} for gid in goals}
        frozen = {'status': 'frozen', 'cohort_id': C.protocol()['cohort_id'],
                  'goals': {gid: {'qualified': True} for gid in goals}, 'artifacts': {},
                  'planned_heldout_episodes': 544 * len(goals)}
        path = self.old / 'all-heldout-frozen.json'
        write(path, frozen)
        _, qualified = C.verified_global_freeze(readiness)
        self.assertEqual(qualified, goals)
        frozen['goals'].pop('mustard_right')
        write(path, frozen)
        with self.assertRaisesRegex(ValueError, 'every registered goal'):
            C.verified_global_freeze(readiness)

    def test_partial_phase_is_not_automatically_replayed(self):
        directory = self.recovery / 'continuation/temporal-heldout'
        write(directory / 'partial.json', {'status': 'partial'})
        with patch.object(C, 'run_commands') as run:
            with self.assertRaisesRegex(ValueError, 'Partial continuation'):
                C.phase('temporal-heldout', [['worker']], lambda **kwargs: None, 10)
            run.assert_not_called()

    def test_complete_phase_is_reused_only_for_identical_commands(self):
        commands = [['worker', '--gpu', '0']]
        write(self.recovery / 'continuation/temporal-heldout/completed.json',
              {'status': 'complete', 'commands': commands})
        with patch.object(C, 'run_commands') as run:
            C.phase('temporal-heldout', commands, lambda **kwargs: None, 10)
            run.assert_not_called()
            with self.assertRaisesRegex(ValueError, 'different commands'):
                C.phase('temporal-heldout', [['different']], lambda **kwargs: None, 10)


if __name__ == '__main__':
    unittest.main()
