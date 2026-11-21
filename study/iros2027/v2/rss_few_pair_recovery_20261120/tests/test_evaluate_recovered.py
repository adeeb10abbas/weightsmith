import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import evaluate_recovered as E


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


class RecoveredEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patch = patch.object(E, 'goal_root', lambda gid: self.root / gid)
        self.patch.start()

    def tearDown(self):
        self.patch.stop()
        self.tmp.cleanup()

    def test_worker_copy_changes_only_runtime_import_and_identity(self):
        original = (E.STUDY / 'worker.py').read_text()
        expected = original.replace(
            "HERE = Path(__file__).resolve().parent\nREPO = HERE.parents[3]\nsys.path.insert(0, str(REPO))",
            "RECOVERY = Path(__file__).resolve().parent\nHERE = RECOVERY.parent / 'rss_few_pair_validation_20261120'\nREPO = HERE.parents[3]\nsys.path.insert(0, str(HERE))\nsys.path.insert(0, str(REPO))")
        expected = expected.replace("SERVER_RUNTIME = HERE / 'baselines' if plan.get('stage') in ('selection','heldout') else RUNTIME",
            "SERVER_RUNTIME = RECOVERY / 'baselines_v2' if plan.get('stage') in ('selection','heldout') else RUNTIME")
        expected = expected.replace("sources = [args.plan, *native_sources(plan['goal_id'])]",
            "sources = [Path(__file__).resolve(), args.plan, *native_sources(plan['goal_id'])]")
        self.assertEqual(E.WORKER.read_text(),
            '# Recovery copy: import root, baseline server path, and own source identity only.\n' + expected)

    def test_weight_candidate_delegates_to_unchanged_native_validator(self):
        candidate = {'workspace': str(self.root / 'temporal/adaptation/weightsmith/g0')}
        with patch.object(E, 'ORIGINAL_CANDIDATE', return_value={'validated': True}) as original:
            self.assertEqual(E.candidate_identity(candidate, 'temporal'), {'validated': True})
            original.assert_called_once_with(candidate, 'temporal')

    def test_baseline_loader_is_verified_and_record_shape_is_compatible(self):
        workspace = self.root / 'temporal/adaptation/baselines/embedding/strength-1p0'
        write(workspace / 'baseline.json', {'schema': 'legacy'})
        (workspace / 'direction.npy').write_bytes(b'array fixture')
        candidate = {'candidate_id': 'embedding-s1p0', 'workspace': str(workspace),
                     'method': 'embedding', 'strength': 1.0}
        spec = {'method': 'embedding_mean', 'strength': 1.0, 'layer': None}
        from unittest.mock import Mock
        api = SimpleNamespace(load_artifact=Mock(return_value=(spec, None)))
        with patch.object(E, 'artifact_api', return_value=api):
            actual = E.candidate_identity(candidate, 'temporal')
        api.load_artifact.assert_called_once_with(workspace, verify_sources=True)
        self.assertEqual(set(actual), set(candidate) | {'fingerprint', 'artifacts'})
        self.assertEqual(actual['fingerprint'], E.sha(workspace / 'baseline.json'))
        self.assertEqual(set(actual['artifacts']), {str(workspace / 'baseline.json'), str(workspace / 'direction.npy')})

    def test_baseline_strength_mismatch_is_rejected(self):
        workspace = self.root / 'cube_behind/adaptation/baselines-recovery-v2/embedding/strength-1p0'
        write(workspace / 'baseline.json', {'schema': 'v2'})
        candidate = {'workspace': str(workspace), 'method': 'embedding', 'strength': 1.0}
        api = SimpleNamespace(load_artifact=lambda *args, **kwargs:
                              ({'method': 'embedding_mean', 'strength': 2.0}, None))
        with patch.object(E, 'artifact_api', return_value=api):
            with self.assertRaisesRegex(ValueError, 'strength'):
                E.candidate_identity(candidate, 'cube_behind')

    def test_existing_temporal_selection_is_not_resealed(self):
        path = self.root / 'temporal/selection-frozen/schedule.json'
        write(path, {'batches': ['old']})
        with patch.object(E, 'ORIGINAL_SEAL', return_value='existing') as original, \
                patch.object(E, 'runtime_artifacts') as runtime:
            result = E.seal_schedule('temporal', 'selection', ['old'], {}, {})
        self.assertEqual(result, 'existing')
        runtime.assert_not_called()
        original.assert_called_once_with('temporal', 'selection', ['old'], {}, {})

    def test_new_schedule_pins_certified_worker_and_runtime(self):
        gate = self.root / 'gate.json'
        amendment = self.root / 'amendment.json'
        write(gate, {'status': 'pass'})
        write(amendment, {'status': 'authorized_technical_recovery_before_heldout'})
        extra = {str(gate): E.sha(gate)}
        with patch.object(E, 'runtime_artifacts', return_value=extra), \
                patch.object(E, 'GATE', gate), patch.object(E, 'BASELINE_AMENDMENT', amendment), \
                patch.object(E, 'ORIGINAL_SEAL', return_value='sealed') as original:
            E.seal_schedule('cube_behind', 'selection', [], {'original': 'hash'}, {'candidates': []})
        args = original.call_args.args
        self.assertEqual(args[3], {'original': 'hash', **extra})
        self.assertEqual(args[4]['worker_entrypoint'], str(E.WORKER))
        self.assertEqual(args[4]['baseline_runtime_gate_sha256'], E.sha(gate))

    def test_worker_dispatch_keeps_legacy_selection_and_routes_new_phases(self):
        legacy = ['python', str(E.STUDY / 'worker.py'), '--plan', 'plan']
        write(self.root / 'temporal/selection-frozen/schedule.json', {})
        with patch.object(E, 'original_worker_commands', return_value=[legacy.copy()]):
            self.assertEqual(E.worker_commands('temporal', 'selection'), [legacy])
        write(self.root / 'temporal/heldout-frozen/schedule.json', {'worker_entrypoint': str(E.WORKER)})
        with patch.object(E, 'original_worker_commands', return_value=[legacy.copy()]):
            actual = E.worker_commands('temporal', 'heldout')
        self.assertEqual(actual[0][1], str(E.WORKER))
        write(self.root / 'cube_behind/selection-frozen/schedule.json', {})
        with patch.object(E, 'original_worker_commands', return_value=[legacy.copy()]):
            with self.assertRaisesRegex(ValueError, 'existing temporal'):
                E.worker_commands('cube_behind', 'selection')

    def test_failed_native_gate_blocks_new_schedules(self):
        gate = self.root / 'gate.json'
        write(gate, {'status': 'failed', 'new_policy_episodes': 0, 'kernel_batch': 16, 'artifacts': {'a': 'b'}})
        with patch.object(E, 'GATE', gate):
            with self.assertRaisesRegex(ValueError, 'has not passed'):
                E.runtime_artifacts()

    def test_nominal_pass_without_legacy_equivalence_is_rejected(self):
        gate = self.root / 'gate.json'
        write(gate, {'status': 'pass', 'new_policy_episodes': 0, 'kernel_batch': 16,
                     'artifacts': {'a': 'b'}, 'legacy_exact_parity': False})
        with patch.object(E, 'GATE', gate):
            with self.assertRaisesRegex(ValueError, 'every legacy candidate'):
                E.runtime_artifacts()


if __name__ == '__main__':
    unittest.main()
