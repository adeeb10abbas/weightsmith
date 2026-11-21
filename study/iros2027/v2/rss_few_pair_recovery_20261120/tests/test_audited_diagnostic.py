import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import subprocess
import sys

HERE = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


W = load('audited_worker_test', HERE / 'diagnostics/worker_audited.py')
R = load('audited_recovery_test', HERE / 'diagnostics/recover_cube_check.py')


class AuditedWorkerTests(unittest.TestCase):
    def test_exact_source_delta(self):
        text = (HERE / 'diagnostics/worker_audited.py').read_text()
        text = text.replace("HERE = Path(__file__).resolve().parents[2] / 'rss_few_pair_validation_20261120'",
                            'HERE = Path(__file__).resolve().parent')
        text = text.replace('sys.path.insert(0, str(HERE))\n', '', 1)
        start, end = text.index('\n\ndef check_identity('), text.index('\n\ndef main():')
        text = text[:start] + text[end:]
        replacements = [
            ('    check_identity({str(path): reference["sha256"]}, Path(plan["audit_root"]) / "freeze-manifest-check.json",\n'
             '                   "freeze_manifest", "Global freeze manifest changed")',
             '    if digest(path) != reference["sha256"]:\n        raise ValueError("Global freeze manifest changed")'),
            ('    if not expected:\n        raise ValueError("A globally frozen source, observation manifest, or candidate changed")\n'
             '    check_identity(expected, Path(plan["audit_root"]) / "global-freeze-check.json",\n'
             '                   "global_freeze", "A globally frozen source, observation manifest, or candidate changed")',
             '    if not expected or any(digest(p) != checksum for p, checksum in expected.items()):\n'
             '        raise ValueError("A globally frozen source, observation manifest, or candidate changed")'),
            ('            check_identity({path: identity[path]}, root / "source-checks" / "initial-source-mismatch.json",\n'
             '                           "initial_source", "Local runtime/candidate differs from the pre-test freeze", observed={path: checksum})',
             '            raise ValueError("Local runtime/candidate differs from the pre-test freeze")'),
            ('            check_identity(identity, root / "source-checks" / f"batch-{number:02d}-before.json",\n'
             '                           "before_batch", "Plan, runtime, or candidate changed after freeze")',
             '            if any(digest(p) != checksum for p, checksum in identity.items()):\n'
             '                raise RuntimeError("Plan, runtime, or candidate changed after freeze")'),
            ('            check_identity(identity, root / "source-checks" / f"batch-{number:02d}-after.json",\n'
             '                           "after_batch", "Source/candidate changed during batch; preserved without result receipt")',
             '            if any(digest(p) != checksum for p, checksum in identity.items()):\n'
             '                raise RuntimeError("Source/candidate changed during batch; preserved without result receipt")')]
        for new, old in replacements:
            self.assertEqual(text.count(new), 1)
            text = text.replace(new, old, 1)
        self.assertEqual(text, (R.STUDY / 'worker.py').read_text())

    def test_standalone_original_study_import_routing(self):
        program = 'import runpy,sys; d=runpy.run_path(sys.argv[1]); import common; assert common.HERE == d["HERE"]'
        subprocess.run([sys.executable, '-I', '-c', program, str(HERE / 'diagnostics/worker_audited.py')],
                       cwd='/tmp', check=True, capture_output=True, text=True)

    def test_complete_mismatch_log_and_stop(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            first, second = root / 'first', root / 'second'
            first.write_text('old'); second.write_text('old')
            pins = {str(p): W.digest(p) for p in (first, second)}
            first.write_text('new'); second.unlink()
            with self.assertRaises(RuntimeError):
                W.check_identity(pins, root / 'audit.json', 'test', 'must stop')
            result = json.loads((root / 'audit.json').read_text())
            self.assertEqual(result['status'], 'mismatch')
            self.assertEqual(len(result['mismatches']), 2)
            self.assertIn('observed_sha256', result['mismatches'][0])
            self.assertIn('ctime_ns', result['mismatches'][0]['before'])
            self.assertIn('error', result['mismatches'][1])

    def test_matching_pins_record_pass(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root / 'source'; path.write_text('unchanged')
            W.check_identity({str(path): W.digest(path)}, root / 'audit.json', 'test', 'stop')
            self.assertEqual(json.loads((root / 'audit.json').read_text())['status'], 'pass')

    def test_initial_observed_mismatch_cannot_be_cleared_by_later_read(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); path = root / 'source'; path.write_text('restored')
            with self.assertRaises(RuntimeError):
                W.check_identity({str(path): W.digest(path)}, root / 'audit.json', 'initial', 'stop',
                                 observed={str(path): 'prior-mismatching-digest'})
            row = json.loads((root / 'audit.json').read_text())['mismatches'][0]
            self.assertEqual(row['observed_sha256'], 'prior-mismatching-digest')

    def test_full_snapshot_preserves_original_and_canonical_status(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); owner = root / 'old'; canonical = root / 'canonical'
            owner.mkdir(); canonical.mkdir()
            (owner / 'raw-rows.json').write_text('[1]')
            (canonical / 'status.json').write_text('{"status":"stopped"}')
            dest = canonical / 'recovery-v3/failure-snapshot'
            try:
                path, receipt = R.snapshot_failure(owner, canonical, dest)
                self.assertEqual(len(receipt['files']), 2)
                R.verify(receipt['artifacts'])
                (canonical / 'status.json').write_text('{"status":"running"}')
                self.assertEqual(json.loads((dest / 'controller/status.json').read_text())['status'], 'stopped')
                self.assertEqual((owner / 'raw-rows.json').read_text(), '[1]')
                self.assertEqual(path.stat().st_mode & 0o777, 0o444)
            finally:
                for item in [dest, *dest.rglob('*')]:
                    item.chmod(0o755 if item.is_dir() else 0o644)


if __name__ == '__main__':
    unittest.main()
