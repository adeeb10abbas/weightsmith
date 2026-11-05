import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

FILE = Path(__file__).resolve().parents[1]/'runtime/scripts/verify_runtime.py'
spec = importlib.util.spec_from_file_location('v2_runtime_seal_test', FILE)
seal = importlib.util.module_from_spec(spec);spec.loader.exec_module(seal)


class RuntimeSeal(unittest.TestCase):
    def fixture(self, root):
        def write(name, data):
            p=root/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(data);return p
        server=write('extensions/batch_capture_v2/runtime/weightsmith/batched_server.py','server source\n')
        guard=write('extensions/batch_capture_v2/runtime/scripts/verify_runtime.py','guard source\n')
        config=write('wsp.env','PRIVATE_CONFIGURATION=fixture\n')
        workspace=write('workspaces/base/workspace.json','{"base":"fixture"}\n')
        checkpoint=root/'checkpoint'
        lines=[]
        for i in range(26):
            relative=f'params/{i:02d}'
            p=write(f'checkpoint/{relative}',f'parameter bytes{i}\n')
            lines.append(f'{seal.digest(p)}  {relative}\n')
        inventory=write('third_party/pi05_droid_jointpos.sha256',''.join(lines))
        sources={'path_base':'original_isolated_deployment','files':{
            p.relative_to(root).as_posix():seal.digest(p) for p in (server,guard)}}
        manifest=write('operator/rq123-v2/source_manifest.json',json.dumps(sources))
        binding={'capture_protocol_id':'wsp-batch-capture-v2-fixed16','runtime_variant':'fixed16_new_runtime',
                 'source_manifest_sha256':seal.digest(manifest),'native_server_sha256':seal.digest(server),
                 'environment_file_sha256':seal.digest(config),'base_workspace_path':str(workspace.parent),
                 'editable_export_manifest_sha256':seal.digest(workspace),'checkpoint_path':str(checkpoint),
                 'checkpoint_manifest_sha256':seal.digest(inventory)}
        path=write('projects/rq123-v2-validation-v1/v2_runtime_binding.json',json.dumps(binding))
        return path

    def test_intact_sources_and_all_checkpoint_files_verify(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);binding=self.fixture(root)
            report=seal.verify_binding(binding,isolation=root,verify_checkpoint=True)
            self.assertEqual(report['verified_checkpoint_files'],26)
            self.assertEqual(report['verified_source_files'],2)

    def test_mutated_source_environment_or_export_is_rejected(self):
        for relative in ('extensions/batch_capture_v2/runtime/weightsmith/batched_server.py',
                         'wsp.env','workspaces/base/workspace.json'):
            with self.subTest(path=relative),tempfile.TemporaryDirectory() as t:
                root=Path(t);binding=self.fixture(root);(root/relative).write_text('changed')
                with self.assertRaises(ValueError):seal.verify_binding(binding,isolation=root)

    def test_new_unsealed_runtime_module_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);binding=self.fixture(root)
            (root/'extensions/batch_capture_v2/runtime/weightsmith/new.py').write_text('unexpected source')
            with self.assertRaisesRegex(ValueError,'inventory differs'):
                seal.verify_binding(binding,isolation=root)

    def test_checkpoint_content_corruption_rejected_at_phase_check(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);binding=self.fixture(root);(root/'checkpoint/params/00').write_text('wrong')
            with self.assertRaisesRegex(ValueError,'Checkpoint bytes changed'):
                seal.verify_binding(binding,isolation=root,verify_checkpoint=True)

    def test_source_symlink_rejected_even_with_identical_bytes(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);binding=self.fixture(root)
            path=root/'extensions/batch_capture_v2/runtime/weightsmith/batched_server.py'
            target=root/'same-content';target.write_bytes(path.read_bytes());path.unlink();path.symlink_to(target)
            with self.assertRaisesRegex(ValueError,'Symlink'):
                seal.verify_binding(binding,isolation=root)


if __name__=='__main__':unittest.main()
