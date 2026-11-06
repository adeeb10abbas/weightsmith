"""Synthetic CPU fixtures only; no provider, network, simulator or GPU calls."""
import copy
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools'))
import enrollment as E
import provider_driver as P
from broker_fixed16 import ReadinessBroker
from verify_package import verify_archive, checked_name


def write_ref(path,value):
    path.write_text(json.dumps(value,sort_keys=True)+'\n')
    return {'path':str(path),'sha256':E.sha(path)}


def fixture(root):
    agent=root/'agent';agent.mkdir()
    for name in ('BRIEF.md','MODE.md'):
        (agent/name).write_bytes((ROOT/'bundle/weightsmith/operator/rq123-v2/design'/name).read_bytes())
    record=json.loads((ROOT/'examples/enrollment.json').read_text())
    record.update(enrollment_id='synthetic-cpu-fixture-only',model_id='fixture-model-1',effort='high',
                  provider='custom-cli',frozen_at_utc='2020-01-01T00:00:00+00:00')
    config={'provider':'custom-cli','model':'fixture-model-1','effort':'high','mode':'adaptive',
        'host':'fixture-host','agent_dir':str(agent),'state_dir':str(root/'operator'),
        'remote_repo':'/owned/fixture/extensions/batch_capture_v2/runtime',
        'remote_root':'/owned/fixture/projects/rq123-v2-adaptive-r1',
        'enrollment_ref':write_ref(root/'enrollment.json',record)}
    binding={'capture_protocol_id':'wsp-batch-capture-v2-fixed16','runtime_variant':'fixed16_new_runtime',
        'source_manifest_sha256':E.SOURCE_SHA,'checkpoint_manifest_sha256':E.CHECKPOINT_SHA,'protocol_sha256':E.PROTOCOL_SHA}
    config['native_binding_ref']=write_ref(root/'native.json',binding)
    validation={'status':'passed','runtime_binding_sha256':config['native_binding_ref']['sha256'],
        'real_batch_counts':[1,2,4,8,16],'padded_batch_count':16,
        'capture_validation':{'protocol_id':'wsp-batch-capture-v2-fixed16','status':'passed',
            'all_real_positions_checked':True,'zero_skipped':True,'parity_tolerance':.02,
            'real_positions_checked':31,'unique_packets_checked':5},
        'source_checkpoint_verification':{'source_manifest_sha256':E.SOURCE_SHA,'verified_source_files':77,'verified_checkpoint_files':26}}
    config['capture_validation_ref']=write_ref(root/'validation.json',validation)
    rows=[{'slot':f's{i}','cand':'base','fingerprint':'base','success':False} for i in E.SLOTS]
    index=root/'base-index.jsonl';index.write_text(''.join(json.dumps(r)+'\n' for r in rows));config['base_records']=str(index)
    manifest={'protocol':'wsp-manipulation-batch-aware-v2','slots':E.SLOTS,'threshold':.5,
        'files':{'v2_runtime_binding.json':config['native_binding_ref']['sha256'],'data/episodes/index.jsonl':E.sha(index)}}
    config['base_manifest_ref']=write_ref(root/'base-manifest.json',manifest)
    return config


class PackageTests(unittest.TestCase):
    def test_all_77_sealed_sources_are_identical(self):
        self.assertEqual(E.sha(ROOT/'reference/source_manifest.json'),E.SOURCE_SHA)
        manifest=json.loads((ROOT/'reference/source_manifest.json').read_text())
        self.assertEqual(len(manifest['files']),77)
        for name,sha in manifest['files'].items():
            self.assertEqual(E.sha(ROOT/'bundle/weightsmith'/name),sha,name)

    def test_enrollment_accepts_complete_synthetic_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=fixture(Path(tmp));out=E.check_enrollment(config)
            self.assertTrue(out['runtime_evidence_checked'])
            self.assertEqual(out['base_index_sha256'],E.sha(config['base_records']))

    def test_probe_does_not_require_or_allow_gpu_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=fixture(Path(tmp));del config['capture_validation_ref']
            self.assertFalse(E.check_enrollment(config,require_runtime=False)['runtime_evidence_checked'])
            with self.assertRaises(KeyError):E.check_enrollment(config)
            broker=object.__new__(ReadinessBroker);broker.cfg={'readiness_only':True}
            with self.assertRaisesRegex(ValueError,'Readiness'):broker.call('wsp_status',{})

    def test_fresh_workspace_excludes_other_context_but_resume_keeps_own_notes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=fixture(root)
            (root/'agent/NOTEBOOK.md').write_text('synthetic own continuation note')
            for runtime in (False,True):
                with self.assertRaisesRegex(ValueError,'only BRIEF'):
                    E.check_enrollment(config,require_runtime=runtime)
            self.assertTrue(E.check_enrollment(config,fresh_workspace=False)['runtime_evidence_checked'])

    def test_failed_or_wrong_runtime_validation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=fixture(root)
            original=json.loads((root/'validation.json').read_text())
            for variant in ('failed','binding','skip','checkpoint'):
                record=copy.deepcopy(original)
                if variant=='failed':record['status']='failed'
                if variant=='binding':record['runtime_binding_sha256']='f'*64
                if variant=='skip':record['capture_validation']['zero_skipped']=False
                if variant=='checkpoint':record['source_checkpoint_verification']['verified_checkpoint_files']=25
                config['capture_validation_ref']=write_ref(root/'validation.json',record)
                with self.assertRaises(ValueError):E.check_enrollment(config)

    def test_old_base_slots_and_modified_brief_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);config=fixture(root)
            index=Path(config['base_records']);rows=[json.loads(line) for line in index.read_text().splitlines()]
            for r,i in zip(rows,range(20032,20064)):r['slot']=f's{i}'
            index.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            manifest=json.loads((root/'base-manifest.json').read_text());manifest['files']['data/episodes/index.jsonl']=E.sha(index)
            config['base_manifest_ref']=write_ref(root/'base-manifest.json',manifest)
            with self.assertRaisesRegex(ValueError,'slots'):E.check_enrollment(config)
            (root/'agent/BRIEF.md').write_text('modified')
            with self.assertRaisesRegex(ValueError,'brief'):E.check_enrollment(config,require_runtime=False)

    def test_wrong_project_or_another_model_enrollment_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config=fixture(Path(tmp));wrong=copy.deepcopy(config);wrong['model']='different-model'
            with self.assertRaises(ValueError):E.check_enrollment(wrong)
            config['remote_root']='/owned/fixture/projects/another-run'
            with self.assertRaises(ValueError):E.check_enrollment(config)

    def test_adapter_pins_model_tools_and_rejects_fallback(self):
        for model in ('opus','latest','<MODEL>',' padded-model '):
            with self.assertRaises(ValueError):P.exact_model(model)
        command=P.claude_command({'model':'fixture-model-1','effort':'high'},'/operator/mcp.json')
        self.assertEqual(command[command.index('--tools')+1],'')
        for flag in ('--bare','--restricted','--strict-mcp-config'):self.assertIn(flag,command)
        event={'type':'system','subtype':'init','model':'fixture-model-1','session_id':'fixture-session','tools':['mcp__wsp__workspace_read']}
        self.assertEqual(P.event_metadata(json.dumps(event),'fixture-model-1'),'fixture-session')
        event['tools'].append('Bash')
        with self.assertRaises(ValueError):P.event_metadata(json.dumps(event),'fixture-model-1')

    def test_archive_checks_every_member_and_rejects_extra_or_traversal(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);archive=root/'fixture.tar.gz';data=b'fixture-only'
            manifest={'files':{'README.md':{'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest()}}}
            for bad in (None,'extra','traversal','symlink','duplicate','corruption'):
                with tarfile.open(archive,'w:gz') as tar:
                    items=[('README.md',data),('MANIFEST.json',json.dumps(manifest).encode())]
                    if bad=='extra':items.append(('extra.md',b'extra'))
                    if bad=='traversal':items.append(('../outside',b'bad'))
                    if bad=='duplicate':items.append(('README.md',data))
                    if bad=='corruption':items[0]=('README.md',b'wrong')
                    for name,blob in items:
                        item=tarfile.TarInfo(name);item.size=len(blob);tar.addfile(item,io.BytesIO(blob))
                    if bad=='symlink':
                        item=tarfile.TarInfo('link');item.type=tarfile.SYMTYPE;item.linkname='README.md';tar.addfile(item)
                if bad is None:self.assertEqual(verify_archive(archive)['verified_archive_members'],2)
                else:
                    with self.assertRaises(ValueError):verify_archive(archive)


if __name__=='__main__':unittest.main()
