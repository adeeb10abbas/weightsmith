import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('common_reference',HERE/'common_reference.py')
M=importlib.util.module_from_spec(spec);spec.loader.exec_module(M)


class TransferBoundaries(unittest.TestCase):
    def archive(self, root, members):
        p=root/'test.tar.gz'
        with tarfile.open(p,'w:gz') as t:
            for name,raw,kind in members:
                info=tarfile.TarInfo(name);info.size=len(raw);info.type=kind
                if kind==tarfile.SYMTYPE:info.linkname='/tmp/forbidden'
                t.addfile(info,io.BytesIO(raw))
        return p

    def transport(self, files):
        return json.dumps({'reference_manifest_sha256':M.MANIFEST_SHA,'source_project':M.ROOT,'files':files}).encode()

    def test_exact_slots_and_no_foreign_evidence(self):
        for slot in range(30010,30055):
            self.assertEqual(M.allowed(f'data/episodes/base.base.s{slot}/obs.npz'),slot in M.SLOTS)
        for name in ['data/episodes/p1.p1.s30016/obs.npz','data/episodes/base.base.s100000/obs.npz','../config.json','/config.json','data//episodes/index.jsonl','data/episodes/./index.jsonl','traces/index.json']:
            self.assertFalse(M.allowed(name),name)
        self.assertTrue(M.allowed('data/batch_packets/'+'a'*32+'/requests.npz'))

    def test_missing_file_fails(self):
        raw=b'{}';files={'config.json':{'bytes':2,'sha256':hashlib.sha256(raw).hexdigest()}}
        with tempfile.TemporaryDirectory() as d:
            p=self.archive(Path(d),[(M.PREFIX+'/TRANSPORT_MANIFEST.json',self.transport(files),tarfile.REGTYPE)])
            with self.assertRaisesRegex(ValueError,'Incomplete'):M.verify_archive(p)

    def test_corrupt_declared_bytes_fail(self):
        files={'config.json':{'bytes':2,'sha256':hashlib.sha256(b'{}').hexdigest()}}
        with tempfile.TemporaryDirectory() as d:
            p=self.archive(Path(d),[(M.PREFIX+'/TRANSPORT_MANIFEST.json',self.transport(files),tarfile.REGTYPE),(M.PREFIX+'/config.json',b'[]',tarfile.REGTYPE)])
            with self.assertRaisesRegex(ValueError,'checksum'):M.verify_archive(p)

    def test_unsafe_nonregular_and_duplicate_fail(self):
        for name,kind in [(M.PREFIX+'/../escape',tarfile.REGTYPE),(M.PREFIX+'/./config.json',tarfile.REGTYPE),(M.PREFIX+'/config.json',tarfile.SYMTYPE),(M.PREFIX+'/TRANSPORT_MANIFEST.json',tarfile.REGTYPE)]:
            with self.subTest(name=name,kind=kind),tempfile.TemporaryDirectory() as d:
                p=self.archive(Path(d),[(M.PREFIX+'/TRANSPORT_MANIFEST.json',self.transport({}),tarfile.REGTYPE),(name,b'',kind)])
                with self.assertRaises(ValueError):M.verify_archive(p)

    def test_foreign_inventory_fails_before_extract(self):
        with tempfile.TemporaryDirectory() as d:
            p=self.archive(Path(d),[(M.PREFIX+'/TRANSPORT_MANIFEST.json',self.transport({'../escape':{}}),tarfile.REGTYPE)])
            with self.assertRaisesRegex(ValueError,'foreign'):M.verify_archive(p,destination=Path(d)/'staging')
            self.assertFalse((Path(d)/'escape').exists())

    def test_existing_staging_and_unsealed_codec_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError,'Fresh'):M.verify_archive('unused',destination=d)
            p=Path(d)/'codec.py';p.write_text('raise RuntimeError("should not execute")')
            with self.assertRaisesRegex(ValueError,'sealed CPU'):M.verify_packets(d,p)


if __name__=='__main__':unittest.main()
