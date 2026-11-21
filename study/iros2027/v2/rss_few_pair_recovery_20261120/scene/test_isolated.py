from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parent))
import isolated_preflight as I

class PartitionTests(unittest.TestCase):
    def test_partition_preserves_every_original_pose_exactly(self):
        source=I.REPO/'projects/rss-few-pair-validation-20261120/mustard_right/design/collection-layouts.json'
        original=I.read(source)
        with tempfile.TemporaryDirectory() as tmp:
            record=I.prepare(source,tmp)
            rows=[row for part in record['partitions'] for row in I.read(part['path'])['layouts']]
            self.assertEqual(rows,original['layouts'])
            self.assertEqual([len(I.read(p['path'])['layouts']) for p in record['partitions']],[16,16,16])
            self.assertFalse(record['geometry_changed'])
            self.assertFalse(record['scoring_changed'])
            self.assertFalse(record['tolerances_changed'])

    def test_incomplete_reset_set_cannot_receive_certificate(self):
        source=I.REPO/'projects/rss-few-pair-validation-20261120/mustard_right/design/collection-layouts.json'
        with tempfile.TemporaryDirectory() as tmp:
            I.prepare(source,tmp)
            with self.assertRaises(FileNotFoundError):I.certify(source,tmp)

    def test_partition_mutation_is_detected(self):
        source=I.REPO/'projects/rss-few-pair-validation-20261120/mustard_right/design/collection-layouts.json'
        with tempfile.TemporaryDirectory() as tmp:
            record=I.prepare(source,tmp);p=Path(record['partitions'][0]['path'])
            changed=I.read(p);changed['layouts'][0]['poses']['mustard_bottle']['pos'][0]+=.01
            I.save(p,changed)
            with self.assertRaisesRegex(ValueError,'Partition changed'):I.certify(source,tmp)

if __name__=='__main__':unittest.main()
