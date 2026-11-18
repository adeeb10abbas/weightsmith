"""Cost-accounting boundary tests with synthetic receipts only."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import costs


class CostTests(unittest.TestCase):
    def test_unrecorded_queue_and_execution_are_unknown_not_zero(self):
        value=costs.timing({'wall_seconds':20})
        self.assertIsNone(value['queue_seconds']);self.assertIsNone(value['execution_seconds'])
        total=costs.totals([value],'execution_seconds')
        self.assertIsNone(total['measured_sum_seconds']);self.assertEqual(total['unmeasured_components'],1)
        self.assertIsNone(costs.residual(20,None))

    def test_inconsistent_timing_and_negative_durations_are_refused(self):
        with self.assertRaises(ValueError):costs.timing({'wall_seconds':10,'execution_seconds':8,'queue_seconds':3})
        for value in (-1,float('nan'),True):
            with self.assertRaises(ValueError):costs.seconds(value)
        with self.assertRaises(ValueError):costs.residual(4,5)

    def test_parallel_elapsed_is_not_the_sum_or_sum_divided_by_gpus(self):
        value=costs.parallel_interval([(0,10),(5,10)])
        self.assertEqual(value['sum_runner_seconds'],20)
        self.assertEqual(value['elapsed_span_seconds'],15)
        self.assertEqual(value['union_runner_seconds'],15)
        separate=costs.parallel_interval([(0,3),(10,2)])
        self.assertEqual(separate['elapsed_span_seconds'],12)
        self.assertEqual(separate['union_runner_seconds'],5)

    def worker(self,root,duplicate=False):
        root.mkdir();batch=root/'batch-00-original-base';batch.mkdir()
        def save(path,value):path.write_text(json.dumps(value))
        rows=[{'obs':str(root/f'obs-{0 if duplicate else i}.npz'),'ok':True,'split':'development','cohort_id':'synthetic-only'} for i in range(2)]
        save(batch/'episodes.json',rows)
        save(batch/'completed.json',{'batch':0,'cell':'original','condition':'base','n':2,
                                    'episodes_sha256':costs.sha(batch/'episodes.json'),
                                    'wall_seconds':10,'execution_seconds':7,'queue_seconds':2})
        save(root/'plan.json',{'split':'development','case':'temporal','cohort_id':'synthetic-only',
                             'batches':[{'cell':'original','condition':'base','n':2}]})
        save(root/'status.json',{'status':'complete','completed_episodes':2})
        return root

    def test_reused_batch_cannot_inflate_episode_or_wall_counts(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.worker(Path(temporary)/'worker');audit=costs.Audit()
            audit.worker(root,2,'collection','shared','development',{'base'})
            self.assertEqual(sum(row['episodes'] for row in audit.components),2)
            with self.assertRaisesRegex(ValueError,'counted twice'):
                audit.worker(root,2,'selection','lora','development',{'base'})
            self.assertEqual(len(audit.components),1)

    def test_duplicate_observation_and_corrupt_receipt_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=self.worker(Path(temporary)/'duplicate',duplicate=True)
            with self.assertRaisesRegex(ValueError,'counted twice'):
                costs.Audit().worker(root,2,'collection','shared','development',{'base'})
            root=self.worker(Path(temporary)/'corrupt')
            (root/'batch-00-original-base/episodes.json').write_text('[]')
            with self.assertRaisesRegex(ValueError,'hash mismatch'):
                costs.Audit().worker(root,2,'collection','shared','development',{'base'})

    def test_final_cli_refuses_missing_receipts_and_preserves_error_inventory(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'audit'
            command=[sys.executable,str(Path(costs.__file__).resolve()),'--run-root',str(root),
                     '--analysis-report',str(root/'missing-report.json'),'--output',str(out)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,2)
            report=json.loads((out/'costs.json').read_text())
            self.assertFalse(report['receipt_coverage_complete'])
            self.assertTrue(report['errors'])
            self.assertIsNone(report['unmeasured_costs']['manual_wording_and_engineering_seconds'])


if __name__=='__main__':unittest.main()
