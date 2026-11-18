"""Synthetic analysis fixtures only; no learned-policy outcomes."""
import copy
import json
from pathlib import Path
import tempfile
import subprocess
import sys
import unittest

import report


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.protocol = json.loads((Path(__file__).resolve().parents[1]/'protocol.json').read_text())
        self.protocol_path = self.root/'protocol.json'
        report.write(self.protocol_path, self.protocol)
        self.roots = {}
        for case, definition in self.protocol['cases'].items():
            root = self.root/case; root.mkdir(); self.roots[case] = root
            metrics = root/'metrics.json'; report.write(metrics, {'synthetic': True})
            identity = {str(self.protocol_path): report.sha(self.protocol_path)}
            for condition in ('closed_form', 'lora'):
                workspace = root/condition
                identity[str(workspace/'workspace.json')] = 'synthetic-only'
                identity[str(workspace/'edits'/'weight.npy')] = 'synthetic-only'
            batches = []
            goals = sorted({cell['goal'] for cell in definition['cells']})
            for cell in definition['cells']:
                for condition in self.protocol['conditions']:
                    effective = cell['rewrite']['effective_prompt'] if condition == 'saved_rewrite' else cell['prompt']
                    spec = {'cell':cell['id'], 'condition':condition, 'effective_prompt':effective,
                            'slot_start':self.protocol['heldout']['slot_start'][case], 'n':16,
                            'workspace':str(root/condition) if condition in ('closed_form','lora') else None}
                    number = len(batches); batches.append(spec)
                    batch = root/f'batch-{number:02d}-{cell["id"]}-{condition}'; batch.mkdir()
                    rows = []
                    for i in range(16):
                        success = i % 2 == 0
                        if condition == 'closed_form': success = i % 4 != 3
                        if condition == 'lora': success = i % 4 in (0,1)
                        if condition == 'saved_rewrite' and cell['rewrite']['rewritten']: success = i % 4 != 3
                        selected = cell['goal']; competing = next(goal for goal in goals if goal != selected)
                        final = {goal:False for goal in goals}; final[selected] = success
                        if not success: final[competing] = True
                        ever = dict(final); ever[competing] = True
                        rows.append({'case':case,'cell':cell['id'],'condition':condition,
                            'cohort_id':self.protocol['cohort'],'split':'heldout','slot':spec['slot_start']+i,
                            'layout_id':f'{case}-H{i//2+1:02d}','policy_seed':1000+i,
                            'input_prompt':cell['prompt'],'effective_prompt':effective,
                            'selected_goal':selected,'success':success,'strict_final':final,'strict_ever':ever,
                            'stock_final':final,'stock_ever':ever,'ok':True,
                            'first_camera_sha256':{'base_img':f'base-{i}','wrist_img':f'wrist-{i}'},
                            'metrics':str(metrics),'metrics_sha256':report.sha(metrics),
                            'obs':f'/synthetic-only/{case}/{cell["id"]}/{condition}/{i}.npz',
                            'obs_sha256':'synthetic-only'})
                    report.write(batch/'episodes.json', rows)
                    report.write(batch/'completed.json', {'batch':number,'cell':cell['id'],'condition':condition,
                        'n':16,'successes':sum(row['success'] for row in rows),
                        'episodes_sha256':report.sha(batch/'episodes.json')})
            report.write(root/'plan.json', {'case':case,'split':'heldout','cohort_id':self.protocol['cohort'],'batches':batches})
            report.write(root/'status.json', {'status':'complete','completed_episodes':320})
            report.write(root/'identity.json', identity)

    def collect(self):
        return report.collect(self.protocol_path,self.roots)

    def mutate(self, filename, function, reseal=True):
        path = self.roots['temporal']/filename
        rows = report.read(path); function(rows); report.write(path,rows)
        if reseal:
            receipt = report.read(path.parent/'completed.json')
            receipt['episodes_sha256'] = report.sha(path)
            receipt['successes'] = sum(row['success'] for row in rows)
            report.write(path.parent/'completed.json', receipt)

    def test_complete_paired_counts_and_exact_cache_reuses_evidence(self):
        protocol, rows, status = self.collect()
        self.assertTrue(status['complete_and_verified'], status['errors'])
        result = report.summarize(protocol,rows,status,draws=1000)
        self.assertEqual(result['evidence_counts'], {'new_policy_episodes':640,'derived_exact_cache_records':160,
                                                   'derived_exact_cache_new_independent_episodes':0})
        pair = next(p for p in result['paired'] if p['case']=='temporal' and p['cell']=='original'
                    and p['condition']=='closed_form' and p['reference']=='base')
        self.assertEqual((pair['fixed'],pair['broken'],pair['n']), (4,0,16))
        self.assertEqual(pair['success_rate_difference'],.25)
        exact = {r['cell']:r for r in result['exact_cache_derived'] if r['case']=='temporal'}
        self.assertEqual(exact['original']['source_condition'],'saved_rewrite')
        self.assertEqual(exact['unseen_before']['source_condition'],'base')

    def test_missing_receipt_status_only_and_full_analysis_refused(self):
        (self.roots['temporal']/'batch-00-original-base'/'completed.json').unlink()
        protocol,rows,status = self.collect()
        self.assertEqual(status['mode'],'status_only')
        self.assertNotIn('primary',status)
        self.assertEqual(status['verified_new_policy_episodes'],624)
        with self.assertRaises(report.VerificationError): report.summarize(protocol,rows,status)
        report.write_csv(self.root/'inventory.csv',status['inventory'])

    def test_receipt_hash_corruption_is_not_accepted(self):
        self.mutate('batch-00-original-base/episodes.json',lambda rows:rows[0].update(policy_seed=9),reseal=False)
        _,_,status = self.collect()
        self.assertTrue(any('checksum' in error for error in status['errors']))

    def test_duplicate_slot_is_refused_even_with_updated_receipt(self):
        self.mutate('batch-00-original-base/episodes.json',lambda rows:rows[1].update(slot=rows[0]['slot']))
        _,_,status = self.collect()
        self.assertTrue(any('duplicate' in error for error in status['errors']))

    def test_pair_seed_mismatch_is_fatal_but_camera_difference_is_not(self):
        self.mutate('batch-03-original-lora/episodes.json',lambda rows:rows[0].update(policy_seed=99))
        _,_,status = self.collect()
        self.assertTrue(any('Pair layout/seed mismatch' in error for error in status['errors']))
        self.mutate('batch-03-original-lora/episodes.json',lambda rows:rows[0].update(policy_seed=1000,
                    first_camera_sha256={'base_img':'different','wrist_img':'wrist-0'}))
        protocol,rows,status = self.collect()
        result = report.summarize(protocol,rows,status,draws=1000)
        self.assertGreater(result['pairing_diagnostics']['initial_camera_mismatches'],0)
        self.assertEqual(result['pairing_diagnostics']['automatically_excluded_episodes'],0)
        self.assertEqual(result['evidence_counts']['new_policy_episodes'],640)

    def test_metric_hash_and_unfrozen_adapter_are_refused(self):
        path = self.roots['temporal']/'metrics.json'; path.write_text('{}')
        _,_,status = self.collect()
        self.assertTrue(any('Metrics checksum' in error for error in status['errors']))
        path = self.roots['spatial']/'identity.json'; identity = report.read(path)
        identity = {key:value for key,value in identity.items() if 'edits' not in key}
        report.write(path,identity)
        _,_,status = self.collect()
        self.assertTrue(any('were not frozen' in error for error in status['errors']))

    def test_degenerate_cluster_intervals_warn_and_ever_is_separate(self):
        interval = report.clustered_interval([0]*16,[f'L{i//2}' for i in range(16)],draws=1000)
        self.assertTrue(interval['degenerate_ci'])
        self.assertIn('does not establish certainty',interval['ci_warning'])
        protocol,rows,status = self.collect()
        result = report.summarize(protocol,rows,status,draws=1000)
        self.assertTrue(any(r['selected_and_competing_ever']>0 and r['selected_and_competing_final']==0
                            for r in result['collisions']))
        self.assertTrue(all('not a simultaneous' in r['ever_interpretation'] for r in result['collisions']))

    def test_changed_protocol_is_refused(self):
        altered = copy.deepcopy(self.protocol); altered['status']='changed after run'
        report.write(self.protocol_path,altered)
        _,_,status = self.collect()
        self.assertTrue(any('exact protocol' in error for error in status['errors']))

    def test_cli_writes_full_tables_or_status_only_as_requested(self):
        command = [sys.executable,str(Path(report.__file__).resolve()),'--protocol',str(self.protocol_path),
                   '--temporal-root',str(self.roots['temporal']),'--spatial-root',str(self.roots['spatial']),
                   '--bootstrap-draws','1000']
        full = self.root/'full-report'
        subprocess.run([*command,'--output',str(full)],check=True,capture_output=True,text=True)
        self.assertEqual(report.read(full/'report.json')['evidence_counts']['new_policy_episodes'],640)
        self.assertTrue((full/'exact_cache_derived.csv').exists())
        partial = self.root/'status-only'
        subprocess.run([*command,'--output',str(partial),'--allow-partial'],check=True,capture_output=True,text=True)
        self.assertEqual({p.name for p in partial.iterdir()},{'status.json','inventory.csv'})


if __name__=='__main__':
    unittest.main()
