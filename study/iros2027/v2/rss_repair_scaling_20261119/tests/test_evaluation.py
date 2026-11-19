"""Guard the inference requests, complete-grid selection, and paired estimands."""
import copy
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import evaluate as E


class EvaluationContracts(unittest.TestCase):
    def plan(self, cells):
        return {'cohort_id': E.protocol()['cohort_id'], 'case': 'temporal', 'split': 'heldout',
                'inference_batching': 'fixed16', 'protocol_sha256': E.sha(E.HERE/'protocol.json'),
                'batches': cells}

    def test_requests_and_opposing_goal_are_distinct(self):
        cells = E.test_cells('weightsmith_n1', '/tmp/candidate')
        E.validate_plan(self.plan(cells))
        self.assertEqual([r['slot_start'] for r in cells], [240000,240016,240032,240048])
        self.assertEqual(cells[2]['goal'], 'cube_then_banana')
        self.assertEqual(cells[3]['goal'], 'banana_then_cube')
        bad = copy.deepcopy(cells); bad[3]['goal'] = 'cube_then_banana'
        with self.assertRaises(ValueError): E.validate_plan(self.plan(bad))
        bad = copy.deepcopy(cells); bad[2]['effective_prompt'] = E.protocol()['clarified_prompt']
        with self.assertRaises(ValueError): E.validate_plan(self.plan(bad))

    def test_clarification_has_only_original_task(self):
        cells = E.test_cells('clarified', None)
        self.assertEqual(len(cells), 2)
        E.validate_plan(self.plan(cells))
        self.assertTrue(all(c['effective_prompt'] == E.protocol()['clarified_prompt'] for c in cells))

    def test_selection_tie_and_incomplete_grid(self):
        candidates = [{'candidate_id': 'a', 'budget': 1, 'learning_rate': 1e-4},
                      {'candidate_id': 'b', 'budget': 1, 'learning_rate': 1e-3}]
        rows = [{'slot': i, 'success': i<4} for i in range(16)]
        self.assertEqual(E.select_candidates(candidates, {'a': rows, 'b': rows})[0]['candidate_id'], 'a')
        with self.assertRaises(ValueError): E.select_candidates(candidates, {'a': rows, 'b': rows[:-1]})

    def test_reject_partial_available_budget_registry(self):
        record = {'source_manifest': '/source.json', 'source_manifest_sha256': 'digest',
                  'completed_budgets': [1], 'candidates': [
                      {'candidate_id': 'a', 'budget': 1, 'learning_rate': 1e-4},
                      {'candidate_id': 'b', 'budget': 1, 'learning_rate': 1e-3}]}
        source = {'cohort_id': E.protocol()['cohort_id'], 'split': 'development',
                  'correction_budgets_available': [1,3,6,12]}
        with patch.object(E, 'read', side_effect=[record,source]), patch.object(E, 'sha', return_value='digest'), patch.object(E, 'candidate_identity', side_effect=lambda c:c):
            with self.assertRaisesRegex(ValueError, 'Every available'): E.registry('/registry.json','lora')

    def test_pairing_rejects_changed_noise(self):
        a=[{'slot':i,'layout_id':f'l{i}','policy_seed':i,'success':True,'first_camera_sha256':{'a':'x'}} for i in range(16)]
        b=copy.deepcopy(a)
        b[0]['policy_seed'] += 1
        with self.assertRaisesRegex(ValueError, 'identities differ'): analyze.contrast(a,b,'edit','base','original')
        b=copy.deepcopy(a)
        for r in b:r['success']=False
        result=analyze.contrast(a,b,'edit','base','original')
        self.assertEqual((result['fixed'],result['broken'],result['difference']), (16,0,1.0))


if __name__ == '__main__': unittest.main()
