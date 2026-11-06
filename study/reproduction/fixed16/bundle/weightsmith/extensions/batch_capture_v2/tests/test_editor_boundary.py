"""The v2 editor cannot reuse v1 or failed captures as validated evidence."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / 'runtime'
sys.path.insert(0, str(RUNTIME))
sys.path.insert(0, str(RUNTIME / 'drivers'))
from study_validation import normalize_activation_spec, validate_episode_id
from study_tensor_ops import TensorOps


class EditorBoundary(unittest.TestCase):
    def test_disjoint_episode_slots(self):
        for slot in (30016, 30047):
            self.assertEqual(validate_episode_id(f'base.base.s{slot}'), f'base.base.s{slot}')
        for slot in (20032, 20063, 90000, 30015, 30048, 100000):
            with self.assertRaises(ValueError): validate_episode_id(f'base.base.s{slot}')

    def test_edited_capture_requires_own_policy_and_parity(self):
        spec = {'cand':'p1', 'items':[{'episode':'p1.abcdef.s30016','k':0}],
                'action_layers':[0], 'denoise_steps':[0]}
        normalized = normalize_activation_spec(spec, {'base','p1'}, 'adaptive')
        self.assertIs(normalized['require_parity'], True)
        for invalid in ({**spec, 'require_parity':False},
                        {**spec, 'items':[{'episode':'base.base.s30016','k':0}]}):
            with self.assertRaises(ValueError): normalize_activation_spec(invalid, {'base','p1'}, 'adaptive')

    def test_editor_cannot_override_resource_cap(self):
        spec = {'items':[{'episode':'base.base.s30016','k':0}], 'project_capture_bytes':2**40,
                'action_layers':[0], 'denoise_steps':[0]}
        with self.assertRaises(ValueError): normalize_activation_spec(spec, {'base'}, 'adaptive')

    def test_capture_requires_explicit_bounded_sites(self):
        spec = {'items':[{'episode':'base.base.s30016','k':0}]}
        with self.assertRaisesRegex(ValueError, 'requires explicit'):
            normalize_activation_spec(spec, {'base'}, 'adaptive')

    def fixture(self, root, marker=None):
        capture = root/'data/acts/capture'; capture.mkdir(parents=True)
        items = [{'episode':'base.base.s30016','k':0},{'episode':'base.base.s30017','k':0}]
        idx = {'cand':'base','action_layers':[0],'denoise_steps':[0],'items':items,
               'policy_fingerprint':'base'}
        if marker is not None: idx['capture_validation'] = marker
        (capture/'index.json').write_text(json.dumps(idx))
        values = np.stack([np.ones((1,1,1,1024)), np.zeros((1,1,1,1024))])
        np.savez(capture/'acts.npz', **{'action.resid_out':values})
        for item, success in zip(items, (True,False)):
            d=root/'data/episodes'/item['episode'];d.mkdir(parents=True)
            (d/'summary.json').write_text(json.dumps({'success':success}))
        return capture

    def marker(self):
        return {'protocol_id':'wsp-batch-capture-v2-fixed16','status':'passed',
                'all_real_positions_checked':True,'parity_tolerance':0.02,'zero_skipped':True}

    def test_failed_legacy_or_skipped_captures_cannot_make_direction(self):
        markers = (None, {**self.marker(),'status':'failed'},
                   {**self.marker(),'zero_skipped':False},
                   {**self.marker(),'all_real_positions_checked':1})
        for marker in markers:
            with self.subTest(marker=marker), tempfile.TemporaryDirectory() as t:
                root=Path(t);self.fixture(root,marker)
                with self.assertRaises(ValueError):
                    TensorOps(root,root/'unused').contrast({'capture':'capture','key':'action.resid_out','layer':0})
                self.assertFalse((root/'notes/study_directions').exists())

    def test_direction_rechecks_pass_and_exact_source_hash(self):
        for mutation in ('arrays','marker'):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as t:
                root=Path(t);capture=self.fixture(root,self.marker());ops=TensorOps(root,root/'unused')
                result=ops.contrast({'capture':'capture','key':'action.resid_out','layer':0})
                self.assertAlmostEqual(float(np.linalg.norm(ops.direction(result['direction_id'],0))),1.0)
                if mutation=='arrays':
                    with (capture/'acts.npz').open('ab') as f:f.write(b'changed')
                else:
                    p=capture/'index.json';d=json.loads(p.read_text());d['capture_validation']['status']='failed';p.write_text(json.dumps(d))
                with self.assertRaises(ValueError):ops.direction(result['direction_id'],0)


if __name__=='__main__': unittest.main()
