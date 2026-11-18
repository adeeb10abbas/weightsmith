"""CPU contract tests; these are not robot-policy results."""
import json
import hashlib
import tempfile
from pathlib import Path
import unittest

import numpy as np

import train_lora as baseline


class DataBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)

    def spec(self):
        return {'cohort_id': 'test-only', 'split': 'development',
                'original_prompt': 'original', 'teacher_prompt': 'teacher',
                'items': [{'obs_path': str(self.root/'train'/'obs.npz'), 'decision_index': 0, 'split': 'train'},
                          {'obs_path': str(self.root/'validation'/'obs.npz'), 'decision_index': 0, 'split': 'validation'}]}

    def read(self, spec):
        baseline.dump(self.root/'spec.json', spec)
        return baseline.read_spec(self.root/'spec.json')

    def live_item(self,item):
        item=dict(item)
        item['obs_path']=str(Path(item['obs_path']).with_name('obs_r0_e0.npz'))
        member={'env_id':0,'obs_path':item['obs_path'],'decision_index':item['decision_index'],'obs_sha256':'0'*64}
        context={'source_episodes_path':str(self.root/'episodes.json'),'source_episodes_sha256':'0'*64,
                 'source_receipt_path':str(self.root/'completed.json'),'source_receipt_sha256':'0'*64,
                 'source_run_id':0,'source_step':0,'source_batch_observation_sha256':{item['obs_path']:'0'*64},
                 'members':[member]}
        item.update(live_batch_size=1,live_batch_bucket=1,live_batch_members=[member],source_env_id=0,
                    source_run_id=0,source_step=0,live_context=context,
                    live_context_sha256=hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest())
        return item

    def test_rejects_non_development_and_heldout_paths(self):
        spec = self.spec(); spec['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'development'):
            self.read(spec)
        spec = self.spec(); spec['items'][0]['obs_path'] = str(self.root/'heldout_v2'/'obs.npz')
        with self.assertRaisesRegex(ValueError, 'Held-out'):
            self.read(spec)

    def test_validation_cannot_share_episode_even_at_other_decision(self):
        spec = self.spec()
        spec['items'][1].update(obs_path=spec['items'][0]['obs_path'], decision_index=1)
        with self.assertRaisesRegex(ValueError, 'distinct episodes'):
            self.read(spec)

    def test_different_prompts_on_same_state_are_allowed_but_duplicates_are_not(self):
        spec = self.spec()
        spec['items'].append({**spec['items'][0], 'original_prompt': 'other goal'})
        self.read(spec)
        spec['items'].append(dict(spec['items'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            self.read(spec)

    def test_preservation_requires_identity_teacher(self):
        spec = self.spec(); spec['items'][0]['role'] = 'preservation'
        with self.assertRaisesRegex(ValueError, 'Preservation'):
            self.read(spec)
        spec['items'][0]['teacher_prompt'] = 'original'
        self.read(spec)

    def test_flatten_roundtrip_preserves_image_state_and_masks(self):
        source = {'image': {'base': np.arange(18, dtype=np.uint8).reshape(2, 3, 3)},
                  'image_mask': {'base': np.asarray(True)}, 'state': np.zeros(32, dtype=np.float32)}
        result = baseline.flatten(baseline.unflatten(baseline.flatten(source)))
        for key, value in baseline.flatten(source).items():
            np.testing.assert_array_equal(value, result[key])

    def test_prepared_dataset_rejects_corrupt_targets(self):
        spec = self.spec();spec['items']=[self.live_item(item) for item in spec['items']]
        baseline.dump(self.root/'input-spec.json', spec)
        records = []
        for i, item in enumerate(spec['items']):
            p = self.root/f'item-{i:04d}.npz'
            np.savez(p, **{'input/state': np.zeros(32)}, teacher_actions=np.zeros((15, 32)),
                     teacher_actions_physical=np.zeros((15, 8)),live_replay_actions_physical=np.zeros((15,8)),
                     recorded_actions_physical=np.zeros((15,8)))
            records.append({**item, 'file': p.name, 'sha256': baseline.sha(p),
                            'live_replay_max_abs_error':0.,'live_replay_bit_exact':True,'recorded_input_sha256':'0'*64})
        baseline.dump(self.root/'dataset.json', {'split': 'development', 'records': records,
                      'live_replay_gate':{'mode':'exact_pool_bucket','tolerance':1e-6,'status':'pass','maximum_error':0.,'items':2},
                      'stored_spec_sha256': baseline.sha(self.root/'input-spec.json')})
        _, examples = baseline.load_dataset(self.root)
        self.assertEqual(len(examples['train']), 1)
        with (self.root/records[0]['file']).open('ab') as stream:
            stream.write(b'corrupt')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            baseline.load_dataset(self.root)

    def test_missing_or_falsified_live_certificate_is_refused(self):
        baseline.dump(self.root/'dataset.json',{'split':'development','records':[]})
        with self.assertRaisesRegex(ValueError,'live replay certificate'):
            baseline.load_dataset(self.root)
        record={'live_replay_max_abs_error':.001,'live_replay_bit_exact':False,'recorded_input_sha256':'0'*64}
        meta={'records':[record],'live_replay_gate':{'mode':'exact_pool_bucket','status':'pass',
              'tolerance':1e-6,'maximum_error':.001,'items':1}}
        with self.assertRaisesRegex(ValueError,'outside tolerance'):
            baseline.validate_live_certificate(meta)

    def test_live_metadata_rejects_wrong_bucket_source_and_checksum(self):
        item=self.live_item(self.spec()['items'][0]);baseline.validate_live_metadata(item)
        wrong={**item,'live_batch_bucket':2}
        with self.assertRaisesRegex(ValueError,'bucket'):
            baseline.validate_live_metadata(wrong)
        with self.assertRaisesRegex(ValueError,'Source decision'):
            baseline.validate_live_metadata({**item,'source_env_id':1})
        with self.assertRaisesRegex(ValueError,'checksum'):
            baseline.validate_live_metadata({**item,'live_context_sha256':'f'*64})

    def test_live_membership_is_reconstructed_from_all_recorded_steps(self):
        batch=self.root/'batch';observations=batch/'obs-l0a0';observations.mkdir(parents=True)
        rows=[];members=[]
        for env in range(16):
            path=observations/f'obs_r0_e{env}.npz'
            np.savez(path,step=np.asarray([0,15] if env<15 else [0]))
            digest=baseline.sha(path)
            rows.append({'obs':str(path),'obs_sha256':digest,'env':env,'run':0,'case':'spatial',
                         'split':'development','condition':'base','ok':True,'effective_prompt':'teacher','cohort_id':'source-cohort'})
            if env<15:members.append({'env_id':env,'obs_path':str(path),'decision_index':1,'obs_sha256':digest})
        baseline.dump(batch/'episodes.json',rows)
        baseline.dump(batch/'completed.json',{'episodes_sha256':baseline.sha(batch/'episodes.json'),'n':16})
        context={'source_episodes_path':str(batch/'episodes.json'),'source_episodes_sha256':baseline.sha(batch/'episodes.json'),
                 'source_receipt_path':str(batch/'completed.json'),'source_receipt_sha256':baseline.sha(batch/'completed.json'),
                 'source_run_id':0,'source_step':15,'source_batch_observation_sha256':{r['obs']:r['obs_sha256'] for r in rows},
                 'members':members}
        item={'obs_path':rows[0]['obs'],'decision_index':1,'recorded_prompt':'teacher','live_batch_size':15,
              'live_batch_bucket':16,'live_batch_members':members,'source_env_id':0,'source_run_id':0,'source_step':15,
              'live_context':context,'live_context_sha256':hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        got=baseline.validate_live_context(item,{'case':'spatial'},{})
        self.assertEqual(got['source_cohort_id'],'source-cohort')
        wrong_context={**context,'members':members[:-1]}
        wrong={**item,'live_batch_size':14,'live_batch_members':members[:-1],'live_context':wrong_context,
               'live_context_sha256':hashlib.sha256(json.dumps(wrong_context,sort_keys=True,separators=(',',':')).encode()).hexdigest()}
        with self.assertRaisesRegex(ValueError,'membership does not match'):
            baseline.validate_live_context(wrong,{'case':'spatial'},{})

    def test_recorded_float32_storage_is_accounted_for_without_hiding_drift(self):
        recorded=np.ones((15,8),dtype=np.float32)
        exact=baseline.physical_errors(recorded.astype(np.float64)+1e-8,recorded)
        self.assertTrue(exact['bit_exact']);self.assertEqual(exact['max_abs_error'],0.)
        drift=baseline.physical_errors(recorded.astype(np.float64)+.001,recorded)
        self.assertGreater(drift['max_abs_error'],1e-6)


class AdapterTests(unittest.TestCase):
    def test_zero_start_immutable_base_frozen_slices_and_real_gradient(self):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        original = jnp.arange(24, dtype=jnp.float32).reshape(3, 4, 2).astype(jnp.bfloat16)
        state = nnx.State({'PaliGemma': {'llm': {'layers': {'mlp': {'linear': nnx.Param(original).to_state(),
                                                                 'linear_1': nnx.Param(jnp.ones((3, 2, 2))).to_state()}}}}})
        a = {'a': jnp.ones((1, 4, 1)), 'b': jnp.zeros((1, 1, 2))}
        def get(s):
            return s['PaliGemma']['llm']['layers']['mlp']['linear'].value
        zero = baseline.injected_state(state, a, (1,), jax, jnp)
        np.testing.assert_array_equal(get(zero), original)
        changed = baseline.injected_state(state, {**a, 'b': jnp.ones((1, 1, 2))}, (1,), jax, jnp)
        np.testing.assert_array_equal(get(state), original)
        np.testing.assert_array_equal(get(changed)[jnp.asarray([0, 2])], original[jnp.asarray([0, 2])])
        np.testing.assert_array_equal(changed['PaliGemma']['llm']['layers']['mlp']['linear_1'].value,
                                      state['PaliGemma']['llm']['layers']['mlp']['linear_1'].value)
        self.assertTrue(np.any(np.asarray(get(changed)[1]) != np.asarray(original[1])))
        gradient = jax.grad(lambda factors: jnp.sum(get(baseline.injected_state(state, factors, (1,), jax, jnp)).astype(jnp.float32)**2))(a)
        self.assertGreater(float(jnp.linalg.norm(gradient['b'])), 0)

    def test_export_exact_evaluated_weights_and_checks_base_identity(self):
        from weightsmith.distill import bf16_round
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); source = root/'source'; (source/'base').mkdir(parents=True)
            original = np.arange(24, dtype=np.float32).reshape(3, 4, 2)
            edited = original.copy(); edited[1, 0, 0] += .25
            edited = bf16_round(edited)
            entry = {'file': 'layer1.npy', 'numel': 8}
            np.save(source/'base'/'layer1.npy', original[1])
            baseline.dump(source/'workspace.json', {'tensors': {'language.blocks.01.mlp.linear': entry},
                          'total_parameters_and_buffers': 24, 'base_vlm_sha256': 'test-only'})
            destination = root/'good'; destination.mkdir()
            result = baseline.export_workspace(destination, (1,), source, original, edited)
            self.assertEqual(result['changed_elements'], 1)
            np.testing.assert_array_equal(np.load(destination/'workspace'/'edits'/'layer1.npy'), edited[1])
            bad = root/'bad'; bad.mkdir()
            with self.assertRaisesRegex(ValueError, 'base differ'):
                baseline.export_workspace(bad, (1,), source, original+1, edited)


if __name__ == '__main__':
    unittest.main()
