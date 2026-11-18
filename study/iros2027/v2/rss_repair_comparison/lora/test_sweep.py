"""CPU selection/provenance tests; no fitting or policy episodes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import sweep


def training_fixture(repair=.2,preservation=.1):
    items=[{'role':role,'robot_action_mse':value} for role,value in [('repair',repair),('preservation',preservation)] for _ in range(4)]
    return {'rank':32,'layers':sweep.LAYERS,'steps':200,'learning_rate':3e-5,'weight_decay':0.,'seed':17,
            'dataset_manifest_sha256':'dataset','source_code_sha256':'source','train_inputs':24,'validation_inputs':8,
            'closed_loop_evaluation':'not performed by this training command',
            'workspace_validation':{'action_expert_and_head_frozen':False,
                                    'edited_tensors':{f'language.blocks.{layer:02d}.mlp.linear':{} for layer in sweep.LAYERS}},
            'final_validation_actions':{'items':items,'by_role':{'repair':{'robot_action_mse':repair},
                                                               'preservation':{'robot_action_mse':preservation}}}}


def candidates():
    return [{'candidate_id':name,'learning_rate':rate,'balanced_validation_action_mse':mse,
             'stage':stage,'status':'complete'} for name,rate,mse,stage in
            [('initial-1e-3',.001,.001,'initial'),('lr-3e-5',3e-5,.03,'new'),('lr-1e-4',1e-4,.02,'new')]]


class SweepTests(unittest.TestCase):
    def test_nomination_considers_only_new_rates_and_keeps_initial_for_stage_b(self):
        self.assertEqual(sweep.nominate(candidates()),'lr-1e-4')

    def test_exact_metric_tie_uses_smaller_learning_rate(self):
        rows=candidates();rows[1]['balanced_validation_action_mse']=rows[2]['balanced_validation_action_mse']
        self.assertEqual(sweep.nominate(list(reversed(rows))),'lr-3e-5')

    def test_failed_missing_and_nonfinite_trials_cannot_be_omitted(self):
        for rows in (candidates()[:2],[{**r,'status':'stopped'} if r['stage']=='new' else r for r in candidates()],
                     [{**r,'balanced_validation_action_mse':float('nan')} if r['stage']=='new' else r for r in candidates()]):
            with self.assertRaises(ValueError):sweep.nominate(rows)

    def test_equal_role_weight_and_consistent_item_summaries(self):
        fixture=training_fixture(.4,.1)
        fixture['final_validation_actions']['items']=fixture['final_validation_actions']['items'][:5]
        metric,_,counts=sweep.balanced_action_mse(fixture)
        self.assertEqual(counts,{'repair':4,'preservation':1})
        self.assertAlmostEqual(metric,.25)
        fixture['final_validation_actions']['by_role']['repair']['robot_action_mse']=.9
        with self.assertRaisesRegex(ValueError,'summary differs'):sweep.balanced_action_mse(fixture)

    def test_recipe_and_validation_split_cannot_drift(self):
        fixture=training_fixture();sweep.validate_recipe(fixture,'dataset',3e-5,'source')
        for field,value in [('steps',201),('learning_rate',.001),('source_code_sha256','changed'),('dataset_manifest_sha256','other')]:
            altered=copy.deepcopy(fixture);altered[field]=value
            with self.assertRaisesRegex(ValueError,'recipe'):sweep.validate_recipe(altered,'dataset',3e-5,'source')
        altered=copy.deepcopy(fixture);altered['final_validation_actions']['items'].pop()
        with self.assertRaisesRegex(ValueError,'four validation'):sweep.validate_recipe(altered,'dataset',3e-5,'source')
        altered=copy.deepcopy(fixture);altered['workspace_validation']['edited_tensors']['action.head']={}
        with self.assertRaisesRegex(ValueError,'outside the frozen VLM'):sweep.validate_recipe(altered,'dataset',3e-5,'source')

    def test_source_identity_detects_mutation_and_requires_protocol_and_training_code(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);source=root/'source.py';source.write_text('frozen\n')
            identity={str(source):sweep.sha(source)};sweep.verify_identity(identity)
            freeze=root/'freeze.json';sweep.dump(freeze,{'sources':identity})
            with self.assertRaisesRegex(ValueError,'missing required'):sweep.read_freeze(freeze)
            source.write_text('changed\n')
            with self.assertRaisesRegex(ValueError,'changed'):sweep.verify_identity(identity)

    def test_candidate_seal_covers_all_nine_tensors_and_fit_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            fit=Path(temporary);workspace=fit/'workspace';(workspace/'edits').mkdir(parents=True)
            tensors={}
            for layer in sweep.LAYERS:
                name=f'layer{layer}.npy';np.save(workspace/'edits'/name,np.zeros((2,2),dtype=np.float32))
                tensors[f'language.blocks.{layer:02d}.mlp.linear']={'file':name,
                    'model_path':['PaliGemma','llm','layers','mlp','linear'],'scan_index':layer}
            sweep.dump(workspace/'workspace.json',{'tensors':tensors})
            for name in ('training.json','settings.json'):sweep.dump(fit/name,{})
            np.savez(fit/'adapters.npz',a=np.zeros(2))
            seal=sweep.seal_candidate(fit,'lr-3e-5','temporal',fit/'seal.json','identity','dataset')
            self.assertEqual(len(seal['workspace_artifacts']),10)
            self.assertEqual(len(seal['artifacts']),13)
            sweep.verify_identity(seal['artifacts'])
            np.save(workspace/'edits'/'layer8.npy',np.ones((2,2)))
            with self.assertRaisesRegex(ValueError,'changed'):sweep.verify_identity(seal['artifacts'])


if __name__=='__main__':unittest.main()
