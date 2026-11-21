"""Regression coverage for punctuation at the native instruction/state boundary."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parents[1]/'baselines_v2'
REPO=HERE.parents[4]
STUDY=HERE.parent.parent/'rss_few_pair_validation_20261120'
sys.path.insert(0,str(REPO));sys.path.insert(0,str(STUDY))

def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec)
    sys.modules[name]=m;spec.loader.exec_module(m);return m

A=load('_punctuation_artifacts',HERE/'artifacts.py')
N=load('_punctuation_native',HERE/'native.py')
with mock.patch.dict(sys.modules,{'artifacts':A}):P=load('_punctuation_prepare',HERE/'prepare.py')

class TokenizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from weightsmith import config as C
        cls.sp,cls.meta=P.tokenizer_metadata(C.OPENPI_DATA_HOME/'big_vision/paligemma_tokenizer.model')

    def tokens(self,prompt):
        tokens,mask=P.encoded_instruction(self.sp,self.meta,prompt)
        valid=tokens!=0
        return tokens,valid,mask

    def test_all_twelve_registered_prompts_exclude_template_state_and_padding(self):
        from goals.registry import GOAL_IDS,get_goal
        for gid in GOAL_IDS:
            for cell in ('original','teacher','unseen','competing'):
                with self.subTest(goal=gid,cell=cell):
                    tokens,valid,mask=self.tokens(get_goal(gid)[cell])
                    anchor=np.flatnonzero(tokens==3040).item()
                    self.assertEqual(np.flatnonzero(mask).tolist(),list(range(3,anchor-1)))
                    self.assertFalse(mask[:3].any());self.assertFalse(mask[anchor-1:].any())
                    self.assertTrue(valid[mask].all())

    def test_temporal_original_teacher_masks_are_exact_legacy_masks(self):
        old=load('_punctuation_old_artifacts',STUDY/'baselines/artifacts.py')
        from goals.registry import get_goal
        for cell in ('original','teacher','competing'):
            tokens,valid,mask=self.tokens(get_goal('temporal')[cell])
            before=old.instruction_mask(tokens,valid,self.meta['prefix_ids'],self.meta['delimiter_ids'])
            np.testing.assert_array_equal(mask,before)

    def test_period_comma_is_excluded_as_one_whole_token(self):
        a,_,ma=self.tokens('Put the cube in the bowl')
        b,_,mb=self.tokens('Put the cube in the bowl.')
        self.assertIn(1173,b);self.assertNotIn(1173,b[mb])
        np.testing.assert_array_equal(a[ma],b[mb])

    def test_old_code_reproduces_the_period_boundary_failure(self):
        old=load('_punctuation_old_artifacts_2',STUDY/'baselines/artifacts.py')
        tokens,valid,_=self.tokens('Put the cube in the bowl.')
        with self.assertRaises(ValueError):old.instruction_mask(tokens,valid,self.meta['prefix_ids'],self.meta['delimiter_ids'])

    def test_jax_and_host_masks_match_all_registered_prompts(self):
        import jax
        import jax.numpy as jnp
        from goals.registry import GOAL_IDS,get_goal
        cases=[self.tokens(get_goal(gid)[cell]) for gid in GOAL_IDS for cell in ('original','teacher','unseen','competing')]
        tokens=np.stack([x[0] for x in cases]);valid=np.stack([x[1] for x in cases])
        result=jax.jit(lambda t,v:N.instruction_mask_jax(t,v,self.meta['prefix_ids'],self.meta['delimiter_ids'],self.meta['separator_token_ids']))(jnp.array(tokens),jnp.array(valid))
        np.testing.assert_array_equal(np.asarray(result),np.stack([x[2] for x in cases]))

    def test_malformed_boundaries_fail_host_and_disable_native_injection(self):
        import jax.numpy as jnp
        tokens,valid,_=self.tokens('Put the cube in the bowl.')
        anchor=np.flatnonzero(tokens==3040).item()
        cases=[]
        t=tokens.copy();t[anchor-1]=99999;cases.append((t,valid))
        t=tokens.copy();t[anchor]=11111;cases.append((t,valid))
        t=tokens.copy();t[3:5]=[3040,235292];cases.append((t,valid))
        v=valid.copy();v[4]=False;cases.append((tokens,v))
        t=tokens.copy();t[0]=999;cases.append((t,valid))
        for t,v in cases:
            with self.assertRaises(ValueError):A.instruction_mask(t,v,self.meta['prefix_ids'],self.meta['delimiter_ids'],self.meta['separator_token_ids'])
            actual=N.instruction_mask_jax(jnp.asarray(t)[None],jnp.asarray(v)[None],self.meta['prefix_ids'],self.meta['delimiter_ids'],self.meta['separator_token_ids'])
            self.assertFalse(np.asarray(actual).any())

    def test_legacy_load_requires_certificate_and_exact_artifact_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);candidate=root/'legacy/candidate';gate=root/'passed.json'
            A.write_artifact(candidate,'residual_cells',np.ones((2,16,2048),np.float32),{},layer=16)
            spec=A.read(candidate/'baseline.json');spec.update(schema=A.LEGACY_SCHEMA,
                runtime_code_sha256=A.LEGACY_RUNTIME_SHA256,artifact_code_sha256=A.LEGACY_ARTIFACT_SHA256)
            A.save(candidate/'baseline.json',spec)
            with mock.patch.object(A,'LEGACY_ROOT',root/'legacy'),mock.patch.object(A,'COMPATIBILITY_GATE',gate):
                with self.assertRaises(FileNotFoundError):A.load_artifact(candidate)
                cert={'status':'pass','kernel_batch':16,'new_policy_episodes':0,'legacy_exact_parity':True,
                    'artifacts':{str(HERE/name):A.sha(HERE/name) for name in ('artifacts.py','native.py','server.py')},
                    'legacy_artifacts':{str(candidate/name):A.sha(candidate/name) for name in ('baseline.json','direction.npy')}}
                A.save(gate,cert);loaded,_=A.load_artifact(candidate);self.assertEqual(loaded['method'],'residual_cells')
                spec['strength']=2.;A.save(candidate/'baseline.json',spec)
                with self.assertRaises(ValueError):A.load_artifact(candidate)

if __name__=='__main__':unittest.main()
