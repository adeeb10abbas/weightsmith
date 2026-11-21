"""CPU-only load receipt, rejection, and subprocess CLI contract tests."""
import contextlib
import io
import json
import logging
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest import mock
import numpy as np

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import server as S


class ServerContractTest(unittest.TestCase):
    def blank(self):
        srv=object.__new__(S.Server)
        srv._native_jitted=object();srv.jitted=object();srv.edited={}
        srv._baseline_functions={};srv.active_baseline=None;srv._instruction_tokenizer=None
        return srv

    def test_unknown_workspace_does_not_silently_become_base(self):
        with tempfile.TemporaryDirectory() as temp:
            for path in (Path(temp)/'missing',Path(temp)):
                with self.subTest(path=path):
                    with self.assertRaises(FileNotFoundError):self.blank().load(path)

    def test_unknown_baseline_schema_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            (Path(temp)/'baseline.json').write_text(json.dumps({'schema':'unknown','method':'residual_cells'}))
            with self.assertRaises(ValueError):self.blank().load(temp)

    def test_only_final_baseline_workspace_is_logged(self):
        fake_jnp=types.ModuleType('jax.numpy');fake_jnp.asarray=np.asarray;fake_jnp.float32=np.float32
        fake_jax=types.ModuleType('jax');fake_jax.numpy=fake_jnp
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);(root/'baseline.json').write_text('{}')
            spec={'method':'residual_cells','layer':16,'strength':1.}
            srv=self.blank();srv._baseline_functions[('residual_cells',16)]=lambda *args:None
            buf=io.StringIO();handler=logging.StreamHandler(buf);oldlevel=S.base.log.level
            S.base.log.addHandler(handler);S.base.log.setLevel(logging.INFO)
            try:
                def reset(_self,workspace):
                    self.assertIsNone(workspace)
                    S.base.log.info('loaded %s',json.dumps({'workspace':None}))
                    return {'workspace':None}
                with mock.patch.dict(sys.modules,{'jax':fake_jax,'jax.numpy':fake_jnp}), \
                     mock.patch.object(S.A,'load_artifact',return_value=(spec,np.ones((2,16,2048),np.float32))), \
                     mock.patch.object(S.base.Server,'load',reset):
                    report=srv.load(root)
            finally:
                S.base.log.removeHandler(handler);S.base.log.setLevel(oldlevel)
            lines=buf.getvalue().splitlines()
            self.assertEqual(len(lines),1)
            parsed=json.loads(lines[0].split('loaded ',1)[1])
            self.assertEqual(parsed['workspace'],str(root.resolve()))
            self.assertEqual(parsed['fingerprint'],report['fingerprint'])
            self.assertFalse(parsed['checkpoint_weights_modified'])

    def test_rollout_entrypoint_help_does_not_load_a_model(self):
        result=subprocess.run([sys.executable,str(HERE/'batched_server.py'),'--help'],
                              text=True,capture_output=True,timeout=20)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertIn('--workspace',result.stdout);self.assertIn('--port',result.stdout)


if __name__=='__main__':unittest.main()
