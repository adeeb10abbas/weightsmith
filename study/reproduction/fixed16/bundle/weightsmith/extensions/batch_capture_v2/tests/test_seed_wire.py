"""CPU-only live-client seed producer and optional actual msgpack codec checks."""
import ast
from pathlib import Path
import types
import unittest
import numpy as np
RUNTIME=Path(__file__).resolve().parents[1]/'runtime'


def seed_method():
    tree=ast.parse((RUNTIME/'weightsmith/batched_client.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='BatchedPi0Client')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='_seed')
    namespace={'np':np};exec(compile(ast.Module(body=[method],type_ignores=[]),'live-client-seed','exec'),namespace)
    return namespace['_seed']


class SeedContract(unittest.TestCase):
    def test_python_and_numpy_env_ids_produce_same_python_int_seed(self):
        method=seed_method()
        for kind in (int,np.int32,np.int64):
            client=types.SimpleNamespace(_sampling_seed_base=1876,_episode_index=0,_env_replans={})
            first=method(client,kind(3));second=method(client,kind(3))
            self.assertIs(type(first),int);self.assertIs(type(second),int)
            self.assertEqual(first,(1876*64+3)*100);self.assertEqual(second,first+1)

    def test_negative_overflow_float_and_boolean_are_not_silently_coerced(self):
        method=seed_method()
        for base,env in ((-1,0),(2**32,0),(1,1.25),(1,True),(1,-1)):
            client=types.SimpleNamespace(_sampling_seed_base=base,_episode_index=0,_env_replans={})
            with self.assertRaises(ValueError):method(client,env)
            self.assertFalse(client._env_replans)

    def test_actual_openpi_msgpack_codec_keeps_python_int_when_available(self):
        try:from openpi_client import msgpack_numpy
        except ImportError:self.skipTest('OpenPI codec optional locally; required remote CPU environment')
        client=types.SimpleNamespace(_sampling_seed_base=1876,_episode_index=0,_env_replans={})
        value=seed_method()(client,np.int64(3))
        wire=msgpack_numpy.Packer().pack({'sampling_seed':value})
        received=msgpack_numpy.unpackb(wire)['sampling_seed']
        self.assertIs(type(received),int);self.assertEqual(received,value)

if __name__=='__main__':unittest.main()
