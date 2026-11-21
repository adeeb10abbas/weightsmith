"""CPU artifact/alignment tests; native hook tests use the OpenPI environment."""
import contextlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import artifacts as A


class ArtifactsTest(unittest.TestCase):
    def test_instruction_scope_excludes_prefix_state_and_padding(self):
        tokens=np.array([1,10,11,101,102,9,8,7,12,13,0,0])
        mask=A.instruction_mask(tokens,tokens!=0,[1,10,11],[9,8,7])
        self.assertEqual(np.flatnonzero(mask).tolist(),[3,4])
        with self.assertRaises(ValueError):
            A.instruction_mask([1,10,11,9,8,7],[True]*6,[1,10,11],[9,8,7])
        with self.assertRaises(ValueError):
            A.instruction_mask([1,10,11,1,9,8,7,1,9,8,7],[True]*11,[1,10,11],[9,8,7])

    def test_cell_expansion_is_spatial_not_contiguous_chunks(self):
        cells=np.arange(32).reshape(2,16,1)
        expanded=A.expand_image_cells(cells).reshape(2,16,16)
        self.assertTrue(np.all(expanded[0,:4,:4]==0))
        self.assertTrue(np.all(expanded[0,:4,4:8]==1))
        self.assertTrue(np.all(expanded[0,4:8,:4]==4))
        self.assertTrue(np.all(expanded[1,-4:,-4:]==31))

    def test_residual_only_selected_sources_and_grid_cells(self):
        shape=(5,2,2,17,2048)
        original=np.zeros(shape,np.float16);teacher=np.ones(shape,np.float16)
        teacher[0]=2;teacher[2]=4;teacher[4]=6
        teacher[:,:,:,0]=1000
        result=A.residual_direction(original,teacher,[0,2,4],1)
        self.assertEqual(result.shape,(2,16,2048))
        self.assertTrue(np.all(result==4))
        with self.assertRaises(ValueError):A.residual_direction(original,teacher,[0,0,1],1)

    def test_frozen_artifact_detects_direction_tampering(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'candidate';direction=np.ones((2,16,2048),np.float32)
            A.write_artifact(path,'residual_cells',direction,{'artifacts':{}},layer=16)
            spec,loaded=A.load_artifact(path)
            self.assertTrue(np.array_equal(loaded,direction));self.assertFalse(spec['checkpoint_weights_modified'])
            with self.assertRaises(FileExistsError):A.write_artifact(path,'residual_cells',direction,{},layer=16)
            np.save(path/'direction.npy',np.zeros_like(direction))
            with self.assertRaises(ValueError):A.load_artifact(path)


@unittest.skipUnless(importlib.util.find_spec('flax') and os.environ.get('JAX_PLATFORMS')=='cpu',
                     'Native hook tests require OpenPI Python with JAX_PLATFORMS=cpu')
class NativeHooksTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import jax
        import jax.numpy as jnp
        from flax import nnx
        from flax.nnx import bridge
        from openpi.models import gemma
        cls.jax,cls.jnp,cls.nnx=jax,jnp,nnx
        cfg=gemma.Config(width=8,depth=2,mlp_dim=16,num_heads=1,num_kv_heads=1,head_dim=8)
        llm=bridge.ToNNX(gemma.Module(configs=[cfg,cfg],embed_dtype='bfloat16'))
        llm.lazy_init(method='init',use_adarms=[False,False],rngs=nnx.Rngs(42))
        cls.model=nnx.Dict(PaliGemma=nnx.Dict(llm=llm))
        cls.graph,cls.state=nnx.split(cls.model)
        import native
        cls.N=native

    def test_native_scan_zero_equivalence_and_nonzero_cache_effect(self):
        from flax import linen as nn
        jax,jnp,nnx=self.jax,self.jnp,self.nnx
        x=jax.random.normal(jax.random.key(9),(1,516,8)).astype(jnp.bfloat16)
        mask=jnp.ones((1,516,516),dtype=bool)
        positions=jnp.arange(516,dtype=jnp.int32)[None]
        def native_fn(state):
            m=nnx.merge(self.graph,state)
            return m.PaliGemma.llm([x,None],positions=positions,mask=mask)
        def hook_fn(state,scale):
            m=nnx.merge(self.graph,state)
            self.N.add_layer_markers(m)
            with nn.intercept_methods(self.N.residual_interceptor(jnp.ones((2,16,8)),scale,0)):
                return m.PaliGemma.llm([x,None],positions=positions,mask=mask)
        base=jax.jit(native_fn)(self.state)
        zero=jax.jit(hook_fn)(self.state,jnp.array(0.,jnp.float32))
        changed=jax.jit(hook_fn)(self.state,jnp.array(.5,jnp.float32))
        for a,b in zip(jax.tree.leaves(base),jax.tree.leaves(zero),strict=True):
            np.testing.assert_array_equal(np.asarray(a),np.asarray(b))
        # Layer0 cache is constructed before its output is shifted; layer1 sees
        # the changed prefix. This catches ineffective final-layer-only hooks.
        for a,b in zip(base[1],changed[1],strict=True):
            np.testing.assert_array_equal(np.asarray(a[0]),np.asarray(b[0]))
        self.assertTrue(any(not np.array_equal(np.asarray(a[1]),np.asarray(b[1]))
                            for a,b in zip(base[1],changed[1],strict=True)))
        self.assertNotIn(self.N.MARKER,self.model.PaliGemma.llm.layers)
        self.assertNotIn(self.N.MARKER,self.state['PaliGemma']['llm']['layers'])

    def test_embedding_native_zero_and_instruction_mask(self):
        from flax import linen as nn
        jax,jnp,nnx=self.jax,self.jnp,self.nnx
        tokens=jnp.array([[1,10,11,101,102,9,8,7,12,0]])
        valid=tokens!=0;direction=jnp.arange(8,dtype=jnp.float32)+1
        def run(scale):
            m=nnx.merge(self.graph,self.state)
            hook=self.N.embedding_interceptor(direction,scale,valid,[1,10,11],[9,8,7])
            with nn.intercept_methods(hook):return m.PaliGemma.llm(tokens,method='embed')
        m=nnx.merge(self.graph,self.state);base=m.PaliGemma.llm(tokens,method='embed')
        zero=jax.jit(run)(jnp.array(0.,jnp.float32));changed=jax.jit(run)(jnp.array(1.,jnp.float32))
        np.testing.assert_array_equal(np.asarray(base),np.asarray(zero))
        native_mask=np.asarray(self.N.instruction_mask_jax(tokens,valid,[1,10,11],[9,8,7]))
        self.assertEqual(np.flatnonzero(native_mask[0]).tolist(),[3,4])
        np.testing.assert_array_equal(np.asarray(base)[~native_mask],np.asarray(changed)[~native_mask])
        self.assertTrue(np.any(np.asarray(base)[native_mask]!=np.asarray(changed)[native_mask]))


if __name__=='__main__':unittest.main()
