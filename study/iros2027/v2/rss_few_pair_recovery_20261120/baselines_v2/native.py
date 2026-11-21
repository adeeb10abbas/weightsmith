"""Native OpenPI inference hooks; no replacement transformer or sampler.

The residual hook adds an offset after one prefix block, before the next block
fills its KV cache. Temporary integer layer markers are added only to a merged
inference module. Original checkpoint and server-state arrays are never edited.
"""
from __future__ import annotations

MARKER = '_fewpair_layer_index'


def instruction_mask_jax(tokens, valid, prefix_ids, delimiter_ids, separator_token_ids=None):
    import jax.numpy as jnp
    length = tokens.shape[1]
    matches = jnp.ones(tokens.shape, dtype=bool)
    for offset, token in enumerate(delimiter_ids[1:]):
        shifted = jnp.pad(tokens[:, offset:] == token, ((0, 0), (0, offset)), constant_values=False)
        shifted_valid=jnp.pad(valid[:,offset:],((0,0),(0,offset)),constant_values=False)
        matches &= shifted & shifted_valid
    anchor=jnp.argmax(matches,axis=1)
    end=anchor-1
    pos = jnp.arange(length)[None, :]
    separators=jnp.asarray(separator_token_ids or [delimiter_ids[0]])
    preceding=jnp.take_along_axis(tokens,jnp.maximum(end,0)[:,None],axis=1)[:,0]
    valid_prefix=jnp.all(tokens[:,:len(prefix_ids)]==jnp.asarray(prefix_ids),axis=1)
    valid_prefix &= jnp.all(valid[:,:len(prefix_ids)],axis=1)
    contiguous=jnp.all(jnp.diff(valid.astype(jnp.int32),axis=1)<=0,axis=1)
    okay=(jnp.sum(matches,axis=1)==1)&(end>len(prefix_ids))&jnp.any(preceding[:,None]==separators[None,:],axis=1)
    return (pos>=len(prefix_ids))&(pos<end[:,None])&valid&okay[:,None]&valid_prefix[:,None]&contiguous[:,None]


def embedding_interceptor(direction, strength, valid, prefix_ids, delimiter_ids, separator_token_ids=None, *, replacement=None):
    import jax.numpy as jnp
    from openpi.models import gemma

    def intercept(next_fun, args, kwargs, context):
        result = next_fun(*args, **kwargs)
        if isinstance(context.module, gemma.Module) and context.method_name == 'embed':
            if replacement is not None:
                if replacement.shape != result.shape:
                    raise ValueError('Positive-control embedding shape differs')
                return replacement.astype(result.dtype)
            mask = instruction_mask_jax(args[0], valid, prefix_ids, delimiter_ids,separator_token_ids)
            offset = (direction * strength).astype(result.dtype)
            result = result + jnp.where(mask[..., None], offset, jnp.zeros_like(offset))
        return result
    return intercept


def residual_interceptor(direction, strength, layer):
    import jax.numpy as jnp
    from openpi.models import gemma
    grid = direction.reshape(2, 4, 4, direction.shape[-1])
    expanded = jnp.repeat(jnp.repeat(grid, 4, axis=1), 4, axis=2).reshape(512, direction.shape[-1])

    def intercept(next_fun, args, kwargs, context):
        result = next_fun(*args, **kwargs)
        if isinstance(context.module, gemma.Block) and context.method_name == '__call__':
            outputs, cache = result
            if outputs[0] is not None:  # Prefix only; never modify the action expert.
                marker = context.module.get_variable('params', MARKER)
                if marker is None:
                    raise ValueError('Native scanned layer is missing its intervention index')
                if outputs[0].shape[1] < 512 or outputs[0].shape[-1] != direction.shape[-1]:
                    raise ValueError('Unsupported native prefix layout')
                # Native mask is [batch,1,query,key]; infer valid prefix keys.
                attention_mask = args[3] if len(args) > 3 else kwargs['attn_mask']
                valid = jnp.any(attention_mask, axis=-2)[:, 0, :512]
                amount = jnp.where(marker == layer, strength, jnp.asarray(0., dtype=jnp.float32))
                offset = (expanded * amount).astype(outputs[0].dtype)
                offset = jnp.where(valid[..., None], offset[None], jnp.zeros_like(offset)[None])
                changed = outputs[0].at[:, :512].add(offset)
                outputs = [changed, *outputs[1:]]
                result = outputs, cache
        return result
    return intercept


def add_layer_markers(module):
    """Attach index metadata to the transient merged module, not server state."""
    import jax.numpy as jnp
    from flax import nnx
    layers = module.PaliGemma.llm.layers
    if MARKER in layers:
        raise ValueError('Layer markers must not persist in the checkpoint state')
    depth = module.PaliGemma.llm.module.configs[0].depth
    layers[MARKER] = nnx.Param(jnp.arange(depth, dtype=jnp.int32))


def make_embedding_function(graphdef, prefix_ids, delimiter_ids,separator_token_ids=None):
    from flax import nnx, linen as nn

    def infer(state, rng, observation, noise, direction, strength):
        model = nnx.merge(graphdef, state)
        hook = embedding_interceptor(direction, strength, observation.tokenized_prompt_mask,
                                     prefix_ids, delimiter_ids,separator_token_ids)
        with nn.intercept_methods(hook):
            return model.sample_actions(rng, observation, noise=noise)
    return infer


def make_residual_function(graphdef, layer):
    from flax import nnx, linen as nn

    def infer(state, rng, observation, noise, direction, strength):
        model = nnx.merge(graphdef, state)
        add_layer_markers(model)
        with nn.intercept_methods(residual_interceptor(direction, strength, layer)):
            return model.sample_actions(rng, observation, noise=noise)
    return infer


def make_embedding_replacement_function(graphdef):
    """Positive control only: inject exact clarified embeddings and token mask."""
    import dataclasses
    from flax import nnx, linen as nn

    def infer(state, rng, observation, noise, replacement, replacement_mask):
        model = nnx.merge(graphdef, state)
        observation = dataclasses.replace(observation, tokenized_prompt_mask=replacement_mask)
        hook = embedding_interceptor(None, None, None, (), (), replacement=replacement)
        with nn.intercept_methods(hook):
            return model.sample_actions(rng, observation, noise=noise)
    return infer
