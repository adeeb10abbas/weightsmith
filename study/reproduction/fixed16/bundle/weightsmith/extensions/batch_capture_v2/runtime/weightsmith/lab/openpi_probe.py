# Vendored from the author's earlier project `weight-space-policy-lab`
# (src/weight_space_policy_lab/openpi_probe.py), trimmed to the functions this harness uses.
# Logic is unchanged. See NOTICE.
"""Layer-local diagnostics using the trained OpenPI blocks, never a replacement policy.

Only the diagnostic execution is instrumented. All candidates remain ordinary static
weight edits. Simulator stage labels are never passed into model inference.
"""

from __future__ import annotations


def _attention_probs(params, normalized, positions, mask, cache, expert, config):
    """Reconstruct attention from the exact captured norm inputs and cached keys."""
    import einops
    import jax
    import jax.numpy as jnp
    from openpi.models import gemma

    suffix = "_1" if expert else ""
    x = normalized
    q = jnp.einsum("BTD,NDH->BTNH", x, params[f"q_einsum{suffix}"]["w"].astype(x.dtype))
    k, v = jnp.einsum("BSD,2KDH->2BSKH", x, params[f"kv_einsum{suffix}"]["w"].astype(x.dtype))
    q = gemma._apply_rope(q, positions=positions) * config.head_dim**-0.5
    k = gemma._apply_rope(k, positions=positions)
    if cache is not None:
        k = jnp.concatenate([cache[0], k], axis=1)
        v = jnp.concatenate([cache[1], v], axis=1)
    q = einops.rearrange(q, "B T (K G) H -> B T K G H", K=config.num_kv_heads)
    logits = jnp.einsum("BTKGH,BSKH->BKGTS", q, k, preferred_element_type=jnp.float32)
    probabilities = jax.nn.softmax(
        jnp.where(mask[:, :, None], logits, -2.3819763e38), axis=-1
    ).astype(x.dtype)
    encoded = jnp.einsum("BKGTS,BSKH->BTKGH", probabilities, v)
    encoded = einops.rearrange(encoded, "B T K G H -> B T (K G) H")
    output = jnp.einsum(
        "BTNH,NHD->BTD", encoded, params[f"attn_vec_einsum{suffix}"]["w"].astype(x.dtype)
    )
    return probabilities, output


def _block_with_trace(params, x, positions, mask, cond, cache, configs, expert, indices):
    """Capture real module inputs/outputs through Linen's interception interface."""
    from flax import linen as nn
    from openpi.models import gemma, lora

    captured = {}
    suffix = "_1" if expert else ""

    def intercept(next_fun, args, kwargs, context):
        module, method = context.module, context.method_name
        if (
            isinstance(module, lora.FeedForward)
            and method == "_dot"
            and args[1].shape == (configs[expert].mlp_dim, configs[expert].width)
        ):
            captured["mlp_down_input"] = args[0][:, indices]
        result = next_fun(*args, **kwargs)
        if method == "__call__":
            if isinstance(module, gemma.RMSNorm) and module.name == f"pre_attention_norm{suffix}":
                captured["attention_input_full"] = result[0]
                captured["attention_gate"] = result[1]
            elif isinstance(module, gemma.RMSNorm) and module.name == f"pre_ffw_norm{suffix}":
                captured["mlp_input"] = result[0][:, indices]
                captured["mlp_gate"] = result[1]
            elif isinstance(module, gemma.Attention):
                captured["attention_output"] = result[0][expert][:, indices]
        return result

    xs = [None, None]
    xs[expert] = x
    conds = [None, None]
    conds[expert] = cond
    with nn.intercept_methods(intercept):
        outputs, new_cache = gemma.Block(configs=tuple(configs)).apply(
            {"params": params},
            xs,
            cache,
            positions,
            mask,
            conds,
            True,
        )
    norm = captured.pop("attention_input_full")
    probabilities, reconstructed = _attention_probs(
        params["attn"], norm, positions, mask, cache, expert, configs[expert]
    )
    captured["attention_reconstruction_error"] = (
        reconstructed[:, indices] - captured["attention_output"]
    )
    # [batch, heads, sampled query token, all key tokens]. Keep actual masked probabilities.
    captured["attention_probs"] = probabilities[:, 0][:, :, indices, :]
    captured["attention_input"] = norm[:, indices]
    captured["residual_input"] = x[:, indices]
    captured["residual_output"] = outputs[expert][:, indices]
    if expert == 0:
        captured.pop("attention_gate")
        captured.pop("mlp_gate")
    return outputs[expert], new_cache, captured


def trace_llm(model, embedded, positions, mask, cond, cache, expert, spec):
    """Run the original per-layer Block with the original scanned parameters."""
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import gemma

    params = nnx.state(model.PaliGemma.llm, nnx.Param).to_pure_dict()
    configs = model.PaliGemma.llm.module.configs
    x = embedded.astype(model.PaliGemma.llm.module.embed_dtype)
    indices = jnp.asarray(
        spec.action_tokens if expert else tuple(x.shape[1] - 200 + i for i in spec.prompt_tokens)
    )
    full_mask = jnp.asarray(mask)[:, None]

    def step(carry, layer_values):
        layer_params, layer_cache = layer_values
        output, new_cache, trace = _block_with_trace(
            layer_params,
            carry,
            positions,
            full_mask,
            cond,
            layer_cache,
            configs,
            expert,
            indices,
        )
        return output, (new_cache, trace)

    x, (new_cache, traces) = jax.lax.scan(step, x, (params["layers"], cache))
    suffix = "_1" if expert else ""
    x, _ = gemma.RMSNorm().apply({"params": params[f"final_norm{suffix}"]}, x, cond)
    traces = jax.tree.map(lambda a: a[jnp.asarray(spec.layers)], traces)
    return x, new_cache, traces


def make_probe_method(spec):
    # spec: any object with .layers, .action_tokens, .prompt_tokens (tuples of ints); see acts_worker.Spec
    def probe(model, rng, observation):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as openpi_model
        from openpi.models.pi0 import make_attn_mask

        observation = openpi_model.preprocess_observation(None, observation, train=False)
        prefix, prefix_mask, prefix_ar = model.embed_prefix(observation)
        positions = jnp.cumsum(prefix_mask, axis=1) - 1
        prefix_output, cache, vlm = trace_llm(
            model,
            prefix,
            positions,
            make_attn_mask(prefix_mask, prefix_ar),
            None,
            None,
            0,
            spec,
        )
        batch = observation.state.shape[0]
        noise = jax.random.normal(rng, (batch, model.action_horizon, model.action_dim))

        def step(carry, _):
            x_t, time = carry
            suffix, mask, ar, cond = model.embed_suffix(
                observation, x_t, jnp.broadcast_to(time, batch)
            )
            prefix_attention = jnp.broadcast_to(
                prefix_mask[:, None, :], (batch, suffix.shape[1], prefix.shape[1])
            )
            full_mask = jnp.concatenate([prefix_attention, make_attn_mask(mask, ar)], axis=-1)
            positions = jnp.sum(prefix_mask, axis=-1)[:, None] + jnp.cumsum(mask, axis=-1) - 1
            h, _, trace = trace_llm(model, suffix, positions, full_mask, cond, cache, 1, spec)
            velocity = model.action_out_proj(h[:, -model.action_horizon :])
            trace["head_input"] = h
            trace["flow_velocity"] = velocity
            trace["time_condition"] = cond
            return (x_t - 0.1 * velocity, time - 0.1), trace

        (actions, _), action = jax.lax.scan(step, (noise, 1.0), None, length=10)
        return actions, {
            "vlm": vlm,
            "action": action,
            "prefix_output": prefix_output,
            "prefix_mask": prefix_mask,
            "token_ids": observation.tokenized_prompt,
            "token_mask": observation.tokenized_prompt_mask,
        }

    return probe
