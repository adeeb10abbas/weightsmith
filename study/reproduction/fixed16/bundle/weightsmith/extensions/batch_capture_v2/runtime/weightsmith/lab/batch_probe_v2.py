"""New instrumented numerical path: explicit noise, real batch context, no pooling."""
from __future__ import annotations
from types import SimpleNamespace


def make_batch_probe(settings):
    # All imports below are lazy: inspecting/importing this module never initializes a GPU.
    combined=sorted(set(settings['action_layers'])|set(settings['vlm_layers']))
    tokens=list(range(15)) if settings['action_tokens']=='mean' else settings['action_tokens']
    spec=SimpleNamespace(layers=tuple(combined),action_tokens=tuple(tokens),prompt_tokens=tuple(settings['prompt_tokens']))
    ai=[combined.index(x) for x in settings['action_layers']];vi=[combined.index(x) for x in settings['vlm_layers']]
    def probe(model,observation,noise):
        import jax
        import jax.numpy as jnp
        from openpi.models import model as openpi_model
        from openpi.models.pi0 import make_attn_mask
        from weightsmith.lab.openpi_probe import trace_llm
        observation=openpi_model.preprocess_observation(None,observation,train=False)
        if noise.shape!=(16,model.action_horizon,model.action_dim):raise ValueError('explicit fixed16 per-request noise required')
        prefix,prefix_mask,prefix_ar=model.embed_prefix(observation)
        positions=jnp.cumsum(prefix_mask,axis=1)-1
        _,cache,vlm=trace_llm(model,prefix,positions,make_attn_mask(prefix_mask,prefix_ar),None,None,0,spec)
        def step(carry,_):
            x_t,time=carry
            suffix,mask,ar,cond=model.embed_suffix(observation,x_t,jnp.broadcast_to(time,16))
            prefix_attention=jnp.broadcast_to(prefix_mask[:,None,:],(16,suffix.shape[1],prefix.shape[1]))
            full_mask=jnp.concatenate([prefix_attention,make_attn_mask(mask,ar)],axis=-1)
            positions=jnp.sum(prefix_mask,axis=-1)[:,None]+jnp.cumsum(mask,axis=-1)-1
            h,_,trace=trace_llm(model,suffix,positions,full_mask,cond,cache,1,spec)
            velocity=model.action_out_proj(h[:,-model.action_horizon:])
            trace['head_input']=h[:,-model.action_horizon:]
            trace['flow_velocity']=velocity;trace['time_condition']=cond
            return (x_t-.1*velocity,time-.1),trace
        (actions,_),all_action=jax.lax.scan(step,(noise,1.),None,length=10)
        # Selection is inside JIT, before host materialization. Preserve every
        # position (including explicitly identified padding), token and value at
        # the declared sites; the compatibility view may pool later.
        ds=jnp.asarray(settings['denoise_steps']);la=jnp.asarray(ai)
        action={}
        for key,value in all_action.items():
            selected=value[ds]
            action[key]=selected if key in ('head_input','flow_velocity','time_condition') else selected[:,la]
        vlm_selected={key:value[jnp.asarray(vi)] for key,value in vlm.items()} if vi else {}
        return actions,{'action':action,'vlm':vlm_selected,'token_ids':observation.tokenized_prompt,
                        'token_mask':observation.tokenized_prompt_mask}
    return probe
