"""Explicit per-request noise and fixed16 native inference for the v2 fork."""
from __future__ import annotations
import json
import time
from pathlib import Path
import numpy as np
from weightsmith import config as C
from .batch_context_v2 import PROTOCOL_ID, batch_layout, checked_identity, seeded_noise, sha


def project_identity(policy_fingerprint):
    path=C.PROJECT_ROOT/'v2_runtime_binding.json'
    if path.is_symlink():raise ValueError('runtime binding cannot be a symlink')
    binding=json.loads(path.read_text())
    source=sha(C.PKG/'batched_server.py')
    if (binding.get('capture_protocol_id')!=PROTOCOL_ID or binding.get('runtime_variant')!='fixed16_new_runtime'
        or binding.get('native_server_sha256')!=source or Path(binding['checkpoint_path']).resolve()!=C.CHECKPOINT.resolve()):
        raise ValueError('v2 runtime binding differs from actual server/checkpoint/variant')
    return checked_identity({'checkpoint_manifest_sha256':binding['checkpoint_manifest_sha256'],
                             'runtime_manifest_sha256':sha(path),'native_server_sha256':source,
                             'policy_fingerprint':policy_fingerprint})


def prepare_batch(server,requests):
    jax,jnp=server.jax,server.jax.numpy;layout=batch_layout(len(requests),'fixed16_new_runtime')
    inputs=[];seeds=[]
    for request in requests:
        obs=dict(request);seed=obs.pop('sampling_seed')
        if type(seed)is not int or not 0<=seed<2**32:raise ValueError('explicit request sampling seed required')
        seeds.append(seed);inputs.append(server.policy._input_transform(obs))
    order=layout['position_to_source']
    stacked=jax.tree.map(lambda *xs:jnp.stack([jnp.asarray(xs[i]) for i in order]),*inputs)
    noise=seeded_noise(seeds,layout,lambda seed:jax.random.normal(jax.random.key(seed),(1,server.H,server.D)),jnp.concatenate)
    return {'stacked':stacked,'observation':server._model.Observation.from_dict(stacked),
            'noise':noise,'layout':layout,'seeds':seeds}


def postprocess(server,raw,context):
    states=np.asarray(context['stacked']['state']);raw=np.asarray(raw)
    return [server.policy._output_transform({'state':states[i],'actions':raw[i]})
            for i in range(context['layout']['real_count'])]


def native_forward(server,requests):
    context=prepare_batch(server,requests);start=time.monotonic()
    raw=server.jitted(server.state,server.jax.random.key(0),context['observation'],context['noise'])
    outputs=postprocess(server,raw,context);ms=(time.monotonic()-start)*1000
    for output in outputs:output['policy_timing']={'infer_ms':ms,'batch':len(requests),'padded_batch':16}
    return outputs,context
