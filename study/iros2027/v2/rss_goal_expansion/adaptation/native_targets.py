"""Fixed-16 native action targets and exact replay for the additive goal cohort.

Copied from the sealed original comparison; only serving-bucket validation and
replay padding differ. The expansion uses prepare only, not LoRA training.

Uses no human demonstrations: teacher action chunks are sampled under a clearer instruction
on exactly the recorded development observations used for the closed-form edit.
Only rank-r updates to selected VLM MLP down-projections are trained. The action
expert is frozen. The native flow-matching loss is used without image augmentation.
All exported dense tensors are bf16-rounded and consumed by the unchanged pool.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import numpy as np

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO))
LIVE_REPLAY_TOLERANCE = 1e-6


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(8 << 20):
            h.update(b)
    return h.hexdigest()


def dump(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def input_digest(request, seed, prompt):
    """Identical canonical raw-input digest used by the closed-form capture path."""
    h = hashlib.sha256()
    for key in sorted(request):
        value = np.ascontiguousarray(request[key])
        h.update(json.dumps([key, str(value.dtype), list(value.shape)]).encode())
        h.update(value.tobytes())
    h.update(json.dumps({'seed': seed, 'prompt': prompt}, sort_keys=True).encode())
    return h.hexdigest()


def flatten(tree, prefix=()):
    out = {}
    for key, value in tree.items():
        p = (*prefix, key)
        if isinstance(value, dict):
            out.update(flatten(value, p))
        else:
            out['/'.join(p)] = np.asarray(value)
    return out


def unflatten(tree):
    out = {}
    for key, value in tree.items():
        parts = key.split('/')
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def read_spec(path):
    spec = json.loads(Path(path).read_text())
    if spec.get('split') != 'development':
        raise ValueError('Only explicitly development-labeled replay data may train LoRA')
    if not spec.get('cohort_id') or not spec.get('items'):
        raise ValueError('cohort_id and items are required')
    seen = set()
    ids = set()
    split_paths = {'train': set(), 'validation': set()}
    for i, it in enumerate(spec['items']):
        if it.get('split') not in split_paths:
            raise ValueError('Every item needs train or validation split')
        p = Path(it['obs_path']).resolve()
        if any('heldout' in part.lower() or 'held-out' in part.lower() for part in p.parts):
            raise ValueError(f'Held-out path refused: {p}')
        k = it.get('decision_index')
        if type(k) is not int or k < 0:
            raise ValueError('decision_index must be a non-negative integer')
        for field in ['original_prompt', 'teacher_prompt']:
            prompt = it.get(field, spec.get(field))
            if not isinstance(prompt, str) or not prompt.strip():
                raise ValueError(f'{field} is required globally or per item')
        identity = (str(p), k, it.get('original_prompt', spec.get('original_prompt')))
        if it.get('role', 'repair') not in {'repair', 'preservation'}:
            raise ValueError('role must be repair or preservation')
        item_id = it.get('item_id', f'item-{i:04d}')
        if item_id in ids:
            raise ValueError('Duplicate item_id')
        ids.add(item_id)
        if it.get('role') == 'preservation' and it.get('original_prompt', spec.get('original_prompt')) != it.get('teacher_prompt', spec.get('teacher_prompt')):
            raise ValueError('Preservation items must use the same original and teacher instruction')
        if identity in seen:
            raise ValueError(f'Duplicate input in spec: {identity}')
        seen.add(identity)
        split_paths[it['split']].add(str(p))
    if not all(split_paths.values()):
        raise ValueError('Train and development-validation episodes are both required')
    if split_paths['train'] & split_paths['validation']:
        raise ValueError('Train and validation must use distinct episodes, not merely distinct decisions')
    return spec


def load_policy():
    from weightsmith import config as C
    from openpi.policies import policy_config
    from openpi.training import config
    return policy_config.create_trained_policy(config.get_config('pi05_droid_jointpos'), C.CHECKPOINT)


def validate_live_metadata(item):
    """Check the declared serving bucket and its canonical development context."""
    required = ('live_batch_size','live_batch_bucket','live_batch_members','source_env_id',
                'source_run_id','source_step','live_context','live_context_sha256')
    if any(key not in item for key in required):
        raise ValueError('Missing exact live replay context')
    size, bucket = item['live_batch_size'], item['live_batch_bucket']
    if type(size) is not int or not 1 <= size <= 16 or type(bucket) is not int or bucket != 16 or item.get('inference_batching') != 'fixed16':
        raise ValueError('Wrong live batch size or serving bucket')
    members = item['live_batch_members']
    if any(type(item[key]) is not int or item[key] < 0 for key in ('source_env_id','source_run_id','source_step')) or item['source_env_id'] >= 16:
        raise ValueError('Invalid live source environment/run/step metadata')
    if not isinstance(members,list) or len(members) != size:
        raise ValueError('Live batch membership count differs from batch size')
    ids = [member['env_id'] for member in members]
    if any(type(i) is not int or not 0 <= i < 16 for i in ids) or ids != sorted(set(ids)):
        raise ValueError('Live batch members must have unique sorted environment IDs')
    context = item['live_context']
    digest = hashlib.sha256(json.dumps(context,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if digest != item['live_context_sha256']:
        raise ValueError('Live context checksum mismatch')
    if context['members'] != members or any(context[key] != item[key] for key in ('source_run_id','source_step')):
        raise ValueError('Live context differs from flat item metadata')
    source = Path(item['obs_path']).resolve()
    matching = [m for m in members if m['env_id'] == item['source_env_id']]
    if len(matching) != 1 or Path(matching[0]['obs_path']).resolve() != source or matching[0]['decision_index'] != item['decision_index']:
        raise ValueError('Source decision is missing from live batch membership')
    for member in members:
        p = Path(member['obs_path']).resolve()
        match = re.fullmatch(r'obs_r(\d+)_e(\d+)\.npz',p.name)
        if not match or int(match[1]) != item['source_run_id'] or int(match[2]) != member['env_id'] or p.parent != source.parent:
            raise ValueError('Live member filename/run/source-directory mismatch')
        if type(member['decision_index']) is not int or member['decision_index'] < 0:
            raise ValueError('Invalid live member decision index')
        if context['source_batch_observation_sha256'].get(str(p)) != member['obs_sha256']:
            raise ValueError('Live member hash differs from its source-batch manifest')
    return context


def validate_live_context(item, spec, cache):
    """Reconstruct membership from the sealed development batch, before GPU use."""
    context = validate_live_metadata(item)
    hashes = cache.setdefault('hashes',{})
    def checked_hash(path):
        path = str(Path(path).resolve())
        if path not in hashes:
            hashes[path] = sha(path)
        return hashes[path]
    episodes_path = Path(context['source_episodes_path']).resolve()
    receipt_path = Path(context['source_receipt_path']).resolve()
    if receipt_path != episodes_path.parent/'completed.json':
        raise ValueError('Live context receipt is not owned by its source batch')
    for key,path in [('source_episodes_sha256',episodes_path),('source_receipt_sha256',receipt_path)]:
        if checked_hash(path) != context[key]:
            raise ValueError('Live context source receipt or episodes changed')
    receipt = json.loads(receipt_path.read_text())
    rows = json.loads(episodes_path.read_text())
    if receipt['episodes_sha256'] != context['source_episodes_sha256'] or receipt['n'] != 16 or len(rows) != 16:
        raise ValueError('Live context requires a sealed sixteen-episode source batch')
    source = Path(item['obs_path']).resolve()
    manifests = {str(Path(row['obs']).resolve()):row['obs_sha256'] for row in rows}
    if manifests != context['source_batch_observation_sha256']:
        raise ValueError('Live context observation manifest differs from source receipts')
    if len(manifests) != 16 or any(row.get('inference_batching') != 'fixed16' or row.get('split') != 'development' or row.get('condition') != 'base'
                                 or not row.get('ok') or row.get('case') != spec.get('case') for row in rows):
        raise ValueError('Live replay context must contain unchanged-policy development episodes from this case')
    source_row = next((row for row in rows if Path(row['obs']).resolve() == source),None)
    prompt = item.get('recorded_prompt',spec.get('recorded_prompt'))
    if not source_row or source_row['effective_prompt'] != prompt or source_row['env'] != item['source_env_id'] or source_row['run'] != item['source_run_id']:
        raise ValueError('Live source prompt/environment/run does not match its sealed episode')
    steps_cache = cache.setdefault('steps',{})
    actual = []
    for row in sorted(rows,key=lambda r:r['env']):
        p = Path(row['obs']).resolve()
        if not p.is_relative_to(episodes_path.parent) or checked_hash(p) != row['obs_sha256']:
            raise ValueError('Live context observation changed or belongs to another batch')
        if row['run'] != item['source_run_id'] or p.parent != source.parent:
            continue
        if str(p) not in steps_cache:
            with np.load(p,allow_pickle=False) as z:
                steps_cache[str(p)] = z['step'].copy()
        matches = np.flatnonzero(steps_cache[str(p)] == item['source_step'])
        if len(matches) > 1:
            raise ValueError('Duplicate global step in live context')
        if len(matches):
            actual.append({'env_id':row['env'],'obs_path':str(p),'decision_index':int(matches[0]),'obs_sha256':row['obs_sha256']})
    if actual != item['live_batch_members']:
        raise ValueError('Live batch membership does not match the recorded global step')
    if len({row['cohort_id'] for row in rows}) != 1:
        raise ValueError('Live context mixes source cohorts')
    return {'source_cohort_id':source_row['cohort_id'],'source_episodes_sha256':context['source_episodes_sha256'],
            'source_receipt_sha256':context['source_receipt_sha256']}


def physical_errors(actions, recorded):
    a,b = np.asarray(actions,dtype=np.float32),np.asarray(recorded,dtype=np.float32)
    if a.shape != (15,8) or b.shape != (15,8) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid physical replay action shape or finiteness')
    difference = a.astype(np.float64)-b.astype(np.float64)
    return {'bit_exact':bool(np.array_equal(a,b)),'max_abs_error':float(np.max(np.abs(difference))),
            'per_coordinate_max_abs_error':np.max(np.abs(difference),axis=0).tolist()}


def validate_live_certificate(meta):
    gate = meta.get('live_replay_gate',{})
    if gate.get('mode') != 'exact_pool_bucket' or gate.get('status') != 'pass' or gate.get('tolerance') != LIVE_REPLAY_TOLERANCE:
        raise ValueError('Missing or invalid exact live replay certificate')
    maximum = gate.get('maximum_error',float('inf'))
    if not np.isfinite(maximum) or not 0 <= maximum <= LIVE_REPLAY_TOLERANCE or gate.get('items') != len(meta['records']):
        raise ValueError('Live replay certificate is incomplete or outside tolerance')
    errors = []
    for row in meta['records']:
        value = row.get('live_replay_max_abs_error',float('inf'))
        if not np.isfinite(value) or not 0 <= value <= LIVE_REPLAY_TOLERANCE or type(row.get('live_replay_bit_exact')) is not bool:
            raise ValueError('Missing or invalid per-item live replay certificate')
        if not isinstance(row.get('recorded_input_sha256'),str) or len(row['recorded_input_sha256']) != 64:
            raise ValueError('Live replay certificate lacks the recorded input identity')
        errors.append(value)
    if not errors or max(errors) != maximum:
        raise ValueError('Live replay certificate maximum differs from item errors')


def prepare(args):
    spec = read_spec(args.spec)
    out = Path(args.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    dump(out / 'input-spec.json', spec)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.models import model as om
    from weightsmith import config as C
    from weightsmith.batched_server import Server
    start = time.time()
    context_cache = {}
    live_sources = [validate_live_context(item,spec,context_cache) for item in spec['items']]
    # Reuse the exact pool implementation and its one checkpoint copy. The
    # singleton teacher and exact-bucket validation serve distinct purposes.
    server = Server()
    policy = server.policy
    graph, state = server.graphdef, server.state

    @jax.jit
    def sample(state, observation, seed):
        model = nnx.merge(graph, state)
        return model.sample_actions(jax.random.key(seed), observation)

    records = []
    targets = []
    physical_targets = []
    hashes = {}
    live_seconds = 0.0
    singleton_extra_calls = 0
    for i, it in enumerate(spec['items']):
        p = Path(it['obs_path']).resolve()
        if str(p) not in hashes:
            hashes[str(p)] = sha(p)
        k = it['decision_index']
        with np.load(p, allow_pickle=False) as z:
            if k >= len(z['seed']):
                raise IndexError((p, k))
            seed = int(z['seed'][k])
            if seed < 0:
                raise ValueError('A recorded deterministic sampling seed is required')
            raw = {key:z[key][k] for key in ('base_img','wrist_img','joint','gripper')}
            request = {'observation/exterior_image_1_left': raw['base_img'],
                       'observation/wrist_image_left': raw['wrist_img'],
                       'observation/joint_position': raw['joint'],
                       'observation/gripper_position': raw['gripper']}
            recorded_actions = z['actions'][k].copy()
        original = it.get('original_prompt', spec.get('original_prompt'))
        teacher = it.get('teacher_prompt', spec.get('teacher_prompt'))
        source_prompt = it.get('recorded_prompt', spec.get('recorded_prompt'))
        recorded_input_sha = input_digest(raw,seed,source_prompt)
        student_inputs = policy._input_transform({**request, 'prompt': original})
        teacher_inputs = policy._input_transform({**request, 'prompt': teacher})
        inputs = jax.tree.map(lambda x: jnp.asarray(x)[None], teacher_inputs)
        target = np.asarray(sample(state, om.Observation.from_dict(inputs), seed)[0], dtype=np.float32)
        if not np.isfinite(target).all():
            raise ValueError('Nonfinite teacher target')
        robot_target = policy._output_transform({'state': np.asarray(inputs['state'][0]), 'actions': target})['actions']
        src_inputs = teacher_inputs if source_prompt == teacher else (student_inputs if source_prompt == original else
                     policy._input_transform({**request,'prompt':source_prompt}))
        inp = jax.tree.map(lambda x:jnp.asarray(x)[None],src_inputs)
        native = target if source_prompt == teacher else np.asarray(sample(state,om.Observation.from_dict(inp),seed)[0])
        singleton_extra_calls += int(source_prompt != teacher)
        singleton_actions = policy._output_transform({'state':np.asarray(inp['state'][0]),'actions':native})['actions']
        singleton_error = physical_errors(singleton_actions,recorded_actions)
        if input_digest(raw,seed,source_prompt) != recorded_input_sha:
            raise ValueError('Input transformation mutated the recorded raw observation')
        live_start = time.time()
        live_request = {**request,'prompt':source_prompt,'sampling_seed':seed}
        # Examples are independent across the batch dimension. Repeating this
        # exact request recreates the recorded kernel bucket and pool padding
        # without making other validation-context observations training targets.
        live_actions = np.asarray(server.infer_batch([live_request]*it['live_batch_bucket'])[0]['actions'],dtype=np.float32)
        live_wall = time.time()-live_start
        live_seconds += live_wall
        live_error = physical_errors(live_actions,recorded_actions)
        if live_error['max_abs_error'] > LIVE_REPLAY_TOLERANCE:
            raise ValueError(f'Exact pool-bucket replay error {live_error["max_abs_error"]:g} exceeds {LIVE_REPLAY_TOLERANCE:g}')
        name = f'item-{i:04d}.npz'
        np.savez_compressed(out/name, **{'input/'+k:v for k,v in flatten(student_inputs).items()},
                            teacher_actions=target, teacher_actions_physical=robot_target,
                            live_replay_actions_physical=live_actions,
                            recorded_actions_physical=np.asarray(recorded_actions,dtype=np.float32),
                            singleton_recorded_actions_physical=np.asarray(singleton_actions,dtype=np.float32))
        rec = {**it, 'obs_path': str(p), 'policy_seed': seed, 'original_prompt': original,
               'teacher_prompt': teacher, 'recorded_prompt':source_prompt,'item_id': it.get('item_id', f'item-{i:04d}'),
               'source_observation_sha256':hashes[str(p)],
               'original_input_sha256':input_digest(raw,seed,original),
               'teacher_input_sha256':input_digest(raw,seed,teacher),
               'recorded_input_sha256':recorded_input_sha,**live_sources[i],
               'live_replay_max_abs_error':live_error['max_abs_error'],
               'live_replay_bit_exact':live_error['bit_exact'],
               'live_replay_per_coordinate_max_abs_error':live_error['per_coordinate_max_abs_error'],
               'live_replay_wall_seconds':live_wall,
               'singleton_recorded_max_abs_error':singleton_error['max_abs_error'],
               'singleton_recorded_per_coordinate_max_abs_error':singleton_error['per_coordinate_max_abs_error'],
               'file': name, 'sha256': sha(out/name)}
        records.append(rec)
        targets.append(target)
        physical_targets.append(robot_target)
        dump(out/'progress.json', {'stage':'preparing','done':len(records),'total':len(spec['items']), 'wall_seconds':time.time()-start})
        print(f'teacher input {i+1}/{len(spec["items"])} split={it["split"]} '
              f'live_bucket={it["live_batch_bucket"]} live_error={live_error["max_abs_error"]:.8g} '
              f'live_bit_exact={live_error["bit_exact"]} singleton_drift={singleton_error["max_abs_error"]:.8g}',flush=True)
    np.savez_compressed(out/'teacher-actions.npz',actions=np.stack(physical_targets),
                        normalized_actions=np.stack(targets),
                        item_ids=np.asarray([r['item_id'] for r in records]),
                        policy_seeds=np.asarray([r['policy_seed'] for r in records]))
    report = {'cohort_id':spec['cohort_id'],'split':'development','method':'native teacher targets and fixed16 replay; no LoRA fit in this cohort',
              'reuse_or_new':'new teacher samples on shared fresh development observations; reused pretrained checkpoint',
              'base_checkpoint':str(C.CHECKPOINT),'base_workspace':str(C.BASE_WS),
              'base_workspace_manifest_sha256':sha(C.BASE_WS/'workspace.json'),
              'spec_sha256':sha(args.spec),'stored_spec_sha256':sha(out/'input-spec.json'),
              'source_observation_sha256':hashes,'records':records,
              'teacher_actions_file':'teacher-actions.npz','teacher_actions_sha256':sha(out/'teacher-actions.npz'),
              'wall_seconds':time.time()-start, 'teacher_forward_calls':len(records),
              'singleton_extra_forward_calls':singleton_extra_calls,
              'live_replay_gate':{'mode':'exact_pool_bucket','tolerance':LIVE_REPLAY_TOLERANCE,'status':'pass',
                  'maximum_error':max(r['live_replay_max_abs_error'] for r in records),
                  'max_per_coordinate_error':np.max([r['live_replay_per_coordinate_max_abs_error'] for r in records],axis=0).tolist(),
                  'items':len(records),'forward_calls':len(records),
                  'padded_policy_examples':sum(r['live_batch_bucket'] for r in records),
                  'repeated_request_examples':sum(r['live_batch_bucket'] for r in records),'wall_seconds':live_seconds},
              'live_replay_context_hashes':context_cache['hashes'],
              'singleton_replay_note':'Batch-one versus recorded live-bucket drift is diagnostic, not input identity or target-parity proof.',
              'code_sha256':sha(__file__)}
    validate_live_certificate(report)
    dump(out/'dataset.json',report)
    dump(out/'progress.json',{'stage':'complete','items':len(records),'wall_seconds':report['wall_seconds']})


def injected_state(base_state, adapters, layers, jax, jnp):
    """Return independent NNX state; only selected VLM linear slices can change."""
    state = jax.tree.map(lambda x: x, base_state)
    node = state['PaliGemma']['llm']['layers']['mlp']['linear']
    delta = jnp.einsum('lir,lro->lio', adapters['a'], adapters['b'])
    indices = jnp.asarray(layers)
    before = node.value[indices]
    node.value = node.value.at[indices].set((before.astype(jnp.float32)+delta).astype(before.dtype))
    return state


def load_dataset(path):
    path = Path(path)
    meta = json.loads((path/'dataset.json').read_text())
    if meta['split'] != 'development':
        raise ValueError('Training on a held-out dataset is forbidden')
    validate_live_certificate(meta)
    spec = read_spec(path/'input-spec.json')
    if sha(path/'input-spec.json') != meta['stored_spec_sha256']:
        raise ValueError('Stored input specification checksum mismatch')
    if len(spec['items']) != len(meta['records']):
        raise ValueError('Prepared records differ from the input specification')
    examples = {'train': [], 'validation': []}
    for original, r in zip(spec['items'], meta['records'], strict=True):
        validate_live_metadata(original)
        if r.get('live_context_sha256') != original['live_context_sha256'] or r.get('live_batch_bucket') != original['live_batch_bucket']:
            raise ValueError('Live replay certificate context differs from the frozen input specification')
        if (original['split'], str(Path(original['obs_path']).resolve()), original['decision_index']) != (r['split'],r['obs_path'],r['decision_index']):
            raise ValueError('Prepared split or state differs from the input specification')
        f = path/r['file']
        if sha(f) != r['sha256']:
            raise ValueError(f'Dataset checksum mismatch: {f}')
        with np.load(f,allow_pickle=False) as z:
            inputs = unflatten({k[len('input/'):]:z[k].copy() for k in z.files if k.startswith('input/')})
            target = z['teacher_actions'].copy()
            robot_target = z['teacher_actions_physical'].copy()
            live_error = physical_errors(z['live_replay_actions_physical'],z['recorded_actions_physical'])
        if live_error['max_abs_error'] != r['live_replay_max_abs_error'] or live_error['bit_exact'] != r['live_replay_bit_exact']:
            raise ValueError('Stored live replay arrays differ from their exact replay certificate')
        if not np.isfinite(target).all() or not np.isfinite(robot_target).all():
            raise ValueError('Nonfinite action target')
        examples[r['split']].append((inputs,target,robot_target,r))
    if not all(examples.values()):
        raise ValueError('Training and validation examples required')
    return meta,examples


def export_workspace(out, layers, base_ws, base_weights, merged_weights):
    from weightsmith.lab.openpi_vlm_full import validate_workspace
    cfg = json.loads((base_ws/'workspace.json').read_text())
    ws = out/'workspace'
    ws.mkdir()
    (ws/'base').symlink_to(base_ws/'base', target_is_directory=True)
    (ws/'edits').mkdir()
    selected = {f'language.blocks.{l:02d}.mlp.linear':cfg['tensors'][f'language.blocks.{l:02d}.mlp.linear'] for l in layers}
    cfg.update(tensors=selected,max_changed_fraction=1.0,max_changed_elements=cfg['total_parameters_and_buffers'],
               editable_parameters=sum(e['numel'] for e in selected.values()),
               edit_scope='action-supervised LoRA; selected VLM MLP down projections; merged bf16 weights')
    dump(ws/'workspace.json',cfg)
    for i,l in enumerate(layers):
        entry=selected[f'language.blocks.{l:02d}.mlp.linear']
        base=np.load(base_ws/'base'/entry['file'],allow_pickle=False)
        if not np.array_equal(base, np.asarray(base_weights[l], dtype=np.float32)):
            raise ValueError(f'Checkpoint and workspace base differ at layer {l}')
        # Serialize the exact bf16 tensors evaluated in JAX, avoiding NumPy/JAX
        # reduction differences near a bf16 rounding boundary.
        np.save(ws/'edits'/entry['file'],np.asarray(merged_weights[l], dtype=np.float32),allow_pickle=False)
    return validate_workspace(ws)


def train(args):
    meta, examples=load_dataset(args.dataset)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    settings=vars(args).copy();settings.pop('func',None)
    dump(out/'settings.json',settings)
    import jax
    import jax.numpy as jnp
    import optax
    from flax import nnx
    from openpi.models import model as om
    from weightsmith import config as C
    if sha(C.BASE_WS/'workspace.json') != meta['base_workspace_manifest_sha256']:
        raise ValueError('Workspace identity differs from teacher dataset')
    start=time.time();policy=load_policy();graph,base_state=nnx.split(policy._model)
    layers=tuple(args.layers)
    packed=base_state['PaliGemma']['llm']['layers']['mlp']['linear'].value
    rng=np.random.default_rng(args.seed)
    # Standard zero-output LoRA initialization: random A, zero B.
    adapters={'a':jnp.asarray(rng.normal(0,.01,(len(layers),packed.shape[1],args.rank)).astype(np.float32)),
              'b':jnp.zeros((len(layers),args.rank,packed.shape[2]),dtype=jnp.float32)}
    optimizer=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(args.lr,weight_decay=args.weight_decay))
    opt_state=optimizer.init(adapters)
    def loss_fn(a,state,observation,targets,key):
        model=nnx.merge(graph,injected_state(state,a,layers,jax,jnp))
        # train=False prevents image augmentation and preserves the recorded inputs.
        return jnp.mean(model.compute_loss(key,observation,targets,train=False))
    value_grad=jax.value_and_grad(loss_fn)
    @jax.jit
    def step(a,opt,state,observation,targets,key):
        loss,grad=value_grad(a,state,observation,targets,key)
        updates,opt=optimizer.update(grad,opt,a)
        return optax.apply_updates(a,updates),opt,loss,optax.global_norm(grad)
    evaluate=jax.jit(loss_fn)
    @jax.jit
    def sample(a,state,observation,seed):
        model=nnx.merge(graph,injected_state(state,a,layers,jax,jnp))
        return model.sample_actions(jax.random.key(seed),observation)
    def batch(example):
        inp,tgt,_,_=example
        return om.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],inp)),jnp.asarray(tgt)[None]
    validation=[batch(x) for x in examples['validation']]
    def val(a):
        values=[]
        for i,(obs,tgt) in enumerate(validation):
            values.append(float(evaluate(a,base_state,obs,tgt,jax.random.key(args.seed+100000+i))))
        return float(np.mean(values))
    def action_errors(a):
        records=[]
        for inp,tgt,robot_target,record in examples['validation']:
            observation=om.Observation.from_dict(jax.tree.map(lambda x:jnp.asarray(x)[None],inp))
            predicted=np.asarray(sample(a,base_state,observation,record['policy_seed'])[0])
            robot=policy._output_transform({'state':inp['state'],'actions':predicted})['actions']
            error=predicted.astype(np.float64)-tgt
            robot_error=robot.astype(np.float64)-robot_target
            records.append({'item_id':record['item_id'],'role':record.get('role','repair'),
                            'normalized_action_mse':float(np.mean(error**2)),
                            'robot_action_mse':float(np.mean(robot_error**2)),
                            'robot_action_mae':float(np.mean(np.abs(robot_error))),
                            'robot_action_max_abs':float(np.max(np.abs(robot_error)))})
        metric_keys=['normalized_action_mse','robot_action_mse','robot_action_mae','robot_action_max_abs']
        return {'items':records,'mean':{key:float(np.mean([r[key] for r in records])) for key in metric_keys},
                'by_role':{role:{key:float(np.mean([r[key] for r in records if r['role']==role])) for key in metric_keys}
                           for role in sorted({r['role'] for r in records})},
                'note':'Fixed teacher sampling seeds on development-validation states; not closed-loop success. Robot MSE mixes joint and gripper coordinates.'}
    initial=val(adapters)
    initial_actions=action_errors(adapters)
    print(f'initial validation flow loss {initial:.7g}',flush=True)
    history=[]
    for i in range(args.steps):
        item=examples['train'][int(rng.integers(len(examples['train'])))];obs,tgt=batch(item)
        adapters,opt_state,loss,gn=step(adapters,opt_state,base_state,obs,tgt,jax.random.key(args.seed+i))
        loss=float(loss);gn=float(gn)
        if not np.isfinite(loss+gn):
            raise ValueError('Nonfinite LoRA loss/gradient; no candidate exported')
        if i==0 or (i+1)%args.log_every==0 or i+1==args.steps:
            rec={'step':i+1,'train_flow_loss':loss,'gradient_norm':gn,'wall_seconds':time.time()-start}
            history.append(rec)
            dump(out/'progress.json',{'stage':'training',**rec,'planned_steps':args.steps})
            print(json.dumps(rec),flush=True)
    final=val(adapters)
    final_actions=action_errors(adapters)
    factors=jax.tree.map(np.asarray,adapters)
    np.savez_compressed(out/'adapters.npz',**factors,layers=np.asarray(layers))
    merged_state=injected_state(base_state,adapters,layers,jax,jnp)
    merged=export_workspace(out,layers,C.BASE_WS,packed,merged_state['PaliGemma']['llm']['layers']['mlp']['linear'].value)
    mem=jax.devices()[0].memory_stats() or {}
    report={'method':'teacher-action LoRA distillation','objective':'native flow matching on normalized teacher action chunks',
            'reuse_or_new':'new gradient optimization; reused pretrained checkpoint',
            'cohort_id':meta['cohort_id'],'dataset_manifest_sha256':sha(Path(args.dataset)/'dataset.json'),
            'source_code_sha256':sha(__file__),'rank':args.rank,'layers':list(layers),'steps':args.steps,
            'learning_rate':args.lr,'weight_decay':args.weight_decay,'seed':args.seed,
            'adapters_sha256':sha(out/'adapters.npz'),
            'train_inputs':len(examples['train']),'validation_inputs':len(examples['validation']),
            'initial_validation_flow_loss':initial,'final_validation_flow_loss':final,
            'initial_validation_actions':initial_actions,'final_validation_actions':final_actions,
            'trainable_parameters':sum(v.size for v in factors.values()),'wall_seconds':time.time()-start,
            'device':str(jax.devices()[0]),'memory_stats':mem,'history':history,'workspace_validation':merged,
            'closed_loop_evaluation':'not performed by this training command'}
    dump(out/'training.json',report);dump(out/'progress.json',{'stage':'complete','wall_seconds':report['wall_seconds']})
    print(json.dumps({k:v for k,v in report.items() if k not in ['history','workspace_validation']},indent=2),flush=True)


def smoke(args):
    """One synthetic forward/backward, never a robot or policy-success result."""
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    import jax
    import jax.numpy as jnp
    from flax import nnx
    from openpi.training import config as training_config
    import optax
    from weightsmith.locks import GpuLock
    t0=time.time()
    with GpuLock(args.gpu):
        policy=load_policy();graph,state=nnx.split(policy._model)
        cfg=training_config.get_config('pi05_droid_jointpos').model
        obs=cfg.fake_obs(batch_size=1)
        targets=jnp.zeros((1,policy._model.action_horizon,policy._model.action_dim),dtype=jnp.float32)
        packed=state['PaliGemma']['llm']['layers']['mlp']['linear'].value
        layers=tuple(range(8,17));r=32
        adapters={'a':jax.random.normal(jax.random.key(17),(len(layers),packed.shape[1],r))*.01,
                  'b':jnp.zeros((len(layers),r,packed.shape[2]))}
        def loss(a,base,observation,target):
            model=nnx.merge(graph,injected_state(base,a,layers,jax,jnp))
            return jnp.mean(model.compute_loss(jax.random.key(18),observation,target,train=False))
        vg=jax.jit(jax.value_and_grad(loss))
        print('SMOKE: model loaded; compiling one synthetic gradient step',flush=True)
        before,grad=vg(adapters,state,obs,targets)
        before=float(before);norm=float(optax.global_norm(grad))
        assert np.isfinite(before+norm) and norm>0
        mem=jax.devices()[0].memory_stats() or {}
        report={'kind':'synthetic compatibility smoke; not a policy experiment',
                'loss':before,'gradient_norm':norm,'rank':r,'layers':list(layers),
                'wall_seconds':time.time()-t0,'memory_stats':mem,
                'trainable_parameters':sum(x.size for x in adapters.values()),
                'jax_version':jax.__version__,'device':str(jax.devices()[0])}
        dump(out/'smoke.json',report);print(json.dumps(report,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--spec',required=True);prep.add_argument('--output',required=True)
    prep.set_defaults(func=prepare)
    t=sub.add_parser('train');t.add_argument('--dataset',required=True);t.add_argument('--output',required=True)
    t.add_argument('--steps',type=int,default=200);t.add_argument('--rank',type=int,default=32)
    t.add_argument('--lr',type=float,default=1e-3);t.add_argument('--weight-decay',type=float,default=0.)
    t.add_argument('--seed',type=int,default=17);t.add_argument('--layers',type=int,nargs='+',default=list(range(8,17)))
    t.add_argument('--log-every',type=int,default=10);t.set_defaults(func=train)
    sm=sub.add_parser('smoke');sm.add_argument('--output',required=True);sm.add_argument('--gpu',type=int,required=True);sm.set_defaults(func=smoke)
    prep.add_argument('--gpu',type=int,required=True);t.add_argument('--gpu',type=int,required=True)
    a=p.parse_args()
    if a.command=='train' and (a.steps<1 or a.rank<1 or not a.layers or min(a.layers)<0 or max(a.layers)>16 or len(set(a.layers))!=len(a.layers)):
        p.error('Positive steps/rank and unique VLM layers 0–16 are required')
    if a.command=='train' and (a.lr<=0 or a.weight_decay<0 or a.log_every<1):
        p.error('Positive learning rate/log interval and nonnegative weight decay are required')
    if os.environ.get('CUDA_VISIBLE_DEVICES',str(a.gpu)) != str(a.gpu):
        p.error('--gpu must match the sole physical GPU in CUDA_VISIBLE_DEVICES')
    os.environ['CUDA_VISIBLE_DEVICES']=str(a.gpu)
    os.environ.setdefault('XLA_PYTHON_CLIENT_MEM_FRACTION','.90')
    if a.command=='smoke':
        a.func(a)
    else:
        from weightsmith.locks import GpuLock
        with GpuLock(a.gpu):
            a.func(a)

if __name__=='__main__':
    main()
