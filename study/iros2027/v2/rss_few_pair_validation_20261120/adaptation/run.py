"""Construct three fixed-recipe Weightsmith edits after fresh source qualification."""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time
import numpy as np

ADAPTATION = Path(__file__).resolve().parent
sys.path.insert(0, str(ADAPTATION.parent))
from common import HERE, REPO, SCALING, canonical, goal_root, protocol, read, save, sha, verify
from adaptation.sources import qualified
from adaptation.recipe import ARRAYS, LAYERS, capture_alignment, pooled_key_values, fit, merge
from adaptation.compatibility import ensure as ensure_compatibility
from weightsmith import config as C, distill


def run_checked(command, log, *, env=None, input_text=None):
    with Path(log).open('x') as stream:
        result = subprocess.run(command, cwd=REPO, env=env, input=input_text, text=True,
                                stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError(f'Construction stage failed ({result.returncode}); partial evidence preserved at {log}')


def addendum(out):
    path = out/'protocol-addendum.json'
    save(path, {'cohort_id': protocol()['cohort_id'], 'protocol_sha256': sha(HERE/'protocol.json'),
                'field_clarified': 'edit.alignment', 'status': 'frozen before fresh adaptation',
                'replacement': 'Three opening template positions plus a fixed37-position window beginning at the common State anchor; paired token IDs and validity flags must match at all40positions across all9source states.',
                'reason': 'Audit of the archived N=3 recipe found a fixed window, not an all-valid suffix. Across its12source states the40positions contain38,39,or40valid tokens. Replication preserves padding/truncation and all72key rows.',
                'scope': 'Alignment description correction only; collection, scoring, source selection, held-out design, layers, ridge and merge multiplier remain frozen.',
                'historical_source': str(REPO/'projects/rss-repair-scaling-20261119/edits/n3/construction.json'),
                'historical_construction_sha256': '68520cb055690bbd4f42fc6cfb2cbab8b100ec54ddcab0bdfcc234c2d4c93fc2',
                'fresh_heldout_outcomes_used': False}, exclusive=True)
    return path


def prepare_capture(out, spec):
    episodes = out/'episodes'
    episodes.mkdir(exist_ok=True)
    for i, item in enumerate(spec['items']):
        episode = episodes/f'item-{i:04d}'
        episode.mkdir(exist_ok=True)
        path = episode/'obs.npz'
        if not path.exists():
            path.symlink_to(Path(item['obs_path']).resolve())
        if path.resolve() != Path(item['obs_path']).resolve():
            raise ValueError('Capture observation link changed')
        save(episode/'summary.json', {'prompt': item['recorded_prompt']}, exclusive=True)
    for which in ('original', 'teacher'):
        request = {'cand': 'base', 'episodes_dir': str(episodes), 'vlm_layers': list(LAYERS),
                   'image_tokens': True, 'prompt_tokens': list(range(96)), 'action_layers': [0],
                   'denoise_steps': [0], 'action_tokens': 'mean', 'action_attention': False,
                   'prompt': spec[which+'_prompt'],
                   'items': [{'episode': f'item-{i:04d}', 'k': item['decision_index'], 'prompt': item[which+'_prompt']}
                             for i, item in enumerate(spec['items'])]}
        save(out/f'{which}-capture-spec.json', request, exclusive=True)


def capture(out, gpu):
    from weightsmith.locks import GpuLock as SharedGpuLock
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), XLA_PYTHON_CLIENT_PREALLOCATE='false',
               XLA_PYTHON_CLIENT_MEM_FRACTION=C.ACTS_MEM_FRACTION, JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE))
    (out/'captures').mkdir(exist_ok=True)
    for which in ('original', 'teacher'):
        path, receipt = out/'captures'/which, out/'captures'/f'{which}-receipt.json'
        request = out/f'{which}-capture-spec.json'
        if receipt.exists():
            record = read(receipt)
            verify(record['artifacts'])
            if record['spec_sha256'] != sha(request):
                raise ValueError('Capture request changed')
            continue
        if path.exists():
            raise ValueError(f'Partial capture preserved; no automatic retry: {path}')
        start = time.time()
        with SharedGpuLock(gpu):
            run_checked([str(C.OPENPI_PY), str(ADAPTATION/'capture_worker.py'), '--output', str(path)],
                        out/'captures'/f'{which}.log', env=env, input_text=request.read_text())
        save(receipt, {'spec_sha256': sha(request), 'wall_seconds': time.time()-start,
                       'artifacts': {str(p): sha(p) for p in (path/'acts.npz', path/'index.json', ADAPTATION/'capture_worker.py', REPO/'weightsmith/acts_worker.py')}})


def validate_capture(out, spec, dataset):
    data = read(dataset/'dataset.json')
    if (data.get('arm') != 'matched_decision' or data.get('train_inputs') != 9
            or data.get('correction_count') != 9 or data.get('validation_inputs') != 0
            or read(dataset/'input-spec.json') != spec):
        raise ValueError('Native replay targets do not match the nine qualified source states')
    gate = data.get('live_replay_gate', {})
    if gate.get('status') != 'pass' or gate.get('tolerance') != 1e-6 or gate.get('maximum_error', float('inf')) > 1e-6:
        raise ValueError('Missing exact fixed16 source replay certificate')
    verify(data['source_artifacts'])
    verify(data['live_replay_context_hashes'])
    verify(data['adaptation_driver_artifacts'])
    if sha(dataset/'teacher-actions.npz') != data['teacher_actions_sha256']:
        raise ValueError('Native physical action export changed')
    indices, arrays = {}, {}
    for which in ('original', 'teacher'):
        path = out/'captures'/which
        indices[which] = read(path/'index.json')
        if indices[which].get('prompt_token_slots') != list(range(96)):
            raise ValueError('Wrong fresh capture token support')
        with np.load(path/'acts.npz', allow_pickle=False) as z:
            arrays[which] = {key: z[key].copy() for key in (*ARRAYS, 'actions', 'recorded_actions')}
        if not all(np.isfinite(value).all() for value in arrays[which].values()):
            raise ValueError('Nonfinite paired capture')
    slots, alignment = capture_alignment(indices['original'], indices['teacher'], expected_count=9)
    rows = []
    for i, (item, record) in enumerate(zip(spec['items'], data['records'], strict=True)):
        for key in ('item_id', 'obs_path', 'decision_index', 'pair_id', 'live_context_sha256'):
            if item[key] != record[key]:
                raise ValueError('Native and instrumented source identities differ')
        with np.load(item['obs_path'], allow_pickle=False) as z:
            seed, step = int(z['seed'][item['decision_index']]), int(z['step'][item['decision_index']])
        for which in ('original', 'teacher'):
            entry = indices[which]['items'][i]
            expected = {'episode': f'item-{i:04d}', 'k': item['decision_index'], 'seed': seed,
                        'sim_step': step, 'prompt': item[which+'_prompt']}
            if any(entry[k] != v for k, v in expected.items()):
                raise ValueError('Instrumented input identity differs from the selected source')
        target = dataset/record['file']
        if sha(target) != record['sha256']:
            raise ValueError('Native target array changed')
        with np.load(target, allow_pickle=False) as z:
            teacher, live, recorded = z['teacher_actions_physical'], z['live_replay_actions_physical'], z['recorded_actions_physical']
        parity = float(np.max(np.abs(arrays['teacher']['actions'][i]-teacher)))
        replay = float(np.max(np.abs(live-recorded)))
        if not np.isfinite(parity+replay) or parity > .01 or replay > 1e-6:
            raise ValueError(f'Unqualified native/instrumented replay: teacher={parity}, live={replay}')
        rows.append({'pair_id': item['pair_id'], 'teacher_max_abs_error': parity,
                     'live_replay_max_abs_error': replay,
                     'singleton_live_original_max_abs_error': float(np.max(np.abs(arrays['original']['actions'][i]-recorded))),
                     **alignment[i]})
    save(out/'parity.json', {'status': 'pass', 'teacher_tolerance': .01, 'source_replay_tolerance': 1e-6,
                            'slot_map': slots, 'records': rows,
                            'note': 'Exact fixed16 source replay and separate fresh singleton native/instrumented teacher parity. Singleton/live original differences are diagnostic only.'}, exclusive=True)
    return arrays, slots


def solve(out, arrays, slots, manifest_path, manifest, gate_path):
    from weightsmith.lab.openpi_vlm_full import validate_workspace
    dest = out/'weightsmith'
    dest.mkdir(exist_ok=True)
    base = Path(C.BASE_WS).resolve()
    base_config = read(base/'workspace.json')
    candidates = []
    for group in manifest['groups']:
        root = dest/group['group_id']
        report = root/'construction.json'
        if report.exists():
            previous = read(report)
            verify(previous['artifacts'])
            if previous['source_indices'] != group['indices'] or previous['source_manifest_sha256'] != sha(manifest_path):
                raise ValueError('Completed edit belongs to different source group')
            candidates.append(previous)
            continue
        root.mkdir(exist_ok=False)
        workspace = root/'workspace'
        workspace.mkdir()
        (workspace/'edits').mkdir()
        (workspace/'scratch').mkdir()
        (workspace/'base').symlink_to(base/'base', target_is_directory=True)
        selected = {distill.MLP_DOWN.format(layer): base_config['tensors'][distill.MLP_DOWN.format(layer)] for layer in LAYERS}
        config = {**base_config, 'tensors': selected, 'max_changed_fraction': 1.,
                  'max_changed_elements': base_config['total_parameters_and_buffers'], 'include_action': False,
                  'editable_parameters': sum(value['numel'] for value in selected.values()),
                  'edit_scope': 'Closed-form Weightsmith correction; VLM MLP down projections0–16 only',
                  'frozen': ['vision encoder', 'VLM embeddings and other projections', 'action expert and action heads']}
        save(workspace/'workspace.json', config)
        start = time.time()
        kv = pooled_key_values(arrays['original'], arrays['teacher'], group['indices'], slots)
        layers = []
        for layer in LAYERS:
            delta, stats = fit(*kv[layer])
            name = distill.MLP_DOWN.format(layer)
            entry = selected[name]
            base_path = base/'base'/entry['file']
            weight = np.load(base_path, allow_pickle=False)
            edited = merge(weight, delta)
            target = workspace/'edits'/entry['file']
            np.save(target, edited, allow_pickle=False)
            layers.append({'layer': layer, 'tensor': name, **stats,
                           'changed_values': int(np.count_nonzero(edited != weight)),
                           'relative_delta_l2': float(np.linalg.norm(delta)/max(np.linalg.norm(weight), 1e-30)),
                           'base_sha256': sha(base_path), 'edited_sha256': sha(target)})
            print({'group': group['group_id'], 'layer': layer, 'changed_values': layers[-1]['changed_values']}, flush=True)
        artifacts = {str(path): sha(path) for path in (workspace/'workspace.json', *sorted((workspace/'edits').glob('*.npy')))}
        candidate = {'candidate_id': f'weightsmith_{group["group_id"]}', 'group_id': group['group_id'],
                     'source_indices': group['indices'], 'pair_ids': group['pair_ids'], 'source_pairs': 3,
                     'workspace': str(workspace), 'status': 'complete', 'source_manifest': str(manifest_path),
                     'source_manifest_sha256': sha(manifest_path), 'parity_sha256': sha(out/'parity.json'),
                     'protocol_addendum_sha256': sha(out/'protocol-addendum.json'),
                     'compatibility_gate_sha256': sha(gate_path), 'layers': layers, 'artifacts': artifacts,
                     'workspace_validation': validate_workspace(workspace), 'wall_seconds': time.time()-start,
                     'heldout_used': False, 'development_outcomes_used': False}
        save(report, candidate, exclusive=True)
        candidates.append(candidate)
    gate_artifacts = [manifest_path, out/'protocol-addendum.json', out/'parity.json', out/'construction-seal.json',
                      gate_path, out/'teacher-parity/dataset.json', out/'teacher-parity/native-receipt.json']
    for which in ('original', 'teacher'):
        gate_artifacts.extend([out/f'{which}-capture-spec.json', out/'captures'/f'{which}-receipt.json',
                               out/'captures'/which/'index.json', out/'captures'/which/'acts.npz'])
    registry = {'status': 'complete', 'cohort_id': protocol()['cohort_id'], 'goal_id': manifest['goal_id'],
                'source_manifest': str(manifest_path), 'source_manifest_sha256': sha(manifest_path),
                'captures': {which: str(out/'captures'/which) for which in ('original', 'teacher')},
                'gate': {'historical_compatibility': str(gate_path), 'native_and_instrumented_parity': str(out/'parity.json')},
                'artifacts': {str(path): sha(path) for path in gate_artifacts}, 'candidates': candidates,
                'heldout_used': False, 'development_outcomes_used': False}
    save(dest/'candidates.json', registry, exclusive=True)
    return registry


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--goal', required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    manifest_path, manifest, spec_path, spec, cache = qualified(args.goal)
    out = goal_root(args.goal)/'adaptation'
    out.mkdir(exist_ok=True)
    status_path = out/'status.json'
    if status_path.exists() and read(status_path).get('status') == 'stopped':
        raise ValueError('A stopped adaptation requires a separately recorded recovery decision; no automatic retry')
    addendum_path = addendum(out)
    compatibility_path = ensure_compatibility()
    paths = [*sorted(ADAPTATION.glob('*.py')), HERE/'common.py', HERE/'protocol.json',
             SCALING/'lora/fresh_lora.py', SCALING.parent/'rss_weightsmith_goals/adaptation/native_targets.py',
             SCALING.parent/'rss_repair_comparison/lora/train_lora.py',
             SCALING.parent/'rss_repair_comparison/runtime/batched_server.py',
             spec_path, manifest_path, addendum_path, compatibility_path,
             REPO/'weightsmith/distill.py', REPO/'weightsmith/acts_worker.py', REPO/'weightsmith/lab/openpi_probe.py',
             C.BASE_WS/'workspace.json', C.CHECKPOINT/'assets/droid/norm_stats.json']
    seal = {'source_manifest': str(manifest_path), 'artifacts': {str(path.resolve()): sha(path) for path in paths}}
    save(out/'construction-seal.json', seal, exclusive=True)
    verify(seal['artifacts'])
    if status_path.exists() and read(status_path).get('status') == 'complete':
        registry = read(out/'weightsmith/candidates.json')
        verify(registry['artifacts'])
        for candidate in registry['candidates']:
            verify(candidate['artifacts'])
        print({'status': 'complete', 'registry': str(out/'weightsmith/candidates.json')})
        return
    status = {'status': 'running', 'goal_id': args.goal, 'stage': 'native_source_replay', 'started_unix': time.time()}
    save(status_path, status)
    try:
        dataset = out/'teacher-parity'
        receipt = dataset/'native-receipt.json'
        if receipt.exists():
            verify(read(receipt)['artifacts'])
        else:
            if dataset.exists():
                raise ValueError('Partial native preparation preserved; no automatic retry')
            run_checked([str(C.OPENPI_PY), str(ADAPTATION/'native.py'), '--goal', args.goal,
                         '--output', str(dataset), '--gpu', str(args.gpu)], out/'teacher-parity.log')
            meta = read(dataset/'dataset.json')
            save(receipt, {'status': 'complete', 'spec_sha256': sha(spec_path),
                           'artifacts': {str(path): sha(path) for path in
                                         [dataset/'dataset.json', dataset/'input-spec.json', dataset/'teacher-actions.npz',
                                          *[dataset/record['file'] for record in meta['records']]]}}, exclusive=True)
        verify(seal['artifacts'])
        verify(cache['hashes'])
        prepare_capture(out, spec)
        status['stage'] = 'paired_activation_capture'
        save(status_path, status)
        capture(out, args.gpu)
        verify(seal['artifacts'])
        arrays, slots = validate_capture(out, spec, dataset)
        status['stage'] = 'closed_form_solve'
        save(status_path, status)
        result = solve(out, arrays, slots, manifest_path, manifest, compatibility_path)
        verify(seal['artifacts'])
        verify(cache['hashes'])
        status.update(status='complete', stage='complete', candidates=len(result['candidates']), completed_unix=time.time())
        print({'status': 'complete', 'registry': str(out/'weightsmith/candidates.json')})
    except BaseException as exc:
        status.update(status='stopped', reason=repr(exc))
        raise
    finally:
        save(status_path, status)


if __name__ == '__main__':
    main()
