"""CPU-only identities and frozen layouts for the fresh correction-scaling cohort."""
from __future__ import annotations
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
ROOT = REPO / 'projects/rss-repair-scaling-20261119'
PRIOR = HERE.parent / 'rss_repair_comparison'
POOL = HERE.parent / 'rss_goal_expansion_v2/runtime/single_attempt_pool.py'
sys.path.insert(0, str(REPO))

def read(path):
    return json.loads(Path(path).read_text())

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while block := f.read(8 << 20): h.update(block)
    return h.hexdigest()

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

def save(path, value, *, exclusive=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and path.exists():
        if read(path) != value: raise ValueError(f'Refusing to replace frozen content: {path}')
        return
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temp.replace(path)

def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec); sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

def protocol(): return read(HERE / 'protocol.json')

def layouts(kind):
    p = protocol(); source = load('scaling_source_layouts', PRIOR / 'runtime/generate_layouts.py')
    if kind == 'collection': n, seed = p['collection']['layouts'], p['collection']['generator_seed']
    elif kind in ('near', 'far'): n, seed = 16, p['evaluation'][kind + '_seed']
    else: raise ValueError(kind)
    rng = np.random.default_rng(seed); rows = []
    for i in range(n):
        shift = rng.uniform(-.012, .012, 2); placed = {}
        for name, (xyz, quat) in source.POSES['temporal'].items():
            pos = np.array(xyz); pos[:2] += shift + rng.uniform(-.007, .007, 2)
            if kind == 'far':
                # Prespecified changed arrangements: swap object sides in half,
                # and translate the bowl by 4--6 cm. No policy-based selection.
                if i % 2 and name in ('rubiks_cube', 'banana'):
                    pos[0] = (.530 if name == 'rubiks_cube' else .395) + shift[0] + rng.uniform(-.007, .007)
                if name == 'bowl': pos[0] += (.04 + .02 * rng.random()) * (1 if i % 4 < 2 else -1)
            placed[name] = {'pos': pos.round(10).tolist(), 'quat': quat}
        rows.append({'id': f'WSS-{kind}-{i:03d}', 'poses': placed})
    return {'case': 'temporal', 'split': 'development' if kind in ('collection', 'near') else 'heldout',
            'generator_seed': seed, 'objects': list(source.POSES['temporal']),
            'settle_steps': 15, 'renders': 8, 'layouts': rows,
            'generation': 'Fixed before policy access. Far arrangements swap object sides in half and translate bowl by 4--6 cm.' if kind == 'far' else 'Fixed small XY perturbations, no outcome selection.'}

def source_inventory():
    prior = load('scaling_prior_inventory', PRIOR / 'make_eval_plan.py')
    return sorted(set([*prior.runtime_sources(), POOL, HERE / 'protocol.json',
                       HERE / 'common.py', HERE / 'worker.py', HERE / 'collect.py',
                       HERE / 'prepare_sources.py', PRIOR / 'prepare_shared.py']))

def validate_collection_plan(plan):
    p = protocol(); c = p['collection']
    if plan.get('cohort_id') != p['cohort_id'] or plan.get('split') != 'development' or plan.get('inference_batching') != 'fixed16':
        raise ValueError('Wrong source cohort/split/inference mode')
    if plan['case'] != 'temporal' or plan['protocol_sha256'] != sha(HERE / 'protocol.json'):
        raise ValueError('Wrong frozen protocol')
    if len(plan['batches']) != 1: raise ValueError('Exactly one complete source batch per worker')
    b = plan['batches'][0]
    if b['cell'] not in ('original', 'teacher') or b['condition'] != 'base' or b.get('workspace') is not None:
        raise ValueError('Source collection must use unchanged policy')
    prompt = p['original_prompt'] if b['cell'] == 'original' else p['clarified_prompt']
    if b['effective_prompt'] != prompt or b['input_prompt'] != p['original_prompt'] or b['task'] != p['task'] or b['goal'] != p['goal']:
        raise ValueError('Source request or task differs')
    start = b['slot_start']
    if b['n'] != 16 or start % 16 or not c['slot_start'] <= start < c['slot_start'] + c['max_pairs']:
        raise ValueError('Source budget/slots differ')
    if b['layouts'] != str(ROOT / 'design/collection-layouts.json'):
        raise ValueError('Unexpected source layouts')

