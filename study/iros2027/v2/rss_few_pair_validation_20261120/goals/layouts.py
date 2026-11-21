"""Fresh deterministic layout families; no policy or outcome access."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import numpy as np

try:
    from .registry import GOAL_IDS, get_goal, load_runtime_module
except ImportError:
    from registry import GOAL_IDS, get_goal, load_runtime_module

STAGES = ('collection', 'development', 'heldout')
COUNTS = {'collection': 48, 'development': 16, 'heldout': 16}


def default_seed(goal_id: str, split: str) -> int:
    if goal_id not in GOAL_IDS or split not in STAGES:
        raise ValueError((goal_id, split))
    return 2026112001 + GOAL_IDS.index(goal_id) * 100 + STAGES.index(split)


def _base_poses(goal_id: str) -> dict:
    goal = get_goal(goal_id)
    module = load_runtime_module(goal_id, 'generate_layouts.py')
    if goal_id == 'cube_behind':
        return module.stock_poses(goal['scene_path'], goal['objects'])
    return {name: {'pos': list(pos), 'quat': list(quat)}
            for name, (pos, quat) in module.POSES[goal['runtime_case']].items()}


def build_layouts(goal_id: str, split: str, n: int | None = None,
                  seed: int | None = None) -> dict:
    if split not in STAGES:
        raise ValueError(f'Unknown layout stage: {split!r}')
    n = COUNTS[split] if n is None else n
    if type(n) is not int or n < 1 or n % 16:
        raise ValueError('Layout count must be a positive multiple of sixteen')
    seed = default_seed(goal_id, split) if seed is None else seed
    if type(seed) is not int or seed < 0:
        raise ValueError('Seed must be a nonnegative integer')
    goal = get_goal(goal_id)
    base = _base_poses(goal_id)
    rng = np.random.default_rng(seed)
    rows = []
    for index in range(n):
        shift = rng.uniform(-.012, .012, size=2)
        poses = {}
        for name, pose in base.items():
            pos = np.array(pose['pos'], dtype=float)
            pos[:2] += shift + rng.uniform(-.007, .007, size=2)
            if split == 'heldout':
                if goal_id in ('temporal', 'cube_behind'):
                    if index % 2 and name in ('rubiks_cube', 'banana'):
                        other = 'banana' if name == 'rubiks_cube' else 'rubiks_cube'
                        pos[0] = base[other]['pos'][0] + shift[0] + rng.uniform(-.007, .007)
                    if name == 'bowl':
                        pos[0] += (.04 + .02 * rng.random()) * (1 if index % 4 < 2 else -1)
                else:
                    # Exchange near/far starts in half the cases, maintaining a
                    # large X separation and nearly equal Y: neither left nor
                    # right is true initially. A common Y offset changes the
                    # arrangement while retaining collision clearance.
                    if index % 2:
                        other = 'raisin_box' if name == 'mustard_bottle' else 'mustard_bottle'
                        pos[0] = base[other]['pos'][0] + shift[0] + rng.uniform(-.007, .007)
                    pos[1] += .03 if index % 4 < 2 else -.03
            poses[name] = {'pos': pos.round(10).tolist(), 'quat': list(pose['quat'])}
        rows.append({'id': f'WFP-{goal_id}-{split}-{index:03d}', 'poses': poses})
    asset = Path(goal['scene_path'])
    result = {'case': goal['runtime_case'], 'goal_id': goal_id,
              'split': 'heldout' if split == 'heldout' else 'development',
              'stage': split, 'generator_seed': seed,
              'objects': goal['objects'], 'settle_steps': 15, 'renders': 8,
              'asset': str(asset), 'asset_sha256': hashlib.sha256(asset.read_bytes()).hexdigest(),
              'layouts': rows,
              'generation': 'Fixed independent seeds and XY jitter. Held-out temporal/cube layouts swap cube and banana X starts in half and translate bowl X by 4-6cm. Held-out mustard layouts swap object X starts in half and add a common +/-3cm Y shift. No policy-based layout selection.',
              'scope': 'Changed arrangements of existing objects and scenes. Every layout still requires a policy-free reset gate for pose/stability and initially false goals.'}
    validate_layouts(result)
    return result


def validate_layouts(spec: dict) -> None:
    goal = get_goal(spec['goal_id'])
    if spec['case'] != goal['runtime_case'] or spec['objects'] != goal['objects']:
        raise ValueError('Layout case/objects disagree with selected goal')
    if spec.get('stage') not in STAGES:
        raise ValueError('Layout stage missing or invalid')
    expected_split = 'heldout' if spec['stage'] == 'heldout' else 'development'
    if spec.get('split') != expected_split:
        raise ValueError('Layout stage and evaluation split disagree')
    rows = spec['layouts']
    if not rows or len(rows) % 16 or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Layout identities incomplete or duplicated')
    for row in rows:
        if set(row['poses']) != set(goal['objects']):
            raise ValueError('Missing or extra object pose')
        for pose in row['poses'].values():
            xyz, quat = np.asarray(pose['pos']), np.asarray(pose['quat'])
            if xyz.shape != (3,) or quat.shape != (4,) or not np.isfinite(xyz).all() or not np.isfinite(quat).all():
                raise ValueError('Invalid pose geometry')
            if not np.isclose(np.linalg.norm(quat), 1., atol=.005):
                raise ValueError('Invalid orientation norm')


def write_layouts(path, spec: dict) -> None:
    """Write once, permitting only an identical existing artifact."""
    validate_layouts(spec)
    path = Path(path)
    text = json.dumps(spec, indent=2, allow_nan=False) + '\n'
    if path.exists():
        if path.read_text() != text:
            raise FileExistsError(f'Refusing to replace frozen layouts: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as f:
        f.write(text)
