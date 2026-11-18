"""Deterministic, outcome-independent small XY variations of stock object poses."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np

SCENES = {'banana_left_plate': ('bagel_plate_banana_bowl', ['banana', 'plate_large', 'bowl'], 'N'), 'butter_behind_raisin': ('butter_raisin_box', ['butter', 'raisin_box'], 'U'), 'left_bowl_on_right': ('bowls_2_table', ['bowl_1', 'bowl_2'], 'L')}
SEEDS = {'development':202611182001,'heldout':202611182002}


def stock_poses(asset,names):
    text = Path(asset).read_text()
    poses = {}
    for name in names:
        match = re.search(r'(?m)^    def (?:Xform )?"'+re.escape(name)+r'"[^\n]*',text)
        if not match:
            raise ValueError(f'No stock root object {name}')
        block = text[match.start():].split('\n    }',1)[0]
        def values(pattern,n):
            hit = re.search(pattern,block)
            if not hit:
                raise ValueError(f'Missing stock transform {name}')
            result = [float(x.strip()) for x in hit[1].split(',')]
            if len(result) != n:
                raise ValueError(name)
            return result
        poses[name] = {'pos':values(r'(?:double|float)3 xformOp:translate = \(([^\n]+)\)',3),
                       'quat':values(r'quat[fd] xformOp:orient = \(([^\n]+)\)',4)}
    return poses


def build(case,split,asset_root):
    scene,names,prefix = SCENES[case]
    asset = Path(asset_root)/(scene+'.usda')
    poses = stock_poses(asset,names)
    seed = SEEDS[split]+100*list(SCENES).index(case)
    rng = np.random.default_rng(seed)
    layouts = []
    for index in range(4 if split=='development' else 8):
        shift = rng.uniform(-.007,.007,size=2)
        perturbed = {}
        for name,pose in poses.items():
            pos = np.array(pose['pos'])
            pos[:2] += shift+rng.uniform(-.003,.003,size=2)
            perturbed[name] = {'pos':pos.round(10).tolist(),'quat':pose['quat']}
        layouts.append({'id':f'{prefix}-{"D" if split=="development" else "H"}{index+1:02}','poses':perturbed})
    return {'case':case,'split':split,'generator_seed':seed,
            'generation':'Predeclared small XY shifts with stock Z/orientation, authored before policy evaluation; existing stock scenes, no outcome-selected geometry.',
            'asset':str(asset.resolve()),'asset_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),
            'objects':names,'settle_steps':15,'renders':8,'layouts':layouts}


if __name__ == '__main__':
    import sys
    sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
    from weightsmith import config
    dest = Path(__file__).resolve().parent/'layouts'
    dest.mkdir(exist_ok=True)
    for case in SCENES:
        for split in SEEDS:
            p = dest/f'{case}_{split}.json'
            text = json.dumps(build(case,split,config.ROBOLAB/'assets/scenes'),indent=2)+'\n'
            if p.exists() and p.read_text()!=text:
                raise RuntimeError(f'Refusing to replace changed layout {p}')
            p.write_text(text)
