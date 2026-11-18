"""Regenerate frozen layouts without model access or policy-outcome selection.

Uses known scene asset heights/orientations; all XY placements are new. Spatial
starts separate objects along robot X, making neither left nor right true.
This is small in-distribution pose variation, not a new-scene generalization claim.
"""
import json
from pathlib import Path
import numpy as np

POSES = {
    "temporal": {
        "rubiks_cube": ([.395, -.105, .0811058581], [.9999317945, -.0005057244, .0009222017, -.0116285298]),
        "banana": ([.530, -.105, .0681846961], [.9966595087, -.0005555861, -.0012671497, .08165821495]),
        "bowl": ([.465, .130, .0773495957], [.9994019883, -.0003871697, -.0004394573, .0345750991]),
    },
    "spatial": {
        "mustard_bottle": ([.350, .150, .0823110640], [.5523233904, -.0046616404, .0160821507, .8334617465]),
        "raisin_box": ([.495, .150, .02150323056], [.7115456714, -.0144917467, .7024691031, .00546625855]),
    },
}
SEEDS = {"development": 2026111701, "heldout": 2026111702}


def build(case, split):
    seed = SEEDS[split] + (100 if case == "spatial" else 0)
    rng = np.random.default_rng(seed)
    layouts = []
    for i in range(4 if split == "development" else 8):
        shift = rng.uniform(-.012, .012, size=2)
        poses = {}
        for name, (base, quat) in POSES[case].items():
            pos = np.array(base)
            pos[:2] += shift + rng.uniform(-.007, .007, size=2)
            poses[name] = {"pos": pos.round(10).tolist(), "quat": quat}
        layouts.append({"id": f"{case[0].upper()}-{'D' if split == 'development' else 'H'}{i+1:02}", "poses": poses})
    return {"case": case, "split": split, "generator_seed": seed,
            "generation": "fixed deterministic XY perturbations, no policy feedback",
            "objects": list(POSES[case]), "settle_steps": 15, "renders": 8, "layouts": layouts}


if __name__ == "__main__":
    dest = Path(__file__).resolve().parent / "layouts"
    dest.mkdir(exist_ok=True)
    for case in POSES:
        for split in SEEDS:
            path = dest / f"{case}_{split}.json"
            text = json.dumps(build(case, split), indent=2) + "\n"
            if path.exists() and path.read_text() != text:
                raise RuntimeError(f"Refusing to replace a changed frozen layout: {path}")
            path.write_text(text)
