import json
pts = [
    # approach (unrotated)
    ('base.base.s70012', 2), ('base.base.s70012', 4), ('base.base.s70007', 2), ('base.base.s70007', 3),
    ('base.base.s70000', 2), ('base.base.s70030', 3), ('base.base.s70018', 3), ('base.base.s70018', 5),
    # grasp-ready rotated low open
    ('base.base.s70005', 9), ('base.base.s70005', 10), ('base.base.s70005', 16), ('base.base.s70018', 25),
    ('base.base.s70018', 27), ('base.base.s70024', 24), ('base.base.s70024', 26), ('base.base.s70025', 23),
    # holding-ish
    ('base.base.s70005', 17), ('base.base.s70005', 21), ('base.base.s70024', 7), ('base.base.s70025', 26),
]
prompts = [None, 'Pick up the green box', 'Pick up the box of frozen vegetables', 'Pick up the green vegetable box']
items = []
for p in prompts:
    for eid, k in pts:
        it = {'episode': eid, 'k': k}
        if p is not None:
            it['prompt'] = p
        items.append(it)
spec = {'id': 'a1-prompts', 'cand': 'base', 'prompt_tokens': list(range(24)),
        'items': items, 'action_layers': [17], 'vlm_layers': [4, 8, 12, 17], 'denoise_steps': [0, 9], 'action_tokens': 'mean'}
print(json.dumps(spec))
