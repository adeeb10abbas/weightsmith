import json
fk = json.load(open('/work/notes/fk_base.json'))
cats = {'HOLD': [], 'EMPTY': [], 'OPENLOW': [], 'OPENHIGH': []}
for eid, v in sorted(fk.items()):
    rows = v['rows']
    for r in rows:
        s, x, y, z, g, a0, am = r
        if 0.6 <= g <= 0.95:
            cats['HOLD'].append((eid, s))
        elif g >= 0.97:
            cats['EMPTY'].append((eid, s))
        elif g < 0.05 and z <= 0.10:
            cats['OPENLOW'].append((eid, s))
        elif g < 0.05 and z >= 0.18 and s > 0:
            cats['OPENHIGH'].append((eid, s))
for k, v in cats.items():
    print(k, len(v))
rng = np.random.RandomState(0)
want = {'HOLD': 36, 'EMPTY': 14, 'OPENLOW': 24, 'OPENHIGH': 16}
items = []; labels = []
for k, n in want.items():
    v = cats[k]
    if len(v) > n:
        # spread across episodes: stride sample
        idx = np.linspace(0, len(v) - 1, n).round().astype(int)
        v = [v[i] for i in idx]
    for eid, s in v:
        items.append({'episode': eid, 'step': int(s)}); labels.append(k)
json.dump({'items': items, 'labels': labels}, open('/work/notes/items_a1.json', 'w'))
spec = {"id": "a1-states", "cand": "base", "items": items, "action_layers": [17],
        "denoise_steps": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], "action_tokens": "mean"}
print(len(items))
print(json.dumps(spec))
