import json, re

eps = episodes(cand="base")
print(len(eps))
e0 = eps[0]
for k, v in e0.items():
    s = repr(v)
    print(k, ":", s[:300])
print()
for e in eps:
    ev = e.get('events', [])
    toks = []
    for x in ev:
        n = x['name']; info = x.get('info', '')
        if n == 'GRIPPER_FULLY_CLOSED':
            toks.append(f"C@{x['step']}")
        elif n == 'WRONG_OBJECT_GRABBED_FAILURE':
            m = re.search(r"grabbed: '([^']+)'", info)
            toks.append(f"W[{m.group(1) if m else '?'}]@{x['step']}")
        elif n == 'OBJECT_BUMPED':
            m = re.search(r"bumped: '([^']+)' nudged ([0-9.]+)", info)
            toks.append(f"B[{m.group(1)},{m.group(2)}]@{x['step']}" if m else f"B@{x['step']}")
        elif n == 'OBJECT_GRABBED_SUCCESS':
            toks.append(f"GRAB@{x['step']}")
        elif n == 'STACKED_SUCCESS':
            toks.append(f"STACK@{x['step']}")
        else:
            toks.append(f"{n}@{x['step']}")
    print(e.get('eid'), 'OK ' if e.get('success') else 'FAIL', ' '.join(toks))
o = obs(e0.get("eid"))
for k, v in o.items():
    print(k, getattr(v, "shape", None), getattr(v, "dtype", None))
