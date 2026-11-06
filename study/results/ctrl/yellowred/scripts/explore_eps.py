import json
eps = episodes(cand="base")
print(len(eps))
e0 = eps[0]
print(type(e0))
if isinstance(e0, dict):
    for k, v in e0.items():
        s = repr(v)
        print(k, ":", s[:300])
o = obs(eps[0]["eid"] if isinstance(eps[0], dict) and "eid" in eps[0] else eps[0]["id"])
print(type(o))
if isinstance(o, dict):
    for k, v in o.items():
        try:
            print(k, getattr(v, "shape", None), getattr(v, "dtype", None))
        except Exception as ex:
            print(k, ex)
