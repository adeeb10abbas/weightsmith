eps = episodes()
print(len(eps))
e = eps[0]
print({k: (str(v)[:200]) for k, v in e.items()} if isinstance(e, dict) else e)
eid = e['id'] if isinstance(e, dict) and 'id' in e else e.get('eid') if isinstance(e, dict) else None
print('eid', eid)
o = obs(eid)
print(type(o))
if isinstance(o, dict):
    for k, v in o.items():
        print(k, getattr(v, 'shape', None), getattr(v, 'dtype', None), str(v)[:120] if not hasattr(v, 'shape') or v.size < 20 else '')
