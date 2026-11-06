import numpy as np, json
eps = episodes(cand="base")
print("n episodes", len(eps))
e0 = eps[0]
print("keys", list(e0.keys()) if isinstance(e0, dict) else type(e0))
short = {"GRIPPER_FULLY_CLOSED": "FC", "WRONG_OBJECT_GRABBED_FAILURE": "G", "OBJECT_BUMPED": "B",
         "STACKED_SUCCESS": "S"}
for e in sorted(eps, key=lambda e: e["eid"]):
    evs = e.get("events", [])
    s = []
    for ev in evs:
        nm = short.get(ev["name"], ev["name"])
        info = ev.get("info", "")
        tag = ""
        if nm == "G":
            tag = info.split("'")[1].replace("rubiks_cube", "c")
        elif nm == "S":
            tag = "1" if "1/2" in info else "2"
        elif nm == "B":
            tag = info.split("'")[1].replace("rubiks_cube", "c") + info.split("nudged")[1].strip()
        s.append(f"{ev['step']}:{nm}{tag}")
    print(e["eid"][-6:], "OK " if e["success"] else "FAIL", e["steps"], " ".join(s))
o = obs(e0["eid"])
print(type(o))
if isinstance(o, dict):
    for k, v in o.items():
        a = np.asarray(v)
        print(k, a.shape, a.dtype)
