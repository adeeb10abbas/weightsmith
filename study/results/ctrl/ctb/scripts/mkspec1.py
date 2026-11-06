import json
rel = [("s70001",22),("s70002",33),("s70003",23),("s70004",7),("s70007",18),("s70008",22),("s70011",4),("s70012",19),
       ("s70018",21),("s70021",25),("s70023",5),("s70024",22),("s70027",4),("s70028",4),("s70029",32),("s70031",5)]
hold = [("s70000",5),("s70025",5),("s70016",5),("s70009",8),("s70022",9),("s70017",4),("s70013",5),("s70020",5),("s70010",39),("s70030",7)]
miss = [("s70001",3),("s70002",3),("s70005",3),("s70007",3),("s70009",4),("s70012",4),("s70019",4),("s70021",4),("s70026",4),("s70010",4)]
close = [("s70000",4),("s70013",4),("s70017",3),("s70020",4)]
items = []
meta = []
def add(lst, tag, prompt=None):
    for s, k in lst:
        it = {"episode": "base.base." + s, "k": k}
        if prompt:
            it["prompt"] = prompt
        items.append(it)
        meta.append({"slot": s, "k": k, "tag": tag, "prompt": prompt or "orig"})
add(rel, "rel"); add(hold, "hold"); add(miss, "miss"); add(close, "close")
add(rel, "rel", "Put the cube in the bowl"); add(miss, "miss", "Put the cube in the bowl")
add(hold[:6], "hold", "Put the banana in the bowl")
spec = {"id": "a-diag1", "cand": "base", "prompt_tokens": list(range(24)), "items": items,
        "action_layers": list(range(18)), "vlm_layers": [6, 12, 17], "denoise_steps": [0, 5, 9],
        "action_tokens": "mean", "action_attention": True}
print(json.dumps(spec))
json.dump(meta, open("/work/notes/diag1_meta.json", "w"))
print(len(items))
