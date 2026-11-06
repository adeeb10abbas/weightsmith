import json
groups = {
    "bowlrel": [("s70000", 8), ("s70013", 7), ("s70017", 6), ("s70025", 8), ("s70017", 27), ("s70030", 12), ("s70001", 35), ("s70024", 39), ("s70016", 52), ("s70027", 15)],
    "carry": [("s70000", 6), ("s70000", 7), ("s70025", 6), ("s70025", 7), ("s70017", 5), ("s70013", 6)],
    "stage2": [("s70000", 10), ("s70000", 11), ("s70000", 12), ("s70000", 13), ("s70013", 9), ("s70013", 10), ("s70013", 11),
               ("s70017", 8), ("s70017", 9), ("s70017", 10), ("s70025", 9), ("s70025", 10), ("s70025", 11), ("s70025", 12)],
    "approach": [("s70000", 2), ("s70005", 2), ("s70011", 2), ("s70020", 2)],
}
items, meta = [], []
for tag, lst in groups.items():
    for s, k in lst:
        items.append({"episode": "base.base." + s, "k": k})
        meta.append({"slot": s, "k": k, "tag": tag})
spec = {"id": "a-pres2", "cand": "base", "prompt_tokens": list(range(16)), "items": items,
        "action_layers": [12, 13, 14, 15, 16, 17], "vlm_layers": [12], "denoise_steps": [0, 5, 9],
        "action_tokens": "mean"}
json.dump(meta, open("/work/notes/pres2_meta.json", "w"))
print(json.dumps(spec))
print(len(items))
