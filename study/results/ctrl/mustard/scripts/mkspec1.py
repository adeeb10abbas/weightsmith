import json
dec = json.load(open("/work/notes/decisions.json"))
items = []
first = [d for d in dec if d["first"]]
for p in ["Put the mustard in the left bin", "Put the mustard in the right bin"]:
    for d in first:
        items.append({"episode": d["eid"], "k": d["k"], "prompt": p})
spec = {"id": "a1-lr-postgrasp", "cand": "base", "prompt_tokens": list(range(24)), "items": items,
        "action_layers": list(range(18)), "vlm_layers": list(range(18)), "denoise_steps": [0, 5, 9],
        "action_tokens": "mean", "action_attention": True}
print(json.dumps(spec))
print(len(items))
