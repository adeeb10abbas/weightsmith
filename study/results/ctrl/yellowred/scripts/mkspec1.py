import json
decs = [("base.base.s70004", 1), ("base.base.s70013", 1), ("base.base.s70028", 1), ("base.base.s70030", 1),
        ("base.base.s70004", 10), ("base.base.s70013", 3), ("base.base.s70020", 3), ("base.base.s70028", 5)]
prompts = [None,
           "Pick up the yellow block",
           "Stack the lime block on the red block",
           "Stack the yellow cube on the red cube",
           "Stack the yellow-green block on the red block",
           "Pick up the yellow block next to the green block",
           "Stack the green block on the red block",
           "Put the yellow block on top of the red block"]
items = []
for p in prompts:
    for (e, k) in decs:
        it = {"episode": e, "k": k}
        if p is not None:
            it["prompt"] = p
        items.append(it)
spec = {"id": "a1-prompts", "cand": "base", "prompt_tokens": list(range(24)),
        "items": items, "action_layers": [0, 4, 8, 12, 17], "vlm_layers": [17],
        "denoise_steps": [0, 9], "action_tokens": "mean", "action_attention": True}
print(json.dumps(spec))
