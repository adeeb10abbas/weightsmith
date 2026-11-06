import json
dec = [('s70005', 15), ('s70005', 16), ('s70013', 5), ('s70017', 4), ('s70018', 4), ('s70021', 5),
       ('s70024', 6), ('s70031', 6), ('s70009', 8), ('s70022', 13), ('s70003', 16), ('s70006', 10)]
prompts = ["Put the rubiks cube to the front of the bowl",
           "Put the rubiks cube to the back of the bowl"]
items = []
for p in prompts:
    for s, k in dec:
        items.append({"episode": "base.base." + s, "k": k, "prompt": p})
spec = {"id": "a2-frontback", "cand": "base", "prompt_tokens": list(range(28)), "items": items,
        "action_layers": [17], "vlm_layers": [0, 1, 3, 8, 17], "denoise_steps": [9], "action_tokens": "mean",
        "image_tokens": True}
print(json.dumps(spec))
