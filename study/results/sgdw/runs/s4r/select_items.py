"""Fixed selection rule for the S4-R build and check decisions (run from study/results/sgdw)."""
import json
d1 = json.load(open("runs/s4r/diverge.json"))["rows"]   # RF-failure episodes: base RF vs TF replays of RF inputs
d2 = json.load(open("runs/s4r/div2.json"))              # TF-success episodes: RF vs TF replays of TF inputs
pairs = json.load(open("runs/s4r/pairs.json"))["pairs"]
RF = "Move the mustard bottle on the table so that the raisin box is to the left of the mustard bottle."
TF = "Move the mustard bottle on the table so that the mustard bottle is to the right of the raisin box."
# approach: per replayed RF-failure episode, the replan k in 2..6 with the largest TF-vs-RF chunk-end EE distance
approach = []
for e in sorted({r["episode"] for r in d1}):
    rr = [r for r in d1 if r["episode"] == e and 2 <= r["k"] <= 6]
    approach.append(max(rr, key=lambda r: r["ee_dist_cm"]))
# transport: TF-episode replans holding the mustard (measured gripper 0.3..0.9, both wordings command closed),
# top 6 by the same distance, at most 2 per episode
hold = sorted([r for r in d2 if 0.3 <= r["grip_state"] <= 0.9 and r["closed_rf"] and r["closed_tf"]],
              key=lambda r: -r["ee_dist_cm"])
transport, used = [], {}
for r in hold:
    if used.get(r["episode"], 0) >= 2:
        continue
    transport.append(r)
    used[r["episode"]] = used.get(r["episode"], 0) + 1
    if len(transport) == 6:
        break
build = [{"episode": r["episode"], "k": r["k"]} for r in approach + transport]
bset = {(b["episode"], b["k"]) for b in build}
# held-out decisions: approach replans 2..6 of six other RF failures; other holding replans; TF no-harm replans
used_rf = {r["episode"] for r in d1}
other_rf = [p["rf"] for p in pairs if p["tf_ok"] and not p["rf_ok"] and p["rf"] not in used_rf][:6]
appr_chk = [{"episode": e, "k": k} for e in other_rf for k in range(2, 7)]
tr_chk = [{"episode": r["episode"], "k": r["k"]} for r in hold if (r["episode"], r["k"]) not in bset]
used_tf = {r["episode"] for r in d2}
nh = [p for p in pairs if p["tf_ok"] and p["tf"] not in used_tf][:5]
nh_items = [{"episode": p["tf"], "k": k} for p in nh for k in range(1, min(p["tf_K"], 9))]
chk = build + appr_chk + tr_chk
groups = {"build": [0, len(build)], "approach_heldout": [len(build), len(build) + len(appr_chk)],
          "transport_heldout": [len(build) + len(appr_chk), len(chk)]}
print("approach:", [(r["episode"][-6:], r["k"], r["ee_dist_cm"], r["dy_tf_minus_rf_cm"]) for r in approach])
print("transport:", [(r["episode"][-6:], r["k"], r["ee_dist_cm"], r["dy_tf_minus_rf_cm"]) for r in transport])
print(len(build), "build;", len(appr_chk), "approach check from", [e[-6:] for e in other_rf], ";", len(tr_chk),
      "transport check;", len(nh_items), "TF no-harm items from", [p["tf"][-6:] for p in nh])
print("groups:", groups)
json.dump({"build": build, "approach_check": appr_chk, "transport_check": tr_chk, "tf_noharm": nh_items, "groups": groups},
          open("runs/s4r/items.json", "w"), indent=1)


def spec(aid, cand, items, prompt, full=False):
    s = {"id": aid, "cand": cand, "prompt": prompt, "items": items, "action_layers": [17], "action_tokens": [0],
         "denoise_steps": [0], "vlm_layers": []}
    if full:
        s.update({"vlm_layers": list(range(7, 17)), "image_tokens": True, "prompt_tokens": [0]})
    return s


# Merged captures (fewer GPU-lock acquisitions): per-item prompts select the wording inside one capture.
def tag(items, prompt):
    return [dict(it, prompt=prompt) for it in items]


M = {"s4r-ox": spec("s4r-ox", "base", tag(build, RF) + tag(build, TF), None, True),    # build: items 0..11 RF, 12..23 TF
     "chk1": spec("chk1", "base", tag(chk, RF) + tag(nh_items, TF), None),            # base RF on check items + base TF no-harm
     "chk2": spec("chk2", "base", tag(chk, TF), None)}                                # base TF on check items (the target)
for k, v in M.items():
    v.pop("prompt")
    assert len(v["items"]) <= 96, (k, len(v["items"]))
    json.dump(v, open(f"runs/s4r/specs/{k}.json", "w"), indent=1)
print({k: len(v["items"]) for k, v in M.items()})
