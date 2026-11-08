# owsp py base: list the paired practice slots of two arms (RF = the task's own wording, TF = prompt override),
# with replan counts, so that capture items can be chosen from RF failures whose TF twin succeeded.
# Edit ARM_RF / ARM_TF below (episode labels as in data/episodes/index.jsonl). Prints one JSON line at the end.
ARM_RF, ARM_TF = "base", None  # None: the single base@p... label present in the project

rows = episodes()
labels = sorted({r["cand"] for r in rows})
if ARM_TF is None:
    tf = [l for l in labels if l.startswith("base@p")]
    assert len(tf) == 1, f"expected one base@p label, got {tf}"
    ARM_TF = tf[0]
by = {}
for r in rows:
    by.setdefault(r["cand"], {}).setdefault(r["slot"], r)
shared = sorted(set(by[ARM_RF]) & set(by[ARM_TF]))
out = {"arm_rf": ARM_RF, "arm_tf": ARM_TF, "prompt_tf": by[ARM_TF][shared[0]]["prompt"], "pairs": []}
for s in shared:
    a, b = by[ARM_RF][s], by[ARM_TF][s]
    oa, ob = obs(a["eid"]), obs(b["eid"])
    out["pairs"].append({"slot": s, "rf": a["eid"], "tf": b["eid"], "rf_ok": a["success"], "tf_ok": b["success"],
                         "rf_steps": a["steps"], "tf_steps": b["steps"], "rf_K": int(len(oa["step"])),
                         "tf_K": int(len(ob["step"])),
                         "k0_same_obs": bool(np.array_equal(oa["base_img"][0], ob["base_img"][0])
                                             and np.array_equal(oa["joint"][0], ob["joint"][0])),
                         "k0_seed": [int(oa["seed"][0]), int(ob["seed"][0])],
                         "rf_grip_closed_k": [int(k) for k in np.flatnonzero((oa["actions"][:, :, 7] > 0.5).mean(1) >= 0.5)[:3]],
                         "tf_grip_closed_k": [int(k) for k in np.flatnonzero((ob["actions"][:, :, 7] > 0.5).mean(1) >= 0.5)[:3]]})
n = len(out["pairs"])
c = {"both": 0, "neither": 0, "tf_only": 0, "rf_only": 0}
for p in out["pairs"]:
    c["both" if p["rf_ok"] and p["tf_ok"] else "neither" if not (p["rf_ok"] or p["tf_ok"]) else "tf_only" if p["tf_ok"] else "rf_only"] += 1
out["counts"] = c
print(f"{n} shared slots: {c}")
for p in out["pairs"]:
    print(p["slot"], "RF", "OK " if p["rf_ok"] else "FAIL", "TF", "OK " if p["tf_ok"] else "FAIL",
          "K", p["rf_K"], p["tf_K"], "k0 same obs", p["k0_same_obs"], "grip k RF", p["rf_grip_closed_k"], "TF", p["tf_grip_closed_k"])
print("PAIRS_JSON " + json.dumps(out))
