"""Recipe 1: find the failing decisions in recorded episodes.

When: you have run the unedited policy (``owsp eval base``) and want the exact replans where it goes wrong,
plus the replans that must not change, as capture items. See docs/COOKBOOK.md, section 1.

Run (read-only):  bin/owsp py base examples/recipes/01_find_decisions.py

The policy is queried every 15 sim steps (a "replan"); each query returns a 15-step chunk. This recipe reads every
recorded episode of one candidate and splits it into *holds*: runs of replans at which the measured gripper says an
object is in the hand (its width sets the reading: about 0.35 for the cube, 0.58 for the banana here). A hold that
ends with a chunk commanding open is a *release decision*; it is labelled by where it happens relative to the goal
(where successful episodes release). Output: counts, a few episode timelines, and JSON item lists
(``{"episode": ..., "k": ...}``) ready to paste into an ``owsp acts`` spec: targets (premature releases) and
preservation sets (releases at the goal, carries and approaches in successful episodes).

Check it worked: the targets should be the failure you saw in the videos and event timelines (``owsp show``):
here, the cube let go a few centimetres after the grasp, far from the bowl. On cube-then-banana the default
settings find 22 targets in 64 base episodes: all 21 of the paper's decisions and one more.
"""
import json

import numpy as np

DATA_ROOT = "/work/data"  # where `owsp py` mounts the project's episodes and captures

SETTINGS = {
    "cand": "base",                    # whose recorded episodes to scan (`owsp episodes` lists the names)
    "grip_range": [0.1, 0.9],          # measured gripper while holding something: 0 = open, ~1 = closed on nothing
    "object_grip_range": [0.2, 0.45],  # band of the object you care about (cube ~0.35, banana ~0.58); None = any
    "min_goal_dist": 0.10,             # a release farther than this (m, in x-y) from the goal is premature
    "max_hold": None,                  # optionally also require at most this many replans in hand
    "goal_xy": None,                   # goal x-y in the robot base frame (m); None = mean release point of successes
    "first_hold_only": True,           # targets: only the episode's first hold, and only if it is of the object
    "targets_from_failed_only": False,  # True: skip successful episodes (a policy can drop, re-grasp and succeed)
    "show_episodes": 3,                # print the hold timeline of this many failed episodes
    # optional: items to compare with, e.g. the paper's 21 premature releases (notes/sel_a1.json, PREM1)
    "compare_with": [["base.base.s20001", 5], ["base.base.s20007", 5], ["base.base.s20009", 4],
                     ["base.base.s20012", 5], ["base.base.s20020", 12], ["base.base.s20022", 4],
                     ["base.base.s20023", 4], ["base.base.s20024", 4], ["base.base.s20028", 4],
                     ["base.base.s20030", 7], ["base.base.s20032", 7], ["base.base.s20040", 4],
                     ["base.base.s20043", 5], ["base.base.s20044", 4], ["base.base.s20046", 5],
                     ["base.base.s20048", 5], ["base.base.s20049", 7], ["base.base.s20051", 6],
                     ["base.base.s20055", 5], ["base.base.s20060", 6], ["base.base.s20063", 4]],
}


def panda_fk(q):
    """Fingertip position (m, robot base frame) of a Franka Panda: modified DH, flange + 0.257 m."""
    dh = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
          (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]
    T = np.eye(4)
    for i, (a, d, al) in enumerate(dh):
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(q[i]), np.sin(q[i])
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
    F = np.eye(4)
    F[2, 3] = 0.257
    return (T @ F)[:3, 3]


def chunk_closed(actions, threshold=0.5):
    """True where a [.., 15, 8] chunk commands the gripper closed (most of its 15 steps above the threshold)."""
    return (np.asarray(actions)[..., 7] > threshold).mean(-1) >= 0.5


def replans(eid):
    """Per-replan gripper command chunks, measured gripper, joints and sim step of one recorded episode.

    Reads only these arrays: the prelude's ``obs(eid)`` also loads both camera streams (about 18 MB per episode).
    """
    z = np.load(f"{DATA_ROOT}/episodes/{eid}/obs.npz")
    return {k: z[k] for k in ("actions", "gripper", "joint", "step")}


def holds_of(o, lo, hi):
    """Holds of one episode: runs of replans k0..k1 with the measured gripper in [lo, hi].

    ``release`` is k1 when the chunk at k1 commands open (a decision to let go), else None (the run ended because
    the object slipped while the gripper stayed closed, or the episode ended).
    """
    closed = chunk_closed(o["actions"])
    g = np.asarray(o["gripper"], dtype=float).reshape(len(closed), -1)[:, 0]
    inside = (g >= lo) & (g <= hi)
    out, k, K = [], 0, len(g)
    while k < K:
        if not inside[k]:
            k += 1
            continue
        k0 = k
        while k + 1 < K and inside[k + 1]:
            k += 1
        k1 = k
        rel = k1 if not closed[k1] else None
        out.append({"k0": k0, "k1": k1, "release": rel, "length": k1 - k0 + 1, "grip": float(np.median(g[k0:k1 + 1])),
                    "grab_xyz": panda_fk(o["joint"][k0]), "end_xyz": panda_fk(o["joint"][k1])})
        k += 1
    return out


def main(cfg=SETTINGS):
    lo, hi = cfg["grip_range"]
    olo, ohi = cfg["object_grip_range"] or cfg["grip_range"]
    rows = sorted(episodes(cand=cfg["cand"]), key=lambda r: r["eid"])
    if not rows:
        raise SystemExit(f"no recorded episodes for {cfg['cand']!r}; run `owsp eval {cfg['cand']}` first")
    per_ep = {r["eid"]: (r, replans(r["eid"])) for r in rows}
    per_ep = {eid: (r, o, holds_of(o, lo, hi)) for eid, (r, o) in per_ep.items()}

    succ_rel = [h["end_xyz"][:2] for r, _, hs in per_ep.values() if r["success"] for h in hs
                if h["release"] is not None]
    if cfg["goal_xy"] is not None:
        goal, how = np.asarray(cfg["goal_xy"], dtype=float), "given"
    elif succ_rel:
        goal, how = np.mean(succ_rel, axis=0), f"mean of {len(succ_rel)} releases in successful episodes"
    else:
        raise SystemExit("no successful episode with a release: set goal_xy")
    n_fail = sum(not r["success"] for r in rows)
    print(f"{len(rows)} episodes of {cfg['cand']!r}, {len(rows) - n_fail} successful; goal x-y "
          f"{goal.round(3).tolist()} m ({how})")

    def dist(h):
        return float(np.linalg.norm(h["end_xyz"][:2] - goal))

    def is_obj(h):
        return olo <= h["grip"] <= ohi

    lists = {"targets": [], "goal_releases": [], "carries": [], "approaches": []}
    kinds = {"premature": 0, "at goal": 0, "slipped or never released": 0}
    for eid, (r, o, hs) in per_ep.items():
        for h in hs:
            if not is_obj(h):
                continue
            if h["release"] is None:
                kinds["slipped or never released"] += 1
            elif dist(h) <= cfg["min_goal_dist"]:
                kinds["at goal"] += 1
                if r["success"]:
                    lists["goal_releases"].append({"episode": eid, "k": h["release"]})
            else:
                kinds["premature"] += 1
        cand_hs = hs[:1] if cfg["first_hold_only"] else hs
        for h in cand_hs:
            short = cfg["max_hold"] is None or h["length"] <= cfg["max_hold"]
            eligible = not (r["success"] and cfg["targets_from_failed_only"])
            if eligible and is_obj(h) and h["release"] is not None and dist(h) > cfg["min_goal_dist"] and short:
                lists["targets"].append({"episode": eid, "k": h["release"], "step": int(o["step"][h["release"]]),
                                         "replans_in_hand": h["length"], "grip": round(h["grip"], 3),
                                         "goal_dist_cm": round(100 * dist(h), 1)})
                break
        if r["success"]:  # replans inside holds of successful episodes: the policy must keep holding there
            closed = chunk_closed(o["actions"])
            lists["carries"] += [{"episode": eid, "k": k} for h in hs if is_obj(h)
                                 for k in range(h["k0"], h["k1"] + 1) if closed[k] and k != h["release"]]
        first = hs[0]["k0"] if hs else len(o["step"])
        lists["approaches"] += [{"episode": eid, "k": k} for k in range(1, first)
                                if float(o["gripper"][k][0]) < 0.05][-1:]

    print(f"holds of the object (grip {olo}-{ohi}): {kinds['premature']} released > "
          f"{100 * cfg['min_goal_dist']:.0f} cm from the goal, {kinds['at goal']} released at the goal, "
          f"{kinds['slipped or never released']} slipped or never released")
    print(f"targets: {len(lists['targets'])} premature releases (at most one per episode); "
          + ", ".join(f"{k}: {len(v)}" for k, v in lists.items() if k != "targets"))
    for it in lists["targets"][:8]:
        print(f"  {it['episode']} k={it['k']} step {it['step']}: {it['replans_in_hand']} replan(s) in hand, "
              f"grip {it['grip']}, {it['goal_dist_cm']} cm from the goal")

    shown = 0
    for eid, (r, _, hs) in per_ep.items():
        if r["success"] or shown >= cfg["show_episodes"]:
            continue
        shown += 1
        print(f"\n{eid} (failed, {r['steps']} steps): every hold")
        for h in hs:
            end = (f"released at k={h['release']} {100 * dist(h):.0f} cm from the goal" if h["release"] is not None
                   else "slipped or never released")
            print(f"  k={h['k0']}-{h['k1']} grip {h['grip']:.2f} grabbed at {h['grab_xyz'].round(2).tolist()}, {end}")

    if cfg.get("compare_with"):
        want = {(e, int(k)) for e, k in cfg["compare_with"]}
        got = {(it["episode"], it["k"]) for it in lists["targets"]}
        print(f"\ncompare_with: {len(got & want)} of {len(want)} found; extra {sorted(got - want)[:6]}; "
              f"missed {sorted(want - got)[:6]}")
    for name, items in lists.items():
        if items:
            print(f"\n{name} ({len(items)}), as capture items (first 96):")
            print(json.dumps([{"episode": it["episode"], "k": int(it["k"])} for it in items[:96]]))
    return lists


if __name__ == "__main__":
    main()
