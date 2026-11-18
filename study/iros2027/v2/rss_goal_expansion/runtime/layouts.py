"""Start environments from fixed object layouts. Opt-in: nothing changes unless WSP_START_LAYOUTS is set.

WSP_START_LAYOUTS names a JSON file:

  {"objects": [...], "settle_steps": 15, "renders": 8,
   "layouts": [{"id": "C01", "poses": {object: {"pos": [x, y, z], "quat": [w, x, y, z]}}}, ...],
   "probe_module": "<path relative to the file>", "probes": [...]}        (the last two are optional)

Environment slot s = group * num_envs + env_id starts from layouts[s % len(layouts)]. This is the pool's own
slot rule, so the layout of a slot does not depend on how groups are spread over lanes.

WSP_START_LAYOUT_IDS (optional) is a comma list of layout ids, e.g. "C01,C02,C03". Only those layouts are then
used, in the listed order: slot s starts from the (s % k)-th listed layout, k = number of listed ids. Without
the variable every layout of the file is used, as before.

After every stock reset the listed objects of each environment are moved to their layout pose (environment-local
frame, zero velocity), the robot holds its joint positions while physics runs for `settle_steps` control steps,
the cameras are rendered again and the observation is recomputed. What was written and what was measured is
saved as layout_r{run}.json next to the recorded policy inputs (OWSP_OBS_DIR), else in the run's output folder.
If `probe_module` is given, its `probe(env, probes)` result (e.g. goal predicates at reset) is saved with it.
"""
from __future__ import annotations

import importlib.util
import hashlib
import numpy as np
import inspect
import json
import os
import sys
import time
from pathlib import Path

ENV_VAR = "WSP_START_LAYOUTS"
IDS_ENV_VAR = "WSP_START_LAYOUT_IDS"


def layout_index(slot: int, n_layouts: int) -> int:
    """Which layout a slot starts from."""
    return int(slot) % int(n_layouts)


def allowed(layouts: list, ids=None) -> list:
    """The layouts slots may start from. `ids` is a list of layout ids or a comma string; None reads
    WSP_START_LAYOUT_IDS. No ids: every layout, in file order (the behaviour without the variable)."""
    if ids is None:
        ids = os.environ.get(IDS_ENV_VAR, "")
    if isinstance(ids, str):
        ids = [x.strip() for x in ids.split(",") if x.strip()]
    ids = list(ids)
    if not ids:
        return list(layouts)
    by_id = {x["id"]: x for x in layouts}
    bad = [i for i in ids if i not in by_id]
    if bad or len(set(ids)) != len(ids):
        raise ValueError(f"allowed layout ids {ids}: unknown {bad} or repeated")
    return [by_id[i] for i in ids]


def load(path) -> dict:
    spec = json.loads(Path(path).read_text())
    spec["_path"] = str(Path(path).resolve())
    spec["sha256"] = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if spec.get("case") not in ("cube_behind", "glasses_pickup", "mustard_left_bin"):
        raise ValueError("Goal-expansion layout must identify a declared case")
    spec["layouts"] = allowed(spec["layouts"])
    return spec


def _probe_fn(spec: dict):
    rel = spec.get("probe_module")
    if not rel or not spec.get("probes"):
        return None
    path = (Path(spec["_path"]).parent / rel).resolve()
    mod = sys.modules.get(path.stem)
    if mod is None or Path(getattr(mod, "__file__", "") or ".").resolve() != path:
        s = importlib.util.spec_from_file_location(path.stem, path)
        mod = importlib.util.module_from_spec(s)
        sys.modules[path.stem] = mod
        s.loader.exec_module(mod)
    return mod.probe


def apply(env, spec: dict, group: int, run: int, call: int = 0) -> dict:
    """Move the objects of every environment to its layout, settle, re-render. Returns the saved record."""
    import torch

    t0 = time.time()
    n, dev = env.num_envs, env.device
    lays = spec["layouts"]
    slots = [group * n + e for e in range(n)]
    pick = [lays[layout_index(s, len(lays))] for s in slots]
    origins = env.scene.env_origins
    names = list(spec["objects"])

    def local(name):
        return (env.scene[name].data.root_pos_w - origins).clone()

    def norm(x):
        return torch.linalg.norm(x, dim=-1).tolist()

    stock = {name: local(name) for name in names}
    written = {}
    for name in names:
        pos = torch.tensor([p["poses"][name]["pos"] for p in pick], dtype=torch.float32, device=dev)
        quat = torch.tensor([p["poses"][name]["quat"] for p in pick], dtype=torch.float32, device=dev)
        asset = env.scene[name]
        asset.write_root_pose_to_sim(torch.cat([pos + origins, quat], dim=-1))
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev))
        written[name] = (pos, quat)
    env.sim.forward()
    env.scene.update(dt=0.0)
    placed = {name: (local(name), env.scene[name].data.root_quat_w.clone()) for name in names}

    # the robot holds where it is while the objects settle
    held = {}
    for key, art in env.scene.articulations.items():
        q = art.data.joint_pos.clone()
        art.set_joint_position_target(q)
        art.set_joint_velocity_target(torch.zeros_like(q))
        held[key] = q
    steps = int(spec.get("settle_steps", 15))
    track = {name: [] for name in names}
    for _ in range(steps):
        for _ in range(int(env.cfg.decimation)):
            env.scene.write_data_to_sim()
            env.sim.step(render=False)
            env.scene.update(dt=env.physics_dt)
        for name in names:
            track[name].append(local(name))

    rec = {"file": spec["_path"], "run": int(run), "group": int(group), "num_envs": n, "reset_call": int(call),
           "slots": slots, "layout_ids": [p["id"] for p in pick], "settle_steps": steps,
           "control_dt": float(env.step_dt), "objects": {}, "probes": None}
    last = min(9, steps)   # drift over the last 9 control ticks, as in the layouts' source
    for name in names:
        pos, quat = written[name]
        end = track[name][-1] if steps else placed[name][0]
        data = env.scene[name].data
        rec["objects"][name] = {
            "stock_pos": stock[name].tolist(),
            "written_pos": pos.tolist(), "written_quat": quat.tolist(),
            "placed_pos": placed[name][0].tolist(), "placed_quat": placed[name][1].tolist(),
            "settled_pos": end.tolist(), "settled_quat": data.root_quat_w.tolist(),
            "moved_in_settle_m": norm(end - pos),
            "drift_last_ticks_m": norm(end - track[name][-last]) if steps else [0.0] * n,
            "speed_mps": norm(data.root_lin_vel_w), "spin_radps": norm(data.root_ang_vel_w)}
    rec["robot_joint_drift_rad"] = {key: float((art.data.joint_pos - held[key]).abs().max())
                                    for key, art in env.scene.articulations.items()}
    # Preserve failed reset evidence as well as successful reset evidence.
    out = Path(os.environ.get("OWSP_OBS_DIR") or env.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"layout_r{run}_reset{call}.json").write_text(json.dumps(rec))
    # A bad reset invalidates the batch instead of becoming a policy failure.
    validate_record(rec)
    from tasks import observe
    from scoring import Scorer
    positions, initial_predicates = observe(env, spec["case"])
    if any(np.any(v) for k, v in initial_predicates.items()
           if k in ("behind", "cube_in_bowl", "glasses_picked_up", "cube_picked_up", "mustard_left", "mustard_right")):
        raise RuntimeError("Layout starts with a scored relation already true")
    env._rss_scorer = Scorer(spec["case"], positions, float(env.step_dt))
    env._rss_selected_goal = None
    rec["initial_predicates"] = {k: v.tolist() for k, v in initial_predicates.items()}
    rec["case"] = spec["case"]
    rec["sha256"] = spec["sha256"]
    robot = env.scene["robot"].data
    rec["robot_base_pos"] = (robot.root_pos_w - origins).tolist()
    rec["robot_base_quaternion_wxyz"] = robot.root_quat_w.tolist()
    env._rss_layout_receipt = rec
    probe = _probe_fn(spec)
    if probe is not None:
        rec["probes"] = probe(env, spec["probes"])

    for _ in range(int(spec.get("renders", 8))):
        env.sim.render()
    for sensor in env.scene.sensors.values():
        if "Camera" in type(sensor).__name__:
            sensor.update(0.0, force_recompute=True)
    env.obs_buf = env.observation_manager.compute()
    rec["wall_s"] = round(time.time() - t0, 2)

    out = Path(os.environ.get("OWSP_OBS_DIR") or env.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"layout_r{run}_reset{call}.json").write_text(json.dumps(rec))
    (out / f"layout_r{run}.json").write_text(json.dumps(rec))
    print(f"[layouts] run {run} group {group} reset {call}: {' '.join(rec['layout_ids'])} "
          f"({rec['wall_s']} s)", flush=True)
    return rec


def install(seed_base=None) -> bool:
    """Make every episode reset end in `apply`. Returns False (and does nothing) unless WSP_START_LAYOUTS is set."""
    path = os.environ.get(ENV_VAR)
    if not path:
        raise RuntimeError("RSS runtime requires WSP_START_LAYOUTS; stock resets are not allowed")
    spec = load(path)
    import robolab.eval.episode as episode_mod

    stock_run = episode_mod.run_episode
    sig = inspect.signature(stock_run)
    base = int(seed_base or 0)
    state = {"run": 0, "calls": 0}

    def run_episode(*args, **kwargs):
        bound = sig.bind(*args, **kwargs)
        env = bound.arguments["env"]
        state.update(run=int(bound.arguments["episode"]), calls=0)
        if not getattr(env, "_wsp_layout_reset", False):
            stock_reset = env.reset

            def reset(*a, **k):
                env._rss_scorer = None
                _, extras = stock_reset(*a, **k)
                state["calls"] += 1
                apply(env, spec, base + state["run"], state["run"], state["calls"])
                return env.obs_buf, extras

            env.reset = reset
            env._wsp_layout_reset = True
        result = stock_run(*args, **kwargs)
        scorer = getattr(env, "_rss_scorer", None)
        if scorer is None or state["calls"] != 2:
            raise RuntimeError(f"Expected two layout resets, observed {state['calls']}")
        rec = scorer.record()
        rec.update(selected_goal=env._rss_selected_goal, layout=env._rss_layout_receipt,
                   run=state["run"], reset_calls=state["calls"], environment_results=result[0])
        out = Path(os.environ.get("OWSP_OBS_DIR") or env.output_dir)
        (out / f"metrics_r{state['run']}.json").write_text(json.dumps(rec, indent=2))
        return result

    episode_mod.run_episode = run_episode
    print(f"[layouts] starting environments from {spec['_path']} ({len(spec['layouts'])} layouts, "
          f"group base {base})", flush=True)
    return True


def validate_record(rec):
    """Reject missing/wrong placements and unsettled physics before policy evaluation."""
    for name, data in rec["objects"].items():
        written, placed = np.array(data["written_pos"]), np.array(data["placed_pos"])
        settled = np.array(data["settled_pos"])
        if not np.isfinite(settled).all() or not np.isfinite(placed).all():
            raise RuntimeError(f"Nonfinite reset pose for {name}")
        if np.max(np.linalg.norm(written - placed, axis=1)) > .002:
            raise RuntimeError(f"Written reset did not take effect for {name}")
        wq, pq = np.array(data["written_quat"]), np.array(data["placed_quat"])
        cosine = np.abs(np.sum(wq * pq, axis=1) / (np.linalg.norm(wq, axis=1) * np.linalg.norm(pq, axis=1)))
        if not np.isfinite(cosine).all() or np.min(cosine) < np.cos(np.deg2rad(1) / 2):
            raise RuntimeError(f"Written reset orientation did not take effect for {name}")
        if np.max(np.linalg.norm(written[:, :2] - settled[:, :2], axis=1)) > .012:
            raise RuntimeError(f"Reset position changed by over 12mm in XY for {name}")
        if max(data["drift_last_ticks_m"]) > .005 or max(data["speed_mps"]) > .03:
            raise RuntimeError(f"Object did not settle for {name}")
        q = np.array(data["settled_quat"])
        if not np.isfinite(q).all() or np.max(np.abs(np.linalg.norm(q, axis=1) - 1)) > .005:
            raise RuntimeError(f"Invalid quaternion for {name}")
