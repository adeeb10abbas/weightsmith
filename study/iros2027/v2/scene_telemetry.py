"""Opt-in pose logging for the new development cohort; does not change resets or scoring."""
import json
import os
from pathlib import Path


def install():
    from robolab.core.environments.env import RobolabEnv

    original_reset, original_step = RobolabEnv.reset, RobolabEnv.step

    def snapshot(env):
        return {
            name: {"position": (obj.data.root_pos_w - env.scene.env_origins).tolist(),
                   "quaternion_wxyz": obj.data.root_quat_w.tolist()}
            for name, obj in env.scene.rigid_objects.items()
        }

    def write(env, name):
        out = Path(os.environ["OWSP_OBS_DIR"])
        out.mkdir(parents=True, exist_ok=True)
        record = {"objects": snapshot(env), "num_envs": env.num_envs,
                  "control_dt": float(env.step_dt), "environment_seed": 0}
        if hasattr(env, "_iros_bowl_displacement"):
            record["max_bowl_displacement_m"] = env._iros_bowl_displacement.tolist()
        (out / name).write_text(json.dumps(record))

    def reset(env, *args, **kwargs):
        result = original_reset(env, *args, **kwargs)
        env._iros_first_step = True
        return result

    def step(env, action):
        if getattr(env, "_iros_first_step", True):
            env._iros_first_step = False
            env._iros_bowl_start = env.scene["bowl"].data.root_pos_w.clone()
            env._iros_bowl_displacement = env._iros_bowl_start[:, 0] * 0
            write(env, "scene-start.json")
        result = original_step(env, action)
        displacement = (env.scene["bowl"].data.root_pos_w - env._iros_bowl_start).norm(dim=-1)
        env._iros_bowl_displacement = env._iros_bowl_displacement.maximum(displacement)
        if bool(env._frozen_envs.all()):
            write(env, "scene-final.json")
        return result

    RobolabEnv.reset, RobolabEnv.step = reset, step
