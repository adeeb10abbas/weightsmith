"""RoboLab client: one websocket request per sim step for all envs needing a replan.

Per-env deterministic noise seeds: seed = (base + episode)*100000 + env_id*1000 + replan_k.
If OWSP_OBS_DIR is set, every policy request (exact 224x224 inputs + seed + sim step)
and returned action chunk is saved per env as obs_r{episode}_e{env}.npz.
"""
from __future__ import annotations

import os

import numpy as np
from policies.pi0_family.client import Pi0DroidJointposClient


class BatchedPi0Client(Pi0DroidJointposClient):
    def start_episode(self, episode_index: int) -> None:
        super().start_episode(episode_index)
        self._env_replans: dict[int, int] = {}
        self._step = 0
        self._rec: dict[int, dict[str, list]] = {}

    def _flush(self):
        out = os.environ.get("OWSP_OBS_DIR")
        if not out or not getattr(self, "_rec", None):
            return
        os.makedirs(out, exist_ok=True)
        for e, r in self._rec.items():
            np.savez_compressed(
                os.path.join(out, f"obs_r{self._episode_index}_e{e}.npz"),
                **{k: np.stack(v) for k, v in r.items()},
            )
        self._rec = {}

    def reset(self, *, env_id=None) -> None:
        if env_id is None:
            self._flush()
            self._env_replans = {}
        super().reset(env_id=env_id)

    def _seed(self, env_id: int) -> int | None:
        if self._sampling_seed_base is None:
            return None
        k = self._env_replans.get(env_id, 0)
        self._env_replans[env_id] = k + 1
        return ((self._sampling_seed_base + self._episode_index) * 64 + env_id) * 100 + k

    def infer_batch(self, obs, instruction, *, env_ids):
        need = [e for e in env_ids if self._needs_refresh(e)]
        extracted = {e: self._extract_observation(obs, env_id=e) for e in env_ids}
        if need:
            reqs = []
            override = os.environ.get("OWSP_PROMPT")
            for e in need:
                r = self._pack_request(extracted[e], override or instruction)
                s = self._seed(e)
                if s is not None:
                    r["sampling_seed"] = s
                reqs.append(r)
            resp = self._infer_with_retry({"__batch__": reqs})
            if isinstance(resp, str):
                raise RuntimeError(resp)
            record = bool(os.environ.get("OWSP_OBS_DIR"))
            for e, r, out in zip(need, reqs, resp["__batch__"]):
                chunk = np.asarray(out["actions"])
                self._set_chunk(e, self._postprocess_chunk(chunk))
                if record:
                    d = self._rec.setdefault(e, {k: [] for k in ("base_img", "wrist_img", "joint", "gripper", "seed", "step", "actions")})
                    d["base_img"].append(np.asarray(r["observation/exterior_image_1_left"], np.uint8))
                    d["wrist_img"].append(np.asarray(r["observation/wrist_image_left"], np.uint8))
                    d["joint"].append(np.asarray(r["observation/joint_position"], np.float32))
                    d["gripper"].append(np.asarray(r["observation/gripper_position"], np.float32).reshape(-1))
                    d["seed"].append(np.int64(r.get("sampling_seed", -1)))
                    d["step"].append(np.int64(self._step))
                    d["actions"].append(chunk.astype(np.float32))
        self._step += 1
        rets = {}
        for e in env_ids:
            rets[e] = {"action": self._next_action(e), "viz": self._build_visualization(extracted[e]) if e == 0 else None}
        return rets
