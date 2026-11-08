"""RoboLab client for GR00T N1.7 DROID: one batched request per sim step for all envs that need a replan.

Built on RoboLab's validated ``policies/gr00t/client.py`` (observation contract unchanged: exterior
``over_shoulder_left_camera`` and ``wrist_cam`` sent as HWC uint8 180x320 without letterboxing, state
``eef_9d`` / ``joint_position`` / ``gripper_position``, actions = absolute ``joint_position`` (7) +
``gripper_position`` (1) binarised at 0.5, open-loop horizon 8). Additions, as in weightsmith's pi0.5 client:

* paired seeds: the flow-matching noise of env e at its k-th replan in episode (run) r uses
  seed = ((sampling_seed_base + r) * 64 + e) * 100 + k, sent as ``options.seeds`` (see groot_server.py);
* ``OWSP_PROMPT`` overrides the task instruction (wording arms), the task's own wording otherwise;
* ``OWSP_OBS_DIR``: every request (exact images and state sent, seed, sim step) and the returned chunk
  (unbinarised, all 40 rows) are saved per env as ``obs_r{run}_e{env}.npz`` for offline replay.
"""
from __future__ import annotations

import os

import numpy as np

from policies.gr00t.client import RESOLUTION, GR00TDroidJointposClient, resize_no_pad, _MsgSerializer


class BatchedGR00TClient(GR00TDroidJointposClient):
    def __init__(self, *args, sampling_seed_base: int | None = None, **kw):
        super().__init__(*args, **kw)
        self._sampling_seed_base = sampling_seed_base
        self._episode_index = 0
        self._env_replans: dict[int, int] = {}
        self._step = 0
        self._rec: dict[int, dict[str, list]] = {}

    def start_episode(self, episode_index: int) -> None:
        super().start_episode(episode_index)
        self._episode_index = int(episode_index)
        self._env_replans = {}
        self._step = 0
        self._rec = {}

    def _flush(self):
        out = os.environ.get("OWSP_OBS_DIR")
        if not out or not self._rec:
            return
        os.makedirs(out, exist_ok=True)
        for e, r in self._rec.items():
            np.savez_compressed(os.path.join(out, f"obs_r{self._episode_index}_e{e}.npz"),
                                **{k: np.stack(v) for k, v in r.items()})
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

    def _get_action_batched(self, observation: dict, seeds):
        req = {"endpoint": "get_action", "data": {"observation": observation,
                                                   "options": {"seeds": seeds} if seeds else None}}
        self.client.socket.send(_MsgSerializer.to_bytes(req))
        msg = self.client.socket.recv()
        if msg == b"ERROR":
            raise RuntimeError("GR00T server error")
        resp = _MsgSerializer.from_bytes(msg)
        if isinstance(resp, dict) and "error" in resp:
            raise RuntimeError(f"Server error: {resp['error']}")
        return resp[0]

    def infer_batch(self, obs, instruction, *, env_ids):
        need = [e for e in env_ids if self._needs_refresh(e)]
        extracted = {e: self._extract_observation(obs, env_id=e) for e in env_ids}
        if need:
            text = os.environ.get("OWSP_PROMPT") or instruction
            ext = [resize_no_pad(extracted[e]["external_image"], *RESOLUTION) for e in need]
            wri = [resize_no_pad(extracted[e]["wrist_image"], *RESOLUTION) for e in need]
            seeds = [self._seed(e) for e in need]
            seeds = None if any(s is None for s in seeds) else seeds
            o = {
                "video.exterior_image_1_left": np.stack(ext)[:, None].astype(np.uint8),
                "video.wrist_image_left": np.stack(wri)[:, None].astype(np.uint8),
                "state.eef_9d": np.stack([extracted[e]["eef_9d"] for e in need])[:, None].astype(np.float32),
                "state.joint_position": np.stack([extracted[e]["joint_position"] for e in need])[:, None].astype(np.float32),
                "state.gripper_position": np.stack([extracted[e]["gripper_position"] for e in need])[:, None].astype(np.float32),
                "annotation.language.language_instruction": [text] * len(need),
            }
            act = self._get_action_batched(o, seeds)
            joint = np.asarray(act["action.joint_position"], np.float32)        # [B, 40, 7] absolute
            grip = np.asarray(act["action.gripper_position"], np.float32)       # [B, 40, 1] absolute
            if grip.ndim == 2:
                grip = grip[..., None]
            record = bool(os.environ.get("OWSP_OBS_DIR"))
            for i, e in enumerate(need):
                chunk = np.concatenate([joint[i], grip[i]], axis=1)
                self._set_chunk(e, self._postprocess_chunk(chunk))
                if record:
                    d = self._rec.setdefault(e, {k: [] for k in ("ext_img", "wrist_img", "joint", "gripper", "eef_9d",
                                                                 "seed", "step", "actions", "eef_actions")})
                    d["ext_img"].append(o["video.exterior_image_1_left"][i, 0])
                    d["wrist_img"].append(o["video.wrist_image_left"][i, 0])
                    d["joint"].append(o["state.joint_position"][i, 0])
                    d["gripper"].append(o["state.gripper_position"][i, 0].reshape(-1))
                    d["eef_9d"].append(o["state.eef_9d"][i, 0])
                    d["seed"].append(np.int64(seeds[i] if seeds else -1))
                    d["step"].append(np.int64(self._step))
                    d["actions"].append(chunk.astype(np.float32))
                    eef = act.get("action.eef_9d")
                    d["eef_actions"].append(np.asarray(eef[i], np.float32) if eef is not None else np.zeros((1, 9), np.float32))
        self._step += 1
        return {e: {"action": self._next_action(e),
                    "viz": self._build_visualization(extracted[e]) if e == 0 else None} for e in env_ids}
