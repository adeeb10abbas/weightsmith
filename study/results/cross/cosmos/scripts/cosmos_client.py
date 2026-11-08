"""Batched RoboLab client for Cosmos3 Edge with weightsmith's paired slot scheme (Isaac Sim venv).

Wire format and image composite are RoboLab's policies/cosmos3 Cosmos3Client unchanged (wrist 360x640 over the
left|right shoulder views at 180x320 each = 540x640; joint positions; gripper; 32-action open-loop chunks; gripper
command binarised at 0.5). Differences from the stock client:
- one {"__batch__": [...]} request per sim step for all envs that need a new chunk (answers are served one by one);
- per-env deterministic sampling seed ((group * 64 + env) * 100 + k), k = the env's replan index, as weightsmith's
  BatchedPi0Client: a slot always receives the same noise stream whatever runs beside it (paired evaluations);
- OWSP_PROMPT overrides the task instruction (wording arms);
- OWSP_OBS_DIR records every request (composite image, joints, gripper, seed, sim step, returned raw chunk) per env as
  obs_r{episode}_e{env}.npz, for offline replay and activation capture;
- OWSP_VIDEO_DIR writes one MP4 per slot (<slot>.mp4, 15 fps, every control step): left shoulder | right shoulder |
  wrist camera, 320x180 each, from the observations the env already rendered (no extra rendering).
"""
from __future__ import annotations

import os

import cv2
import numpy as np
from policies.cosmos3.client import Cosmos3Client

VIDEO_CAMS = ("over_shoulder_left_camera", "over_shoulder_right_camera", "wrist_cam")
VIDEO_TILE = (320, 180)  # (w, h) per camera


class BatchedCosmos3Client(Cosmos3Client):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._env_replans: dict[int, int] = {}
        self._step = 0
        self._rec: dict[int, dict[str, list]] = {}
        self._vw: dict[int, cv2.VideoWriter] = {}

    def start_episode(self, episode_index: int) -> None:
        super().start_episode(episode_index)
        self._env_replans, self._step, self._rec = {}, 0, {}

    def _flush(self):
        out = os.environ.get("OWSP_OBS_DIR")
        if not out or not getattr(self, "_rec", None):
            return
        os.makedirs(out, exist_ok=True)
        for e, r in self._rec.items():
            np.savez_compressed(os.path.join(out, f"obs_r{self._episode_index}_e{e}.npz"),
                                **{k: np.stack(v) for k, v in r.items()})
        self._rec = {}

    def _close_videos(self):
        for w in getattr(self, "_vw", {}).values():
            w.release()
        self._vw = {}

    def _video(self, obs, env_ids):
        out = os.environ.get("OWSP_VIDEO_DIR")
        if not out:
            return
        import torch
        import torch.nn.functional as F
        imgs = {}
        with torch.no_grad():  # downsample on the GPU, then copy 320x180 tiles (not 1280x720 frames) to the CPU
            for c in VIDEO_CAMS:
                x = obs["image_obs"][c][list(env_ids)].permute(0, 3, 1, 2).float()
                x = F.interpolate(x, size=(VIDEO_TILE[1], VIDEO_TILE[0]), mode="area")
                imgs[c] = dict(zip(env_ids, x.round().clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()))
        off = int(os.environ.get("OWSP_ENV_OFFSET", "0"))
        for e in env_ids:
            if e not in self._vw:
                os.makedirs(out, exist_ok=True)
                slot = (int(self._sampling_seed_base or 0) + self._episode_index) * 16 + off + e
                self._vw[e] = cv2.VideoWriter(os.path.join(out, f"{slot}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                              (VIDEO_TILE[0] * len(VIDEO_CAMS), VIDEO_TILE[1]))
            tiles = [imgs[c][e] for c in VIDEO_CAMS]
            self._vw[e].write(cv2.cvtColor(np.concatenate(tiles, 1), cv2.COLOR_RGB2BGR))

    def reset(self, *, env_id=None) -> None:
        if env_id is None:
            self._flush()
            self._close_videos()
            self._env_replans = {}
        super().reset(env_id=env_id)

    def close(self) -> None:
        self._flush()
        self._close_videos()
        super().close()

    def _seed(self, env_id: int) -> int:
        k = self._env_replans.get(env_id, 0)
        self._env_replans[env_id] = k + 1
        j = env_id + int(os.environ.get("OWSP_ENV_OFFSET", "0"))  # slot index within the 16-slot group
        return ((int(self._sampling_seed_base or 0) + self._episode_index) * 64 + j) * 100 + k

    def infer_batch(self, obs, instruction, *, env_ids):
        need = [e for e in env_ids if self._needs_refresh(e)]
        if need:
            prompt = os.environ.get("OWSP_PROMPT") or instruction
            reqs = []
            for e in need:
                r = self._pack_request(self._extract_observation(obs, env_id=e), prompt)
                r["sampling_seed"] = self._seed(e)
                reqs.append(r)
            resp = self._infer_with_retry({"__batch__": reqs})
            record = bool(os.environ.get("OWSP_OBS_DIR"))
            for e, r, out in zip(need, reqs, resp["__batch__"]):
                chunk = np.asarray(out["action"], dtype=np.float32)
                self._set_chunk(e, self._postprocess_chunk(chunk))
                if record:
                    d = self._rec.setdefault(e, {k: [] for k in ("image", "joint", "gripper", "seed", "step", "actions")})
                    d["image"].append(np.asarray(r["observation/image"], np.uint8))
                    d["joint"].append(np.asarray(r["observation/joint_position"], np.float32).reshape(-1))
                    d["gripper"].append(np.asarray(r["observation/gripper_position"], np.float32).reshape(-1))
                    d["seed"].append(np.int64(r["sampling_seed"]))
                    d["step"].append(np.int64(self._step))
                    d["actions"].append(chunk)
        self._step += 1
        try:
            self._video(obs, env_ids)
        except Exception as exc:  # a video problem must never stop an evaluation
            print(f"[BatchedCosmos3Client] video disabled: {exc}", flush=True)
            os.environ.pop("OWSP_VIDEO_DIR", None)
        return {e: {"action": self._next_action(e), "viz": None} for e in env_ids}
