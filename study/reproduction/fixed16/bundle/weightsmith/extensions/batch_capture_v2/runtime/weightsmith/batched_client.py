"""RoboLab client: one websocket request per sim step for all envs needing a replan.

Per-env deterministic noise seeds: ((base + episode)*64 + env_id)*100 + replan_k.
If OWSP_OBS_DIR is set, every policy request (exact 224x224 inputs + seed + sim step)
and returned action chunk is saved per env as obs_r{episode}_e{env}.npz.
"""
from __future__ import annotations

import os
import math

import numpy as np
from policies.pi0_family.client import Pi0DroidJointposClient


class BatchedPi0Client(Pi0DroidJointposClient):
    def _postprocess_chunk(self, chunk):
        result = super()._postprocess_chunk(chunk)
        # Operator-only calibration control. The ordinary policy path retains
        # the upstream postprocessor exactly when the threshold is unchanged.
        threshold = float(os.environ.get("WSP_GRIPPER_THRESHOLD", "0.5"))
        if not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError("Gripper threshold must be finite and strictly between 0 and 1")
        if threshold != 0.5:
            result = result.copy()
            result[..., -1] = (np.asarray(chunk)[..., -1] > threshold).astype(result.dtype)
        return result

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
        values = (self._sampling_seed_base, self._episode_index, env_id, k)
        if any(isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) for value in values):
            raise ValueError("sampling coordinates must be integers")
        if any(int(value) < 0 for value in values):
            raise ValueError("sampling coordinates must be nonnegative")
        # Normalize NumPy integer env IDs at the producer, before msgpack.
        # The server retains its strict Python-int uint32 request contract.
        seed = ((int(self._sampling_seed_base) + int(self._episode_index)) * 64 + int(env_id)) * 100 + int(k)
        if not 0 <= seed < 2**32:
            raise ValueError("sampling seed exceeds uint32 range")
        self._env_replans[env_id] = k + 1
        return seed

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
                    d = self._rec.setdefault(e, {k: [] for k in ("base_img", "wrist_img", "joint", "gripper", "seed", "step", "actions", "prompt", "batch_packet_id", "batch_packet_sha256", "batch_position", "batch_real_count", "batch_padded_count")})
                    d["base_img"].append(np.asarray(r["observation/exterior_image_1_left"], np.uint8))
                    d["wrist_img"].append(np.asarray(r["observation/wrist_image_left"], np.uint8))
                    d["joint"].append(np.asarray(r["observation/joint_position"]).copy())
                    d["gripper"].append(np.asarray(r["observation/gripper_position"]).reshape(-1).copy())
                    d["seed"].append(np.int64(r.get("sampling_seed", -1)))
                    d["step"].append(np.int64(self._step))
                    d["actions"].append(chunk.astype(np.float32))
                    d["prompt"].append(np.asarray(r["prompt"]))
                    packet = out.get("_wsp_batch_v2")
                    if not packet or packet.get("protocol_id") != "wsp-batch-capture-v2-fixed16" or packet.get("padded_count") != 16:
                        raise RuntimeError("v2 recording requires actual fixed16 server packet provenance")
                    d["batch_packet_id"].append(np.asarray(packet["packet_id"]))
                    d["batch_packet_sha256"].append(np.asarray(packet["packet_sha256"]))
                    d["batch_position"].append(np.int64(packet["position"]))
                    d["batch_real_count"].append(np.int64(packet["real_count"]))
                    d["batch_padded_count"].append(np.int64(packet["padded_count"]))
        self._step += 1
        rets = {}
        for e in env_ids:
            rets[e] = {"action": self._next_action(e), "viz": self._build_visualization(extracted[e]) if e == 0 else None}
        return rets
