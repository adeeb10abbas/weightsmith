import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from scene_telemetry import install
from screen_worker import batches


def test_abba_matches_exact_starts_with_reversed_order():
    runs = batches({"slot_start": 70000})
    assert runs == [("original", 70000), ("explicit", 70000),
                    ("explicit", 70016), ("original", 70016)]
    assert {start for wording, start in runs if wording == "original"} == {70000, 70016}


@pytest.mark.parametrize("start", [70001, 90000, 120000, 69984, True])
def test_only_declared_development_range_is_available(start):
    with pytest.raises(ValueError):
        batches({"slot_start": start})


class Tensor(np.ndarray):
    def clone(self):
        return self.copy()

    def norm(self, dim):
        return np.linalg.norm(self, axis=dim).view(Tensor)

    def maximum(self, other):
        return np.maximum(self, other).view(Tensor)


def test_telemetry_preserves_reset_step_and_records_reference_motion(tmp_path, monkeypatch):
    class Scene(dict):
        pass

    class Env:
        def __init__(self):
            self.scene = Scene(bowl=SimpleNamespace(data=SimpleNamespace(
                root_pos_w=np.zeros((2, 3)).view(Tensor), root_quat_w=np.ones((2, 4)).view(Tensor))))
            self.scene.rigid_objects = self.scene
            self.scene.env_origins = np.zeros((2, 3)).view(Tensor)
            self.num_envs, self.step_dt = 2, 1 / 15
            self._frozen_envs = np.array([False, False])
            self.actions = []

        def reset(self):
            self.scene["bowl"].data.root_pos_w[:] = 0
            return "original reset result"

        def step(self, action):
            self.actions.append(action)
            self.scene["bowl"].data.root_pos_w[:, 0] += 0.003
            self._frozen_envs[:] = True
            return "original step result"

    monkeypatch.setitem(sys.modules, "robolab.core.environments.env", SimpleNamespace(RobolabEnv=Env))
    monkeypatch.setenv("OWSP_OBS_DIR", str(tmp_path))
    install()
    env = Env()
    assert env.reset() == "original reset result"
    action = object()
    assert env.step(action) == "original step result"
    assert env.actions == [action]
    start = json.loads((tmp_path / "scene-start.json").read_text())
    end = json.loads((tmp_path / "scene-final.json").read_text())
    assert start["objects"]["bowl"]["position"] == [[0, 0, 0], [0, 0, 0]]
    assert end["max_bowl_displacement_m"] == pytest.approx([0.003, 0.003])
