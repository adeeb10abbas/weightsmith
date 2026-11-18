import ast
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
ROBOLAB = Path("/home/ali/vibes/weight-space-policy-lab/robolab_checkout")
ISAAC = Path("/home/ali/vibes/weight-space-policy-lab/robolab_env/.venv/lib/python3.11/site-packages/isaaclab/source/isaaclab/isaaclab")


def function(path, name):
    node = next(n for n in ast.walk(ast.parse(path.read_text())) if isinstance(n, ast.FunctionDef) and n.name == name)
    node.decorator_list = []
    namespace = {"torch": SimpleNamespace(Tensor=np.ndarray)}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


spec = importlib.util.spec_from_file_location("rss_full_horizon_scoring", RUNTIME / "scoring.py")
scoring = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scoring)


class FullHorizonTests(unittest.TestCase):
    def test_early_cube_completion_followed_by_banana_motion_fails_full_horizon(self):
        gate = function(RUNTIME / "RSSCubeOnlyFullHorizonTask.py", "at_horizon")
        initial = {"rubiks_cube": [[0., 0., 0.]], "banana": [[1., 0., 0.]], "bowl": [[2., 0., 0.]]}
        s = scoring.Scorer("temporal", initial, 1 / 15)
        pred = {"rubiks_cube_placed": [True], "banana_placed": [False]}
        horizon = 900
        early_stopped = False
        full_terminated_early = False
        for tick in range(1, horizon + 1):
            p = dict(initial)
            if tick >= 120:
                p["banana"] = [[1.05, 0., 0.]]
            s.update(tick, p, pred)
            early_stopped |= bool(s.current["cube_only"][0])
            full = gate(s.current["cube_only"], np.array([tick]), horizon)
            if tick < horizon:
                full_terminated_early |= bool(full[0])
        self.assertTrue(early_stopped)
        self.assertFalse(full_terminated_early)
        self.assertFalse(full[0])
        self.assertTrue(s.ever["cube_only"][0])

    def test_installed_manager_and_robolab_record_simultaneous_timeout_success(self):
        compute = function(ISAAC / "managers/termination_manager.py", "compute")
        true = lambda env: np.array([True])
        manager = SimpleNamespace(_truncated_buf=np.array([False]), _terminated_buf=np.array([False]),
            _term_names=["time_out", "success"], _term_cfgs=[SimpleNamespace(func=true, params={}, time_out=True),
            SimpleNamespace(func=true, params={}, time_out=False)], _term_dones={"time_out": np.array([False]),
            "success": np.array([False])}, _env=None)
        self.assertTrue(compute(manager)[0])
        self.assertTrue(manager._truncated_buf[0])
        self.assertTrue(manager._terminated_buf[0])
        manager.terminated = manager._terminated_buf
        reset_idx = function(ROBOLAB / "robolab/core/environments/env.py", "_reset_idx")
        env = SimpleNamespace(_has_stepped=True, _frozen_envs=np.array([False]), episode_length_buf=np.array([900]),
            termination_manager=manager, _env_results={}, _env_term_step={}, recorder_manager=None)
        reset_idx(env, np.array([0]))
        self.assertIs(env._env_results[0], True)
        self.assertEqual(env._env_term_step[0], 900)


if __name__ == "__main__":
    unittest.main()
