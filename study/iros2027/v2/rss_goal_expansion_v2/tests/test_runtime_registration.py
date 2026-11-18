"""Exercise the real external-file resolver without loading Isaac Sim."""
import importlib.util
from pathlib import Path
import sys
from types import ModuleType
import unittest
from unittest.mock import patch

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
ROBOLAB = Path("/home/ali/vibes/weight-space-policy-lab/robolab_checkout")


class RuntimeRegistrationTests(unittest.TestCase):
    def test_real_resolver_selects_exactly_one_requested_class(self):
        names = ("RSSGCubeBehindTask", "RSSGCubeInBowlTask", "RSSGGlassesPickupTask",
                 "RSSGDeskCubePickupTask", "RSSGMustardLeftBinTask", "RSSGMustardRightBinTask")
        class Task:
            pass
        task_module = ModuleType("robolab.core.task.task")
        task_module.Task = Task
        definitions = ModuleType("tasks")
        for name in names:
            setattr(definitions, name, type(name, (Task,), {}))
        original_path = list(sys.path)
        try:
            with patch.dict(sys.modules, {"robolab.core.task.task": task_module, "tasks": definitions}):
                path = ROBOLAB / "robolab/core/task/task_utils.py"
                spec = importlib.util.spec_from_file_location("rss_real_task_utils", path)
                resolver = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(resolver)
                for name in names:
                    target = str(RUNTIME / (name + ".py"))
                    self.assertEqual(resolver.resolve_task_path(target, ROBOLAB)[1], name)
                    self.assertEqual(len(resolver.load_task_from_file(target, allow_multiple=True)), 1)
        finally:
            sys.path[:] = original_path


if __name__ == "__main__":
    unittest.main()
