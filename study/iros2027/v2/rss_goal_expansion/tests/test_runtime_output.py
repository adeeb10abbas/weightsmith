import ast
import importlib.util
from pathlib import Path
import unittest

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
spec = importlib.util.spec_from_file_location("rss_output_redirect", RUNTIME / "output_redirect.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RuntimeOutputTests(unittest.TestCase):
    def test_actual_runner_changes_only_the_output_assignment(self):
        source = Path("/home/ali/vibes/weight-space-policy-lab/robolab_checkout/robolab/eval/runner.py").read_text()
        updated = module.rewrite_source(source)
        self.assertEqual(updated.replace(module.REPLACEMENT, module.ORIGINAL), source)
        ast.parse(updated)
        self.assertIn('from robolab.constants import PACKAGE_DIR', updated)
        with self.assertRaises(RuntimeError):
            module.rewrite_source(updated)


if __name__ == "__main__":
    unittest.main()
