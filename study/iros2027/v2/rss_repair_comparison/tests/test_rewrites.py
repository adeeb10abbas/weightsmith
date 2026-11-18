import importlib.util
from pathlib import Path
import unittest

p = Path(__file__).resolve().parents[1] / "rewrites.py"
spec = importlib.util.spec_from_file_location("rss_rewrites", p)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class RewriteTests(unittest.TestCase):
    def test_order_minimal_pairs(self):
        for connective in ["then", "before", "next"]:
            self.assertEqual(m.intent(f"Put cube in bowl {connective} banana in bowl"), "cube_then_banana")
            self.assertEqual(m.intent(f"Put banana in bowl {connective} cube in bowl"), "banana_then_cube")
        self.assertEqual(m.intent("Banana goes in bowl after cube"), "cube_then_banana")
        self.assertFalse(m.route("Put banana in bowl then cube in bowl")["rewritten"])

    def test_source_rewrites(self):
        self.assertEqual(m.route(m.TEMPORAL_SOURCE)["effective_prompt"], m.TEMPORAL_TARGET)
        self.assertEqual(m.route(m.SPATIAL_SOURCE)["effective_prompt"], m.SPATIAL_TARGET)

    def test_inverse_spatial_language(self):
        self.assertEqual(m.intent("Move mustard right of raisin"), "mustard_right")
        self.assertEqual(m.intent("Move mustard so raisin is left of mustard"), "mustard_right")
        self.assertFalse(m.route("Move mustard left of raisin")["rewritten"])
        self.assertFalse(m.route("Move raisin left of mustard")["rewritten"])

    def test_abstention(self):
        for request in ["Put only cube in bowl and leave banana", "Do not put cube then banana in bowl",
                        "Move mustard not right of raisin", "Move cube and banana in bowl",
                        "Place mustard on the raisin box"]:
            self.assertFalse(m.route(request)["rewritten"])


if __name__ == "__main__":
    unittest.main()
