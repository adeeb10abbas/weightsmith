import copy
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np

RUNTIME = Path(__file__).resolve().parents[1] / "runtime"
spec = importlib.util.spec_from_file_location("rss_fixed_batch_server", RUNTIME / "batched_server.py")
fixed = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fixed)
OriginalServer = fixed.Server.__bases__[0]


class FixedBatchTests(unittest.TestCase):
    def server(self):
        # The wrapper is tested without model construction, JAX, or any GPU work.
        return object.__new__(fixed.Server)

    def test_all_active_set_sizes_keep_order_seed_and_fixed_kernel(self):
        for n in (1, 2, 3, 8, 9, 15, 16):
            with self.subTest(n=n):
                requests = [{"sampling_seed": 100 + i, "prompt": f"request{i}",
                             "image": np.array([i], dtype=np.uint8)} for i in range(n)]
                observed, parent_outputs = [], []

                def infer(_server, batch):
                    observed.extend(batch)
                    parent_outputs.extend({"actions": np.array([i, row["sampling_seed"]]),
                                           "sentinel": row["prompt"],
                                           "policy_timing": {"infer_ms": 7.25, "batch": len(batch)}}
                                          for i, row in enumerate(batch))
                    return parent_outputs

                with patch.object(OriginalServer, "infer_batch", infer):
                    result = self.server().infer_batch(requests)
                self.assertEqual(len(observed), 16)
                self.assertEqual([r["sampling_seed"] for r in observed[:n]], list(range(100, 100 + n)))
                self.assertEqual([r["sampling_seed"] for r in observed[n:]], [100] * (16 - n))
                self.assertEqual(len(result), n)
                self.assertEqual([r["sentinel"] for r in result], [f"request{i}" for i in range(n)])
                for i, out in enumerate(result):
                    np.testing.assert_array_equal(out["actions"], [i, 100 + i])
                    self.assertIs(out["actions"], parent_outputs[i]["actions"])
                    self.assertEqual(out["policy_timing"],
                                     {"infer_ms": 7.25, "batch": n, "requested_n": n, "kernel_batch": 16})
                self.assertTrue(all(out["policy_timing"] == {"infer_ms": 7.25, "batch": 16}
                                    for out in parent_outputs))

    def test_parent_mutations_cannot_change_callers_or_alias_padding_rows(self):
        request = {"sampling_seed": 314, "nested": {"image": np.array([7, 8], dtype=np.uint8)}}
        original = copy.deepcopy(request)

        def infer(_server, batch):
            batch[0]["sampling_seed"] = -1
            batch[0]["nested"]["image"][0] = 99
            for row in batch[1:]:
                self.assertEqual(row["sampling_seed"], 314)
                np.testing.assert_array_equal(row["nested"]["image"], [7, 8])
            return [{"actions": i} for i in range(16)]

        requests = [request]
        with patch.object(OriginalServer, "infer_batch", infer):
            self.server().infer_batch(requests)
        self.assertEqual(len(requests), 1)
        self.assertIs(requests[0], request)
        self.assertEqual(request["sampling_seed"], original["sampling_seed"])
        np.testing.assert_array_equal(request["nested"]["image"], original["nested"]["image"])

    def test_invalid_batch_sizes_fail_before_inference(self):
        with patch.object(OriginalServer, "infer_batch") as infer:
            for n in (0, 17):
                with self.subTest(n=n), self.assertRaises(ValueError):
                    self.server().infer_batch([{}] * n)
            infer.assert_not_called()

    def test_wrong_parent_output_count_is_not_silently_truncated(self):
        with patch.object(OriginalServer, "infer_batch", return_value=[{}] * 15):
            with self.assertRaises(RuntimeError):
                self.server().infer_batch([{"sampling_seed": 1}])

    def test_main_uses_wrapper_class_with_original_entrypoint(self):
        # Restore the module attribute after inspecting the process-local patch.
        with patch.object(fixed.base, "Server", OriginalServer):
            with patch.object(fixed.base, "main") as main:
                main.side_effect = lambda: self.assertIs(fixed.base.Server, fixed.Server)
                fixed.main()
                main.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
