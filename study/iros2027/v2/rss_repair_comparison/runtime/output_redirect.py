"""Scoped output relocation without mutating RoboLab's asset root or checkout.

The installed runner hardcodes PACKAGE_DIR/output. Compile a process-local copy
changing exactly that one assignment; fail if the upstream implementation drifts.
"""
import inspect
import os
from pathlib import Path

ORIGINAL = 'output_dir = os.path.join(PACKAGE_DIR, "output", output_folder_name)'
REPLACEMENT = 'output_dir = os.path.join(os.environ["RSS_OUTPUT_DIR"], output_folder_name)'


def rewrite_source(source):
    if source.count(ORIGINAL) != 1:
        raise RuntimeError("RoboLab output assignment changed; refusing a broad runtime patch")
    return source.replace(ORIGINAL, REPLACEMENT, 1)


def redirected_runner(stock):
    if not os.environ.get("RSS_OUTPUT_DIR"):
        raise RuntimeError("RSS_OUTPUT_DIR must name the new cohort's local output root")
    output = Path(os.environ["RSS_OUTPUT_DIR"]).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["RSS_OUTPUT_DIR"] = str(output)
    source = rewrite_source(inspect.getsource(stock))
    namespace = dict(stock.__globals__)
    exec(compile(source, "<RSS output-only copy of RoboLab run_evaluation>", "exec"), namespace)
    print(f"[rss] scoped output root: {output}", flush=True)
    return namespace[stock.__name__]
