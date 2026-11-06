#!/usr/bin/env python3
"""Package weightsmith results or an exported edit as a Hugging Face Hub folder, with a card.

    # results -> dataset folder (comparisons, held-out runs, evidence record)
    python integrations/hub/export_to_hub.py dataset --out hub/weightsmith-results

    # an edit file (from `weightsmith edit export`) -> model folder, with its held-out runs
    python integrations/hub/export_to_hub.py edit v2mi-f.safetensors --out hub/v2mi-f \\
        --heldout study/results/opus/heldout/cube-then-banana__v2mi-f-RubiksCubeThenBananaTask-90000-128.json

Nothing is uploaded unless you add ``--push OWNER/NAME`` (needs ``pip install 'weightsmith[hub]'`` and a
Hugging Face token); repositories are created private unless ``--public`` is given. Every number in a
generated card is read from the packaged files.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))  # run from a checkout without installing

from weightsmith import edits  # noqa: E402

GITHUB = "https://github.com/adeeb10abbas/weightsmith"
RESULTS = REPO / "study" / "results" / "opus"
COMPARISON_FIELDS = ["level", "task", "label", "editor", "type", "values_changed", "n", "base", "cand", "fixed",
                     "broke", "source", "note"]
HELDOUT_FIELDS = ["file", "cand", "fingerprint", "task", "first_slot", "last_slot", "paired_n", "base_success",
                  "cand_success", "base_rate", "cand_rate", "base_wilson95_low", "base_wilson95_high",
                  "cand_wilson95_low", "cand_wilson95_high", "fixed", "broke", "mcnemar_exact_p", "crashed_slots"]


def _commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return out.stdout.strip() or None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _prepare(out: Path) -> None:
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} exists and is not empty; choose a new folder")
    out.mkdir(parents=True, exist_ok=True)


def _manifest(out: Path, kind: str) -> list[Path]:
    files = sorted(p for p in out.rglob("*") if p.is_file() and p.name != "manifest.json")
    (out / "manifest.json").write_text(json.dumps({
        "kind": kind, "generator": "integrations/hub/export_to_hub.py", "source": GITHUB, "commit": _commit(),
        "files": {str(p.relative_to(out)): {"bytes": p.stat().st_size, "sha256": _sha256(p)} for p in files},
    }, indent=1) + "\n")
    return files + [out / "manifest.json"]


_SUPERSCRIPT = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")


def _p(value: float) -> str:
    """A p-value as printed in the report: 0.012, or 7.2×10⁻²² below 0.001."""
    if value >= 1e-3:
        return f"{value:.3f}"
    mantissa, exponent = f"{value:.1e}".split("e")
    return f"{mantissa}×10{str(int(exponent)).translate(_SUPERSCRIPT)}"


def _heldout_rows(paths: list[Path]) -> list[dict]:
    rows = []
    for path in paths:
        s = json.loads(path.read_text())["summary"]
        rows.append({"file": path.name, "cand": s["cand"], "fingerprint": s["fingerprint"], "task": s["task"],
                     "first_slot": s["slots"][0], "last_slot": s["slots"][1], "paired_n": s["paired_n"],
                     "base_success": s["base_success"], "cand_success": s["cand_success"],
                     "base_rate": s["base_rate"], "cand_rate": s["cand_rate"],
                     "base_wilson95_low": s["base_wilson95"][0], "base_wilson95_high": s["base_wilson95"][1],
                     "cand_wilson95_low": s["cand_wilson95"][0], "cand_wilson95_high": s["cand_wilson95"][1],
                     "fixed": s["fixed"], "broke": s["broke"], "mcnemar_exact_p": s["mcnemar_exact_p"],
                     "crashed_slots": s["crashed_slots"]})
    return rows


def _heldout_table(rows: list[dict]) -> str:
    lines = ["| Task | Candidate | Held-out slots | Base | Edit | Fixed / broke | Exact McNemar p |",
             "|---|---|---|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(f"| `{r['task']}` | `{r['cand']}` | {r['first_slot']}–{r['last_slot']} | "
                     f"{r['base_success']}/{r['paired_n']} | {r['cand_success']}/{r['paired_n']} | "
                     f"{r['fixed']} / {r['broke']} | {_p(r['mcnemar_exact_p'])} |")
    return "\n".join(lines)


def _write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


# ------------------------------------------------------------------------------------------- dataset
def build_dataset(out: Path, results: Path = RESULTS) -> list[Path]:
    """Write a dataset folder: comparisons, held-out runs and the evidence record, plus a card."""
    _prepare(out)
    comparisons = json.loads((results / "comparisons.json").read_text())
    shutil.copy2(results / "comparisons.json", out / "comparisons.json")
    _write_csv(out / "comparisons.csv", COMPARISON_FIELDS, comparisons["rows"])
    heldout = sorted((results / "heldout").glob("*.json"))
    (out / "heldout").mkdir()
    for path in heldout:
        shutil.copy2(path, out / "heldout" / path.name)
    rows = _heldout_rows(heldout)
    _write_csv(out / "heldout.csv", HELDOUT_FIELDS, rows)
    if (results / "EVIDENCE.md").is_file():
        shutil.copy2(results / "EVIDENCE.md", out / "EVIDENCE.md")
    card = f"""---
license: mit
pretty_name: weightsmith results
task_categories:
- robotics
tags:
- robotics
- weight-editing
- vision-language-action
- pi0.5
- robolab
- benchmark
size_categories:
- n<1K
configs:
- config_name: comparisons
  data_files: comparisons.csv
- config_name: heldout
  data_files: heldout.csv
---

# weightsmith results

Paired closed-loop results from [weightsmith]({GITHUB}): LLM editors read and edit the weights of OpenPI π0.5
(`pi05_droid_jointpos`, 3.35B parameters) directly, without training, and every candidate is compared with the
unedited policy on identical simulator seeds in NVIDIA RoboLab (Isaac Sim).

## Held-out runs

Scored by the operator with `weightsmith/heldout.py` on seeds the editor never saw; base and edit alternate run order.

{_heldout_table(rows)}

## Files

| File | Contents |
|---|---|
| `comparisons.csv`, `comparisons.json` | {len(comparisons["rows"])} paired comparisons: level, task, change, editor, \
successes with and without, fixed and broke |
| `heldout.csv`, `heldout/*.json` | {len(rows)} held-out runs; the JSON files are `heldout.py`'s output, unmodified |
| `EVIDENCE.md` | Protocol, budgets, costs and every proposal |
| `manifest.json` | sha256 of every file and the source commit |

Field definitions are in the repository's [results/README.md]({GITHUB}/blob/main/results/README.md). The report
([failure-levels.md]({GITHUB}/blob/main/study/blog/failure-levels.md)) interprets these numbers; the
[leaderboard]({GITHUB}/blob/main/LEADERBOARD.md) ranks them. Cite with the repository's
[CITATION.cff]({GITHUB}/blob/main/CITATION.cff).
"""
    (out / "README.md").write_text(card)
    return _manifest(out, "dataset")


# ------------------------------------------------------------------------------------------- edit
def build_edit(edit_file: Path, out: Path, heldout: list[Path] = ()) -> list[Path]:
    """Write a model folder for one edit file: the file, its metadata, held-out runs and a card."""
    edit = edits.load(edit_file)
    _prepare(out)
    meta = edit.metadata()
    src, base, summary = meta.get("source") or {}, meta.get("base") or {}, meta["summary"]
    name = src.get("candidate") or edit_file.stem
    target = out / f"{name}{edit_file.suffix}"
    shutil.copy2(edit_file, target)
    (out / "edit.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    rows = _heldout_rows(list(heldout))
    if heldout:
        (out / "heldout").mkdir()
        for path in heldout:
            shutil.copy2(path, out / "heldout" / path.name)
    fraction = (f" ({100 * summary['changed_fraction']:.2f}% of {base['total_parameters']:,} parameters)"
                if summary.get("changed_fraction") else "")
    encodings = sorted({t.encoding for t in edit})
    digest = (base.get("editable_sha256") or "?")[:16]
    results = (f"## Held-out results\n\nScored with `weightsmith/heldout.py` on seeds never used while editing.\n\n"
               f"{_heldout_table(rows)}\n\n") if rows else ""
    card = f"""---
license: mit
tags:
- robotics
- weight-editing
- vision-language-action
- pi0.5
- openpi
- weightsmith
---

# `{name}`: a weight edit for π0.5

A weight edit for OpenPI π0.5 (`{base.get("checkpoint") or "pi05_droid_jointpos"}`) in the portable
[weightsmith edit format]({GITHUB}/blob/main/docs/INTEGRATIONS.md#portable-edit-files). It changes
{summary["changed_values"]:,} values in {summary["tensors"]} tensors{fraction}.
{"Applying it reproduces the edited weights bit for bit." if summary["exact"] else
 f"It is lossy: {summary['values_off']:,} values differ from the source after bf16 rounding."}

| | |
|---|---|
| Source candidate | `{name}`, owsp fingerprint `{src.get("fingerprint", "?")}` |
| Base export | `{base.get("workspace_format") or "?"}`, editable sha256 `{digest}…` |
| Encoding | {", ".join(encodings)}; payload {edits._human_bytes(summary["payload_bytes"])} |
| Metadata | `edit.json` (per tensor: shape, changed values, sha256 of base and result, low-rank error) |

{results}## Use it

```bash
pip install "weightsmith @ git+{GITHUB}"
weightsmith edit info {target.name}
weightsmith edit apply {target.name} --workspace $BASE_WORKSPACE --out $WSP_ROOT/cands/{name}
```

`$BASE_WORKSPACE` is the bf16 export of the public checkpoint made by `scripts/prepare_workspace.sh`
([setup]({GITHUB}/blob/main/docs/SETUP.md)). From Python, `weightsmith.edits.load(path)` returns an edit
whose `apply(arrays)` and `apply_to_params(params)` act on numpy arrays or an OpenPI params tree.
"""
    (out / "README.md").write_text(card)
    return _manifest(out, "model")


# ------------------------------------------------------------------------------------------- upload
def push(out: Path, repo_id: str, repo_type: str, public: bool = False) -> str:
    """Upload ``out`` to the Hub (explicit opt-in). Returns the repository URL."""
    try:
        from huggingface_hub import HfApi
    except ImportError:
        raise SystemExit("--push needs huggingface_hub: pip install 'weightsmith[hub]'") from None
    api = HfApi()
    url = api.create_repo(repo_id, repo_type=repo_type, private=not public, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type=repo_type,
                      commit_message="Upload with weightsmith integrations/hub/export_to_hub.py")
    return str(url)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="kind", required=True)
    d = sub.add_parser("dataset", help="results -> dataset folder")
    d.add_argument("--results", type=Path, default=RESULTS,
                   help=f"results folder (default: {RESULTS.relative_to(REPO)})")
    e = sub.add_parser("edit", help="an edit file -> model folder")
    e.add_argument("edit", type=Path, help="file written by `weightsmith edit export`")
    e.add_argument("--heldout", type=Path, nargs="*", default=[], help="heldout.py results.json files for this edit")
    for p in (d, e):
        p.add_argument("--out", type=Path, required=True, help="new folder to write")
        p.add_argument("--push", metavar="OWNER/NAME", help="upload the folder to this Hub repository")
        p.add_argument("--public", action="store_true", help="create the repository public (default: private)")
    a = ap.parse_args(argv)
    files = build_dataset(a.out, a.results) if a.kind == "dataset" else build_edit(a.edit, a.out, a.heldout)
    size = sum(f.stat().st_size for f in files)
    print(f"wrote {a.out}: {len(files)} files, {edits._human_bytes(size)}")
    for f in files:
        print(f"  {f.relative_to(a.out)}")
    if a.push:
        url = push(a.out, a.push, "dataset" if a.kind == "dataset" else "model", a.public)
        print(f"uploaded to {url}")
    else:
        print("dry run: nothing was uploaded (add --push OWNER/NAME to upload)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
