#!/usr/bin/env bash
# Check published weightsmith edit files from a checkout of this repository.
#
#   edits/verify.sh edits/ctrl-ctb-p1.safetensors              # checksum, metadata, held-out record
#   edits/verify.sh --download v2mi-f abl-imgl8                 # fetch from the edits-v1 release first
#   edits/verify.sh v2mi-f.safetensors --workspace $BASE_WORKSPACE   # also apply it and check the result
#
# For each file it checks that:
#   1. its sha256 is one of the published checksums in edits/SHA256SUMS;
#   2. it is an exact weightsmith.edit file, and the owsp fingerprint it records belongs to a held-out
#      record under study/results/*/heldout/ (the run that scored it);
#   3. with --workspace, applying it to your base export (scripts/prepare_workspace.sh) reproduces that
#      fingerprint, with every tensor matching its recorded sha256. The candidate is written to a
#      temporary directory (honours TMPDIR; v2mi-f needs about 2.3 GB) and deleted afterwards.
#
# Needs python3 with numpy. The weightsmith package is taken from this checkout. --download uses
# `gh release download` when gh is installed and logged in, otherwise curl.
# Exit status: 0 when every check passed, 1 otherwise.
set -euo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$repo${PYTHONPATH:+:$PYTHONPATH}"
exec "${PYTHON:-python3}" - "$repo" "$@" <<'PY'
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

repo = Path(sys.argv[1])
TAG, GH_REPO = "edits-v1", "adeeb10abbas/weightsmith"
URL = f"https://github.com/{GH_REPO}/releases/download/{TAG}/"

ap = argparse.ArgumentParser(prog="edits/verify.sh", description="Check published weightsmith edit files.")
ap.add_argument("files", nargs="+", metavar="FILE",
                help="an edit file, or the name of a published edit (for example v2mi-f)")
ap.add_argument("--workspace", type=Path,
                help="base export ($BASE_WORKSPACE): also apply each edit there and compare fingerprints")
ap.add_argument("--download", action="store_true",
                help=f"fetch published edits that are not found locally from the {TAG} release")
ap.add_argument("--dir", type=Path, default=repo / "edits" / "downloads",
                help="where --download saves files (default: edits/downloads/, which Git ignores)")
a = ap.parse_args(sys.argv[2:])

try:
    from weightsmith import edits
except ImportError as e:
    sys.exit(f"verify.sh: cannot import weightsmith from {repo} ({e}); install numpy")

published = {}  # file name -> sha256
for line in (repo / "edits" / "SHA256SUMS").read_text().splitlines():
    if line.strip():
        digest, name = line.split(maxsplit=1)
        published[name.lstrip("*")] = digest


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def download(name):
    a.dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".download-", dir=a.dir))
    try:
        print(f"  downloading {name} from the {TAG} release ...", flush=True)
        got = shutil.which("gh") and subprocess.run(
            ["gh", "release", "download", TAG, "-R", GH_REPO, "-p", name, "-D", str(tmp)]).returncode == 0
        if not got and shutil.which("curl"):
            got = subprocess.run(["curl", "-fL", "--progress-bar", "-o", str(tmp / name), URL + name]).returncode == 0
        if not got or not (tmp / name).is_file():
            return None
        (tmp / name).replace(a.dir / name)
        return a.dir / name
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def locate(arg):
    path = Path(arg)
    if path.is_file():
        return path
    name = path.name if path.name.endswith(".safetensors") else path.name + ".safetensors"
    for folder in (repo / "edits", a.dir):
        if (folder / name).is_file():
            return folder / name
    if a.download and name in published:
        return download(name)
    return None


def heldout_records(fingerprint):
    for f in sorted(repo.glob("study/results/*/heldout/*.json")):
        if f.name.endswith(".plan.json"):
            continue
        try:
            s = json.loads(f.read_text()).get("summary") or {}
        except (OSError, ValueError):
            continue
        if s.get("fingerprint") == fingerprint:
            yield f.relative_to(repo), s


failed = 0
for arg in a.files:
    print(arg)
    path = locate(arg)
    if path is None:
        hint = "" if a.download else "; pass --download to fetch a published edit"
        print(f"  FAIL  not found{hint}")
        failed += 1
        continue
    ok = True
    digest = sha256(path)
    names = sorted(n for n, d in published.items() if d == digest)
    if names:
        print(f"  ok    sha256 {digest[:16]}... matches the published {names[0]} ({path})")
    else:
        print(f"  FAIL  sha256 {digest} of {path} is not in edits/SHA256SUMS")
        ok = False
    try:
        edit = edits.load(path)
    except Exception as e:  # noqa: BLE001 - not an edit file, or a damaged one
        print(f"  FAIL  not a readable weightsmith edit: {e}")
        failed += 1
        continue
    summary, source = edit.summary(), edit.meta.get("source") or {}
    fingerprint = source.get("fingerprint")
    print(f"  {'ok  ' if summary['exact'] else 'FAIL'}  edit of candidate {source.get('candidate', '?')}: "
          f"{summary['changed_values']:,} values in {summary['tensors']} tensor(s), "
          f"exact: {'yes' if summary['exact'] else 'no'}, fingerprint {fingerprint}")
    ok = ok and summary["exact"]
    records = list(heldout_records(fingerprint)) if fingerprint else []
    for rel, s in records:
        print(f"  ok    held-out record {rel}: {s['task']}, slots {s['slots'][0]}-{s['slots'][1]}, "
              f"{s['base_success']}/{s['paired_n']} -> {s['cand_success']}/{s['paired_n']} "
              f"({s['fixed']} fixed, {s['broke']} broke, exact McNemar p = {s['mcnemar_exact_p']:.2g})")
    if not records:
        print(f"  FAIL  no held-out record under study/results/*/heldout/ has fingerprint {fingerprint}")
        ok = False
    if a.workspace:
        tmp = Path(tempfile.mkdtemp(prefix="wsverify-"))
        try:
            report = edits.apply_to_workspace(edit, a.workspace, tmp / "candidate")
            if report["fingerprint"] == fingerprint:
                print(f"  ok    applied to {a.workspace}: every tensor matches its recorded sha256, "
                      f"fingerprint {report['fingerprint']}")
            else:
                print(f"  FAIL  applied to {a.workspace}: fingerprint {report['fingerprint']}, expected {fingerprint}")
                ok = False
        except (edits.EditError, OSError) as e:
            print(f"  FAIL  apply to {a.workspace}: {e}")
            ok = False
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    else:
        print("  skip  apply: pass --workspace $BASE_WORKSPACE to check the applied fingerprint")
    failed += not ok

print(f"{len(a.files) - failed} of {len(a.files)} passed" if failed else f"all {len(a.files)} passed")
sys.exit(1 if failed else 0)
PY
