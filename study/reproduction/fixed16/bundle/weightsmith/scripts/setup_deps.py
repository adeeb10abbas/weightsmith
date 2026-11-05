#!/usr/bin/env python3
"""Install the pinned reference stack inside this dedicated checkout only."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import uuid

REPO = Path(__file__).resolve().parents[1]
PHASES = ("sources", "assets", "policy", "simulator", "checkpoint")
SOURCES = {
    "openpi": ("https://github.com/xuningy/openpi.git", "aa6420561529593114160d05e5ad155792b272f3", "openpi", 1),
    "RoboLab": ("https://github.com/NVlabs/RoboLab.git", "0aef241fb088ca21bb4ebd24448940ed56620d17", "robolab", 6),
}
CHECKPOINT_URI = "gs://openpi-assets-simeval/pi05_droid_jointpos"


def digest_files(paths):
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.name.encode() + b"\0" + path.read_bytes() + b"\0")
    return digest.hexdigest()


def contained(path, root):
    path, root = Path(path).resolve(), Path(root).resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Path must be inside the dedicated checkout: {path}")
    return path


def private_environment(repo, environ=None):
    env = dict(os.environ if environ is None else environ)
    defaults = {"UV_CACHE_DIR": "runtime/cache/uv", "UV_PYTHON_INSTALL_DIR": "runtime/python",
                "PIP_CACHE_DIR": "runtime/cache/pip", "XDG_CACHE_HOME": "runtime/cache/xdg",
                "OPENPI_DATA_HOME": "runtime/cache/openpi", "HF_HOME": "runtime/cache/huggingface",
                "HF_HUB_CACHE": "runtime/cache/huggingface/hub", "HUGGINGFACE_HUB_CACHE": "runtime/cache/huggingface/hub",
                "TRANSFORMERS_CACHE": "runtime/cache/huggingface/transformers",
                "TORCH_HOME": "runtime/cache/torch", "XDG_DATA_HOME": "runtime/data",
                "XDG_CONFIG_HOME": "runtime/config", "XDG_STATE_HOME": "runtime/state",
                "CUDA_CACHE_PATH": "runtime/cache/cuda", "NUMBA_CACHE_DIR": "runtime/cache/numba",
                "MPLCONFIGDIR": "runtime/config/matplotlib", "JAX_COMPILATION_CACHE_DIR": "runtime/cache/jax",
                "TMPDIR": "runtime/tmp"}
    for key, relative in defaults.items():
        path = contained(env.get(key) or repo / relative, repo)
        path.mkdir(parents=True, exist_ok=True)
        env[key] = str(path)
    for key in ("VIRTUAL_ENV", "CONDA_PREFIX", "UV_PROJECT_ENVIRONMENT", "PYTHONPATH", "PYTHONHOME",
                "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    env.update(GIT_CONFIG_GLOBAL=os.devnull, GIT_CONFIG_NOSYSTEM="1", GIT_LFS_SKIP_SMUDGE="1",
               UV_LINK_MODE="hardlink", PYTHONNOUSERSITE="1")
    return env


class Installer:
    def __init__(self, repo=REPO):
        self.repo = Path(repo).resolve()
        self.runtime = contained(self.repo / "runtime", self.repo)
        self.runtime.mkdir(parents=True, exist_ok=True)
        self.state_path = self.runtime / "setup-state.json"
        if self.state_path.exists():
            self.state = json.loads(self.state_path.read_text())
            if self.state.get("repo") != str(self.repo):
                raise RuntimeError("Setup state belongs to a different deployment")
        else:
            self.state = {"repo": str(self.repo), "owner_token": uuid.uuid4().hex, "sources": {}, "phases": {}}
        self.env = private_environment(self.repo)

    def save(self):
        temporary = self.state_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self.state, indent=2) + "\n")
        temporary.replace(self.state_path)

    def run(self, args, *, cwd=None, capture=False, env=None):
        print("+ " + shlex.join(str(v) for v in args), flush=True)
        result = subprocess.run([str(v) for v in args], cwd=cwd or self.repo, env=env or self.env,
                                text=True, stdout=subprocess.PIPE if capture else None, check=True)
        return result.stdout.strip() if capture else ""

    def git(self, path, *args, capture=False):
        return self.run(["git", "-C", path, *args], capture=capture)

    def source_details(self, name):
        url, base, patchdir, count = SOURCES[name]
        patches = sorted((self.repo / "third_party/patches" / patchdir).glob("*.patch"))
        if len(patches) != count:
            raise RuntimeError(f"Expected {count} patches for {name}; found {len(patches)}")
        return {"url": url, "base": base, "patch_digest": digest_files(patches)}, patches

    def require_source(self, name):
        path = contained(self.repo / "third_party" / name, self.repo)
        if (path / ".git").is_symlink():
            raise RuntimeError(f"Refusing symlinked Git metadata: {path}")
        entry = self.state["sources"].get(name)
        marker = path / ".git/weightsmith-owner.json"
        expected, _ = self.source_details(name)
        if not entry or not marker.is_file() or json.loads(marker.read_text()) != {
            "repo": str(self.repo), "owner_token": self.state["owner_token"], "source": name
        }:
            raise RuntimeError(f"Refusing an unowned source checkout: {path}")
        if any(entry.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Pins or patches changed for {name}; use a fresh deployment")
        head = self.git(path, "rev-parse", "HEAD", capture=True)
        if head != entry.get("head") or entry.get("stage") != "ready":
            raise RuntimeError(f"{name} is incomplete or its HEAD changed; inspect setup-state.json without resetting it")
        if self.git(path, "status", "--porcelain", "--untracked-files=no", capture=True):
            raise RuntimeError(f"Refusing to modify dirty source checkout: {path}")
        return path

    def sources(self):
        for name in SOURCES:
            path = self.repo / "third_party" / name
            if path.exists() or path.is_symlink():
                self.require_source(name)
                print(f"{name}: existing owned pinned source verified", flush=True)
                continue
            expected, patches = self.source_details(name)
            self.state["sources"][name] = {**expected, "stage": "cloning"}
            self.save()
            path.parent.mkdir(parents=True, exist_ok=True)
            self.run(["git", "clone", "--filter=blob:none", "--no-checkout", expected["url"], path])
            contained(path, self.repo)
            (path / ".git/weightsmith-owner.json").write_text(json.dumps({
                "repo": str(self.repo), "owner_token": self.state["owner_token"], "source": name}))
            self.git(path, "lfs", "install", "--local", "--skip-smudge")
            self.git(path, "checkout", "--detach", expected["base"])
            self.git(path, "-c", "user.name=weightsmith", "-c", "user.email=weightsmith@users.noreply.github.com",
                     "am", "--committer-date-is-author-date", *patches)
            self.git(path, "submodule", "update", "--init", "--recursive")
            self.state["sources"][name].update(stage="ready", head=self.git(path, "rev-parse", "HEAD", capture=True))
            self.save()

    def assets(self, purge=False):
        path = self.require_source("RoboLab")
        self.git(path, "-c", f"lfs.storage={path / '.git/lfs'}", "lfs", "pull")
        if purge:
            self.purge_lfs_objects(path)

    def purge_lfs_objects(self, path):
        expected = self.require_source("RoboLab")
        if Path(path).resolve() != expected:
            raise RuntimeError("LFS cleanup is restricted to the owned RoboLab checkout")
        objects = path / ".git/lfs/objects"
        if objects.is_symlink() or (path / ".git/lfs").is_symlink():
            raise RuntimeError("Refusing symlinked LFS cache cleanup")
        contained(objects, path / ".git")
        if objects.exists():
            shutil.rmtree(objects)
        print("Removed only owned RoboLab .git/lfs/objects; working-tree assets retained", flush=True)

    def sync(self, project):
        project = contained(project, self.repo)
        lock = project / "uv.lock"
        before = hashlib.sha256(lock.read_bytes()).hexdigest()
        venv = project / ".venv"
        if venv.is_symlink():
            raise RuntimeError(f"Refusing symlinked virtual environment: {venv}")
        key = str(project.relative_to(self.repo))
        owned = self.state.setdefault("environments", {})
        if venv.exists() and key not in owned:
            raise RuntimeError(f"Refusing an existing unowned virtual environment: {venv}")
        if key in owned and owned[key] != {"path": str(venv), "lock_sha256": before}:
            raise RuntimeError(f"Environment lock or path changed: {venv}")
        owned[key] = {"path": str(venv), "lock_sha256": before}
        self.save()
        env = dict(self.env, UV_PROJECT_ENVIRONMENT=str(venv))
        self.run(["uv", "sync", "--frozen", "--python", "3.11"], cwd=project, env=env)
        if hashlib.sha256(lock.read_bytes()).hexdigest() != before:
            raise RuntimeError(f"Lockfile unexpectedly changed: {lock}")

    def policy(self):
        self.sync(self.require_source("openpi"))

    def simulator(self):
        self.require_source("openpi")
        self.require_source("RoboLab")
        self.sync(self.repo / "envs/robolab-sim")

    def checkpoint(self):
        source = self.require_source("openpi")
        python = source / ".venv/bin/python"
        location = self.runtime / "checkpoint-location.json"
        code = ("import json,sys; from openpi.shared import download; "
                "from pathlib import Path; p=download.maybe_download(sys.argv[1]); "
                "Path(sys.argv[2]).write_text(json.dumps({'path':str(p)}))")
        self.run([python, "-c", code, CHECKPOINT_URI, location], cwd=source)
        checkpoint = contained(json.loads(location.read_text())["path"], self.repo)
        verified = verify_checkpoint(checkpoint, self.repo / "third_party/pi05_droid_jointpos.sha256")
        self.state["checkpoint"] = {"uri": CHECKPOINT_URI, "path": str(checkpoint), **verified}
        self.save()
        print(f"Checkpoint SHA256 verified: {verified['files']} files", flush=True)

    def execute(self, phases, purge=False):
        self.save()
        for phase in phases:
            print(f"=== phase {phase} ===", flush=True)
            self.state["phases"][phase] = {"status": "running", "started": time.time()}
            self.save()
            try:
                self.assets(purge) if phase == "assets" else getattr(self, phase)()
            except BaseException as exc:
                self.state["phases"][phase].update(status="failed", error=str(exc), finished=time.time())
                self.save()
                raise
            self.state["phases"][phase].update(status="complete", finished=time.time())
            self.save()


def verify_checkpoint(checkpoint, manifest):
    lines = [line for line in manifest.read_text().splitlines() if line.strip()]
    if len(lines) != 26:
        raise RuntimeError(f"Expected the 26-file reference checksum manifest, got {len(lines)} entries")
    size = 0
    for line in lines:
        expected, name = line.split(maxsplit=1)
        path = contained(checkpoint / name.removeprefix("*"), checkpoint)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
                size += len(block)
        if digest.hexdigest() != expected:
            raise RuntimeError(f"Checkpoint checksum mismatch: {name}")
    return {"files": len(lines), "bytes": size, "manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest()}


def selected_phases(phase, no_sync):
    if no_sync:
        if phase not in ("all", "sources"):
            raise ValueError("--no-sync can only be combined with --phase sources")
        return ("sources",)
    return PHASES if phase == "all" else (phase,)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("all", *PHASES), default="all")
    parser.add_argument("--no-sync", action="store_true", help="Sources only: no LFS assets, environments or checkpoint")
    parser.add_argument("--purge-lfs-cache", action="store_true", help="After assets, remove only the owned duplicate LFS object cache")
    args = parser.parse_args()
    try:
        phases = selected_phases(args.phase, args.no_sync)
        runtime = contained(REPO / "runtime", REPO)
        runtime.mkdir(exist_ok=True)
        with (runtime / "setup.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            installer = Installer()
            installer.execute(phases, args.purge_lfs_cache)
        print(json.dumps({"completed_phases": phases, "state": str(installer.state_path),
                          "OPENPI_DIR": str(REPO / "third_party/openpi"),
                          "OPENPI_PYTHON": str(REPO / "third_party/openpi/.venv/bin/python"),
                          "ROBOLAB_DIR": str(REPO / "third_party/RoboLab"),
                          "SIM_PYTHON": str(REPO / "envs/robolab-sim/.venv/bin/python"),
                          "OPENPI_DATA_HOME": installer.env["OPENPI_DATA_HOME"],
                          "CHECKPOINT_DIR": installer.state.get("checkpoint", {}).get("path")}, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"setup failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
