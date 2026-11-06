#!/usr/bin/env python3
"""Run a command with private caches, a disk reserve, and an optional GPU guard."""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


def private_directory(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Isolation path escapes the dedicated checkout: {path}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def isolated_environment(root, source=None):
    """Scope application state without changing HOME or any global config."""
    env = dict(os.environ if source is None else source)
    for key in ("PYTHONHOME", "PYTHONPATH", "VIRTUAL_ENV", "CONDA_PREFIX", "CONDA_DEFAULT_ENV",
                "UV_PROJECT_ENVIRONMENT", "UV_ACTIVE", "PIP_TARGET", "PIP_PREFIX", "PIP_USER"):
        env.pop(key, None)
    paths = {
        "XDG_CACHE_HOME": "runtime/cache", "XDG_CONFIG_HOME": "runtime/config",
        "XDG_DATA_HOME": "runtime/data", "XDG_STATE_HOME": "runtime/state",
        "XDG_RUNTIME_DIR": "runtime/run", "OMNI_CONFIG_PATH": "runtime/omni-config",
        "UV_CACHE_DIR": "runtime/cache/uv", "UV_PYTHON_INSTALL_DIR": "runtime/python",
        "UV_TOOL_DIR": "runtime/tools", "UV_TOOL_BIN_DIR": "runtime/bin",
        "PIP_CACHE_DIR": "runtime/cache/pip", "PYTHONUSERBASE": "runtime/python-user",
        "PYTHONPYCACHEPREFIX": "runtime/cache/pycache",
        "HF_HOME": "runtime/cache/huggingface", "HUGGINGFACE_HUB_CACHE": "runtime/cache/huggingface/hub",
        "HF_HUB_CACHE": "runtime/cache/huggingface/hub", "TRANSFORMERS_CACHE": "runtime/cache/huggingface/transformers",
        "TORCH_HOME": "runtime/cache/torch", "NUMBA_CACHE_DIR": "runtime/cache/numba",
        "MPLCONFIGDIR": "runtime/config/matplotlib", "CUDA_CACHE_PATH": "runtime/cache/cuda",
        "__GL_SHADER_DISK_CACHE_PATH": "runtime/cache/nvidia-gl", "OPTIX_CACHE_PATH": "runtime/cache/optix",
        "TRITON_CACHE_DIR": "runtime/cache/triton", "OPENPI_DATA_HOME": "runtime/cache/openpi",
        "WSP_JAX_CACHE": "runtime/cache/jax", "JAX_COMPILATION_CACHE_DIR": "runtime/cache/jax",
        "TMPDIR": "runtime/tmp",
    }
    for name, relative in paths.items():
        env[name] = str(private_directory(root, relative))
    Path(env["XDG_RUNTIME_DIR"]).chmod(0o700)
    omni_paths = {name: str(private_directory(root, "runtime/omni/" + name))
                  for name in ("data", "cache", "logs")}
    config = Path(env["OMNI_CONFIG_PATH"]) / "omniverse.toml"
    if config.is_symlink():
        raise ValueError(f"Refusing symlinked private Omniverse configuration: {config}")
    config.write_text("[paths]\n" + "".join(f"{name}_root = {json.dumps(path)}\n" for name, path in omni_paths.items()))
    private_directory(root, "runtime/kit")
    native = root / "runtime/sysroot/usr/lib/x86_64-linux-gnu"
    if native.exists() or native.is_symlink():
        if not native.resolve().is_relative_to(root) or not native.is_dir():
            raise ValueError(f"Private native-library path escapes checkout or is not a directory: {native}")
        env["LD_LIBRARY_PATH"] = str(native) + (os.pathsep + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
    env.update(PYTHONNOUSERSITE="1", PIP_CONFIG_FILE=os.devnull,
               UV_LINK_MODE="hardlink", WSP_MANAGED_PROCESS_GROUP="1",
               UV_CONCURRENT_DOWNLOADS="2", UV_CONCURRENT_INSTALLS="2", UV_CONCURRENT_BUILDS="2",
               OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", MAX_JOBS="2",
               WSP_ISOLATED_ROOT=str(root), WSP_ENV_FILE=str(root / "wsp.env"),
               WSP_KIT_PORTABLE_ROOT=str(root / "runtime/kit"))
    env["PATH"] = str(root / "runtime/bin") + ":/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
    return env


def gpu_processes(gpu):
    result = subprocess.run(["nvidia-smi", "-i", str(gpu), "--query-compute-apps=pid",
                             "--format=csv,noheader,nounits"], text=True, capture_output=True, timeout=10)
    if result.returncode:
        raise RuntimeError("Cannot verify GPU availability: " + result.stderr.strip()[-1000:])
    pids = set()
    for line in result.stdout.splitlines():
        line = line.strip()
        if not line or line == "No running processes found":
            continue
        if not line.isdecimal():
            raise RuntimeError(f"Cannot identify GPU process: {line!r}")
        pids.add(int(line))
    return pids


def foreign_gpu_processes(gpu, group=None):
    foreign = set()
    for pid in gpu_processes(gpu):
        try:
            own = group is not None and os.getpgid(pid) == group
        except ProcessLookupError:
            # NVIDIA may expose host PIDs not resolvable in this container. That
            # is not sufficient evidence that a workload belongs to this run.
            own = False
        if not own:
            foreign.add(pid)
    return foreign


def check_resources(root, min_free_gib, gpu=None, group=None):
    free = shutil.disk_usage(root).free
    if free < min_free_gib * 2**30:
        raise RuntimeError(f"Container disk reserve reached: {free / 2**30:.2f} GiB free, minimum {min_free_gib:g} GiB")
    if gpu is not None:
        devices = [gpu] if isinstance(gpu, int) else gpu
        for device in devices:
            foreign = foreign_gpu_processes(device, group)
            if foreign:
                raise RuntimeError(f"GPU {device} has unrelated or unidentifiable process IDs: {sorted(foreign)}")


def stop_group(proc, grace=5.0):
    """The child is the group leader; no unrelated process group is targeted."""
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    try:
        proc.wait(timeout=grace)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    proc.wait()


def run_guarded(command, root, min_free_gib=8.0, gpu=None, interval=1.0):
    check_resources(root, min_free_gib, gpu)
    env = isolated_environment(root)
    proc = subprocess.Popen(command, cwd=root, env=env, start_new_session=True)
    try:
        try:
            os.setpriority(os.PRIO_PROCESS, proc.pid, 10)
        except ProcessLookupError:
            pass
        while proc.poll() is None:
            check_resources(root, min_free_gib, gpu, proc.pid)
            time.sleep(interval)
        return proc.returncode
    finally:
        # Includes orphaned workers in the group after the command leader exits.
        stop_group(proc)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path, help="Dedicated existing weightsmith checkout")
    ap.add_argument("--min-free-gib", type=float, default=8.0)
    ap.add_argument("--gpu", type=int, action="append", help="Refuse or stop on foreign compute jobs; repeat for multiple GPUs")
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args(argv)
    root = args.root.expanduser().resolve()
    if not (root / "weightsmith/config.py").is_file():
        ap.error("--root must be an existing dedicated weightsmith checkout")
    if not math.isfinite(args.min_free_gib) or args.min_free_gib <= 0:
        ap.error("--min-free-gib must be positive and finite")
    if args.gpu is not None and any(gpu < 0 for gpu in args.gpu):
        ap.error("--gpu must be nonnegative")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        ap.error("provide a command after --")

    def interrupted(signum, frame):
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        return run_guarded(command, root, args.min_free_gib, args.gpu)
    except KeyboardInterrupt:
        print("isolated run interrupted; owned worker group stopped", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, subprocess.TimeoutExpired) as exc:
        print(f"isolated run stopped: {exc}", file=sys.stderr)
        return 1
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == "__main__":
    raise SystemExit(main())
