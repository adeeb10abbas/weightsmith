#!/usr/bin/env python3
"""Extract pinned Ubuntu 24.04 libraries privately; never install system packages."""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import subprocess
import tarfile
import tempfile

REPO = Path(__file__).resolve().parents[1]
PACKAGES = (
    {"name": "libglu1-mesa", "version": "9.0.2-1.1build1", "sha256": "4288833eebfbf6b00d1ae98159f429e1f7f6398288c9305edeeb185d9ed048d9"},
    {"name": "libxt6t64", "version": "1:1.2.1-1.2build1", "sha256": "2e4317d5ae1a45e517771ea0b8d93519b5f915ab08983facecd480d64dbd1489"},
    {"name": "libopengl0", "version": "1.7.0-1build1", "sha256": "da36bd4d30c13b40b697fa8b72cc09d309ff36a5bd9750341ef78492bf3f3966"},
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def safe_path(root, relative):
    path = root / relative
    for part in (path, *path.parents):
        if part == root:
            break
        if part.is_symlink():
            raise ValueError(f"Refusing symlinked setup path: {part}")
    if not path.resolve().is_relative_to(root):
        raise ValueError(f"Setup path escapes checkout: {path}")
    return path


def require_host():
    if platform.system() != "Linux" or platform.machine() not in ("x86_64", "amd64"):
        raise RuntimeError("These native-library pins require Ubuntu 24.04 amd64")
    values = dict(line.split("=", 1) for line in Path("/etc/os-release").read_text().splitlines() if "=" in line)
    if values.get("ID", "").strip('"') != "ubuntu" or values.get("VERSION_ID", "").strip('"') != "24.04":
        raise RuntimeError("These native-library pins require Ubuntu 24.04; do not substitute another distribution")


def obtain_package(debs, pin):
    filename = f"{pin['name']}_{pin['version'].replace(':', '_')}_amd64.deb"
    destination = debs / filename
    for candidate in sorted(debs.glob("*.deb")):
        if candidate.is_symlink():
            raise ValueError(f"Refusing symlinked package cache file: {candidate}")
        if digest(candidate) == pin["sha256"]:
            return candidate
    if destination.exists():
        raise RuntimeError(f"Cached package has the wrong checksum: {destination}")
    with tempfile.TemporaryDirectory(prefix="download-", dir=debs) as temporary:
        subprocess.run(["apt-get", "download", f"{pin['name']}={pin['version']}"], cwd=temporary, check=True)
        downloads = list(Path(temporary).glob("*.deb"))
        if len(downloads) != 1 or downloads[0].is_symlink() or digest(downloads[0]) != pin["sha256"]:
            raise RuntimeError(f"Downloaded package does not match pinned SHA256: {pin['name']}")
        downloads[0].replace(destination)
    return destination


def validate_archive_members(members):
    paths, links = [], set()
    for member in members:
        path = PurePosixPath(member.name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Unsafe package archive path: {member.name}")
        if not (member.isfile() or member.isdir() or member.issym()):
            raise ValueError(f"Unsupported package archive entry: {member.name}")
        if member.issym():
            target = PurePosixPath(member.linkname)
            if target.is_absolute():
                raise ValueError(f"Unsafe package symlink: {member.name} -> {member.linkname}")
            resolved = list(path.parent.parts)
            for part in target.parts:
                if part == "..":
                    if not resolved:
                        raise ValueError(f"Unsafe package symlink: {member.name} -> {member.linkname}")
                    resolved.pop()
                else:
                    resolved.append(part)
            links.add(path)
        paths.append(path)
    if any(parent in links for path in paths for parent in path.parents):
        raise ValueError("Refusing package archive entries nested below a symlink")


def verify_and_extract(deb, pin, destination):
    if digest(deb) != pin["sha256"]:
        raise RuntimeError(f"Package checksum changed before extraction: {deb}")
    fields = subprocess.check_output(["dpkg-deb", "--show", "--showformat=${Package}\\n${Version}\\n${Architecture}\\n", str(deb)], text=True).splitlines()
    if fields != [pin["name"], pin["version"], "amd64"]:
        raise RuntimeError(f"Package identity differs from pin: {fields}")
    proc = subprocess.Popen(["dpkg-deb", "--fsys-tarfile", str(deb)], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|") as archive:
            validate_archive_members(list(archive))
    finally:
        proc.stdout.close()
        proc.wait()
    if proc.returncode:
        raise RuntimeError(f"Cannot inspect package archive: {deb}")
    subprocess.run(["dpkg-deb", "--extract", str(deb), str(destination)], check=True)


def tree_manifest(root):
    result = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            if not path.resolve().is_relative_to(root):
                raise ValueError(f"Private sysroot symlink escapes deployment: {path}")
            result[relative] = {"symlink": os.readlink(path)}
        elif path.is_file():
            result[relative] = {"sha256": digest(path)}
        elif not path.is_dir():
            raise ValueError(f"Unsupported entry in private sysroot: {path}")
    return result


def setup(root, adopt_existing=False):
    require_host()
    runtime = safe_path(root, "runtime")
    runtime.mkdir(exist_ok=True)
    debs = safe_path(root, "runtime/debs")
    debs.mkdir(exist_ok=True)
    sysroot = safe_path(root, "runtime/sysroot")
    record = safe_path(root, "runtime/native-libs.json")
    if record.exists() and json.loads(record.read_text()).get("root") != str(root):
        raise RuntimeError("Native-library manifest belongs to another deployment")
    if sysroot.exists() and not record.exists() and not adopt_existing:
        raise RuntimeError("Private sysroot already exists without ownership metadata; use --adopt-existing to verify it before adoption")
    with tempfile.TemporaryDirectory(prefix="native-stage-", dir=runtime) as temporary:
        staged = Path(temporary) / "sysroot"
        staged.mkdir()
        packages = []
        for pin in PACKAGES:
            deb = obtain_package(debs, pin)
            verify_and_extract(deb, pin, staged)
            packages.append({**pin, "architecture": "amd64", "deb": str(deb.relative_to(root))})
        files = tree_manifest(staged)
        if sysroot.exists():
            if tree_manifest(sysroot) != files:
                raise RuntimeError("Existing private sysroot differs from the verified packages; refusing to overwrite it")
        else:
            staged.replace(sysroot)
        manifest = {"format": "weightsmith-native-libs-v1", "root": str(root), "distribution": "Ubuntu 24.04",
                    "packages": packages, "files": files, "library_path": str(sysroot / "usr/lib/x86_64-linux-gnu")}
        with tempfile.NamedTemporaryFile("w", prefix="native-manifest-", dir=runtime, delete=False) as handle:
            json.dump(manifest, handle, indent=2)
            handle.write("\n")
            temporary_record = Path(handle.name)
        temporary_record.replace(record)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=REPO)
    parser.add_argument("--adopt-existing", action="store_true", help="Adopt only an existing sysroot identical to freshly verified packages")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    if not (root / "weightsmith/config.py").is_file():
        parser.error("--root must be the dedicated weightsmith checkout")
    try:
        runtime = safe_path(root, "runtime")
        runtime.mkdir(exist_ok=True)
        lock = safe_path(root, "runtime/native-libs.lock")
        with lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            manifest = setup(root, args.adopt_existing)
        print(json.dumps({"manifest": str(runtime / "native-libs.json"), "library_path": manifest["library_path"],
                          "packages": len(manifest["packages"])}, indent=2))
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.exit(1, f"private native-library setup failed: {exc}\n")


if __name__ == "__main__":
    main()
