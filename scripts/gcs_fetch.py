#!/usr/bin/env python3
"""Fast parallel download of a public Google Cloud Storage prefix (stdlib only).

    python3 scripts/gcs_fetch.py gs://openpi-assets-simeval/pi05_droid_jointpos ~/.cache/openpi

writes ~/.cache/openpi/openpi-assets-simeval/pi05_droid_jointpos/..., the layout OpenPI's
maybe_download() expects, so OpenPI finds the checkpoint without downloading it again.
Large objects are fetched as many byte ranges at once, which is several times faster than a
single stream on most cloud hosts. Files land under a .partial directory that is renamed
when every object is complete.
"""
import json
import os
import shutil
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

CHUNK = 64 << 20  # bytes per ranged request
WORKERS = int(os.environ.get("GCS_FETCH_WORKERS", "48"))


def list_objects(bucket, prefix):
    items, token = [], None
    while True:
        q = {"prefix": prefix, "fields": "items(name,size),nextPageToken", "maxResults": "1000"}
        if token:
            q["pageToken"] = token
        url = f"https://storage.googleapis.com/storage/v1/b/{bucket}/o?" + urllib.parse.urlencode(q)
        with urllib.request.urlopen(url, timeout=60) as r:
            d = json.load(r)
        items += [(i["name"], int(i["size"])) for i in d.get("items", [])]
        token = d.get("nextPageToken")
        if not token:
            return items


def fetch_range(url, path, start, end, tries=6):
    for k in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
            with urllib.request.urlopen(req, timeout=120) as r, open(path, "r+b") as f:
                f.seek(start)
                shutil.copyfileobj(r, f, 1 << 20)
                if f.tell() != end + 1:
                    raise IOError(f"short read {f.tell()} != {end + 1}")
            return end - start + 1
        except Exception as e:  # noqa: BLE001 - retry any network error
            if k == tries - 1:
                raise
            time.sleep(2 ** k)
            print(f"  retry {os.path.basename(path)} [{start}-{end}]: {e}", file=sys.stderr)


def main():
    src, cache = sys.argv[1], os.path.expanduser(sys.argv[2])
    bucket, _, prefix = src[len("gs://"):].partition("/")
    prefix = prefix.strip("/")
    final = os.path.join(cache, bucket, prefix)
    if os.path.exists(final):
        print(f"exists: {final}")
        return
    tmp = final + ".partial"
    objs = list_objects(bucket, prefix + "/")
    total = sum(s for _, s in objs)
    print(f"{len(objs)} objects, {total / 1e9:.2f} GB -> {final}")
    jobs = []
    for name, size in objs:
        path = os.path.join(tmp, os.path.relpath(name, prefix))
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.truncate(size)
        url = f"https://storage.googleapis.com/{bucket}/{urllib.parse.quote(name)}"
        for start in range(0, max(size, 1), CHUNK):
            if size:
                jobs.append((url, path, start, min(start + CHUNK, size) - 1))
    t0, done = time.time(), 0
    with ThreadPoolExecutor(WORKERS) as ex:
        for n in ex.map(lambda j: fetch_range(*j), jobs):
            done += n
            if done and (done * 20 // total) != ((done - n) * 20 // total):
                dt = time.time() - t0
                print(f"  {done / 1e9:6.2f}/{total / 1e9:.2f} GB  {done / dt / 1e6:6.1f} MB/s", flush=True)
    os.rename(tmp, final)
    print(f"done in {time.time() - t0:.0f} s: {final}")


if __name__ == "__main__":
    main()
