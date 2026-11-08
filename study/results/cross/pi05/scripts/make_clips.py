"""Before/after clips from the same start: the unedited episode on top, the edited (or reworded) one below.

Runs on the GPU host, where the per-slot sensor videos are (<root>/videos/<code>/<arm>/<slot>.mp4, written by
pool_pi05.py). Each clip: both episodes of one slot (same scene, same noise seeds), stacked, labelled with the arm and
its outcome from the run records, sped up, H.264, at most 2 MB.

    python3 make_clips.py --root /workspace/cross/pi05 SPEC.json OUT_DIR
SPEC.json: [{"code": "T1", "slot": 96005, "top": "base", "bottom": "v2mi-f", "name": "t1_96005_base_vs_v2mi-f"}, ...]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
LABEL = {"base": "unedited", "v2mi-f": "edited (v2mi-f)", "abl-imgl8": "edited (abl-imgl8)",
         "ctl-rand": "random-direction control", "s4r-grp3": "edited (s4r-grp3)",
         "ctl-rand3": "random-direction control", "teacher": "unedited, known-good wording"}
TASKS = {"T1": "cube then banana", "T2": "pick glasses", "T3": "cube left of bowl", "T4": "stack yellow on red",
         "T5": "mustard in left bin", "T6": "stack 3 cubes", "T7": "pick green block",
         "T8": "mustard right of raisin", "R1": "cube in bowl"}
MAX_BYTES = 2_000_000


def outcome(root: Path, code: str, arm: str, slot: int):
    s0 = slot - (slot - 30000) % 16 if slot < 96000 else slot - (slot - 96000) % 16
    rec = json.loads((root / "runs" / f"{code}-{arm}-s{s0}.json").read_text())
    row = next(r for r in rec["rows"] if r["slot"] == slot)
    return bool(row["success"]), int(row["steps"]), rec.get("prompt")


def duration(p: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace(",", "\\,")


def make(root: Path, spec: dict, out_dir: Path, speed: float):
    code, slot = spec["code"], int(spec["slot"])
    vids, labels = [], []
    for arm in (spec["top"], spec["bottom"]):
        v = root / "videos" / code / arm / f"{slot}.mp4"
        ok, steps, prompt = outcome(root, code, arm, slot)
        vids.append(v)
        lab = spec.get(f"{'top' if arm == spec['top'] else 'bottom'}_label") or LABEL.get(arm, arm)
        labels.append(f"{lab}: {'SUCCESS' if ok else 'failure'} ({steps / 15:.0f} s)")
    d = [duration(v) for v in vids]
    T = max(d)
    title = f"{code} {TASKS[code]}, start {slot} (same scene and seeds), {speed:g}x speed"
    fl = []
    for i in range(2):
        pad = T - d[i]
        fl.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration={pad:.3f},scale=960:-2,"
                  f"drawtext=fontfile={FONT}:text='{esc(labels[i])}':x=10:y=8:fontsize=20:fontcolor=white:"
                  f"box=1:boxcolor=black@0.6:boxborderw=6[v{i}]")
    fl.append(f"[v0][v1]vstack=inputs=2,drawtext=fontfile={FONT}:text='{esc(title)}':x=10:y=h-30:fontsize=17:"
              f"fontcolor=white:box=1:boxcolor=black@0.6:boxborderw=5,setpts=PTS/{speed},fps=30[out]")
    out = out_dir / f"{spec['name']}.mp4"
    for crf in (28, 31, 34, 37, 40):
        cmd = ["ffmpeg", "-y", "-loglevel", "error", "-i", str(vids[0]), "-i", str(vids[1]),
               "-filter_complex", ";".join(fl), "-map", "[out]", "-c:v", "libx264", "-preset", "slow",
               "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
        subprocess.run(cmd, check=True)
        if out.stat().st_size <= MAX_BYTES:
            break
    return {**spec, "file": out.name, "bytes": out.stat().st_size, "crf": crf, "labels": labels,
            "source_videos": [str(v) for v in vids]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/workspace/cross/pi05")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("spec")
    ap.add_argument("out_dir")
    a = ap.parse_args()
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    done = [make(Path(a.root), s, out_dir, a.speed) for s in json.loads(Path(a.spec).read_text())]
    (out_dir / "clips.json").write_text(json.dumps(done, indent=1) + "\n")
    for c in done:
        print(c["file"], c["bytes"], c["labels"])


if __name__ == "__main__":
    main()
