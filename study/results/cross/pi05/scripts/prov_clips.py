"""PROVISIONAL before/after clips of π0.5 DROID from episode videos that already exist on ali-ms.

The fresh run (shared slots, default preset) produced no episodes (the pod cannot run Isaac Sim; see README), so
these clips come from the earlier held-out runs the provisional tables cite: RTX 3090, RoboLab `performance` preset,
weightsmith pool, each arm on the same slots (same scene, same noise seeds). Every panel of a clip is one arm on one
slot, stacked top to bottom, labelled with its arm and the outcome RoboLab's checker recorded.

Selection rule (no hand-picking): for each clip, the lowest slot in the run's block that (a) was in the first group
of its Isaac process in every arm (`run` = 0 in the per-slot record, so the pairing is exact) and (b) has the clip's
outcome pattern (e.g. unedited fails, random control fails, edit succeeds). The first frames of all panels are
compared (PSNR vs the top panel, recorded in clips.json) to check that they start from the same scene.

    python3 study/results/cross/pi05/scripts/prov_clips.py [--cache DIR]
Needs ssh access to ali-ms (reads per-slot rows.json and copies the source MP4s; CPU only) and a local ffmpeg.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
OUT = HERE / "videos"
HOST = "ali-ms"
PROJ = "/home/ali/vibes/opus-weight-surgery/projects"
FONT = next((p for p in [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/opt/homebrew/Caskroom/miniconda/base/lib/python3.12/site-packages/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf",
] if Path(p).exists()), None)
FPS_SRC = 15
SPEED = 3.0
W = 960            # panel width; each source frame is 1280x360 (external camera | wrist camera)
MAX_BYTES = 2_000_000

CTB = "cube-then-banana/heldout/"
V2 = CTB + "v2mi-f-RubiksCubeThenBananaTask-90000-128/"
CLIPS = [
    {"name": "t1_v2mi-f_vs_control_vs_teacher", "code": "T1", "task": "cube then banana",
     "block": [90000, 90063], "run_note": "held-out (paper Fig. 1 runs)",
     "panels": [("unedited", V2 + "logs-base-90000", False),
                ("random-direction control (same layers, ranks, singular values)", CTB + "ctl1-90000-64/logs-rand-90000", False),
                ("edited: v2mi-f", V2 + "logs-cand-90000", True),
                ("unedited + explicit wording (teacher)", CTB + "ctl1-90000-64/logs-teacher-90000", True)]},
    {"name": "t1_v2mi-f_second_lane", "code": "T1", "task": "cube then banana", "block": [90064, 90127],
     "run_note": "held-out (paper runs)",
     "panels": [("unedited", V2 + "logs-base-90064", False), ("edited: v2mi-f", V2 + "logs-cand-90064", True)]},
    {"name": "t3_v2mi-f_side_effect_fixed", "code": "T3", "task": "cube left of bowl", "block": [90000, 90063],
     "run_note": "held-out side-effect run (T1 edit, T3's own instruction)",
     "panels": [("unedited", "tx-rubikscubeleftofbowl/heldout/ctb-v2mi-RubiksCubeLeftOfBowlTask-90000-64/logs-base-{b}", False),
                ("edited: v2mi-f (built for T1)", "tx-rubikscubeleftofbowl/heldout/ctb-v2mi-RubiksCubeLeftOfBowlTask-90000-64/logs-cand-{b}", True)],
     "lanes": [90000, 90032]},
    {"name": "t2_v2mi-f_side_effect_fixed", "code": "T2", "task": "pick glasses", "block": [90000, 90063],
     "run_note": "held-out side-effect run (T1 edit, T2's own instruction)",
     "panels": [("unedited", "thr-pickglasses/heldout/ctb-v2mi-PickGlassesTask-90000-64/logs-base-{b}", False),
                ("edited: v2mi-f (built for T1)", "thr-pickglasses/heldout/ctb-v2mi-PickGlassesTask-90000-64/logs-cand-{b}", True)],
     "lanes": [90000, 90032]},
    {"name": "t2_v2mi-f_side_effect_broke", "code": "T2", "task": "pick glasses", "block": [90000, 90063],
     "run_note": "held-out side-effect run; one of the 9 starts the edit breaks",
     "panels": [("unedited", "thr-pickglasses/heldout/ctb-v2mi-PickGlassesTask-90000-64/logs-base-{b}", True),
                ("edited: v2mi-f (built for T1)", "thr-pickglasses/heldout/ctb-v2mi-PickGlassesTask-90000-64/logs-cand-{b}", False)],
     "lanes": [90000, 90032]},
    {"name": "t8_s4r-grp3_vs_control", "code": "T8", "task": "mustard right of raisin, reference-first wording",
     "block": [92000, 92063], "run_note": "held-out (sgdw study)",
     "panels": [("unedited, RF wording", "sgdw-s4r/heldout/s4r-grp2-MustardRightOfRaisinRFTask-92000-64/logs-base-{b}", False),
                ("RF + random-direction control (same layers, ranks, singular values)", "sgdw-s4r/heldout/ctl-92000-64/logs-rand3-{b}", False),
                ("RF + edit s4r-grp3", "sgdw-s4r/heldout/ctl-92000-64/logs-g3-{b}", True)],
     "lanes": [92000, 92032]},
]


def sh(cmd, **kw):
    return subprocess.run(cmd, check=True, capture_output=True, text=True, **kw).stdout


def rows_of(remote_dir: str, cache: Path) -> dict[int, dict]:
    local = cache / "rows" / remote_dir / "rows.json"
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_text(sh(["ssh", HOST, f"cat {PROJ}/{remote_dir}/rows.json"], stdin=subprocess.DEVNULL))
    r = json.loads(local.read_text())
    return {x["slot"]: x for x in (r if isinstance(r, list) else r["rows"]) if x.get("ok")}


def pick(clip, cache):
    """Lowest slot with run 0 in every arm and the clip's outcome pattern; returns (slot, per-panel rows)."""
    lanes = clip.get("lanes", [None])
    arms = []
    for label, path, want in clip["panels"]:
        rows = {}
        for b in lanes:
            rows.update(rows_of(path.format(b=b) if b else path, cache))
        arms.append((label, rows, want))
    lo, hi = clip["block"]
    for s in range(lo, hi + 1):
        if all(s in r and r[s].get("run", 0) == 0 and bool(r[s]["success"]) == w for _, r, w in arms):
            return s, [(label, r[s]) for label, r, _ in arms]
    raise SystemExit(f"no slot matches {clip['name']}")


def fetch(remote: str, cache: Path) -> Path:
    local = cache / "src" / re.sub(r"[^A-Za-z0-9_.-]", "_", remote.split("/output/")[-1])
    if not local.exists():
        local.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["scp", "-q", f"{HOST}:{remote}", str(local)], check=True, stdin=subprocess.DEVNULL)
    return local


def duration(p: Path) -> float:
    return float(sh(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(p)]).strip())


def first_frame_psnr(a: Path, b: Path) -> float:
    out = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(a), "-i", str(b), "-lavfi",
                          "[0:v]trim=end_frame=1[x];[1:v]trim=end_frame=1[y];[x][y]psnr", "-f", "null", "-"],
                         capture_output=True, text=True).stderr
    m = re.search(r"average:([0-9.]+|inf)", out)
    return float(m.group(1)) if m else float("nan")


def esc(t: str) -> str:
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "’").replace(",", "\\,").replace("%", "\\%")


def short_reason(r: str | None) -> list[str]:
    """RoboLab's failure reason, shortened only by dropping argument names and the frame of reference."""
    r = (r or "?").replace("object=", "").replace("reference_", "ref ")
    r = re.sub(r", frame_of_reference=[a-z_]+", "", r)
    r = r.replace("Conditions not satisfied: ", "").replace("Condition not satisfied: ", "")
    return [("checker, not satisfied: " if k == 0 else "  and: ") + part for k, part in enumerate(r.split("; "))]


def build(clip, cache: Path):
    slot, panels = pick(clip, cache)
    vids, labels, prov = [], [], []
    for label, row in panels:
        v = fetch(row["video"], cache)
        ok, steps = bool(row["success"]), int(row["steps"])
        vids.append(v)
        labels.append(f"{label}: {'SUCCESS' if ok else 'failure'} "
                      f"({'done after' if ok else 'time limit,'} {steps / FPS_SRC:.0f} s)")
        prov.append({"label": label, "success": ok, "steps": steps, "reason": row.get("reason"),
                     "instruction": row.get("instruction"), "source_video": row["video"], "run": row.get("run")})
    d = [duration(v) for v in vids]
    T = max(d)
    for i in range(1, len(vids)):
        prov[i]["first_frame_psnr_vs_top_db"] = round(first_frame_psnr(vids[0], vids[i]), 1)
    fs = 19 if len(vids) > 2 else 21
    title = f"{clip['code']} {clip['task']}  \u00b7  start {slot}  \u00b7  {SPEED:g}x speed"
    sub = "Same start in every panel (scene and noise seeds); " + clip["run_note"]
    sub2 = "PROVISIONAL: earlier \u03c00.5 DROID runs, RTX 3090, RoboLab performance preset"
    fl = []
    for i in range(len(vids)):
        why = ""
        if not prov[i]["success"]:
            for j, line in enumerate(short_reason(prov[i]["reason"])):
                why += (f",drawtext=fontfile={FONT}:text='{esc(line)}':x=10:y={fs + 24 + j * 20}:fontsize=14:"
                        f"fontcolor=white:box=1:boxcolor=black@0.62:boxborderw=4")
        fl.append(f"[{i}:v]tpad=stop_mode=clone:stop_duration={T - d[i] + 0.05:.3f},trim=duration={T:.3f},"
                  f"scale={W}:-2,drawtext=fontfile={FONT}:text='{esc(labels[i])}':x=10:y=8:fontsize={fs}:"
                  f"fontcolor=white:box=1:boxcolor=black@0.62:boxborderw=6{why}[v{i}]")
    ins = "".join(f"[v{i}]" for i in range(len(vids)))
    fl.append(f"{ins}vstack=inputs={len(vids)},pad=iw:ih+70:0:0:color=0x111111,"
              f"drawtext=fontfile={FONT}:text='{esc(title)}':x=10:y=h-64:fontsize=17:fontcolor=white,"
              f"drawtext=fontfile={FONT}:text='{esc(sub)}':x=10:y=h-40:fontsize=14:fontcolor=0xcccccc,"
              f"drawtext=fontfile={FONT}:text='{esc(sub2)}':x=10:y=h-20:fontsize=14:fontcolor=0xcccccc,"
              f"setpts=PTS/{SPEED},fps=30[out]")
    out = OUT / f"{clip['name']}_{slot}.mp4"
    cap = MAX_BYTES * (1.5 if len(vids) > 3 else 1)
    for crf in (26, 29, 32, 35, 38):
        cmd = ["ffmpeg", "-y", "-loglevel", "error"]
        for v in vids:
            cmd += ["-i", str(v)]
        cmd += ["-filter_complex", ";".join(fl), "-map", "[out]", "-c:v", "libx264", "-preset", "slow",
                "-crf", str(crf), "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(out)]
        subprocess.run(cmd, check=True)
        if out.stat().st_size <= cap:
            break
    return {"file": out.name, "code": clip["code"], "slot": slot, "block": clip["block"], "speed": SPEED,
            "bytes": out.stat().st_size, "crf": crf, "seconds": round(T / SPEED, 1), "panels": prov,
            "status": "provisional", "hardware": "ali-ms RTX 3090", "render_preset": "performance",
            "selection": "lowest slot of the block with run 0 in every arm and this outcome pattern"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(Path.home() / ".cache/xpi05-clips"))
    ap.add_argument("--only", default=None)
    a = ap.parse_args()
    cache = Path(a.cache)
    OUT.mkdir(exist_ok=True)
    idx = OUT / "clips.json"
    done = {c["file"]: c for c in (json.loads(idx.read_text()) if idx.exists() else [])}
    for clip in CLIPS:
        if a.only and a.only not in clip["name"]:
            continue
        c = build(clip, cache)
        done = {k: v for k, v in done.items() if not k.startswith(clip["name"] + "_")}
        done[c["file"]] = c
        print(c["file"], c["bytes"], c["crf"], [(p["label"], p["success"], p.get("first_frame_psnr_vs_top_db"))
                                               for p in c["panels"]])
    idx.write_text(json.dumps(sorted(done.values(), key=lambda c: c["file"]), indent=1) + "\n")
    dump_slots(cache)


def dump_slots(cache: Path):
    """Compact per-slot outcomes of every earlier run read above (results/prior_slots_ali-ms.json), so that
    analyze.py can recompute the provisional control pairings from files in this repository."""
    runs = {}
    for f in sorted((cache / "rows").rglob("rows.json")):
        key = str(f.parent.relative_to(cache / "rows"))
        rows = rows_of(key, cache)
        runs[key] = {str(sl): [int(bool(x["success"])), int(x["steps"]), x.get("run", 0)] for sl, x in sorted(rows.items())}
    out = HERE / "results" / "prior_slots_ali-ms.json"
    out.write_text(json.dumps({"what": "per-slot [success, steps, run-in-process] of earlier pi0.5 runs on ali-ms "
                                       "(RTX 3090, performance preset), from each run's rows.json under "
                                       f"{HOST}:{PROJ}/<key>/rows.json; written by scripts/prov_clips.py",
                               "runs": runs}, indent=0) + "\n")


if __name__ == "__main__":
    main()
