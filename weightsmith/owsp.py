#!/usr/bin/env python3
"""owsp: weight-space programming tools for pi0.5 on RoboLab (the editor's only tool surface).

All commands run on the GPU host. The editor only reaches this file through the
`bin/owsp` wrapper (local or over ssh). Held-out seeds and results are never readable from here.
Configuration: see weightsmith/config.py and .env.example.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> `weightsmith` importable
from weightsmith import config as C  # noqa: E402

R = C.PROJECT_ROOT  # per-project root (config, cands, data, state); env WSP_ROOT
HARNESS = C.PKG
LOCKS = C.LOCK_DIR  # GPU locks are global across projects
BASE_WS = C.BASE_WS  # full VLM+action export, read-only
OPENPI_PY = C.OPENPI_PY
CANDS = R / "cands"
DATA = R / "data"
EPI = DATA / "episodes"
ACTS = DATA / "acts"
FRAMES = DATA / "frames"
NOTES = R / "notes"
STATE = R / "state"
if not (R / "config.json").is_file():
    sys.exit(f"owsp: no project config at {R / 'config.json'} (set WSP_ROOT to a project root)")
CFG = json.loads((R / "config.json").read_text())
TASK = CFG["task"]
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")


# ---------------------------------------------------------------- utilities
def budget():
    p = STATE / "budget.json"
    return json.loads(p.read_text()) if p.exists() else {"episodes": 0, "acts_calls": 0, "acts_items": 0, "evals": 0}


def charge(**kw):
    b = budget()
    for k, v in kw.items():
        b[k] = b.get(k, 0) + v
    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "budget.json").write_text(json.dumps(b, indent=1))
    return b


def check_budget(episodes=0, acts_items=0):
    b, lim = budget(), CFG["limits"]
    if b["episodes"] + episodes > lim["practice_episodes"]:
        sys.exit(f"budget: practice episodes {b['episodes']}+{episodes} > {lim['practice_episodes']}")
    if b["acts_items"] + acts_items > lim["acts_items"]:
        sys.exit(f"budget: acts items {b['acts_items']}+{acts_items} > {lim['acts_items']}")


from weightsmith.locks import GpuLock  # noqa: E402  (shared with heldout.py)


def cand_dir(name):
    if name == "base":
        return None
    if not NAME_RE.match(name):
        sys.exit(f"bad candidate name {name!r}")
    d = CANDS / name
    if not d.exists():
        sys.exit(f"no candidate {name!r}; create with `owsp new {name}`")
    return d


def fingerprint(name):
    d = cand_dir(name)
    if d is None:
        return "base"
    import hashlib
    h = hashlib.sha256()
    for f in sorted((d / "edits").glob("*.npy")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    return h.hexdigest()[:16]


def episodes_index():
    p = EPI / "index.jsonl"
    return [json.loads(l) for l in p.read_text().splitlines()] if p.exists() else []


# ---------------------------------------------------------------- commands
def cmd_tensors(a):
    cfg = json.loads((BASE_WS / "workspace.json").read_text())
    rx = re.compile(a.regex) if a.regex else None
    n = 0
    for name, e in sorted(cfg["tensors"].items()):
        if rx and not rx.search(name):
            continue
        n += 1
        print(f"{name:55s} {str(e['shape']):22s} mean={e['mean']:+.4g} std={e['std']:.4g} min={e['min']:+.4g} max={e['max']:+.4g}")
    print(f"[{n} tensors]")


def cmd_new(a):
    if not NAME_RE.match(a.name) or a.name == "base":
        sys.exit("name must match [a-z0-9][a-z0-9_-]* and not be 'base'")
    d = CANDS / a.name
    if d.exists():
        sys.exit(f"{a.name} exists")
    d.mkdir(parents=True)
    cfg = json.loads((BASE_WS / "workspace.json").read_text())
    cfg["max_changed_fraction"] = 1.0
    cfg["sandbox_memory_bytes"] = 100 << 30
    cfg["sandbox_timeout_seconds"] = 1800
    (d / "workspace.json").write_text(json.dumps(cfg))
    (d / "base").symlink_to(BASE_WS / "base")
    (d / "edits").mkdir()
    (d / "scratch").mkdir()
    if a.from_:
        src = cand_dir(a.from_)
        if src is not None:
            for f in (src / "edits").glob("*.npy"):
                shutil.copy2(f, d / "edits" / f.name)
    (d / "meta.json").write_text(json.dumps({"created": time.time(), "from": a.from_, "note": a.note}))
    print(f"created candidate {a.name} (edits copied from {a.from_ or 'nothing'})")


PRELUDE_EXTRA = r'''
import glob as _glob
DATA = "/work/data"
NOTES = "/work/notes"
def episodes(cand=None, success=None):
    """Practice episode records (dicts). Filter by candidate name and/or success."""
    rows = [json.loads(l) for l in open(DATA + "/episodes/index.jsonl")] if os.path.exists(DATA + "/episodes/index.jsonl") else []
    return [r for r in rows if (cand is None or r["cand"] == cand) and (success is None or r["success"] == success)]
def obs(eid):
    """Exact policy inputs at every replan of practice episode `eid`:
    base_img/wrist_img [K,224,224,3] uint8, joint [K,7], gripper [K,1], seed [K], step [K] (sim step), actions [K,15,8] (policy output chunk)."""
    z = np.load(f"{DATA}/episodes/{eid}/obs.npz")
    return {k: z[k] for k in z.files}
def acts(aid):
    """Activation capture `aid` -> (index dict, dict name->array stacked over items)."""
    idx = json.load(open(f"{DATA}/acts/{aid}/index.json"))
    z = np.load(f"{DATA}/acts/{aid}/acts.npz")
    return idx, {k: z[k] for k in z.files}
'''


def sandbox(cand: Path, code: str) -> str:
    base_real = (cand / "base").resolve()
    cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", "--clearenv",
           "--ro-bind", "/usr", "/usr", "--symlink", "usr/lib", "/lib", "--symlink", "usr/lib64", "/lib64",
           "--symlink", "usr/bin", "/bin", "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
           "--dir", "/work", "--ro-bind", str((cand / "workspace.json").resolve()), "/work/workspace.json",
           "--ro-bind", str(base_real), "/work/base",
           "--bind", str((cand / "edits").resolve()), "/work/edits",
           "--bind", str((cand / "scratch").resolve()), "/work/scratch",
           "--ro-bind", str(DATA.resolve()), "/work/data",
           "--bind", str(NOTES.resolve()), "/work/notes",
           "--chdir", "/work", "--setenv", "HOME", "/tmp", "--setenv", "PATH", "/usr/bin:/bin",
           "--setenv", "CUDA_VISIBLE_DEVICES", "", "--setenv", "OMP_NUM_THREADS", "16",
           "/usr/bin/python3", "-I", "-c"]
    from weightsmith.lab.sandbox import PRELUDE
    cmd.append(PRELUDE + PRELUDE_EXTRA + "\n" + code)
    try:
        p = subprocess.run(cmd, text=True, capture_output=True, timeout=1800)
        out = p.stdout + (("\n[stderr]\n" + p.stderr[-6000:]) if p.stderr.strip() else "") + (f"\n[exit {p.returncode}]" if p.returncode else "")
    except subprocess.TimeoutExpired:
        out = "[timed out after 1800 s]"
    if len(out) > 40000:
        out = out[:40000] + "\n[output truncated at 40000 chars]"
    return out or "(no output)"


def cmd_py(a):
    d = cand_dir(a.name)
    if d is None:
        # read-only analysis against base: use a scratch candidate that is never evaluated
        d = CANDS / "_scratch_base"
        if not d.exists():
            cmd_new(argparse.Namespace(name="_scratch_base", from_=None, note="read-only scratch")) if False else None
            d.mkdir(parents=True)
            cfg = json.loads((BASE_WS / "workspace.json").read_text())
            cfg["writable_override"] = False
            for e in cfg["tensors"].values():
                e["writable"] = False
            (d / "workspace.json").write_text(json.dumps(cfg))
            (d / "base").symlink_to(BASE_WS / "base")
            (d / "edits").mkdir()
            (d / "scratch").mkdir()
    code = sys.stdin.read()
    log = STATE / "py_log.jsonl"
    out = sandbox(d, code)
    print(out, flush=True)
    with log.open("a") as f:
        f.write(json.dumps({"t": time.time(), "cand": a.name, "code": code, "out": out[:4000]}) + "\n")


def validate(name):
    d = cand_dir(name)
    from weightsmith.lab.openpi_vlm_full import validate_workspace
    return validate_workspace(d)


def cmd_diff(a):
    rep = validate(a.name)
    rep.pop("editable_parameters", None)
    print(json.dumps(rep, indent=1, default=str))


def run_eval(name, slots, tag, task=None, prompt=None, extra_gpus=()):
    """Run one episode per slot on the GPU pool. Returns pool rows."""
    from weightsmith.pool import run_pool
    d = cand_dir(name)
    rows, rep = run_pool(workspace=str(d) if d else None, task=task or TASK, tag=f"{tag}-{name}", slots=slots,
                         envs=CFG["envs"], rendering_type=CFG["rendering_type"], video_mode="sensor", prompt=prompt,
                         extra_gpus=extra_gpus)
    return rows


def episode_events(out_dir: Path, run_idx: int, env_id: int):
    for cand in [out_dir / f"log_{run_idx}_env{env_id}.json"]:
        if cand.exists():
            log = json.loads(cand.read_text())
            keep = {"OBJECT_GRABBED_SUCCESS", "OBJECT_IN_CONTAINER_SUCCESS", "TARGET_OBJECT_DROPPED",
                    "WRONG_OBJECT_GRABBED_FAILURE", "GRIPPER_FULLY_CLOSED", "OBJECT_BUMPED", "MULTIPLE_OBJECTS_GRABBED"}
            return [{"step": int(e["step"]), "name": e["name"], "info": e.get("info", "")[:160]}
                    for e in log.get("events", []) if e.get("name") in keep or "subtask" in e.get("info", "").lower()][:80]
    return []


def cmd_eval(a):
    d = cand_dir(a.name)
    if d is not None:
        validate(a.name)
    E = CFG["envs"]
    if a.n % E or a.n <= 0:
        sys.exit(f"--n must be a positive multiple of {E}")
    if a.offset % E or a.offset < 0 or a.offset + a.n > CFG["practice_slots"]:
        sys.exit(f"--offset must be a multiple of {E} and offset+n <= {CFG['practice_slots']} practice slots")
    check_budget(episodes=a.n)
    fp = fingerprint(a.name)
    label = a.name
    if a.prompt and not CFG.get("prompt_overrides", True):
        sys.exit("prompt overrides are disabled in this project: the instruction is fixed; only weights may change")
    if a.prompt:
        import hashlib
        label = f"{a.name}@p{hashlib.sha256(a.prompt.encode()).hexdigest()[:6]}"
        print(f"[DIAGNOSTIC closed-loop prompt override: {a.prompt!r} -> recorded as {label!r}; not a weight edit]")
    slots = range(CFG["practice_seed_base"] + a.offset, CFG["practice_seed_base"] + a.offset + a.n)
    with GpuLock(C.SERVER_GPU, C.SIM_GPU, optional=C.EVAL_EXTRA_GPUS) as lk:
        t0 = time.time()
        if lk.acquired:
            print(f"[eval using extra GPU lane(s): {lk.acquired}]", flush=True)
        rows = run_eval(a.name, slots, "prac", prompt=a.prompt, extra_gpus=tuple(lk.acquired))
        dt = time.time() - t0
    bad = [r for r in rows if r["ok"] and (r.get("task_name") not in (None, TASK)
                                          or (r.get("instruction") or "").strip().lower() != CFG["instruction"].strip().lower())]
    if bad:
        sys.exit(f"instruction/task mismatch: config says {TASK!r} / {CFG['instruction']!r}, sim reported "
                 f"{bad[0].get('task_name')!r} / {bad[0].get('instruction')!r}; fix config.json (no budget charged)")
    done = [r for r in rows if r["ok"]]
    charge(episodes=len(done), evals=1)
    recs = []
    for r in rows:
        if not r["ok"]:
            continue
        slot = f"s{r['slot']}"
        eid = f"{label}.{fp[:6]}.{slot}"
        k = 1
        while (EPI / eid).exists():
            k += 1
            eid = f"{label}.{fp[:6]}.{slot}.r{k}"
        (EPI / eid).mkdir(parents=True)
        if Path(r["obs"]).exists():
            shutil.move(r["obs"], EPI / eid / "obs.npz")
        out_dir = Path(r["out_dir"])
        rec = {"eid": eid, "cand": label, "prompt": a.prompt or CFG["instruction"], "fingerprint": fp, "slot": slot, "success": bool(r["success"]),
               "steps": r["steps"], "score": r["score"], "reason": r["reason"],
               "events": episode_events(out_dir, r["run"], r["env"]), "video": [r["video"]] if r["video"] else []}
        (EPI / eid / "summary.json").write_text(json.dumps(rec, indent=1))
        recs.append(rec)
    with (EPI / "index.jsonl").open("a") as f:
        for r in recs:
            f.write(json.dumps({k: v for k, v in r.items() if k not in ("events", "video")}) + "\n")
    k = sum(r["success"] for r in recs)
    miss = len(rows) - len(recs)
    print(f"[eval {label} fp={fp} n={len(recs)} success={k}/{len(recs)} ({100*k/max(1,len(recs)):.1f}%) wall={dt/60:.1f} min"
          + (f"; {miss} episodes crashed and were not charged" if miss else "") + "]")
    if label != "base":
        base = {}
        for r in episodes_index():
            if r["cand"] == "base":
                base.setdefault(r["slot"], []).append(r["success"])
        both = [(sum(base[r["slot"]]) / len(base[r["slot"]]), r["success"]) for r in recs if r["slot"] in base]
        if both:
            print(f"[paired vs base on {len(both)} shared slots: base {sum(b for b, _ in both):.1f} vs cand {sum(c for _, c in both)} | "
                  f"fixed {sum((b < 0.5) and c for b, c in both)}, broke {sum((b >= 0.5) and (not c) for b, c in both)}]")
    for r in sorted(recs, key=lambda r: r["slot"]):
        ev = ",".join(f"{e['name'].replace('_SUCCESS','').replace('OBJECT_','')}@{e['step']}" for e in r["events"] if e["name"] != "GRIPPER_FULLY_CLOSED")[:170]
        print(f"  {r['eid']:34s} {'OK  ' if r['success'] else 'FAIL'} steps={r['steps']} {(r['reason'] or '')[:40]} | {ev}")


def cmd_episodes(a):
    rows = episodes_index()
    if a.cand:
        rows = [r for r in rows if r["cand"] == a.cand]
    by = {}
    for r in rows:
        by.setdefault((r["cand"], r["fingerprint"]), []).append(r["success"])
    for (c, fp), v in by.items():
        print(f"{c:25s} fp={fp} {sum(v)}/{len(v)} ({100*sum(v)/len(v):.1f}%)")
    if a.verbose:
        for r in rows:
            print(json.dumps(r))


def cmd_show(a):
    s = json.loads((EPI / a.eid / "summary.json").read_text())
    s.pop("video", None)
    print(json.dumps(s, indent=1))


FRAME_SCRIPT = r"""
import sys, json, numpy as np, imageio.v3 as iio
from PIL import Image, ImageDraw
vid, out, label = sys.argv[1], sys.argv[2], sys.argv[3]
steps = [int(x) for x in sys.argv[4].split(",")]
tiles = []
for st in steps:
    try:
        fr = iio.imread(vid, index=st)
    except Exception:
        fr = np.zeros((360, 1280, 3), np.uint8)
    im = Image.fromarray(fr)
    ImageDraw.Draw(im).text((8, 8), f"{label} step~{st}", fill=(255, 255, 0))
    tiles.append(im)
W, H = tiles[0].width, tiles[0].height
cols = 1 if W > 900 else 2
rows = (len(tiles) + cols - 1) // cols
sheet = Image.new("RGB", (W * cols, H * rows))
for i, t in enumerate(tiles):
    sheet.paste(t, ((i % cols) * W, (i // cols) * H))
sheet.save(out, quality=85)
"""


def cmd_frames(a):
    s = json.loads((EPI / a.eid / "summary.json").read_text())
    if not s.get("video"):
        sys.exit("no video recorded for this episode")
    FRAMES.mkdir(parents=True, exist_ok=True)
    steps = [int(x) for x in a.steps.split(",")][:12]
    out = FRAMES / f"{a.eid}_{'-'.join(map(str, steps))}.jpg"
    sim_py = str(C.SIM_PY)
    p = subprocess.run([sim_py, "-c", FRAME_SCRIPT, s["video"][0], str(out), a.eid, ",".join(map(str, steps))],
                       capture_output=True, text=True)
    if p.returncode:
        sys.exit(p.stderr[-2000:])
    print(out)


def cmd_acts(a):
    spec = json.loads(sys.stdin.read())
    items = spec["items"]
    if not CFG.get("prompt_overrides", True) and (spec.get("prompt") or any(it.get("prompt") for it in items)):
        sys.exit("prompt overrides are disabled in this project: the instruction is fixed; only weights may change")
    if not 1 <= len(items) <= CFG["limits"]["acts_items_per_call"]:
        sys.exit(f"1..{CFG['limits']['acts_items_per_call']} items per call")
    for it in items:
        if not (EPI / it["episode"] / "obs.npz").exists():
            sys.exit(f"no recorded obs for episode {it['episode']}")
    check_budget(acts_items=len(items))
    d = cand_dir(spec.get("cand", "base"))
    if d is not None:
        validate(spec["cand"])
    aid = spec.get("id") or time.strftime("a%m%d-%H%M%S")
    if not NAME_RE.match(aid) or (ACTS / aid).exists():
        sys.exit(f"bad or existing acts id {aid}")
    out = ACTS / aid
    spec["episodes_dir"] = str(EPI)
    spec["task_instruction"] = CFG["instruction"]
    with GpuLock(C.ACTS_GPU):
        env = dict(os.environ, JAX_COMPILATION_CACHE_DIR=str(C.JAX_CACHE), CUDA_VISIBLE_DEVICES=str(C.ACTS_GPU), XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_PYTHON_CLIENT_MEM_FRACTION=C.ACTS_MEM_FRACTION)
        p = subprocess.run([OPENPI_PY, str(HARNESS / "acts_worker.py"), "--output", str(out), "--workspace", str(d) if d else ""],
                           input=json.dumps(spec), text=True, capture_output=True, env=env)
    if p.returncode:
        shutil.rmtree(out, ignore_errors=True)
        sys.exit("acts failed:\n" + (p.stdout + p.stderr)[-5000:])
    charge(acts_calls=1, acts_items=len(items))
    print(p.stdout[-6000:])


def cmd_status(a):
    print(json.dumps({"budget_used": budget(), "limits": CFG["limits"], "task": TASK, "instruction": CFG["instruction"],
                      "candidates": sorted(p.name for p in CANDS.iterdir() if not p.name.startswith("_")) if CANDS.exists() else [],
                      "acts": sorted(p.name for p in ACTS.iterdir()) if ACTS.exists() else []}, indent=1))


def main():
    ap = argparse.ArgumentParser(prog="owsp")
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("tensors"); p.add_argument("regex", nargs="?"); p.set_defaults(f=cmd_tensors)
    p = sp.add_parser("new"); p.add_argument("name"); p.add_argument("--from", dest="from_"); p.add_argument("--note", default=""); p.set_defaults(f=cmd_new)
    p = sp.add_parser("py"); p.add_argument("name"); p.set_defaults(f=cmd_py)
    p = sp.add_parser("diff"); p.add_argument("name"); p.set_defaults(f=cmd_diff)
    p = sp.add_parser("eval"); p.add_argument("name"); p.add_argument("--n", type=int, default=64); p.add_argument("--offset", type=int, default=0); p.add_argument("--prompt", default=None); p.set_defaults(f=cmd_eval)
    p = sp.add_parser("episodes"); p.add_argument("--cand"); p.add_argument("-v", "--verbose", action="store_true"); p.set_defaults(f=cmd_episodes)
    p = sp.add_parser("show"); p.add_argument("eid"); p.set_defaults(f=cmd_show)
    p = sp.add_parser("frames"); p.add_argument("eid"); p.add_argument("--steps", required=True); p.set_defaults(f=cmd_frames)
    p = sp.add_parser("acts"); p.set_defaults(f=cmd_acts)
    p = sp.add_parser("status"); p.set_defaults(f=cmd_status)
    a = ap.parse_args()
    for d in (NOTES, STATE, EPI, ACTS, FRAMES, CANDS):
        d.mkdir(parents=True, exist_ok=True)
    a.f(a)


if __name__ == "__main__":
    main()
