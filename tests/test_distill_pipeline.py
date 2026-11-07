"""`weightsmith distill run` end to end on CPU: select -> capture -> build -> check against a fake GPU host.

The fake host stands in for bin/owsp. It runs the pipeline's real rendered sandbox scripts (select, build,
fingerprint, check) against a synthetic project on disk, and answers `owsp acts` with synthetic captures in the
real on-disk format. The real GPU run is in docs/TUTORIAL.md.
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from weightsmith import distill as D  # noqa: E402
from weightsmith.lab.sandbox import PRELUDE  # noqa: E402

ORIG = "Put the cube then the banana in the bowl"
EXPL = "Put the cube in the bowl then put the banana in the bowl"
F, DM = 64, 6  # toy MLP hidden and model widths


def _bf16():
    g: dict = {}
    exec(PRELUDE.split("\nnp.set_printoptions")[0] + "\n" + PRELUDE[PRELUDE.index("def bf16"):], g)
    return g["bf16"]


def _chunks(closed):
    a = np.zeros((len(closed), 15, 8), np.float32)
    a[:, :, 7] = np.asarray(closed, np.float32)[:, None]
    return a


# replans: 0 = open, 1 = closed; the measured gripper is ~0.3 while an object is held
EPISODES = {  # eid: (gripper commands, measured gripper, success)
    "base.base.s20000": ([0, 0, 1, 0, 0, 0], [0, 0, 0, .3, 0, 0], False),         # grasp k=2, early release k=3
    "base.base.s20001": ([0, 1, 1, 0, 0, 0], [0, 0, .3, .3, 0, 0], False),        # grasp k=1, early release k=3
    "base.base.s20002": ([0, 0, 0, 1, 0, 0], [0, 0, 0, 0, .35, 0], False),        # grasp k=3, early release k=4
    "base.base.s20003": ([0, 1, 1, 1, 1, 1], [0, 0, .3, .3, .3, .3], True),       # grasp k=1, carried: no release
}


class FakeHost:
    """bin/owsp for one synthetic project: rendered `owsp py` scripts run in-process, `owsp acts` is synthetic."""

    def __init__(self, root: Path):
        self.root, self.dry_run, self.calls = root, False, []
        self.data = root / "data"
        epi = self.data / "episodes"
        epi.mkdir(parents=True)
        rows = []
        for eid, (closed, grip, ok) in EPISODES.items():
            (epi / eid).mkdir()
            np.savez(epi / eid / "obs.npz", actions=_chunks(closed), gripper=np.asarray(grip, np.float32)[:, None],
                     step=np.arange(len(closed)) * 15)
            rows.append({"eid": eid, "cand": "base", "fingerprint": "base", "slot": eid[-6:], "success": ok})
        (epi / "index.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (self.data / "acts").mkdir()
        r = np.random.default_rng(3)
        self.W = {D.MLP_DOWN.format(L): _bf16()(r.normal(size=(F, DM)) * 0.1) for L in range(17)}
        self.cands: set[str] = set()
        self.unresponsive: set[str] = set()  # episodes whose decision the explicit wording does not change

    # ---- bin/owsp
    def run(self, *args, file_text=None, file_name=None):
        self.calls.append(args[0] if args[0] != "py" else f"py {args[1]}")
        if args[0] == "status":
            return json.dumps({"instruction": ORIG, "candidates": sorted(self.cands),
                               "acts": sorted(p.name for p in (self.data / "acts").iterdir())}, indent=1)
        if args[0] == "new":
            self.cands.add(args[1])
            (self.root / "cands" / args[1] / "edits").mkdir(parents=True)
            return f"created candidate {args[1]}"
        if args[0] == "py":
            return self._py(args[1], file_text)
        if args[0] == "acts":
            return self._acts(json.loads(file_text))
        raise AssertionError(args)

    def _py(self, cand, script):
        edits, scratch = self.root / "cands" / cand / "edits", self.root / "cands" / cand / "scratch"
        scratch.mkdir(parents=True, exist_ok=True)
        script = (script.replace("/work/data", str(self.data)).replace("/work/edits", str(edits))
                  .replace("/work/scratch", str(scratch)))

        def path(name):
            return edits / f"{name}.npy"

        def load(name):
            return np.load(path(name)) if path(name).exists() else self.W[name].copy()

        def save(name, value):
            assert cand != "base", "the base sandbox is read-only"
            np.save(path(name), np.asarray(value, np.float32))

        def episodes(cand=None, success=None):
            rows = [json.loads(x) for x in (self.data / "episodes" / "index.jsonl").read_text().splitlines()]
            return [r for r in rows
                    if (cand is None or r["cand"] == cand) and (success is None or r["success"] == success)]

        def obs(eid):
            z = np.load(self.data / "episodes" / eid / "obs.npz")
            return {k: z[k] for k in z.files}

        g = {"np": np, "json": json, "os": os, "__name__": "__main__", "bf16": _bf16(), "load": load, "save": save,
             "base": lambda n: self.W[n].copy(), "episodes": episodes, "obs": obs,
             "saved": lambda: sorted(p.stem for p in edits.glob("*.npy")) if edits.exists() else []}
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            exec(compile(script, "<owsp py>", "exec"), g)
        return buf.getvalue()

    def _acts(self, spec):
        out = self.data / "acts" / spec["id"]
        assert not out.exists(), "owsp acts refuses an existing id"
        layers = spec.get("vlm_layers") or []
        rows, hid, res, act = [], [], [], []
        for it in spec["items"]:
            prompt = it.get("prompt") or spec["prompt"]
            seed = int(hashlib.sha256(f"{it['episode']}/{it['k']}".encode()).hexdigest()[:8], 16)
            r = np.random.default_rng(seed)
            h = r.normal(size=(len(layers), 2, 17, F))
            x = r.normal(size=(len(layers), 2, 17, DM))
            if prompt == EXPL:  # the explicit wording shifts every layer's residual a little more
                x = x + np.cumsum(r.normal(size=(len(layers), 2, 17, DM)) * 0.1, axis=0)
            hid.append(h)
            res.append(x)
            recorded = obs_chunk = np.load(self.data / "episodes" / it["episode"] / "obs.npz")["actions"][it["k"]]
            a = obs_chunk.copy()
            explicit = prompt == EXPL and it["episode"] not in self.unresponsive
            if explicit or spec["cand"] != "base":  # the explicit wording and the edit both keep the gripper closed
                a[:, 7] = 1.0
            act.append(a)
            rows.append({"episode": it["episode"], "k": int(it["k"]), "prompt": prompt, "recorded": recorded.tolist()})
        arrays = {"actions": np.stack(act), "recorded_actions": np.stack([np.asarray(r.pop("recorded")) for r in rows])}
        if layers and spec.get("image_tokens"):
            arrays["vlm.img_mlp_hidden"] = np.stack(hid).astype(np.float16)
            arrays["vlm.img_resid_out"] = np.stack(res).astype(np.float16)
        out.mkdir()
        np.savez(out / "acts.npz", **arrays)
        (out / "index.json").write_text(json.dumps({"items": rows, "vlm_layers": layers, "prompt_token_slots": [0]}))
        return json.dumps({"id": spec["id"], "items": len(rows)})


CFG = {
    "name": "t-imgl8", "instruction": ORIG, "explicit_instruction": EXPL,
    "select": {"rule": "early_release", "limit": 2},
    "check_items": {"other_early_releases": {"rule": "early_release", "limit": 8},
                    "grasps": {"rule": "grasp", "success": None, "limit": 8}},
    "edit": {"mode": "image", "layers": [8, 9], "alpha": 1.0, "lam_rel": 1e-3},
}


@pytest.fixture
def host(tmp_path, monkeypatch):
    h = FakeHost(tmp_path / "project")
    monkeypatch.setattr(D, "Owsp", lambda root, owsp_bin, workdir, dry_run: h)
    monkeypatch.setenv("WSP_ROOT", str(h.root))
    return h


def test_run_select_capture_build_check_end_to_end(host, tmp_path, capsys):
    wd = tmp_path / "wd"
    res = D.run_pipeline(CFG, workdir=wd)
    out = capsys.readouterr().out
    # select: the first early release of each failed episode, limited to 2
    assert res["items"] == [{"episode": "base.base.s20000", "k": 3}, {"episode": "base.base.s20001", "k": 3}]
    # check groups come from the same rules but never reuse a build decision
    groups = json.loads((wd / "check_items.json").read_text())
    assert groups["other_early_releases"] == [{"episode": "base.base.s20002", "k": 4}]
    assert [it["episode"] for it in groups["grasps"]] == sorted(EPISODES)
    # captures: original and explicit wording of the decisions, the edited policy, base on the check items
    fp = res["fingerprint"]
    acts = sorted(p.name for p in (host.data / "acts").iterdir())
    assert acts == sorted(["t-imgl8-o", "t-imgl8-x", f"t-imgl8-c-{fp[:6]}", "t-imgl8-cb"])
    # build: one closed-form edit per layer, saved in the candidate
    edits = host.root / "cands" / "t-imgl8" / "edits"
    assert sorted(p.stem for p in edits.glob("*.npy")) == [D.MLP_DOWN.format(8), D.MLP_DOWN.format(9)]
    assert [r["layer"] for r in res["build"]["layers"]] == [8, 9] and res["build"]["other_edits_kept"] == []
    # the fingerprint is the one `owsp eval` prints: sha256 over the sorted edit files' names and bytes
    h = hashlib.sha256()
    for f in sorted(edits.glob("*.npy")):
        h.update(f.name.encode())
        h.update(f.read_bytes())
    assert fp == h.hexdigest()[:16]
    # check: the edit holds at the decisions it was built from and at the other release; grasps unchanged
    d, c = res["check"]["decisions"], res["check"]["check"]
    assert (d["closed_base"], d["closed_expl"], d["closed_cand"], d["cand_agrees_expl"]) == (0, 2, 2, 2)
    assert (c["other_early_releases"]["closed_base"], c["other_early_releases"]["closed_cand"]) == (0, 1)
    assert c["grasps"]["n"] == 4 and c["grasps"]["flips_vs_base"] == 0
    assert "next steps" in out and "bin/owsp eval t-imgl8 --n 32" in out and "heldout.py --cand t-imgl8" in out

    # a second run reuses every capture (no new `owsp acts`) and rebuilds the same edit
    host.calls.clear()
    res2 = D.run_pipeline(CFG, workdir=wd)
    assert "acts" not in host.calls and res2["fingerprint"] == fp
    assert "exists with the same 2 decisions: reusing it" in capsys.readouterr().out


def test_a_changed_selection_never_reuses_stale_captures(host, tmp_path):
    D.run_pipeline(CFG, workdir=tmp_path / "wd", steps=("select", "capture"))
    cfg = {**CFG, "select": {"rule": "early_release", "limit": 3}}
    with pytest.raises(RuntimeError, match="holds other decisions"):
        D.run_pipeline(cfg, workdir=tmp_path / "wd", steps=("select", "capture"))


def test_a_rebuilt_edit_gets_a_fresh_check_capture(host, tmp_path):
    first = D.run_pipeline(CFG, workdir=tmp_path / "wd")
    second = D.run_pipeline({**CFG, "edit": {**CFG["edit"], "alpha": 0.5}}, workdir=tmp_path / "wd")
    assert first["fingerprint"] != second["fingerprint"]
    acts = {p.name for p in (host.data / "acts").iterdir()}
    assert {f"t-imgl8-c-{first['fingerprint'][:6]}", f"t-imgl8-c-{second['fingerprint'][:6]}"} <= acts


def test_project_comes_from_wsp_root_or_the_config(host, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("WSP_ROOT")
    with pytest.raises(KeyError, match="WSP_ROOT"):
        D.run_pipeline(CFG, workdir=tmp_path / "wd", steps=("select",))
    res = D.run_pipeline({**CFG, "project": str(host.root)}, workdir=tmp_path / "wd", steps=("select",))
    assert res["project"] == str(host.root) and "(config)" in capsys.readouterr().out


def test_the_build_reports_edits_it_did_not_write(host, tmp_path):
    D.run_pipeline(CFG, workdir=tmp_path / "wd", steps=("select", "capture", "build"))
    res = D.run_pipeline({**CFG, "edit": {**CFG["edit"], "layers": [9]}}, workdir=tmp_path / "wd",
                         steps=("select", "capture", "build"))
    assert res["build"]["other_edits_kept"] == [D.MLP_DOWN.format(8)]


def test_responsive_only_builds_from_the_decisions_the_wording_changes(host, tmp_path, capsys):
    host.unresponsive = {"base.base.s20001"}
    res = D.run_pipeline({**CFG, "responsive_only": True}, workdir=tmp_path / "wd")
    out = capsys.readouterr().out
    assert res["responsive"] == [0] and res["build"]["n_items"] == 1
    assert "1 of 2 decisions change with the wording" in out
    assert res["check"]["decisions"]["n"] == 2  # the check still replays every selected decision


def test_responsive_only_stops_when_no_decision_changes(host, tmp_path):
    host.unresponsive = set(EPISODES)
    with pytest.raises(RuntimeError, match="nothing to distil"):
        D.run_pipeline({**CFG, "responsive_only": True}, workdir=tmp_path / "wd")
    assert not (host.root / "cands" / "t-imgl8").exists()  # stopped before creating the candidate
