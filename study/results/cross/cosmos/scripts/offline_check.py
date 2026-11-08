"""Offline check of an edit from a replay capture (numpy): do the edited policy's actions under the original wording
move toward the unedited policy's actions under the known-good wording, on the same inputs and noise?

Reads the actions of every variant in a capture.py output ([D, 32, 8] per variant, D decisions) and reports, per
variant against the teacher (the unedited policy under the known-good wording) and against the student (the unedited
policy under the original wording): the chunk-end end-effector distance (Franka FK of the commanded joints), the
mean joint distance over the chunk, gripper-command agreement over the 32 steps, and how many decisions moved closer
to the teacher than the unedited student is.

usage: offline_check.py CAPTURE.npz --teacher G --student O [--variants O+edit,O+ctl] [--out OUT.json]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Franka Panda DH, flange + 0.257 m (weightsmith.distill.panda_fk, copied so this runs without the package)
_DH = [(0.0, 0.333, 0.0), (0.0, 0.0, -np.pi / 2), (0.0, 0.316, np.pi / 2), (0.0825, 0.0, np.pi / 2),
       (-0.0825, 0.384, -np.pi / 2), (0.0, 0.0, np.pi / 2), (0.088, 0.0, np.pi / 2)]


def panda_fk(q):
    T = np.eye(4)
    for i, (a, d, al) in enumerate(_DH):
        th = float(q[i])
        ca, sa, ct, st = np.cos(al), np.sin(al), np.cos(th), np.sin(th)
        T = T @ np.array([[ct, -st, 0, a], [st * ca, ct * ca, -sa, -d * sa], [st * sa, ct * sa, ca, d * ca],
                          [0, 0, 0, 1]])
    F = np.eye(4)
    F[2, 3] = 0.257
    return (T @ F)[:3, 3]


def ee_end(a):
    return np.array([panda_fk(x[-1, :7]) for x in a])


def compare(a, b):
    return {"ee_end_cm": 100 * np.linalg.norm(ee_end(a) - ee_end(b), axis=1),
            "joint_mean_rad": np.abs(a[:, :, :7] - b[:, :, :7]).mean((1, 2)),
            "grip_agree": ((a[:, :, 7] > 0.5) == (b[:, :, 7] > 0.5)).mean(1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("capture")
    ap.add_argument("--teacher", default="G")
    ap.add_argument("--student", default="O")
    ap.add_argument("--variants", default="")
    ap.add_argument("--groups", default="", help="name:i0-i1,name:i0-i1 decision index ranges to report separately")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    X = np.load(a.capture)
    T, S = X[f"{a.teacher}/actions"], X[f"{a.student}/actions"]
    groups = {"all": np.arange(len(T))}
    for g in filter(None, a.groups.split(",")):
        name, rng = g.split(":")
        i0, i1 = map(int, rng.split("-"))
        groups[name] = np.arange(i0, i1 + 1)
    base = compare(S, T)
    rep = {"capture": a.capture, "teacher": a.teacher, "student": a.student, "decisions": int(len(T)), "groups": {}}
    for gname, idx in groups.items():
        g = {"student_vs_teacher": {k: float(v[idx].mean()) for k, v in base.items()}}
        for v in filter(None, a.variants.split(",")):
            A = X[f"{v}/actions"]
            c, cs = compare(A, T), compare(A, S)
            g[v] = {"vs_teacher": {k: float(x[idx].mean()) for k, x in c.items()},
                    "vs_student": {k: float(x[idx].mean()) for k, x in cs.items()},
                    "closer_to_teacher_ee": int((c["ee_end_cm"][idx] < base["ee_end_cm"][idx]).sum()),
                    "n": int(len(idx)),
                    "gap_closed_ee_pct": float(100 * (1 - c["ee_end_cm"][idx].mean() / max(base["ee_end_cm"][idx].mean(), 1e-9)))}
        rep["groups"][gname] = g
    s = json.dumps(rep, indent=1)
    if a.out:
        Path(a.out).write_text(s)
    print(s)


if __name__ == "__main__":
    main()
