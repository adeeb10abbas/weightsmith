#!/usr/bin/env python3
"""Operator: create a project root (on the GPU host) and an editor working dir.

usage: scripts/new_project.py --config configs/stack-yellow-red.json \\
           --root /data/wsp/projects/stack-yellow-red-claude \\
           --brief briefs/stack-yellow-red/BRIEF.md [--prior briefs/stack-yellow-red/PRIOR_FINDINGS.md] \\
           --agent-dir agents/stack-yellow-red-claude

The project root gets config.json (cands/, data/, state/ are created by owsp on first use).
It is created over ssh when WSP_HOST is set, locally otherwise. The agent dir gets BRIEF.md,
PRIOR_FINDINGS.md (optional) and an empty NOTEBOOK.md. Nothing is overwritten.
"""
import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True, type=Path)
ap.add_argument("--root", required=True, help="project root on the GPU host (becomes WSP_ROOT)")
ap.add_argument("--brief", required=True, type=Path)
ap.add_argument("--prior", type=Path)
ap.add_argument("--agent-dir", required=True, type=Path)
a = ap.parse_args()

cfg = a.config.read_text()
host = os.environ.get("WSP_HOST", "")
mk = f"test ! -e {shlex.quote(a.root)}/config.json && mkdir -p {shlex.quote(a.root)} && cat > {shlex.quote(a.root)}/config.json"
cmd = ["ssh", "-o", "BatchMode=yes", host, mk] if host else ["bash", "-c", mk]
if subprocess.run(cmd, input=cfg, text=True).returncode:
    sys.exit(f"could not create {a.root}/config.json (exists already?)")

agent = a.agent_dir
agent.mkdir(parents=True, exist_ok=False)
shutil.copy(a.brief, agent / "BRIEF.md")
if a.prior:
    shutil.copy(a.prior, agent / "PRIOR_FINDINGS.md")
(agent / "NOTEBOOK.md").write_text(
    "# Notebook\n\n(The editor keeps this file current: findings, hypotheses, candidates, results, current best.)\n")
print(f"project root: {a.root} (on {host or 'this machine'})\nagent dir:    {agent.resolve()}\n"
      f"next: WSP_ROOT={a.root} scripts/base_practice.sh 64   # on the GPU host\n"
      f"then: WSP_ROOT={a.root} drivers/claude.sh {agent}    # or drivers/astra.sh")
