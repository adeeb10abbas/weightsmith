# Editor briefs

Each task dir holds the `BRIEF.md` the editor reads first and, optionally, a `PRIOR_FINDINGS.md`
of hints from earlier editors. `scripts/new_project.py` copies them into the editor's working dir.

- `banana-then-cube/`: the brief used for Claude run 1 (verbatim).
- `cube-then-banana/`: the brief and prior findings used for Claude run 2 (verbatim, frame paths anonymized).
- `stack-yellow-red/`: the capability brief (fixed instruction, prompt overrides disabled) and its general prior findings.

The "Your tools" paragraph describes the reference GPU host. On that host, evals ran on GPU1
and captures on GPU0, which was clock-limited; timings and budgets are quoted for that box.
If your hardware differs, edit that paragraph, and give **both** editors the same text.
The briefs name Claude Code tools (Read/Write/Edit, Bash `run_in_background`).
`drivers/astra.sh` adds a fixed mapping paragraph for Codex instead of changing the brief.
