# Editor drivers

A driver runs one LLM editor headless on one task. It puts only [`bin/owsp`](../bin/owsp) on the editor's `PATH`, starts it in its agent directory with a fixed kick-off prompt, and resumes the same session until the editor writes `FINAL.md`, the budget is spent, or you `touch STOP` in the agent directory. Both drivers send identical prompts, so editors can be compared on the same brief, budget and seeds ([COMPARING_EDITORS.md](../docs/COMPARING_EDITORS.md)).

| Driver | Editor | Tool surface | Status |
|---|---|---|---|
| [`claude.sh`](claude.sh) | Claude Code (`claude -p`, model `opus`, effort `xhigh`) | Bash restricted to `owsp *`, plus Read/Write/Edit/Glob/Grep in the agent directory; no MCP servers, web or subagents | Ready: used for the Claude Opus runs on the [leaderboard](../LEADERBOARD.md) |
| [`astra.sh`](astra.sh) | GPT-6 Astra through the Codex CLI (`codex exec`) | Shell in a workspace-write sandbox; [`codex/owsp.rules`](codex/owsp.rules) lets only `owsp` out and forbids interpreters and network clients | Experimental: flags to confirm on the first run are listed in [COMPARING_EDITORS.md](../docs/COMPARING_EDITORS.md#astra-driver-todo--unverified) |

## Run an editor

```bash
scripts/new_project.py --config configs/cube-then-banana.json --root /data/wsp/projects/ctb-claude \
    --brief briefs/cube-then-banana/BRIEF.md --prior briefs/cube-then-banana/PRIOR_FINDINGS.md \
    --agent-dir agents/ctb-claude
WSP_ROOT=/data/wsp/projects/ctb-claude scripts/base_practice.sh 64      # base on 64 practice slots, uncharged
WSP_ROOT=/data/wsp/projects/ctb-claude drivers/claude.sh agents/ctb-claude   # or drivers/astra.sh
WSP_ROOT=/data/wsp/projects/ctb-claude scripts/progress.sh agents/ctb-claude # budget, candidates, last turns
```

The editor keeps `NOTEBOOK.md` current and ends with `FINAL.md`, naming one candidate or `base`. Only the operator then scores that candidate on held-out slots with `weightsmith/heldout.py`.

| Setting | Default | Meaning |
|---|---|---|
| `WSP_ROOT` | (required) | project root on the GPU host |
| `CLAUDE_MODEL`, `CLAUDE_EFFORT` | `opus`, `xhigh` | Claude Code model and effort |
| `ASTRA_MODEL`, `ASTRA_EFFORT` | `gpt-6-astra`, `max` | Codex model and reasoning effort |
| `MAX_RESUMES` | `40` | resumes before the driver gives up |
| `EDITOR_LOG_DIR` | `<agent dir>-logs/<driver>` | one JSONL transcript per turn, plus `DONE` |
| `WSP_FRAMES_DIR` | `<agent dir>/frames` | where `owsp frames` copies images |

To drive the same tools interactively from an MCP client instead (Claude Desktop, Cursor, an IDE), use `weightsmith mcp`; see [INTEGRATIONS.md](../docs/INTEGRATIONS.md#mcp-server). Benchmark entries should use a driver, so every editor gets the identical tool surface.
