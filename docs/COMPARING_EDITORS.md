# Comparing editors: Claude (Claude Code) vs GPT-6 Astra (Codex CLI)

Both drivers give the editor the **same brief**, the **same `owsp` tool surface**, the same
budget and the same practice seeds. Scoring is done by the operator on held-out seeds that
neither editor can see.

## What is matched

| | `drivers/claude.sh` | `drivers/astra.sh` |
|---|---|---|
| CLI | `claude -p` (Claude Code), resumed with `-r SESSION` | `codex exec`, resumed with `codex exec resume THREAD` |
| model / effort | `opus`, `--effort xhigh` (`CLAUDE_MODEL`, `CLAUDE_EFFORT`) | `gpt-6-astra`, `model_reasoning_effort="max"` (`ASTRA_MODEL`, `ASTRA_EFFORT`) |
| commands | Bash allowlist: `owsp *` only (`--allowedTools "Bash(owsp *)"`, `--permission-mode dontAsk`) | shell tool in a `workspace-write` sandbox (no network); `owsp` allowed outside the sandbox by `drivers/codex/owsp.rules`; interpreters and network clients forbidden |
| files | Read/Write/Edit/Glob/Grep in the agent dir | file edits in the agent dir (sandbox-writable root) |
| images | Read tool on the JPG printed by `owsp frames` | `view_image` on the same JPG |
| off | web, MCP (`--strict-mcp-config`), subagents (`--tools` list) | `web_search="disabled"`, user config/MCP (`--ignore-user-config` + private `CODEX_HOME`), multi_agent, apps, plugins, browser, computer use, image generation |
| prompts | KICK, then CONT on every resume | the same KICK/CONT text + a fixed tool-name mapping paragraph (the brief names Claude Code tools) |
| stop | `FINAL.md` exists, `STOP` exists, or `MAX_RESUMES` (40) | same |

Effort levels are vendor-specific. The defaults are the settings used so far: Claude run 1
used `xhigh`, and the earlier lab's Astra runs used `max`. Record whatever you choose.

## Protocol

1. **One task config, two project roots.** Each editor gets its own candidates, notes and
   budget, and both see the *same* base practice episodes.
   ```bash
   # on the GPU host
   T=stack-yellow-red; P=/data/wsp/projects
   scripts/new_project.py --config configs/$T.json --root $P/$T-claude \
       --brief briefs/$T/BRIEF.md --prior briefs/$T/PRIOR_FINDINGS.md --agent-dir agents/$T-claude
   WSP_ROOT=$P/$T-claude scripts/base_practice.sh 64
   # clone base episodes + config into the second root (hard links), fresh budget
   mkdir -p $P/$T-astra/state && cp -al $P/$T-claude/config.json $P/$T-claude/data $P/$T-astra/
   cp $P/$T-claude/state/budget.json $P/$T-astra/state/
   ```
   Then create `agents/$T-astra` the same way (or copy `agents/$T-claude` before the run starts).
   Run `new_project.py` on the editor machine with `WSP_HOST` set if the editors run there.
2. **Same budgets and seeds.** `limits` (practice episodes, activation items, items per call),
   `practice_slots` and `practice_seed_base` come from the shared config. Every eval of either
   editor draws from the same 256 practice slots, so the paired-vs-base numbers are comparable.
3. **Run the editors.**
   ```bash
   WSP_ROOT=$P/$T-claude drivers/claude.sh agents/$T-claude
   WSP_ROOT=$P/$T-astra  drivers/astra.sh  agents/$T-astra
   ```
   - Both drivers share the GPU locks. If they run at the same time, their evals queue behind
     each other, and wall-clock waiting differs between them.
   - For a clean comparison, run them one after the other, or on two identical GPU hosts.
   - `touch agents/<dir>/STOP` ends a run after the current turn.
4. **Score (operator only).** Each editor names one candidate in `FINAL.md`. On the GPU host:
   ```bash
   WSP_ROOT=$P/$T-claude python3 weightsmith/heldout.py --cand <name> --slot-start 90000 --n 192
   WSP_ROOT=$P/$T-astra  python3 weightsmith/heldout.py --cand <name> --slot-start 90000 --n 192
   ```
   - Use the **same held-out slots for both**. Held-out slots must be ≥ 80000. Practice slots
     are 20000..20255. Editors cannot read anything under `heldout/`.
   - `heldout.py` freezes a copy of the candidate and checks its fingerprint. It then runs base
     and candidate on identical slots, alternating order by half.
   - It reports base/candidate success with Wilson 95% CIs, fixed/broke counts and an exact
     McNemar p-value. Both editors are paired against the same base slots.
   - A submission of `base` scores as no change.
   - Report the held-out deltas. Also report practice budget used, activation items used and
     wall-clock time, and whether the edit's mechanism story survived its own predictions
     (from `NOTEBOOK.md`).

## Astra driver: TODO / unverified

Verified without spending model calls:
- All `codex exec` / `codex exec resume` flags used appear in `--help` (codex-cli 0.166), and
  most come from the lab's working invocation.
- `codex execpolicy check --rules drivers/codex/owsp.rules …` returns `allow` for `owsp …`
  and `forbidden` for `python3`, `curl`, `ssh` and `git`.

Not verified (check on the first run, by reading `agents/<dir>-logs/astra/turn-000.jsonl`):
- [ ] **Rules location.** `codex exec` must load rules from `$CODEX_HOME/rules/*.rules`
  (the driver uses a private `CODEX_HOME`).
- [ ] **`allow` semantics.** An `allow` rule must make `owsp` run *outside* the
  workspace-write sandbox, because it needs network for ssh. If it does not, `owsp` fails with
  a network error. There is no fallback that keeps "no web": do **not** enable
  `sandbox_workspace_write.network_access`.
- [ ] **Auth.** The driver symlinks `~/.codex/auth.json` into the private `CODEX_HOME`.
  Otherwise run `CODEX_HOME=<dir> codex login` once.
- [ ] **Long evals.** `owsp eval` blocks 25-35 min. Check that the shell tool does not time it
  out, and that a backgrounded or polled eval (unified exec) keeps running.
- [ ] **Feature names.** The `features.*=false` names vary across Codex versions. Unknown
  keys are not rejected unless `--strict-config` is set.
- [ ] **Parity gap.** Claude can run only `owsp`. Astra can still run harmless local
  read-only commands (`ls`, `cat`, `sed`) in its sandbox. These have no network, write only
  in the agent dir, and Codex reads files that way. The sandbox does not confine *reads* to
  the agent dir.
