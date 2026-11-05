#!/usr/bin/env bash
# Headless GPT-6 Astra weight editor via the OpenAI Codex CLI. Resumes the same thread
# (`codex exec resume`) until FINAL.md exists (or STOP exists, or MAX_RESUMES is reached).
#
# usage: WSP_ROOT=<project root on GPU host> drivers/astra.sh AGENT_DIR
# env: ASTRA_MODEL (gpt-6-astra), ASTRA_EFFORT (max), MAX_RESUMES (40), EDITOR_LOG_DIR,
#      WSP_FRAMES_DIR, WSP_CODEX_HOME (private Codex home; default <log dir>/codex-home)
#
# Invocation follows the earlier weight-space-policy-lab Astra runs (editor.py: CodexSession):
#   --ignore-user-config --skip-git-repo-check --json --model ... -c model_reasoning_effort=...
#   -c approval_policy="never" -c web_search="disabled" -c features.<x>=false ...,
#   OPENAI_API_KEY stripped so the logged-in ChatGPT session is used, resume by thread id.
# Differences from the lab (which had NO shell and parsed code blocks from replies): the shell
# tool stays on so Astra can call `owsp` itself, like Claude's Bash tool; the sandbox is
# workspace-write (writes only in AGENT_DIR, no network); drivers/codex/owsp.rules lets `owsp`
# (and only owsp) run outside the sandbox and forbids interpreters / network clients.
# Unverified points are marked TODO here and listed in docs/COMPARING_EDITORS.md.
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$(cd "${1:?usage: drivers/astra.sh AGENT_DIR}" && pwd)" || exit 1
: "${WSP_ROOT:?set WSP_ROOT to the project root on the GPU host}"
export WSP_ROOT
export WSP_FRAMES_DIR="${WSP_FRAMES_DIR:-$AGENT_DIR/frames}"
export PATH="$repo/bin:$PATH"
LOG="${EDITOR_LOG_DIR:-$AGENT_DIR-logs/astra}"
MAX_RESUMES="${MAX_RESUMES:-40}"
mkdir -p "$LOG" "$WSP_FRAMES_DIR"

# Private Codex home: no user config, skills, MCP servers or rules leak in; sessions for
# `resume` are stored here. Auth is reused from the operator's normal Codex login.
export CODEX_HOME="${WSP_CODEX_HOME:-$LOG/codex-home}"
mkdir -p "$CODEX_HOME/rules"
cp "$repo/drivers/codex/owsp.rules" "$CODEX_HOME/rules/owsp.rules"
# TODO(verify): that codex exec loads $CODEX_HOME/rules/*.rules (the rule file itself is checked
#   with `codex execpolicy check --rules drivers/codex/owsp.rules owsp status` -> allow).
# TODO(verify): auth.json location/format for your Codex version; alternatively run
#   `CODEX_HOME=$CODEX_HOME codex login` once.
if [ ! -e "$CODEX_HOME/auth.json" ] && [ -f "$HOME/.codex/auth.json" ]; then
  ln -s "$HOME/.codex/auth.json" "$CODEX_HOME/auth.json"
fi
unset OPENAI_API_KEY   # as in the lab: never bill an API key that happens to be in the shell

cd "$AGENT_DIR" || exit 1
COMMON=(--ignore-user-config --skip-git-repo-check --json
        --model "${ASTRA_MODEL:-gpt-6-astra}"
        -c "model_reasoning_effort=\"${ASTRA_EFFORT:-max}\""
        -c 'approval_policy="never"'
        -c 'sandbox_mode="workspace-write"'
        -c 'web_search="disabled"')
# Same feature switches as the lab, except the shell / exec / view_image tools stay ON
# (owsp calls, background evals, looking at `owsp frames` JPGs). Extra switches for tools the
# lab's Codex version did not have (browser_use_external, in_app_browser, computer_use) are
# feature names listed by `codex features list` (codex-cli 0.166).
for f in code_mode multi_agent apps plugins remote_plugin browser_use browser_use_external \
         in_app_browser computer_use image_generation; do
  COMMON+=(-c "features.$f=false")
done
# TODO(verify): `owsp eval` blocks ~25-35 min. Confirm the Codex shell tool does not time it
#   out; unified_exec (default on) lets the model start it and poll, like Claude's
#   run_in_background.

# Kick-off and resume prompts: identical wording to drivers/claude.sh, plus a tool-name mapping
# because BRIEF.md names Claude Code tools.
KICK="Read BRIEF.md in your working directory carefully, then do the experiment it describes. Work autonomously for as long as it takes; there is no human to answer questions. Keep NOTEBOOK.md current."
CONT="Continue the experiment from where you left off. Re-read BRIEF.md and NOTEBOOK.md first, check \`owsp status\`, then keep going until FINAL.md is written or the budget is spent."
TOOLS="Tool mapping: BRIEF.md names Claude Code tools. In this session, run \`owsp\` with your shell tool; it is the only command that can reach the GPU host, and nothing can reach the internet. 'Read/Write/Edit file tools' means reading and editing files in your working directory. 'Open it with the Read tool' means viewing the JPG path that \`owsp frames\` prints with your image-viewing tool. 'Bash run_in_background' means starting a long \`owsp eval\` and polling it while you do other \`owsp\` work."

thread_of() {  # first "thread_id" anywhere in a JSONL event stream (as in the lab's CodexSession)
  python3 -c '
import json, sys
def find(v):
    if isinstance(v, dict):
        if isinstance(v.get("thread_id"), str):
            return v["thread_id"]
        v = list(v.values())
    if isinstance(v, list):
        for c in v:
            f = find(c)
            if f:
                return f
for line in open(sys.argv[1]):
    try:
        t = find(json.loads(line))
    except ValueError:
        continue
    if t:
        print(t)
        break
' "$1"
}

i=0
if [ ! -s thread_id ]; then
  printf '%s\n\n%s\n' "$KICK" "$TOOLS" | codex exec "${COMMON[@]}" -o "$LOG/turn-000.last.md" - \
    > "$LOG/turn-000.jsonl" 2> "$LOG/turn-000.err"
  thread_of "$LOG/turn-000.jsonl" > thread_id
  [ -s thread_id ] || { echo "no thread id in $LOG/turn-000.jsonl; see $LOG/turn-000.err" >&2; exit 1; }
fi
while [ ! -f FINAL.md ] && [ "$i" -lt "$MAX_RESUMES" ] && [ ! -f STOP ]; do
  i=$((i+1)); n=$(printf %03d "$i")
  printf '%s\n\n%s\n' "$CONT" "$TOOLS" | codex exec resume "${COMMON[@]}" -o "$LOG/turn-$n.last.md" "$(cat thread_id)" - \
    > "$LOG/turn-$n.jsonl" 2> "$LOG/turn-$n.err"
  sleep 20
done
echo "editor finished: FINAL=$( [ -f FINAL.md ] && echo yes || echo no ) resumes=$i" > "$LOG/DONE"
