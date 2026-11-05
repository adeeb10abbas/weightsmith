#!/usr/bin/env bash
# Headless Claude Code weight editor. Resumes the same session until FINAL.md exists
# (or STOP exists in the agent dir, or MAX_RESUMES is reached).
#
# usage: WSP_ROOT=<project root on GPU host> drivers/claude.sh AGENT_DIR
#   AGENT_DIR holds BRIEF.md (+ PRIOR_FINDINGS.md, NOTEBOOK.md); see scripts/new_project.py.
# env: CLAUDE_MODEL (opus), CLAUDE_EFFORT (xhigh), MAX_RESUMES (40), EDITOR_LOG_DIR, WSP_FRAMES_DIR
#
# Tool surface: Bash restricted to `owsp *` (bin/owsp is the only thing added to PATH), plus
# Read/Write/Edit/Glob/Grep in the working directory. No MCP servers, no web, no subagents.
set -u
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
AGENT_DIR="$(cd "${1:?usage: drivers/claude.sh AGENT_DIR}" && pwd)" || exit 1
: "${WSP_ROOT:?set WSP_ROOT to the project root on the GPU host}"
export WSP_ROOT
export WSP_FRAMES_DIR="${WSP_FRAMES_DIR:-$AGENT_DIR/frames}"
export PATH="$repo/bin:$PATH"
LOG="${EDITOR_LOG_DIR:-$AGENT_DIR-logs/claude}"
MAX_RESUMES="${MAX_RESUMES:-40}"
mkdir -p "$LOG" "$WSP_FRAMES_DIR"
cd "$AGENT_DIR" || exit 1

COMMON=(--model "${CLAUDE_MODEL:-opus}" --effort "${CLAUDE_EFFORT:-xhigh}" --restricted --strict-mcp-config
        --tools "Bash,Read,Write,Edit,Glob,Grep"
        --allowedTools "Bash(owsp *)" "Read" "Write" "Edit" "Glob" "Grep"
        --permission-mode dontAsk --add-dir "$WSP_FRAMES_DIR"
        --output-format stream-json --verbose)
# Kick-off and resume prompts: identical wording in drivers/astra.sh.
KICK="Read BRIEF.md in your working directory carefully, then do the experiment it describes. Work autonomously for as long as it takes; there is no human to answer questions. Keep NOTEBOOK.md current."
CONT="Continue the experiment from where you left off. Re-read BRIEF.md and NOTEBOOK.md first, check \`owsp status\`, then keep going until FINAL.md is written or the budget is spent."

i=0
if [ ! -s session_id ]; then
  claude -p "$KICK" "${COMMON[@]}" < /dev/null > "$LOG/turn-000.jsonl" 2> "$LOG/turn-000.err"
  grep -m1 '"session_id"' "$LOG/turn-000.jsonl" \
    | python3 -c 'import sys,json;print(json.loads(sys.stdin.readline())["session_id"])' > session_id
fi
while [ ! -f FINAL.md ] && [ "$i" -lt "$MAX_RESUMES" ] && [ ! -f STOP ]; do
  i=$((i+1)); n=$(printf %03d "$i")
  claude -p "$CONT" -r "$(cat session_id)" "${COMMON[@]}" < /dev/null > "$LOG/turn-$n.jsonl" 2> "$LOG/turn-$n.err"
  sleep 20
done
echo "editor finished: FINAL=$( [ -f FINAL.md ] && echo yes || echo no ) resumes=$i" > "$LOG/DONE"
