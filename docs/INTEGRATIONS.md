# Integrations

weightsmith plugs into the tools you already use. The editor-side toolkit needs only numpy; install it from a clone, or use the [Docker image](../docker/README.md):

```bash
pip install -e '.[mcp]'      # the `weightsmith` command and the MCP server; add `hub` for the Hugging Face export
```

| Integration | Status | What it is | Tested by |
|---|---|---|---|
| [MCP server](#mcp-server) | Ready | `weightsmith mcp`: the `owsp` tools as 10 typed MCP tools | `tests/test_mcp_server.py` (MCP SDK v1 and v2) and `tests/test_runtime_integration.py`; end to end with the official MCP client against the GPU host behind the paper's results |
| [Claude Code](#claude-code), [Claude Desktop](#claude-desktop), [Cursor](#cursor), [Codex CLI](#codex-cli) | Ready | Standard stdio MCP configs, below | The server side, as above |
| [Docker](#docker) | Ready | CPU image of the toolkit; serves MCP over ssh | `edit info / apply` and the MCP session above, run from the image |
| [Portable edit files](#portable-edit-files) | Ready | `weightsmith edit export / info / apply` | `tests/test_edits.py`, `tests/test_cli.py`; both published edits round-trip to their scored fingerprints |
| [Claude Code driver](#headless-editors-benchmark-runs) | Ready | `drivers/claude.sh`: a headless editor for benchmark runs | The Claude Opus runs on the [leaderboard](../LEADERBOARD.md) |
| [Codex CLI driver](#headless-editors-benchmark-runs) | Experimental | `drivers/astra.sh`: the same for GPT-6 Astra | Flags checked against `codex --help`; first-run checks are [listed](COMPARING_EDITORS.md#astra-driver-todo--unverified) |
| [Hugging Face Hub](#hugging-face-hub) | Experimental | Hub-ready folders (card, manifest) for results and edits | `tests/test_hub_export.py`; nothing has been pushed to the Hub yet |
| [`weightsmith distill`](#roadmap) | Roadmap | Closed-form instruction distillation as one command | Today: the [published recipe](../study/results/opus/ctb/v2mi_edit.py), run in `owsp py` |

## MCP server

`weightsmith mcp` is a stdio [Model Context Protocol](https://modelcontextprotocol.io) server. Each tool runs `bin/owsp` with validated arguments, exactly as an editor typing `owsp` would, so it is bound by the same project budget, sandbox and checks. It never runs a shell, and held-out scoring is not exposed.

| Tool | `owsp` command | Effect |
|---|---|---|
| `status` | `owsp status` | read-only |
| `list_tensors` | `owsp tensors [REGEX]` | read-only |
| `list_episodes` | `owsp episodes [--cand C] [-v]` | read-only |
| `show_episode` | `owsp show EID` | read-only |
| `view_frames` | `owsp frames EID --steps ...` | read-only; returns the image |
| `diff_candidate` | `owsp diff C` | read-only |
| `create_candidate` | `owsp new C [--from=B] [--note=...]` | creates a candidate |
| `run_python` | `owsp py C script.py` | NumPy in the GPU host's sandbox; `save()` writes the candidate's edits |
| `evaluate_candidate` | `owsp eval C --n=N --offset=K` | closed-loop episodes, charged to the budget |
| `capture_activations` | `owsp acts spec.json` | activation replay, charged to the budget |

The server finds the GPU host and project the same way `bin/owsp` does: from the environment, or from `wsp.env` at the repository root (template: [`.env.example`](../.env.example)).

| Setting | Flag | Variable | Meaning |
|---|---|---|---|
| Project | `--root` | `WSP_ROOT` | project root on the GPU host (required) |
| GPU host | `--host` | `WSP_HOST` | ssh host of the GPU box; empty runs locally |
| Remote checkout | | `WSP_REMOTE_REPO` | path of this repository on the GPU host (with `WSP_HOST`) |
| Frames | `--frames-dir` | `WSP_FRAMES_DIR` | where `view_frames` copies JPEGs |
| owsp | `--owsp` | `WSP_OWSP` | the owsp command (default: this checkout's `bin/owsp`) |
| Timeout | `--timeout` | | seconds before a call is abandoned (default: none) |

The examples below use `/data/wsp/projects/cube-then-banana` as the project and `gpu-box` as the GPU host. Leave out `WSP_HOST` and `WSP_REMOTE_REPO` when the client runs on the GPU host itself.

**Allow for long calls.** One `evaluate_candidate` call of 64 episodes takes about 25–35 minutes on one 24 GB GPU, so raise the client's tool timeout where it has one (shown below for Claude Code and the Codex CLI).

### Claude Code

```bash
claude mcp add weightsmith \
  -e WSP_HOST=gpu-box -e WSP_REMOTE_REPO=weightsmith \
  -- weightsmith mcp --root /data/wsp/projects/cube-then-banana
claude mcp list                      # weightsmith should show as connected
export MCP_TOOL_TIMEOUT=3600000      # milliseconds; set before starting claude
```

Add `-s project` to share the server with your team through a checked-in `.mcp.json`.

### Claude Desktop

Add to `claude_desktop_config.json` (macOS: `~/Library/Application Support/Claude/`, Windows: `%APPDATA%\Claude\`), then restart Claude Desktop. Desktop does not read your shell's `PATH`, so give the absolute path of `weightsmith` (`which weightsmith`):

```json
{
  "mcpServers": {
    "weightsmith": {
      "command": "/absolute/path/to/venv/bin/weightsmith",
      "args": ["mcp", "--root", "/data/wsp/projects/cube-then-banana"],
      "env": { "WSP_HOST": "gpu-box", "WSP_REMOTE_REPO": "weightsmith" }
    }
  }
}
```

### Cursor

Add the same block to `.cursor/mcp.json` in your project, or to `~/.cursor/mcp.json` for every project:

```json
{
  "mcpServers": {
    "weightsmith": {
      "command": "weightsmith",
      "args": ["mcp", "--root", "/data/wsp/projects/cube-then-banana"],
      "env": { "WSP_HOST": "gpu-box", "WSP_REMOTE_REPO": "weightsmith" }
    }
  }
}
```

### Codex CLI

```bash
codex mcp add weightsmith --env WSP_HOST=gpu-box --env WSP_REMOTE_REPO=weightsmith \
  -- weightsmith mcp --root /data/wsp/projects/cube-then-banana
```

or in `~/.codex/config.toml`, where `tool_timeout_sec` allows long evaluations:

```toml
[mcp_servers.weightsmith]
command = "weightsmith"
args = ["mcp", "--root", "/data/wsp/projects/cube-then-banana"]
env = { WSP_HOST = "gpu-box", WSP_REMOTE_REPO = "weightsmith" }
tool_timeout_sec = 3600
```

### Docker

The [CPU image](../docker/README.md) runs the same server; mount your `~/.ssh` read-only so it can reach the GPU host:

```bash
docker build -f docker/Dockerfile -t weightsmith .
claude mcp add weightsmith -- docker run -i --rm -v ~/.ssh:/home/editor/.ssh:ro \
  -e WSP_HOST=gpu-box -e WSP_REMOTE_REPO=weightsmith \
  weightsmith mcp --root /data/wsp/projects/cube-then-banana
```

### Without installing

From a plain checkout with the MCP SDK present (`pip install 'mcp>=1.10'`), use `python integrations/mcp/server.py` in place of `weightsmith mcp` in any config above.

### Headless editors (benchmark runs)

For benchmark runs, the drivers start an editor with nothing but `owsp` on its `PATH` and resume it until it writes `FINAL.md`: `drivers/claude.sh` (Claude Code) and `drivers/astra.sh` (Codex CLI; experimental until the first-run checks in [COMPARING_EDITORS.md](COMPARING_EDITORS.md#astra-driver-todo--unverified) are done). They do not use MCP, so every editor gets the identical tool surface. See [drivers/](../drivers/README.md).

## Portable edit files

An `owsp` candidate is a directory of full edited tensors that only works next to the base export it was made from. `weightsmith edit` turns it into one self-describing file that anyone with the public checkpoint can inspect, verify and apply.

```bash
weightsmith edit export $WSP_ROOT/cands/abl-imgl8 -o abl-imgl8.safetensors --rank auto   # on the GPU host
weightsmith edit info   abl-imgl8.safetensors                         # add --json for all metadata
weightsmith edit apply  abl-imgl8.safetensors --workspace $BASE_WORKSPACE --out $WSP_ROOT/cands/abl-imgl8-copy
weightsmith edit apply  abl-imgl8.safetensors --arrays base.npz --out edited.npz   # arrays keyed by tensor name
```

- **Format.** A safetensors file (or an npz with the same keys) that stores, per changed tensor, a dense, sparse or low-rank encoding of `target - base`, plus exact fix-ups. JSON metadata records the source candidate and its `owsp` fingerprint, the base checkpoint and export, and per tensor the shape, encoding, changed values, sha256 of base and target values, and the OpenPI parameter path. The full specification is the docstring of [`weightsmith/edits.py`](../weightsmith/edits.py).
- **Exact.** Export checks the reconstruction bit for bit. Apply checks every tensor's base and result against the recorded sha256, and `--workspace` reproduces the source candidate's fingerprint. `--rank N --lossy` drops the fix-ups for a smaller file and reports how many values then differ.
- **Checked on the published edits.** On the GPU host, with `--rank auto`, each exports exactly and re-applies to the fingerprint scored in its [held-out run](../study/results/opus/heldout/):

  | Edit | Changed values | File (dense float32) | Fingerprint after apply |
  |---|---|---|---|
  | `v2mi-f` | 450,468,837 in 17 tensors | 878 MB (2.3 GB) | `63db0c52a51306c9` |
  | `abl-imgl8` | 248,057,499 in 9 tensors | 274 MB (1.2 GB) | `5bf8c4329c1e416a` |

  The low-rank factors take 2–10 MB per layer; most of each file is exact fix-ups for bf16 rounding.
- **No safetensors dependency.** A small numpy reader and writer is built in; files load with the official `safetensors` library in any framework.

From Python, `weightsmith.edits.load(path)` returns an `Edit`; `Edit.apply(arrays)` edits a dict of arrays, and `Edit.apply_to_params(params)` edits an OpenPI params tree restored with `restore_type=np.ndarray`.

## Hugging Face Hub

*Experimental.* [`integrations/hub/export_to_hub.py`](../integrations/hub/export_to_hub.py) packages results or an edit file as a Hub-ready folder with a generated card and a sha256 manifest. Every number in a card is read from the packaged files. Nothing is uploaded unless you pass `--push OWNER/NAME`, which needs `pip install -e '.[hub]'` and a Hugging Face token; repositories are created private unless you add `--public`.

```bash
python integrations/hub/export_to_hub.py dataset --out hub/weightsmith-results
python integrations/hub/export_to_hub.py edit abl-imgl8.safetensors --out hub/abl-imgl8 \
    --heldout study/results/opus/heldout/cube-then-banana__abl-imgl8-RubiksCubeThenBananaTask-90000-128.json
```

## Roadmap

- `weightsmith distill`: closed-form instruction distillation (instruction-swap capture, per-layer update, offline check) as one command. Today the recipe is [`v2mi_edit.py`](../study/results/opus/ctb/v2mi_edit.py), run in `owsp py`, with the method in Section 2 of the [paper](https://adeeb10abbas.github.io/weightsmith/paper.pdf).
- The published edits as files on the Hugging Face Hub.
