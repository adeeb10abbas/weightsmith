"""MCP server: the ``owsp`` tool surface as typed tools, so any MCP client can act as a weight editor.

    weightsmith mcp                       # stdio; same as: python integrations/mcp/server.py

Every tool runs ``bin/owsp`` with validated arguments, exactly as an editor typing ``owsp`` would.
It reaches the same GPU host (``WSP_HOST`` over ssh, or local when empty) and project
(``WSP_ROOT``), and is bound by the same budget, sandbox and checks. The server never runs a shell
and exposes nothing else. Held-out scoring stays operator-only. Client setup: docs/INTEGRATIONS.md.

Needs the official MCP Python SDK (``pip install 'weightsmith[mcp]'``); v1 (FastMCP) and v2
(MCPServer) are both supported. Importing this module does not import the SDK.
"""
import argparse
import functools
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated, Any

REPO = Path(__file__).resolve().parents[1]
CANDIDATE_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}$")  # owsp's candidate-name rule
# a candidate label: a name, or name@p<hash> for a diagnostic prompt override
LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,40}(@p[0-9a-f]{6})?$")
EPISODE_RE = re.compile(r"^[A-Za-z0-9_@][A-Za-z0-9_.@-]{0,159}$")
MAX_CODE_BYTES = 256_000
MAX_SPEC_BYTES = 1_000_000
MAX_OUTPUT_CHARS = 100_000

INSTRUCTIONS = """\
weightsmith: read, probe and edit the weights of the pi0.5 robot policy (OpenPI pi05_droid_jointpos) running \
closed-loop in RoboLab on a GPU host. Every tool wraps one `owsp` command and is bound by the project's budget.
Typical loop: status -> list_episodes / show_episode / view_frames on base practice episodes -> \
capture_activations on recorded decisions -> run_python on 'base' to analyse weights and activations -> \
create_candidate and run_python with save(name, bf16(...)) to build a closed-form edit -> diff_candidate -> \
evaluate_candidate (paired with base on the same practice slots). Write the hypothesis down before evaluating. \
The task brief (BRIEF.md) sets the rules; held-out seeds are scored by the operator, never through these tools."""


class OwspError(RuntimeError):
    """owsp refused or failed; the message is meant for the editor to read."""


def find_owsp(command: str | None = None) -> list[str]:
    """The owsp command: ``command``, else $WSP_OWSP, else this checkout's bin/owsp, else `owsp` on PATH."""
    command = command or os.environ.get("WSP_OWSP")
    if command:
        return shlex.split(command)
    local = REPO / "bin" / "owsp"
    if local.is_file():
        return [str(local)]
    found = shutil.which("owsp")
    if found:
        return [found]
    raise OwspError("cannot find bin/owsp: run from a weightsmith checkout or set WSP_OWSP")


def _candidate(name: str, *, allow_base: bool = True) -> str:
    if allow_base and name == "base":
        return name
    if not isinstance(name, str) or not CANDIDATE_RE.fullmatch(name) or name == "base":
        raise OwspError(f"bad candidate name {name!r}: use [a-z0-9][a-z0-9_-]* (max 41 chars)"
                        + ("" if allow_base else ", not 'base'"))
    return name


def _episode(eid: str) -> str:
    if not isinstance(eid, str) or not EPISODE_RE.fullmatch(eid) or ".." in eid:
        raise OwspError(f"bad episode id {eid!r}: use an id printed by list_episodes or evaluate_candidate")
    return eid


def _ints(values: Any, what: str, lo: int, hi: int, max_len: int) -> list[int]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= max_len:
        raise OwspError(f"{what}: give 1..{max_len} integers")
    if any(isinstance(v, bool) or not isinstance(v, int) or not lo <= v <= hi for v in values):
        raise OwspError(f"{what}: integers in {lo}..{hi}")
    return [int(v) for v in values]


class Owsp:
    """Runs ``bin/owsp`` with validated arguments: an argv list, never a shell."""

    def __init__(self, command: str | None = None, *, env: dict | None = None,
                 timeout: float | None = None) -> None:
        self.argv = find_owsp(command)
        self.env = dict(os.environ if env is None else env)
        self.timeout = timeout

    def run(self, *args: str) -> str:
        try:
            p = subprocess.run([*self.argv, *args], capture_output=True, text=True, env=self.env,
                               timeout=self.timeout, stdin=subprocess.DEVNULL)
        except subprocess.TimeoutExpired:
            raise OwspError(f"owsp {args[0]} timed out after {self.timeout} s") from None
        except OSError as e:
            raise OwspError(f"cannot run owsp ({self.argv[0]}): {e}") from None
        out = (p.stdout or "").rstrip()
        if p.stderr and p.stderr.strip():
            out = (out + "\n" + p.stderr.rstrip()).strip()
        if len(out) > MAX_OUTPUT_CHARS:
            out = out[:MAX_OUTPUT_CHARS] + f"\n[output truncated at {MAX_OUTPUT_CHARS} chars]"
        if p.returncode:
            raise OwspError(f"owsp {args[0]} failed (exit {p.returncode}):\n{out[-8000:]}")
        return out or "(no output)"

    def _run_with_file(self, args: list[str], text: str, suffix: str) -> str:
        fd, path = tempfile.mkstemp(prefix="wsp-mcp-", suffix=suffix)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(text)
            return self.run(*args, path)
        finally:
            os.unlink(path)

    # ---- read-only
    def status(self) -> str:
        return self.run("status")

    def tensors(self, regex: str = "") -> str:
        if regex:
            if len(regex) > 200 or regex.startswith("-"):
                raise OwspError("regex: at most 200 chars and must not start with '-'")
            try:
                re.compile(regex)
            except re.error as e:
                raise OwspError(f"regex: {e}") from None
        return self.run("tensors", *([regex] if regex else []))

    def episodes(self, candidate: str = "", verbose: bool = False) -> str:
        args = ["episodes"]
        if candidate:
            if not LABEL_RE.fullmatch(candidate) and candidate != "base":
                raise OwspError(f"bad candidate label {candidate!r}")
            args += ["--cand", candidate]
        return self.run(*args, *(["-v"] if verbose else []))

    def show(self, episode: str) -> str:
        return self.run("show", _episode(episode))

    def frames(self, episode: str, steps: list[int]) -> Path:
        steps = _ints(steps, "steps", 0, 100_000, 12)
        out = self.run("frames", _episode(episode), "--steps", ",".join(map(str, steps)))
        path = Path(out.strip().splitlines()[-1])
        if not path.is_file():
            raise OwspError(f"owsp frames did not produce a local image:\n{out[-2000:]}")
        return path

    def diff(self, candidate: str) -> str:
        return self.run("diff", _candidate(candidate, allow_base=False))

    # ---- mutating
    def new(self, name: str, from_candidate: str = "", note: str = "") -> str:
        args = ["new", _candidate(name, allow_base=False)]
        if from_candidate:
            args.append(f"--from={_candidate(from_candidate)}")
        if len(note) > 4000:
            raise OwspError("note: at most 4000 chars")
        return self.run(*args, f"--note={note}")

    def py(self, candidate: str, code: str) -> str:
        if not isinstance(code, str) or not code.strip():
            raise OwspError("code: give the Python source to run")
        if len(code.encode()) > MAX_CODE_BYTES:
            raise OwspError(f"code: at most {MAX_CODE_BYTES} bytes")
        return self._run_with_file(["py", _candidate(candidate)], code, ".py")

    def eval(self, candidate: str, n: int = 64, offset: int = 0, prompt: str = "") -> str:
        if isinstance(n, bool) or not isinstance(n, int) or n <= 0 or isinstance(offset, bool) \
                or not isinstance(offset, int) or offset < 0:
            raise OwspError("n must be a positive integer and offset a non-negative integer")
        if len(prompt) > 500:
            raise OwspError("prompt: at most 500 chars")
        return self.run("eval", _candidate(candidate), f"--n={n}", f"--offset={offset}",
                        *([f"--prompt={prompt}"] if prompt else []))

    def acts(self, spec: dict[str, Any]) -> str:
        if not isinstance(spec, dict) or not isinstance(spec.get("items"), list) or not spec["items"]:
            raise OwspError("spec needs a non-empty 'items' list")
        for item in spec["items"]:
            if not isinstance(item, dict) or "episode" not in item or ("k" in item) == ("step" in item):
                raise OwspError('each item is {"episode": EID, "k": replan} or {"episode": EID, "step": sim_step}')
            _episode(item["episode"])
        _candidate(spec.get("cand", "base"))
        if spec.get("id") is not None:
            _candidate(spec["id"], allow_base=False)
        text = json.dumps(spec)
        if len(text) > MAX_SPEC_BYTES:
            raise OwspError(f"spec: at most {MAX_SPEC_BYTES} bytes")
        return self._run_with_file(["acts"], text, ".json")


# -------------------------------------------------------------------------------------------- MCP server
def _sdk():
    """(server class, Image, ToolError) from MCP SDK v2 (MCPServer) or v1 (FastMCP)."""
    try:
        from mcp.server.mcpserver import Image, MCPServer
        from mcp.server.mcpserver.exceptions import ToolError
        return MCPServer, Image, ToolError
    except ImportError:
        pass
    try:
        from mcp.server.fastmcp import FastMCP, Image
        from mcp.server.fastmcp.exceptions import ToolError
        return FastMCP, Image, ToolError
    except ImportError:
        raise ImportError("the MCP server needs the MCP Python SDK: pip install 'weightsmith[mcp]'") from None


def build_server(owsp: Owsp | None = None):
    """Create the MCP server (not started). Tools are listed read-only first, then mutating."""
    import anyio.to_thread
    from mcp.types import ToolAnnotations
    from pydantic import Field

    Server, Image, ToolError = _sdk()
    owsp = owsp or Owsp()
    server = Server("weightsmith", instructions=INSTRUCTIONS)

    def annotations(title: str, *, read_only: bool, destructive: bool = False, idempotent: bool = False):
        hints = {"title": title, "readOnlyHint": read_only, "openWorldHint": False}
        if not read_only:
            hints.update(destructiveHint=destructive, idempotentHint=idempotent)
        return ToolAnnotations.model_validate(hints)

    async def call(fn, *args, **kwargs):
        try:
            return await anyio.to_thread.run_sync(functools.partial(fn, *args, **kwargs))
        except OwspError as e:
            raise ToolError(str(e)) from None

    def tool(title: str, **hints):
        return server.tool(annotations=annotations(title, **hints), structured_output=False)

    def arg(kind, description: str):
        return Annotated[kind, Field(description=description)]

    Candidate = arg(str, "Candidate name, or 'base' for the unedited policy.")
    NewName = arg(str, "New candidate name: [a-z0-9][a-z0-9_-]*, not 'base'.")
    Edited = arg(str, "Candidate name (not 'base').")
    Episode = arg(str, "Episode id, as printed by list_episodes or evaluate_candidate.")
    Regex = arg(str, r"Python regex on tensor names, e.g. 'language\.blocks\.1[0-6]\.mlp\.linear'. Empty: all.")
    Label = arg(str, "Only this candidate label, e.g. 'base', 'v2' or 'base@p1a2b3c'. Empty: all.")
    Steps = arg(list[int], "1-12 simulator steps to render, e.g. [0, 150, 300].")
    Code = arg(str, "NumPy code to run in the GPU host's CPU sandbox; print what you need.")
    Prompt = arg(str, "Diagnostic instruction override, only if the project allows it; never a weight edit.")
    Layers = arg(list[int], "Layer, step or token indices (see the tool description).")
    Items = arg(list[dict[str, Any]], 'Recorded decisions to replay: [{"episode": EID, "k": replan_index} or '
                                      '{"episode": EID, "step": sim_step}], each with an optional "prompt".')

    # ------------------------------------------------------------ read-only
    @tool("Project status", read_only=True)
    async def status() -> str:
        """Budget used and limits, task and instruction, candidates and activation captures (`owsp status`)."""
        return await call(owsp.status)

    @tool("List editable tensors", read_only=True)
    async def list_tensors(regex: Regex = "") -> str:
        """Editable tensors with shape and statistics (`owsp tensors`). Names look like
        'language.blocks.08.mlp.linear' (VLM) and 'action.blocks.12.mlp_1.linear' (action expert)."""
        return await call(owsp.tensors, regex)

    @tool("List practice episodes", read_only=True)
    async def list_episodes(candidate: Label = "",
                            verbose: arg(bool, "Also print one JSON row per episode.") = False) -> str:
        """Success counts per candidate and fingerprint on practice slots (`owsp episodes`)."""
        return await call(owsp.episodes, candidate, verbose)

    @tool("Show one episode", read_only=True)
    async def show_episode(episode: Episode) -> str:
        """Outcome, failure reason and event timeline (grasps, drops, placements) of a practice episode
        (`owsp show`)."""
        return await call(owsp.show, episode)

    @tool("View camera frames", read_only=True)
    async def view_frames(episode: Episode, steps: Steps):
        """Render camera frames of a practice episode at the given steps; returns the image (`owsp frames`)."""
        path = await call(owsp.frames, episode, steps)
        return [f"{path}", Image(path=path)]

    @tool("Check a candidate", read_only=True)
    async def diff_candidate(candidate: Edited) -> str:
        """Validate a candidate (bf16-exact, allowed tensors, change budget) and report changed values per tensor
        (`owsp diff`)."""
        return await call(owsp.diff, candidate)

    # ------------------------------------------------------------ mutating
    @tool("Create a candidate", read_only=False, destructive=False)
    async def create_candidate(name: NewName,
                               from_candidate: arg(str, "Copy this candidate's edits; empty: start from base.") = "",
                               note: arg(str, "Hypothesis or note stored with the candidate.") = "") -> str:
        """Create an empty candidate to edit, or a copy of another candidate (`owsp new`)."""
        return await call(owsp.new, name, from_candidate, note)

    @tool("Run Python in the weight sandbox", read_only=False, destructive=True)
    async def run_python(candidate: Candidate, code: Code) -> str:
        """Run NumPy code in the GPU host's sandbox: no network, no GPU (`owsp py`). Helpers: load(name), base(name),
        save(name, value), bf16(x), stats(name), saved(), drop_edit(name), episodes(), obs(eid), acts(aid).
        With a candidate, save() overwrites that candidate's edits; with 'base' the weights are read-only."""
        return await call(owsp.py, candidate, code)

    @tool("Evaluate a candidate in closed loop", read_only=False, destructive=False)
    async def evaluate_candidate(candidate: Candidate,
                                 n: arg(int, "Episodes: a multiple of the project's envs (16).") = 64,
                                 offset: arg(int, "First practice slot offset: a multiple of 16.") = 0,
                                 prompt: Prompt = "") -> str:
        """Run the candidate on n practice slots in simulation, paired with base on the same slots (`owsp eval`).
        Spends n practice episodes of budget; 64 episodes take about 25-35 min on one 24 GB GPU."""
        return await call(owsp.eval, candidate, n, offset, prompt)

    @tool("Capture activations", read_only=False, destructive=False)
    async def capture_activations(
        items: Items,
        candidate: Candidate = "base",
        vlm_layers: Layers = [],  # noqa: B006 - pydantic copies defaults per call
        action_layers: Layers = [],  # noqa: B006
        denoise_steps: Layers = [],  # noqa: B006
        action_tokens: Layers = [],  # noqa: B006
        prompt_tokens: Layers = [],  # noqa: B006
        image_tokens: arg(bool, "Also record VLM image tokens (per camera: mean plus a 4x4 grid).") = False,
        action_attention: arg(bool, "Also record action-expert attention.") = False,
        prompt: Prompt = "",
        capture_id: arg(str, "Name for this capture (default: a timestamp).") = "",
    ) -> str:
        """Replay recorded practice decisions through an instrumented forward pass and save activations
        (`owsp acts`); spends one activation item per decision. Defaults: vlm_layers none (VLM layers 0-17),
        action_layers all (0-17), denoise_steps all (0-9), action_tokens the mean over tokens (0-14),
        prompt_tokens 0-23 (0-199). Load the result in run_python with acts('<capture id>')."""
        spec: dict[str, Any] = {"cand": candidate, "items": items}
        for key, value in (("vlm_layers", vlm_layers), ("action_layers", action_layers),
                           ("denoise_steps", denoise_steps), ("action_tokens", action_tokens),
                           ("prompt_tokens", prompt_tokens), ("prompt", prompt), ("id", capture_id)):
            if value:
                spec[key] = value
        if image_tokens:
            spec["image_tokens"] = True
        if action_attention:
            spec["action_attention"] = True
        return await call(owsp.acts, spec)

    return server


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="weightsmith mcp",
        description="MCP server (stdio) exposing the owsp tools, so any MCP client can act as a weight editor.")
    ap.add_argument("--owsp", help="owsp command (default: $WSP_OWSP, this checkout's bin/owsp, or owsp on PATH)")
    ap.add_argument("--root", help="project root on the GPU host (sets WSP_ROOT)")
    ap.add_argument("--host", help="ssh host of the GPU box (sets WSP_HOST; empty = local)")
    ap.add_argument("--frames-dir", help="where view_frames copies JPEGs (sets WSP_FRAMES_DIR)")
    ap.add_argument("--timeout", type=float, default=None,
                    help="seconds before an owsp call is abandoned (default: no limit)")
    a = ap.parse_args(argv)
    env = dict(os.environ)
    for key, value in (("WSP_ROOT", a.root), ("WSP_HOST", a.host), ("WSP_FRAMES_DIR", a.frames_dir)):
        if value is not None:
            env[key] = value
    try:
        server = build_server(Owsp(a.owsp, env=env, timeout=a.timeout))
    except (ImportError, OwspError) as e:
        print(f"weightsmith mcp: {e}", file=sys.stderr)
        return 2
    server.run()  # stdio
    return 0


if __name__ == "__main__":
    sys.exit(main())
