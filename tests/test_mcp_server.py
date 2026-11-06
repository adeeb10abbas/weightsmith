"""MCP server: argument validation and argv building (no SDK needed), tool registration, and a real
JSON-RPC session over stdio, all against a fake owsp so no GPU host is needed."""
import asyncio
import base64
import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path

import pytest

from weightsmith import mcp_server as M

REPO = Path(__file__).resolve().parents[1]
READ_ONLY = ["status", "list_tensors", "list_episodes", "show_episode", "view_frames", "diff_candidate"]
MUTATING = ["create_candidate", "run_python", "evaluate_candidate", "capture_activations"]


# ------------------------------------------------------------------------------ tool layer (no SDK)
def test_commands_map_one_to_one_onto_owsp(fake_owsp):
    o = M.Owsp(fake_owsp.command, env=fake_owsp.env)
    assert json.loads(o.status())["task"] == "FakeTask"
    o.tensors(r"language\.blocks\.0[89]")
    o.episodes("base@p1a2b3c", verbose=True)
    o.show("v2mi-f@p1a2b3c.63db0c.s20000.r2")
    o.diff("v2")
    o.new("v3", from_candidate="v2", note="-- keep the cube; hypothesis: 'in the bowl' binds late")
    o.eval("v3", n=32, offset=16, prompt="Put the cube in the bowl then the banana")
    args = [c["args"] for c in fake_owsp.calls()]
    assert args == [
        ["status"],
        ["tensors", r"language\.blocks\.0[89]"],
        ["episodes", "--cand", "base@p1a2b3c", "-v"],
        ["show", "v2mi-f@p1a2b3c.63db0c.s20000.r2"],
        ["diff", "v2"],
        ["new", "v3", "--from=v2", "--note=-- keep the cube; hypothesis: 'in the bowl' binds late"],
        ["eval", "v3", "--n=32", "--offset=16", "--prompt=Put the cube in the bowl then the banana"],
    ]
    assert {(c["root"], c["host"]) for c in fake_owsp.calls()} == {("/projects/demo", "gpu-box")}


def test_code_and_specs_travel_as_temporary_files(fake_owsp):
    o = M.Owsp(fake_owsp.command, env=fake_owsp.env)
    code = "W = load('language.blocks.08.mlp.linear')\nprint(W.shape)\n"
    o.py("base", code)
    spec = {"cand": "v2", "items": [{"episode": "base.base.s20000", "k": 3}], "vlm_layers": [8, 9], "id": "c1"}
    o.acts(spec)
    py, acts = fake_owsp.calls()
    assert py["args"][:2] == ["py", "base"] and py["file"] == code
    assert acts["args"][0] == "acts" and json.loads(acts["file"]) == spec
    assert not Path(py["file_path"]).exists() and not Path(acts["file_path"]).exists()


def test_frames_returns_the_local_image(fake_owsp):
    path = M.Owsp(fake_owsp.command, env=fake_owsp.env).frames("base.base.s20000", [0, 150, 300])
    assert path.is_file() and path.read_bytes().startswith(b"\xff\xd8")
    assert fake_owsp.calls()[0]["args"] == ["frames", "base.base.s20000", "--steps", "0,150,300"]


def test_owsp_failures_reach_the_editor(fake_owsp):
    with pytest.raises(M.OwspError, match="budget: practice episodes 1200"):
        M.Owsp(fake_owsp.command, env=fake_owsp.env).eval("over-budget")


@pytest.mark.parametrize("call", [
    lambda o: o.show("../../etc/passwd"),
    lambda o: o.show("a/b"),
    lambda o: o.show("-v"),
    lambda o: o.diff("base"),
    lambda o: o.diff("Bad Name"),
    lambda o: o.new("base"),
    lambda o: o.new("--help"),
    lambda o: o.new("ok", from_candidate="../x"),
    lambda o: o.py("x;rm", "print(1)"),
    lambda o: o.py("base", "   "),
    lambda o: o.eval("v1", n=0),
    lambda o: o.eval("v1", n=True),
    lambda o: o.eval("v1", offset=-16),
    lambda o: o.tensors("-x"),
    lambda o: o.tensors("("),
    lambda o: o.episodes("x/y"),
    lambda o: o.frames("base.base.s1", []),
    lambda o: o.frames("base.base.s1", list(range(13))),
    lambda o: o.frames("base.base.s1", [-1]),
    lambda o: o.acts({"items": []}),
    lambda o: o.acts({"items": [{"episode": "e1"}]}),
    lambda o: o.acts({"items": [{"episode": "e1", "k": 1, "step": 2}]}),
    lambda o: o.acts({"items": [{"episode": "../e1", "k": 1}]}),
    lambda o: o.acts({"cand": "Bad", "items": [{"episode": "e1", "k": 1}]}),
])
def test_rejects_unsafe_or_malformed_arguments_before_running_anything(fake_owsp, call):
    with pytest.raises(M.OwspError):
        call(M.Owsp(fake_owsp.command, env=fake_owsp.env))
    assert fake_owsp.calls() == []


def test_finds_this_checkouts_owsp(monkeypatch):
    monkeypatch.delenv("WSP_OWSP", raising=False)
    assert M.find_owsp() == [str(REPO / "bin" / "owsp")]
    monkeypatch.setenv("WSP_OWSP", "python3 /x/fake owsp.py")
    assert M.find_owsp() == ["python3", "/x/fake", "owsp.py"]
    assert M.find_owsp("ssh gpu owsp") == ["ssh", "gpu", "owsp"]


def test_importing_the_module_does_not_import_the_sdk():
    code = ("import sys, weightsmith.mcp_server\n"
            "print(any(m == 'mcp' or m.startswith('mcp.') for m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=REPO, check=True)
    assert out.stdout.strip() == "False"


# ------------------------------------------------------------------------------ MCP SDK (in process)
def _text(result):
    """Text of a tool result across SDK versions (v2: CallToolResult; v1: content list or (content, structured))."""
    content = getattr(result, "content", None)
    if content is None:
        content = result[0] if isinstance(result, tuple) else result
    return "\n".join(c.text for c in content if getattr(c, "type", "") == "text")


def test_registers_typed_tools_read_only_first(fake_owsp):
    pytest.importorskip("mcp")
    server = M.build_server(M.Owsp(fake_owsp.command, env=fake_owsp.env))
    tools = asyncio.run(server.list_tools())
    names = [t.name for t in tools]
    assert names == READ_ONLY + MUTATING
    for t in tools:
        hints = t.annotations.model_dump(by_alias=True, exclude_none=True)
        assert hints["readOnlyHint"] is (t.name in READ_ONLY) and hints["openWorldHint"] is False
        assert t.description and "owsp" in t.description
    schema = {t.name: t.model_dump(by_alias=True)["inputSchema"] for t in tools}
    assert schema["view_frames"]["properties"]["steps"]["type"] == "array"
    assert schema["capture_activations"]["required"] == ["items"]
    assert set(schema["evaluate_candidate"]["required"]) == {"candidate"}
    run_python = next(t for t in tools if t.name == "run_python")
    assert run_python.annotations.model_dump(by_alias=True)["destructiveHint"] is True


def test_tools_call_through_to_owsp(fake_owsp):
    pytest.importorskip("mcp")
    server = M.build_server(M.Owsp(fake_owsp.command, env=fake_owsp.env))
    out = asyncio.run(server.call_tool("capture_activations", {
        "items": [{"episode": "base.base.s20000", "step": 300}], "vlm_layers": [8], "image_tokens": True}))
    assert _text(out).startswith("ok acts")
    spec = json.loads(fake_owsp.calls()[-1]["file"])
    assert spec == {"cand": "base", "items": [{"episode": "base.base.s20000", "step": 300}],
                    "vlm_layers": [8], "image_tokens": True}
    with pytest.raises(Exception, match="budget"):
        asyncio.run(server.call_tool("evaluate_candidate", {"candidate": "over-budget"}))


# ------------------------------------------------------------------------------ MCP over stdio (what clients do)
class StdioSession:
    """A minimal MCP client: newline-delimited JSON-RPC over the server's stdin/stdout."""

    def __init__(self, env, cwd):
        self.proc = subprocess.Popen([sys.executable, str(REPO / "integrations" / "mcp" / "server.py")],
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                     text=True, env=env, cwd=cwd)
        self.lines: queue.Queue = queue.Queue()
        threading.Thread(target=lambda: [self.lines.put(line) for line in self.proc.stdout], daemon=True).start()
        self.next_id = 0

    def notify(self, method):
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "method": method}) + "\n")
        self.proc.stdin.flush()

    def request(self, method, params=None, timeout=60):
        self.next_id += 1
        self.proc.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.next_id, "method": method,
                                          "params": params or {}}) + "\n")
        self.proc.stdin.flush()
        while True:
            try:
                message = json.loads(self.lines.get(timeout=timeout))
            except queue.Empty:
                stderr = self.proc.stderr.read() if self.proc.poll() is not None else "(server still running)"
                raise AssertionError(f"no reply to {method}; server stderr: {stderr}") from None
            if message.get("id") == self.next_id:
                assert "error" not in message, message
                return message["result"]

    def initialize(self):
        result = self.request("initialize", {"protocolVersion": "2025-06-18", "capabilities": {},
                                             "clientInfo": {"name": "pytest", "version": "0"}})
        self.notify("notifications/initialized")
        return result

    def close(self):
        self.proc.stdin.close()
        try:
            self.proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def test_stdio_session_like_an_mcp_client(fake_owsp, tmp_path):
    pytest.importorskip("mcp")
    env = dict(fake_owsp.env, WSP_OWSP=fake_owsp.command, PYTHONDONTWRITEBYTECODE="1")
    s = StdioSession(env, tmp_path)
    try:
        init = s.initialize()
        assert init["serverInfo"]["name"] == "weightsmith" and "owsp" in init.get("instructions", "")
        assert [t["name"] for t in s.request("tools/list")["tools"]] == READ_ONLY + MUTATING
        status = s.request("tools/call", {"name": "status", "arguments": {}})
        assert not status.get("isError") and json.loads(status["content"][0]["text"])["task"] == "FakeTask"
        frames = s.request("tools/call", {"name": "view_frames",
                                          "arguments": {"episode": "base.base.s20000", "steps": [0, 300]}})
        image = next(c for c in frames["content"] if c["type"] == "image")
        assert image["mimeType"] == "image/jpeg" and base64.b64decode(image["data"]).startswith(b"\xff\xd8")
        failed = s.request("tools/call", {"name": "evaluate_candidate", "arguments": {"candidate": "over-budget"}})
        assert failed["isError"] and "budget: practice episodes" in failed["content"][0]["text"]
        unsafe = s.request("tools/call", {"name": "show_episode", "arguments": {"episode": "../../etc/passwd"}})
        assert unsafe["isError"] and "bad episode id" in unsafe["content"][0]["text"]
    finally:
        s.close()
    assert [c["args"][0] for c in fake_owsp.calls()] == ["status", "frames", "eval"]


def test_stdio_session_against_the_real_owsp_runtime(synthetic_project, tmp_path):
    """MCP client -> server -> bin/owsp -> weightsmith/owsp.py on a synthetic project (local mode, no GPU)."""
    pytest.importorskip("mcp")
    env = {k: v for k, v in synthetic_project.env.items() if k != "WSP_OWSP"}
    s = StdioSession(dict(env, PYTHONDONTWRITEBYTECODE="1"), tmp_path)
    try:
        s.initialize()
        status = json.loads(s.request("tools/call", {"name": "status", "arguments": {}})["content"][0]["text"])
        assert status["task"] == "RubiksCubeThenBananaTask" and status["budget_used"]["episodes"] == 0
        made = s.request("tools/call", {"name": "create_candidate", "arguments": {"name": "v1", "note": "test"}})
        assert "created candidate v1" in made["content"][0]["text"]
        episodes = s.request("tools/call", {"name": "list_episodes", "arguments": {"candidate": "base"}})
        assert "1/2 (50.0%)" in episodes["content"][0]["text"]
    finally:
        s.close()
    assert os.path.isdir(synthetic_project.root / "cands" / "v1")
