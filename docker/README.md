# Docker: the editor-side toolkit

A CPU-only image with the `weightsmith` command: portable edit files (`weightsmith edit`) and the MCP server (`weightsmith mcp`). The GPU-host runtime (OpenPI, RoboLab, Isaac Sim) is not in the image; the MCP server reaches it over ssh, exactly like [`bin/owsp`](../bin/owsp). To set up the GPU host itself, see [docs/SETUP.md](../docs/SETUP.md).

```bash
docker build -f docker/Dockerfile -t weightsmith .     # from the repository root
docker run --rm weightsmith --version
```

## Inspect and apply edit files

Mount the folder that holds your files at `/work`:

```bash
docker run --rm -v "$PWD:/work" weightsmith edit info  /work/v2mi-f.safetensors
docker run --rm -v "$PWD:/work" weightsmith edit apply /work/v2mi-f.safetensors --arrays /work/base.npz --out /work/edited.npz
```

The container runs as uid 1000, so the mounted folder must be writable by it for `apply`.

## Serve MCP from the container

The server needs ssh access to the GPU host. Mount your `~/.ssh` read-only (a key without a passphrase prompt, and the host in `known_hosts`), and pass the host and project:

```bash
claude mcp add weightsmith -- docker run -i --rm -v ~/.ssh:/home/editor/.ssh:ro \
  -e WSP_HOST=gpu-box -e WSP_REMOTE_REPO=weightsmith \
  weightsmith mcp --root /data/wsp/projects/cube-then-banana
```

For Claude Desktop, Cursor or the Codex CLI, use the same command as `"command": "docker"` with the rest as `args`; the config formats are in [docs/INTEGRATIONS.md](../docs/INTEGRATIONS.md). Keep `-i`: MCP talks over stdin and stdout. Paths in a mounted `~/.ssh/config` must work inside the container (use `~/.ssh/...`, not absolute host paths).

## Verified

- `weightsmith edit info` and `weightsmith edit apply --arrays` on a mounted edit file reproduce the edited values bit for bit.
- An MCP client (the official Python SDK over stdio) started this image as above, listed the 10 tools, and ran `status` and `list_tensors` on the GPU host behind the paper's results. Checked on macOS with colima; on Linux, make sure uid 1000 can read the mounted key.

With BuildKit, [`Dockerfile.dockerignore`](Dockerfile.dockerignore) keeps the build context to the files the image copies.
