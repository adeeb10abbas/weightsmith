# weightsmith editor-side toolkit, CPU only: the `weightsmith` CLI (portable edit files) and the MCP server.
# The GPU-host runtime (OpenPI, RoboLab, Isaac Sim) is not in this image; the MCP server reaches it over ssh.
#
#   docker build -f docker/Dockerfile -t weightsmith .
#   docker run --rm -v "$PWD:/work" weightsmith edit info /work/v2mi-f.safetensors
#
# See docker/README.md.
FROM python:3.11-slim

# bin/owsp is a bash script that reaches the GPU host with ssh.
RUN apt-get update \
 && apt-get install -y --no-install-recommends bash openssh-client \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/weightsmith
COPY pyproject.toml README.md LICENSE NOTICE ./
COPY weightsmith ./weightsmith
COPY bin ./bin
COPY integrations ./integrations
# Editable, so the MCP server finds this checkout's bin/owsp. `hub` adds the Hugging Face client.
RUN find . -name __pycache__ -prune -exec rm -rf {} + \
 && pip install --no-cache-dir -e '.[mcp,hub]' \
 && useradd --create-home --uid 1000 editor

USER editor
WORKDIR /work
ENTRYPOINT ["weightsmith"]
CMD ["--help"]
