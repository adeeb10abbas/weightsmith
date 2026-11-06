#!/usr/bin/env python3
"""Launch the weightsmith MCP server over stdio; the same as `weightsmith mcp`.

    python integrations/mcp/server.py [--root WSP_ROOT] [--host WSP_HOST]

Works from a plain checkout (no install) once the MCP SDK is present: pip install 'mcp>=1.10'.
Client configuration: docs/INTEGRATIONS.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the checkout's weightsmith package

from weightsmith.mcp_server import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
