"""Entrypoint name required by weightsmith.rollout.start_server."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from server import main

if __name__ == '__main__':
    main()
