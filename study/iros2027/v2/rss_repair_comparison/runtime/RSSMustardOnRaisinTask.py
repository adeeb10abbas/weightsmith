"""Single-class entry point required by RoboLab's external task resolver."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tasks as _tasks


class RSSMustardOnRaisinTask(_tasks.RSSMustardOnRaisinTask):
    pass
