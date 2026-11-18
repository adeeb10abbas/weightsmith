"""Single-class external task registration."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import tasks as _tasks
class RSSWBananaLeftPlateTask(_tasks.RSSWBananaLeftPlateTask):
    pass
