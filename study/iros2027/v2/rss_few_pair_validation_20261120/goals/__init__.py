"""Goal definitions and deterministic inputs for the few-pair validation study."""
from .registry import GOAL_IDS, get_goal, runtime_sources
from .layouts import build_layouts, validate_layouts, write_layouts
from .selection import select_failure_decision

__all__ = ["GOAL_IDS", "get_goal", "runtime_sources", "build_layouts",
           "validate_layouts", "write_layouts", "select_failure_decision"]
