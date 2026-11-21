"""Prespecified single-state selectors; no search over behavioral outcomes."""
from __future__ import annotations
import sys
import numpy as np

try:
    from .registry import REPO, get_goal
except ImportError:
    from registry import REPO, get_goal


def select_failure_decision(goal_id: str, actions, gripper) -> int | None:
    """Choose one original-failure state after the pair-level eligibility gate.

    The successful clarified execution qualifies the pair, but its states are
    not used here. ``None`` rejects an unusable source; callers must record it.
    """
    rule = get_goal(goal_id)['source_selector']
    actions, gripper = np.asarray(actions), np.asarray(gripper)
    if actions.ndim != 3 or actions.shape[1] < 1 or actions.shape[2] < 8:
        raise ValueError('Expected recorded [decision, chunk, action] array')
    if gripper.ndim not in (1, 2) or (gripper.ndim == 2 and gripper.shape[1] != 1):
        raise ValueError('Expected one measured gripper coordinate per decision')
    if len(actions) != len(gripper) or not np.isfinite(actions).all() or not np.isfinite(gripper).all():
        raise ValueError('Malformed source action/gripper sequence')
    if len(actions) == 0:
        return None
    if rule['rule'] == 'midpoint':
        return (len(actions) - 1) // 2
    sys.path.insert(0, str(REPO))
    from weightsmith.distill import select_early_releases
    found = select_early_releases(actions, gripper, max_hold=rule['max_hold'],
                                  min_grip=rule['min_grip'], max_grip=rule['max_grip'],
                                  first_only=True)
    return int(found[0]) if found else None
