"""
validate_layer/log_manager.py

Manages the override_log.csv file for the Validation Layer.
The log lives inside validate_layer/ and appends rows during operation.
It is only reset (deleted and re-created) when LOG_RESET is True.
"""

import csv
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Hardcoded reset flag — set to True to wipe the log on next run.
# LOG_ENABLED flag — set to False during PPO training to prevent disk I/O thrashing.
# ---------------------------------------------------------------------------
LOG_RESET: bool = True
LOG_ENABLED: bool = False

# Log file lives inside validate_layer/
_LOG_DIR = str(Path(__file__).resolve().parent)
_LOG_FILENAME = "override_log.csv"
_LOG_PATH = os.path.join(_LOG_DIR, _LOG_FILENAME)

_HEADER = ["timestamp", "original_ppo_a", "corrected_a", "constraint_id"]


def get_log_path() -> str:
    """Return the absolute path to override_log.csv."""
    return _LOG_PATH


def init_log() -> None:
    """
    Initialise the override log.

    * If LOG_RESET is True  → delete existing log and create a fresh one.
    * If LOG_RESET is False → create the log only if it does not exist
      (preserving previous rows).
    """
    if not LOG_ENABLED:
        return

    if LOG_RESET and os.path.exists(_LOG_PATH):
        os.remove(_LOG_PATH)

    if not os.path.exists(_LOG_PATH):
        with open(_LOG_PATH, "w", newline="") as fh:
            csv.writer(fh).writerow(_HEADER)


def append_row(timestamp: float, original: float, corrected: float,
               constraint_id: str) -> None:
    """Append a single override event row to the log."""
    if not LOG_ENABLED:
        return

    with open(_LOG_PATH, "a", newline="") as fh:
        csv.writer(fh).writerow(
            [timestamp, original, corrected, constraint_id]
        )
