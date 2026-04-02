"""
Draft: 3
Corrected to match internal_layers.md (draft 4).

validate_layer/validator.py

This module implements a deterministic, safety-critical Validation Layer (VL)
for a PPO-driven autonomous train agent.  It acts as a 5-Layer "Safety Sieve"
based on ETCS Level 2 Fixed Block (Train-to-Zone) signalling principles.

Layer 1 — Ingestion & Static Limits
Layer 2 — Simulation  (3-step SUVAT lookahead at T+5, T+10, T+15 s)
Layer 3 — Multi-Factor Violation Check  (spatial, temporal, segment)
Layer 4 — Decision Node  (4a pass / 4b sieve)
Layer 5 — XAI Log  (override_log.csv)
"""

import csv
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# Data structures  (internal_layers.md  §3)
# ---------------------------------------------------------------------------

@dataclass
class Segment:
    """One fixed-block segment of Rail Line 104."""
    id: str
    start: float          # metres along track
    end: float            # metres along track
    limit_ms: float       # speed limit  (m/s)
    spatial_headway: float   # SH — metres, no buffer
    temporal_headway: float  # TH — seconds, no buffer


# ---------------------------------------------------------------------------
# Safety constants  (internal_layers.md  §4)
# ---------------------------------------------------------------------------

ACCEL: float = 0.5           # traction acceleration  (m/s²)
EMERGENCY_DECEL: float = -1.0  # service / emergency braking  (m/s²)
TEMPORAL_BUFFER: float = 5.0   # headway reaction buffer  (seconds)
SIM_STEPS: list[int] = [5, 10, 15]  # simulation lookahead times  (seconds)

# 4-Aspect signalling — fixed speed targets for aspects 0-2
# Aspect 3 (Green) is segment-dependent (= segment speed limit)
_ASPECT_SPEEDS: dict[int, float] = {
    0: 0.0,    # Red    — stop
    1: 8.33,   # Yellow — 30 km/h
    2: 16.67,  # Double-Yellow — 60 km/h
}


# ---------------------------------------------------------------------------
# Validation Layer
# ---------------------------------------------------------------------------

class ValidationLayer:
    """
    5-layer deterministic safety sieve.

    Public API
    ----------
    get_safe_action(proposed_act, x, v, dtz) -> (safe_action, was_overridden)
    """

    def __init__(self, log_dir: str = "."):
        # Segment Map  (internal_layers.md  §3 — table)
        self.segments: list[Segment] = [
            Segment("S0",   582,  5481, 25.00, 312.5, 12.5),
            Segment("S1",  5481, 14951, 25.00, 312.5, 12.5),
            Segment("S2", 14951, 37160, 16.67, 138.9,  8.3),
            Segment("S3", 37160, 47017,  8.33,  34.7,  4.2),
            Segment("S4", 47017, 67394,  8.33,  34.7,  4.2),
            Segment("S5", 67394, 76651,  8.33,  34.7,  4.2),
        ]

        # Layer 5 — XAI log path
        self._log_path = os.path.join(log_dir, "override_log.csv")
        self._init_log()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _init_log(self) -> None:
        """Create the override log with a header row if it doesn't exist."""
        if not os.path.exists(self._log_path):
            with open(self._log_path, "w", newline="") as fh:
                csv.writer(fh).writerow(
                    ["timestamp", "original_ppo_a", "corrected_a", "constraint_id"]
                )

    def get_segment(self, x: float) -> Segment:
        """Return the segment that contains position *x*."""
        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg
        # Clamp: before first segment → first; after last → last
        if x >= self.segments[-1].end:
            return self.segments[-1]
        return self.segments[0]

    @staticmethod
    def _speed_for_action(action: int, segment: Segment) -> float:
        """Map a 4-aspect action integer to a target speed (m/s)."""
        if action == 3:
            return segment.limit_ms          # Green = segment limit
        return _ASPECT_SPEEDS.get(action, 0.0)

    # ------------------------------------------------------------------
    # Layer 2 — SUVAT Projection
    # ------------------------------------------------------------------

    @staticmethod
    def _project(x: float, v: float, target_v: float,
                 t: float) -> Tuple[float, float]:
        """
        Two-phase SUVAT projection over *t* seconds.

        Phase 1 — accelerate / brake towards *target_v*.
        Phase 2 — cruise at *target_v* for any remaining time.

        Returns (x_projected, v_projected).
        """
        if abs(target_v - v) < 0.01:
            # Already at target → cruise
            return x + v * t, v

        accel = ACCEL if target_v > v else EMERGENCY_DECEL

        # Time needed to reach target_v from v
        t_to_target = (target_v - v) / accel       # always positive

        if t <= t_to_target:
            # Still accelerating / braking throughout the window
            v_proj = v + accel * t
            x_proj = x + v * t + 0.5 * accel * t ** 2
        else:
            # Reach target partway, then cruise
            x_at_target = x + v * t_to_target + 0.5 * accel * t_to_target ** 2
            x_proj = x_at_target + target_v * (t - t_to_target)
            v_proj = target_v

        v_proj = max(0.0, v_proj)        # speed can never go negative
        return x_proj, v_proj

    # ------------------------------------------------------------------
    # Layer 1 + 3 — Ingestion, Static Limits & Violation Checks
    # ------------------------------------------------------------------

    def _check_action_safety(
        self,
        action: int,
        x: float,
        v: float,
        dtz: float,
    ) -> Tuple[bool, str]:
        """
        Run the 3-step simulation (Layer 2) and perform the multi-factor
        violation check (Layer 3) for a given *action*.

        Returns
        -------
        (is_safe, constraint_id)
            constraint_id is "" when safe.
        """
        current_seg = self.get_segment(x)

        # Layer 1 — static limit: cap target speed at segment limit
        target_v = min(
            self._speed_for_action(action, current_seg),
            current_seg.limit_ms,
        )

        # Layer 2 — simulate at T+5, T+10, T+15
        for t in SIM_STEPS:
            x_proj, v_proj = self._project(x, v, target_v, t)

            # --- Layer 3a: Spatial Violation ---
            # Train must not enter / pass the occupied zone
            if x_proj >= dtz:
                return False, "DTZ_Spatial_Violation"

            # --- Layer 3b: Temporal Violation ---
            # Ensure enough reaction time before reaching DTZ
            if v_proj > 0.01:                       # avoid div-by-zero
                time_to_zone = (dtz - x_proj) / v_proj
                if time_to_zone < TEMPORAL_BUFFER:
                    return False, "DTZ_Temporal_Violation"

            # --- Layer 3c: Segment Violation ---
            # Projected position may have crossed into a new segment
            proj_seg = self.get_segment(x_proj)
            if v_proj > proj_seg.limit_ms + 0.01:   # small tolerance
                return False, f"{proj_seg.id}_Limit"

        return True, ""

    # ------------------------------------------------------------------
    # Layer 4 — Decision Node  (4a / 4b)
    # ------------------------------------------------------------------

    def get_safe_action(
        self,
        proposed_act: int,
        x: float,
        v: float,
        dtz: float,
    ) -> Tuple[int, bool]:
        """
        Layer 4 — The Decision Node.

        4a  If the proposed action passes L2/L3 → return it unchanged.
        4b  Otherwise sieve downward through the action space and return
            the highest safe action.  Log the override (Layer 5).

        Parameters
        ----------
        proposed_act : int   Action from PPO (0-3, 4-aspect).
        x            : float Current position (m).
        v            : float Current speed (m/s).
        dtz          : float Coordinate of the next red signal / occupied
                             block (m).

        Returns
        -------
        (safe_action, was_overridden)
        """
        # 4a — try the proposed action first
        is_safe, constraint = self._check_action_safety(proposed_act, x, v, dtz)
        if is_safe:
            return proposed_act, False

        # 4b — sieve: decrement through action space
        original_constraint = constraint       # why the proposal failed
        for act in range(proposed_act - 1, -1, -1):
            is_safe, _ = self._check_action_safety(act, x, v, dtz)
            if is_safe:
                self._log_override(proposed_act, act, original_constraint)
                return act, True

        # Emergency-stop fallback
        self._log_override(proposed_act, 0, original_constraint)
        return 0, True

    # ------------------------------------------------------------------
    # Layer 5 — XAI Log
    # ------------------------------------------------------------------

    def _log_override(
        self,
        original: int,
        corrected: int,
        constraint_id: str,
    ) -> None:
        """Append one row to override_log.csv."""
        with open(self._log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [time.time(), original, corrected, constraint_id]
            )
