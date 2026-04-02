"""
Draft: 5
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

Assume two trains, train 1 and train 2, with train 1 in front of train 2.

DTZ (Distance-to-Zone) definition
----------------------------------
Each segment is divided into fixed blocks of length SH (spatial headway).
DTZ is the *distance* from the train's front to the start of the next
block boundary ahead.  DTZ is therefore always ≤ SH.
"""

import csv
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.graph import VLSegment, build_vl_segments   # noqa: E402


# ---------------------------------------------------------------------------
# Safety constants  (internal_layers.md  §4)
# ---------------------------------------------------------------------------

ACCEL: float = 0.5           # traction acceleration  (m/s²)
EMERGENCY_DECEL: float = -1.0  # service / emergency braking  (m/s²)
TEMPORAL_BUFFER: float = 5.0   # headway reaction buffer  (seconds)
SIM_STEPS: list[int] = [5, 10, 15]  # simulation lookahead times  (seconds)

# Number of 4-aspect actions per train
ACTIONS_PER_TRAIN: int = 4

# Action space layout (single flat space for both trains):
#   0–3 → Train 1 (front)   |   4–7 → Train 2 (rear)
# Aspect 0 = Red (stop), 1 = Yellow (1/3 limit), 2 = Dbl-Yellow (2/3 limit),
# 3 = Green (segment limit).


# ---------------------------------------------------------------------------
# Validation Layer
# ---------------------------------------------------------------------------

class ValidationLayer:
    """
    5-layer deterministic safety sieve.

    Public API
    ----------
    get_safe_action(proposed_act, x, v, dtz) -> (safe_action, was_overridden)
    compute_dtz(x)                           -> float  (distance to next block)

    The action integers follow the combined layout:
        0–3  →  Train 1       4–7  →  Train 2
    Internally the per-train action (0–3) is decoded for the sieve.

    DTZ is the distance from the train to the next fixed-block boundary.
    It is always ≤ SH for the current segment.
    """

    def __init__(self, log_dir: str = "."):
        # Segments (with block boundaries) are built once by graph.py
        # and cached at module level — no repeated computation.
        self.segments: list[VLSegment] = build_vl_segments()

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

    def get_segment(self, x: float) -> VLSegment:
        """Return the segment that contains position *x*."""
        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg
        # Clamp: before first segment → first; after last → last
        if x >= self.segments[-1].end:
            return self.segments[-1]
        return self.segments[0]

    def compute_dtz(self, x: float) -> float:
        """
        Compute Distance-to-Zone (DTZ).

        DTZ = distance from position *x* to the start of the next
        fixed-block boundary ahead.  Blocks within each segment are
        spaced SH apart, so DTZ is always ≤ SH.

        Returns
        -------
        float   Distance in metres (always > 0 unless exactly on a
                boundary, in which case the *following* boundary is used).
        """
        seg = self.get_segment(x)
        for boundary in seg.block_boundaries:
            if boundary > x:
                return boundary - x
        # x is at or past the last boundary in this segment →
        # next boundary is the start of the following segment.
        seg_idx = self.segments.index(seg)
        if seg_idx + 1 < len(self.segments):
            return self.segments[seg_idx + 1].start - x
        # Beyond all segments — return a large safe value
        return seg.spatial_headway

    @staticmethod
    def _speed_for_action(action: int, segment: VLSegment) -> float:
        """
        Map a per-train 4-aspect action (0–3) to a target speed (m/s).

        The target speeds are proportional to the *current segment's*
        speed limit so they adapt automatically as the train crosses
        segment boundaries:
            0  Red          →  0           (stop)
            1  Yellow       →  1/3 × limit (cautious approach)
            2  Double-Yellow→  2/3 × limit (moderate approach)
            3  Green        →  limit       (full speed)
        """
        fractions = {0: 0.0, 1: 1.0 / 3.0, 2: 2.0 / 3.0, 3: 1.0}
        return fractions.get(action, 0.0) * segment.limit_ms

    @staticmethod
    def _decode_action(combined_action: int) -> Tuple[int, int]:
        """
        Decode a combined action into (train_index, per_train_action).

        Combined layout:  0–3 → Train 1,  4–7 → Train 2.
        """
        train_idx = combined_action // ACTIONS_PER_TRAIN
        per_train = combined_action % ACTIONS_PER_TRAIN
        return train_idx, per_train

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
            # The train has NOT yet reached target_v within the
            # lookahead window, so it is still accelerating or braking
            # throughout.  We use the *actual* current speed (v) and
            # the constant acceleration directly in SUVAT — no
            # "projected / intermediate" values are needed because the
            # motion is a single constant-acceleration phase from t = 0
            # to t = t.
            v_proj = v + accel * t
            x_proj = x + v * t + 0.5 * accel * t ** 2
        else:
            # The train reaches target_v at t_to_target, BEFORE the end
            # of the lookahead window.  After that instant the
            # acceleration drops to zero and the train cruises.  We
            # therefore split into two SUVAT phases:
            #   Phase 1 (0 → t_to_target):  constant accel, uses actual v.
            #   Phase 2 (t_to_target → t):  zero accel, uses target_v
            #                                (the "projected" cruise speed).
            # x_at_target is the position at the moment the train
            # finishes accelerating / braking; the remaining time is
            # covered at the constant cruise speed target_v.
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
        violation check (Layer 3) for a given per-train *action* (0–3).

        Parameters
        ----------
        dtz : float   Distance (metres) from the train to the next
                       fixed-block boundary.  Always ≤ SH.

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

        # Absolute coordinate of the next block boundary
        boundary_x = x + dtz

        # Layer 2 — simulate at T+5, T+10, T+15
        for t in SIM_STEPS:
            x_proj, v_proj = self._project(x, v, target_v, t)

            # --- Layer 3a: Spatial Violation ---
            # Train must not reach or cross the next block boundary
            if x_proj >= boundary_x:
                return False, "DTZ_Spatial_Violation"

            # --- Layer 3b: Temporal Violation ---
            # Ensure enough reaction time before reaching the boundary
            if v_proj > 0.01:                       # avoid div-by-zero
                time_to_zone = (boundary_x - x_proj) / v_proj
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
        proposed_act : int   Per-train action (0–3, 4-aspect).
        x            : float Current position (m).
        v            : float Current speed (m/s).
        dtz          : float Distance to the next fixed-block boundary
                             (m).  Always ≤ SH.

        Returns
        -------
        (safe_action, was_overridden)

        Performance note — sieve cost on segments S0 / S1
        -------------------------------------------------
        The sieve decrements through at most 4 actions (3 → 2 → 1 → 0),
        each running 3 SUVAT projections — a worst-case of 12 lightweight
        arithmetic operations.  On S0/S1 the higher speed limit (90 km/h)
        does *not* increase the number of iterations; it only makes it
        more likely that the higher-speed actions violate a constraint
        and are skipped quickly.  The fixed, small action space (4 actions)
        means the sieve completes in constant O(1) time regardless of
        segment, so performance is not a concern. ??? fact check.
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
