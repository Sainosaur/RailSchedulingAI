"""
Draft: 8    

validate_layer/validator.py

This module implements a deterministic, safety-critical Validation Layer (VL)
for a PPO-driven autonomous train agent.  It acts as a 4-Layer Pipeline
based on ETCS Level 2 Fixed Block (Train-to-Zone) signalling principles.

Layer 1 — Ingestion & Dynamic Limits
Layer 2 — Kinematic Simulation & Multi-Factor Violation Check
Layer 3 — The Decision Node (Direct Interceptor)
Layer 4 — Explainable AI (XAI) Logging

Assume two trains, train 1 and train 2, with train 1 in front of train 2.

DTZ (Distance-to-Zone) definition
----------------------------------
Each segment is divided into fixed blocks of length SH (spatial headway).
DTZ is the *distance* from the train's front to the start of the next
block boundary ahead.  DTZ is therefore always ≤ SH.
"""

import time
from typing import Tuple
import math

from graph.graph import VLSegment, build_vl_segments
from validate_layer.log_manager import init_log, append_row


# ---------------------------------------------------------------------------
# Safety constants  (internal_layers.md  §4)
# ---------------------------------------------------------------------------

ACCEL: float = 0.5           # traction acceleration  (m/s²)
SERVICE_DECEL: float = -0.5  # comfortable/service braking  (m/s²)
EMERGENCY_DECEL: float = -1.0  # emergency braking limit  (m/s²)
TEMPORAL_BUFFER: float = 5.0   # headway reaction buffer  (seconds)
SIM_STEPS: list[int] = [5, 10, 15]  # simulation lookahead times  (seconds)

# AI Action Space (Continuous Throttle)
# The RL Agent outputs a continuous float representing proposed acceleration:
# proposed_a ∈ [-1.0, 0.5]
# Where -1.0 = Emergency Brake, 0.0 = Coast, 0.5 = Max Acceleration

# The environment independently provides the 4-aspect signal bounding this system:
# Aspect 0 = Red (stop), 1 = Orange (cautious), 2 = Flashing Green (moderate), 3 = Green (clear).


# ---------------------------------------------------------------------------
# Validation Layer
# ---------------------------------------------------------------------------

class ValidationLayer:
    """
    4-layer deterministic safety pipeline.

    Public API
    ----------
    get_safe_action(proposed_a, env_aspect, x, v, dtz) -> (safe_action, was_overridden)
    compute_dtz(x)                                     -> float  (distance to next block)

    The AI outputs a continuous acceleration parameter `proposed_a` in [-1.0, 0.5].
    The VL tests this requested physical trajectory directly against the safe dynamic
    boundaries strictly established by the environment's `env_aspect`.
    Joint centralized action spaces (e.g. 0-7) are not used.

    DTZ is the distance from the train to the next fixed-block boundary.
    It is always ≤ SH for the current segment.
    """

    def __init__(self):
        # Segments (with block boundaries) are built once by graph.py
        # and cached at module level — no repeated computation.
        self.segments: list[VLSegment] = build_vl_segments()

        # Layer 4 — XAI log (managed by log_manager.py)
        init_log()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def get_segment(self, x: float) -> VLSegment:
        """
        Return the segment that contains position *x*.
        Raises ValueError if *x* is outside the track boundaries.
        """
        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg
        # Allow exact endpoint of the track
        if x == self.segments[-1].end:
            return self.segments[-1]
        
        raise ValueError(f"Position x={x} is off the track "
                         f"[{self.segments[0].start}, {self.segments[-1].end}]")

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

    # ------------------------------------------------------------------
    # Layer 1 — Ingestion & Dynamic Limits
    # ------------------------------------------------------------------

    @staticmethod
    def _speed_for_aspect(aspect: int, segment: VLSegment, dtz: float) -> Tuple[float, float]:
        """
        Map a per-train 4-aspect L2 signal (0-3) to a target speed and available distance.
        
        Aspect mapping (Distance to Hazard):
            0  Red          →  DTZ
            1  Orange       →  DTZ + 1 * SH
            2  Flash-Green  →  DTZ + 2 * SH
            3  Green        →  DTZ + 3 * SH
        """
        # Determine the available clear distance based on the signal aspect (0-3)
        if aspect == 0:
            distance_available = dtz
        elif aspect == 1:
            distance_available = dtz + (1 * segment.spatial_headway)
        elif aspect == 2:
            distance_available = dtz + (2 * segment.spatial_headway)
        else:
            distance_available = dtz + (3 * segment.spatial_headway)

        a_comfort = abs(SERVICE_DECEL)
        v_dynamic = math.sqrt(2 * a_comfort * max(0.0, distance_available))

        # Output the minimum of the dynamically derived speed and the segment speed limit
        target_v = min(v_dynamic, segment.limit_ms)

        return target_v, distance_available

    # ------------------------------------------------------------------
    # Layer 2 — Kinematic Simulation & Multi-Factor Violation Check
    # ------------------------------------------------------------------

    @staticmethod
    def _project(x: float, v: float, proposed_a: float, limit_v: float,
                 t: float) -> Tuple[float, float]:
        """
        Two-phase SUVAT projection over *t* seconds.
        Phase 1 — apply `proposed_a` until speed safely reaches `limit_v` (or strictly 0).
        Phase 2 — coast at `limit_v` (or 0) for any remaining time.
        """
        if proposed_a == 0:
            return x + v * t, v

        if proposed_a > 0:
            if v >= limit_v:
                return x + v * t, v
            t_to_limit = (limit_v - v) / proposed_a
        else:
            if v <= 0.0:
                return x, 0.0
            t_to_limit = (0.0 - v) / proposed_a

        if t <= t_to_limit:
            v_proj = v + proposed_a * t
            x_proj = x + v * t + 0.5 * proposed_a * (t ** 2)
        else:
            x_at_limit = x + v * t_to_limit + 0.5 * proposed_a * (t_to_limit ** 2)
            terminal_v = limit_v if proposed_a > 0 else 0.0
            x_proj = x_at_limit + terminal_v * (t - t_to_limit)
            v_proj = terminal_v

        return max(0.0, x_proj), max(0.0, v_proj)

    def _check_action_safety(
        self,
        proposed_a: float,
        env_aspect: int,
        x: float,
        v: float,
        dtz: float,
    ) -> Tuple[bool, str]:
        """
        Run the 3-step simulation and perform the multi-factor violation 
        check (Layer 2) for a given continuous proposed acceleration.
        """
        current_seg = self.get_segment(x)

        max_safe_v, distance_available = self._speed_for_aspect(env_aspect, current_seg, dtz)
        boundary_x = x + distance_available

        v_ceiling = min(max_safe_v, current_seg.limit_ms)

        for t in SIM_STEPS:
            x_proj, v_proj = self._project(x, v, proposed_a, v_ceiling, t)

            # --- Spatial Violation ---
            if x_proj >= boundary_x:
                return False, "Aspect_Spatial_Violation"

            # --- Temporal Violation ---
            if v_proj > 0.01:
                time_to_zone = (boundary_x - x_proj) / v_proj
                if time_to_zone < TEMPORAL_BUFFER:
                    return False, "Aspect_Temporal_Violation"
            
            # --- Dynamic Kinematics Bounds ---
            if v_proj > max_safe_v + 0.01:
                return False, "Kinematic_Target_Violation"

            # --- Segment Limit Violation ---
            try:
                proj_seg = self.get_segment(x_proj)
                if v_proj > proj_seg.limit_ms + 0.01:
                    return False, f"{proj_seg.id}_Limit"
            except ValueError:
                return False, "Track_Bounds_Violation"

        return True, ""

    # ------------------------------------------------------------------
    # Layer 3 — The Decision Node (Direct Interceptor)
    # ------------------------------------------------------------------

    def get_safe_action(
        self,
        proposed_a: float,
        env_aspect: int,
        x: float,
        v: float,
        dtz: float,
    ) -> Tuple[float, bool]:
        """
        Layer 3 — Continuous Decision Node (Direct Interceptor).

        O(1) execution validating the AI's requested continuous acceleration against 
        environmental bounds. If unsafe, it throws away the float and returns 
        the maximum emergency braking float (-1.0).

        Parameters
        ----------
        proposed_a   : float Continuous AI Throttle Action [-1.0, 0.5].
        env_aspect   : int   Environmental Signal Boundary (0-3).
        x            : float Current position (m).
        v            : float Current speed (m/s).
        dtz          : float Distance to the next fixed-block boundary.

        Return
        -------
        (safe_action, was_overridden) -> (float, bool)

        """
        # Clamp proposed acceleration purely to system capability limits
        proposed_a = max(EMERGENCY_DECEL, min(ACCEL, proposed_a))

        # Absolute check against True environmental bounds
        is_safe, constraint = self._check_action_safety(proposed_a, env_aspect, x, v, dtz)

        if is_safe:
            return proposed_a, False

        # Apply absolute emergency deceleration on safety violation
        self._log_override(proposed_a, float(EMERGENCY_DECEL), constraint)
        return float(EMERGENCY_DECEL), True

    # ------------------------------------------------------------------
    # Layer 4 — Explainable AI (XAI) Logging
    # ------------------------------------------------------------------

    @staticmethod
    def _log_override(
        original: float,
        corrected: float,
        constraint_id: str,
    ) -> None:
        """Append one override event row to validate_layer/override_log.csv."""
        append_row(time.time(), original, corrected, constraint_id)

