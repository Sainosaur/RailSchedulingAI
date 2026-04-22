"""
Draft: 9

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

import math
import time
from typing import Tuple

try:
    from graph.graph import VLSegment, build_vl_segments
except ImportError:
    from graph import VLSegment, build_vl_segments

from validate_layer.log_manager import append_row, init_log

# ---------------------------------------------------------------------------
# Safety constants  (internal_layers.md  §4)
# ---------------------------------------------------------------------------

ACCEL: float = 0.5  # traction acceleration  (m/s²)
SERVICE_DECEL: float = -0.5  # comfortable/service braking  (m/s²)
EMERGENCY_DECEL: float = -1.0  # emergency braking limit  (m/s²)
# Lookahead times are computed per-segment in _check_action_safety,
# scaled to t_stop = v_limit / |EMERGENCY_DECEL| for each segment.

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
    get_safe_action(proposed_a, env_aspect, x, u, dtz) -> (safe_action, was_overridden)
    compute_dtz(x)                                     -> float  (distance to next block)

    The AI outputs a continuous acceleration parameter `proposed_a`.
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
        If *x* slightly overshoots the track boundaries computationally, it clamps 
        to the nearest segment to prevent brittle ValueError crashes in PPO.
        """
        if x <= self.segments[0].start:
            return self.segments[0]
            
        if x >= self.segments[-1].end:
            return self.segments[-1]

        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg

        raise ValueError(
            f"Position x={x} is completely unresolvable "
            f"[{self.segments[0].start}, {self.segments[-1].end}]"
        )

    def compute_dtz(self, x: float) -> float:
        """
        Compute Distance-to-Zone (DTZ).

        DTZ = distance from position *x* to the start of the next
        fixed-block boundary ahead.  Blocks within each segment are
        spaced SH apart, so DTZ is always ≤ SH.
        # Note: zones shorter than SH are merged at generation time by
        # build_vl_segments() in graph.py, so every zone has length >= SH.

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
    def _speed_for_aspect(
        aspect: int, segment: VLSegment, dtz: float
    ) -> Tuple[float, float]:
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

        # BUG 4/5 FIX: Use EMERGENCY_DECEL (1.0 m/s²) — the train's actual
        # braking capability — not SERVICE_DECEL (0.5 m/s²).  Using 0.5 made
        # the VL compute a safe-speed ceiling only half what the train can
        # actually achieve, causing spurious overrides throughout training.
        a_brake = abs(EMERGENCY_DECEL)
        v_dynamic = math.sqrt(2 * a_brake * max(0.0, distance_available))

        # Output the minimum of the dynamically derived speed and the segment speed limit
        target_v = min(v_dynamic, segment.limit_ms)

        return target_v, distance_available

    # ------------------------------------------------------------------
    # Layer 2 — Kinematic Simulation & Multi-Factor Violation Check
    # ------------------------------------------------------------------

    @staticmethod
    def _project(
        x: float, u: float, proposed_a: float, v_ceil: float, t: float
    ) -> Tuple[float, float]:
        """
        Two-phase SUVAT projection over *t* seconds.

        Phase 1 — apply proposed_a until speed reaches v_ceil (or 0 on braking).
        Phase 2 — cruise at v_ceil (or rest at 0) for remaining time.

        Notation follows SUVAT convention: u = initial velocity, v = final velocity.
        Returns (x_final, v) — projected position and final velocity.
        """
        if proposed_a == 0:
            return x + u * t, u

        if proposed_a > 0:
            if u >= v_ceil:
                # BUG 6 FIX: coast at the ceiling, not at u (which may be
                # above v_ceil due to a prior segment with a higher limit).
                return x + v_ceil * t, v_ceil 
            t_to_limit = (v_ceil - u) / proposed_a
        else:
            if u <= 0.0:
                return x, 0.0
            t_to_limit = (0.0 - u) / proposed_a

        if t <= t_to_limit:
            v = u + proposed_a * t
            x_proj = x + u * t + 0.5 * proposed_a * (t**2)
        else:
            x_at_limit = x + u * t_to_limit + 0.5 * proposed_a * (t_to_limit**2)
            v = v_ceil if proposed_a > 0 else 0.0
            x_proj = x_at_limit + v * (t - t_to_limit)

        # BUG 19 FIX: do not clamp x_proj to 0.  A projected position going
        # negative is physically impossible in normal operation; if it occurs
        # it indicates a numerical edge case that should surface as a
        # Track_Bounds_Violation via get_segment(), not be silently hidden.
        return x_proj, max(0.0, v)

    def _check_action_safety(
        self,
        proposed_a: float,
        env_aspect: int,
        x: float,
        u: float,
        dtz: float,
    ) -> Tuple[bool, str]:
        """
        Run the multi-factor violation check for a given continuous proposed
        acceleration using adaptive per-segment lookahead times.

        Lookahead times are scaled to the segment's stopping distance:
            t_stop = v_limit / |EMERGENCY_DECEL|
        Sampled at 33%, 67%, and 100% of t_stop.  This prevents false
        Spatial Violations in short-block segments (S3-S5) where a fixed
        15-second window overshoots the 3-block Green authority even at a
        safe cruising speed.  The horizon is capped at the minimum time
        physically needed to guarantee a safe stop — consistent with ETCS
        Level 2 movement authority principles.

        The 100% horizon is safe: at t_stop, a train coasting at v_limit travels
        at most v_limit * t_stop = v_limit² / |a| metres — approximately one
        stopping distance (1 × SH).  Green authority spans 3 × SH, so the
        100% horizon never reaches the Green boundary under normal operation.

        The Temporal Violation check has been removed as it is provably redundant.
        Proof: if time_to_zone < 5s at sim step t₁, remaining distance = v × t_zone < v × 5.
        The last sim step t₃ = t_stop and t₃-t₁ = 0.67×t_stop ≥ 5.36s (all segments,
        min t_stop = 8s for S3). At t₃ the train travels v×5.36 > remaining distance
        → Spatial Violation fires. There is no case where temporal triggers but spatial
        does not.
        """
        current_seg = self.get_segment(x)

        max_safe_v, distance_available = self._speed_for_aspect(
            env_aspect, current_seg, dtz
        )
        boundary_x = x + distance_available

        v_ceiling = current_seg.limit_ms

        # Adaptive lookahead: scale to segment stopping distance.
        # t_stop is the minimum time to decelerate from the segment speed limit
        # to rest under emergency braking.  Evaluating beyond this is redundant
        # — the train will already have stopped.
        # BUG 17 FIX: use floor() not round(); round() can produce t > t_stop
        # (e.g. round(8.5) = 9 when t_stop=8.33), projecting past the safe
        # stop horizon.
        t_stop = current_seg.limit_ms / abs(EMERGENCY_DECEL)
        sim_steps = [
            max(1, math.floor(t_stop * 0.33)),
            max(2, math.floor(t_stop * 0.67)),
            max(3, math.floor(t_stop)),
        ]

        for t in sim_steps:
            x_proj, v = self._project(x, u, proposed_a, v_ceiling, t)

            # BUG 7 FIX: update v_ceiling after each projection step so that
            # if the train crosses into a slower segment the remaining
            # projections use the tighter limit.  This prevents the validator
            # from modelling the train as coasting at the old (higher) ceiling
            # through a segment with a lower speed limit.
            try:
                proj_seg = self.get_segment(x_proj)
                v_ceiling = min(v_ceiling, proj_seg.limit_ms)
            except ValueError:
                return False, "Track_Bounds_Violation"

            # --- Spatial Violation ---
            if x_proj >= boundary_x:
                return False, "Aspect_Spatial_Violation"

            # --- Dynamic Kinematics Bounds ---
            if v > max_safe_v + 0.01:
                return False, "Kinematic_Target_Violation"

            # --- Segment Limit Violation ---
            if v > proj_seg.limit_ms + 0.01:
                return False, f"{proj_seg.id}_Limit"

        return True, ""

    # ------------------------------------------------------------------
    # Layer 3 — The Decision Node (Direct Interceptor)
    # ------------------------------------------------------------------

    def get_safe_action(
        self,
        proposed_a: float,
        env_aspect: int,
        x: float,
        u: float,
        dtz: float,
    ) -> Tuple[float, bool]:
        """
        Layer 3 — Continuous Decision Node (Direct Interceptor).

        Validates the AI's requested continuous acceleration against environmental
        bounds.  If unsafe, computes the minimum deceleration required to stop
        within the available distance or respect the limit.

        OVERRIDE DIFFERENTIATION:
        - Safety Overrides (True): Aspect_Spatial, Kinematic_Target, Track_Bounds.
          These represent mistakes by the AI and are punished.
        - System Clamps (False): Hardware_Limit, Segment_Limit.
          These are purely physical/maintainance constraints and are NOT punished.

        Returns
        -------
        (safe_a, was_safety_overridden) -> (float, bool)
        """
        # 1. Hardware clamp (System Clamp)
        clamped_a = max(EMERGENCY_DECEL, min(ACCEL, proposed_a))
        was_hardware_clamped = clamped_a != proposed_a

        # 2. Safety check (Layer 2)
        is_safe, constraint = self._check_action_safety(
            clamped_a, env_aspect, x, u, dtz
        )

        if is_safe:
            if was_hardware_clamped:
                self._log_override(proposed_a, clamped_a, "Hardware_Limit_Clamp")
                # Hardware clamp is NOT a safety violation — return False
                return clamped_a, False
            return clamped_a, False

        # 3. Already stopped
        if u <= 0.01:
            return 0.0, False

        # 4. Violation Resolution
        seg = self.get_segment(x)
        _, distance_available = self._speed_for_aspect(env_aspect, seg, dtz)

        if distance_available > 0.1:
            a_needed = -(u**2) / (2.0 * distance_available)
            safe_a = float(max(EMERGENCY_DECEL, min(0.0, a_needed)))
        else:
            safe_a = float(EMERGENCY_DECEL)

        self._log_override(proposed_a, safe_a, constraint)

        # 5. Differentiate: Is this a "Safety" violation?
        # Safety violations = Aspect, Kinematic Target, or Track Bounds.
        # System Clamps = Segment Limits (e.g. S3_Limit).
        is_safety = True
        if "_Limit" in constraint or constraint == "Hardware_Limit_Clamp":
            is_safety = False
        
        return safe_a, is_safety

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
