"""
validate_layer/validator.py
Safety-first validation layer for Polish Rail Line 104.
"""

import math
import time
from typing import List, Tuple

try:
    from graph.graph import VLSegment, build_vl_segments
except ImportError:
    from graph import VLSegment, build_vl_segments

from validate_layer.log_manager import append_row, init_log

# Physical Constants
ACCEL: float = 0.5
SERVICE_DECEL: float = -0.5
EMERGENCY_DECEL: float = -1.0


class ValidationLayer:
    def __init__(self):
        self.segments: List[VLSegment] = build_vl_segments()
        init_log()
        self.dt: float = 1.0  # Synchronized with env.DT

    def get_segment(self, x: float) -> VLSegment:
        """Find the track segment containing position x."""
        if x <= self.segments[0].start:
            return self.segments[0]
        if x >= self.segments[-1].end:
            return self.segments[-1]
        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg
        return self.segments[-1]

    def compute_dtz(self, x: float) -> float:
        """Compute Distance To Zone (distance to the next block boundary)."""
        seg = self.get_segment(x)
        for boundary in seg.block_boundaries:
            if boundary > x + 0.01:  # Epsilon to prevent 0.0 at boundaries
                return boundary - x
        return float(seg.spatial_headway)

    def _speed_for_aspect(
        self, aspect: int, segment: VLSegment, dtz: float, current_v: float
    ) -> Tuple[float, float]:
        """
        Calculates maximum allowed speed and total authority distance for a signal aspect.
        """
        sh = float(segment.spatial_headway)

        # Grant 'Green Departure Runway' (3 blocks) if stationary at Green
        if current_v < 0.1 and aspect == 3:
            distance_available = dtz + (3.0 * sh)
        else:
            distance_available = dtz + (float(aspect) * sh)

        # Basic kinematic safety limit: v = sqrt(2 * a * s)
        v_dynamic = math.sqrt(2.0 * abs(EMERGENCY_DECEL) * max(0.0, distance_available))

        # Target speed is the lower of the kinematic safety and segment limit
        v_target = min(v_dynamic, float(segment.limit_ms))

        return v_target, distance_available

    def _project(
        self, x: float, u: float, proposed_a: float, v_ceil: float, t: float
    ) -> Tuple[float, float]:
        """Predicts position and speed after t seconds using SUVAT."""
        if proposed_a == 0:
            return x + u * t, u

        if proposed_a > 0:
            if u >= v_ceil:
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
            # Reached speed limit or stop mid-interval
            x_at_limit = x + u * t_to_limit + 0.5 * proposed_a * (t_to_limit**2)
            v = v_ceil if proposed_a > 0 else 0.0
            x_proj = x_at_limit + v * (t - t_to_limit)

        return x_proj, max(0.0, v)

    def _check_action_safety(
        self, proposed_a: float, env_aspect: int, x: float, u: float, dtz: float
    ) -> Tuple[bool, str]:
        """Simulates trajectory to verify if an action violates safety constraints."""
        current_seg = self.get_segment(x)
        max_safe_v, dist_avail = self._speed_for_aspect(env_aspect, current_seg, dtz, u)
        boundary_x = x + dist_avail

        # Look ahead based on stopping time
        t_stop = current_seg.limit_ms / abs(EMERGENCY_DECEL)
        sim_steps = [max(1.0, math.floor(t_stop * f)) for f in [0.33, 0.67, 1.0]]

        v_ceiling = float(current_seg.limit_ms)
        for t in sim_steps:
            x_proj, v = self._project(x, u, proposed_a, v_ceiling, t)
            proj_seg = self.get_segment(x_proj)

            # Constraint 1: Spatial limit (Aspect Authority)
            # TOLERANCE: If speed is very low (< 1 m/s) and we are close to the boundary (< 2m), 
            # allow a small overshoot to prevent nuisance overrides during station stops.
            tolerance = 2.0 if (u < 1.0 and dist_avail < 2.0) else 0.0
            if x_proj >= boundary_x + tolerance:
                return False, "Aspect_Spatial_Violation"

            # Constraint 2: Speed limit of the current or future segment
            if v > proj_seg.limit_ms + 0.5:
                return False, f"{proj_seg.id}_Limit"

            # Constraint 3: Kinematic safety for the current aspect
            if v > max_safe_v + 0.5:
                return False, "Kinematic_Target_Violation"

        return True, ""

    def get_safe_action(
        self, proposed_a: float, env_aspect: int, x: float, u: float, dtz: float
    ) -> Tuple[float, bool]:
        """
        Main entry point. Validates action and returns a safe alternative if needed.
        """
        seg = self.get_segment(x)

        # 1. Hardware/Software Clamping
        clamped_a = max(EMERGENCY_DECEL, min(ACCEL, proposed_a))

        # 2. Stationary interlock
        if u < 0.1 and env_aspect == 0 and proposed_a <= 0:
            return 0.0, False

        # 3. Speed Limit Governor
        if u >= seg.limit_ms - 0.01:
            clamped_a = min(0.0, clamped_a)

        # 4. Trajectory Safety Check
        is_safe, constraint = self._check_action_safety(
            clamped_a, env_aspect, x, u, dtz
        )

        if is_safe:
            return clamped_a, False

        # 5. RESOLUTION (If unsafe, find the best possible safe action)
        v_target, dist_avail = self._speed_for_aspect(env_aspect, seg, dtz, u)

        if "_Limit" in constraint:
            # Resolve speed limit violations
            seg_id = constraint.split("_")[0]
            target_limit = float(seg.limit_ms)
            for s in self.segments:
                if str(s.id) == str(seg_id):
                    target_limit = float(s.limit_ms)
                    break
            a_needed = (target_limit - u - 0.01) / self.dt
            safe_a = max(EMERGENCY_DECEL, min(ACCEL, a_needed))
        elif dist_avail > 0.1:
            # Resolve spatial violations using SUVAT: v² = u² + 2as => a = -u² / 2s
            a_needed = -(u**2) / (2.0 * dist_avail)
            safe_a = max(EMERGENCY_DECEL, min(0.0, a_needed))
        else:
            # Critical violation
            safe_a = EMERGENCY_DECEL

        # Only mark as 'overridden' if the AI's intent was less safe than the correction
        was_safety_failure = True if proposed_a > safe_a + 0.01 else False
        if "_Limit" in constraint:
            was_safety_failure = False  # System-level speed limit clamp

        self._log_override(proposed_a, safe_a, constraint)
        return float(safe_a), was_safety_failure

    @staticmethod
    def _log_override(original: float, corrected: float, constraint_id: str) -> None:
        append_row(time.time(), original, corrected, constraint_id)
