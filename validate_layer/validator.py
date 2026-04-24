"""
validate_layer/validator.py
Anti-Lazy Update: Fixes the Zero-Authority bug at station boundaries.
"""

import math
import time
from typing import Tuple

try:
    from graph.graph import VLSegment, build_vl_segments
except ImportError:
    from graph import VLSegment, build_vl_segments

from validate_layer.log_manager import append_row, init_log

ACCEL: float = 0.5
SERVICE_DECEL: float = -0.5
EMERGENCY_DECEL: float = -1.0


class ValidationLayer:
    def __init__(self):
        self.segments: list[VLSegment] = build_vl_segments()
        init_log()
        self.dt = 1.0  # Synchronized with env.DT

    def get_segment(self, x: float) -> VLSegment:
        if x <= self.segments[0].start:
            return self.segments[0]
        if x >= self.segments[-1].end:
            return self.segments[-1]
        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg
        return self.segments[-1]

    def compute_dtz(self, x: float) -> float:
        seg = self.get_segment(x)
        for boundary in seg.block_boundaries:
            if boundary > x:
                return boundary - x
        return seg.spatial_headway

    def _speed_for_aspect(
        self, aspect: int, segment: VLSegment, dtz: float, current_v: float
    ) -> Tuple[float, float]:
        sh = segment.spatial_headway

        # NEW COMPLEXITY: Only grant the 1km Departure Authority if the signal is GREEN (3).
        # If the signal is Orange (1) or FlashGreen (2), the "Zero Distance" math stays
        # at 0.0, keeping the train locked.
        if current_v < 0.1 and aspect == 3:
            distance_available = dtz + 3 * sh  # The 'Green Departure Runway'
        else:
            # Normal authority calculation for all other states
            distance_available = dtz + (aspect * sh)

        v_dynamic = math.sqrt(2 * abs(EMERGENCY_DECEL) * max(0.0, distance_available))
        return min(v_dynamic, segment.limit_ms), distance_available

    def _project(
        self, x: float, u: float, proposed_a: float, v_ceil: float, t: float
    ) -> Tuple[float, float]:
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
            x_at_limit = x + u * t_to_limit + 0.5 * proposed_a * (t_to_limit**2)
            v = v_ceil if proposed_a > 0 else 0.0
            x_proj = x_at_limit + v * (t - t_to_limit)
        return x_proj, max(0.0, v)

    def _check_action_safety(
        self, proposed_a: float, env_aspect: int, x: float, u: float, dtz: float
    ) -> Tuple[bool, str]:
        current_seg = self.get_segment(x)
        max_safe_v, dist_avail = self._speed_for_aspect(env_aspect, current_seg, dtz, u)
        boundary_x = x + dist_avail

        t_stop = current_seg.limit_ms / abs(EMERGENCY_DECEL)
        sim_steps = [max(1, math.floor(t_stop * f)) for f in [0.33, 0.67, 1.0]]

        v_ceiling = current_seg.limit_ms
        for t in sim_steps:
            x_proj, v = self._project(x, u, proposed_a, v_ceiling, t)
            if x_proj >= boundary_x:
                return False, "Aspect_Spatial_Violation"
            if v > max_safe_v + 0.5:
                return False, "Kinematic_Target_Violation"
            if v > v_ceiling + 0.5:
                return False, f"{current_seg.id}_Limit"
        return True, ""

    def get_safe_action(
        self, proposed_a: float, env_aspect: int, x: float, u: float, dtz: float
    ) -> Tuple[float, bool]:
        """
        Validates AI action. Includes logic to differentiate system clamps from safety failures.
        """
        clamped_a = max(EMERGENCY_DECEL, min(ACCEL, proposed_a))

        # PROACTIVE SAFETY FIX: If stopped at Red and staying stopped, no override.
        if u < 0.1 and env_aspect == 0 and proposed_a <= 0:
            return 0.0, False

        # --- PHYSICAL GOVERNOR ---
        # If we are at or above the limit, cap accel at 0.0
        v_ceiling = seg.limit_ms
        if u >= v_ceiling - 0.01:
            clamped_a = min(0.0, clamped_a)

        is_safe, constraint = self._check_action_safety(
            clamped_a, env_aspect, x, u, dtz
        )

        if is_safe:
            return clamped_a, False

        # RESOLUTION LOGIC
        seg = self.get_segment(x)
        v_target, dist_avail = self._speed_for_aspect(env_aspect, seg, dtz, u)

        if "_Limit" in constraint:
            a_needed = (v_target - u - 0.01) / self.dt
            safe_a = float(max(EMERGENCY_DECEL, min(ACCEL, a_needed)))
        elif dist_avail > 0.1:
            a_needed = -(u**2) / (2.0 * dist_avail)
            safe_a = float(max(EMERGENCY_DECEL, min(0.0, a_needed)))
        else:
            safe_a = EMERGENCY_DECEL

        # Only set overridden=True if the AI was LESS safe than required.
        # Clamping for speed limits is a 'System Clamp' (False).
        was_safety_failure = True if proposed_a > safe_a + 0.01 else False
        if "_Limit" in constraint:
            was_safety_failure = False

        self._log_override(proposed_a, safe_a, constraint)
        return safe_a, was_safety_failure

    @staticmethod
    def _log_override(original: float, corrected: float, constraint_id: str) -> None:
        append_row(time.time(), original, corrected, constraint_id)
