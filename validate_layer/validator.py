"""
Draft: 10 (Anti-Lazy Update)

validate_layer/validator.py

This module implements a deterministic, safety-critical Validation Layer (VL)
for a PPO-driven autonomous train agent. 
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
# Safety constants
# ---------------------------------------------------------------------------

ACCEL: float = 0.5  
SERVICE_DECEL: float = -0.5  
EMERGENCY_DECEL: float = -1.0  

# ---------------------------------------------------------------------------
# Validation Layer
# ---------------------------------------------------------------------------

class ValidationLayer:
    """
    4-layer deterministic safety pipeline updated to prevent "Lazy Agent" traps.
    """

    def __init__(self):
        self.segments: list[VLSegment] = build_vl_segments()
        init_log()

        # FIXED: Synchronized with ModernizedLine104.DT = 1.0
        # Prevents the "10x jerk" bug where VL corrections were too aggressive.
        self.dt = 1.0 

    def get_segment(self, x: float) -> VLSegment:
        if x <= self.segments[0].start:
            return self.segments[0]
        if x >= self.segments[-1].end:
            return self.segments[-1]
        for seg in self.segments:
            if seg.start <= x < seg.end:
                return seg
        raise ValueError(f"Position x={x} unresolvable.")

    def compute_dtz(self, x: float) -> float:
        seg = self.get_segment(x)
        for boundary in seg.block_boundaries:
            if boundary > x:
                return boundary - x
        seg_idx = self.segments.index(seg)
        if seg_idx + 1 < len(self.segments):
            return self.segments[seg_idx + 1].start - x
        return seg.spatial_headway

    @staticmethod
    def _speed_for_aspect(aspect: int, segment: VLSegment, dtz: float) -> Tuple[float, float]:
        # Maps signal to distance: 0=Red(DTZ), 1=Orange(DTZ+1SH), etc.
        distance_available = dtz + (aspect * segment.spatial_headway)
        
        # Uses EMERGENCY_DECEL to find the physical limit.
        a_brake = abs(EMERGENCY_DECEL)
        v_dynamic = math.sqrt(2 * a_brake * max(0.0, distance_available))
        
        return min(v_dynamic, segment.limit_ms), distance_available

    @staticmethod
    def _project(x: float, u: float, proposed_a: float, v_ceil: float, t: float) -> Tuple[float, float]:
        if proposed_a == 0:
            return x + u * t, u
        if proposed_a > 0:
            if u >= v_ceil: return x + v_ceil * t, v_ceil 
            t_to_limit = (v_ceil - u) / proposed_a
        else:
            if u <= 0.0: return x, 0.0
            t_to_limit = (0.0 - u) / proposed_a

        if t <= t_to_limit:
            v = u + proposed_a * t
            x_proj = x + u * t + 0.5 * proposed_a * (t**2)
        else:
            x_at_limit = x + u * t_to_limit + 0.5 * proposed_a * (t_to_limit**2)
            v = v_ceil if proposed_a > 0 else 0.0
            x_proj = x_at_limit + v * (t - t_to_limit)
        return x_proj, max(0.0, v)

    def _check_action_safety(self, proposed_a: float, env_aspect: int, x: float, u: float, dtz: float) -> Tuple[bool, str]:
        current_seg = self.get_segment(x)
        max_safe_v, distance_available = self._speed_for_aspect(env_aspect, current_seg, dtz)
        boundary_x = x + distance_available
        v_ceiling = current_seg.limit_ms
        t_stop = current_seg.limit_ms / abs(EMERGENCY_DECEL)
        
        sim_steps = [max(1, math.floor(t_stop * f)) for f in [0.33, 0.67, 1.0]]

        for t in sim_steps:
            x_proj, v = self._project(x, u, proposed_a, v_ceiling, t)
            try:
                proj_seg = self.get_segment(x_proj)
                v_ceiling = min(v_ceiling, proj_seg.limit_ms)
            except ValueError:
                return False, "Track_Bounds_Violation"

            if x_proj >= boundary_x: return False, "Aspect_Spatial_Violation"
            if v > max_safe_v + 0.01: return False, "Kinematic_Target_Violation"
            if v > proj_seg.limit_ms + 0.01: return False, f"{proj_seg.id}_Limit"

        return True, ""

    def get_safe_action(self, proposed_a: float, env_aspect: int, x: float, u: float, dtz: float) -> Tuple[float, bool]:
        """
        Layer 3 — Continuous Decision Node.
        """
        # 1. Hardware clamp
        clamped_a = max(EMERGENCY_DECEL, min(ACCEL, proposed_a))
        
        # 2. FIXED: Red Signal Stop Logic
        # If stopped at Red and AI stays stopped (proposed_a <= 0), it's safe and NO override is flagged.
        # This prevents the AI from being punished for obeying the signal.
        if u < 0.1 and env_aspect == 0 and proposed_a <= 0.0:
            return 0.0, False

        # 3. Safety check
        is_safe, constraint = self._check_action_safety(clamped_a, env_aspect, x, u, dtz)

        if is_safe:
            # Clamping to max hardware limits is NOT a punishable safety override.
            return clamped_a, False

        # 4. Violation Resolution
        seg = self.get_segment(x)
        v_safe_limit, distance_available = self._speed_for_aspect(env_aspect, seg, dtz)

        if "_Limit" in constraint:
            # Syncing with self.dt ensures smooth correction
            a_needed = (v_safe_limit - u - 0.001) / self.dt
            safe_a = float(max(EMERGENCY_DECEL, min(ACCEL, a_needed)))
        elif distance_available > 0.1:
            a_needed = -(u**2) / (2.0 * distance_available)
            safe_a = float(max(EMERGENCY_DECEL, min(0.0, a_needed)))
        else:
            safe_a = float(EMERGENCY_DECEL)

        self._log_override(proposed_a, safe_a, constraint)

        # 5. FIXED: Anti-Lazy Override Flag
        # Only flag a safety violation if the AI was LESS safe than required.
        # If the AI proposed braking (e.g., -0.8) and VL wanted -0.5, was_safety = False.
        was_safety = True if proposed_a > safe_a + 0.01 else False
        
        # Speed limit violations remain System Clamps (False).
        if "_Limit" in constraint:
            was_safety = False
        
        return safe_a, was_safety

    @staticmethod
    def _log_override(original: float, corrected: float, constraint_id: str) -> None:
        append_row(time.time(), original, corrected, constraint_id)