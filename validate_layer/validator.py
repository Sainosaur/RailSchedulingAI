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
    def __init__(self, vl_active: bool = True): # toggle to activate VL in simulation. True is on, False if off.
        self.vl_active = vl_active
        self.segments: List[VLSegment] = build_vl_segments()
        if self.vl_active:
            init_log()
        self.dt: float = 1.0

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

    def compute_zone_boundaries(self, x: float, seg: VLSegment) -> Tuple[float, float]:
        """Return (x_ai_zone_end, zone_idx) for position x within seg.

        The last zone of each segment is oversized (>= SH) because the safety merge
        absorbs any short fractional remainder into it. Must detect and handle this.
        n_full = number of full SH-length zones before the oversized last zone.
        Any x past start + n_full*SH belongs to the last zone, which ends at seg.end.
        """
        sh = seg.spatial_headway
        n_full = int((seg.end - seg.start) / sh)
        zone_idx = int((x - seg.start) / sh)
        zone_idx = min(zone_idx, n_full)
        x_zone_end = seg.start + (zone_idx + 1) * sh
        if x_zone_end > seg.end:
            x_zone_end = seg.end
        return x_zone_end, zone_idx

    def compute_dtz(self, x: float, seg: VLSegment) -> float:
        """Distance from x to the end of the current zone."""
        x_ai_zone_end, _ = self.compute_zone_boundaries(x, seg)
        return max(0.0, x_ai_zone_end - x)



    def check_and_log(
        self,
        x: float,
        u: float,
        proposed_a: float,
        seg: VLSegment,
        x_obs_zone_start: float,
        train_aspect: int,
        station_cleared: bool
    ) -> dict:
        """
        Check all constraints. Log any violations. Return dict of violation flags.
        Does NOT override the action. Returns results for env/reward to use.

        Returns dict with keys:
          spatial_violation: bool
          speed_limit_violation: bool
          station_signal_violation: bool
          lead_train_signal_violation: bool
          negative_speed_violation: bool
          accel_not_zero_violation: bool
          any_violation: bool
        """
        violations = {
            "spatial_violation": False,
            "speed_limit_violation": False,
            "station_signal_violation": False,
            "lead_train_signal_violation": False,
            "negative_speed_violation": False,
            "accel_not_zero_violation": False,
            "any_violation": False,
        }

        # When VL is inactive, skip all checks — return clean violations dict
        if not self.vl_active:
            return violations

        sh = seg.spatial_headway
        x_ai_zone_end, _ = self.compute_zone_boundaries(x, seg)

        # 1. Negative speed check
        projected_v = u + proposed_a * self.dt
        if projected_v < 0.0:
            violations["negative_speed_violation"] = True
            self._log(x_ai_zone_end, x_obs_zone_start, u, "NEGATIVE_SPEED_VIOLATION")

        # 2. Speed limit check
        if u > seg.limit_ms + 0.01:
            violations["speed_limit_violation"] = True
            self._log(x_ai_zone_end, x_obs_zone_start, u, "SPEED_LIMIT_VIOLATION")

        # 3. Accel not zero at stationary or at speed limit
        if (u < 0.1 and proposed_a < -0.01) or \
           (abs(u - seg.limit_ms) < 0.01 and proposed_a > 0.01):
            violations["accel_not_zero_violation"] = True
            self._log(x_ai_zone_end, x_obs_zone_start, u, "ACCEL_NOT_ZERO_VIOLATION")

        # 4. Spatial violation (SUVAT stopping distance check)
        # dist_vio_check = u^2 / (2 * |EMERGENCY_DECEL|)
        dist_vio_check = (u * u) / (2.0 * abs(EMERGENCY_DECEL))
        if x_ai_zone_end + dist_vio_check >= x_obs_zone_start:
            violations["spatial_violation"] = True
            self._log(x_ai_zone_end, x_obs_zone_start, u, "SPATIAL_VIOLATION")

        # 5. Station signal violation
        # If the obstacle IS the station and dwell not complete, and train tries to pass
        if not station_cleared and x >= x_obs_zone_start:
            violations["station_signal_violation"] = True
            self._log(x_ai_zone_end, x_obs_zone_start, u, "STATION_SIGNAL_VIOLATION")

        # 6. Lead train signal violation
        # If aspect is Red and train is accelerating
        if train_aspect == 0 and proposed_a > 0.01:
            violations["lead_train_signal_violation"] = True
            self._log(x_ai_zone_end, x_obs_zone_start, u, "LEAD_TRAIN_SIGNAL_VIOLATION")

        violations["any_violation"] = any([
            violations["spatial_violation"],
            violations["speed_limit_violation"],
            violations["station_signal_violation"],
            violations["lead_train_signal_violation"],
            violations["negative_speed_violation"],
            violations["accel_not_zero_violation"],
        ])

        return violations

    def _log(self, x_ai_zone_end: float, x_obs_zone_start: float, speed: float, constraint_id: str):
        append_row(time.time(), x_ai_zone_end, x_obs_zone_start, speed, constraint_id)
