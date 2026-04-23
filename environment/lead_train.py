"""
environment/lead_train.py

Manages the physics, state, and schedule of the autonomous lead train.
Acts as a completely separate agent from the Gymnasium environment.
"""

import math

class LeadTrain:
    LEAD_DWELL_RANGE: tuple[int, int] = (15, 60)
    LEAD_SERVICE_DECEL: float = -0.5
    ACCEL: float = 0.5
    DECEL: float = -1.0

    def __init__(self, stations: list[float]):
        self.stations = stations
        self.x: float = 0.0
        self.v: float = 0.0
        self.dwell_timer: int = 0
        self.station_idx: int = 1
        self.stalled: bool = False
        self.held: bool = False
        self.random_stall_timer: int = 0
        self.np_random = None  # Injected on reset

    def reset(self, start_x: float, start_v: float, np_random):
        """Reset the lead train to its starting state."""
        self.x = start_x
        self.v = start_v
        self.dwell_timer = 0
        self.station_idx = 1
        self.stalled = False
        self.held = False
        self.random_stall_timer = 0
        self.np_random = np_random

    def advance(self, dt: float, vl, active_hazards: set, training_mode: bool):
        """Advance the lead train one timestep with realistic physics."""
        # --- Random Stalls for Training Mode ---
        if training_mode and self.dwell_timer == 0 and not self.stalled and not self.held:
            if self.np_random.random() < 0.001:
                self.random_stall_timer = self.np_random.integers(15, 60)

        # --- Stall override: emergency brake to stop ---
        if self.stalled or self.random_stall_timer > 0:
            if self.random_stall_timer > 0:
                self.random_stall_timer -= 1
            if self.v > 0:
                self.v = max(0.0, self.v + self.DECEL * dt)
                self.x += self.v * dt
            return

        # --- Dwelling at station ---
        if self.dwell_timer > 0:
            if not self.held:
                self.dwell_timer -= 1
            if self.dwell_timer == 0 and not self.held:
                if self.station_idx < len(self.stations) - 1:
                    self.station_idx += 1
                else:
                    self.x = self.stations[-1] + 34.7
                    self.v = 0.0
                    self.station_idx = len(self.stations)
            return

        if self.station_idx >= len(self.stations):
            return

        # --- Moving: determine speed ceiling from constraints ---
        seg = vl.get_segment(self.x)
        seg_limit = seg.limit_ms
        brake_a = abs(self.LEAD_SERVICE_DECEL)

        d_station = max(0.01, self.stations[self.station_idx] - self.x)
        v_ceil_station = math.sqrt(2 * brake_a * d_station)

        v_ceil_seg = float('inf')
        seg_idx = vl.segments.index(seg)
        if seg_idx + 1 < len(vl.segments):
            next_seg = vl.segments[seg_idx + 1]
            if next_seg.limit_ms < seg_limit:
                d_boundary = max(0.01, seg.end - self.x)
                v_ceil_seg = math.sqrt(next_seg.limit_ms ** 2 + 2 * brake_a * d_boundary)

        v_ceil_hazard = float('inf')
        for h_start, _h_end in active_hazards:
            if h_start >= self.x:
                d_hazard = h_start - self.x
                v_ceil_hazard = min(v_ceil_hazard, math.sqrt(2 * brake_a * max(0.0, d_hazard)))

        v_ceil = min(seg_limit, v_ceil_station, v_ceil_seg, v_ceil_hazard)

        # --- Choose acceleration ---
        if self.v > v_ceil + 0.01:
            a = self.LEAD_SERVICE_DECEL
        elif self.v < v_ceil - 0.01:
            a = self.ACCEL
        else:
            a = 0.0

        # --- Two-phase SUVAT position update ---
        prev_v = self.v
        if a == 0.0:
            self.v = min(prev_v, v_ceil)
            dx = self.v * dt
        elif a > 0:
            if prev_v >= v_ceil:
                self.v = v_ceil
                dx = v_ceil * dt
            else:
                t_to_ceil = (v_ceil - prev_v) / a
                if dt <= t_to_ceil:
                    self.v = prev_v + a * dt
                    dx = prev_v * dt + 0.5 * a * dt ** 2
                else:
                    dx = (prev_v * t_to_ceil + 0.5 * a * t_to_ceil ** 2 + v_ceil * (dt - t_to_ceil))
                    self.v = v_ceil
        else:
            if prev_v <= v_ceil:
                self.v = max(0.0, v_ceil)
                dx = self.v * dt
            else:
                t_to_ceil = (prev_v - v_ceil) / abs(a)
                if dt <= t_to_ceil:
                    self.v = prev_v + a * dt
                    dx = prev_v * dt + 0.5 * a * dt ** 2
                else:
                    dx = (prev_v * t_to_ceil + 0.5 * a * t_to_ceil ** 2 + v_ceil * (dt - t_to_ceil))
                    self.v = v_ceil

        self.v = max(0.0, self.v)
        prev_lead_x = self.x
        self.x += max(0.0, dx)

        # --- Snaps ---
        for h_start, _h_end in active_hazards:
            if prev_lead_x < h_start <= self.x:
                self.x = h_start
                self.v = 0.0
                return

        if self.station_idx < len(self.stations):
            next_station = self.stations[self.station_idx]
            if self.x >= next_station:
                self.x = next_station
                self.v = 0.0
                self.dwell_timer = self.np_random.integers(
                    self.LEAD_DWELL_RANGE[0], self.LEAD_DWELL_RANGE[1]
                )