"""
environment/railway_env.py

Gymnasium environment for a single train traversing Polish Rail Line 104.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.lead_train import LeadTrain
from environment.timetable import (
    STATION_NAMES,
    Timetable,
    compute_eta_to_station,
    generate_timetable,
)
from reward.reward_function import TrainState, compute_reward
from validate_layer.validator import ValidationLayer


class ModernizedLine104(gym.Env):
    metadata = {"render_modes": ["human"]}

    # --- Physics & Limits ---
    DT: float = 1.0
    ACCEL: float = 0.5
    DECEL: float = -1.0
    MAX_STEPS: int = 10_000
    TRACK_START: float = 582.0
    TRACK_END: float = 76651.0
    STATIONS: list[float] = [582, 5481, 14951, 37160, 47017, 67394, 76651]

    def __init__(
        self,
        lead_train_speed: float = 20.0,
        slack_factor: float = 1.1,
        render_mode: str | None = None,
        training_mode: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.lead_train_speed = lead_train_speed
        self.training_mode = training_mode
        self.slack_factor = slack_factor
        self.active_hazards: set[tuple[float, float]] = set()

        self.vl = ValidationLayer()
        self.lead_train = LeadTrain(stations=self.STATIONS)  # <-- INITIALIZE LEAD TRAIN

        # Spaces: We use a symmetric [-1, 1] space for the agent.
        # Inside step(), we map [-1, 0] -> [DECEL, 0] and [0, 1] -> [0, ACCEL]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array(
                [80000.0, 30.0, 80000.0, 3.0, 80000.0, 30.0, 10000.0], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # State variables
        self.x: float = self.TRACK_START
        self.v: float = 0.0
        self.dtz: float = 0.0
        self.time: float = 0.0
        self.last_station_idx: int = 0
        self.visited_stations: set[int] = set()
        self.last_a: float = 0.0
        self.step_count: int = 0

        self.step_count = 0

        self.timetable: Timetable = generate_timetable(
            self.vl.segments,
            self.STATIONS,
            dwell_seconds=float(LeadTrain.LEAD_DWELL_RANGE[1]),
            slack_factor=self.slack_factor,
        )
        self.ai_arrival_times: dict[int, float] = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.TRACK_START
        self.v = 0.0
        self.time = 0.0
        self.last_station_idx = 0
        self.visited_stations = {0}
        self.last_a = 0.0
        self.step_count = 0
        self.ai_arrival_times = {}
        self.ai_departure_time = 0.0  # Initialize departure time

        # Reset the lead train ~2km ahead
        start_lead_x = self.TRACK_START + 2000.0
        start_lead_v = self.vl.get_segment(start_lead_x).limit_ms
        self.lead_train.reset(
            start_x=start_lead_x, start_v=start_lead_v, np_random=self.np_random
        )

        self._update_dtz()
        return self._get_obs(), {
            "overridden": False,
            "timetable": self.timetable.to_dict(),
        }

    def step(self, action: np.ndarray):
        # Action Remapping: Map symmetric [-1, 1] to physical [DECEL, ACCEL]
        # This ensures that an initial random action of 0.0 maps to 0.0 (coasting).
        raw_action = float(action[0])
        if raw_action >= 0:
            proposed_a = raw_action * self.ACCEL
        else:
            proposed_a = (
                abs(raw_action) * self.DECEL
            )  # DECEL is negative, so we use abs()

        self._cached_nearest_pos = self._nearest_obstruction(self.x)
        self._cached_dist_to_occupied = self._dist_to_nearest_occupied_from_cache()
        env_aspect = self._get_signal_aspect_from_cache()
        self._cached_aspect = env_aspect


        # --- CLEAN INTERLOCK ---
        # Any physical interlocks have been removed to give full throttle control.
        # We rely on rewards and the Validation Layer for safety.

        # --- VALIDATION LAYER ---
        safe_a, safety_overridden = self.vl.get_safe_action(
            proposed_a, env_aspect, self.x, self.v, self.dtz, self._cached_dist_to_occupied
        )

        action_delta = abs(safe_a - self.last_a)
        self.last_a = safe_a

        # --- AI PHYSICS ---
        prev_x, prev_v = self.x, self.v
        seg = self.vl.get_segment(self.x)
        limit_v = seg.limit_ms

        if safe_a == 0.0:
            self.v = min(prev_v, limit_v)
            dx = self.v * self.DT
        elif safe_a > 0.0:
            if prev_v >= limit_v:
                self.v, dx = limit_v, limit_v * self.DT
            else:
                t_to_limit = (limit_v - prev_v) / safe_a
                if self.DT <= t_to_limit:
                    self.v = prev_v + safe_a * self.DT
                    dx = prev_v * self.DT + 0.5 * safe_a * self.DT**2
                else:
                    dx = (
                        prev_v * t_to_limit
                        + 0.5 * safe_a * t_to_limit**2
                        + limit_v * (self.DT - t_to_limit)
                    )
                    self.v = limit_v
        else:
            if prev_v <= 0.0:
                self.v, dx = 0.0, 0.0
            else:
                t_to_zero = prev_v / abs(safe_a)
                if self.DT <= t_to_zero:
                    self.v = prev_v + safe_a * self.DT
                    dx = prev_v * self.DT + 0.5 * safe_a * self.DT**2
                else:
                    dx = prev_v * t_to_zero + 0.5 * safe_a * t_to_zero**2
                    self.v = 0.0

        self.v = max(0.0, self.v)
        self.x = min(prev_x + max(0.0, dx), self.TRACK_END)
        applied_traction = max(0.0, safe_a)
        self.time += self.DT
        self.step_count += 1

        # --- LEAD TRAIN UPDATE ---
        self.lead_train.advance(
            self.DT, self.vl, self.active_hazards, self.training_mode
        )
        self._update_dtz()

        # --- STATION ARRIVAL ---
        reached_new_station = False
        next_st_idx = self.last_station_idx + 1
        scheduled_arrival_time = None
        actual_arrival_time = None

        if next_st_idx < len(self.STATIONS) and self.x >= self.STATIONS[next_st_idx]:
            reached_new_station = True
            self.last_station_idx = next_st_idx
            self.visited_stations.add(next_st_idx)
            self.x = self.STATIONS[next_st_idx]
            self.v = 0.0

            actual_arrival_time = self.time
            entry = self.timetable.get_entry(next_st_idx)
            scheduled_arrival_time = (
                entry.scheduled_arrival if entry else self._ideal_schedule[next_st_idx]
            )
            self.ai_arrival_times[next_st_idx] = self.time

            # NEW DWELL LOGIC: Random dwell (matching lead train behavior 20-40s)
            if next_st_idx < len(self.STATIONS) - 1 and entry:
                self.ai_departure_time = self.time + self.np_random.integers(20, 40)

        # --- REWARDS ---
        last_st_pos = self.STATIONS[self.last_station_idx]
        next_idx = min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        next_st_pos = self.STATIONS[next_idx]

        # Calculate the next scheduled arrival time
        next_entry = self.timetable.get_entry(next_idx)
        # Use the station entry for the scheduled time
        next_scheduled = next_entry.scheduled_arrival if next_entry else 0.0

        in_dwell = (
            hasattr(self, "ai_departure_time") and self.time < self.ai_departure_time
        )

        state = TrainState(
            current_position=self.x,
            previous_position=prev_x,
            last_station_position=last_st_pos,
            next_station_position=next_st_pos,
            current_speed=self.v,
            speed_limit=seg.limit_ms,
            headway=self._compute_headway(),
            temporal_headway=seg.temporal_headway,
            spatial_headway=seg.spatial_headway,
            reached_new_station=reached_new_station,
            scheduled_arrival_time=scheduled_arrival_time,
            actual_arrival_time=actual_arrival_time,
            collision=(self.x >= self.lead_train.x),
            safety_overridden=safety_overridden,
            action_delta=action_delta,
            applied_traction=applied_traction,
            distance_to_occupied=self._dist_to_nearest_occupied(),
            is_dwelling=in_dwell,
            station_index=self.last_station_idx,
            signal_aspect=self._get_signal_aspect(),
            applied_acceleration=safe_a,
            proposed_acceleration=proposed_a,
            current_time=self.time,
            next_scheduled_arrival_time=next_scheduled,
        )

        reward_out = compute_reward(state)
        terminated = bool(self.x >= self.TRACK_END or reward_out.terminate)
        truncated = bool(self.step_count >= self.MAX_STEPS)

        # Clear cache before returning observation to ensure next state gets fresh data
        if hasattr(self, "_cached_aspect"): del self._cached_aspect
        if hasattr(self, "_cached_dist_to_occupied"): del self._cached_dist_to_occupied
        if hasattr(self, "_cached_nearest_pos"): del self._cached_nearest_pos

        info = {
            "safety_overridden": safety_overridden,
            "safe_a": safe_a,
            "proposed_a": proposed_a,
            "aspect": env_aspect,
            "time": self.time,
            "segment": seg.id,
            "reward_breakdown": {
                "progress": reward_out.r_progress,
                "headway": reward_out.r_headway,
                "speed": reward_out.r_speed,
                "heartbeat": reward_out.r_heartbeat,
                "signal_compliance": reward_out.r_signal_compliance,
                "station": reward_out.r_station,
                "punctuality": reward_out.r_time,
                "override": reward_out.r_override,
                "jerk": reward_out.r_jerk,
                "energy": reward_out.r_energy,
                "patience": reward_out.r_patience,
                "creep": reward_out.r_creep,
                "violation": reward_out.r_violation,
                "collision": reward_out.r_collision,
            },
            "punctuality_status": self.get_punctuality_status(),
        }

        return self._get_obs(), reward_out.r_total, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        return np.array(
            [
                self.x,
                self.v,
                self.dtz,
                float(
                    self._cached_aspect
                    if hasattr(self, "_cached_aspect")
                    else self._get_signal_aspect()
                ),
                self._cached_dist_to_occupied
                if hasattr(self, "_cached_dist_to_occupied")
                else self._dist_to_nearest_occupied(),
                self.lead_train.v,
                self.time,
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode == "human":
            seg = self.vl.get_segment(self.x)
            print(
                f"t={self.time:7.1f}s | x={self.x:8.1f}m | v={self.v:5.2f}m/s | seg={seg.id} | dtz={self.dtz:8.1f}m | lead={self.lead_train.x:8.1f}m | sig={self._get_signal_aspect()}"
            )

    @property
    def lead_x(self) -> float:
        return self.lead_train.x

    @property
    def lead_v(self) -> float:
        return self.lead_train.v

    @property
    def lead_dwell_timer(self) -> int:
        return self.lead_train.dwell_timer

    @property
    def lead_stalled(self) -> bool:
        return self.lead_train.stalled

    @property
    def lead_held(self) -> bool:
        return self.lead_train.held

    @property
    def ai_dwell_timer(self) -> int:
        """Remaining dwell time in seconds."""
        if not hasattr(self, "ai_departure_time"):
            return 0
        return max(0, int(self.ai_departure_time - self.time))

    # --- API HELPER METHODS ---
    def _update_dtz(self):
        self.dtz = self.vl.compute_dtz(self.x)

    def _nearest_obstruction(self, from_x: float) -> float:
        # We removed the 'end of segment is barrier' logic as it causes deadlocks.
        # Barriers are now only actual trains and hazards.
        nearest = self.TRACK_END

        # Also check for lead train
        nearest = min(nearest, self.lead_train.x)

        # Also check for active hazards
        for h_start, _ in self.active_hazards:
            if h_start > from_x:
                nearest = min(nearest, h_start)
        return nearest

    def _dist_to_nearest_occupied(self) -> float:
        nearest = self._nearest_obstruction(self.x)
        # Only treat next station as an obstruction if we are significantly before it
        # This prevents the "Deadlock at 0m" issue.
        if self.last_station_idx + 1 < len(self.STATIONS):
            next_st_pos = self.STATIONS[self.last_station_idx + 1]
            if next_st_pos > self.x + 1.0:
                nearest = min(nearest, next_st_pos)
        return max(0.0, nearest - self.x)

    def _get_signal_aspect(self) -> int:
        # Use the distance that includes stations
        d = self._dist_to_nearest_occupied()
        sh = self.vl.get_segment(self.x).spatial_headway
        if d > 3 * sh:
            return 3
        if d > 2 * sh:
            return 2
        if d > sh:
            return 1
        return 0

    def _get_signal_aspect_from_cache(self) -> int:
        d = self._cached_nearest_pos - self.x
        sh = self.vl.get_segment(self.x).spatial_headway
        if d > 3 * sh:
            return 3
        if d > 2 * sh:
            return 2
        if d > sh:
            return 1
        return 0

    def _dist_to_nearest_occupied_from_cache(self) -> float:
        nearest = self._cached_nearest_pos
        if self.last_station_idx + 1 < len(self.STATIONS):
            next_st_pos = self.STATIONS[self.last_station_idx + 1]
            if next_st_pos > self.x + 1.0:
                nearest = min(nearest, next_st_pos)
        return max(0.0, nearest - self.x)

    def _compute_headway(self) -> float:
        return (self.lead_train.x - self.x) / self.v if self.v > 0.01 else 9999.0


    # --- EXTERNAL COMMANDS ---
    def stall_lead(self):
        self.lead_train.stalled = True

    def release_lead(self):
        self.lead_train.stalled = self.lead_train.held = False

    def hold_lead(self):
        self.lead_train.held = True

    def set_block_hazard(self, start: float, end: float, active: bool):
        self.active_hazards.add(
            (start, end)
        ) if active else self.active_hazards.discard((start, end))

    def clear_all_hazards(self):
        self.active_hazards.clear()

    # --- PUNCTUALITY ---
    def _train_punctuality(
        self, position: float, last_visited_idx: int, arrival_log: dict[int, float]
    ) -> dict:
        next_idx = last_visited_idx + 1
        if next_idx >= len(self.STATIONS):
            return {
                "next_station_idx": None,
                "status": "arrived",
                "arrival_log": arrival_log,
            }
        entry = self.timetable.get_entry(next_idx)
        if entry is None:
            return {"status": "unknown"}
        eta = self.time + compute_eta_to_station(
            position, entry.position_m, self.vl.segments
        )
        slack = entry.scheduled_arrival - eta
        status = "on_time" if abs(slack) <= 60.0 else "early" if slack > 0 else "late"
        return {
            "next_station_idx": next_idx,
            "next_station_name": entry.station_name,
            "eta": round(eta, 1),
            "status": status,
        }

    def get_punctuality_status(self) -> dict:
        return {
            "sim_time": round(self.time, 1),
            "ai": self._train_punctuality(
                self.x, self.last_station_idx, self.ai_arrival_times
            ),
        }
