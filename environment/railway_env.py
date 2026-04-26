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
    STATIONS: list[float] = [582.0, 5481.0, 14951.0, 37160.0, 47017.0, 67394.0, 76651.0]
    TRACK_START: float = 582.0
    TRACK_END: float = 76651.0

    def __init__(
        self,
        lead_train_speed: float = 20.0,
        lead_stop_offset: float = 94.7,
        slack_factor: float = 1.1,
        render_mode: str | None = None,
        training_mode: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.lead_train_speed = lead_train_speed
        self.lead_stop_offset = lead_stop_offset
        self.training_mode = training_mode
        self.slack_factor = slack_factor
        self.active_hazards: set[tuple[float, float]] = set()

        self.vl = ValidationLayer()
        self.lead_train = LeadTrain(
            stations=self.STATIONS,
            stop_offset=self.lead_stop_offset,
        )  # <-- INITIALIZE LEAD TRAIN

        # Action space: [-1.0, 0.5] directly maps to [full emergency brake, full traction].
        # Upper bound is 0.5 m/s² (max traction); lower bound is -1.0 m/s² (emergency brake).
        self.action_space = gym.spaces.Box(
            low=-1.0, high=0.5, shape=(1,), dtype=np.float32
        )
        # New observation space: 9 dimensions
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array(
                [80000.0, 30.0, 1000.0, 3.0, 3.0, 80000.0, 2000.0, 10000.0, 6.0], dtype=np.float32
            ),
            dtype=np.float32,
        )

        # State variables
        self.x: float = self.TRACK_START
        self.v: float = 0.0
        self.dtz: float = 0.0
        self.time: float = 0.0
        self.last_station_idx: int = 0
        self.next_station_num: int = 1
        self.visited_stations: set[int] = set()
        self.last_a: float = 0.0
        self.previous_a: float = 0.0
        self.step_count: int = 0
        self.station_cleared: bool = False
        self.station_cleared_prev: bool = False

        self.timetable: Timetable = generate_timetable(
            self.vl.segments,
            self.STATIONS,
            dwell_seconds=float(LeadTrain.LEAD_DWELL_RANGE[1]),
            slack_factor=self.slack_factor,
        )
        self.ai_arrival_times: dict[int, float] = {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.STATIONS[0]
        self.v = 0.0
        self.time = 0.0
        self.last_station_idx = 0
        self.next_station_num = 1
        self.visited_stations = {0}
        self.last_a = 0.0
        self.previous_a = 0.0
        self.step_count = 0
        self.ai_arrival_times = {}
        self.ai_departure_time = 0.0
        # Start cleared at Chabówka (origin)
        self.station_cleared = True
        self.station_cleared_prev = True

        # Reset the lead train ~2km ahead
        start_lead_x = self.TRACK_START + 2000.0
        start_lead_v = self.vl.get_segment(start_lead_x).limit_ms
        self.lead_train.reset(
            start_x=start_lead_x, start_v=start_lead_v, np_random=self.np_random
        )

        self._update_dtz()

        # Initial aspects for obs
        seg = self.vl.get_segment(self.x)
        lead_seg = self.vl.get_segment(self.lead_train.x)
        lead_zone_idx = int((self.lead_train.x - lead_seg.start) / lead_seg.spatial_headway)
        x_lead_zone_start = lead_seg.start + lead_zone_idx * lead_seg.spatial_headway
        train_aspect = self.vl.compute_signal_aspect(self.x, seg, x_lead_zone_start)
        
        # Origin is cleared at Chabówka (index 0)
        station_aspect = 3 
        optimal_braking_distance = (self.v ** 2) / (2 * 0.5)

        return self._get_obs(train_aspect, station_aspect, x_lead_zone_start, optimal_braking_distance), {
            "overridden": False,
            "timetable": self.timetable.to_dict(),
        }
    def step(self, action: np.ndarray):
        # 1. Action space check
        raw_action = float(action[0])
        if raw_action > 0.5 or raw_action < -1.0:
            raw_action = max(-1.0, min(0.5, raw_action))  # clamp silently, VL logs violation

        # proposed_a is the raw action value directly (no remapping needed).
        proposed_a = raw_action
        
        # 2. Physics & State Update (AI Train)
        prev_x, prev_v = self.x, self.v
        seg = self.vl.get_segment(self.x)
        limit_v = seg.limit_ms
        dt = self.DT
        
        # Track last acceleration for dashboard/reward
        self.last_a = proposed_a

        # Simple SUVAT with clamping to local speed limit
        new_v = prev_v + proposed_a * dt
        new_v = max(0.0, min(new_v, limit_v))
        
        # To avoid sticking at 0 due to tiny actions, if proposed_a > 0 and v=0, 
        # ensure a minimum displacement if possible. 
        # But SUVAT already gives dx = 0.5 * a * dt^2 which is > 0 if a > 0.
        dx = (prev_v + new_v) / 2.0 * dt
        
        # Ensure that if proposed_a > 0, we actually move some distance
        if proposed_a > 1e-4 and dx < 1e-4:
            dx = 1e-4
            
        new_x = min(prev_x + max(0.0, dx), self.TRACK_END)
        
        self.x = new_x
        self.v = new_v
        self.time += dt
        self.step_count += 1
        
        # 3. Lead Train Update
        self.lead_train.advance(dt, self.vl, self.active_hazards, self.training_mode)
        
        # 4. Aspect & Signal Calculations
        seg = self.vl.get_segment(self.x)
        sh = seg.spatial_headway
        
        # Capture cleared status from the PREVIOUS step before any mutation this step.
        # reward_function.py needs this to detect "agent stalling after clearance".
        station_cleared_prev = self.station_cleared_prev
        self.station_cleared_prev = self.station_cleared
        
        # Lead train zone start
        lead_seg = self.vl.get_segment(self.lead_train.x)
        lead_zone_idx = int((self.lead_train.x - lead_seg.start) / lead_seg.spatial_headway)
        x_lead_zone_start = lead_seg.start + lead_zone_idx * lead_seg.spatial_headway
        
        train_aspect = self.vl.compute_signal_aspect(self.x, seg, x_lead_zone_start)
        
        # Station aspect
        x_station_zone_start = seg.end - sh
        
        # Check for station arrival
        next_st_idx = self.last_station_idx + 1
        reached_new_station = False
        
        # Ensure we only check for the station physically ahead of us
        target_station_pos = self.STATIONS[next_st_idx] if next_st_idx < len(self.STATIONS) else 999999.0

        # Arrival trigger: within spatial headway (SH) of the absolute station coordinate
        if next_st_idx < len(self.STATIONS) and self.x >= (target_station_pos - sh):
            # Arrived at station zone
            if next_st_idx not in self.visited_stations:
                reached_new_station = True
                self.last_station_idx = next_st_idx
                self.visited_stations.add(next_st_idx)
                self.station_cleared = False
                dwell_time = self.np_random.integers(20, 41) # 20-40 inclusive
                self.ai_departure_time = self.time + dwell_time
                self.ai_arrival_times[next_st_idx] = self.time

        if not self.station_cleared:
            if hasattr(self, "ai_departure_time") and self.time >= self.ai_departure_time:
                self.station_cleared = True
                self.next_station_num = min(self.next_station_num + 1, len(self.STATIONS) - 1)
        
        if not self.station_cleared:
            station_aspect = 0
        else:
            station_aspect = 3
            
        # 5. Validation Layer Integration
        violations = self.vl.check_and_log(
            self.x, self.v, proposed_a, seg, x_lead_zone_start, train_aspect, self.station_cleared
        )
        
        # 6. Rewards
        self._update_dtz()
        optimal_braking_distance = (self.v ** 2) / (2 * 0.5)
        
        state = TrainState(
            current_position=self.x,
            previous_position=prev_x,
            last_station_position=self.STATIONS[self.last_station_idx],
            next_station_position=self.STATIONS[min(self.last_station_idx + 1, len(self.STATIONS) - 1)],
            current_speed=self.v,
            speed_limit=seg.limit_ms,
            headway=self._compute_headway(),
            temporal_headway=seg.temporal_headway,
            spatial_headway=sh,
            reached_new_station=reached_new_station,
            scheduled_arrival_time=self.timetable.get_entry(self.last_station_idx).scheduled_arrival if self.timetable.get_entry(self.last_station_idx) else 0.0,
            actual_arrival_time=self.ai_arrival_times.get(self.last_station_idx),
            collision=(train_aspect == -1),
            safety_overridden=False, # Overrides removed
            action_delta=abs(proposed_a - self.previous_a),
            applied_traction=max(0.0, proposed_a),
            distance_to_occupied=x_lead_zone_start - self.x,
            is_dwelling=not self.station_cleared,
            station_index=self.last_station_idx,
            signal_aspect=train_aspect, # Use train_aspect for now
            applied_acceleration=proposed_a,
            proposed_acceleration=proposed_a,
            current_time=self.time,
            next_scheduled_arrival_time=self.timetable.get_entry(min(self.last_station_idx + 1, len(self.STATIONS) - 1)).scheduled_arrival if self.timetable.get_entry(min(self.last_station_idx + 1, len(self.STATIONS) - 1)) else 0.0,
        )
        # We need to pass more info to reward function as per instructions
        reward_info = {
            "train_aspect": train_aspect,
            "station_aspect": station_aspect,
            "station_cleared": self.station_cleared,
            "x_lead_zone_start": x_lead_zone_start,
            "x_station_zone_start": x_station_zone_start,
            "x_ai_zone_end": self.vl.compute_zone_boundaries(self.x, seg)[0],
            "optimal_braking_distance": optimal_braking_distance,
            "violations": violations,
            "lead_train_stalled": self.lead_train.stalled,
            "lead_train_held": self.lead_train.held,
            "previous_a": self.previous_a,
            "station_cleared_prev": station_cleared_prev,
        }
        
        reward_out = compute_reward(state, reward_info)
        self.previous_a = proposed_a

        # 7. Termination
        terminated = False
        if train_aspect == -1: # Collision / Zone overlap
            terminated = True
        if self.x >= self.TRACK_END and self.v < 0.1:
            terminated = True
            
        truncated = bool(self.step_count >= self.MAX_STEPS)

        # 8. Info Dict
        x_ai_zone_end, _ = self.vl.compute_zone_boundaries(self.x, seg)
        info = {
            "train_aspect": train_aspect,
            "station_aspect": station_aspect,
            "station_cleared": self.station_cleared,
            "x_ai_zone_end": x_ai_zone_end,
            "x_lead_zone_start": x_lead_zone_start,
            "x_station_zone_start": x_station_zone_start,
            "optimal_braking_distance": optimal_braking_distance,
            "violations": violations,
            "reward_breakdown": reward_out.to_dict() if hasattr(reward_out, "to_dict") else {},
            "punctuality_status": self.get_punctuality_status(),
        }

        return self._get_obs(train_aspect, station_aspect, x_lead_zone_start, optimal_braking_distance), reward_out.r_total, terminated, truncated, info

    def _get_obs(self, train_aspect: int, station_aspect: int, x_lead_zone_start: float, optimal_braking_distance: float) -> np.ndarray:
        return np.array(
            [
                self.x,
                self.v,
                self.dtz,
                float(train_aspect),
                float(station_aspect),
                x_lead_zone_start,
                optimal_braking_distance,
                self.time,
                float(self.next_station_num),
            ],
            dtype=np.float32,
        )

    def render(self):
        if self.render_mode == "human":
            seg = self.vl.get_segment(self.x)
            print(
                f"t={self.time:7.1f}s | x={self.x:8.1f}m | v={self.v:5.2f}m/s | seg={seg.id} | dtz={self.dtz:8.1f}m | lead={self.lead_train.x:8.1f}m"
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
        seg = self.vl.get_segment(self.x)
        self.dtz = self.vl.compute_dtz(self.x, seg)

    def _compute_headway(self) -> float:
        return (self.lead_train.x - self.x) / self.v if self.v > 0.01 else 9999.0

    def _get_signal_aspect(self) -> int:
        """Dashboard helper: The current most restrictive signal aspect."""
        seg = self.vl.get_segment(self.x)
        # 1. Aspect based on lead train position
        lead_seg = self.vl.get_segment(self.lead_train.x)
        lead_zone_idx = int((self.lead_train.x - lead_seg.start) / lead_seg.spatial_headway)
        x_lead_zone_start = lead_seg.start + lead_zone_idx * lead_seg.spatial_headway
        train_aspect = self.vl.compute_signal_aspect(self.x, seg, x_lead_zone_start)

        # 2. Aspect based on station clearance
        station_aspect = 3 if self.station_cleared else 0
        return int(min(train_aspect, station_aspect))

    def _dist_to_nearest_occupied(self) -> float:
        """Dashboard helper: distance in metres to the nearest occupied zone start."""
        seg = self.vl.get_segment(self.x)
        lead_seg = self.vl.get_segment(self.lead_train.x)
        lead_zone_idx = int((self.lead_train.x - lead_seg.start) / lead_seg.spatial_headway)
        x_lead_zone_start = lead_seg.start + lead_zone_idx * lead_seg.spatial_headway

        if not self.station_cleared:
            x_station_zone_start = seg.end - seg.spatial_headway
            return max(0.0, x_station_zone_start - self.x)

        return max(0.0, x_lead_zone_start - self.x)


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