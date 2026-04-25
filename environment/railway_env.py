"""
environment/railway_env.py

Gymnasium environment for a single train traversing Polish Rail Line 104
(Chabówka → Nowy Sącz, ~76 km).

Convoy model: one AI-controlled train follows a lead train that cruises
at a configurable constant speed.  The agent chooses a 4-aspect signal
action each timestep; the Validation Layer ensures safety before the
physics update runs.
"""

from __future__ import annotations
import sys
from pathlib import Path

import math

import gymnasium as gym
import numpy as np

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_layer.validator import ValidationLayer   # noqa: E402
from reward.reward_function import compute_reward, TrainState  # noqa: E402
from environment.timetable import (                             # noqa: E402
    Timetable, generate_timetable, compute_eta_to_station,
    STATION_NAMES,
)
from dataclasses import dataclass


@dataclass
class Landslide:
    """A named, toggleable hazard at a specific track position."""
    position: float   # exact position passed by the caller (metres)
    active: bool      # whether the hazard is currently in effect
    _block_start: float  # start of the enclosing fixed block
    _block_end: float    # end   of the enclosing fixed block

class ModernizedLine104(gym.Env):
    """
    Gymnasium environment — Line 104 single-train convoy.

    Observation  (Box, shape=(7,)):
        [position, speed, dtz, signal_aspect,
         distance_to_next_station, speed_limit, headway]

    Action (Discrete(4)):
        0 = Red (stop)          1 = Yellow (1/3 limit)
        2 = Double-Yellow (2/3) 3 = Green  (full limit)
    """

    metadata = {"render_modes": ["human"]}
    
    # --- Lead train behaviour ---
    LEAD_DWELL_RANGE: tuple[int, int] = (15, 60)  # random dwell bounds (seconds)
    LEAD_SERVICE_DECEL: float = -0.5  # comfortable service braking for lead (m/s²)

    # --- Physics ---
    DT: float = 1.0      # timestep  (seconds)
    ACCEL: float = 0.5    # traction acceleration  (m/s²)
    DECEL: float = -1.0   # emergency braking  (m/s²)

    # --- Episode limits ---
    MAX_STEPS: int = 10_000  # ~2.8 hours simulated at DT=1.0

    # --- Track extents ---
    TRACK_START: float = 582.0
    TRACK_END: float = 76651.0

    # --- Station positions (metres) — from segment map ---
    STATIONS: list[float] = [582, 5481, 14951, 37160, 47017, 67394, 76651]

    # Speed targets now derived from segment limit via
    # ValidationLayer._speed_for_action (1/3, 2/3, full).

    def __init__(
        self,
        lead_train_speed: float = 20.0,
        render_mode: str | None = None,
        training_mode: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.lead_train_speed = lead_train_speed
        self.training_mode = training_mode
        self.active_hazards: set[tuple[float, float]] = set()  # {(block_start, block_end), ...}
        self.landslides: dict[int, Landslide] = {}   # idx → Landslide
        self._landslide_counter: int = 0             # monotonically increasing key

        # Validation Layer (safety sieve)
        self.vl = ValidationLayer()

        # Spaces
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0], dtype=np.float32),
            high=np.array([0.5], dtype=np.float32),
            dtype=np.float32,
        )
        self.observation_space = gym.spaces.Box(
            low=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            high=np.array([80000.0, 30.0, 80000.0, 3.0, 80000.0, 30.0, 10000.0], dtype=np.float32),
            dtype=np.float32,
        )

        # State variables — initialised properly in reset()
        self.x: float = self.TRACK_START
        self.v: float = 0.0
        self.dtz: float = 0.0
        self.lead_x: float = 0.0 
        self.lead_v: float = self.lead_train_speed # current speed ﹀
        self.time: float = 0.0
        self.lead_dwell_timer: int = 0 # seconds remaining at station ﹀ 
        self.lead_station_idx: int = 1 # which station the lead is targeting next ﹀
        self.last_station_idx: int = 0
        self.visited_stations: set[int] = set()
        self.last_a: float = 0.0  # Track the last *executed* acceleration for jerk calculation
        self.step_count: int = 0
        self.lead_stalled: bool = False   # external stall command (frontend)
        self.lead_held: bool = False      # external hold at station (frontend)
        self.random_stall_timer: int = 0  # mid-track random stalls during training

        # Pre-compute ideal arrival times for punctuality tracking
        self._ideal_schedule = self._compute_ideal_schedule()

        # Timetable — richer schedule with dwell, used for API/dashboard
        self.timetable: Timetable = generate_timetable(
            segments=self.vl.segments,
            station_positions=self.STATIONS,
            dwell_seconds=float(self.LEAD_DWELL_RANGE[1]),  # max dwell
        )
        # Actual arrival timestamps — populated as stations are reached
        self.ai_arrival_times: dict[int, float] = {}

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.TRACK_START
        self.v = 0.0
        self.time = 0.0
        self.lead_dwell_timer = 0 # seconds remaining at station ﹀
        self.lead_station_idx = 1 # which station the lead is targeting next ﹀
        self.last_station_idx = 0
        self.visited_stations = {0}  # starting station already "visited"
        self.last_a = 0.0
        self.step_count = 0

        # Lead train begins ~2 km ahead
        self.lead_x = self.TRACK_START + 2000.0
        lead_seg = self.vl.get_segment(self.lead_x)
        self.lead_v = lead_seg.limit_ms   # ideal train cruises at segment limit
        self.lead_stalled = False
        self.lead_held = False
        self.random_stall_timer = 0
        # Note: active_hazards are NOT cleared on reset — hazards persist
        # across episodes because they represent external physical events
        # injected via the API, not simulation state.
        self._update_dtz()

        # Reset arrival tracking
        self.ai_arrival_times = {}

        return self._get_obs(), {
            "overridden": False,
            "timetable": self.timetable.to_dict(),
        }

    def step(self, action: np.ndarray):
        """
        Execute one environment step:
            1.  Validate the proposed continuous acceleration through the safety sieve.
            2.  Apply direct acceleration physics.
            3.  Advance the lead train.
            4.  Compute the reward.
        """
        # Ensure action is a float scalar
        proposed_a = float(action[0])

        # Current signal aspect (derived from distance to lead train)
        env_aspect = self._get_signal_aspect()

        # ----- 1. Validation Layer (Layers 1–4) -----
        # VL handles the safety check against the aspect and dtz.
        safe_a, overridden = self.vl.get_safe_action(
            proposed_a, env_aspect, self.x, self.v, self.dtz,
        )

        # Jerk tracking: delta of the *physically executed* acceleration
        action_delta = abs(safe_a - self.last_a)
        self.last_a = safe_a

        # ----- 2. Physics — two-phase SUVAT update -----
        # Matches validator._project() so safety projection and actual
        # kinematics use the same model.
        prev_x = self.x
        prev_v = self.v

        seg = self.vl.get_segment(self.x)
        limit_v = seg.limit_ms

        if safe_a == 0.0:
            # Coasting at current speed (capped at segment limit)
            self.v = min(prev_v, limit_v)
            dx = self.v * self.DT
        elif safe_a > 0.0:
            if prev_v >= limit_v:
                # Already at limit — coast
                self.v = limit_v
                dx = limit_v * self.DT
            else:
                t_to_limit = (limit_v - prev_v) / safe_a
                if self.DT <= t_to_limit:
                    # Full acceleration phase
                    self.v = prev_v + safe_a * self.DT
                    dx = prev_v * self.DT + 0.5 * safe_a * self.DT ** 2
                else:
                    # Accelerate to limit, then coast for remainder
                    dx = (prev_v * t_to_limit
                          + 0.5 * safe_a * t_to_limit ** 2
                          + limit_v * (self.DT - t_to_limit))
                    self.v = limit_v
        else:  # safe_a < 0
            if prev_v <= 0.0:
                # Already stopped
                self.v = 0.0
                dx = 0.0
            else:
                t_to_zero = prev_v / abs(safe_a)
                if self.DT <= t_to_zero:
                    # Full braking phase
                    self.v = prev_v + safe_a * self.DT
                    dx = prev_v * self.DT + 0.5 * safe_a * self.DT ** 2
                else:
                    # Brake to stop, then stationary for remainder
                    dx = prev_v * t_to_zero + 0.5 * safe_a * t_to_zero ** 2
                    self.v = 0.0

        self.v = max(0.0, self.v)
        self.x = min(prev_x + max(0.0, dx), self.TRACK_END)

        # Calculate applied traction (positive acceleration only)
        # Note: if safe_a was overridden to -1.0, applied_traction is 0.0
        applied_traction = max(0.0, safe_a)

        self.time += self.DT
        self.step_count += 1

        # ----- 3. Lead train (ideal, respects segment speed limits) -----
        self._advance_lead_train()
        self._update_dtz()

        # ----- 4. Station arrival check -----
        reached_new_station = False
        next_st_idx = self.last_station_idx + 1
        scheduled_arrival_time = None
        actual_arrival_time = None
        
        if next_st_idx < len(self.STATIONS):
            if self.x >= self.STATIONS[next_st_idx]:
                reached_new_station = True
                self.last_station_idx = next_st_idx
                self.visited_stations.add(next_st_idx)
                
                # BUG 13 FIX: Supply arrival times to TrainState so the
                # punctuality penalty is actually calculated.
                actual_arrival_time = self.time
                
                # Grade against the simulated timetable (which includes dwells), 
                # rather than the physically 'ideal' no-stops schedule.
                entry = self.timetable.get_entry(next_st_idx)
                scheduled_arrival_time = entry.scheduled_arrival if entry else self._ideal_schedule[next_st_idx]

                # Record AI arrival for timetable punctuality tracking
                self.ai_arrival_times[next_st_idx] = self.time

        # ----- 5. Reward computation -----
        last_st_pos = self.STATIONS[self.last_station_idx]
        next_st_pos = self.STATIONS[
            min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        ]

        # Temporal headway: time gap to the lead train (seconds)
        headway = self._compute_headway() # ﹀

        state = TrainState(
            current_position=self.x,
            previous_position=prev_x,
            last_station_position=last_st_pos,
            next_station_position=next_st_pos,
            current_speed=self.v,
            speed_limit=seg.limit_ms,
            headway=headway,
            reached_new_station=reached_new_station,
            scheduled_arrival_time=scheduled_arrival_time,
            actual_arrival_time=actual_arrival_time,
            collision=(self.x >= self.lead_x),
            temporal_headway=seg.temporal_headway,
            overridden=overridden,
            action_delta=action_delta,
            applied_traction=applied_traction,
        )

        reward_out = compute_reward(state)
        total_reward = reward_out.r_total

        # ----- 6. Termination & Truncation -----
        terminated = bool(self.x >= self.TRACK_END or reward_out.terminate)
        truncated = bool(self.step_count >= self.MAX_STEPS)

        info = {
            "overridden": overridden,
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
                "station": reward_out.r_station,
                "punctuality": reward_out.r_time,
                "override": reward_out.r_override,
                "jerk": reward_out.r_jerk,
                "energy": reward_out.r_energy,
                "violation": reward_out.r_violation,
                "collision": reward_out.r_collision,
            },
            "punctuality_status": self.get_punctuality_status(),
        }

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            seg = self.vl.get_segment(self.x)
            print(
                f"t={self.time:7.1f}s | "
                f"x={self.x:8.1f}m | "
                f"v={self.v:5.2f} m/s ({self.v * 3.6:5.1f} km/h) | "
                f"seg={seg.id} | "
                f"dtz={self.dtz:8.1f}m | "
                f"lead={self.lead_x:8.1f}m"
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_dtz(self) -> None:
        """
        Update Distance-to-Zone.

        DTZ = distance from the train to the next fixed-block boundary.
        Blocks are SH-length subdivisions within each segment, so
        DTZ is always ≤ SH.
        """
        self.dtz = self.vl.compute_dtz(self.x)

    def _nearest_obstruction(self, from_x: float) -> float:
        """Return position of the nearest obstruction ahead of *from_x*.

        Obstructions are:
            1. The lead train (at self.lead_x)
            2. Any active hazard block whose start is ahead of *from_x*

        Returns the position (metres) of the closest one.
        """
        nearest = self.lead_x
        for h_start, _h_end in self.active_hazards:
            if h_start > from_x:
                nearest = min(nearest, h_start)
        return nearest

    def _get_signal_aspect(self) -> int:
        """
        Derive the 4-aspect signal from the distance to the nearest
        obstruction (lead train OR hazard block) and the current
        segment's spatial headway (SH).

        The nearest obstruction determines how many blocks ahead are
        clear.  SH is the block length:
            Green  (3) — next 3 blocks clear    (dist > 3 × SH)
            DblYlw (2) — next 2 blocks clear    (dist > 2 × SH)
            Yellow (1) — next 1 block  clear     (dist > 1 × SH)
            Red    (0) — next block occupied     (dist ≤ 1 × SH)
        """
        nearest = self._nearest_obstruction(self.x)
        dist_to_obstruction = nearest - self.x
        sh = self.vl.get_segment(self.x).spatial_headway

        if dist_to_obstruction > 3 * sh:
            return 3
        if dist_to_obstruction > 2 * sh:
            return 2
        if dist_to_obstruction > sh:
            return 1
        return 0

    def _compute_headway(self) -> float: # ﹀
        """
        Temporal headway — time gap in seconds between the AI train
        and the lead train, based on current AI speed.
        Returns 9999.0 when the AI is stationary to avoid division by zero.
        """
        if self.v > 0.01:
            return (self.lead_x - self.x) / self.v
        return 9999.0

    def _compute_ideal_schedule(self) -> list[float]:
        """Pre-compute ideal travel times to each station (seconds).

        Uses segment speed limits with no dwell — the fastest physically
        possible arrival at each station.  Used by the punctuality reward.
        """
        times = [0.0]
        cumulative = 0.0
        for i, station_pos in enumerate(self.STATIONS[1:], start=1):
            cumulative += compute_eta_to_station(
            self.STATIONS[i - 1], station_pos, self.vl.segments
        )
        times.append(cumulative)
        return times

    def _advance_lead_train(self) -> None:
        """Advance the lead train one timestep with realistic physics.

        The lead train is an *ideal* driver: it targets the segment
        speed limit, brakes smoothly for station stops and speed-limit
        transitions, and accelerates at the standard traction rate.

        External controls:
            lead_stalled  — forces emergency braking to a full stop
            lead_held     — freezes the dwell timer at a station
        """
        # --- Random Stalls for Training Mode ---
        if self.training_mode and self.lead_dwell_timer == 0 and not self.lead_stalled and not self.lead_held:
            # 0.1% chance per step (~7 expected stalls per trip) to trigger a random mid-track stall
            if self.np_random.random() < 0.001:
                self.random_stall_timer = self.np_random.integers(15, 60)  # stall for 15-60 seconds

        # --- Stall override: emergency brake to stop ---
        if self.lead_stalled or self.random_stall_timer > 0:
            if self.random_stall_timer > 0:
                self.random_stall_timer -= 1
            if self.lead_v > 0:
                self.lead_v = max(0.0, self.lead_v + self.DECEL * self.DT)
                self.lead_x += self.lead_v * self.DT
            return

        # --- Dwelling at station ---
        if self.lead_dwell_timer > 0:
            if not self.lead_held:
                self.lead_dwell_timer -= 1
            if self.lead_dwell_timer == 0 and not self.lead_held:
                if self.lead_station_idx < len(self.STATIONS) - 1:
                    self.lead_station_idx += 1
                else:
                    # Final station — pull forward one block and park
                    self.lead_x = self.STATIONS[-1] + 34.7
                    self.lead_v = 0.0
                    self.lead_station_idx = len(self.STATIONS)  # mark done
            return

        # --- Past all stations ---
        if self.lead_station_idx >= len(self.STATIONS):
            return

        # --- Moving: determine speed ceiling from constraints ---
        seg = self.vl.get_segment(self.lead_x)
        seg_limit = seg.limit_ms
        brake_a = abs(self.LEAD_SERVICE_DECEL)

        # Constraint 1: must be able to stop at next station
        d_station = max(0.01, self.STATIONS[self.lead_station_idx] - self.lead_x)
        v_ceil_station = math.sqrt(2 * brake_a * d_station)

        # Constraint 2: must be able to slow for a slower upcoming segment
        v_ceil_seg = float('inf')
        seg_idx = self.vl.segments.index(seg)
        if seg_idx + 1 < len(self.vl.segments):
            next_seg = self.vl.segments[seg_idx + 1]
            if next_seg.limit_ms < seg_limit:
                d_boundary = max(0.01, seg.end - self.lead_x)
                v_ceil_seg = math.sqrt(
                    next_seg.limit_ms ** 2 + 2 * brake_a * d_boundary
                )

        # Constraint 3: must be able to stop before any hazard block ahead
        # Use >= so that a train parked exactly AT the hazard boundary
        # gets v_ceil=0 and stays stopped (d_hazard=0 → v_ceil=0).
        v_ceil_hazard = float('inf')
        for h_start, _h_end in self.active_hazards:
            if h_start >= self.lead_x:
                d_hazard = h_start - self.lead_x  # 0 when parked at boundary
                v_ceil_hazard = min(v_ceil_hazard, math.sqrt(2 * brake_a * max(0.0, d_hazard)))

        # Effective ceiling: lowest of all constraints
        v_ceil = min(seg_limit, v_ceil_station, v_ceil_seg, v_ceil_hazard)

        # --- Choose acceleration ---
        if self.lead_v > v_ceil + 0.01:
            a = self.LEAD_SERVICE_DECEL          # brake
        elif self.lead_v < v_ceil - 0.01:
            a = self.ACCEL                       # accelerate
        else:
            a = 0.0                              # coast

        # --- Two-phase SUVAT position update ---
        prev_v = self.lead_v
        if a == 0.0:
            self.lead_v = min(prev_v, v_ceil)
            dx = self.lead_v * self.DT
        elif a > 0:
            if prev_v >= v_ceil:
                self.lead_v = v_ceil
                dx = v_ceil * self.DT
            else:
                t_to_ceil = (v_ceil - prev_v) / a
                if self.DT <= t_to_ceil:
                    self.lead_v = prev_v + a * self.DT
                    dx = prev_v * self.DT + 0.5 * a * self.DT ** 2
                else:
                    dx = (prev_v * t_to_ceil
                          + 0.5 * a * t_to_ceil ** 2
                          + v_ceil * (self.DT - t_to_ceil))
                    self.lead_v = v_ceil
        else:  # braking
            if prev_v <= v_ceil:
                self.lead_v = max(0.0, v_ceil)
                dx = self.lead_v * self.DT
            else:
                t_to_ceil = (prev_v - v_ceil) / abs(a)
                if self.DT <= t_to_ceil:
                    self.lead_v = prev_v + a * self.DT
                    dx = prev_v * self.DT + 0.5 * a * self.DT ** 2
                else:
                    dx = (prev_v * t_to_ceil
                          + 0.5 * a * t_to_ceil ** 2
                          + v_ceil * (self.DT - t_to_ceil))
                    self.lead_v = v_ceil

        self.lead_v = max(0.0, self.lead_v)
        prev_lead_x = self.lead_x  # position before this step's move
        self.lead_x += max(0.0, dx)

        # --- Snap to hazard block boundary if reached ---
        for h_start, _h_end in self.active_hazards:
            if prev_lead_x < h_start <= self.lead_x:
                self.lead_x = h_start
                self.lead_v = 0.0
                return  # parked at hazard — skip station snap

        # --- Snap to station on arrival ---
        if self.lead_station_idx < len(self.STATIONS):
            next_station = self.STATIONS[self.lead_station_idx]
            if self.lead_x >= next_station:
                self.lead_x = next_station
                self.lead_v = 0.0
                self.lead_dwell_timer = self.np_random.integers(
                    self.LEAD_DWELL_RANGE[0], self.LEAD_DWELL_RANGE[1]
                )

    def stall_lead(self) -> None:
        """External command: force the lead train to emergency brake."""
        self.lead_stalled = True

    def release_lead(self) -> None:
        """External command: release the stall / hold on the lead train."""
        self.lead_stalled = False
        self.lead_held = False

    def hold_lead(self) -> None:
        """External command: freeze the lead train at its current station."""
        self.lead_held = True

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector (7 values)."""
        aspect = self._get_signal_aspect()
        seg = self.vl.get_segment(self.x)
        next_st_pos = self.STATIONS[
            min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        ]
        dist_to_next_station = max(0.0, next_st_pos - self.x)
        headway = self._compute_headway() # ﹀
        return np.array(
            [self.x, self.v, self.dtz, float(aspect),
             dist_to_next_station, seg.limit_ms, min(headway, 9999.0)],
            dtype=np.float32,
        )
        
    def set_block_hazard(self, start: float, end: float, active: bool) -> None:
        """Set or clear a hazard on the block spanning [start, end).

        When active, this block acts as a virtual obstruction:
        both the AI and lead trains will see a degraded signal aspect
        and brake before entering the block.
        """
        key = (start, end)
        if active:
            self.active_hazards.add(key)
        else:
            self.active_hazards.discard(key)

    def clear_all_hazards(self) -> None:
        """Remove all active hazards."""
        self.active_hazards.clear()
        
        # ------------------------------------------------------------------
    # Landslide API  (named, indexed, toggleable hazards)
    # ------------------------------------------------------------------

    def _find_block_for_position(self, position: float) -> tuple[float, float]:
        """Return the (block_start, block_end) of the fixed block that
        contains *position*.  Used to translate a landslide position into
        the (start, end) key expected by active_hazards / set_block_hazard.
        """
        seg = self.vl.get_segment(position)
        bounds = seg.block_boundaries
        for i in range(len(bounds) - 1):
            if bounds[i] <= position < bounds[i + 1]:
                return bounds[i], bounds[i + 1]
        # Fallback: position is at the very last boundary
        return bounds[-2], bounds[-1]

    def set_landslide(self, position: float) -> int:
        """Register a new active landslide at *position* and return its index.

        The landslide immediately blocks the enclosing fixed block (as if
        set_block_hazard were called with active=True).
        """
        block_start, block_end = self._find_block_for_position(position)
        idx = self._landslide_counter
        self._landslide_counter += 1
        self.landslides[idx] = Landslide(
            position=position,
            active=True,
            _block_start=block_start,
            _block_end=block_end,
        )
        self.set_block_hazard(block_start, block_end, True)
        return idx

    def toggle_landslide(self, idx: int) -> bool:
        """Toggle the landslide at *idx* on/off.  Returns the new active state."""
        ls = self.landslides[idx]
        ls.active = not ls.active
        # Only deactivate block if no other landslide in the same block is still active
        block_still_needed = any(
        other.active and other._block_start == ls._block_start
        for i, other in self.landslides.items() if i != idx
        )
        if ls.active or not block_still_needed:
            self.set_block_hazard(ls._block_start, ls._block_end, ls.active)
        return ls.active

    def clear_landslide(self, idx: int) -> None:
        """Remove the landslide at *idx* and deactivate its block hazard."""
        ls = self.landslides.pop(idx)
        self.set_block_hazard(ls._block_start, ls._block_end, False)

    # ------------------------------------------------------------------
    # Timetable & Punctuality
    # ------------------------------------------------------------------

    def _train_punctuality(
        self,
        position: float,
        last_visited_idx: int,
        arrival_log: dict[int, float],
    ) -> dict:
        """Compute live punctuality for one train.

        Parameters
        ----------
        position         : Current position in metres.
        last_visited_idx : Index of the last station this train visited.
        arrival_log      : Dict mapping station_idx → actual arrival time.

        Returns
        -------
        dict with keys: next_station_idx, next_station_name, scheduled_arrival,
                        eta, slack_seconds, status, arrival_log.
        """
        next_idx = last_visited_idx + 1
        if next_idx >= len(self.STATIONS):
            # Past all stations — journey complete
            return {
                "next_station_idx": None,
                "next_station_name": None,
                "scheduled_arrival": None,
                "eta": None,
                "slack_seconds": None,
                "status": "arrived",
                "arrival_log": {
                    str(k): round(v, 1) for k, v in arrival_log.items()
                },
            }

        entry = self.timetable.get_entry(next_idx)
        if entry is None:
            return {
                "next_station_idx": next_idx,
                "next_station_name": STATION_NAMES[next_idx] if next_idx < len(STATION_NAMES) else f"Station {next_idx}",
                "scheduled_arrival": None,
                "eta": None,
                "slack_seconds": None,
                "status": "unknown",
                "arrival_log": {
                    str(k): round(v, 1) for k, v in arrival_log.items()
                },
            }

        # ETA: segment-aware with accel/decel buffer
        remaining_travel = compute_eta_to_station(
            position, entry.position_m, self.vl.segments,
        )
        eta = self.time + remaining_travel

        slack = entry.scheduled_arrival - eta  # positive = ahead, negative = behind

        if abs(slack) <= 60.0:
            status = "on_time"
        elif slack > 0:
            status = "early"
        else:
            status = "late"

        return {
            "next_station_idx": next_idx,
            "next_station_name": entry.station_name,
            "scheduled_arrival": round(entry.scheduled_arrival, 1),
            "eta": round(eta, 1),
            "slack_seconds": round(slack, 1),
            "status": status,
            "arrival_log": {
                str(k): round(v, 1) for k, v in arrival_log.items()
            },
        }

    def get_punctuality_status(self) -> dict:
        """Compute live schedule status for the AI train.

        Returns a dict suitable for JSON serialisation and WebSocket broadcast.
        """
        return {
            "sim_time": round(self.time, 1),
            "ai": self._train_punctuality(
                self.x, self.last_station_idx, self.ai_arrival_times
            ),
        }

# -----------------------------------------------------------------------
# Quick smoke-test
# -----------------------------------------------------------------------
if __name__ == "__main__":
    env = ModernizedLine104(render_mode="human")
    obs, info = env.reset()
    print("=== Line 104 Environment — Smoke Test ===\n")
    env.render()

    total_r = 0.0
    for step_i in range(env.MAX_STEPS):
        # Continuous action: try to accelerate at max (0.5)
        action = np.array([0.5], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        total_r += reward

        if step_i % 500 == 0:
            env.render()
            print(f"     step {step_i:4d}  reward={reward:+8.3f}  "
                  f"overridden={info['overridden']}  "
                  f"safe_a={info['safe_a']:.2f}")

        if terminated or truncated:
            print(f"\n--- Episode ended at step {step_i} ---")
            break
        
        if step_i % 100 == 0:
            print(f"  lead_x={env.lead_x:.1f}  lead_v={env.lead_v:.2f}  "
                f"dwell={env.lead_dwell_timer}  lead_target_st={env.lead_station_idx}")

    env.render()
    print(f"\nTotal reward: {total_r:+.2f}")
    print(f"Stations visited: {sorted(env.visited_stations)}")
