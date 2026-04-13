"""
environment/railway_env.py

Gymnasium environment for a single train traversing Polish Rail Line 104
(Chabówka → Nowy Sącz, ~76 km).

Convoy model:
    Lead train  — rule-based; cruises at lead_train_speed, decelerates to
                  stop at every station (SUVAT physics, ±0.5 m/s²), dwells
                  for STATION_DWELL_TIME steps, then re-accelerates.
                  Station stops propagate to the AI train naturally via the
                  headway / signal-aspect mechanism.
    AI train    — PPO agent; continuous acceleration in [-1.0, 0.5] m/s².
                  Every action passes through the Validation Layer before
                  physics run.

Landslides (operator-configurable):
    Up to MAX_LANDSLIDES (3) landslides can be placed at arbitrary track
    positions during a live demo.  Each active landslide is treated as a
    virtual stopped train; the AI train sees it only through a reduced signal
    aspect — no new observation dimensions are required.
    Use set_landslide(), clear_landslide(), and toggle_landslide() to manage
    them at runtime.

Physics constants match validate_layer/validator.py:
    ACCEL           = +0.5  m/s²
    EMERGENCY_DECEL = -1.0  m/s²
"""

import sys
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import bisect
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_layer.validator import ValidationLayer   # noqa: E402
from reward.reward_function import compute_reward, TrainState  # noqa: E402


# ---------------------------------------------------------------------------
# Landslide — operator-placed track obstruction
# ---------------------------------------------------------------------------

@dataclass
class Landslide:
    """A landslide blocking the track at a fixed position.

    Treated as a virtual stopped train for signal-aspect calculation.
    Can be toggled on/off during a live demo without restarting the episode.
    """
    position: float   # metres along track
    active: bool = True


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class ModernizedLine104(gym.Env):
    """
    Gymnasium environment — Line 104 single-train convoy.

    Observation  (Box, shape=(7,)):
        [position, speed, dtz, signal_aspect,
         distance_to_next_station, speed_limit, headway]

    Action  (Box, shape=(1,), range=[-1.0, 0.5]):
        Continuous acceleration in m/s²:
            -1.0 = emergency brake   0.0 = coast   +0.5 = max traction
    """

    metadata = {"render_modes": ["human"]}

    # --- Physics (must match validate_layer/validator.py constants) ---
    DT: float = 1.0
    ACCEL: float = 0.5
    EMERGENCY_DECEL: float = -1.0

    # --- Lead train ---
    LEAD_SERVICE_DECEL: float = 0.5   # braking magnitude for station stop (m/s²)
    LEAD_ACCEL: float = 0.5           # traction magnitude for acceleration (m/s²)
    STATION_DWELL_TIME: int = 30      # steps the lead train waits at each station (deterministic)

    # --- Lead train randomisation (training mode only) ---
    LEAD_SPEED_MIN: float = 15.0      # m/s
    LEAD_SPEED_MAX: float = 25.0      # m/s
    DWELL_MIN: int = 20               # steps
    DWELL_MAX: int = 60               # steps
    SLOWDOWN_SPEED_MIN: float = 5.0   # m/s
    SLOWDOWN_SPEED_MAX: float = 12.0  # m/s
    SLOWDOWN_DURATION_MIN: int = 30   # steps
    SLOWDOWN_DURATION_MAX: int = 120  # steps

    # --- Landslides ---
    MAX_LANDSLIDES: int = 3

    # --- Episode limits ---
    MAX_STEPS: int = 10_000

    # --- Track extents ---
    TRACK_START: float = 582.0
    TRACK_END: float = 76651.0

    # --- Station positions (metres) ---
    STATIONS: list[float] = [582, 5481, 14951, 37160, 47017, 67394, 76651]

    def __init__(
        self,
        lead_train_speed: float = 20.0,
        render_mode: str | None = None,
        training_mode: bool = False,
    ):
        super().__init__()
        self.render_mode = render_mode
        self.lead_train_speed = lead_train_speed
        self._base_lead_speed = lead_train_speed  # preserved for reset in sim mode
        self.training_mode = training_mode

        self.vl = ValidationLayer()

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

        # AI train state (initialised in reset)
        self.x: float = self.TRACK_START
        self.v: float = 0.0
        self.dtz: float = 0.0
        self.time: float = 0.0
        self.last_station_idx: int = 0
        self.visited_stations: set[int] = set()
        self.last_a: float = 0.0
        self.step_count: int = 0

        # Lead train state (initialised in reset)
        self.lead_x: float = 0.0
        self.lead_v: float = 0.0
        self.lead_station_idx: int = 0
        self.lead_dwell_timer: int = 0

        # Landslides — persist across resets; managed via public API
        self.landslides: list[Landslide] = []

<<<<<<< HEAD
        # Pre-compute unified block boundaries for RBC signal calculation
        self._all_boundaries = self._build_unified_boundaries()
=======
        # Lead train slowdown state — set via set_lead_slowdown() or scheduled in training
        self.lead_slow_until: int = 0    # step_count at which slowdown ends
        self.lead_slow_speed: float = 0.0  # target speed during slowdown
        self._scheduled_slowdowns: list[tuple[int, int, float]] = []  # (activate_at, duration, speed)
>>>>>>> 415f571 (Changes...)

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # AI train
        self.x = self.TRACK_START
        self.v = 0.0
        self.time = 0.0
        self.last_station_idx = 0
        self.visited_stations = {0}
        self.last_a = 0.0
        self.step_count = 0

        # Lead train — starts 2 km ahead, already at cruise speed
        if self.training_mode:
            self.lead_train_speed = float(self.np_random.uniform(
                self.LEAD_SPEED_MIN, self.LEAD_SPEED_MAX
            ))
        else:
            self.lead_train_speed = self._base_lead_speed

        self.lead_x = self.TRACK_START + 2000.0
        self.lead_v = self.lead_train_speed
        self.lead_station_idx = 0
        self.lead_dwell_timer = 0

        # Slowdown state
        self.lead_slow_until = 0
        self.lead_slow_speed = 0.0
        self._scheduled_slowdowns = []
        if self.training_mode:
            self._schedule_lead_slowdowns()

        self._update_dtz()
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        proposed_a = float(action[0])
        env_aspect = self._get_signal_aspect()

        # ----- 1. Validation Layer -----
        safe_a, overridden = self.vl.get_safe_action(
            proposed_a, env_aspect, self.x, self.v, self.dtz,
        )

        action_delta = abs(safe_a - self.last_a)
        self.last_a = safe_a

        # ----- 2. AI train physics — SUVAT -----
        prev_x = self.x
        prev_v = self.v

        # s = u·t + ½·a·t²  (prev_v avoids forward-Euler position overestimate)
        self.x += prev_v * self.DT + 0.5 * safe_a * (self.DT ** 2)
        self.x = max(self.TRACK_START, min(self.x, self.TRACK_END))

        # v = u + a·t; segment resolved after position so correct limit applies
        v_new = max(0.0, prev_v + safe_a * self.DT)
        seg = self.vl.get_segment(self.x)
        self.v = min(v_new, seg.limit_ms)

        applied_traction = max(0.0, safe_a)
        self.time += self.DT
        self.step_count += 1

        # ----- 3. Lead train (with station stops) -----
        self._step_lead_train()
        self._update_dtz()

        # ----- 4. Landslides (stationary — no step required) -----

        # ----- 5. AI train station arrival -----
        reached_new_station = False
        while self.last_station_idx + 1 < len(self.STATIONS):
            next_st_idx = self.last_station_idx + 1
            if self.x >= self.STATIONS[next_st_idx]:
                reached_new_station = True
                self.last_station_idx = next_st_idx
                self.visited_stations.add(next_st_idx)
            else:
                break

        # ----- 6. Reward -----
        last_st_pos = self.STATIONS[self.last_station_idx]
        next_st_pos = self.STATIONS[
            min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        ]
        headway = self._compute_headway()

        state = TrainState(
            current_position=self.x,
            previous_position=prev_x,
            last_station_position=last_st_pos,
            next_station_position=next_st_pos,
            current_speed=self.v,
            speed_limit=seg.limit_ms,
            headway=headway,
            reached_new_station=reached_new_station,
            collision=(self.x >= self.lead_x),
            temporal_headway=seg.temporal_headway,
            overridden=overridden,
            action_delta=action_delta,
            applied_traction=applied_traction,
        )

        reward_out = compute_reward(state)
        total_reward = reward_out.r_total

        # ----- 7. Termination & Truncation -----
        terminated = bool(self.x >= self.TRACK_END or reward_out.terminate)
        truncated = bool(self.step_count >= self.MAX_STEPS)

        info = {
            "overridden": overridden,
            "safe_a": safe_a,
            "proposed_a": proposed_a,
            "aspect": env_aspect,
            "time": self.time,
            "segment": seg.id,
            "lead_x": self.lead_x,
            "lead_v": self.lead_v,
            "lead_dwell": self.lead_dwell_timer,
            "landslides": [
                {"position": ls.position, "active": ls.active}
                for ls in self.landslides
            ],
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
        }

        return self._get_obs(), total_reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            seg = self.vl.get_segment(self.x)
            dwell = f"  [DWELL {self.lead_dwell_timer}s]" if self.lead_dwell_timer > 0 else ""
<<<<<<< HEAD
            active_ls = [ls for ls in self.landslides if ls.active]
            hz = f"  hazards={len(active_ls)}" if active_ls else ""
=======
            hz = f"  landslides={len(self.landslides)}" if self.landslides else ""
>>>>>>> 415f571 (Changes...)
            print(
                f"t={self.time:7.1f}s | "
                f"x={self.x:8.1f}m | "
                f"v={self.v:5.2f} m/s ({self.v * 3.6:5.1f} km/h) | "
                f"seg={seg.id} | "
                f"dtz={self.dtz:8.1f}m | "
                f"lead={self.lead_x:8.1f}m ({self.lead_v * 3.6:.1f} km/h)"
                f"{dwell}{hz}"
            )

    # ------------------------------------------------------------------
    # Lead train
    # ------------------------------------------------------------------

    def _step_lead_train(self) -> None:
        """
        Advance the lead train one timestep using SUVAT integration.

        State machine:
            DWELL   — timer > 0: count down; position and velocity frozen.
            BRAKE   — within stopping distance of next station: decelerate
                      at LEAD_SERVICE_DECEL (-0.5 m/s²) until stopped.
            ACCEL   — below cruise speed: accelerate at LEAD_ACCEL (+0.5 m/s²).
            CRUISE  — at cruise speed: coast (a = 0).
            DONE    — past the final station: halt.

        SUVAT (s = u·t + ½·a·t²) is used for position, consistent with the
        AI train.  Arrival snap prevents overshoot accumulation.
        """
        if self.lead_dwell_timer > 0:
            self.lead_dwell_timer -= 1
            return

        next_lead_st = self.lead_station_idx + 1
        if next_lead_st >= len(self.STATIONS):
            # Route complete — stay at final station
            self.lead_v = 0.0
            return

        target_x = self.STATIONS[next_lead_st]
        dist = target_x - self.lead_x

        # Activate any scheduled slowdowns
        remaining = []
        for activate_at, duration, speed in self._scheduled_slowdowns:
            if self.step_count >= activate_at:
                self.lead_slow_until = self.step_count + duration
                self.lead_slow_speed = speed
            else:
                remaining.append((activate_at, duration, speed))
        self._scheduled_slowdowns = remaining

        # Cruise target — reduced during an active slowdown
        cruise_target = (
            self.lead_slow_speed
            if self.step_count < self.lead_slow_until
            else self.lead_train_speed
        )

        # Stopping distance under service braking: s = v² / (2·a)
        stopping_dist = (self.lead_v ** 2) / (2.0 * self.LEAD_SERVICE_DECEL)

        if dist <= stopping_dist:
            lead_a = -self.LEAD_SERVICE_DECEL
        elif self.lead_v < cruise_target:
            lead_a = self.LEAD_ACCEL
        elif self.lead_v > cruise_target:
            lead_a = -self.LEAD_SERVICE_DECEL
        else:
            lead_a = 0.0

        # SUVAT — position then velocity (mirrors AI train integration)
        prev_lead_v = self.lead_v
        self.lead_x += prev_lead_v * self.DT + 0.5 * lead_a * (self.DT ** 2)
        self.lead_v = float(max(0.0, min(
            self.lead_train_speed,
            prev_lead_v + lead_a * self.DT,
        )))

        # Arrival — snap to station coordinate and start dwell
        if self.lead_x >= target_x:
            self.lead_x = target_x
            self.lead_v = 0.0
            self.lead_station_idx = next_lead_st
            self.lead_dwell_timer = (
                int(self.np_random.integers(self.DWELL_MIN, self.DWELL_MAX + 1))
                if self.training_mode
                else self.STATION_DWELL_TIME
            )

    # ------------------------------------------------------------------
    # Lead train slowdown — public API for live-demo use
    # ------------------------------------------------------------------

    def set_lead_slowdown(self, speed: float, duration: int) -> None:
        """Immediately slow the lead train to *speed* m/s for *duration* steps.

        Intended for frontend-triggered demo scenarios.
        Speed is clamped to [0, lead_train_speed].
        """
        self.lead_slow_until = self.step_count + duration
        self.lead_slow_speed = max(0.0, min(float(speed), self.lead_train_speed))

    def _schedule_lead_slowdowns(self) -> None:
        """Pre-schedule 1–2 random slowdowns for the episode (training only)."""
        n = int(self.np_random.integers(1, 3))
        bands = [(300, 1200), (1500, 3000)]
        for i in range(n):
            lo, hi = bands[i % len(bands)]
            activate_at = int(self.np_random.integers(lo, hi + 1))
            duration = int(self.np_random.integers(
                self.SLOWDOWN_DURATION_MIN, self.SLOWDOWN_DURATION_MAX + 1
            ))
            speed = float(self.np_random.uniform(
                self.SLOWDOWN_SPEED_MIN, self.SLOWDOWN_SPEED_MAX
            ))
            self._scheduled_slowdowns.append((activate_at, duration, speed))

    # ------------------------------------------------------------------
    # Landslide control — public API for live-demo use
    # ------------------------------------------------------------------

    def set_landslide(self, position: float) -> int:
        """Place a new active landslide at *position* metres along the track.

        Returns the index of the new landslide so the caller can reference it
        for later toggle/clear operations.

        Raises ValueError if MAX_LANDSLIDES (3) are already placed.
        """
        if len(self.landslides) >= self.MAX_LANDSLIDES:
            raise ValueError(
                f"Maximum {self.MAX_LANDSLIDES} landslides already placed. "
                "Clear one before adding another."
            )
        self.landslides.append(Landslide(position=position, active=True))
        return len(self.landslides) - 1

    def clear_landslide(self, idx: int) -> None:
        """Remove the landslide at *idx* permanently."""
        if idx < 0 or idx >= len(self.landslides):
            raise IndexError(f"No landslide at index {idx}.")
        self.landslides.pop(idx)

    def toggle_landslide(self, idx: int) -> bool:
        """Flip the active state of landslide *idx*.

        Returns the new active state (True = blocking, False = cleared).
        """
        if idx < 0 or idx >= len(self.landslides):
            raise IndexError(f"No landslide at index {idx}.")
        self.landslides[idx].active = not self.landslides[idx].active
        return self.landslides[idx].active

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_dtz(self) -> None:
        """Update Distance-to-Zone (distance to next fixed-block boundary)."""
        self.dtz = self.vl.compute_dtz(self.x)

    def _compute_headway(self) -> float:
        """
        Temporal headway — seconds until the AI train reaches the lead
        train's current position.  Capped at 10 000 s (obs-space high).
        Returns 10 000 when the AI train is stationary.
        """
        if self.v > 0.01:
            return min((self.lead_x - self.x) / self.v, 10_000.0)
        return 10_000.0

    def _build_unified_boundaries(self) -> list[float]:
        """Build a sorted, deduplicated list of every fixed-block boundary
        on the track.  Two consecutive entries define one block.
        Called once in __init__; the result is cached in self._all_boundaries.
        """
        boundaries: set[float] = set()
        for seg in self.vl.segments:
            boundaries.update(seg.block_boundaries)
        return sorted(boundaries)

    def _find_block_index(self, position: float) -> int:
        """Return the index of the block containing *position*.
        Block *i* spans [_all_boundaries[i], _all_boundaries[i+1]).
        Uses binary search for O(log n) performance.
        """
        idx = bisect.bisect_right(self._all_boundaries, position) - 1
        return max(0, min(idx, len(self._all_boundaries) - 2))

    def _get_signal_aspect(self) -> int:
        """
        Derive the 4-aspect ETCS signal using Radio Block Centre (RBC) logic.

        Instead of using raw continuous distance, this simulates an RBC that
        reports which fixed block ahead is occupied.  The aspect is determined
        by counting the number of free (unoccupied) blocks between the AI
        train's current block and the nearest occupied block.

        Two occupancy sources are evaluated; the most restrictive (lowest)
        aspect is returned:

            1. Lead train — occupies the block containing lead_x.
            2. Active landslides — each occupies the block at its position.
               Landslides behind the AI train are ignored.

        Aspect mapping (number of free blocks between AI and obstacle):
            3+ free blocks  →  Green  (3)
            2 free blocks   →  Double-Yellow (2)
            1 free block    →  Yellow (1)
            0 free blocks   →  Red    (0)
        """
        ai_block = self._find_block_index(self.x)

        # Lead train occupancy
        lead_block = self._find_block_index(self.lead_x)
        free_blocks = lead_block - ai_block - 1
        aspect = max(0, min(3, free_blocks))

        # Landslide occupancy — each active landslide ahead is a virtual
        # stopped train; take the most restrictive signal.
        for ls in self.landslides:
            if ls.active and ls.position > self.x:
                ls_block = self._find_block_index(ls.position)
                ls_free = ls_block - ai_block - 1
                aspect = min(aspect, max(0, min(3, ls_free)))

        return aspect

    def _get_obs(self) -> np.ndarray:
        """Build the 7-dimensional observation vector."""
        aspect = self._get_signal_aspect()
        seg = self.vl.get_segment(self.x)
        next_st_pos = self.STATIONS[
            min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        ]
        dist_to_next_station = max(0.0, next_st_pos - self.x)
        return np.array(
            [self.x, self.v, self.dtz, float(aspect),
             dist_to_next_station, seg.limit_ms, self._compute_headway()],
            dtype=np.float32,
        )
