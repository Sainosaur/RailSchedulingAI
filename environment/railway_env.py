"""
environment/railway_env.py

Gymnasium environment for a single train traversing Polish Rail Line 104
(Chabówka → Nowy Sącz, ~76 km).

Convoy model: one AI-controlled train follows a lead train that cruises
at a configurable constant speed.  The agent chooses a 4-aspect signal
action each timestep; the Validation Layer ensures safety before the
physics update runs.
"""

import sys
from pathlib import Path

import gymnasium as gym
import numpy as np

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from validate_layer.validator import ValidationLayer   # noqa: E402
from reward.reward_function import compute_reward, TrainState  # noqa: E402


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

    # --- Physics ---
    DT: float = 1.0      # timestep  (seconds)
    ACCEL: float = 0.5    # traction acceleration  (m/s²)
    DECEL: float = -1.0   # service braking  (m/s²)

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
    ):
        super().__init__()
        self.render_mode = render_mode
        self.lead_train_speed = lead_train_speed

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
        self.time: float = 0.0
        self.last_station_idx: int = 0
        self.visited_stations: set[int] = set()
        self.last_a: float = 0.0  # Track the last *executed* acceleration for jerk calculation
        self.step_count: int = 0

    # ------------------------------------------------------------------
    # Gymnasium API
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.x = self.TRACK_START
        self.v = 0.0
        self.time = 0.0
        self.last_station_idx = 0
        self.visited_stations = {0}  # starting station already "visited"
        self.last_a = 0.0
        self.step_count = 0

        # Lead train begins ~2 km ahead
        self.lead_x = self.TRACK_START + 2000.0
        self._update_dtz()

        return self._get_obs(), {}

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

        # ----- 3. Lead train moves forward -----
        self.lead_x += self.lead_train_speed * self.DT
        self._update_dtz()

        # ----- 4. Station arrival check -----
        reached_new_station = False
        next_st_idx = self.last_station_idx + 1
        if next_st_idx < len(self.STATIONS):
            if self.x >= self.STATIONS[next_st_idx]:
                reached_new_station = True
                self.last_station_idx = next_st_idx
                self.visited_stations.add(next_st_idx)

        # ----- 5. Reward computation -----
        last_st_pos = self.STATIONS[self.last_station_idx]
        next_st_pos = self.STATIONS[
            min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        ]

        # Temporal headway: time gap to the lead train (seconds)
        headway = (
            (self.lead_x - self.x) / self.v
            if self.v > 0.01 else 9999.0
        )

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

    def _get_signal_aspect(self) -> int:
        """
        Derive the 4-aspect signal from the distance to the lead train
        and the current segment's spatial headway (SH).

        The lead train's distance determines how many blocks ahead are
        clear.  SH is the block length:
            Green  (3) — next 3 blocks clear    (dist > 3 × SH)
            DblYlw (2) — next 2 blocks clear    (dist > 2 × SH)
            Yellow (1) — next 1 block  clear     (dist > 1 × SH)
            Red    (0) — next block occupied     (dist ≤ 1 × SH)
        """
        dist_to_lead = self.lead_x - self.x
        sh = self.vl.get_segment(self.x).spatial_headway

        if dist_to_lead > 3 * sh:
            return 3
        if dist_to_lead > 2 * sh:
            return 2
        if dist_to_lead > sh:
            return 1
        return 0

    def _get_obs(self) -> np.ndarray:
        """Build the observation vector (7 values)."""
        aspect = self._get_signal_aspect()
        seg = self.vl.get_segment(self.x)
        next_st_pos = self.STATIONS[
            min(self.last_station_idx + 1, len(self.STATIONS) - 1)
        ]
        dist_to_next_station = max(0.0, next_st_pos - self.x)
        headway = (
            (self.lead_x - self.x) / self.v
            if self.v > 0.01 else 9999.0
        )
        return np.array(
            [self.x, self.v, self.dtz, float(aspect),
             dist_to_next_station, seg.limit_ms, min(headway, 9999.0)],
            dtype=np.float32,
        )


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

    env.render()
    print(f"\nTotal reward: {total_r:+.2f}")
    print(f"Stations visited: {sorted(env.visited_stations)}")
