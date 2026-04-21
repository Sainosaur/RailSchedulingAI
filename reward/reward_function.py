"""
Railway RL Reward Function

Structure:
  - Continuous rewards  (every timestep)
  - Event rewards       (on specific events)
  - Terminal penalties  (once, ends episode)

Total: r_t = r_continuous + r_event + r_terminal
"""

from dataclasses import dataclass
from typing import Optional
import math

# State container

@dataclass
class RewardConfig:
    """Hyperparameters for the reward function."""
    k_p: float = 0.1  # per-metre progress reward scale (≈ 0.0015 reward per metre → ~1.5/km)
    station_reward: float = 25.0
    station_overshoot_penalty: float = -25.0  # penalty for blasting through station at speed
    station_arrival_speed_limit: float = 2.0  # m/s — max speed to qualify for station reward
    punctuality_factor: float = 0.5       # penalty per second *outside* tolerance
    punctuality_tolerance: float = 60.0   # seconds, "on time" window

    k_h: float = 3.0  # HEADWAY_WARNING_SCALE
    headway_warning_multiplier: float = 3.0  # e.g., 3x TH starts warning zone
    headway_violation_multiplier: float = 1.0  # e.g., 1x TH is hard safety limit

    k_vel: float = 0.5   # velocity bonus weight (continuous positive reward for speed)
    k_over: float = 0.05  # overspeed penalty weight (quadratic, vs. segment limit only)
    k_under: float = 0.0  # underspeed penalty weight — disabled: progress + velocity bonus carry departure signal

    # Kinematic Comfort Limits
    comfortable_acceleration: float = 0.5  # m/s²
    comfortable_deceleration: float = 0.5  # m/s²

    # Behavioural and Operational penalties
    heartbeat_penalty: float = -0.05          # per-step cost of doing nothing
    override_penalty: float = 0.0             # overrides are correct safety behaviour, not an agent failure
    # jerk penalty must be small: early Gaussian exploration naturally swings
    # action by ~1.5 per step, so -0.5 × 1.5² = -1.125/step (~-600/rollout)
    # teaches PPO to pick a constant action — which collapses to full brake.
    jerk_penalty: float = -0.01               # penalty for flip-flopping actions abruptly
    energy_penalty_weight: float = -0.05      # penalty for positive traction use

    # Safety costs — non-terminal so the agent can recover and learn.
    headway_violation_penalty: float = -10.0
    collision_penalty: float = -20.0

DEFAULT_CONFIG = RewardConfig()

@dataclass
class TrainState:
    """All variables required to compute the reward at one timestep."""

    # position
    current_position: float  # meters along the track
    previous_position: float  # psition at previous timestamp t-1
    last_station_position: float  # position of the last station passed
    next_station_position: float  # position of the next station ahead

    # speed
    current_speed: float  # m/s
    speed_limit: float  # m/s (track speed limit)

    # Safety
    headway: float  # seconds to the train ahead

    # Timetable / events
    reached_new_station: bool  # True only in the timestep of arrival
    scheduled_arrival_time: Optional[float] = None  # seconds
    actual_arrival_time: Optional[float] = None  # seconds

    # Safety events
    collision: bool = False

    # Default parameters that must follow non-defaults (to fix dataclass syntax error)
    temporal_headway: float = 12.5  # default to highest TH
    
    # Validation / Smoothing (Defaults set to safe values so the environment won't break until we integrate them)
    overridden: bool = False
    # BUG 11/21 FIX: action_delta is the absolute difference between two
    # continuous accelerations — a float in [0.0, 1.5], not a discrete int.
    action_delta: float = 0.0  # |safe_a - last_a|  (m/s²)
    applied_traction: float = 0.0  # Positive throttle applied by agent (m/s²), defaults to 0
    dwelling: bool = False  # True when the environment is enforcing a station dwell
    arrival_speed: Optional[float] = None  # speed when crossing station (before snap to 0)


# Reward components


def _compute_progress_reward(state: TrainState, config: RewardConfig) -> float:
    # Per-metre progress reward — uniform across spans of wildly different lengths
    # (5 km vs. 22 km) so the per-step signal is comparable throughout the line.
    delta_m = max(0.0, state.current_position - state.previous_position)
    return config.k_p * delta_m


def _compute_headway_penalty(state: TrainState, config: RewardConfig) -> float:
    """Monotonically increasing penalty as headway shrinks toward the violation
    threshold.  Scaled from 0 at the warning threshold to -k_h at the violation
    threshold.

    BUG 8 FIX: the previous formula used a downward-opening parabola
    (-k_h * h * (warning_thresh - h) / 1000) which was *maximally* punishing
    at the midpoint of the warning zone and got weaker as the train approached
    the danger zone — exactly backwards.  Replaced with a linear ramp:
        t = 0  at warning_thresh (no penalty)
        t = 1  at violation_thresh (full penalty = -k_h)
    """
    warning_thresh = state.temporal_headway * config.headway_warning_multiplier
    violation_thresh = state.temporal_headway * config.headway_violation_multiplier
    h = state.headway

    if violation_thresh < h < warning_thresh:
        zone_width = warning_thresh - violation_thresh
        if zone_width <= 0:
            return 0.0
        t = (warning_thresh - h) / zone_width  # 0 → 1 as h drops to violation_thresh
        return -config.k_h * t
    return 0.0


def _compute_velocity_bonus(state: TrainState, config: RewardConfig) -> float:
    """Continuous positive reward proportional to speed / speed_limit.

    This replaces the underspeed penalty with a direct incentive:
    faster = more reward.  Suppressed during environment-enforced
    station dwells so the agent isn't punished for physics it can't control.
    """
    if state.dwelling or state.speed_limit <= 0:
        return 0.0
    return config.k_vel * (state.current_speed / state.speed_limit)


def _compute_speed_reward(state: TrainState, config: RewardConfig) -> float:
    # Penalise only overspeed above the posted segment limit.  The earlier
    # kinematic v_accel/v_brake profile created a catastrophic trap: post-
    # station, d_from resets to 0 so v_accel ≈ 0 and any cruising speed
    # looks like massive overspeed.  Safe braking into a station is
    # enforced by the Validation Layer and by the collision/headway
    # penalties — the reward does not need to re-encode it.
    v: float = state.current_speed
    v_max: float = state.speed_limit

    overspeed: float = max(0.0, v - v_max)
    underspeed: float = max(0.0, v_max - v)  # only used if k_under > 0

    return -(config.k_over * overspeed**2) - (config.k_under * underspeed)

def _compute_energy_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize purely positive acceleration applications (traction). Coasting and braking are free.
    return config.energy_penalty_weight * state.applied_traction

def _compute_heartbeat_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.heartbeat_penalty

def _compute_override_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.override_penalty if state.overridden else 0.0

def _compute_jerk_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize the magnitude of the action shift squared (e.g., jump of 1 = -20, jump of 3 = -180)
    return config.jerk_penalty * (state.action_delta ** 2)

def _compute_station_reward(state: TrainState, config: RewardConfig) -> float:
    """Station arrival reward — conditional on arrival speed.
    
    +station_reward  if the train arrives slowly (< station_arrival_speed_limit)
    +overshoot_penalty if the train blasts through at high speed
    """
    if not state.reached_new_station:
        return 0.0
    # Use arrival_speed (pre-snap) if available, otherwise current_speed
    check_speed = state.arrival_speed if state.arrival_speed is not None else state.current_speed
    if check_speed <= config.station_arrival_speed_limit:
        return config.station_reward
    return config.station_overshoot_penalty
 
def _compute_punctuality_penalty(state: TrainState, config: RewardConfig) -> float:
    """Penalty for arriving at a station outside the on-time tolerance window.

    BUG 13 FIX: previously always returned 0 because scheduled/actual arrival
    times were never set by the environment.  The environment now passes
    actual_arrival_time=self.time and scheduled_arrival_time computed from
    the expected journey time at lead_train_speed. 
    """
    if not state.reached_new_station:
        return 0.0
    if state.scheduled_arrival_time is None or state.actual_arrival_time is None:
        return 0.0

    deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)
    excess = max(0.0, deviation - config.punctuality_tolerance)
    return -config.punctuality_factor * excess

def _compute_headway_violation(state: TrainState, config: RewardConfig) -> tuple[float, bool]:
    # Non-terminal: apply the cost but let the episode continue so the agent
    # can experience recovery. Episode only truncates on MAX_STEPS or TRACK_END.
    violation_thresh = state.temporal_headway * config.headway_violation_multiplier
    if state.headway <= violation_thresh:
        return config.headway_violation_penalty, False
    return 0.0, False

def _compute_collision_penalty(state: TrainState, config: RewardConfig) -> tuple[float, bool]:
    # Non-terminal: see _compute_headway_violation note.
    if state.collision:
        return config.collision_penalty, False
    return 0.0, False

# Main reward function

@dataclass
class RewardOutput:
    """Structured reward breakdown for easy debugging."""
    # Continuous
    r_progress: float
    r_headway: float
    r_speed: float
    r_heartbeat: float
    r_velocity: float
    # Event
    r_station: float
    r_time: float
    r_override: float
    r_jerk: float
    r_energy: float
    # Terminal
    r_violation: float
    r_collision: float
    # Aggregate
    r_continuous: float
    r_event: float
    r_terminal: float
    r_total: float
    # Episode control
    terminate: bool

def compute_reward(state: TrainState, config: RewardConfig = DEFAULT_CONFIG) -> RewardOutput:
    """
    Compute the full reward for one timestep.
 
    r_t = r_continuous + r_event + r_terminal
 
    Also returns a `terminate` flag that the environment loop should
    check to end the episode immediately.
    """
    # --- Continuous ---
    r_progress  = _compute_progress_reward(state, config)
    r_headway   = _compute_headway_penalty(state, config)
    r_speed     = _compute_speed_reward(state, config)
    r_heartbeat = _compute_heartbeat_penalty(state, config)
    r_velocity  = _compute_velocity_bonus(state, config)
 
    # --- Event ---
    r_station   = _compute_station_reward(state, config)
    r_time      = _compute_punctuality_penalty(state, config)
    r_override  = _compute_override_penalty(state, config)
    r_jerk      = _compute_jerk_penalty(state, config)
    r_energy    = _compute_energy_penalty(state, config)
 
    # --- Terminal ---
    r_violation, terminate_violation = _compute_headway_violation(state, config)
    r_collision, terminate_collision = _compute_collision_penalty(state, config)
 
    # Aggregates
    r_continuous = r_progress + r_headway + r_speed + r_heartbeat + r_velocity
    r_event      = r_station + r_time + r_override + r_jerk + r_energy
    r_terminal   = r_violation + r_collision
    r_total      = r_continuous + r_event + r_terminal
    terminate    = terminate_violation or terminate_collision
 
    return RewardOutput(
        r_progress=r_progress,
        r_headway=r_headway,
        r_speed=r_speed,
        r_heartbeat=r_heartbeat,
        r_velocity=r_velocity,
        r_station=r_station,
        r_time=r_time,
        r_override=r_override,
        r_jerk=r_jerk,
        r_energy=r_energy,
        r_violation=r_violation,
        r_collision=r_collision,
        r_continuous=r_continuous,
        r_event=r_event,
        r_terminal=r_terminal,
        r_total=r_total,
        terminate=terminate)



# Example usage

 
if __name__ == "__main__":
    # Typical mid-journey timestep
    state = TrainState(
        current_position=1500.0,
        previous_position=1450.0,
        last_station_position=1000.0,
        next_station_position=2000.0,
        current_speed=30.0,
        speed_limit=30.0,
        headway=15.0, # e.g. 15s headway
        temporal_headway=12.5,
        reached_new_station=False,
    )
 
    result = compute_reward(state)
 
    print("=== Reward Breakdown ===")
    print(f"  Progress       : {result.r_progress:+.4f}")
    print(f"  Headway        : {result.r_headway:+.4f}")
    print(f"  Speed          : {result.r_speed:+.4f}")
    print(f"  Heartbeat      : {result.r_heartbeat:+.4f}")
    print(f"  Station        : {result.r_station:+.4f}")
    print(f"  Punctuality    : {result.r_time:+.4f}")
    print(f"  Override       : {result.r_override:+.4f}")
    print(f"  Jerk           : {result.r_jerk:+.4f}")
    print(f"  Energy         : {result.r_energy:+.4f}")
    print(f"  HW Violation   : {result.r_violation:+.4f}")
    print(f"  Collision      : {result.r_collision:+.4f}")
    print("------------------------")
    print(f"  Continuous     : {result.r_continuous:+.4f}")
    print(f"  Event          : {result.r_event:+.4f}")
    print(f"  Terminal       : {result.r_terminal:+.4f}")
    print(f"  TOTAL          : {result.r_total:+.4f}")
    print(f"  Terminate      : {result.terminate}")