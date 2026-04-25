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
    """Hyperparameters for the reward function — elimination-diet baseline.

    Only the signals needed to teach the agent to move and survive are active.
    Reintroduce one component at a time once the base behaviour is stable.

    Mathematical basis
    ------------------
    Perfect episode: ~3 800 steps at 20 m/s average across 76 km.
    Progress total  = k_p * 6 segments = 20 * 6 = 120.
    Heartbeat fires only while stationary (v < 0.5 m/s); penalty per
    stall-step = -0.5.  Moving earns +0.04/step; sitting costs -0.5/step
    → 12.5:1 gradient favouring movement.
    Velocity bonus = +0.05/step at speed limit, 0 at standstill.
    Collision = -100 (≈ 83% of a perfect run): very painful, recoverable.
    Violation = -50  (≈ 42% of a perfect run): serious, survivable.
    """

    # --- Active ---
    k_p: float = 20.0  # progress reward scale (segment-span fraction × k_p)
    heartbeat_penalty: float = -0.5   # fires only when v < 0.5 m/s
    headway_violation_penalty: float = -50.0   # non-terminal
    collision_penalty: float = -100.0          # non-terminal

    k_h: float = 1.0   # headway warning ramp: 0 at 3×TH → -1.0/step at 1×TH
    k_vel: float = 0.1   # velocity reward: +k_vel at speed limit, 0 at standstill (green only)
    headway_warning_multiplier: float = 3.0
    headway_violation_multiplier: float = 1.0

    # --- Station stopping ---
    station_reward: float = 100.0             # one-shot on clean arrival
    station_overshoot_penalty: float = -25.0 # one-shot on overspeed pass-through
    station_arrival_speed_limit: float = 0.5  # m/s — must arrive nearly stopped
    station_approach_zone: float = 500.0      # metres — taper + braking shaping zone
    k_brake: float = 0.2   # kinematic braking shaping: penalises v > v_stop inside zone
    max_decel: float = 1.0  # m/s² — physical max braking (mirrors environment DECEL)
    dwell_reward: float = 0.75  # per-step reward while dwelling at a station
    approach_heartbeat_scale: float = 0.3  # heartbeat inside zone = penalty × this (reduced, not zero)

    # --- Zeroed (reintroduce one at a time) ---
    punctuality_factor: float = 0.0
    punctuality_tolerance: float = 60.0

    k_over: float = 0.0   # overspeed penalty
    k_under: float = 0.0  # underspeed penalty

    comfortable_acceleration: float = 0.5  # m/s²
    comfortable_deceleration: float = 0.5  # m/s²

    override_penalty: float = 0.0
    jerk_penalty: float = 0.0
    energy_penalty_weight: float = 0.0

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
    signal_aspect: int = 3  # 0=Red, 1=Yellow, 2=Double-Yellow, 3=Green
    
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
    # Segment-span fractional progress scaled by signal aspect (0–3).
    # aspect/3 multiplier: Green=1.0, DblYellow=0.67, Yellow=0.33, Red=0.0
    # This creates a speed gradient — slower is fine on yellow, stop on red.
    aspect_scale = state.signal_aspect / 3.0
    if aspect_scale == 0.0:
        return 0.0

    span = state.next_station_position - state.last_station_position
    if span <= 0:
        return 0.0
    p_current = (state.current_position - state.last_station_position) / span
    p_previous = (state.previous_position - state.last_station_position) / span
    p_current = max(0.0, min(1.0, p_current))
    p_previous = max(0.0, min(1.0, p_previous))
    return config.k_p * aspect_scale * (p_current - p_previous)


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
        # Scale penalty by signal aspect — amber signals carry less braking
        # pressure so the agent can still move at reduced speed, while red
        # keeps the full penalty to reinforce stopping.
        #   Red (0): 1.00 × k_h  — full penalty
        #   Yellow (1): 0.05 × k_h  — gentle, ~15 km/h equilibrium
        #   Dbl Yellow (2): 0.10 × k_h  — moderate, ~30 km/h equilibrium
        #   Green (3): 1.00 × k_h  — full (agent is never in zone on green)
        aspect_penalty_scale = [1.0, 0.05, 0.10, 1.0][state.signal_aspect]
        return -config.k_h * t * aspect_penalty_scale
    return 0.0


def _compute_velocity_bonus(state: TrainState, config: RewardConfig) -> float:
    """Continuous positive reward proportional to speed / speed_limit.

    Scaled by signal aspect so the agent has a reason to keep moving on
    yellow/double-yellow, just not at full speed:
        Green (3):        1.00 × k_vel  (+0.100/step at limit)
        Double Yellow (2): 0.75 × k_vel  (+0.075/step at limit)
        Yellow (1):        0.50 × k_vel  (+0.050/step at limit)
        Red (0):           0.00          (no incentive to move)

    Tapers linearly to zero inside the station approach zone so there is a
    passive incentive to decelerate before reaching the platform.
    """
    if state.speed_limit <= 0:
        return 0.0
    vel_scale = [0.0, 0.5, 0.75, 1.0][state.signal_aspect]
    if vel_scale == 0.0:
        return 0.0
    dist_to_station = state.next_station_position - state.current_position
    # Taper from 1.0 outside zone to 0.5 at the platform — agent still has
    # reason to move, just not at full speed.
    zone_frac = min(1.0, max(0.0, dist_to_station / config.station_approach_zone))
    approach_scale = 0.5 + 0.5 * zone_frac
    return config.k_vel * vel_scale * approach_scale * min(1.0, state.current_speed / state.speed_limit)


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
    # Only penalise stalling on a Green signal — on yellow/red the agent
    # should be free to stop without being punished for it.
    # Suppressed in the approach zone so the agent isn't penalised for braking
    # toward the station on an otherwise-green line.
    if state.signal_aspect != 3:
        return 0.0
    if state.current_speed >= 0.5:
        return 0.0
    dist_to_station = state.next_station_position - state.current_position
    if dist_to_station < config.station_approach_zone:
        # Reduced penalty inside approach zone — still pushes agent to move
        # but doesn't overwhelm the braking gradient.
        return config.heartbeat_penalty * config.approach_heartbeat_scale
    return config.heartbeat_penalty

def _compute_braking_reward(state: TrainState, config: RewardConfig) -> float:
    """Dense kinematic shaping inside the station approach zone.

    Computes v_stop = sqrt(2 × max_decel × dist) — the maximum speed that
    still allows a full stop at the station.  Any excess above v_stop is
    penalised quadratically, giving the agent a dense per-step gradient that
    grows as it falls further behind the ideal braking curve.
    """
    dist_to_station = state.next_station_position - state.current_position
    if dist_to_station >= config.station_approach_zone or dist_to_station <= 0:
        return 0.0
    v_stop = math.sqrt(2.0 * config.max_decel * dist_to_station)
    excess = state.current_speed - v_stop
    if excess <= 0:
        return 0.0
    return -config.k_brake * excess ** 2

def _compute_dwell_reward(state: TrainState, config: RewardConfig) -> float:
    return config.dwell_reward if state.dwelling else 0.0

def _compute_override_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.override_penalty if state.overridden else 0.0

def _compute_jerk_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize the magnitude of the action shift squared (e.g., jump of 1 = -20, jump of 3 = -180)
    return config.jerk_penalty * (state.action_delta ** 2)

def _compute_station_reward(state: TrainState, config: RewardConfig) -> float:
    """Station arrival reward — conditional on arrival speed.

    Provides a gradient of rewards to guide the agent to a clean stop.
    """
    if not state.reached_new_station:
        return 0.0
    # Use arrival_speed (pre-snap) if available, otherwise current_speed
    check_speed = state.arrival_speed if state.arrival_speed is not None else state.current_speed

    if check_speed <= config.station_arrival_speed_limit:
        return config.station_reward
    elif check_speed <= 2.0:
        return 10.0
    elif check_speed <= 5.0:
        return -5.0
    else:
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
    r_brake: float
    r_dwell: float
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
    r_brake     = _compute_braking_reward(state, config)
    r_dwell     = _compute_dwell_reward(state, config)

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
    r_continuous = r_progress + r_headway + r_speed + r_heartbeat + r_velocity + r_brake + r_dwell
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
        r_brake=r_brake,
        r_dwell=r_dwell,
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
    print(f"  Velocity       : {result.r_velocity:+.4f}")
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