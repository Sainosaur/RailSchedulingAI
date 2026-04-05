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

# State container

@dataclass
class RewardConfig:
    """Hyperparameters for the reward function."""
    k_p: float = 5.0 # incremental progress reward scale
    station_reward: float = 5.0
    punctuality_factor: float = 0.5       # penalty per second *outside* tolerance
    punctuality_tolerance: float = 60.0   # seconds, "on time" window
    
    k_h: float = 3.0  # HEADWAY_WARNING_SCALE
    headway_warning_multiplier: float = 3.0  # e.g., 3x TH starts warning zone
    headway_violation_multiplier: float = 1.0  # e.g., 1x TH is hard safety limit
    
    k_over: float = 0.5  # overspeed penalty weight  (quadratic)
    k_under: float = 0.0  # underspeed penalty weight (linear, default 0)
    
    # New addition: Behavioral and Operational penalties
    heartbeat_penalty: float = -0.1          # penalty applied every step to prevent stalling
    override_penalty: float = -500.0         # penalty when the Validation Layer intervenes
    jerk_penalty: float = -20.0               # penalty for flip-flopping actions abruptly
    
    headway_violation_penalty: float = -150.0
    collision_penalty: float = -200.0

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
    action_delta: int = 0  # Magnitude of change (e.g. 0 to 3)

    # Buffer distances for speed profile
    acceleration_buffer: float = 200.0  # D_a  (metres) — tune per prototype
    braking_buffer: float = 300.0  # D_b  (metres) — tune per prototype


# Reward components


def _compute_progress_reward(state: TrainState, config: RewardConfig) -> float:
    
    span = state.next_station_position - state.last_station_position
    if span <= 0:
        return 0.0

    p_current = (state.current_position - state.last_station_position) / span
    p_previous = (state.previous_position - state.last_station_position) / span

    p_current = max(0.0, min(1.0, p_current))  # clamp to [0,1]
    p_previous = max(0.0, min(1.0, p_previous))  # clamp to [0,1]

    return config.k_p * (p_current - p_previous)


def _compute_headway_penalty(state: TrainState, config: RewardConfig) -> float:
    
    warning_thresh = state.temporal_headway * config.headway_warning_multiplier
    violation_thresh = state.temporal_headway * config.headway_violation_multiplier
    h = state.headway
    
    if violation_thresh < h < warning_thresh:
        return -config.k_h * (h * (warning_thresh - h)) / 1000.0
    return 0.0


def _compute_speed_reward(state: TrainState, config: RewardConfig) -> float:
    
    v: float = state.current_speed
    v_max: float = state.speed_limit
    d_from: float = max(0.0, state.current_position - state.last_station_position)
    d_to: float = max(0.0, state.next_station_position - state.current_position)
    D_a: float = state.acceleration_buffer
    D_b: float = state.braking_buffer

    v_target: float = max(0.0,min(v_max,
                                  v_max * (d_from / D_a) if D_a > 0 else v_max,
                                  v_max * (d_to / D_b) if D_b > 0 else v_max))
    
    overspeed: float = max(0.0, v - v_target)
    underspeed: float = max(0.0, v_target - v)

    return -(config.k_over * overspeed**2) - (config.k_under * underspeed)

def _compute_heartbeat_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.heartbeat_penalty

def _compute_override_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.override_penalty if state.overridden else 0.0

def _compute_jerk_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize the magnitude of the action shift squared (e.g., jump of 1 = -20, jump of 3 = -180)
    return config.jerk_penalty * (state.action_delta ** 2)

def _compute_station_reward(state: TrainState, config: RewardConfig) -> float:
    
    return config.station_reward if state.reached_new_station else 0.0 
 
def _compute_punctuality_penalty(state: TrainState, config: RewardConfig) -> float:
    
    if not state.reached_new_station:
        return 0.0
    if state.scheduled_arrival_time is None or state.actual_arrival_time is None:
        return 0.0
    
    deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)
    excess = max(0.0, deviation - config.punctuality_tolerance)
    return -config.punctuality_factor * excess

def _compute_headway_violation(state: TrainState, config: RewardConfig) -> tuple[float, bool]:

    violation_thresh = state.temporal_headway * config.headway_violation_multiplier
    if state.headway <= violation_thresh:
        return config.headway_violation_penalty, True
    
    return 0.0, False

def _compute_collision_penalty(state: TrainState, config: RewardConfig) -> tuple[float, bool]:
   
    if state.collision:
        return config.collision_penalty, True
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
    # Event
    r_station: float
    r_time: float
    r_override: float
    r_jerk: float
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
 
    # --- Event ---
    r_station   = _compute_station_reward(state, config)
    r_time      = _compute_punctuality_penalty(state, config)
    r_override  = _compute_override_penalty(state, config)
    r_jerk      = _compute_jerk_penalty(state, config)
 
    # --- Terminal ---
    r_violation, terminate_violation = _compute_headway_violation(state, config)
    r_collision, terminate_collision = _compute_collision_penalty(state, config)
 
    # Aggregates
    r_continuous = r_progress + r_headway + r_speed + r_heartbeat
    r_event      = r_station + r_time + r_override + r_jerk
    r_terminal   = r_violation + r_collision
    r_total      = r_continuous + r_event + r_terminal
    terminate    = terminate_violation or terminate_collision
 
    return RewardOutput(
        r_progress=r_progress,
        r_headway=r_headway,
        r_speed=r_speed,
        r_heartbeat=r_heartbeat,
        r_station=r_station,
        r_time=r_time,
        r_override=r_override,
        r_jerk=r_jerk,
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
        acceleration_buffer=200.0,
        braking_buffer=300.0,
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
    print(f"  HW Violation   : {result.r_violation:+.4f}")
    print(f"  Collision      : {result.r_collision:+.4f}")
    print("------------------------")
    print(f"  Continuous     : {result.r_continuous:+.4f}")
    print(f"  Event          : {result.r_event:+.4f}")
    print(f"  Terminal       : {result.r_terminal:+.4f}")
    print(f"  TOTAL          : {result.r_total:+.4f}")
    print(f"  Terminate      : {result.terminate}")