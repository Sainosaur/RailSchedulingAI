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
    # 1. Progress & Station Rewards (The "Carrots")
    k_p: float = 2000.0                      # incremental progress reward scale
    station_reward_base: float = 500.0        # base milestone for station arrival
    station_escalation: float = 0.3           
    
    # 2. Speed Rewards (The "Driver")
    # Overspeed: linear penalty for exceeding the target speed.
    # Underspeed: replaced with a ratio-based bonus that rewards *approaching*
    # the target speed rather than punishing the current gap.
    # This eliminates the -25/step trap at v=0 that caused learned helplessness.
    k_over: float = 1.0                       
    k_speed_bonus: float = 5.0                # Positive reward scaled by v/v_target ratio
    speed_penalty_cap: float = -100.0         # Cap for overspeed penalty
    
    # 3. Moving Incentives (The "Anti-Granny" kick)
    # Rewards pushing the throttle hard when on Green.
    signal_compliance_bonus: float = 1.0      
    k_stall: float = 3.0                      # Reduced from 15.0 — old value created a binary cliff
    k_lazy: float = 5.0                       # Penalty for slow acceleration

    # 4. Punctuality & Temporal (The "Schedule")
    punctuality_factor: float = 0.5           
    punctuality_tolerance: float = 60.0       
    punctuality_penalty_cap: float = -1000.0  # Cap the punctuality "Black Hole"

    # 5. Continuous Taxes (The "Anti-Cowardice" clock)
    existence_penalty: float = -0.1           # Tax per step (Forces movement)
    heartbeat_penalty: float = -0.5           # Forces movement at Green signals

    # 6. Safety & Operational (The "Guardrails")
    override_penalty: float = -25.0            # Low cost for triggering VL (to prevent cowardice)
    jerk_penalty: float = -0.1                
    k_jerk: float = 2.0
    comfortable_acceleration: float = 0.5     
    comfortable_deceleration: float = 0.5     
    energy_penalty_weight: float = 0.0        

    # 7. Headway & Collision (The "Disasters")
    k_h: float = 15.0                          
    headway_warning_multiplier: float = 3.0   
    headway_violation_multiplier: float = 1.0 
    headway_violation_penalty: float = -500.0
    collision_penalty: float = -1000.0

    # 8. Training mode — softens terminal penalties for longer exploration
    training_mode: bool = False

DEFAULT_CONFIG = RewardConfig()

@dataclass
class TrainState:
    """All variables required to compute the reward at one timestep."""

    # position
    current_position: float  # meters along the track
    previous_position: float  # psition at previous timestamp t-1
    last_station_position: float  # position of the last station passed
    next_station_position: float  # position of the next station ahead
    distance_to_occupied: float   # distance to the obstruction ahead (m)

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
    """Monotonically increasing penalty as headway shrinks toward the violation
    threshold.  Scaled from 0 at the warning threshold to -k_h at the violation
    threshold.
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


def _compute_speed_reward(state: TrainState, config: RewardConfig) -> float:
    """Speed shaping: penalise overspeed, reward approaching the target.
    """
    v: float = state.current_speed
    v_max: float = state.speed_limit
    
    d_brake: float = min(state.next_station_position - state.current_position, state.distance_to_occupied)
    
    # v_target is the min of the speed limit and the square-root braking curve
    v_brake = math.sqrt(2 * config.comfortable_deceleration * max(0.0, d_brake))
    v_target: float = min(v_max, v_brake)
    
    overspeed = max(0.0, v - v_target)

    # Unified Physics Tracking Reward:
    speed_error = abs(v - v_target)
    tracking_ratio = max(0.0, 1.0 - (speed_error / v_max)) if v_max > 0 else 1.0
    speed_bonus = config.k_speed_bonus * tracking_ratio

    return -(config.k_over * overspeed) + speed_bonus


def _compute_energy_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize purely positive acceleration applications (traction). Coasting and braking are free.
    return config.energy_penalty_weight * state.applied_traction

def _compute_heartbeat_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.heartbeat_penalty

def _compute_override_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.override_penalty if state.overridden else 0.0

def _compute_jerk_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize the magnitude of the action shift squared
    return config.jerk_penalty * (state.action_delta ** 2)

def _compute_signal_compliance_reward(state: TrainState, config: RewardConfig) -> float:
    # Deprecated: The unified speed tracking reward perfectly handles all 
    # signal and physics states natively. 
    return 0.0

def _compute_station_reward(state: TrainState, config: RewardConfig) -> float:
    # Use station_reward_base for arrivals
    return config.station_reward_base if state.reached_new_station else 0.0 
 
def _compute_punctuality_penalty(state: TrainState, config: RewardConfig) -> float:
    """Penalty for arriving at a station outside the on-time tolerance window.
    """
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
        terminate = not config.training_mode
        return config.headway_violation_penalty, terminate
    
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
    r_energy    = _compute_energy_penalty(state, config)
 
    # --- Terminal ---
    r_violation, terminate_violation = _compute_headway_violation(state, config)
    r_collision, terminate_collision = _compute_collision_penalty(state, config)
 
    # Aggregates
    r_continuous = r_progress + r_headway + r_speed + r_heartbeat
    r_event      = r_station + r_time + r_override + r_jerk + r_energy
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
        r_energy=r_energy,
        r_violation=r_violation,
        r_collision=r_collision,
        r_continuous=r_continuous,
        r_event=r_event,
        r_terminal=r_terminal,
        r_total=r_total,
        terminate=terminate)

if __name__ == "__main__":
    # Typical mid-journey timestep
    state = TrainState(
        current_position=1500.0,
        previous_position=1450.0,
        last_station_position=1000.0,
        next_station_position=2000.0,
        distance_to_occupied=500.0,
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