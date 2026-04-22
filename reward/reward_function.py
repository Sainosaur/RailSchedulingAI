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
    k_p: float = 2000.0                      # incremental progress reward scale
    station_reward_base: float = 5000.0       # base milestone for station arrival
    station_escalation: float = 0.3           
    punctuality_factor: float = 0.5           
    punctuality_tolerance: float = 60.0       

    k_h: float = 3.0                          
    headway_warning_multiplier: float = 3.0   
    headway_violation_multiplier: float = 1.0 

    k_over: float = 0.5                       
    k_under: float = 0.01                     
    speed_penalty_cap: float = -2.0           

    # Kinematic Comfort Limits
    comfortable_acceleration: float = 0.5     
    comfortable_deceleration: float = 0.5     

    # --- THE "ANTI-COWARDICE" TWEAKS ---
    existence_penalty: float = -0.1           # Constant tax per step (Forces movement)
    heartbeat_penalty: float = -0.5           # Increased 50x (Forces movement at Green)
    override_penalty: float = -25.0           # Increased 25x (Teaches respect for VL)
    k_stall: float = 5.0                      # Heavy penalty for sitting at Green
    k_lazy: float = 1.0                       # Penalty for slow acceleration

    # Behavioural and Operational penalties
    jerk_penalty: float = -0.1                
    signal_compliance_bonus: float = 0.3      
    energy_penalty_weight: float = 0.0        

    headway_violation_penalty: float = -5000.0
    collision_penalty: float = -10000.0

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
    safety_overridden: bool = False
    # BUG 11/21 FIX: action_delta is the absolute difference between two
    # continuous accelerations — a float in [0.0, 1.5], not a discrete int.
    action_delta: float = 0.0  # |safe_a - last_a|  (m/s²)
    applied_traction: float = 0.0  # Positive throttle applied by agent (m/s²), defaults to 0

    # Gap 1 fix: distance to nearest obstruction (lead train, hazard, or station)
    distance_to_occupied: float = 99999.0
    # Gap 3 fix: whether the AI is in a forced station dwell
    is_dwelling: bool = False
    # Principle 2: station index for escalating rewards
    station_index: int = 0
    # Signal awareness
    signal_aspect: int = 3  # 0=Red, 1=Orange, 2=FlashGreen, 3=Green
    applied_acceleration: float = 0.0  # the actual acceleration applied this step
    proposed_acceleration: float = 0.0  # the agent's chosen action (before VL clamping)


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


def _compute_speed_reward(state: TrainState, config: RewardConfig) -> float:
    
    v: float = state.current_speed
    v_max: float = state.speed_limit
    # Force a minimum distance so the target speed is NEVER 0.0 when cleared to depart. 
    # This prevents an inescapable cowardice trap at the station.
    d_from: float = max(20.0, state.current_position - state.last_station_position)
    d_to: float = max(0.0, state.next_station_position - state.current_position)

    # Gap 1 fix: use the nearest obstruction (lead train, hazard, station)
    # instead of just the next station for the braking envelope.
    # This ensures the target speed drops to 0 as the AI approaches
    # a Red signal, aligning the reward with the Validation Layer.
    d_brake: float = min(d_to, state.distance_to_occupied)

    # Braking envelope only (v^2 = 2as => v = sqrt(2as)).
    # v_accel removed: it clamped v_target to 4.47 m/s right after every station
    # (d_from clamped to 20m), teaching the agent to crawl away from stops.
    # Departure acceleration is guided by heartbeat + progress already.
    # Trapezoid analysis confirmed: dynamic formula applies to BRAKING only.
    v_brake = math.sqrt(2 * config.comfortable_deceleration * d_brake)

    v_target: float = min(v_max, v_brake)
    
    overspeed: float = max(0.0, v - v_target)
    underspeed: float = max(0.0, v_target - v)

    # Signal compliance: don't punish underspeed when signal is Red/Orange
    # and AI is correctly slowing down or stopped. The AI is doing the RIGHT
    # thing by being cautious — only penalise underspeed on Green/FlashGreen.
    if state.signal_aspect <= 1 and v < 1.0:
        underspeed = 0.0

    raw = -(config.k_over * overspeed**2) - (config.k_under * underspeed)
    # Principle 3: cap dense speed penalty so it can't drown sparse station rewards
    return max(config.speed_penalty_cap, raw)

def _compute_energy_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize purely positive acceleration applications (traction). Coasting and braking are free.
    return config.energy_penalty_weight * state.applied_traction

def _compute_heartbeat_penalty(state: TrainState, config: RewardConfig) -> float:
    # Gap 3 fix: don't penalise the AI for sitting still during a
    # forced station dwell — it has no control over this period.
    if state.is_dwelling:
        return 0.0
    # Signal compliance: don't punish for being stopped at Red/Orange.
    # The AI is correctly obeying the signal — penalising it here
    # teaches cowardice ("stopping is always bad").
    if state.signal_aspect <= 1 and state.current_speed < 0.5:
        return 0.0
    return config.heartbeat_penalty

def _compute_override_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.override_penalty if state.safety_overridden else 0.0

def _compute_jerk_penalty(state: TrainState, config: RewardConfig) -> float:
    # Gap 2 fix: only penalise voluntary jerk from the AI's own action
    # changes.  When the VL overrides, the jerk is the VL's doing — the
    # AI is already punished via the override penalty.
    if state.safety_overridden:
        return 0.0
    # Penalize the magnitude of the action shift squared (e.g., jump of 1 = -2, jump of 1.5 = -4.5)
    return config.jerk_penalty * (state.action_delta ** 2)

def _compute_signal_compliance_reward(state: TrainState, config: RewardConfig) -> float:
    if state.safety_overridden:
        return 0.0

    v = state.current_speed
    v_lim = state.speed_limit
    a = state.applied_acceleration
    bonus = config.signal_compliance_bonus

    if state.signal_aspect == 3:  # Green
        if v < 0.5:
            return -config.k_stall # Punish stalling heavily
        return bonus * min(1.0, v / v_lim)
    elif state.signal_aspect == 0:  # Red
        # Return 0 instead of positive bonus. 
        # Combined with existence_penalty, sitting at Red is now a net loss.
        return 0.0
    
    # Keep your existing logic for FlashGreen and Orange...
    return 0.0


def _compute_station_reward(state: TrainState, config: RewardConfig) -> float:
    # Principle 2: rewards escalate toward destination.
    # Station 0 = base × 1.0, Station 5 = base × 2.5
    # Creates a reward gradient pulling AI toward completion.
    if not state.reached_new_station:
        return 0.0
    escalation = 1.0 + state.station_index * config.station_escalation
    return config.station_reward_base * escalation
 
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
    r_signal_compliance: float
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
    Compute the full reward for one timestep, incorporating anti-cowardice logic.
    """
    # 1. Existence tax (The "Anti-Cowardice" clock)
    r_existence = config.existence_penalty 

    # 2. Continuous rewards (every timestep)
    r_progress  = _compute_progress_reward(state, config)
    r_headway   = _compute_headway_penalty(state, config)
    r_speed     = _compute_speed_reward(state, config)
    r_heartbeat = _compute_heartbeat_penalty(state, config)
    r_signal_compliance = _compute_signal_compliance_reward(state, config)
 
    # 3. Event rewards (on specific triggers)
    r_station   = _compute_station_reward(state, config)
    r_time      = _compute_punctuality_penalty(state, config)
    r_override  = _compute_override_penalty(state, config)
    r_jerk      = _compute_jerk_penalty(state, config)
    r_energy    = _compute_energy_penalty(state, config)
 
    # 4. Terminal penalties (end of episode)
    r_violation, terminate_violation = _compute_headway_violation(state, config)
    r_collision, terminate_collision = _compute_collision_penalty(state, config)
 
    # --- Aggregation ---
    
    # We fold the existence tax into the continuous reward total.
    # This means even if the train is stationary, it's losing points every step.
    r_continuous = (r_progress + r_headway + r_speed + 
                    r_heartbeat + r_signal_compliance + r_existence)
    
    r_event      = r_station + r_time + r_override + r_jerk + r_energy
    r_terminal   = r_violation + r_collision
    
    r_total      = r_continuous + r_event + r_terminal
    
    # Episode should end if a safety violation or collision occurs
    terminate    = terminate_violation or terminate_collision
 
    return RewardOutput(
        r_progress=r_progress,
        r_headway=r_headway,
        r_speed=r_speed,
        r_heartbeat=r_heartbeat,
        r_signal_compliance=r_signal_compliance,
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
        terminate=terminate
    )

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
    print(f"  Sig Compliance : {result.r_signal_compliance:+.4f}")
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