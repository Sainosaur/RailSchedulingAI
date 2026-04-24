"""
Railway RL Reward Function

Structure:
  - Continuous rewards  (every timestep)
  - Event rewards       (on specific events)
  - Terminal penalties  (once, ends episode)

Total: r_t = r_continuous + r_event + r_terminal
"""

import math
from dataclasses import dataclass
from typing import Optional

# State container


@dataclass
class RewardConfig:
    # 1. Progress & Station Rewards (The "Carrots")
    k_p: float = (
        2000.0  # incremental progress reward scale (increase to 4000 if still slow)
    )
    station_reward_base: float = 8000.0  # milestone for station arrival
    station_escalation: float = 0.3

    # 2. Speed & Cruise Incentives
    k_over: float = 1.0
    k_under: float = 1.0
    speed_penalty_cap: float = -2.0
    cruise_bonus: float = 1.5  # NEW: Carrot for staying at 90%+ of speed limit

    # 3. Moving Incentives (The "Anti-Granny" kick)
    signal_compliance_bonus: float = 1.0
    k_stall: float = 20.0  # DOUBLED: sitting at Green is now EXTREMELY painful
    k_lazy: float = 10.0  # DOUBLED: punish hanging back too far
    clear_road_bonus: float = 1.0  # NEW: Bonus for having 3+ blocks clear

    # 4. Punctuality & Temporal (The "Schedule")
    punctuality_factor: float = 0.5
    punctuality_tolerance: float = 60.0
    ontime_bonus: float = 10000.0  # INCREASED: Massive carrot for hitting the window
    station_destination_bonus: float = 50000.0  # NEW: The "Grand Prize" for reaching Nowy Sącz
    punctuality_penalty_cap: float = -1000.0  # Cap the punctuality "Black Hole"

    # 5. Continuous Taxes (The "Anti-Cowardice" clock)
    existence_penalty: float = -0.1  # Tax per step (Forces movement)
    heartbeat_penalty: float = -0.5  # Forces movement at Green signals

    # 6. Safety & Operational (The "Guardrails")
    override_penalty: float = -25.0  # High cost for triggerring VL
    jerk_penalty: float = -0.1
    comfortable_acceleration: float = 0.5
    comfortable_deceleration: float = 0.5
    energy_penalty_weight: float = 0.0

    # 7. Headway & Collision (The "Disasters")
    k_h: float = 3.0
    headway_warning_multiplier: float = 3.0
    headway_violation_multiplier: float = 1.0
    headway_violation_penalty: float = -100000.0  # FATAL: Run this and lose everything
    collision_penalty: float = -100000.0  # FATAL

    # 8. Lateness & Dwell (The "Professionalism")
    lateness_violation_threshold: float = 600.0 
    lateness_violation_penalty: float = -10000.0 
    dwell_patience_bonus: float = 5.0  # NEW: Reward for staying at V=0 when signal is RED


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
    spatial_headway: float = 500.0  # distance of one block

    # Validation / Smoothing (Defaults set to safe values so the environment won't break until we integrate them)
    safety_overridden: bool = False
    # BUG 11/21 FIX: action_delta is the absolute difference between two
    # continuous accelerations — a float in [0.0, 1.5], not a discrete int.
    action_delta: float = 0.0  # |safe_a - last_a|  (m/s²)
    applied_traction: float = (
        0.0  # Positive throttle applied by agent (m/s²), defaults to 0
    )

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
    current_time: float = 0.0  # environment current time
    next_scheduled_arrival_time: float = 0.0  # scheduled arrival at the next station


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

    # 1. Braking envelope targeting the next station or occupied block ahead
    d_to: float = max(0.0, state.next_station_position - state.current_position)
    d_brake: float = min(d_to, state.distance_to_occupied)

    # v_target is the min of the speed limit and the square-root braking curve
    v_brake = math.sqrt(2 * config.comfortable_deceleration * d_brake)
    v_target: float = min(v_max, v_brake)

    overspeed = max(0.0, v - v_target)
    underspeed = max(0.0, v_target - v)

    if state.signal_aspect <= 1 and v < 1.0:
        underspeed = 0.0

    # FIXED: Using linear penalties instead of quadratic to stop emergency braking
    raw = -(config.k_over * overspeed) - (config.k_under * underspeed)

    return max(config.speed_penalty_cap, raw)


def _compute_energy_penalty(state: TrainState, config: RewardConfig) -> float:
    # Penalize purely positive acceleration applications (traction). Coasting and braking are free.
    return config.energy_penalty_weight * state.applied_traction


def _compute_heartbeat_penalty(state: TrainState, config: RewardConfig) -> float:
    # Punish only if stationary (v < 0.5) when the signal is Green (aspect 3).
    # This stops the "slow creep" reward drain while moving.
    if not state.is_dwelling and state.signal_aspect == 3 and state.current_speed < 0.5:
        return config.heartbeat_penalty
    return 0.0


def _compute_clear_road_bonus(state: TrainState, config: RewardConfig) -> float:
    # Bonus for being in the "Sweet Spot": Green signal but just barely.
    # Encourages the train to close the gap and stay at the front of the green wave (exactly 3-4 zones back).
    # If d < 3*SH, aspect is no longer 3 (turns to Double Yellow).
    # If d > 4*SH, we cut the bonus to prevent the AI from lagging too far behind.
    sh = state.spatial_headway
    d = state.distance_to_occupied

    if state.signal_aspect == 3 and (3.0 * sh <= d <= 4.0 * sh):
        return config.clear_road_bonus
    return 0.0


def _compute_cruise_reward(state: TrainState, config: RewardConfig) -> float:
    # Strong positive reinforcement for actually hitting the speed limit.
    # We reward being within 90% to 105% of the limit.
    v = state.current_speed
    v_lim = state.speed_limit
    reward = 0.0

    if v_lim > 5.0 and 0.9 * v_lim <= v <= 1.05 * v_lim:
        reward += config.cruise_bonus

    # Technical Instruction: Add a specific bonus (+2.0) if abs(applied_acceleration) < 0.01
    # and the train is within 95% of the speed_limit.
    if v >= 0.95 * v_lim and abs(state.applied_acceleration) < 0.01:
        reward += 2.0

    return reward


def _compute_override_penalty(state: TrainState, config: RewardConfig) -> float:
    return config.override_penalty if state.safety_overridden else 0.0


def _compute_jerk_penalty(state: TrainState, config: RewardConfig) -> float:
    # Gap 2 fix: only penalise voluntary jerk from the AI's own action
    # changes.  When the VL overrides, the jerk is the VL's doing — the
    # AI is already punished via the override penalty.
    if state.safety_overridden:
        return 0.0
    # Penalize the magnitude of the action shift squared (e.g., jump of 1 = -2, jump of 1.5 = -4.5)
    return config.jerk_penalty * (state.action_delta**2)


def _compute_creep_penalty(state: TrainState, config: RewardConfig) -> float:
    # Punish "creeping" (moving very slowly) when the signal is not Green
    if state.signal_aspect < 3 and 0.1 <= state.current_speed <= 2.0:
        return -5.0
    return 0.0


def _compute_patience_reward(state: TrainState, config: RewardConfig) -> float:
    """Reward for remaining stationary when required (Signals or Stations)."""
    # 1. Stationary at Restrictive Signal (Double Yellow or Yellow)
    if (state.signal_aspect == 1 or state.signal_aspect == 2) and state.current_speed < 0.1:
        return 2.0  # Encourage waiting for signals to clear

    # 2. Stationary at Station (Dwell) or RED Signal
    # If we are within 5 meters of a station OR signal is RED, and speed is zero, provide dwell bonus
    dist_to_station = min(
        abs(state.current_position - state.last_station_position),
        abs(state.current_position - state.next_station_position)
    )
    if (dist_to_station < 5.0 or state.signal_aspect == 0) and state.current_speed < 0.1:
        return config.dwell_patience_bonus  # Strong bonus specifically for waiting correctly

    return 0.0


def _compute_signal_compliance_reward(state: TrainState, config: RewardConfig) -> float:
    if state.safety_overridden:
        return 0.0

    v = state.current_speed
    v_lim = state.speed_limit
    a = state.proposed_acceleration  # Use proposed_acceleration to judge AI intent
    bonus = config.signal_compliance_bonus

    # RESTART PUNISHMENT: Massive penalty for trying to move from stop at Red/Yellow/DoubleYellow
    if v < 0.1 and a > 0.05 and state.signal_aspect < 3:
        return -5000.0  # CRITICAL: Punish intent to run a red light BEFORE they even move

    if state.signal_aspect == 3:  # Green
        # 1. Punish if stationary at Green
        if v < 0.5:
            return -config.k_stall * 2.0  # Increased penalty for stalling

        # 2. Sweet Spot Logic: Formation Driving (3-4 blocks behind lead)
        d_occ = state.distance_to_occupied
        sh = state.spatial_headway
        
        if 3.0 * sh <= d_occ <= 4.0 * sh:
            return bonus * 5.0  # High reward for staying in the 'Sweet Spot'
        
        # Removed lazy penalty to allow normal green signal progression

        # 3. Normal speed progression bonus for 0-3 blocks
        return config.signal_compliance_bonus * (v / v_lim)

    elif state.signal_aspect == 2:  # Double Yellow — MUST decelerate
        if a < -0.1:
            return bonus * 4.0  # DOUBLED: Reward active braking more aggressively
        elif a > 0.05:
            return -config.override_penalty * 0.5
        return 0.0

    elif state.signal_aspect == 1:  # Yellow — MUST decelerate
        if a < -0.1:
            return bonus * 6.0  # DOUBLED: Very high reward for braking at yellow
        elif a > 0.01:
            return -config.override_penalty * 0.8
        return 0.0

    elif state.signal_aspect == 0:  # RED: Next Zone Occupied — Stop within current zone
        if v > 0.1:
            if a < -0.1:
                return bonus * 5.0  # Massive reward for active braking to target stop line
            else:
                return -config.override_penalty  # Critical penalty for not slowing down
        return 0.0

    return 0.0


def _compute_station_reward(state: TrainState, config: RewardConfig) -> float:
    # Principle 2: rewards escalate toward destination.
    # Station 0 = base × 1.0, Station 5 = base × 2.5
    # Creates a reward gradient pulling AI toward completion.
    if not state.reached_new_station:
        return 0.0
    escalation = 1.0 + state.station_index * config.station_escalation
    reward = config.station_reward_base * escalation
    
    # DESTINATION BONUS: If this is the final station (Index 6: Nowy Sącz)
    if state.station_index >= 6:
        reward += config.station_destination_bonus
        
    return reward


def _compute_punctuality_penalty(state: TrainState, config: RewardConfig) -> float:
    """Reward for arriving on-time, or penalty for missing the window.

    BUG 13 FIX: The environment now feeds accurate scheduled/actual arrival times.
    """
    if not state.reached_new_station:
        return 0.0
    if state.scheduled_arrival_time is None or state.actual_arrival_time is None:
        return 0.0

    deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)

    # 1. On-Time Reward
    if deviation <= config.punctuality_tolerance:
        return config.ontime_bonus

    # 2. Missed Window Penalty
    excess = max(0.0, deviation - config.punctuality_tolerance)
    raw_penalty = -config.punctuality_factor * excess
    return max(config.punctuality_penalty_cap, raw_penalty)


def _compute_headway_violation(
    state: TrainState, config: RewardConfig
) -> tuple[float, bool]:

    violation_thresh = state.temporal_headway * config.headway_violation_multiplier
    if state.headway <= violation_thresh:
        return config.headway_violation_penalty, True

    return 0.0, False


def _compute_collision_penalty(
    state: TrainState, config: RewardConfig
) -> tuple[float, bool]:

    if state.collision:
        return config.collision_penalty, True
    return 0.0, False


def _compute_lateness_violation(
    state: TrainState, config: RewardConfig
) -> tuple[float, bool]:
    if (
        state.next_scheduled_arrival_time > 0
        and state.current_time
        > state.next_scheduled_arrival_time + config.lateness_violation_threshold
    ):
        return config.lateness_violation_penalty, True
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
    r_clear_road: float
    r_cruise: float
    r_patience: float
    r_creep: float
    # Event
    r_station: float
    r_time: float
    r_override: float
    r_jerk: float
    r_energy: float
    # Terminal
    r_violation: float
    r_collision: float
    r_lateness: float
    # Aggregate
    r_continuous: float
    r_event: float
    r_terminal: float
    r_total: float
    # Episode control
    terminate: bool


# 2. Update the compute_reward function
def compute_reward(
    state: TrainState, config: RewardConfig = DEFAULT_CONFIG
) -> RewardOutput:
    r_existence = config.existence_penalty

    r_progress = _compute_progress_reward(state, config)
    r_headway = _compute_headway_penalty(state, config)
    r_speed = _compute_speed_reward(state, config)
    r_heartbeat = _compute_heartbeat_penalty(state, config)
    r_signal_compliance = _compute_signal_compliance_reward(state, config)
    r_clear_road = _compute_clear_road_bonus(state, config)
    r_cruise = _compute_cruise_reward(state, config)
    r_patience = _compute_patience_reward(state, config)
    r_creep = _compute_creep_penalty(state, config)

    r_station = _compute_station_reward(state, config)
    r_time = _compute_punctuality_penalty(state, config)
    r_override = _compute_override_penalty(state, config)
    r_jerk = _compute_jerk_penalty(state, config)
    r_energy = _compute_energy_penalty(state, config)

    r_violation, terminate_violation = _compute_headway_violation(state, config)
    r_collision, terminate_collision = _compute_collision_penalty(state, config)
    r_lateness, terminate_lateness = _compute_lateness_violation(state, config)

    # Aggregate including r_patience and r_creep
    r_continuous = (
        r_progress
        + r_headway
        + r_speed
        + r_heartbeat
        + r_signal_compliance
        + r_clear_road
        + r_cruise
        + r_existence
        + r_patience
        + r_creep
    )

    r_event = r_station + r_time + r_override + r_jerk + r_energy
    r_terminal = r_violation + r_collision + r_lateness
    r_total = r_continuous + r_event + r_terminal

    terminate = terminate_violation or terminate_collision or terminate_lateness

    return RewardOutput(
        r_progress=r_progress,
        r_headway=r_headway,
        r_speed=r_speed,
        r_heartbeat=r_heartbeat,
        r_signal_compliance=r_signal_compliance,
        r_clear_road=r_clear_road,
        r_cruise=r_cruise,
        r_patience=r_patience,
        r_creep=r_creep,
        r_station=r_station,
        r_time=r_time,
        r_override=r_override,
        r_jerk=r_jerk,
        r_energy=r_energy,
        r_violation=r_violation,
        r_collision=r_collision,
        r_lateness=r_lateness,
        r_continuous=r_continuous,
        r_event=r_event,
        r_terminal=r_terminal,
        r_total=r_total,
        terminate=terminate,
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
        headway=15.0,  # e.g. 15s headway
        temporal_headway=12.5,
        reached_new_station=False,
        current_time=100.0,
        next_scheduled_arrival_time=500.0,
    )

    result = compute_reward(state)

    print("=== Reward Breakdown ===")
    print(f"  Progress       : {result.r_progress:+.4f}")
    print(f"  Headway        : {result.r_headway:+.4f}")
    print(f"  Speed          : {result.r_speed:+.4f}")
    print(f"  Heartbeat      : {result.r_heartbeat:+.4f}")
    print(f"  Sig Compliance : {result.r_signal_compliance:+.4f}")
    print(f"  Clear Road     : {result.r_clear_road:+.4f}")
    print(f"  Cruise         : {result.r_cruise:+.4f}")
    print(f"  Station        : {result.r_station:+.4f}")
    print(f"  Punctuality    : {result.r_time:+.4f}")
    print(f"  Override       : {result.r_override:+.4f}")
    print(f"  Jerk           : {result.r_jerk:+.4f}")
    print(f"  Energy         : {result.r_energy:+.4f}")
    print(f"  HW Violation   : {result.r_violation:+.4f}")
    print(f"  Collision      : {result.r_collision:+.4f}")
    print(f"  Lateness       : {result.r_lateness:+.4f}")
    print("------------------------")
    print(f"  Continuous     : {result.r_continuous:+.4f}")
    print(f"  Event          : {result.r_event:+.4f}")
    print(f"  Terminal       : {result.r_terminal:+.4f}")
    print(f"  TOTAL          : {result.r_total:+.4f}")
    print(f"  Terminate      : {result.terminate}")
