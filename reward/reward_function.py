"""
reward/reward_function.py
Railway RL Reward Function
"""

import math
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any

@dataclass
class TrainState:
    """All variables required to compute the reward at one timestep."""
    current_position: float
    previous_position: float
    last_station_position: float
    next_station_position: float
    current_speed: float
    speed_limit: float
    headway: float  # temporal headway in seconds
    temporal_headway: float
    spatial_headway: float
    reached_new_station: bool
    scheduled_arrival_time: Optional[float] = None
    actual_arrival_time: Optional[float] = None
    collision: bool = False
    safety_overridden: bool = False
    action_delta: float = 0.0
    applied_traction: float = 0.0
    distance_to_occupied: float = 99999.0
    is_dwelling: bool = False
    station_index: int = 0
    signal_aspect: int = 3
    applied_acceleration: float = 0.0
    proposed_acceleration: float = 0.0
    current_time: float = 0.0
    next_scheduled_arrival_time: float = 0.0

@dataclass
class RewardOutput:
    """Structured reward breakdown for easy debugging."""
    r_progress: float = 0.0
    r_headway: float = 0.0
    r_speed: float = 0.0
    r_heartbeat: float = 0.0
    r_signal_compliance: float = 0.0
    r_station: float = 0.0
    r_time: float = 0.0
    r_jerk: float = 0.0
    r_regen: float = 0.0
    r_patience: float = 0.0
    r_existence: float = 0.0
    r_violation: float = 0.0
    r_collision: float = 0.0
    r_total: float = 0.0
    terminate: bool = False

    def to_dict(self):
        return asdict(self)

def compute_reward(state: TrainState, info: Dict[str, Any]) -> RewardOutput:
    # Extract info
    train_aspect = info.get("train_aspect", 3)
    station_aspect = info.get("station_aspect", 3)
    station_cleared = info.get("station_cleared", False)
    x_lead_zone_start = info.get("x_lead_zone_start", 99999.0)
    x_station_zone_start = info.get("x_station_zone_start", 99999.0)
    violations = info.get("violations", {})
    lead_train_stalled = info.get("lead_train_stalled", False)
    lead_train_held = info.get("lead_train_held", False)
    
    sh = state.spatial_headway
    dt = 1.0
    u = state.current_speed
    a = state.applied_acceleration
    
    out = RewardOutput()

    # 1. Progress Reward
    span = state.next_station_position - state.last_station_position
    if span > 0:
        p_current = max(0.0, min(1.0, (state.current_position - state.last_station_position) / span))
        p_previous = max(0.0, min(1.0, (state.previous_position - state.last_station_position) / span))
        out.r_progress = 2000.0 * (p_current - p_previous)
        # Suspend if lead train issues
        if lead_train_stalled or lead_train_held:
            out.r_progress = max(0.0, out.r_progress) # Don't penalise slow progress

    # 2. Speed Compliance Reward
    # Regime is determined by the more restrictive of train_aspect and station_aspect.
    # Green  (3): clear road ahead — reward staying near the speed limit.
    # DY/Yellow (2/1): approaching obstacle — reward service braking at 0.5 m/s/s.
    # Red    (0): must stop — reward braking or being stopped.
    effective_aspect = min(train_aspect, station_aspect)

    if effective_aspect == 3:
        out.r_speed = -abs(u - state.speed_limit)
    elif effective_aspect in (1, 2):
        if -0.55 <= a <= -0.45:
            out.r_speed = 5.0
        elif -1.0 < a < -0.55:
            out.r_speed = -2.0   # Harder than service braking but not emergency
        elif a <= -1.0:
            # Justified only if Red aspect and train is near the speed limit
            if effective_aspect == 0 and u >= state.speed_limit * 0.8:
                out.r_speed = 0.0    # Justified emergency brake
            else:
                out.r_speed = -10.0  # Unnecessary emergency brake
    elif effective_aspect == 0:
        if u < 0.1:
            out.r_speed = 0.0    # Correctly stopped
        elif a <= -1.0:
            # Justified only if Red aspect and train is near the speed limit
            if effective_aspect == 0 and u >= state.speed_limit * 0.8:
                out.r_speed = 0.0    # Justified emergency brake
            else:
                out.r_speed = -10.0  # Unnecessary emergency brake
        elif a <= -0.45:
            out.r_speed = 2.0    # Braking correctly toward stop
        else:
            out.r_speed = -5.0   # Moving at Red without adequate braking

    # 3. Regenerative Braking Reward
    if -0.5 <= a <= -0.01:
        out.r_regen = 2.0
    elif -1.0 < a < -0.5:
        out.r_regen = -1.0 # Small penalty for non-regen non-emergency

    # 4. Jerk Reward/Penalty
    jerk = abs(state.applied_acceleration - info.get("previous_a", 0.0)) / dt
    if jerk < 0.5:
        out.r_jerk = 1.0
    elif jerk > 1.0:
        out.r_jerk = -2.0 * (jerk - 1.0)

    # 5. Signal Compliance
    # Train Signal
    if train_aspect == 0 and state.proposed_acceleration > 0.01:
        out.r_signal_compliance -= 500.0 # Red punishment
    if train_aspect == 3:
        # Sweet spot bonus
        x_ai_zone_end = state.current_position + (state.spatial_headway - (state.current_position % state.spatial_headway)) # Rough estimate
        # Better: use info for x_ai_zone_end
        x_ai_zone_end = info.get("x_ai_zone_end", state.current_position)
        x_diff = x_lead_zone_start - x_ai_zone_end
        if 3*sh <= x_diff <= 4*sh:
            out.r_signal_compliance += 5.0

    # Station Signal
    if station_aspect == 0:
        if state.current_position >= x_station_zone_start and u > 0.1:
            out.r_signal_compliance -= 100.0

    # 6. Headway Penalty
    x_ai_zone_end = info.get("x_ai_zone_end", state.current_position)
    x_diff = x_lead_zone_start - x_ai_zone_end
    k_h = 3.0
    if x_diff < 3 * sh:
        if x_diff <= sh:
            out.r_headway = -k_h
        else:
            # Linear scale from 0 at 3*SH to -k_h at 1*SH
            t = (3 * sh - x_diff) / (2 * sh)
            out.r_headway = -k_h * t

    # 7. Station Milestone Reward
    if state.reached_new_station:
        escalation = 1.0 + state.station_index * 0.3
        out.r_station = 8000.0 * escalation
        if state.station_index >= 6:
            out.r_station += 50000.0

    # 8. Punctuality Reward
    if state.reached_new_station:
        if state.scheduled_arrival_time is not None and state.actual_arrival_time is not None:
            deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)
            if deviation <= 60.0:
                out.r_time = 10000.0
            # Suspend accumulation if stalled? No, just the milestone.
    
    # Lateness accumulation penalty (if any) should be suspended if lead stalled
    # But we don't have one now (existence is different)

    # 9. Patience Reward
    if state.is_dwelling and u < 0.1:
        out.r_patience = 10.0
    elif station_cleared and train_aspect == 3 and u < 0.1 and x_diff > 3*sh:
        out.r_patience = -20.0 # Penalty for stalling at Green

    # 10. Existence Penalty
    out.r_existence = -0.1

    # Terminal Penalties
    if x_diff <= 0:
        out.r_collision = -100000.0
        out.terminate = True
    
    if violations.get("any_violation"):
        out.r_violation = -10.0 # Small penalty for logged violations

    # Sum all
    out.r_total = (out.r_progress + out.r_headway + out.r_speed + 
                   out.r_signal_compliance + out.r_station + out.r_time + out.r_jerk + 
                   out.r_regen + out.r_patience + out.r_existence + out.r_violation + 
                   out.r_collision)
    
    return out
