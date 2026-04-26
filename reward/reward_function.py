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
    r_step: float = 0.0
    r_progress: float = 0.0
    r_speed: float = 0.0
    r_signal_compliance: float = 0.0
    r_station: float = 0.0
    r_time: float = 0.0
    r_jerk: float = 0.0
    r_patience: float = 0.0
    r_total: float = 0.0

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

    # Step penalty — drives directed exploration toward the terminus
    out.r_step = -0.5

    # 1. Progress Reward
    distance_travelled = state.current_position - state.previous_position
    if distance_travelled > 0:
        out.r_progress = min(2.0, distance_travelled * 0.02)
    else:
        out.r_progress = 0.0
    
    # Suspend if lead train issues
    if lead_train_stalled or lead_train_held:
        out.r_progress = max(0.0, out.r_progress) # Don't penalise slow progress

    # 2. Speed Compliance Reward
    effective_aspect = min(train_aspect, station_aspect)

    if effective_aspect == 3:
        out.r_speed = max(-10.0, -abs(u - state.speed_limit))
    elif effective_aspect in (1, 2):
        if a <= -1.0:
            out.r_speed = -8.0   # Emergency braking not justified at yellow/double-yellow
        elif -0.55 <= a <= -0.45:
            out.r_speed = 5.0    # Correct service braking
        elif -1.0 < a < -0.55:
            out.r_speed = -2.0   # Harder than service braking but not emergency
        else:
            out.r_speed = -5.0   # Coasting or accelerating toward obstacle
    elif effective_aspect == 0:
        if u < 0.1:
            out.r_speed = 0.0    # Correctly stopped
        elif a <= -1.0:
            # Justified only if Red aspect and train is near the speed limit
            if u >= state.speed_limit * 0.8:
                out.r_speed = 0.0    # Justified emergency brake
            else:
                out.r_speed = -8.0   # Unnecessary emergency brake
        elif a <= -0.45:
            out.r_speed = 5.0    # Braking correctly toward stop
        else:
            out.r_speed = -5.0   # Moving at Red without adequate braking

    # 4. Jerk Reward/Penalty
    jerk = abs(state.applied_acceleration - info.get("previous_a", 0.0)) / dt
    if jerk < 0.5:
        out.r_jerk = 2.0
    elif jerk > 1.0:
        out.r_jerk = -4.0

    # 5. Signal Compliance
    # Train Signal
    if train_aspect == 0 and state.proposed_acceleration > 0.01:
        out.r_signal_compliance = -10.0
    if train_aspect == 3:
        # Sweet spot bonus
        x_ai_zone_end = info.get("x_ai_zone_end", state.current_position)
        x_diff = x_lead_zone_start - x_ai_zone_end
        if 3*sh <= x_diff <= 4*sh:
            out.r_signal_compliance = 5.0

    # Station Signal
    if station_aspect == 0:
        if state.current_position >= x_station_zone_start and u > 0.1:
            out.r_signal_compliance = -10.0

    # 7. Station Milestone Reward
    STATION_REWARDS = {
        1: +2.0,   # Rabka-Zdrój
        2: +3.0,   # Mszana Dolna
        3: +4.0,   # Tymbark
        4: +5.0,   # Limanowa
        5: +7.0,   # Marcinkowice
        6: +10.0,  # Nowy Sącz (terminus — maximum reward)
    }
    if state.reached_new_station:
        out.r_station = STATION_REWARDS.get(state.station_index, 0.0)

    # 8. Punctuality Reward
    if state.reached_new_station:
        if state.scheduled_arrival_time is not None and state.actual_arrival_time is not None:
            deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)
            if deviation <= 60.0:
                out.r_time = 5.0
            else:
                out.r_time = -4.0
    
    # 9. Patience Reward
    if state.is_dwelling and u < 0.1:
        out.r_patience = 5.0
    elif station_cleared and train_aspect == 3 and u < 0.1:
        # Check if we should be moving
        x_ai_zone_end = info.get("x_ai_zone_end", state.current_position)
        x_diff = x_lead_zone_start - x_ai_zone_end
        if x_diff > 3*sh:
            out.r_patience = -10.0

    # Sum all
    out.r_total = (out.r_step + out.r_progress + out.r_speed + 
                   out.r_signal_compliance + out.r_station + out.r_time + out.r_jerk + 
                   out.r_patience)
    
    return out
