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
    x_ai_zone_end = info.get("x_ai_zone_end", state.current_position)
    x_diff = x_lead_zone_start - x_ai_zone_end  # zones between AI and lead train

    out = RewardOutput()

    # Step penalty — drives directed exploration toward the terminus
    out.r_step = -1

    # 1. Progress Reward
    # Scaled so that driving at the speed limit always roughly offsets the step penalty,
    # regardless of which segment the train is on. At 90 km/h (25 m/s): 25 * 0.028 = 0.7.
    # At 30 km/h (8.33 m/s): 8.33 * 0.028 = 0.23 — used to bleed -0.27/step on slow segments.
    # Fix: scale the multiplier by (max_limit / current_limit) so the reward is always ~0.7
    # when driving at the local speed limit.
    MAX_LINE_SPEED_MS = 25.0  # 90 km/h in m/s — fastest segment on line 104
    speed_limit_scale = MAX_LINE_SPEED_MS / max(state.speed_limit, 1.0)
    distance_travelled = state.current_position - state.previous_position
    if distance_travelled > 0:
        out.r_progress = min(2.0, distance_travelled * 0.028 * speed_limit_scale)
    else:
        out.r_progress = 0.0
    
    # Suspend if lead train issues
    if lead_train_stalled or lead_train_held:
        out.r_progress = max(0.0, out.r_progress)

    # 2. Speed Compliance Reward
    effective_aspect = min(train_aspect, station_aspect)

    # At terminus, stopped — no speed penalty applies
    at_terminus = (state.current_position >= state.next_station_position - 10.0 and u < 0.1)
    if at_terminus:
        out.r_speed = 0.0
    elif effective_aspect == 3:
        # Green: penalise overspeed only.
        if a <= -1.0:
            out.r_speed = -15.0 # High penalty for emergency braking on green
        elif u > state.speed_limit + 0.1:
            out.r_speed = max(-10.0, -(u - state.speed_limit))  # overspeed penalty
        else:
            out.r_speed = 0.0  # at or below limit — fine

    elif effective_aspect in (1, 2):
        # Yellow / Double-Yellow: train does NOT need to brake right now — it just
        # cannot be accelerating at/above the limit, and must not use emergency braking.
        # Coasting at or below the speed limit is perfectly valid.
        if a <= -1.0:
            out.r_speed = -10.0 # Penalty for emergency braking on cautionary signals
        elif a > 0.01 and u >= state.speed_limit - 0.1:
            out.r_speed = -5.0   # Accelerating at or above speed limit toward obstacle
        elif -0.55 <= a <= -0.45:
            out.r_speed = 5.0    # Correct service braking
        elif -1.0 < a < -0.55:
            out.r_speed = -2.0   # Harder than service braking but not emergency
        else:
            out.r_speed = 0.0    # Coasting or gentle accel below limit — acceptable

    elif effective_aspect == 0:
        # Red: must stop. Reward braking, penalise anything else.
        if u < 0.1:
            out.r_speed = 2.0 # Bonus for being perfectly stopped
        elif a <= -1.0:
            out.r_speed = -20.0 # Heavy penalty for slamming brakes at red (use service braking!)
        elif -0.6 <= a <= -0.4:
            out.r_speed = 5.0 # Bonus for smooth service braking
        else:
            out.r_speed = -5.0   # Moving at Red without adequate braking

    # 4. Jerk Penalty (penalty-only — smooth control is the expected baseline, not a bonus)
    # A constant +2.0 bonus every cruise step was inflating cumulative reward by ~30,000+
    jerk = abs(state.applied_acceleration - info.get("previous_a", 0.0)) / dt
    if jerk > 1.0:
        out.r_jerk = -1.0
    # else: 0.0 — smooth control is expected, not rewarded

    # 5. Signal Compliance
    # Train Signal
    if train_aspect == 0 and state.proposed_acceleration > 0.01:
        out.r_signal_compliance += -10.0
    if train_aspect == 3:
        # Sweet spot bonus: reward maintaining the ideal following gap (3–4 SH).
        # Only award when the agent is moving (not coasting at a stop).
        if 3*sh <= x_diff <= 4*sh and u > 0.5:
            out.r_signal_compliance += 5.0  # Sweet spot bonus
        elif x_diff > 4*sh and u > 0.5:
            out.r_signal_compliance -= 5.0  # Penalty for lagging too far behind

    # Station Signal
    if station_aspect == 0:
        if state.current_position >= x_station_zone_start and u > 0.1:
            out.r_signal_compliance += -10.0

    # 7. Station Milestone Reward
    STATION_REWARDS = {
        1: +1000.0,   # Rabka-Zdrój
        2: +2000.0,   # Mszana Dolna
        3: +3000.0,   # Tymbark
        4: +4000.0,   # Limanowa
        5: +5000.0,   # Marcinkowice
        6: +20000.0,  # Nowy Sącz (terminus — maximum reward)
    }
    if state.reached_new_station:
        out.r_station = STATION_REWARDS.get(state.station_index, 0.0)

    # 8. Punctuality Reward
    if state.reached_new_station:
        if state.scheduled_arrival_time is not None and state.actual_arrival_time is not None:
            deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)
            if deviation <= 60.0:
                out.r_time = 200.0
            else:
                out.r_time = -100.0
    
    # 9. Patience Reward
    if state.is_dwelling and u < 0.1:
        out.r_patience = 1.0
    elif station_cleared and train_aspect == 3 and u < 0.1:
        # Only penalise if station was already cleared last step too (avoid race condition
        # on the exact frame clearance flips — agent has no chance to react that step)
        if info.get("station_cleared_prev", False) and x_diff > 3*sh:
            out.r_patience = -10.0

    # Sum all
    out.r_total = (out.r_step + out.r_progress + out.r_speed + 
                   out.r_signal_compliance + out.r_station + out.r_time + out.r_jerk + 
                   out.r_patience)
    
    return out