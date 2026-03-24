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

# Constants / hyperparameters

K_P: float = 5.0 # incremental progress rewad scale
STATION_REWARD: float = 5.0
PUNCTUALITY_FACTOR: float = 0.5
 
HEADWAY_WARNING_THRESHOLD: float = 240.0   # seconds — warning zone begins
HEADWAY_VIOLATION_THRESHOLD: float = 192.0 # seconds — hard safety limit
 
K_OVER: float = 1.0        # overspeed penalty weight  (quadratic)
K_UNDER: float = 0.0       # underspeed penalty weight (linear, default 0)
 
HEADWAY_VIOLATION_PENALTY: float = -150.0
COLLISION_PENALTY: float = -200.0

# State container

@dataclass
class TrainState:
    """All variables required to compute the reward at one timestep."""

    #postion 
    current_positions: float         #meters along the track
    previouse_position: float        #psition at previous timestamp t-1
    last_station_position: float     #position of the last station passed
    next_station_position: float     #position of the next station ahead

    #speed
    current_speed: float             # m/s
    speed_limit: float               # m/s (track speed limit)

    # Distances (can be derived but passed in for clarity)
    distance_from_last_station: float  # metres
    distance_to_next_station: float    # metres
 
    # Safety
    headway: float                   # seconds to the train ahead
 
    # Timetable / events
    reached_new_station: bool        # True only in the timestep of arrival
    scheduled_arrival_time: Optional[float] = None  # seconds
    actual_arrival_time: Optional[float] = None     # seconds
 
    # Safety events
    collision: bool = False
 
    # Buffer distances for speed profile
    acceleration_buffer: float = 200.0  # D_a  (metres) — tune per prototype
    braking_buffer: float = 300.0       # D_b  (metres) — tune per prototype


# Reward components 

def _compute_progress_reward(state:TrainState) -> float:
    """
    1.1 Incremental progress reward.
 
    Normalised progress change between the last and current timestep.
 
        p_t = (x_t - x_last) / (x_next - x_last)
 
    r_progress = k_p * (p_t - p_{t-1})
 
    Using previous_position as the proxy for p_{t-1}.
    """
    span = state.next_station_position - state.last_station_position
    if span<=0:
        return 0.0
    
    p_current = (state.current_positions - state.last_station_position)/span
    p_previous = (state.previouse_position - state.last_station_position)/span

    p_current = max(0.0, min(1.0, p_current))   #clamp to [0,1]
    p_previous = max(0.0, min(1.0, p_previous)) #clamp to [0,1]

    return K_P*(p_current - p_previous)

def _compute_headway_penalty(state: TrainState) -> float:
    """
    1.2 Headway warning-zone penalty.
 
    Applied only when headway < 240 s (above the hard 192 s limit).
 
        r_headway = -h_t * (240 - h_t) / 1000   if h_t < 240
                  = 0                             otherwise
    """
    h = state.headway
    if HEADWAY_VIOLATION_THRESHOLD < h < HEADWAY_WARNING_THRESHOLD:
        return -(h * (HEADWAY_WARNING_THRESHOLD - h)) / 1000.0
    return 0.0

def _compute_speed_reward(state: TrainState) -> float:
     """
    1.3 Speed compliance reward.
 
    Phase-aware target speed:
 
        v* = min(v_max,
                 v_max * d_from / D_a,   # acceleration buffer
                 v_max * d_to   / D_b)   # braking buffer
 
    r_speed = -k_over  * max(0, v - v*)^2
              -k_under * max(0, v* - v)
    """
     v: float = state.current_speed
     v_max: float = state.speed_limit
     d_from: float = state.distance_from_last_station
     d_to: float = state.distance_to_next_station
     D_a:float = state.acceleration_buffer
     D_b:float = state.braking_buffer

     v_target: float = min(v_max,
                          v_max * (d_from/D_a) if D_a>0 else v_max,
                          v_max * (d_to/D_b) if D_b>0 else v_max)
     overspeed:float=max(0.0,v-v_target)
     underspeed:float=max(0.0, v_target-v)

     return -(K_OVER * overspeed**2) - (K_UNDER * underspeed)

def _compute_station_reward(state: TrainState) -> float:
    """
    2.1 Station reached milestone reward.
 
        r_station = +5   if reached_new_station
                  =  0   otherwise
    """
    return STATION_REWARD if state.reached_new_station else 0.0 
 
def _compute_punctuality_penalty(state: TrainState) ->float:
    """
    2.2 Punctuality penalty at station arrival.
 
    Only applied when the train has just reached a new station and
    both scheduled and actual arrival times are provided.
 
        r_time = -0.5 * |T_sched - T_arr|   if station reached
               =  0                          otherwise
    """
    if not state.reached_new_station:
        return 0.0
    if state.scheduled_arrival_time is None or state.actual_arrival_time is None:
        return 0.0
    return -PUNCTUALITY_FACTOR * abs(state.scheduled_arrival_time - state.actual_arrival_time)

