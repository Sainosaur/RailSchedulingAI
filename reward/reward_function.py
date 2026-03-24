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
PUNCTUALITY_FACTOR: float = 5.0
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
    previouse_position: float        #psition at previous timestamp
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