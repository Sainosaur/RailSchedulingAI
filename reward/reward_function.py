"""
Railway RL Reward Function
 
Structure:
  - Continuous rewards  (every timestep)
  - Event rewards       (on specific events)
  - Terminal penalties  (once, ends episode)
 
Total: r_t = r_continuous + r_event + r_terminal
"""
from dataclasses import dataclass
from typing import optional 

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