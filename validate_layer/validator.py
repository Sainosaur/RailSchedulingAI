"""
Draft: 2
This draft was fully AI-generated - but used my specification in documentation.
Next steps are to read and understand code, find errors and fix them.
Once done, then make a separate test file to force values and see if code works as expected.

validate_layer/validator.py

This module implements a deterministic, safety-critical Validation Layer (VL)
for a PPO-driven autonomous train agent. It acts as a 5-Layer "Safety Sieve"
based on ETCS Level 2 Fixed Block (Train-to-Zone) signaling principles.
"""

# Save as validator.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np


@dataclass
class Segment:
    id: str
    start: float
    end: float
    limit_ms: float


class ValidationLayer:
    def __init__(self):
        # Coordinates from internal_layers.md [cite: 3]
        self.segments = [
            Segment("S0", 582, 5481, 25.0),
            Segment("S1", 5481, 14951, 25.0),
            Segment("S2", 14951, 37160, 16.67),
            Segment("S3", 37160, 47017, 8.33),
            Segment("S4", 47017, 67394, 8.33),
            Segment("S5", 67394, 76651, 8.33),
        ]
        self.lookahead = [5, 10, 15]  #

    def get_safe_action(self, proposed_act, x, v, dtz):
        current_seg = next(s for s in self.segments if s.start <= x < s.end)
        # 4-Aspect Mapping
        speed_map = {0: 0.0, 1: 8.33, 2: 16.67, 3: current_seg.limit_ms}

        # Layer 4b: Sieve
        for act in range(proposed_act, -1, -1):
            target_v = speed_map[act]
            accel = 0.5 if target_v >= v else -1.0

            is_safe = True
            for t in self.lookahead:
                v_proj = max(0, min(target_v, v + accel * t))
                x_proj = x + v * t + 0.5 * accel * (t**2)

                # Check violations
                if x_proj >= dtz or v_proj > current_seg.limit_ms:
                    is_safe = False
                    break

            if is_safe:
                return act, (act != proposed_act)
        return 0, True  # Emergency Stop fallback
