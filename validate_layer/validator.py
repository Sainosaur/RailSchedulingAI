"""
Draft: 1
This draft was full AI-generated, but used my specification in documentation.
Next steps are to read and understand code, find errors and fix them.
Once done, then make a separate test file to force values and see if code works as expected.

validate_layer/validator.py

This module implements a deterministic, safety-critical Validation Layer (VL)
for a PPO-driven autonomous train agent. It acts as a 5-Layer "Safety Sieve"
based on ETCS Level 2 Fixed Block (Train-to-Zone) signaling principles.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass(frozen=True)
class Segment:
    """
    Represents a Fixed Block/Zone with a Virtual Balise at the start.
    Frozen for immutability and performance during RL training.

    Attributes:
        id (str): The unique identifier for the segment (e.g., "S1").
        start_m (float): The physical start coordinate of the zone in meters.
        end_m (float): The physical end coordinate of the zone in meters.
        limit_kmh (float): The maximum allowed speed in km/h.
        limit_ms (float): The pre-computed maximum allowed speed in m/s.
    """

    id: str
    start_m: float
    end_m: float
    limit_kmh: float
    limit_ms: float = field(init=False)

    def __post_init__(self):
        """Pre-compute m/s conversion upon initialization to save CPU cycles in the loop."""
        object.__setattr__(self, "limit_ms", self.limit_kmh / 3.6)


class InternalValidator:
    """
    The Safety Sieve intercepting PPO actions. Validates spatial and temporal
    constraints using SUVAT projections to guarantee ETCS Level 2 compliance.
    """

    def __init__(self) -> None:
        # Layer 0: The Synthetic Track Map (ETCS L2 Representative)
        self._segments: List[Segment] = [
            Segment("S0", 0.0, 2000.0, 90.0),
            Segment("S1", 2000.0, 4000.0, 90.0),
            Segment("S2", 4000.0, 6000.0, 60.0),
            Segment("S3", 6000.0, 7500.0, 30.0),
            Segment("S4", 7500.0, 9000.0, 30.0),
            Segment("S5", 9000.0, 10000.0, 30.0),
        ]

        # Safety Constants
        self.a_service: float = -1.0  # m/s^2 (Standard Service Braking)
        self.lookahead_times: Tuple[float, float, float] = (5.0, 10.0, 15.0)

    def _get_active_segment(self, x: float) -> Optional[Segment]:
        """
        Layer 1: Spatial Ingestion (The Lookup)
        Identifies which ETCS segment the train occupies at a given coordinate.
        """
        for seg in self._segments:
            if seg.start_m <= x < seg.end_m:
                return seg
        return None

    def _predict_future_state(
        self, x_initial: float, v_initial: float, a: float, t: float
    ) -> Tuple[float, float]:
        """
        Layer 2: SUVAT Simulation
        Projects the train's future position and velocity under constant acceleration.
        """
        future_x = x_initial + (v_initial * t) + (0.5 * a * (t**2))
        future_v = max(0.0, v_initial + (a * t))
        return future_x, future_v

    def validate_action(
        self, current_x: float, current_v: float, proposed_a: float
    ) -> Tuple[float, bool]:
        """
        The 5-Layer Entry Point for the PPO Environment.
        Evaluates the proposed acceleration against future spatial and velocity constraints.

        Args:
            current_x (float): The train's current position in meters.
            current_v (float): The train's current velocity in m/s.
            proposed_a (float): The acceleration requested by the PPO agent.

        Returns:
            Tuple[float, bool]: (Safe Acceleration to apply, Was_Overridden flag).
        """
        # Early-Exit Check: Iterate through lookahead buffers
        for t in self.lookahead_times:
            # Layer 2: Project state
            pred_x, pred_v = self._predict_future_state(
                current_x, current_v, proposed_a, t
            )

            # Layer 1: Identify future zone
            target_seg = self._get_active_segment(pred_x)

            # Layer 5: Spatial Exit (Off-track / End of MA)
            if target_seg is None:
                return self.a_service, True

            # Layer 3 & 4: Violation Check & Deterministic Override
            if pred_v > target_seg.limit_ms:
                return self.a_service, True

        # Action is safe
        return proposed_a, False
