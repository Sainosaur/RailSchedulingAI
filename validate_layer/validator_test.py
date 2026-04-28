"""
validate_layer/validator_test.py

Unit tests for the Validation Layer (VL).
Run with:   python -m unittest validate_layer/validator_test.py
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.graph import VLSegment
from validate_layer import log_manager
from validate_layer.validator import ValidationLayer

_TEST_LOG = os.path.join(str(Path(__file__).resolve().parent), "override_log_test.csv")


class _BaseValidatorTest(unittest.TestCase):
    @staticmethod
    def _make_blocks(start: float, end: float, sh: float) -> list[float]:
        bounds = []
        pos = start
        while pos < end:
            bounds.append(pos)
            pos += sh
        if bounds[-1] < end:
            bounds.append(end)
        return bounds

    def setUp(self):
        self.vl = ValidationLayer.__new__(ValidationLayer)
        self.vl.segments = [
            VLSegment(
                "S0", 582, 5481, 25.00, 312.5, 12.5, self._make_blocks(582, 5481, 312.5)
            ),
            VLSegment(
                "S1",
                5481,
                14951,
                25.00,
                312.5,
                12.5,
                self._make_blocks(5481, 14951, 312.5),
            ),
            VLSegment(
                "S2",
                14951,
                37160,
                16.67,
                138.9,
                8.3,
                self._make_blocks(14951, 37160, 138.9),
            ),
            VLSegment(
                "S3",
                37160,
                47017,
                8.33,
                34.7,
                4.2,
                self._make_blocks(37160, 47017, 34.7),
            ),
            VLSegment(
                "S4",
                47017,
                67394,
                8.33,
                34.7,
                4.2,
                self._make_blocks(47017, 67394, 34.7),
            ),
            VLSegment(
                "S5",
                67394,
                76651,
                8.33,
                34.7,
                4.2,
                self._make_blocks(67394, 76651, 34.7),
            ),
        ]
        self.vl.dt = 1.0
        log_manager._LOG_PATH = _TEST_LOG
        log_manager.init_log()

    def tearDown(self):
        if os.path.exists(_TEST_LOG):
            os.remove(_TEST_LOG)


# -----------------------------------------------------------------------
# DTZ Tests — compute_dtz()
# -----------------------------------------------------------------------


class TestComputeDTZ(_BaseValidatorTest):
    def test_dtz_mid_segment_S0(self):
        x = 1000.0
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)
        self.assertLessEqual(dtz, 312.5)

    def test_dtz_on_block_boundary_S0(self):
        x = 894.5
        dtz = self.vl.compute_dtz(x)
        self.assertAlmostEqual(dtz, 312.5, places=1)


# -----------------------------------------------------------------------
# Speed-for-Aspect Tests — _speed_for_aspect()
# -----------------------------------------------------------------------


class TestSpeedForAspect(_BaseValidatorTest):
    def test_red_is_dtz(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(0, seg, 100.0, 10.0)
        self.assertAlmostEqual(dist, 100.0)

    def test_green_is_dtz_plus_3_sh(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(3, seg, 100.0, 10.0)
        self.assertAlmostEqual(dist, 100.0 + (3 * seg.spatial_headway))

    def test_orange_is_dtz_plus_1_sh(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(1, seg, 100.0, 10.0)
        self.assertAlmostEqual(dist, 100.0 + (1 * seg.spatial_headway))

    def test_flash_green_is_dtz_plus_2_sh(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(2, seg, 100.0, 10.0)
        self.assertAlmostEqual(dist, 100.0 + (2 * seg.spatial_headway))


# -----------------------------------------------------------------------
# SUVAT Projection Tests — _project()
# -----------------------------------------------------------------------


class TestProject(_BaseValidatorTest):
    def test_coast(self):
        # u=10 m/s, no acceleration. SUVAT: s = u*t = 10*5 = 50m. v = u = 10 m/s.
        x, u, proposed_a, v_ceil, t = 1000.0, 10.0, 0.0, 25.0, 5.0
        x_proj, v = self.vl._project(x, u, proposed_a, v_ceil, t)
        self.assertAlmostEqual(v, 10.0)
        self.assertAlmostEqual(x_proj, 1050.0)

    def test_accelerating(self):
        # u=0, a=0.5. SUVAT: s = u*t + 0.5*a*t² = 0 + 0.5*0.5*100 = 25m. v = u + a*t = 5 m/s.
        x, u, proposed_a, v_ceil, t = 1000.0, 0.0, 0.5, 25.0, 10.0
        x_proj, v = self.vl._project(x, u, proposed_a, v_ceil, t)
        self.assertAlmostEqual(v, 5.0)
        self.assertAlmostEqual(x_proj, 1025.0)

    def test_heavy_braking_never_negative(self):
        # u=2, a=-1.0. Train stops at t_stop=2s. Remaining 13s the train stays at rest.
        # v must not go negative (no reverse motion in physics model).
        x, u, proposed_a, v_ceil, t = 1000.0, 2.0, -1.0, 0.0, 15.0
        x_proj, v = self.vl._project(x, u, proposed_a, v_ceil, t)
        self.assertGreaterEqual(v, 0.0)


# -----------------------------------------------------------------------
# Direct Interceptor Tests — get_safe_action()
# -----------------------------------------------------------------------


class TestGetSafeAction(_BaseValidatorTest):
    def test_safe_action_continuous_green(self):
        # Scenario: open track, Green aspect, large clearance.
        # distance_available = dtz + 3*SH = 300 + 3*312.5 = 1237.5m.
        # Projected trajectory is well within bounds — no override expected.
        proposed_a = +0.5
        env_aspect = 3
        x, u, dtz = 1000.0, 10.0, 300.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, u, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, 0.5)

    def test_override_spatial_violation(self):
        # Scenario: Red aspect (clearance = dtz only = 10m), u=20 m/s.
        # distance_available = 10m.
        # At t_stop=25s first sim step (t~8s): x_proj ~= 1160m >> boundary 1010m.
        # Spatial violation fires. a_needed = -u²/(2s) = -(400)/(20) = -20 m/s².
        # Clamped to EMERGENCY_DECEL: safe_a = -1.0.
        # overridden=True tells us the VL intervened.
        # safe_a=-1.0 tells us the violation was so severe that even emergency braking
        # is the minimum physically available response.
        proposed_a = +0.5
        env_aspect = 0
        x, u, dtz = 1000.0, 20.0, 10.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, u, dtz)
        self.assertTrue(overridden)
        self.assertEqual(safe, -1.0)

    def test_clamp_extreme_action(self):
        # Scenario: PPO outputs -5.0 m/s² (outside the action space [-1.0, 0.5]).
        # Hardware clamp: clamped_a = max(-1.0, min(0.5, -5.0)) = -1.0.
        # Safety check runs on -1.0: train decelerates safely, no safety violation detected.
        # Current logic: Only set overridden=True if AI was LESS safe than required.
        # Since -5.0 is MORE safe (more braking) than needed, overridden should be False.
        proposed_a = -5.0
        env_aspect = 3
        x, u, dtz = 1000.0, 10.0, 300.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, u, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, -1.0)

    def test_override_segment_overspeed(self):
        # Scenario: Green aspect, approaching x=14951m (S2 boundary, limit=16.67 m/s), u=20 m/s.
        # distance_available = dtz + 3*SH = 300 + 3*312.5 = 1237.5m.
        # At first sim step (~t=8s): x_proj enters S2 where limit=16.67 m/s and v=24 m/s.
        # S2_Limit violation fires despite Green aspect.
        # Clamping for speed limits is currently considered a 'System Clamp' (overridden=False).
        proposed_a = +0.5
        env_aspect = 3
        x, u, dtz = 14900.0, 20.0, 300.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, u, dtz)
        self.assertFalse(overridden)
        # a_needed = (v_target - u - 0.01) / self.dt
        # v_target for Green at S2 is min(sqrt(2*1*1237.5), 16.67) = 16.67
        # a_needed = (16.67 - 20.0 - 0.01) / 1.0 = -3.34
        # Clamped to EMERGENCY_DECEL = -1.0
        self.assertEqual(safe, -1.0)


# -----------------------------------------------------------------------
# XAI Log Tests
# -----------------------------------------------------------------------


class TestXAILog(_BaseValidatorTest):
    def test_override_creates_log_entry(self):
        # Scenario: Spatial violation (same as test_override_spatial_violation).
        # Any override must append a row to override_log.csv.
        # Verifies Layer 4 XAI logging executes for every Layer 3 intervention.
        proposed_a, env_aspect = +0.5, 0
        x, u, dtz = 1000.0, 20.0, 10.0
        self.vl.get_safe_action(proposed_a, env_aspect, x, u, dtz)

        with open(_TEST_LOG) as f:
            lines = f.readlines()
        self.assertGreaterEqual(len(lines), 2)  # header + at least 1 data row


if __name__ == "__main__":
    unittest.main()
