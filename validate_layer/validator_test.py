"""
validate_layer/validator_test.py

Unit tests for the Validation Layer (VL).
Run with:   python -m unittest validate_layer/validator_test.py
"""

import os
import unittest
import sys
from pathlib import Path

# Ensure project-root imports work regardless of working directory
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.graph import VLSegment
from validate_layer.validator import ValidationLayer
from validate_layer import log_manager


_TEST_LOG = os.path.join(
    str(Path(__file__).resolve().parent), "override_log_test.csv"
)

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
            VLSegment("S0",   582,  5481, 25.00, 312.5, 12.5, self._make_blocks(582, 5481, 312.5)),
            VLSegment("S1",  5481, 14951, 25.00, 312.5, 12.5, self._make_blocks(5481, 14951, 312.5)),
            VLSegment("S2", 14951, 37160, 16.67, 138.9,  8.3, self._make_blocks(14951, 37160, 138.9)),
            VLSegment("S3", 37160, 47017,  8.33,  34.7,  4.2, self._make_blocks(37160, 47017, 34.7)),
            VLSegment("S4", 47017, 67394,  8.33,  34.7,  4.2, self._make_blocks(47017, 67394, 34.7)),
            VLSegment("S5", 67394, 76651,  8.33,  34.7,  4.2, self._make_blocks(67394, 76651, 34.7)),
        ]
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
        _, dist = self.vl._speed_for_aspect(0, seg, 100.0)
        self.assertAlmostEqual(dist, 100.0)

    def test_green_is_dtz_plus_3_sh(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(3, seg, 100.0)
        self.assertAlmostEqual(dist, 100.0 + (3 * seg.spatial_headway))

    def test_orange_is_dtz_plus_1_sh(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(1, seg, 100.0)
        self.assertAlmostEqual(dist, 100.0 + (1 * seg.spatial_headway))

    def test_flash_green_is_dtz_plus_2_sh(self):
        seg = self.vl.get_segment(1000.0)
        _, dist = self.vl._speed_for_aspect(2, seg, 100.0)
        self.assertAlmostEqual(dist, 100.0 + (2 * seg.spatial_headway))


# -----------------------------------------------------------------------
# SUVAT Projection Tests — _project()
# -----------------------------------------------------------------------

class TestProject(_BaseValidatorTest):
    def test_coast(self):
        x, v, proposed_a, limit_v, t = 1000.0, 10.0, 0.0, 25.0, 5.0
        x_proj, v_proj = self.vl._project(x, v, proposed_a, limit_v, t)
        self.assertAlmostEqual(v_proj, 10.0)
        self.assertAlmostEqual(x_proj, 1050.0)

    def test_accelerating(self):
        x, v, proposed_a, limit_v, t = 1000.0, 0.0, 0.5, 25.0, 10.0
        x_proj, v_proj = self.vl._project(x, v, proposed_a, limit_v, t)
        self.assertAlmostEqual(v_proj, 5.0)
        self.assertAlmostEqual(x_proj, 1025.0)

    def test_heavy_braking_never_negative(self):
        x, v, proposed_a, limit_v, t = 1000.0, 2.0, -1.0, 0.0, 15.0
        x_proj, v_proj = self.vl._project(x, v, proposed_a, limit_v, t)
        self.assertGreaterEqual(v_proj, 0.0)


# -----------------------------------------------------------------------
# Direct Interceptor Tests — get_safe_action()
# -----------------------------------------------------------------------

class TestGetSafeAction(_BaseValidatorTest):
    def test_safe_action_continuous_green(self):
        proposed_a = +0.5
        env_aspect = 3
        x, v, dtz = 1000.0, 10.0, 300.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, v, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, 0.5)

    def test_override_spatial_violation(self):
        proposed_a = +0.5
        env_aspect = 0
        x, v, dtz = 1000.0, 20.0, 10.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, v, dtz)
        self.assertTrue(overridden)
        self.assertEqual(safe, -1.0)
        
    def test_clamp_extreme_action(self):
        proposed_a = -5.0
        env_aspect = 3
        x, v, dtz = 1000.0, 10.0, 300.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, v, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, -1.0)
        
    def test_override_temporal_violation(self):
        proposed_a = +0.5
        env_aspect = 0
        x, v, dtz = 1000.0, 15.0, 100.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, v, dtz)
        self.assertTrue(overridden)
        self.assertEqual(safe, -1.0)

    def test_override_segment_overspeed(self):
        proposed_a = +0.5
        env_aspect = 3
        x, v, dtz = 14900.0, 20.0, 300.0
        safe, overridden = self.vl.get_safe_action(proposed_a, env_aspect, x, v, dtz)
        self.assertTrue(overridden)
        self.assertEqual(safe, -1.0)


# -----------------------------------------------------------------------
# XAI Log Tests
# -----------------------------------------------------------------------

class TestXAILog(_BaseValidatorTest):
    def test_override_creates_log_entry(self):
        proposed_a, env_aspect = +0.5, 0
        x, v, dtz = 1000.0, 20.0, 10.0
        self.vl.get_safe_action(proposed_a, env_aspect, x, v, dtz)

        with open(_TEST_LOG) as f:
            lines = f.readlines()
        self.assertGreaterEqual(len(lines), 2)


if __name__ == "__main__":
    unittest.main()
