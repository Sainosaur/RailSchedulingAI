"""
validate_layer/validator_test.py

Unit tests for the Validation Layer (VL).
Run with:   python -m unittest validate_layer/validator_test.py
From:       /home/dharms/RailSchedulingAI/
"""

import os
import sys
import unittest
from pathlib import Path

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from graph.graph import VLSegment
from validate_layer.validator import ValidationLayer
from validate_layer import log_manager


# Path for the test-only log (inside validate_layer/)
_TEST_LOG = os.path.join(
    str(Path(__file__).resolve().parent), "override_log_test.csv"
)


class _BaseValidatorTest(unittest.TestCase):
    """
    Shared setup for all validator tests.

    Creates a ValidationLayer with hardcoded segments (no geocoding API
    calls) and a temporary override log that is cleaned up after each test.
    """

    @staticmethod
    def _make_blocks(start: float, end: float, sh: float) -> list[float]:
        """Build SH-spaced block boundaries within [start, end)."""
        bounds = []
        pos = start
        while pos < end:
            bounds.append(pos)
            pos += sh
        if bounds[-1] < end:
            bounds.append(end)
        return bounds

    def setUp(self):
        """Create a VL instance with hardcoded segments for testing."""
        self.vl = ValidationLayer.__new__(ValidationLayer)
        self.vl.segments = [
            VLSegment("S0",   582,  5481, 25.00, 312.5, 12.5,
                      self._make_blocks(582, 5481, 312.5)),
            VLSegment("S1",  5481, 14951, 25.00, 312.5, 12.5,
                      self._make_blocks(5481, 14951, 312.5)),
            VLSegment("S2", 14951, 37160, 16.67, 138.9,  8.3,
                      self._make_blocks(14951, 37160, 138.9)),
            VLSegment("S3", 37160, 47017,  8.33,  34.7,  4.2,
                      self._make_blocks(37160, 47017, 34.7)),
            VLSegment("S4", 47017, 67394,  8.33,  34.7,  4.2,
                      self._make_blocks(47017, 67394, 34.7)),
            VLSegment("S5", 67394, 76651,  8.33,  34.7,  4.2,
                      self._make_blocks(67394, 76651, 34.7)),
        ]
        # Point the log manager at a test-only file
        log_manager._LOG_PATH = _TEST_LOG
        log_manager.init_log()

    def tearDown(self):
        """Remove the temporary override log after each test."""
        if os.path.exists(_TEST_LOG):
            os.remove(_TEST_LOG)


# -----------------------------------------------------------------------
# DTZ Tests — compute_dtz()
# -----------------------------------------------------------------------

class TestComputeDTZ(_BaseValidatorTest):
    """Tests for compute_dtz(): distance to next block boundary."""

    def test_dtz_mid_segment_S0(self):
        """DTZ should be > 0 and ≤ SH when inside a segment."""
        x = ...    # position in metres, inside S0
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)
        self.assertLessEqual(dtz, 312.5)   # SH for S0

    def test_dtz_on_block_boundary_S0(self):
        """When exactly on a boundary, DTZ should jump to the next one."""
        x = ...    # position in metres, exactly on a block boundary in S0
        dtz = self.vl.compute_dtz(x)
        self.assertAlmostEqual(dtz, 312.5, places=1)  # full SH

    def test_dtz_inside_S3(self):
        """DTZ should be ≤ SH (34.7 m) on S3."""
        x = ...    # position in metres, inside S3
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)
        self.assertLessEqual(dtz, 34.7 + 0.1)   # SH for S3 + tolerance

    def test_dtz_near_segment_boundary(self):
        """DTZ near end of a segment should point to the next segment."""
        x = ...    # position in metres, near the end of a segment
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)


# -----------------------------------------------------------------------
# Segment Lookup Tests — get_segment()
# -----------------------------------------------------------------------

class TestGetSegment(_BaseValidatorTest):
    """Tests for get_segment(): find which segment a position belongs to."""

    def test_segment_lookup_S0(self):
        """A position inside S0 should return segment S0."""
        x = ...    # position in metres, inside S0
        seg = self.vl.get_segment(x)
        self.assertEqual(seg.id, "S0")

    def test_segment_lookup_S3(self):
        """A position inside S3 should return segment S3."""
        x = ...    # position in metres, inside S3
        seg = self.vl.get_segment(x)
        self.assertEqual(seg.id, "S3")

    def test_segment_clamp_before_track(self):
        """A position before the track should clamp to S0."""
        x = ...    # position in metres, before S0 start (582)
        seg = self.vl.get_segment(x)
        self.assertEqual(seg.id, "S0")

    def test_segment_clamp_after_track(self):
        """A position after the track should clamp to S5."""
        x = ...    # position in metres, after S5 end (76651)
        seg = self.vl.get_segment(x)
        self.assertEqual(seg.id, "S5")


# -----------------------------------------------------------------------
# Speed-for-Action Tests — _speed_for_action()
# -----------------------------------------------------------------------

class TestSpeedForAction(_BaseValidatorTest):
    """Tests for _speed_for_action(): 4-aspect to speed mapping."""

    def test_red_is_zero(self):
        """Action 0 (Red) should always map to 0 m/s."""
        seg = self.vl.get_segment(1000.0)  # any segment
        speed = self.vl._speed_for_action(0, seg)
        self.assertAlmostEqual(speed, 0.0)

    def test_green_equals_segment_limit(self):
        """Action 3 (Green) should map to the segment speed limit."""
        x = ...    # position in metres, pick a segment
        seg = self.vl.get_segment(x)
        speed = self.vl._speed_for_action(3, seg)
        self.assertAlmostEqual(speed, seg.limit_ms)

    def test_yellow_is_one_third(self):
        """Action 1 (Yellow) should map to 1/3 of the segment limit."""
        x = ...    # position in metres, pick a segment
        seg = self.vl.get_segment(x)
        speed = self.vl._speed_for_action(1, seg)
        self.assertAlmostEqual(speed, seg.limit_ms / 3.0, places=2)

    def test_double_yellow_is_two_thirds(self):
        """Action 2 (Double-Yellow) should map to 2/3 of the segment limit."""
        x = ...    # position in metres, pick a segment
        seg = self.vl.get_segment(x)
        speed = self.vl._speed_for_action(2, seg)
        self.assertAlmostEqual(speed, seg.limit_ms * 2.0 / 3.0, places=2)


# -----------------------------------------------------------------------
# SUVAT Projection Tests — _project()
# -----------------------------------------------------------------------

class TestProject(_BaseValidatorTest):
    """Tests for _project(): two-phase SUVAT forward projection."""

    def test_cruise_at_target(self):
        """When v == target_v, train should cruise (no acceleration)."""
        x = ...       # start position (m)
        v = ...       # current speed (m/s), equal to target
        target_v = v  # same as v
        t = ...       # lookahead time (s)
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertAlmostEqual(v_proj, v)
        self.assertAlmostEqual(x_proj, x + v * t)

    def test_accelerating(self):
        """Accelerating from rest: position and speed should increase."""
        x = ...         # start position (m)
        v = ...         # current speed (m/s), low or zero
        target_v = ...  # target speed (m/s), higher than v
        t = ...         # lookahead time (s)
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertGreater(v_proj, v)
        self.assertGreater(x_proj, x)

    def test_braking(self):
        """Braking: speed should decrease towards target."""
        x = ...         # start position (m)
        v = ...         # current speed (m/s), high
        target_v = ...  # target speed (m/s), lower than v
        t = ...         # lookahead time (s)
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertLess(v_proj, v)

    def test_speed_never_negative(self):
        """Speed should never go below zero, even with heavy braking."""
        x = ...         # start position (m)
        v = ...         # current speed (m/s), low
        target_v = 0.0  # emergency stop
        t = ...         # lookahead time (s), long enough to stop
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertGreaterEqual(v_proj, 0.0)


# -----------------------------------------------------------------------
# Safety Sieve Tests — get_safe_action()
# -----------------------------------------------------------------------

class TestGetSafeAction(_BaseValidatorTest):
    """
    Tests for get_safe_action(): the Layer 4 decision node.

    Each test injects:
        proposed_act : int    — the PPO-proposed action (0–3)
        x            : float  — current position (m)
        v            : float  — current speed (m/s)
        dtz          : float  — distance to next block boundary (m), ≤ SH
    """

    def test_green_passes_when_safe(self):
        """Green should pass when speed is low and DTZ is large."""
        proposed_act = ...  # action (0–3)
        x = ...             # position (m)
        v = ...             # speed (m/s)
        dtz = ...           # distance to zone (m), ≤ SH
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, proposed_act)

    def test_green_overrides_when_close(self):
        """Green should be overridden when DTZ is tiny."""
        proposed_act = ...  # action (0–3)
        x = ...             # position (m)
        v = ...             # speed (m/s)
        dtz = ...           # distance to zone (m), very small
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)
        self.assertLess(safe, proposed_act)

    def test_red_always_safe(self):
        """Red (stop) should always be accepted, no matter the DTZ."""
        proposed_act = 0
        x = ...    # position (m)
        v = ...    # speed (m/s), can be zero
        dtz = ...  # distance to zone (m)
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, 0)

    def test_spatial_violation(self):
        """Projected position past boundary → must override."""
        proposed_act = ...  # action (0–3)
        x = ...             # position (m)
        v = ...             # speed (m/s), high enough to overshoot
        dtz = ...           # distance to zone (m), very small
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)

    def test_temporal_violation(self):
        """Not enough reaction time → must override."""
        proposed_act = ...  # action (0–3)
        x = ...             # position (m)
        v = ...             # speed (m/s)
        dtz = ...           # distance to zone (m)
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)

    def test_segment_violation(self):
        """Crossing into a slower segment at speed → must override."""
        proposed_act = ...  # action (0–3)
        x = ...             # position (m), near a segment boundary
        v = ...             # speed (m/s), above next segment's limit
        dtz = ...           # distance to zone (m)
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)


# -----------------------------------------------------------------------
# XAI Log Tests — _log_override()
# -----------------------------------------------------------------------

class TestXAILog(_BaseValidatorTest):
    """Tests for Layer 5: override logging to CSV."""

    def test_override_creates_log_entry(self):
        """When an override occurs, override_log_test.csv should gain a row."""
        proposed_act = ...  # action that will be overridden
        x = ...             # position (m)
        v = ...             # speed (m/s)
        dtz = ...           # distance to zone (m), chosen to force override
        self.vl.get_safe_action(proposed_act, x, v, dtz)

        with open(_TEST_LOG) as f:
            lines = f.readlines()
        # Header + at least one data row
        self.assertGreaterEqual(len(lines), 2)

    def test_no_override_no_log_entry(self):
        """When no override occurs, only the header row should exist."""
        proposed_act = ...  # action that will pass safely
        x = ...             # position (m)
        v = ...             # speed (m/s)
        dtz = ...           # distance to zone (m), chosen so action passes
        self.vl.get_safe_action(proposed_act, x, v, dtz)

        with open(_TEST_LOG) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)  # header only


if __name__ == "__main__":
    unittest.main()
