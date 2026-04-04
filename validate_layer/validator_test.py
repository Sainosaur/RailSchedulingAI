"""
validate_layer/validator_test.py

Unit tests for the Validation Layer (VL).
Run with:   python -m unittest validate_layer/validator_test.py
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
        """
        DTZ should be > 0 and ≤ SH when inside a segment.

        Expected value derivation
        -------------------------
        S0 block boundaries start at 582 and repeat every SH = 312.5 m:
            582.0, 894.5, 1207.0, 1519.5, …

        Position x = 1000.0 sits between boundaries 894.5 and 1207.0.
        DTZ = next_boundary − x = 1207.0 − 1000.0 = 207.0 m.

        Assert: 0 < DTZ ≤ 312.5  (within one SH of S0).
        """
        x = 1000.0
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)
        self.assertLessEqual(dtz, 312.5)   # SH for S0

    def test_dtz_on_block_boundary_S0(self):
        """
        When exactly on a boundary, DTZ should jump to the next one.

        Expected value derivation
        -------------------------
        S0 boundaries: 582.0, 894.5, 1207.0, …
        Position x = 894.5 is exactly on the second boundary.

        compute_dtz() iterates block_boundaries and returns the first
        boundary *strictly greater than* x.  Since 894.5 is not > 894.5,
        it skips that entry and finds 1207.0.

        DTZ = 1207.0 − 894.5 = 312.5 m  (exactly one full SH).
        """
        x = 894.5
        dtz = self.vl.compute_dtz(x)
        self.assertAlmostEqual(dtz, 312.5, places=1)  # full SH

    def test_dtz_inside_S3(self):
        """
        DTZ should be ≤ SH (34.7 m) on S3.

        Expected value derivation
        -------------------------
        S3 runs from 37160 to 47017 with SH = 34.7 m.
        Boundaries: 37160.0, 37194.7, 37229.4, 37264.1, …

        Position x = 37180.0 sits between 37160.0 and 37194.7.
        DTZ = 37194.7 − 37180.0 = 14.7 m.

        Assert: 0 < DTZ ≤ 34.8. We allow a tiny float tolerance (e.g. + 0.1)
        because in Python, floating point arithmetic can sometimes
        yield e.g. 14.700000000000003 instead of exactly 14.7.
        """
        x = 37180.0
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)
        self.assertLessEqual(dtz, 34.7 + 0.1)   # SH for S3 + tolerance

    def test_dtz_near_segment_boundary(self):
        """
        DTZ near the end of a segment should point to the next segment.

        Expected value derivation
        -------------------------
        S0 ends at 5481.  The last block boundaries in S0 are:
            …, 4957, 5269.5, 5481 (segment end).

        The 16th block begins at 5269.5 and ends at 5481.0, so it is a
        short final block of length 211.5 m.

        Position x = 5480.0 is 1 m before the S0 end-boundary (5481).
        DTZ = 5481, the segment-end boundary, minus 5480.0 = 1.0 m.

        Assert: DTZ > 0 (the train has not yet reached the boundary).
        """
        x = 5480.0
        dtz = self.vl.compute_dtz(x)
        self.assertGreater(dtz, 0)


# -----------------------------------------------------------------------
# Segment Lookup Tests — get_segment()
# -----------------------------------------------------------------------

class TestGetSegment(_BaseValidatorTest):
    """Tests for get_segment(): find which segment a position belongs to."""

    def test_segment_lookup_S0(self):
        """
        A position inside S0 should return segment S0.

        Expected value derivation
        -------------------------
        S0 spans [582, 5481).  x = 1000 clearly falls within that interval.
        get_segment() iterates segments and returns the first where
        seg.start ≤ x < seg.end  →  582 ≤ 1000 < 5481  →  S0.
        """
        x = 1000.0
        seg = self.vl.get_segment(x)
        self.assertEqual(seg.id, "S0")

    def test_segment_lookup_S3(self):
        """
        A position inside S3 should return segment S3.

        Expected value derivation
        -------------------------
        S3 spans [37160, 47017).  x = 40000 falls inside.
        37160 ≤ 40000 < 47017  →  S3.
        """
        x = 40000.0
        seg = self.vl.get_segment(x)
        self.assertEqual(seg.id, "S3")

    def test_segment_clamp_before_track(self):
        """
        A position before the track should clamp to S0.

        Expected value derivation
        -------------------------
        The track begins at S0.start = 582.  x = 100 is before that.
        This represents an invalid position outside the track bounds.
        get_segment() is designed to raise a ValueError in this case.
        """
        x = 100.0
        with self.assertRaises(ValueError):
            seg = self.vl.get_segment(x)

    def test_segment_clamp_after_track(self):
        """
        A position after the track should clamp to S5.

        Expected value derivation
        -------------------------
        The track ends at S5.end = 76651.  x = 80000 is beyond that.
        This represents an invalid position.
        get_segment() should raise a ValueError.
        """
        x = 80000.0
        with self.assertRaises(ValueError):
            seg = self.vl.get_segment(x)


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
        """
        Action 3 (Green) should map to the segment speed limit.

        Expected value derivation
        -------------------------
        x = 1000 → S0 (limit_ms = 25.0).
        fractions[3] = 1.0  →  speed = 1.0 × 25.0 = 25.0 m/s.
        """
        x = 1000.0
        seg = self.vl.get_segment(x)
        speed = self.vl._speed_for_action(3, seg)
        self.assertAlmostEqual(speed, seg.limit_ms)

    def test_yellow_is_one_third(self):
        """
        Action 1 (Yellow) should map to 1/3 of the segment limit.

        Expected value derivation
        -------------------------
        x = 40000 → S3 (limit_ms = 8.33).
        fractions[1] = 1/3  →  speed = (1/3) × 8.33 ≈ 2.777 m/s.
        """
        x = 40000.0
        seg = self.vl.get_segment(x)
        speed = self.vl._speed_for_action(1, seg)
        self.assertAlmostEqual(speed, seg.limit_ms / 3.0, places=2)

    def test_double_yellow_is_two_thirds(self):
        """
        Action 2 (Double-Yellow) should map to 2/3 of the segment limit.

        Expected value derivation
        -------------------------
        x = 40000 → S3 (limit_ms = 8.33).
        fractions[2] = 2/3  →  speed = (2/3) × 8.33 ≈ 5.553 m/s.
        """
        x = 40000.0
        seg = self.vl.get_segment(x)
        speed = self.vl._speed_for_action(2, seg)
        # `places=2` ensures the assertion passes if they match up to
        # 2 decimal places. 8.33 * 2/3 = 5.55333..., so we check
        # approximately 5.55.
        self.assertAlmostEqual(speed, seg.limit_ms * 2.0 / 3.0, places=2)


# -----------------------------------------------------------------------
# SUVAT Projection Tests — _project()
# -----------------------------------------------------------------------

class TestProject(_BaseValidatorTest):
    """Tests for _project(): two-phase SUVAT forward projection."""

    def test_cruise_at_target(self):
        """
        When v == target_v, train should cruise (no acceleration).

        Expected value derivation
        -------------------------
        x = 1000, v = 10.0, target_v = 10.0, t = 5.
        |target_v − v| = 0 < 0.01 → cruise branch.
          x_proj = x + v × t = 1000 + 10 × 5 = 1050.
          v_proj = v = 10.0.
        """
        x = 1000.0
        v = 10.0
        target_v = v        # same as v → cruise
        t = 5.0
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertAlmostEqual(v_proj, v)
        # assertAlmostEqual handles potential minor floating point math
        # inaccuracies built into standard IEEE 754 floats.
        self.assertAlmostEqual(x_proj, x + v * t)

    def test_accelerating(self):
        """
        Accelerating from rest: position and speed should increase.

        Expected value derivation
        -------------------------
        x = 1000, v = 0.0, target_v = 25.0, t = 10.
        Note: `t` represents the lookahead time for the SUVAT equation
        (it can be 5, 10, or 15 seconds in the simulation steps).
        
        accel = +0.5 m/s² (ACCEL constant, since target_v > v).
        t_to_target = (25 − 0) / 0.5 = 50 s.
        Since t (10) < t_to_target (50), the train is still accelerating
        throughout the entire lookahead:
        
          # v_proj predicts the speed 10s into the future
          v_proj = 0 + 0.5 × 10 = 5.0 m/s  (> 0 = v). 
          
          # x_proj predicts the geographical position 10s into the future
          x_proj = 1000 + 0 × 10 + 0.5 × 0.5 × 100 = 1025.0 m  (> 1000). 
        """
        x = 1000.0
        v = 0.0
        target_v = 25.0
        t = 10.0
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertGreater(v_proj, v)
        self.assertGreater(x_proj, x)

    def test_braking(self):
        """
        Braking: speed should decrease towards target.

        Expected value derivation
        -------------------------
        x = 5000, v = 20.0, target_v = 5.0, t = 10.
        accel = −1.0 m/s² (EMERGENCY_DECEL, since target_v < v).
        t_to_target = (5 − 20) / (−1) = 15 s.
        Since t (10) < t_to_target (15), the train is still braking
        throughout:
          v_proj = 20 + (−1) × 10 = 10.0 m/s.
        10.0 < 20.0 → speed decreased.
        """
        x = 5000.0
        v = 20.0
        target_v = 5.0
        t = 10.0
        x_proj, v_proj = self.vl._project(x, v, target_v, t)
        self.assertLess(v_proj, v)

    def test_speed_never_negative(self):
        """
        Speed should never go below zero, even with heavy braking.

        Expected value derivation
        -------------------------
        x = 1000, v = 2.0, target_v = 0.0, t = 15.
        accel = −1.0 m/s².
        t_to_target = (0 − 2) / (−1) = 2 s.
        Since t (15) > t_to_target (2), the two-phase branch fires:
          Phase 1: accelerate/brake for 2 s to reach target_v = 0.
          Phase 2: cruise at 0 m/s for 13 s.
          v_proj = target_v = 0.0.
        The max(0.0, v_proj) clamp ensures v_proj ≥ 0. 
        
        Note: Checking at multiple intervals (e.g. 5, 10, 15s) rather than 
        just 15s is necessary because a train might project safely at t=15s 
        (e.g., completely stopped) but during the intermediate time (t=5s)
        it might have overshot the block boundary before reversing or
        stopping. Step-by-step checks catch these mid-trajectory violations.
        """
        x = 1000.0
        v = 2.0
        target_v = 0.0      # emergency stop
        t = 15.0
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
        """
        Green should pass when speed is low and DTZ is large. 
        Explanation: If the train is moving slowly and the Distance-To-Zone 
        is very large, accelerating to top limit speed (Action: Green)
        won't cause a boundary overshoot within the lookahead window,
        so it is safe to permit.

        Expected value derivation
        -------------------------
        proposed_act = 3 (Green), x = 1000, v = 1.0, dtz = 300.0.
        Segment = S0 (limit = 25.0); target_v = 25.0 (Green = full limit).
        boundary_x = 1000 + 300 = 1300. 
        (Note: In Layer 4 tests, `dtz` is explicitly injected as an
        isolated parameter to independently test the simulation sieve logic, 
        bypassing the real track boundaries).

        Sim step T+5  (accel = 0.5 m/s², since target_v > v):
          v_proj = 1.0 + 0.5 × 5 = 3.5 m/s
          x_proj = 1000 + 1 × 5 + 0.5 × 0.5 × 25 = 1011.25
          Spatial: 1011.25 < 1300  ✓
          Temporal: (1300 − 1011.25) / 3.5 = 82.5 s > 5  ✓
          Segment: 3.5 ≤ 25.0  ✓

        All three sim steps pass (speed stays well below limit, position
        stays far from boundary) → action passes Layer 4a unchanged.
        """
        proposed_act = 3
        x = 1000.0
        v = 1.0
        dtz = 300.0
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, proposed_act)

    def test_green_overrides_when_close(self):
        """
        Green should be overridden when DTZ is tiny. 
        Explanation: If DTZ is tiny, the train is about to hit the boundary.
        If it proposes Green (accelerating to full speed), it will easily 
        violate the boundary in the simulation, so the VL must intervene.

        # Evaluating the aspect signaling method:
        # If the true state is that the next 3 zones are clear, the "real" DTZ 
        # (distance to the next OCCUPIED block) would be large (e.g., 3 * SH). 
        # In that scenario, PPO proposing Green is safe and the VL will allow it
        # (as shown in `test_green_passes_when_safe`). 
        # 
        # The purpose of THIS specific test (`test_green_overrides_when_close`) 
        # is to prove the VL catches AI hallucinations. If PPO proposes Green 
        # BUT the very next zone is actually occupied (so true DTZ is only 5m), 
        # the VL correctly overrides it to protect against a crash.
        # 
        # Note on validator.py: Our current `compute_dtz` dummy implementation 
        # simply returns the distance to the *very next* boundary. For a real 
        # 4-aspect system, `compute_dtz` will need to be swapped or upgraded 
        # later to find the distance to the next *occupied* boundary. But the 
        # core sieve logic in Layer 4 (which this tests) remains mathematically 
        # sound and robust.

        Expected value derivation
        -------------------------
        proposed_act = 3 (Green), x = 1000, v = 20.0, dtz = 5.0.
        Segment = S0 (limit = 25.0); target_v = 25.0.
        boundary_x = 1000 + 5 = 1005.
        (Note: Again, `dtz` is an explicitly injected parameter to test
        the sieve logic strictly, ignoring real map coordinates).

        Sim step T+5  (accel = 0.5 m/s²):
          x_proj = 1000 + 20 × 5 + 0.5 × 0.5 × 25 = 1106.25
          Spatial: 1106.25 ≥ 1005  → DTZ_Spatial_Violation ✗

        The proposed Green fails Layer 3a.  The sieve tries action 2
        (target_v = 16.67), then 1 (target_v = 8.33), then 0 (stop).
        Each is also projected — because v = 20 m/s and boundary is only
        5 m away, braking alone cannot prevent overshooting at T+5, so
        the sieve falls through to the lowest safe action (likely 0).

        Result: overridden = True, safe < proposed_act.
        """
        proposed_act = 3
        x = 1000.0
        v = 20.0
        dtz = 5.0
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)
        self.assertLess(safe, proposed_act)

    def test_red_always_safe(self):
        """
        Red (stop) should always be accepted, no matter the DTZ.

        Expected value derivation
        -------------------------
        proposed_act = 0 (Red), x = 1000, v = 0.0, dtz = 10.0.
        target_v = 0.0 (Red → 0 m/s).
        boundary_x = 1000 + 10 = 1010.

        At every sim step, |target_v − v| = 0 < 0.01 → cruise at 0 m/s.
          x_proj = 1000 (no movement), v_proj = 0.0.
          Spatial: 1000 < 1010  ✓
          Temporal: v_proj ≤ 0.01, check skipped  ✓
          Segment: 0.0 ≤ any limit  ✓

        A stopped train cannot violate any constraint → always passes 4a.
        """
        proposed_act = 0
        x = 1000.0
        v = 0.0
        dtz = 10.0
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertFalse(overridden)
        self.assertEqual(safe, 0)

    def test_spatial_violation(self):
        """
        Projected position past boundary → must override.

        Expected value derivation
        -------------------------
        proposed_act = 3 (Green), x = 5000, v = 24.0, dtz = 10.0.
        Segment = S0 (limit = 25.0); target_v = 25.0.
        boundary_x = 5000 + 10 = 5010.

        Sim step T+5  (accel = 0.5 m/s²):
          t_to_target = (25 − 24) / 0.5 = 2 s.
          Phase 1 (0–2 s): x_at_target = 5000 + 24 × 2 + 0.5 × 0.5 × 4
                                        = 5049.0
          5049.0 ≥ 5010  → DTZ_Spatial_Violation at the very first step.

        The train is going 24 m/s and has only 10 m of headroom — it
        overshoots the boundary in under half a second, triggering
        Layer 3a immediately.
        """
        proposed_act = 3
        x = 5000.0
        v = 24.0
        dtz = 10.0
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)

    def test_temporal_violation(self):
        """
        Not enough reaction time → must override.

        Expected value derivation
        -------------------------
    
        proposed_act = 2 (Dbl-Yellow), x = 1000, v = 15.0, dtz = 100.0.
        Segment = S0 (limit = 25.0); target_v = 2/3 × 25 = 16.67.
        boundary_x = 1000 + 100 = 1100. 
        
        (Note: We use synthetic injecting for `dtz` to test all boundary edge 
        cases precisely. If we used hardcoded zones, it's brittle if the map 
        changes. By isolating the sieve from Layer 1, the test is robust.
        Also, adding explicit spatial buffers to SH isn't needed because 
        TEMPORAL_BUFFER (5.0s) mathematically converts to a dynamic spatial 
        buffer based on present speed.)

        Sim step T+5  (accel = 0.5 m/s², since 16.67 > 15.0):
    
          t_to_target = (16.67 − 15) / 0.5 = 3.34 s.
          Since t (5) > t_to_target (3.34), two-phase:
            Phase 1: x_at_target = 1000 + 15 × 3.34 + 0.5 × 0.5 × 3.34²
                                 = 1000 + 50.1 + 2.79 ≈ 1052.89
            Phase 2: x_proj = 1052.89 + 16.67 × (5 − 3.34)
                            = 1052.89 + 27.67 ≈ 1080.56
            v_proj = 16.67.
          Spatial: 1080.56 < 1100  ✓
          Temporal: (1100 − 1080.56) / 16.67 ≈ 1.17 s < 5.0 s  → Violation ✗

        The train will be within 19.4 m of the boundary at 16.67 m/s,
        giving only ~1.17 s of reaction time — below the 5-second buffer.
        """
        proposed_act = 2
        x = 1000.0
        v = 15.0
        dtz = 100.0
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)

    def test_segment_violation(self):
        """
        Crossing into a slower segment at speed → must override.

        Expected value derivation
        -------------------------
        proposed_act = 3 (Green), x = 14900, v = 20.0, dtz = 50.0.
        Segment = S2 starts at 14951; x is just inside S1 (limit = 25.0).
        target_v = 25.0 (Green on S1).
        boundary_x = 14900 + 50 = 14950.

        Sim step T+5  (accel = 0.5 m/s²):
          t_to_target = (25 − 20) / 0.5 = 10 s.
          Since t (5) < t_to_target (10):
            v_proj = 20 + 0.5 × 5 = 22.5 m/s
            x_proj = 14900 + 20 × 5 + 0.5 × 0.5 × 25 = 15006.25
          Spatial: 15006.25 ≥ 14950  → DTZ_Spatial_Violation ✗

        Even without the segment check, the spatial violation fires first
        because the train crosses the block boundary.  But conceptually
        the train also crosses into S2 (limit = 16.67 m/s) at 22.5 m/s,
        which would also trigger a segment violation (22.5 > 16.67).
        Either way the action must be overridden.
        """
        proposed_act = 3
        x = 14900.0
        v = 20.0
        dtz = 50.0
        safe, overridden = self.vl.get_safe_action(proposed_act, x, v, dtz)
        self.assertTrue(overridden)


# -----------------------------------------------------------------------
# XAI Log Tests — _log_override()
# -----------------------------------------------------------------------

class TestXAILog(_BaseValidatorTest):
    """Tests for Layer 5: override logging to CSV."""

    def test_override_creates_log_entry(self):
        """
        When an override occurs, override_log_test.csv should gain a row.

        Expected value derivation
        -------------------------
        proposed_act = 3, x = 1000, v = 20.0, dtz = 5.0.
        These are the same values as test_green_overrides_when_close():
        the train is going 20 m/s with only 5 m before the boundary,
        so the sieve cannot find a safe higher action and overrides.

        When an override happens, _log_override() calls append_row(),
        which appends one CSV row to the log file.  After init_log()
        writes the header, the file should contain ≥ 2 lines (header +
        at least one data row).
        """
        proposed_act = 3
        x = 1000.0
        v = 20.0
        dtz = 5.0
        self.vl.get_safe_action(proposed_act, x, v, dtz)

        with open(_TEST_LOG) as f:
            lines = f.readlines()
        # Header + at least one data row
        self.assertGreaterEqual(len(lines), 2)

    def test_no_override_no_log_entry(self):
        """
        When no override occurs, only the header row should exist.

        Expected value derivation
        -------------------------
        proposed_act = 0 (Red / stop), x = 1000, v = 0.0, dtz = 300.0.
        A stopped train requesting "stop" will never violate any
        constraint (see test_red_always_safe), so the action passes
        Layer 4a with no override.  Since _log_override() is only called
        in the 4b sieve branch, no data row is appended.

        The log file should contain exactly 1 line (the CSV header).
        """
        proposed_act = 0
        x = 1000.0
        v = 0.0
        dtz = 300.0
        self.vl.get_safe_action(proposed_act, x, v, dtz)

        with open(_TEST_LOG) as f:
            lines = f.readlines()
        self.assertEqual(len(lines), 1)  # header only


if __name__ == "__main__":
    unittest.main()
