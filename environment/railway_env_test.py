import pytest
import numpy as np
from environment.railway_env import ModernizedLine104

def _make_env() -> ModernizedLine104:
    """Return a freshly reset environment."""
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    env.reset()
    return env

def test_smoke_reset():
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    obs, info = env.reset()
    assert len(obs) == 7, "Observation space mismatch"
    assert "overridden" in info, "Info dictionary missing overridden flag"
    assert env.time == 0.0, "Time should reset to 0.0"
    assert env.x == env.TRACK_START, "AI Train should start at TRACK_START"

def test_smoke_step():
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    env.reset()
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert not terminated
    assert env.time == env.DT
    assert env.x > env.TRACK_START, "Train should have moved forward"
    
def test_landslide_api():
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    env.reset()
    assert len(env.landslides) == 0
    idx = env.set_landslide(4000.0)
    assert len(env.landslides) == 1
    assert env.landslides[idx].position == 4000.0
    assert env.landslides[idx].active == True
    
    # Toggle off
    new_state = env.toggle_landslide(idx)
    assert new_state == False
    assert env.landslides[idx].active == False
    
    # Toggle on
    new_state = env.toggle_landslide(idx)
    assert new_state == True
    
    # Remove
    env.clear_landslide(idx)
    assert len(env.landslides) == 0

def test_station_visited():
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    env.reset()
    # Force position past first tracking station
    if len(env.STATIONS) > 1:
        env.x = env.STATIONS[1] + 1.0
    action = np.array([0.5], dtype=np.float32)
    obs, reward, terminated, truncated, info = env.step(action)
    assert 1 in env.visited_stations, "Should register visiting station 1"
    
# ---------------------------------------------------------------------------
# Hazard / signal aspect tests
# ---------------------------------------------------------------------------

def test_hazard_degrades_signal_aspect():
    """A landslide 625 m ahead (2xSH) should degrade aspect from Green to Yellow."""
    env = _make_env()
 
    # Baseline: lead is 2000 m ahead -> Green (3)
    assert env._get_signal_aspect() == 3, "Expected Green before hazard"
 
    # Place landslide at 1500 m; enclosing block starts at 1207 m (625 m from AI)
    # 625 m = 2xSH -> exactly on the 2-block boundary -> aspect drops to Yellow (1)
    idx = env.set_landslide(1500.0)
    assert env._get_signal_aspect() == 1, "Hazard should degrade aspect to Yellow"
 
    # Removing the hazard restores the original aspect
    env.clear_landslide(idx)
    assert env._get_signal_aspect() == 3, "Aspect should restore after clearing hazard"
 
 
def test_hazard_in_active_hazards():
    """set_landslide should add the enclosing block to active_hazards."""
    env = _make_env()
    idx = env.set_landslide(1500.0)
    block_key = (env.landslides[idx]._block_start, env.landslides[idx]._block_end)
    assert block_key in env.active_hazards
 
    env.clear_landslide(idx)
    assert block_key not in env.active_hazards
 
 
def test_hazards_persist_across_reset():
    """Hazards represent external physical events and must survive episode resets."""
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    env.reset()
    idx = env.set_landslide(4000.0)
 
    env.reset()
    assert idx in env.landslides, "Landslide should survive reset"
    assert env.landslides[idx].active == True
 
 
# ---------------------------------------------------------------------------
# Overlapping landslides
# ---------------------------------------------------------------------------
 
def test_overlapping_landslides_do_not_cancel():
    """Deactivating one landslide must not remove the block if another is still active."""
    env = _make_env()
 
    # Both 1300 m and 1350 m fall in block 1207-1519.5 (confirmed by probe)
    idx_a = env.set_landslide(1300.0)
    idx_b = env.set_landslide(1350.0)
    block_key = (env.landslides[idx_a]._block_start, env.landslides[idx_a]._block_end)
 
    assert env.landslides[idx_a]._block_start == env.landslides[idx_b]._block_start, \
        "Both landslides must be in the same block for this test to be meaningful"
    assert block_key in env.active_hazards
 
    # Toggle A off - B is still active, block must stay
    env.toggle_landslide(idx_a)
    assert block_key in env.active_hazards, \
        "Block should remain active while landslide B is still on"
 
    # Toggle B off - no active landslides remain in this block
    env.toggle_landslide(idx_b)
    assert block_key not in env.active_hazards, \
        "Block should deactivate when all landslides within it are off"
 
 
# ---------------------------------------------------------------------------
# Lead train controls
# ---------------------------------------------------------------------------
 
def test_lead_stall_stops_train():
    """stall_lead() should emergency-brake the lead to a full stop."""
    env = _make_env()
    assert env.lead_v > 0.0, "Lead should be moving at reset"
 
    env.stall_lead()
    assert env.lead_stalled == True
 
    # At DECEL=-1.0 m/s^2 from 25 m/s the lead stops in 25 steps; use 30 for margin
    action = np.array([0.0], dtype=np.float32)
    for _ in range(30):
        env.step(action)
 
    assert env.lead_v == 0.0, "Lead should be fully stopped after stall"
 
 
def test_lead_release_after_stall():
    """release_lead() should clear the stall flag."""
    env = _make_env()
    env.stall_lead()
    env.release_lead()
    assert env.lead_stalled == False
    assert env.lead_held == False
 
 
def test_lead_hold_freezes_dwell_timer():
    """hold_lead() must prevent the dwell timer from counting down."""
    env = _make_env()
 
    env.lead_dwell_timer = 30
    env.lead_v = 0.0
    env.hold_lead()
 
    action = np.array([0.0], dtype=np.float32)
    for _ in range(5):
        env.step(action)
 
    assert env.lead_dwell_timer == 30, "Dwell timer must not decrease while held"
 
 
def test_lead_hold_release_resumes_timer():
    """After release_lead(), the dwell timer should count down normally."""
    env = _make_env()
    env.lead_dwell_timer = 30
    env.lead_v = 0.0
    env.hold_lead()
    env.release_lead()
 
    action = np.array([0.0], dtype=np.float32)
    env.step(action)
 
    assert env.lead_dwell_timer == 29, "Dwell timer should resume after release"
 
 
# ---------------------------------------------------------------------------
# Termination conditions
# ---------------------------------------------------------------------------
 
def test_truncation_at_max_steps():
    """Episode should truncate (not terminate) when step_count reaches MAX_STEPS."""
    env = _make_env()
    env.step_count = env.MAX_STEPS - 1
 
    _, _, terminated, truncated, _ = env.step(np.array([0.0], dtype=np.float32))
 
    assert truncated, "Episode must truncate at MAX_STEPS"
    assert not terminated, "Truncation should not set terminated flag"
 
 
def test_collision_triggers_termination():
    """AI train reaching lead train position must terminate the episode."""
    env = _make_env()
 
    # Place lead behind the AI - collision check fires immediately
    env.x = env.TRACK_START + 1000.0
    env.lead_x = env.TRACK_START + 500.0
    env.v = 0.0
 
    _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
 
    assert terminated, "Collision should terminate episode"
    assert reward <= -150.0, "Collision must apply a large negative reward"
 
 
def test_headway_violation_terminates():
    """Temporal headway below 1xTH must terminate the episode."""
    env = _make_env()
 
    # At 25 m/s with lead 100 m ahead: headway = 100/25 = 4 s < TH=12.5 s
    env.v = 25.0
    env.lead_x = env.x + 100.0
 
    _, reward, terminated, _, _ = env.step(np.array([0.0], dtype=np.float32))
 
    assert terminated, "Headway violation should terminate episode"
    assert reward <= -100.0, "Headway violation must apply a large negative reward"
 
 
# ---------------------------------------------------------------------------
# Punctuality
# ---------------------------------------------------------------------------
 
def test_punctuality_status_structure():
    """get_punctuality_status() must return the full schema expected by the server."""
    env = _make_env()
    status = env.get_punctuality_status()
 
    assert "sim_time" in status
    assert "ai" in status
 
    ai = status["ai"]
    for key in ("next_station_idx", "next_station_name", "scheduled_arrival",
                "eta", "slack_seconds", "status", "arrival_log"):
        assert key in ai, f"Missing key in punctuality AI dict: '{key}'"
 
    assert ai["status"] in ("on_time", "early", "late"), \
        f"Unexpected status value: {ai['status']}"
 
 
def test_punctuality_logs_arrival_time():
    """Actual arrival time should be recorded when the AI reaches a station."""
    env = _make_env()
    env.x = env.STATIONS[1] + 1.0
 
    env.step(np.array([0.0], dtype=np.float32))
 
    assert 1 in env.ai_arrival_times, "Arrival time should be logged for station 1"
    assert env.ai_arrival_times[1] > 0.0