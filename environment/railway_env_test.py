import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
import numpy as np
from environment.railway_env import ModernizedLine104

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
    
