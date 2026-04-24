"""
Plot reward components across a full simulated journey.
Shows what the AI "sees" at every position along Line 104.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.railway_env import ModernizedLine104
from reward.reward_function import compute_reward, TrainState, RewardConfig

# Paths to trained model
MODEL_DIR = "/home/dharms/RailSchedulingAI/think_layer/models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
STATS_PATH = os.path.join(MODEL_DIR, "final_vecnormalize.pkl")

def run_and_collect():
    """Run full episode with max-throttle policy, collect reward breakdown."""
    env = ModernizedLine104(lead_train_speed=25.0, training_mode=False)
    # Wrap in VecNormalize (Crucial for HPO-tuned models)
    venv = DummyVecEnv([lambda: env])
    venv = VecNormalize.load(STATS_PATH, venv)
    venv.training = False
    venv.norm_reward = False

    # Load Model
    model = PPO.load(MODEL_PATH, env=venv)
    
    data = {
        "position": [], "speed": [], "reward": [], 
        "total_reward": [], "limit": [],
        "lead_x": [], "signal": [],
    }

    obs = venv.reset()
    total_rew = 0
    for idx in range(25000):
        action, _ = model.predict(obs, deterministic=True)
        
        # Take step
        obs, rewards, terminated, info = venv.step(action)
        
        # Unpack BEFORE reset logic can mess with position
        real_env = venv.envs[0].unwrapped
        
        # Get the limit for the CURRENT position (not the reset position)
        current_limit = real_env.vl.get_segment(real_env.x).limit_ms * 3.6

        data["position"].append(real_env.x)
        data["speed"].append(real_env.v)
        data["reward"].append(rewards[0])
        data["total_reward"].append(total_rew)
        data["limit"].append(current_limit)
        data["lead_x"].append(real_env.lead_train.x)
        data["signal"].append(real_env._get_signal_aspect())

        total_rew += rewards[0]

        if idx % 2000 == 0:
            print(f"TRAINED AGENT | Step {idx}: Pos={real_env.x/1000:.1f}km, V={real_env.v*3.6:.1f}km/h")

        if terminated[0]:
            print(f"Terminated at {real_env.x/1000:.1f}km")
            break

    return {k: np.array(v) for k, v in data.items()}

def plot(data):
    """Plot the reward landscape across the line."""
    stations_km = [0, 15, 28, 42, 55, 68, 76] 
    x = data["position"] / 1000.0
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 1. Speed & Limits
    ax = axes[0]
    ax.plot(x, data["speed"] * 3.6, label="Agent Speed", linewidth=2.0, color="blue")
    ax.plot(x, data["limit"], label="Speed Limit", color="red", linestyle="--", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, data["signal"], color="orange", alpha=0.3, label="Signal")
    ax.set_ylabel("Speed (km/h)")
    ax2.set_ylabel("Signal (0-3)")
    ax.set_title("SPEED AUDIT: Trained Agent vs. Limits")
    ax.legend(loc="upper left")

    # 2. Reward density
    ax = axes[1]
    ax.plot(x, data["reward"], color="green", linewidth=0.5)
    ax.set_ylabel("Step Reward")
    ax.set_title("Reward Spikes (Stations/Violations)")

    # 3. Total Reward
    ax = axes[2]
    ax.plot(x, data["total_reward"], color="black", linewidth=2.0)
    ax.set_ylabel("Cumulative Score")
    ax.set_xlabel("Position (km)")
    ax.set_title("The Golden Curve")
    
    for ax_item in axes:
        for s in stations_km: ax_item.axvline(s, color="gray", linestyle="--", alpha=0.2)
    
    plt.tight_layout()
    plt.savefig("reward_landscape.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    data = run_and_collect()
    plot(data)
