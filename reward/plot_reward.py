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
        "total_reward": [], "r_breaks": [],
        "lead_x": [], "signal": [],
    }

    obs = venv.reset()
    total_rew = 0
    for idx in range(25000):  # Long budget for full run
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, terminated, info = venv.step(action)
        
        # Unpack from vector env
        real_env = venv.envs[0].unwrapped
        total_rew += rewards[0]

        if idx % 2000 == 0:
            print(f"TRAINED AGENT | Step {idx}: Pos={real_env.x/1000:.1f}km, V={real_env.v*3.6:.1f}km/h, TotalReward={total_rew:.1f}")

        data["position"].append(real_env.x)
        data["speed"].append(real_env.v)
        data["reward"].append(rewards[0])
        data["total_reward"].append(total_rew)
        data["lead_x"].append(real_env.lead_train.x)
        data["signal"].append(real_env._get_signal_aspect())

        if terminated[0]:
            break

    print(f"Final Position: {real_env.x/1000:.1f}km | Final Reward: {total_rew:.1f}")
    return {k: np.array(v) for k, v in data.items()}


def plot(data):
    """Plot the reward landscape across the line."""
    stations_km = [0, 15, 28, 42, 55, 68, 76]  # approximate
    x = data["position"] / 1000.0  # meters to km
    
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), sharex=True)
    
    # 1. Speed & Signal
    ax = axes[0]
    ax.plot(x, data["speed"] * 3.6, label="Agent Speed", linewidth=1.2, color="blue")
    ax2 = ax.twinx()
    ax2.plot(x, data["signal"], color="orange", alpha=0.3, label="Signal Aspect")
    ax.set_ylabel("Speed (km/h)")
    ax2.set_ylabel("Signal (0-3)")
    ax.set_title("FINAL PERFORMANCE: Trained RL Agent (5M Steps)")
    for s in stations_km: ax.axvline(s, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")

    # 2. Per-Step Reward
    ax = axes[1]
    ax.plot(x, data["reward"], color="green", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Step Reward")
    ax.set_title("Per-Step Reward Density")
    for s in stations_km: ax.axvline(s, color="gray", linestyle="--", alpha=0.3)

    # 3. Cumulative Total
    ax = axes[2]
    ax.plot(x, data["total_reward"], color="black", linewidth=1.5)
    ax.set_ylabel("Total Reward")
    ax.set_xlabel("Position (km)")
    ax.set_title("Cumulative Journey Reward (The Golden Curve)")
    ax.axhline(0, color="red", linestyle="--", alpha=0.5)
    for s in stations_km: ax.axvline(s, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "reward_landscape.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()

if __name__ == "__main__":
    data = run_and_collect()
    plot(data)
