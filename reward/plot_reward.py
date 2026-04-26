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

# Paths to trained model
MODEL_DIR = "/home/dharms/RailSchedulingAI/think_layer/models"
MODEL_PATH = os.path.join(MODEL_DIR, "best_model.zip")
STATS_PATH = os.path.join(MODEL_DIR, "final_vecnormalize.pkl")

def run_and_collect():
    """Run full episode and collect reward breakdown."""
    env = ModernizedLine104(lead_train_speed=25.0, training_mode=False)
    
    use_model = os.path.exists(MODEL_PATH) and os.path.exists(STATS_PATH)
    
    if use_model:
        print(f"--- Plotting AI Mode (Using model {MODEL_PATH}) ---")
        venv = DummyVecEnv([lambda: env])
        venv = VecNormalize.load(STATS_PATH, venv)
        venv.training = False
        venv.norm_reward = False
        model = PPO.load(MODEL_PATH, env=venv)
        obs = venv.reset()
    else:
        print("--- Plotting Heuristic Mode (Simulating 'Perfect' Driver) ---")
        env.reset()
        obs = None

    data = {
        "position": [], "speed": [], "reward": [], 
        "total_reward": [], "limit": [],
        "signal": [],
    }
    breakdown_data = {}

    total_rew = 0
    
    for idx in range(30000):
        # 1. State BEFORE step
        current_x = env.x
        current_v = env.v
        current_signal = env._get_signal_aspect()
        
        # 2. Predict or Heuristic
        if use_model:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, is_done_vec, info_vec = venv.step(action)
            step_reward = rewards[0]
            is_done = is_done_vec[0]
            info = info_vec[0]
        else:
            if hasattr(env, "ai_departure_time") and env.time < env.ai_departure_time:
                target_v = 0.0
            else:
                seg = env.vl.get_segment(env.x)
                v_lim = seg.limit_ms
                if current_signal == 0:
                    dist_to_st = 99999.0
                    if env.last_station_idx + 1 < len(env.STATIONS):
                        dist_to_st = env.STATIONS[env.last_station_idx + 1] - env.x
                    if dist_to_st < seg.spatial_headway + 10.0:
                        target_v = 3.0 
                    else:
                        target_v = 0.0
                elif current_signal == 1: target_v = v_lim * 0.4
                elif current_signal == 2: target_v = v_lim * 0.7
                else: target_v = v_lim
            
            if current_v < target_v - 0.2: action_val = 0.5
            elif current_v > target_v + 0.2: action_val = -1.0
            else: action_val = 0.0
            
            action = np.array([action_val], dtype=np.float32)
            _obs, step_reward, terminated, truncated, info = env.step(action)
            is_done = terminated or truncated

        total_rew += step_reward

        # 3. Append data
        data["position"].append(env.x)
        data["speed"].append(env.v)
        data["reward"].append(step_reward)
        data["total_reward"].append(total_rew)
        data["limit"].append(env.vl.get_segment(env.x).limit_ms * 3.6)
        data["signal"].append(env._get_signal_aspect())

        # Collect breakdown
        if "reward_breakdown" in info:
            for k, v in info["reward_breakdown"].items():
                if k not in breakdown_data: breakdown_data[k] = []
                breakdown_data[k].append(v)

        if is_done:
            print(f"Journey ended at {env.x/1000:.2f} km")
            print(f"Final Total Reward: {total_rew:+.2f}")
            break

    return {k: np.array(v) for k, v in data.items()}, {k: np.array(v) for k, v in breakdown_data.items()}

def plot(data, breakdown):
    """Plot the reward landscape across the line."""
    stations_km = [0.58, 5.48, 14.95, 37.16, 47.01, 67.39, 76.65] 
    x = data["position"] / 1000.0
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
    
    # 1. Speed & Limits
    ax = axes[0]
    ax.plot(x, data["speed"] * 3.6, label="Agent Speed", linewidth=2.0, color="#1f77b4")
    ax.plot(x, data["limit"], label="Speed Limit", color="#d62728", linestyle="--", alpha=0.8)
    ax2 = ax.twinx()
    ax2.plot(x, data["signal"], color="#ff7f0e", alpha=0.3, label="Signal")
    ax.set_ylabel("Speed (km/h)")
    ax2.set_ylabel("Signal")
    ax.set_title("JOURNEY PROFILE: Speed vs. Limit", fontweight="bold")
    ax.legend(loc="upper left")

    # 2. Component Breakdown (Stacked-ish or just key ones)
    ax = axes[1]
    # Filter for interesting ones
    keys = ["r_step", "r_progress", "r_speed", "r_signal_compliance",
            "r_station", "r_time", "r_jerk", "r_patience"]
    for k in keys:
        if k in breakdown:
            ax.plot(x, breakdown[k], label=k, alpha=0.7)
    ax.set_ylabel("Component Reward")
    ax.set_title("REWARD COMPONENTS (Detailed Audit)", fontweight="bold")
    ax.legend(loc="upper left", ncol=3, fontsize='small')
    ax.set_ylim(-12, 12) # Focus on steady state rewards

    # 3. MileStone Rewards (Log scale or high limit to see station jumps)
    ax = axes[2]
    if "r_station" in breakdown:
        ax.plot(x, breakdown["r_station"], label="Station Milestone",
                color="gold", linewidth=2)
    if "r_time" in breakdown:
        ax.plot(x, breakdown["r_time"], label="Punctuality",
                color="green", linestyle=":")
    if "r_patience" in breakdown:
        ax.plot(x, breakdown["r_patience"], label="Patience",
                color="purple", linestyle="--")
    ax.set_ylabel("Milestone Reward")
    ax.set_title("MILESTONES & PUNCTUALITY", fontweight="bold")
    ax.legend(loc="upper left")

    # 4. Cumulative Reward
    ax = axes[3]
    ax.plot(x, data["total_reward"], color="black", linewidth=2.5)
    ax.set_ylabel("Cumulative Score")
    ax.set_xlabel("Position (km)")
    ax.set_title("THE GOLDEN CURVE: Cumulative Reward", fontweight="bold")
    
    for ax_item in axes:
        for s in stations_km: ax_item.axvline(s, color="gray", linestyle="--", alpha=0.2)
        ax_item.grid(True, which='both', linestyle=':', alpha=0.5)
    
    plt.tight_layout()
    # Save in the same directory as the script
    script_dir = Path(__file__).resolve().parent
    output_path = script_dir / "reward_landscape.png"
    plt.savefig(str(output_path), dpi=150)
    print(f"Detailed plot saved to {output_path}")
    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    data, breakdown = run_and_collect()
    plot(data, breakdown)
