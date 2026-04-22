"""
Plot reward components across a full simulated journey.
Shows what the AI "sees" at every position along Line 104.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import matplotlib.pyplot as plt
from environment.railway_env import ModernizedLine104

def run_and_collect():
    """Run full episode with max-throttle policy, collect reward breakdown."""
    env = ModernizedLine104(lead_train_speed=20.0, training_mode=False)
    obs, info = env.reset()

    data = {
        "position": [], "speed": [], "time": [],
        "progress": [], "headway": [], "speed_rew": [],
        "heartbeat": [], "station": [], "override": [],
        "jerk": [], "energy": [], "total": [],
        "lead_x": [], "signal": [],
    }

    for _ in range(env.MAX_STEPS):
        action = np.array([0.5], dtype=np.float32)  # full throttle
        obs, reward, terminated, truncated, info = env.step(action)

        rb = info["reward_breakdown"]
        data["position"].append(env.x)
        data["speed"].append(env.v)
        data["time"].append(env.time)
        data["progress"].append(rb["progress"])
        data["headway"].append(rb["headway"])
        data["speed_rew"].append(rb["speed"])
        data["heartbeat"].append(rb["heartbeat"])
        data["station"].append(rb["station"])
        data["override"].append(rb["override"])
        data["jerk"].append(rb["jerk"])
        data["energy"].append(rb["energy"])
        data["total"].append(reward)
        data["lead_x"].append(env.lead_x)
        data["signal"].append(info["aspect"])

        if terminated or truncated:
            break

    return {k: np.array(v) for k, v in data.items()}


def plot(data):
    fig, axes = plt.subplots(5, 1, figsize=(14, 16), sharex=True)
    x = data["position"] / 1000  # km
    stations_km = np.array([582, 5481, 14951, 37160, 47017, 67394, 76651]) / 1000

    # 1. Speed + Signal
    ax = axes[0]
    ax.plot(x, data["speed"] * 3.6, color="dodgerblue", linewidth=0.8, label="AI Speed (km/h)")
    ax.set_ylabel("Speed (km/h)")
    ax2 = ax.twinx()
    ax2.plot(x, data["signal"], color="orange", alpha=0.4, linewidth=0.5, label="Signal Aspect")
    ax2.set_ylabel("Signal (0-3)")
    ax2.set_ylim(-0.5, 4)
    ax.set_title("Speed & Signal Aspect vs Position")
    for s in stations_km:
        ax.axvline(s, color="gray", linestyle="--", alpha=0.3)
    ax.legend(loc="upper left")
    ax2.legend(loc="upper right")

    # 2. Continuous rewards
    ax = axes[1]
    ax.plot(x, data["progress"], label="Progress", linewidth=0.8)
    ax.plot(x, data["speed_rew"], label="Speed", linewidth=0.8)
    ax.plot(x, data["heartbeat"], label="Heartbeat", linewidth=0.8)
    ax.set_ylabel("Reward")
    ax.set_title("Continuous Rewards (every step)")
    ax.legend()
    for s in stations_km:
        ax.axvline(s, color="gray", linestyle="--", alpha=0.3)

    # 3. Event rewards
    ax = axes[2]
    ax.plot(x, data["override"], label="Override", color="red", linewidth=0.8)
    ax.plot(x, data["jerk"], label="Jerk", color="purple", linewidth=0.8)
    ax.plot(x, data["energy"], label="Energy", color="brown", linewidth=0.8)
    ax.scatter(x[data["station"] > 0], data["station"][data["station"] > 0],
               label="Station (+100)", color="green", zorder=5, s=60)
    ax.set_ylabel("Reward")
    ax.set_title("Event Rewards (on triggers)")
    ax.legend()
    for s in stations_km:
        ax.axvline(s, color="gray", linestyle="--", alpha=0.3)

    # 4. Cumulative total
    ax = axes[3]
    cumulative = np.cumsum(data["total"])
    ax.plot(x, cumulative, color="black", linewidth=1)
    ax.set_ylabel("Cumulative Reward")
    ax.set_title("Cumulative Total Reward")
    ax.axhline(0, color="green", linestyle="--", alpha=0.5, label="Break-even")
    ax.axhline(-500, color="red", linestyle="--", alpha=0.5, label="Cowardice floor (-500)")
    ax.legend()
    for s in stations_km:
        ax.axvline(s, color="gray", linestyle="--", alpha=0.3)

    # 5. Per-step total
    ax = axes[4]
    ax.plot(x, data["total"], color="black", linewidth=0.5, alpha=0.7)
    ax.set_ylabel("Step Reward")
    ax.set_xlabel("Position (km)")
    ax.set_title("Per-Step Total Reward")
    for s in stations_km:
        ax.axvline(s, color="gray", linestyle="--", alpha=0.3)

    plt.tight_layout()
    out = Path(__file__).resolve().parent / "reward_landscape.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


if __name__ == "__main__":
    data = run_and_collect()
    print(f"Episode: {len(data['position'])} steps, "
          f"final pos: {data['position'][-1]/1000:.1f} km, "
          f"total reward: {data['total'].sum():.1f}")
    plot(data)
