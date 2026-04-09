"""
think_layer/evaluate.py

CLI entry point:  python -m think_layer.evaluate

Loads the best (or specified) trained model and runs evaluation episodes.
Outputs:
    - Console summary (total reward, override rate, stations visited, duration)
    - CSV file with per-step data for report plots
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.railway_env import ModernizedLine104  # noqa: E402
from think_layer.config import TrainConfig, DEFAULT_CONFIG  # noqa: E402


def evaluate(
    model_path: str | None = None,
    vecnorm_path: str | None = None,
    n_episodes: int = 5,
    config: TrainConfig = DEFAULT_CONFIG,
    save_csv: bool = True,
) -> dict:
    """
    Run evaluation episodes and collect per-step data.

    Parameters
    ----------
    model_path   : Path to saved model .zip (default: best_model in model_dir)
    vecnorm_path : Path to VecNormalize .pkl (default: best_vecnormalize in model_dir)
    n_episodes   : Number of episodes to evaluate
    config       : TrainConfig for environment settings
    save_csv     : Whether to save per-step data to CSV

    Returns
    -------
    dict with summary statistics
    """
    # Resolve paths
    if model_path is None:
        model_path = os.path.join(config.model_dir, "best_model.zip")
    if vecnorm_path is None:
        vecnorm_path = os.path.join(config.model_dir, "best_vecnormalize.pkl")

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}")
        print("       Have you trained a model yet?  Run: python -m think_layer.train")
        sys.exit(1)

    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)

    # Build evaluation environment
    def _make_eval_env():
        env = ModernizedLine104(lead_train_speed=config.lead_train_speed)
        return env

    venv = DummyVecEnv([_make_eval_env])

    # Load and apply VecNormalize if available
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from: {vecnorm_path}")
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False    # freeze normalisation stats
        venv.norm_reward = False  # use raw rewards for evaluation
    else:
        print("WARNING: VecNormalize stats not found, running without normalisation")

    # Collect data
    all_rows = []
    episode_summaries = []

    for ep in range(n_episodes):
        obs = venv.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_overrides = 0
        ep_rows = []

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)

            info = infos[0]
            raw_reward = reward[0]
            ep_reward += raw_reward
            ep_steps += 1

            if info.get("overridden", False):
                ep_overrides += 1

            # Per-step row
            row = {
                "episode": ep,
                "step": ep_steps,
                "time": info.get("time", 0.0),
                "position": obs[0][0] if hasattr(obs[0], "__len__") else 0.0,
                "speed": obs[0][1] if hasattr(obs[0], "__len__") else 0.0,
                "segment": info.get("segment", ""),
                "aspect": info.get("aspect", -1),
                "proposed_a": info.get("proposed_a", 0.0),
                "safe_a": info.get("safe_a", 0.0),
                "overridden": info.get("overridden", False),
                "reward": raw_reward,
            }

            # Add reward breakdown if present
            breakdown = info.get("reward_breakdown", {})
            for key, val in breakdown.items():
                row[f"r_{key}"] = val

            ep_rows.append(row)
            done = dones[0]

        override_rate = (ep_overrides / ep_steps * 100) if ep_steps > 0 else 0.0

        # Access the raw env to get visited stations
        raw_env = venv.envs[0]
        # Unwrap through Monitor/TimeLimit to get the base env
        base_env = raw_env
        while hasattr(base_env, "env"):
            base_env = base_env.env

        stations_visited = len(getattr(base_env, "visited_stations", set()))

        summary = {
            "episode": ep,
            "total_reward": ep_reward,
            "steps": ep_steps,
            "override_rate": override_rate,
            "overrides": ep_overrides,
            "stations_visited": stations_visited,
        }
        episode_summaries.append(summary)
        all_rows.extend(ep_rows)

        print(f"  Episode {ep}: reward={ep_reward:+.2f}  "
              f"steps={ep_steps}  "
              f"overrides={ep_overrides} ({override_rate:.1f}%)  "
              f"stations={stations_visited}/7")

    # Save CSV
    csv_path = None
    if save_csv and all_rows:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        csv_path = os.path.join(config.results_dir, f"eval_{timestamp}.csv")
        os.makedirs(config.results_dir, exist_ok=True)

        fieldnames = list(all_rows[0].keys())
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)
        print(f"\n  CSV saved: {csv_path}")

    # Aggregate summary
    avg_reward = np.mean([s["total_reward"] for s in episode_summaries])
    avg_override = np.mean([s["override_rate"] for s in episode_summaries])
    avg_stations = np.mean([s["stations_visited"] for s in episode_summaries])

    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  Episodes        : {n_episodes}")
    print(f"  Avg reward      : {avg_reward:+.2f}")
    print(f"  Avg override %  : {avg_override:.1f}%")
    print(f"  Avg stations    : {avg_stations:.1f}/7")
    if csv_path:
        print(f"  Results CSV     : {csv_path}")
    print("=" * 60)

    return {
        "episodes": episode_summaries,
        "avg_reward": avg_reward,
        "avg_override_rate": avg_override,
        "avg_stations": avg_stations,
        "csv_path": csv_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained PPO agent on Line 104"
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Path to model .zip file (default: best_model in model_dir)"
    )
    parser.add_argument(
        "--vecnorm", type=str, default=None,
        help="Path to VecNormalize .pkl file"
    )
    parser.add_argument(
        "--episodes", type=int, default=5,
        help="Number of evaluation episodes (default: 5)"
    )
    parser.add_argument(
        "--no-csv", action="store_true",
        help="Skip saving CSV output"
    )

    args = parser.parse_args()
    config = TrainConfig()

    evaluate(
        model_path=args.model,
        vecnorm_path=args.vecnorm,
        n_episodes=args.episodes,
        config=config,
        save_csv=not args.no_csv,
    )


if __name__ == "__main__":
    main()
