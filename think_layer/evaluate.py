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
from think_layer.config import DEFAULT_CONFIG, TrainConfig  # noqa: E402


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
    model = PPO.load(model_path, device="cpu")

    # Build evaluation environment
    def _make_eval_env():
        env = ModernizedLine104(
            lead_train_speed=config.lead_train_speed,
            lead_stop_offset=config.lead_stop_offset,
            training_mode=False,
        )
        return env

    venv = DummyVecEnv([_make_eval_env])

    # Load and apply VecNormalize if available
    if os.path.exists(vecnorm_path):
        print(f"Loading VecNormalize from: {vecnorm_path}")
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False  # freeze normalisation stats
        venv.norm_reward = False  # use raw rewards for evaluation
    else:
        print("WARNING: VecNormalize stats not found, running without normalisation")

    # Helper to access raw unwrapped env for true position/speed
    def _get_raw_env():
        env = venv.envs[0]
        while hasattr(env, "env"):
            env = env.env
        return env

    # Collect data
    all_rows = []
    episode_summaries = []

    # Station positions for segment timing
    STATIONS = [582.0, 5481.0, 14951.0, 37160.0, 47017.0, 67394.0, 76651.0]

    for ep in range(n_episodes):
        obs = venv.reset()
        done = False
        ep_reward = 0.0
        ep_steps = 0
        ep_overrides = 0
        ep_rows = []

        ep_stations = 1

        # Per-step telemetry for aggregate metrics
        speeds = []
        accelerations = []
        jerks = []
        emergency_brake_count = 0
        prev_accel = 0.0

        # Per-segment timing: track when train crosses each station boundary
        segment_arrival_steps = {}  # station_idx -> sim_time when reached

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)

            info = infos[0]
            raw_reward = reward[0]
            ep_reward += raw_reward
            ep_steps += 1

            raw = _get_raw_env()
            current_speed = raw.v
            current_accel = raw.last_a
            current_time = raw.time

            # Collect telemetry
            speeds.append(current_speed)
            accelerations.append(current_accel)

            # Jerk
            jerk = abs(current_accel - prev_accel)
            jerks.append(jerk)
            prev_accel = current_accel

            # Emergency braking
            if current_accel <= -1.0:
                emergency_brake_count += 1

            # Track segment arrivals
            for si in range(1, len(STATIONS)):
                if si not in segment_arrival_steps and raw.x >= STATIONS[si] - raw.vl.get_segment(raw.x).spatial_headway:
                    segment_arrival_steps[si] = current_time

            # Update max stations reached in this episode
            if "punctuality_status" in info:
                # next_station_idx is the one we are GOING to.
                # So the number of stations reached is the next_idx (since 0 is the start).
                next_idx = info["punctuality_status"]["ai"].get("next_station_idx")
                if next_idx is not None:
                    ep_stations = max(ep_stations, next_idx)
                elif info["punctuality_status"]["ai"].get("status") == "arrived":
                    ep_stations = 7

            if info.get("violations", {}).get("any_violation", False):
                ep_overrides += 1

            # Per-step row
            row = {
                "episode": ep,
                "step": ep_steps,
                "time": info.get("time", 0.0),
                "position": raw.x,
                "speed": current_speed,
                "acceleration": current_accel,
                "jerk": jerk,
                "segment": info.get("segment", ""),
                "train_aspect": info.get("train_aspect", -1),
                "station_aspect": info.get("station_aspect", -1),
                "proposed_a": info.get("proposed_a", 0.0),
                "any_violation": info.get("violations", {}).get("any_violation", False),
                "reward": raw_reward,
            }

            # Add reward breakdown if present
            breakdown = info.get("reward_breakdown", {})
            for key, val in breakdown.items():
                row[f"r_{key}"] = val

            ep_rows.append(row)
            done = dones[0]

        override_rate = (ep_overrides / ep_steps * 100) if ep_steps > 0 else 0.0
        stations_visited = ep_stations

        # Compute aggregate metrics for this episode
        speeds_arr = np.array(speeds)
        accel_arr = np.array(accelerations)
        jerk_arr = np.array(jerks)

        moving_speeds = speeds_arr[speeds_arr > 0.1]  # exclude stationary
        seg_limit = _get_raw_env().vl.get_segment(_get_raw_env().x).limit_ms

        # Station arrival times (cumulative from start)
        arrival_times = {}
        for si in sorted(segment_arrival_steps.keys()):
            arrival_times[si] = segment_arrival_steps[si]

        summary = {
            "episode": ep,
            "total_reward": ep_reward,
            "steps": ep_steps,
            "override_rate": override_rate,
            "overrides": ep_overrides,
            "stations_visited": stations_visited,
            # Speed metrics
            "median_speed": float(np.median(speeds_arr)),
            "mean_speed": float(np.mean(speeds_arr)),
            "median_moving_speed": float(np.median(moving_speeds)) if len(moving_speeds) > 0 else 0.0,
            "max_speed": float(np.max(speeds_arr)),
            # Acceleration metrics
            "mean_abs_accel": float(np.mean(np.abs(accel_arr))),
            "max_accel": float(np.max(accel_arr)),
            "min_accel": float(np.min(accel_arr)),
            # Jerk & smoothness
            "mean_abs_jerk": float(np.mean(jerk_arr)),
            "max_jerk": float(np.max(jerk_arr)),
            "emergency_brakes": emergency_brake_count,
            # Efficiency
            "cruise_efficiency": float(np.sum(speeds_arr >= seg_limit * 0.95) / max(1, len(speeds_arr)) * 100),
            "time_stopped_pct": float(np.sum(speeds_arr < 0.1) / max(1, len(speeds_arr)) * 100),
            # Timing
            "arrival_times": arrival_times,
        }
        episode_summaries.append(summary)
        all_rows.extend(ep_rows)

        print(
            f"  Episode {ep}: reward={ep_reward:+.2f}  "
            f"steps={ep_steps}  "
            f"violations={ep_overrides} ({override_rate:.1f}%)  "
            f"stations={stations_visited}/7"
        )

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

    # Aggregate summary across episodes
    avg_reward = np.mean([s["total_reward"] for s in episode_summaries])
    avg_override = np.mean([s["override_rate"] for s in episode_summaries])
    avg_stations = np.mean([s["stations_visited"] for s in episode_summaries])

    # Timetable for comparison
    raw = _get_raw_env()
    timetable = raw.timetable

    print("\n" + "=" * 60)
    print("  Evaluation Summary")
    print("=" * 60)
    print(f"  Episodes           : {n_episodes}")
    print(f"  Avg reward         : {avg_reward:+.2f}")
    print(f"  Avg override %     : {avg_override:.1f}%")
    print(f"  Avg stations       : {avg_stations:.1f}/7")
    print()
    print("  ── Speed ──")
    print(f"  Median speed       : {np.mean([s['median_speed'] for s in episode_summaries]):.2f} m/s")
    print(f"  Median moving speed: {np.mean([s['median_moving_speed'] for s in episode_summaries]):.2f} m/s")
    print(f"  Max speed          : {np.max([s['max_speed'] for s in episode_summaries]):.2f} m/s")
    print(f"  Cruise efficiency  : {np.mean([s['cruise_efficiency'] for s in episode_summaries]):.1f}%")
    print(f"  Time stopped       : {np.mean([s['time_stopped_pct'] for s in episode_summaries]):.1f}%")
    print()
    print("  ── Smoothness ──")
    print(f"  Mean |accel|       : {np.mean([s['mean_abs_accel'] for s in episode_summaries]):.4f} m/s²")
    print(f"  Mean |jerk|        : {np.mean([s['mean_abs_jerk'] for s in episode_summaries]):.4f} m/s³")
    print(f"  Max jerk           : {np.max([s['max_jerk'] for s in episode_summaries]):.4f} m/s³")
    print(f"  Emergency brakes   : {np.mean([s['emergency_brakes'] for s in episode_summaries]):.1f} avg/episode")
    print()
    print("  ── Station Arrivals (Cumulative) ──")
    station_names = ["", "Rabka-Zdrój", "Mszana Dolna", "Tymbark", "Limanowa", "Marcinkowice", "Nowy Sącz"]
    for si in range(1, 7):
        times = [s["arrival_times"].get(si, None) for s in episode_summaries]
        times = [t for t in times if t is not None]
        if times:
            avg_t = np.mean(times)
            sched_entry = timetable.get_entry(si)
            sched_t = sched_entry.scheduled_arrival if sched_entry else 0.0
            print(f"  Arr at {station_names[si]:15s}: {avg_t:7.0f}s actual  |  {sched_t:7.0f}s scheduled")
        else:
            print(f"  Arr at {station_names[si]:15s}: not reached")

    if csv_path:
        print(f"\n  Results CSV     : {csv_path}")
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
        "--model",
        type=str,
        default=None,
        help="Path to model .zip file (default: best_model in model_dir)",
    )
    parser.add_argument(
        "--vecnorm", type=str, default=None, help="Path to VecNormalize .pkl file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=5,
        help="Number of evaluation episodes (default: 5)",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip saving CSV output")

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