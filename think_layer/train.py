"""
think_layer/train.py

CLI entry point:  python -m think_layer.train

Optional overrides:
    --timesteps 1000000
    --lr 1e-4
    --seed 123
    --lead-speed 25.0
"""

import argparse
import sys
from pathlib import Path

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from think_layer.config import TrainConfig  # noqa: E402
from think_layer.trainer import train  # noqa: E402


def main():
    parser = argparse.ArgumentParser(
        description="Train PPO agent on Line 104 railway environment"
    )
    parser.add_argument(
        "--timesteps", type=int, default=None,
        help="Total training timesteps (default: 500,000)"
    )
    parser.add_argument(
        "--lr", type=float, default=None,
        help="Learning rate (default: 3e-4)"
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--lead-speed", type=float, default=None,
        help="Lead train speed in m/s (default: 20.0)"
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=None,
        help="Checkpoint save frequency in steps (default: 10,000)"
    )
    parser.add_argument(
        "--eval-freq", type=int, default=None,
        help="Evaluation frequency in steps (default: 10,000)"
    )

    args = parser.parse_args()

    # Build config with CLI overrides
    config = TrainConfig()

    if args.timesteps is not None:
        config.total_timesteps = args.timesteps
    if args.lr is not None:
        config.learning_rate = args.lr
    if args.seed is not None:
        config.seed = args.seed
    if args.lead_speed is not None:
        config.lead_train_speed = args.lead_speed
    if args.checkpoint_freq is not None:
        config.checkpoint_freq = args.checkpoint_freq
    if args.eval_freq is not None:
        config.eval_freq = args.eval_freq

    train(config)


if __name__ == "__main__":
    main()
