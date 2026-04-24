"""
think_layer/config.py

Central configuration for the PPO training pipeline.
All hyperparameters, paths, and toggles live here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    """PPO training hyperparameters and runtime paths."""

    # ── PPO Hyperparameters ──────────────────────────────────────────
    total_timesteps: int = 5_000_000
    learning_rate: float = 0.0001
    n_steps: int = 4096
    batch_size: float = 512
    n_epochs: int = 5
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.02
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # ── Network Architecture ─────────────────────────────────────────
    policy_net: list[int] = field(default_factory=lambda: [256, 256])
    value_net: list[int] = field(default_factory=lambda: [256, 256])

    # ── Environment ──────────────────────────────────────────────────
    lead_train_speed: float = 25.0
    max_episode_steps: int = 10_000
    n_envs: int = 12
    slack_factor: float = 1.1  # 10% operational buffer for RL stability

    # ── Normalisation ────────────────────────────────────────────────
    normalize_obs: bool = True
    normalize_reward: bool = True
    norm_obs_clip: float = 10.0
    norm_reward_clip: float = 10.0

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Checkpointing & Evaluation ───────────────────────────────────
    checkpoint_freq: int = 250_000
    eval_freq: int = 50_000
    eval_episodes: int = 5

    # ── Paths ────────────────────────────────────────────────────────
    log_dir: str = str(_PACKAGE_DIR / "runs")
    model_dir: str = str(_PACKAGE_DIR / "models")
    results_dir: str = str(_PACKAGE_DIR / "results")

    def __post_init__(self):
        for d in (self.log_dir, self.model_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)


DEFAULT_CONFIG = TrainConfig()
