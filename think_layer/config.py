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
    total_timesteps: int   = 3_000_000
    learning_rate:   float = 1e-5   # SB3 default
    n_steps:         int   = 2048   # SB3 default
    batch_size:      int   = 512     # SB3 default
    n_epochs:        int   = 5     # SB3 default
    gamma:           float = 0.999   # SB3 default
    gae_lambda:      float = 0.95   # SB3 cdefault
    clip_range:      float = 0.2    # SB3 default
    ent_coef:        float = 0.005    # SB3 default
    vf_coef:         float = 0.5    # SB3 default
    max_grad_norm:   float = 0.5    # SB3 default

    # ── Network Architecture ─────────────────────────────────────────
    policy_net: list[int] = field(default_factory=lambda: [64, 64])  # SB3 default
    value_net:  list[int] = field(default_factory=lambda: [64, 64])  # SB3 default

    # ── Environment ──────────────────────────────────────────────────
    lead_train_speed:  float = 20.0
    max_episode_steps: int   = 15_000
    n_envs:            int   = 12    # 12 of 16 cores; leaves 4 for main process + OS

    # ── Normalisation ────────────────────────────────────────────────
    normalize_obs:    bool  = True
    normalize_reward: bool  = True
    norm_obs_clip:    float = 10.0
    norm_reward_clip: float = 10.0

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Checkpointing & Evaluation ───────────────────────────────────
    checkpoint_freq: int = 250_000
    eval_freq:       int = 50_000
    eval_episodes:   int = 5

    # ── Paths ────────────────────────────────────────────────────────
    log_dir:     str = str(_PACKAGE_DIR / "runs")
    model_dir:   str = str(_PACKAGE_DIR / "models")
    results_dir: str = str(_PACKAGE_DIR / "results")

    def __post_init__(self):
        for d in (self.log_dir, self.model_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)


DEFAULT_CONFIG = TrainConfig()