"""
think_layer/config.py

Central configuration for the PPO training pipeline.
All hyperparameters, paths, and toggles live here.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

# Root of the think_layer package
_PACKAGE_DIR = Path(__file__).resolve().parent


@dataclass
class TrainConfig:
    """PPO training hyperparameters and runtime paths."""

    # ── PPO Hyperparameters ──────────────────────────────────────────
    total_timesteps: int   = 3_000_000
    learning_rate:   float = 3e-4   # SB3 default
    n_steps:         int   = 2048   # SB3 default
    batch_size:      int   = 512     # SB3 default
    n_epochs:        int   = 5     # SB3 default
    gamma:           float = 0.99    # Better for earlier learning horizon
    gae_lambda:      float = 0.95   # SB3 cdefault
    clip_range:      float = 0.2    # SB3 default
    ent_coef:        float = 0.005    # Standard exploration coefficient
    vf_coef:         float = 0.5    # SB3 default
    max_grad_norm:   float = 0.5    # SB3 default
    n_envs:          int   = 1      # Number of parallel environments

    # ── Network Architecture ─────────────────────────────────────────
    # Two hidden layers for both policy and value networks
    policy_net: list[int] = field(default_factory=lambda: [64, 64])
    value_net: list[int] = field(default_factory=lambda: [64, 64])

    # ── Environment ──────────────────────────────────────────────────
    lead_train_speed: float = 20.0  # m/s — midpoint; randomised per episode in training_mode
    max_episode_steps: int = 15_000  # truncation safety net

    # ── Normalisation ────────────────────────────────────────────────
    normalize_obs:    bool  = True
    normalize_reward: bool  = True   # Enabled: terminal penalties have been scaled down to allow this
    norm_obs_clip:    float = 10.0
    norm_reward_clip: float = 10.0

    # ── Reproducibility ──────────────────────────────────────────────
    seed: int = 42

    # ── Checkpointing & Evaluation ───────────────────────────────────
    checkpoint_freq: int = 250_000   # save a checkpoint every N steps
    eval_freq: int = 50_000          # run evaluation every N steps
    eval_episodes: int = 5           # episodes per evaluation round

    # ── Paths ────────────────────────────────────────────────────────
    log_dir: str = str(_PACKAGE_DIR / "runs")
    model_dir: str = str(_PACKAGE_DIR / "models")
    results_dir: str = str(_PACKAGE_DIR / "results")

    def __post_init__(self):
        """Create output directories if they don't exist."""
        for d in (self.log_dir, self.model_dir, self.results_dir):
            os.makedirs(d, exist_ok=True)


# Singleton default config
DEFAULT_CONFIG = TrainConfig()
