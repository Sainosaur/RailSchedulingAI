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
    total_timesteps: int = 5_000_000
    learning_rate: float = 0.00041122409178063574
    n_steps: int = 4096         # rollout buffer size per env per update
                                 # total rollout = n_envs × n_steps = 65,536 steps
                                 # KEEP HIGH: halving to 2048 doubles update frequency
                                 # and cuts FPS ~40% because gradient steps steal time
                                 # from env stepping. 4096 was the pre-regression value.
    batch_size: int = 256        # SGD minibatch size (larger = better GPU utilisation on 4GB VRAM)
    n_epochs: int = 4            # PPO clipping epochs per update (4 balances speed vs learning quality)
    gamma: float = 0.99          # discount factor (prioritizes ~1000s into future for train braking dynamics)
    gae_lambda: float = 0.95     # GAE advantage estimator
    clip_range: float = 0.2      # PPO surrogate clip
    ent_coef: float = 0.05       # entropy bonus for exploration (INCREASED to break cowardice)
    vf_coef: float = 0.5         # value function loss weight
    max_grad_norm: float = 0.5   # gradient clipping

    # ── Network Architecture ─────────────────────────────────────────
    # Two hidden layers for both policy and value networks
    policy_net: list[int] = field(default_factory=lambda: [128, 128])
    value_net: list[int] = field(default_factory=lambda: [128, 128])

    # ── Environment ──────────────────────────────────────────────────
    lead_train_speed: float = 20.0  # m/s — midpoint; randomised per episode in training_mode
    max_episode_steps: int = 15_000  # truncation safety net
    n_envs: int = 16                 # parallel SubprocVecEnv workers (1 per core, ~500MB each)

    # ── Normalisation ────────────────────────────────────────────────
    normalize_obs: bool = True
    normalize_reward: bool = True
    norm_obs_clip: float = 10.0
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