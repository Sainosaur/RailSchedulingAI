"""
think_layer/trainer.py

Training orchestrator — sets up callbacks and runs PPO learning.
"""

import os
import sys
import copy
from pathlib import Path

from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from think_layer.config import TrainConfig  # noqa: E402
from think_layer.agent import build_agent, make_env  # noqa: E402


def train(config: TrainConfig) -> None:
    """
    Run the full PPO training loop.

    Steps:
        1. Build agent + normalised environment
        2. Set up checkpoint and evaluation callbacks
        3. Train for config.total_timesteps
        4. Save final model + VecNormalize stats
    """
    print("=" * 60)
    print("  Line 104 — PPO Training Pipeline")
    print("=" * 60)
    print(f"  Timesteps       : {config.total_timesteps:,}")
    print(f"  Learning rate   : {config.learning_rate}")
    print(f"  Seed            : {config.seed}")
    print(f"  Lead train speed: {config.lead_train_speed} m/s")
    print(f"  Log dir         : {config.log_dir}")
    print(f"  Model dir       : {config.model_dir}")
    print("=" * 60)

    # 1. Build model
    model, vec_env = build_agent(config)

    # 2. Callbacks
    checkpoint_dir = os.path.join(config.model_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)

    checkpoint_cb = CheckpointCallback(
        save_freq=config.checkpoint_freq,
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        verbose=1,
    )

    # Separate eval environment (also normalised, but stats frozen)
    eval_venv = DummyVecEnv([
        make_env(
            seed=config.seed + 1000,
            lead_train_speed=config.lead_train_speed,
            max_episode_steps=config.max_episode_steps,
        )
    ])
    eval_venv = VecNormalize(
        eval_venv,
        norm_obs=config.normalize_obs,
        norm_reward=False,  # raw reward for evaluation
        clip_obs=config.norm_obs_clip,
    )
    # Sync normalisation stats from training env
    # don't drift as training continues to update vec_env.obs_rms in place.
    eval_venv.obs_rms = copy.deepcopy(vec_env.obs_rms)
    eval_venv.training = False   # freeze stats during evaluation
    eval_venv.norm_reward = False

    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=config.model_dir,
        log_path=config.log_dir,
        eval_freq=config.eval_freq,
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    callbacks = CallbackList([checkpoint_cb, eval_cb])

    # 3. Train
    print("\nStarting training...\n")
    model.learn(
        total_timesteps=config.total_timesteps,
        callback=callbacks,
        tb_log_name="PPO_Line104",
    )

    # 4. Save final model + normalisation stats
    final_model_path = os.path.join(config.model_dir, "final_model")
    final_vecnorm_path = os.path.join(config.model_dir, "final_vecnormalize.pkl")

    model.save(final_model_path)
    vec_env.save(final_vecnorm_path)

    # Also save the best model's VecNormalize stats
    best_vecnorm_path = os.path.join(config.model_dir, "best_vecnormalize.pkl")
    vec_env.save(best_vecnorm_path)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Final model  : {final_model_path}.zip")
    print(f"  Best model   : {os.path.join(config.model_dir, 'best_model.zip')}")
    print(f"  VecNormalize : {final_vecnorm_path}")
    print("=" * 60)
