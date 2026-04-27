"""
think_layer/trainer.py

Training orchestrator — sets up callbacks and runs PPO learning.
"""

import os
import sys
from pathlib import Path

from stable_baselines3.common.callbacks import (
    CallbackList,
    CheckpointCallback,
    EvalCallback,
    BaseCallback,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from think_layer.agent import build_agent, make_env  # noqa: E402
from think_layer.config import TrainConfig  # noqa: E402


class SyncNormCallback(BaseCallback):
    """Re-syncs eval VecNormalize obs_rms from training env every N steps."""
    def __init__(self, train_env: VecNormalize, eval_env: VecNormalize, sync_freq: int = 10_000):
        super().__init__()
        self.train_env = train_env
        self.eval_env = eval_env
        self.sync_freq = sync_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.sync_freq == 0:
            self.eval_env.obs_rms = self.train_env.obs_rms
        return True


class SaveBestVecNormalizeCallback(BaseCallback):
    """Saves the VecNormalize statistics whenever a new best model is found by EvalCallback."""
    def __init__(self, vec_env: VecNormalize, eval_cb: EvalCallback, model_dir: str, verbose: int = 1):
        super().__init__(verbose)
        self.vec_env = vec_env
        self.eval_cb = eval_cb
        self.save_path = os.path.join(model_dir, "best_vecnormalize.pkl")
        self.last_best_reward = -float("inf")

    def _on_step(self) -> bool:
        # Check if EvalCallback has updated its best_mean_reward
        if self.eval_cb.best_mean_reward > self.last_best_reward:
            self.last_best_reward = self.eval_cb.best_mean_reward
            if self.verbose > 0:
                print(f"DEBUG: New best model found (reward: {self.last_best_reward:.2f}). Saving VecNormalize stats to {self.save_path}")
            self.vec_env.save(self.save_path)
        return True


class RewardLoggerCallback(BaseCallback):
    """
    Logs per-component reward breakdown to TensorBoard every step.
    Reads from info['reward_breakdown'] which is populated by railway_env.step().
    
    Enables per-component TensorBoard charts so each reward term can be
    monitored independently — essential for diagnosing weight imbalance
    and verifying PBRS shaping is behaving as expected.
    """

    REWARD_KEYS = [
        "r_step",
        "r_progress",
        "r_speed",
        "r_signal_compliance",
        "r_station",
        "r_time",
        "r_jerk",
        "r_patience",
        "r_shaping",   # PBRS shaping term — will be 0.0 until Task 6 is complete
        "r_total",
    ]

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if not infos:
            return True

        for key in self.REWARD_KEYS:
            values = [
                info["reward_breakdown"][key]
                for info in infos
                if "reward_breakdown" in info and key in info["reward_breakdown"]
            ]
            if values:
                self.logger.record(
                    f"reward/{key}",
                    sum(values) / len(values)
                )
        return True


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

    # save_freq is in _on_step calls (1 call = n_envs timesteps)
    checkpoint_cb = CheckpointCallback(
        save_freq=max(1, config.checkpoint_freq // config.n_envs),
        save_path=checkpoint_dir,
        name_prefix="rl_model",
        verbose=1,
    )

    # Separate eval environment (also normalised, but stats frozen)
    eval_venv = DummyVecEnv(
        [
            make_env(
                seed=config.seed + 1000,
                lead_train_speed=config.lead_train_speed,
                slack_factor=config.slack_factor,
                max_episode_steps=config.max_episode_steps,
            )
        ]
    )
    eval_venv = VecNormalize(
        eval_venv,
        norm_obs=config.normalize_obs,
        norm_reward=False,  # raw reward for evaluation
        clip_obs=config.norm_obs_clip,
    )
    # Sync normalisation stats from training env
    eval_venv.obs_rms = vec_env.obs_rms
    eval_venv.training = False  # freeze stats during evaluation

    # eval_freq is in _on_step calls (1 call = n_envs timesteps)
    eval_cb = EvalCallback(
        eval_venv,
        best_model_save_path=config.model_dir,
        log_path=config.log_dir,
        eval_freq=max(1, config.eval_freq // config.n_envs),
        n_eval_episodes=config.eval_episodes,
        deterministic=True,
        verbose=1,
    )

    sync_cb = SyncNormCallback(vec_env, eval_venv, sync_freq=10_000)
    save_best_vecnorm_cb = SaveBestVecNormalizeCallback(vec_env, eval_cb, config.model_dir)
    reward_logger_cb = RewardLoggerCallback()

    callbacks = CallbackList([checkpoint_cb, eval_cb, sync_cb, save_best_vecnorm_cb, reward_logger_cb])

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
    best_vecnorm_path = os.path.join(config.model_dir, "best_vecnormalize.pkl")

    model.save(final_model_path)
    vec_env.save(final_vecnorm_path)
    # Also save as best_vecnormalize.pkl to ensure dashboard fallback works
    vec_env.save(best_vecnorm_path)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print(f"  Final model  : {final_model_path}.zip")
    print(f"  Best model   : {os.path.join(config.model_dir, 'best_model.zip')}")
    print(f"  VecNormalize : {final_vecnorm_path} and {best_vecnorm_path}")
    print("=" * 60)
