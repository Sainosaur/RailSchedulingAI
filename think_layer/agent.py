"""
think_layer/agent.py

Factory functions for creating the Gymnasium environment and the
SB3 PPO agent, including VecNormalize wrapping.
"""

import sys
from pathlib import Path
from typing import Callable

import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from environment.railway_env import ModernizedLine104  # noqa: E402
from think_layer.config import TrainConfig  # noqa: E402


def make_env(
    seed: int = 42,
    lead_train_speed: float = 20.0,
    max_episode_steps: int = 15_000,
) -> Callable[[], gym.Env]:
    """
    Return a thunk that creates a MonitoredLine104 environment.

    SB3's DummyVecEnv expects a list of callables, each returning a Gym env.
    Monitor wraps the env to log episode stats (reward, length).
    """

    def _init() -> gym.Env:
        env = ModernizedLine104(lead_train_speed=lead_train_speed)
        # Wrap in TimeLimit for episode truncation
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
        env = Monitor(env)
        env.reset(seed=seed)
        return env

    return _init


def build_agent(config: TrainConfig) -> tuple[PPO, VecNormalize]:
    """
    Build the PPO agent with VecNormalize and TensorBoard logging.

    Returns
    -------
    (model, vec_env) — the PPO model and the VecNormalize wrapper
                       (needed to save/load normalisation stats).
    """
    # 1. Vectorised environment (single env for now)
    venv = DummyVecEnv([
        make_env(
            seed=config.seed,
            lead_train_speed=config.lead_train_speed,
            max_episode_steps=config.max_episode_steps,
        )
    ])

    # 2. Observation & reward normalisation
    venv = VecNormalize(
        venv,
        norm_obs=config.normalize_obs,
        norm_reward=config.normalize_reward,
        clip_obs=config.norm_obs_clip,
        clip_reward=config.norm_reward_clip,
    )

    # 3. PPO model
    model = PPO(
        policy="MlpPolicy",
        env=venv,
        learning_rate=config.learning_rate,
        n_steps=config.n_steps,
        batch_size=config.batch_size,
        n_epochs=config.n_epochs,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        tensorboard_log=config.log_dir,
        seed=config.seed,
        verbose=1,
        policy_kwargs=dict(
            net_arch=dict(
                pi=config.policy_net,
                vf=config.value_net,
            ),
        ),
    )

    return model, venv
