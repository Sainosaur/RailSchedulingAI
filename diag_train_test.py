"""Verify that training can run with the new config."""
import sys, os
sys.path.insert(0, '.')

from think_layer.config import TrainConfig
from think_layer.agent import build_agent

if __name__ == '__main__':
    print("Checking config...")
    cfg = TrainConfig(total_timesteps=1000, n_envs=2, n_steps=100, batch_size=50, n_epochs=2)
    print(f"normalize_reward: {cfg.normalize_reward}")
    print(f"ent_coef: {cfg.ent_coef}")
    print(f"learning_rate: {cfg.learning_rate}")
    print(f"gamma: {cfg.gamma}")

    print("\nBuilding agent...")
    model, venv = build_agent(cfg)

    print("\nRunning training for 1000 steps...")
    model.learn(total_timesteps=cfg.total_timesteps)

    print("\nTraining completed successfully without crashes!")
    print(f"VecNormalize stats: norm_reward={venv.norm_reward}, ret_rms.var={getattr(venv, 'ret_rms', None).var if hasattr(venv, 'ret_rms') else 'None'}")
