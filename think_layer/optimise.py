"""
think_layer/optimize.py

Optuna Bayesian Hyperparameter Optimization pipeline for the PPO agent.
Uses SQLite persistent storage, prunes unpromising trials, and runs in parallel.

Usage:
    python -m think_layer.optimize --trials 50
"""

import os
import sys
import argparse
from pathlib import Path
import optuna

from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from think_layer.config import TrainConfig
from think_layer.agent import build_agent, make_env


class TrialEvalCallback(EvalCallback):
    """
    Custom Evaluation Callback that reports intermediate results to Optuna.
    If the trial is performing significantly worse than the best trials so far,
    Optuna will prune (kill) it early to save compute time.
    """
    def __init__(self, eval_env, trial, n_eval_episodes=5, eval_freq=10000, *args, **kwargs):
        super().__init__(
            eval_env, 
            n_eval_episodes=n_eval_episodes, 
            eval_freq=eval_freq, 
            best_model_save_path=None, 
            log_path=None, 
            *args, **kwargs
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # self.last_mean_reward is populated by the parent EvalCallback computation
            self.trial.report(self.last_mean_reward, self.eval_idx)
            
            # Prune trial if need be
            if self.trial.should_prune():
                self.is_pruned = True
                return False  # Signals SB3 to halt training
        return continue_training


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    1. Samples hyperparameters for this trial.
    2. Builds the environment and model.
    3. Trains the model while periodically evaluating and pruning.
    4. Returns the final evaluation score.
    """
    # ── 1. Sample Hyperparameters ──
    kwargs = {
        "learning_rate": trial.suggest_float("lr", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048, 4096]),
        "gamma": trial.suggest_categorical("gamma", [0.99, 0.995, 0.999, 0.9999]),
        "ent_coef": trial.suggest_float("ent_coef", 0.0001, 0.5, log=True),
    }
    
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium"])
    net_arch = [64, 64] if net_arch_type == "small" else [128, 128]
    
    # ── 2. Configure Run ──
    config = TrainConfig()
    config.learning_rate = kwargs["learning_rate"]
    config.batch_size = kwargs["batch_size"]
    config.n_steps = kwargs["n_steps"]
    config.gamma = kwargs["gamma"]
    config.ent_coef = kwargs["ent_coef"]
    config.policy_net = net_arch
    config.value_net = net_arch
    
    # For HPO, we use shorter runs to test more configurations, evaluating frequently to enable pruning
    config.total_timesteps = 100_000  
    # eval_freq is in _on_step calls (1 call = n_envs timesteps), so divide by n_envs
    config.eval_freq = 25_000 // config.n_envs  # ~2 evals per 100k trial
    
    # Disable tensorboard logging to save IO and memory across 100s of trials
    model, env = build_agent(config)
    
    # ── 3. Evaluation Setup ──
    eval_venv = DummyVecEnv([
        make_env(
            seed=config.seed + trial.number, 
            lead_train_speed=config.lead_train_speed,
            max_episode_steps=config.max_episode_steps
        )
    ])
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    eval_venv.obs_rms = env.obs_rms
    eval_venv.training = False
    
    eval_cb = TrialEvalCallback(eval_venv, trial, n_eval_episodes=3, eval_freq=config.eval_freq)
    
    # ── 4. Train ──
    try:
        model.learn(total_timesteps=config.total_timesteps, callback=eval_cb)
    except AssertionError as e:
        # Sometimes absurd hyperparameters cause NaN errors in SB3
        print(f"Trial failed due to crash: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        # Prevent memory leaks across hundreds of trials
        model.env.close()
        eval_venv.close()

    # ── 5. Pruning / Return ──
    if eval_cb.is_pruned:
        raise optuna.exceptions.TrialPruned()
        
    return eval_cb.last_mean_reward


def optimize(n_trials: int = 50):
    """Orchestrates the Optuna study."""
    db_path = os.path.join(TrainConfig().log_dir, "optuna_study.db")
    storage = f"sqlite:///{db_path}"

    print("=" * 60)
    print("  Line 104 — Bayesian Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Trials   : {n_trials}")
    print(f"  Database : {storage}")
    print("=" * 60)

    # MedianPruner kills runs that fall below the 50th percentile of previous runs
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
    
    study = optuna.create_study(
        study_name="ppo_line104",
        direction="maximize",
        storage=storage,
        load_if_exists=True,
        pruner=pruner,
    )
    
    study.optimize(objective, n_trials=n_trials)
    
    print("\n" + "=" * 60)
    print("  Optimization Complete!")
    print("=" * 60)
    print("  Best trial:")
    trial = study.best_trial
    
    print(f"    Value (Mean Reward): {trial.value}")
    print("    Params: ")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=50, help="Number of trials to run")
    args = parser.parse_args()
    
    optimize(n_trials=args.trials)
