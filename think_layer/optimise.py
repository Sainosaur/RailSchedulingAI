"""
think_layer/optimize.py

Optuna Bayesian Hyperparameter Optimization pipeline for the PPO agent.
Uses SQLite persistent storage, prunes unpromising trials, and runs in parallel.

Usage:
    python -m think_layer.optimize --trials 50
"""

import argparse
import os
import sys
from pathlib import Path

import optuna
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from think_layer.agent import build_agent, make_env
from think_layer.config import TrainConfig

# ── HPO-specific constants ────────────────────────────────────────────────────
# These are deliberately smaller than the full-training defaults to maximise
# trial throughput.  train.py / trainer.py use "PPO_Line104" as its log prefix;
# HPO trials use "PPO_{trial_number}" so the two namespaces never collide.

_HPO_TOTAL_TIMESTEPS = 150_000  # reduced: 50 trials × 150k is plenty to rank configs
_HPO_N_ENVS = 128  # fewer workers → less SubprocVecEnv IPC overhead
_HPO_MAX_EP_STEPS = 10_000   # halved: cuts stalled-episode tail, still covers full line
_HPO_N_EVAL_EPS = 1  # sequential eval is the bottleneck; 1 ep is enough
_HPO_TB_PREFIX = "PPO"  # saved as PPO_1, PPO_2 … never PPO_Line104_x


class TrialEvalCallback(EvalCallback):
    """
    Custom Evaluation Callback that reports intermediate results to Optuna.
    If the trial is performing significantly worse than the best trials so far,
    Optuna will prune (kill) it early to save compute time.
    """

    def __init__(
        self, eval_env, trial, n_eval_episodes=5, eval_freq=10_000, *args, **kwargs
    ):
        super().__init__(
            eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            best_model_save_path=None,
            log_path=None,
            *args,
            **kwargs,
        )
        self.trial = trial
        self.eval_idx = 0
        self.is_pruned = False

    def _on_step(self) -> bool:
        continue_training = super()._on_step()
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            self.eval_idx += 1
            # self.last_mean_reward is populated by the parent EvalCallback
            self.trial.report(self.last_mean_reward, self.eval_idx)

            if self.trial.should_prune():
                self.is_pruned = True
                return False  # signals SB3 to halt training
        return continue_training


def objective(trial: optuna.Trial) -> float:
    """
    Optuna objective function.
    1. Samples hyperparameters for this trial.
    2. Builds the environment and model.
    3. Trains the model while periodically evaluating and pruning.
    4. Returns the final evaluation score.
    """
    # ── 1. Sample Hyperparameters ──────────────────────────────────────────────
    kwargs = {
        # Tightened around known-good PPO range for continuous control
        "learning_rate": trial.suggest_float("lr", 5e-5, 5e-4, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [256, 512]),
        "n_steps": trial.suggest_categorical("n_steps", [1024, 2048]),
        # Long-horizon journey — high gamma is essential, no point exploring lower
        "gamma": trial.suggest_categorical("gamma", [0.99, 0.999]),
        "gae_lambda": trial.suggest_categorical("gae_lambda", [0.90, 0.95, 0.99]),
        # Keep entropy low — this env has clear signal, not sparse reward
        "ent_coef": trial.suggest_float("ent_coef", 0.001, 0.05, log=True),
        "n_epochs": trial.suggest_categorical("n_epochs", [5, 10]),
    }

    # Drop xlarge/deep variants — overkill for a 9-dim obs space, wastes trial time
    net_arch_type = trial.suggest_categorical(
        "net_arch", ["small", "medium", "large"]
    )
    net_arch_map = {
        "small":  [128, 128],
        "medium": [256, 256],
        "large":  [512, 512],
    }
    net_arch = net_arch_map[net_arch_type]

    # ── 2. Configure Run ───────────────────────────────────────────────────────
    config = TrainConfig()
    config.learning_rate = kwargs["learning_rate"]
    config.batch_size = kwargs["batch_size"]
    config.n_steps = kwargs["n_steps"]
    config.gamma = kwargs["gamma"]
    config.gae_lambda = kwargs["gae_lambda"]
    config.n_epochs = kwargs["n_epochs"]
    config.ent_coef = kwargs["ent_coef"]
    config.policy_net = net_arch
    config.value_net = net_arch

    # HPO-specific overrides for speed
    config.total_timesteps = _HPO_TOTAL_TIMESTEPS
    config.n_envs = _HPO_N_ENVS
    config.max_episode_steps = _HPO_MAX_EP_STEPS

    # eval_freq is in _on_step calls (1 call = n_envs timesteps)
    config.eval_freq = 25_000 // config.n_envs  # ~2 evals per 100 k trial

    # Disable TensorBoard to save IO/memory and sidestep the Protobuf metaclass
    # error on Python 3.14.  train.py uses "PPO_Line104_x"; HPO uses "PPO_x".
    config.log_dir = None

    model, env = build_agent(config)

    # ── 3. Evaluation Setup ────────────────────────────────────────────────────
    eval_venv = DummyVecEnv(
        [
            make_env(
                seed=config.seed + trial.number,
                lead_train_speed=config.lead_train_speed,
                lead_stop_offset=config.lead_stop_offset,
                max_episode_steps=config.max_episode_steps,
            )
        ]
    )
    eval_venv = VecNormalize(eval_venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    # Sync normalisation stats from the training env so evaluation is consistent
    eval_venv.obs_rms = env.obs_rms
    eval_venv.training = False

    eval_cb = TrialEvalCallback(
        eval_venv,
        trial,
        n_eval_episodes=_HPO_N_EVAL_EPS,
        eval_freq=config.eval_freq,
    )

    # ── 4. Train ───────────────────────────────────────────────────────────────
    # tb_log_name is only used when tensorboard_log is set (it's None here),
    # but we pass it explicitly so if logging is ever re-enabled for debugging
    # it writes to PPO_<n> and never stomps on train.py's PPO_Line104_x runs.
    try:
        model.learn(
            total_timesteps=config.total_timesteps,
            callback=eval_cb,
            tb_log_name=f"{_HPO_TB_PREFIX}_{trial.number}",
        )
    except AssertionError as e:
        # Absurd hyperparameter combos can produce NaN losses in SB3
        print(f"Trial {trial.number} crashed: {e}")
        raise optuna.exceptions.TrialPruned()
    finally:
        # Always release resources to prevent memory leaks across hundreds of trials
        model.env.close()
        eval_venv.close()

    # ── 5. Optionally save the trained model for this trial ────────────────────
    # Files are written as  models/hpo/PPO_<trial_number>.zip
    # This is separate from train.py's  models/best_model.zip  /  final_model.zip
    base_config = TrainConfig()  # use default paths
    hpo_model_dir = os.path.join(base_config.model_dir, "hpo")
    os.makedirs(hpo_model_dir, exist_ok=True)
    trial_model_path = os.path.join(hpo_model_dir, f"{_HPO_TB_PREFIX}_{trial.number}")
    model.save(trial_model_path)

    # ── 6. Prune / Return ──────────────────────────────────────────────────────
    if eval_cb.is_pruned:
        raise optuna.exceptions.TrialPruned()

    return eval_cb.last_mean_reward


def optimize(n_trials: int = 150):
    """Orchestrates the Optuna study."""
    db_path = os.path.join(TrainConfig().log_dir, "optuna_study.db")
    storage = f"sqlite:///{db_path}"

    print("=" * 60)
    print("  Line 104 — Bayesian Hyperparameter Optimization")
    print("=" * 60)
    print(f"  Trials        : {n_trials}")
    print(
        f"  Envs/trial    : {_HPO_N_ENVS}  (vs {TrainConfig().n_envs} in full training)"
    )
    print(f"  Steps/trial   : {_HPO_TOTAL_TIMESTEPS:,}")
    print(f"  Max ep steps  : {_HPO_MAX_EP_STEPS:,}")
    print(f"  Model prefix  : {_HPO_TB_PREFIX}_<n>  (train.py uses PPO_Line104_<n>)")
    print(f"  Database      : {storage}")
    print("=" * 60)

    # MedianPruner kills runs that fall below the 50th percentile of previous runs
    pruner = optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=2)

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
    print("    Params:")
    for key, value in trial.params.items():
        print(f"      {key}: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trials", type=int, default=50, help="Number of trials to run"
    )
    args = parser.parse_args()

    optimize(n_trials=args.trials)