"""FULL DIAGNOSTIC: Why is the agent STILL stuck after training?"""
import sys, os, numpy as np
sys.path.insert(0, '.')

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.railway_env import ModernizedLine104
from reward.reward_function import compute_reward, TrainState, RewardConfig

print("=" * 70)
print("PART 1: TRAINING EVALUATION HISTORY")
print("=" * 70)
eval_path = os.path.join("think_layer", "runs", "evaluations.npz")
if os.path.exists(eval_path):
    data = np.load(eval_path)
    timesteps = data['timesteps']
    results = data['results']
    n_pts = min(len(timesteps), 20)
    # Show first 5 and last 15
    indices = list(range(min(5, len(timesteps)))) + list(range(max(0, len(timesteps)-15), len(timesteps)))
    indices = sorted(set(indices))
    for i in indices:
        ts = timesteps[i]
        mean_r = np.mean(results[i])
        std_r = np.std(results[i])
        print(f"  {ts:>12,} steps: mean_reward={mean_r:+10.2f}  std={std_r:8.2f}")
    print(f"\n  First eval: {np.mean(results[0]):+.2f}  Last eval: {np.mean(results[-1]):+.2f}")
    print(f"  Change: {np.mean(results[-1]) - np.mean(results[0]):+.2f}")
else:
    print("  No evaluations.npz found!")

print()
print("=" * 70)
print("PART 2: TRAINED MODEL BEHAVIOR")
print("=" * 70)

model_path = os.path.join("think_layer", "models", "best_model.zip")
vecnorm_path = os.path.join("think_layer", "models", "best_vecnormalize.pkl")

if not os.path.exists(model_path):
    print("  No model found!")
    sys.exit(1)

model = PPO.load(model_path)

def _make_env():
    return ModernizedLine104(lead_train_speed=20.0, training_mode=False)

venv = DummyVecEnv([_make_env])
if os.path.exists(vecnorm_path):
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    print(f"  VecNormalize loaded. obs_rms.mean={venv.obs_rms.mean}")
    print(f"  VecNormalize obs_rms.var={venv.obs_rms.var}")
    print(f"  VecNormalize obs_rms.count={venv.obs_rms.count}")
    # Check if reward normalization is on
    print(f"  norm_obs={venv.norm_obs}  norm_reward={venv.norm_reward}")
    if hasattr(venv, 'ret_rms'):
        print(f"  ret_rms.mean={venv.ret_rms.mean}  ret_rms.var={venv.ret_rms.var}")

obs = venv.reset()
base_env = venv.envs[0]
while hasattr(base_env, 'env'):
    base_env = base_env.env

actions_taken = []
rewards_taken = []

for step in range(200):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, dones, infos = venv.step(action)
    raw_a = float(action[0][0])
    actions_taken.append(raw_a)
    rewards_taken.append(float(reward[0]))
    
    if step < 20:
        info = infos[0]
        print(f"  Step {step:3d}: action={raw_a:+.4f}  "
              f"proposed_a={info.get('proposed_a',0):+.4f}  "
              f"safe_a={info.get('safe_a',0):+.4f}  "
              f"x={base_env.x:.1f}  v={base_env.v:.3f}  "
              f"reward={reward[0]:+.6f}")
    
    if dones[0]:
        print(f"  Episode ended at step {step}")
        break

print(f"\n  Final: x={base_env.x:.1f}  v={base_env.v:.3f}  stations={sorted(base_env.visited_stations)}")
print(f"  Mean action: {np.mean(actions_taken):+.4f}")
print(f"  Std action:  {np.std(actions_taken):.4f}")
print(f"  Min/Max action: {np.min(actions_taken):+.4f} / {np.max(actions_taken):+.4f}")

# Check stochastic behavior
obs2 = venv.reset()
stoch_actions = []
for _ in range(200):
    a, _ = model.predict(obs2, deterministic=False)
    stoch_actions.append(float(a[0][0]))
print(f"\n  Stochastic (200 samples from initial state):")
print(f"    mean={np.mean(stoch_actions):+.4f}  std={np.std(stoch_actions):.6f}")
print(f"    min={np.min(stoch_actions):+.4f}  max={np.max(stoch_actions):+.4f}")
if np.std(stoch_actions) < 0.01:
    print("    *** EXPLORATION COLLAPSED — std is essentially zero! ***")

print()
print("=" * 70)
print("PART 3: TRAINING CONFIG")
print("=" * 70)
from think_layer.config import TrainConfig
cfg = TrainConfig()
print(f"  total_timesteps:   {cfg.total_timesteps:,}")
print(f"  n_envs:            {cfg.n_envs}")
print(f"  n_steps:           {cfg.n_steps}")
print(f"  batch_size:        {cfg.batch_size}")
print(f"  n_epochs:          {cfg.n_epochs}")
print(f"  learning_rate:     {cfg.learning_rate}")
print(f"  gamma:             {cfg.gamma}")
print(f"  gae_lambda:        {cfg.gae_lambda}")
print(f"  clip_range:        {cfg.clip_range}")
print(f"  ent_coef:          {cfg.ent_coef}")
print(f"  max_episode_steps: {cfg.max_episode_steps}")
print(f"  use_sde:           {cfg.use_sde}")
if hasattr(cfg, 'sde_sample_freq'):
    print(f"  sde_sample_freq:   {cfg.sde_sample_freq}")

print()
print("=" * 70)
print("PART 4: RAW ENVIRONMENT CHECK (no model, just env)")
print("=" * 70)
# Verify the env itself gives good rewards for positive actions
env = ModernizedLine104(lead_train_speed=20.0, training_mode=True)
env.reset()

# Check raw obs space
obs_raw = env._get_obs()
print(f"  Raw obs: {obs_raw}")
print(f"  Obs space: {env.observation_space}")
print(f"  Action space: {env.action_space}")

# Run 50 steps with action=+1.0 and check cumulative
total = 0.0
for s in range(50):
    obs, r, term, trunc, info = env.step(np.array([1.0], dtype=np.float32))
    total += r
print(f"\n  50 steps with action=+1.0: cumulative_reward={total:+.4f}  x={env.x:.1f}  v={env.v:.2f}")

env.reset()
total2 = 0.0
for s in range(50):
    obs, r, term, trunc, info = env.step(np.array([0.0], dtype=np.float32))
    total2 += r
print(f"  50 steps with action= 0.0: cumulative_reward={total2:+.4f}  x={env.x:.1f}  v={env.v:.2f}")
print(f"  Difference: {total - total2:+.4f}")

print()
print("=" * 70)
print("PART 5: VecNormalize IMPACT")
print("=" * 70)
# Check what VecNormalize does to the rewards
def _make_train_env():
    return ModernizedLine104(lead_train_speed=20.0, training_mode=True)

venv2 = DummyVecEnv([_make_train_env])
# Check if trainer uses VecNormalize with reward normalization
from think_layer.trainer import train
import inspect
source = inspect.getsource(train)
if 'norm_reward' in source:
    print("  trainer.py uses norm_reward in VecNormalize setup")
    if 'norm_reward=True' in source:
        print("  *** norm_reward=True — rewards ARE being normalized! ***")
    elif 'norm_reward=False' in source:
        print("  norm_reward=False — rewards are NOT normalized")
if 'VecNormalize' in source:
    print("  VecNormalize IS used in training")
    # Check gamma
    if 'gamma' in source:
        import re
        gamma_matches = re.findall(r'gamma\s*=\s*([0-9.]+)', source)
        print(f"  VecNormalize gamma values found: {gamma_matches}")
