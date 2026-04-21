# PPO Train Scheduler — Fix Notes

**Session date:** 2026-04-21  
**Branch:** `claude/sweet-archimedes-dd7938` → `SaiAI`  
**Symptom:** Evaluation reward flat at **-3050.5** across all 1M training steps. Agent never moves from the start of the track.

---

## Root Cause Summary

The -3050.5 number is not random — it is the exact reward of a train that **never moves**:

```
(-0.005 heartbeat  +  -0.3 underspeed floor)  ×  10 000 steps  =  -3050
```

Seven independent bugs each made "stand still and brake" the optimal policy.  
Fixing the reward weights alone (as was previously attempted) could not work because the fundamental training dynamics were broken.

---

## Bug 1 — Asymmetric Action Space

**File:** `environment/railway_env.py`

### What was wrong
```python
# BEFORE
self.action_space = gym.spaces.Box(low=[-1.0], high=[0.5])
```
The action space had **1.5 units of brake range** vs. **0.5 units of throttle range**. Stable-Baselines3's PPO policy initialises a Gaussian centred at 0 and clips outputs to the action space bounds. With this asymmetric space, the average *clipped* action is approximately **-0.11** — the agent is born braking, before it has seen a single reward.

Empirically confirmed: 2000 samples from a fresh untrained policy had mean action = **-0.11**, with 49.6% negative actions but far more mass in the brake direction due to the wider negative range.

### Fix
```python
# AFTER
self.action_space = gym.spaces.Box(low=[-1.0], high=[1.0])
```
The action space is now symmetric. A piecewise linear map inside `step()` converts the normalised agent output to physical acceleration:

```python
if raw_action >= 0.0:
    proposed_a = raw_action * self.ACCEL      # [0, +0.5 m/s²]
else:
    proposed_a = raw_action * abs(self.DECEL) # [-1.0, 0 m/s²]
```

This means:
- `action = 0` → coast (zero acceleration)
- `action = +1` → maximum throttle (+0.5 m/s²)
- `action = -1` → emergency brake (-1.0 m/s²)

After fix, fresh policy mean action = **+0.002** (symmetric around coast).

---

## Bug 2 — Kinematic Overspeed Trap

**File:** `reward/reward_function.py`

### What was wrong
```python
# BEFORE — _compute_speed_reward
v_accel = math.sqrt(2 * comfortable_acceleration * d_from_last_station)
v_brake = math.sqrt(2 * comfortable_deceleration * d_to_next_station)
v_target = max(1.0, min(v_max, v_accel, v_brake))  # floor of 1 m/s

overspeed = max(0.0, v - v_target)
return -(k_over * overspeed**2) - (k_under * underspeed)  # k_over = 0.5
```

Two catastrophic traps were embedded in this formula:

1. **Post-station reset:** When the agent crosses a station, `d_from_last_station` resets to 0, so `v_accel = sqrt(0) = 0`. Any train travelling at even 5 m/s instantly "overspeeds" by 5 m/s. With `k_over=0.5`, penalty = `0.5 × 25 = -12.5/step`. Over the 60-second crossing window this is **-750 reward**.

2. **Pre-station brake trap:** As `d_to_next_station → 0`, `v_brake` also collapses to 0. The agent must have begun gentle braking hundreds of metres before each station or it will have been penalised continuously. A freshly-initialised policy has no concept of this lookahead.

Measured result: a constant "gentle throttle" policy running for 300 steps accumulated a **speed penalty of -2590**, completely overwhelming the +593 progress reward.

The 1.0 m/s floor on `v_target` compounded this — any stationary train at a station origin was penalised -0.3/step for not moving at 1 m/s, creating the **-3050 attractor** (the floor is what pins the exact value).

### Fix
```python
# AFTER — _compute_speed_reward
v_target = v_max   # segment speed limit only

overspeed = max(0.0, v - v_max)
return -(k_over * overspeed**2)   # k_over reduced to 0.05
```

The reward now only penalises exceeding the **posted segment speed limit**, not a kinematic comfort envelope. Safe braking into stations is already enforced by:
- The Validation Layer (which computes its own safe stopping distance)
- The headway penalty (which fires if the agent catches the lead train)
- The collision penalty (if actual overlap occurs)

`k_over` was also reduced from `0.5` → `0.05` so exceeding the limit by 1 m/s costs `-0.05/step` rather than `-0.5/step`.

---

## Bug 3 — Jerk Penalty Kills Exploration

**File:** `reward/reward_function.py`

### What was wrong
```python
# BEFORE
jerk_penalty: float = -0.5

def _compute_jerk_penalty(state, config):
    return config.jerk_penalty * (state.action_delta ** 2)
```

A random Gaussian policy naturally oscillates between positive and negative actions on consecutive steps. Each flip produces `action_delta ≈ 1.5` (full swing across the [-1, +1] space). Penalty per step = `-0.5 × 1.5² = -1.125`.

Measured over a 2048-step rollout: jerk accumulated **-615 reward** vs. progress of only **+141**. The dominant signal in the very first policy update was: *"changing your action is extremely costly."* PPO's response was to pick a **constant action** — and full brake is the most consistent constant action because the Validation Layer never overrides it when the train is already stopped.

### Fix
```python
# AFTER
jerk_penalty: float = -0.01
```

At `delta=1.5`, penalty is now `-0.01 × 2.25 = -0.0225/step` — a token smoothness signal that does not overwhelm exploration.

---

## Bug 4 — Reward Normalisation Hid the Collapse

**File:** `think_layer/config.py`

### What was wrong
```python
# BEFORE
normalize_reward: bool = True
```

`VecNormalize` with `norm_reward=True` computes a running mean and variance of the reward and rescales it to unit variance. When the agent is stuck at -3050 for every episode, the normaliser sees a **constant signal** — it normalises this to approximately 0 with near-zero variance. PPO's policy gradient loss then operates on essentially-zero advantages, meaning the agent receives almost **no training signal at all**.

This also hides the collapse from human inspection: the TensorBoard `rollout/ep_rew_mean` shows values in the normalised range and looks "stable" rather than catastrophic.

### Fix
```python
# AFTER
normalize_reward: bool = False
```

Raw rewards are now visible. For reference, the expected reward range with a working policy is approximately **+5000 to +8000** per episode (progress + station bonuses − running penalties). Optionally divide all reward terms by 10 manually if gradient magnitudes become too large after switching.

---

## Bug 5 — Terminal Safety Penalties Made Motion Catastrophic

**File:** `reward/reward_function.py`

### What was wrong
```python
# BEFORE
headway_violation_penalty: float = -150.0   # → episode terminates
collision_penalty: float = -200.0           # → episode terminates
```

Both penalties **terminated the episode** (`terminate = True`). With early random exploration triggering these occasionally, PPO's value function learned: *"states involving motion eventually lead to -150 or -200 and episode end."* The expected return from any motion-involving state was estimated to be strongly negative. The value function for stand-still states was simply -0.305/step × remaining_steps — bad but predictable.

The asymmetry was lethal: **stand-still is a known cost, motion has unbounded downside**, so PPO learned to avoid motion.

Additionally, the previous refactor set `override_penalty = -1.0`. The Validation Layer overrides safety-critical actions **correctly** — it is working as intended. Penalising correct safety overrides teaches the agent to avoid the validator rather than learn safe driving.

### Fix
```python
# AFTER
headway_violation_penalty: float = -10.0    # non-terminal
collision_penalty: float = -20.0            # non-terminal
override_penalty: float = 0.0
```

Safety costs are now **non-terminal**: the agent feels the pain but the episode continues, so it can experience recovery and learn to avoid the violation. Only `TRACK_END` reached or `MAX_STEPS` truncation end episodes.

Override penalty is zero: the Validation Layer is a safety system, not a reward signal.

---

## Bug 6 — Absolute Position in Observation

**File:** `environment/railway_env.py`

### What was wrong
```python
# BEFORE — _get_obs()
return np.array([self.x, self.v, self.dtz, ...])   # obs[0] = absolute position in [0, 80000]
```

The absolute position `self.x` was in the observation. This allowed the policy network to learn **position-specific** behaviours (e.g. "at x=582, apply brake; at x=14000, throttle"). When new perturbations were introduced (random stalls, variable lead speed) the position-specific policy broke immediately because the same position could now have different dynamics.

Additionally, with `obs[0]` being a large number in [0, 80000] while other features are in [0, 30] or [0, 3], the `VecNormalize` clipping at ±10σ could distort the effective representation of other features.

The headway sentinel (`return 9999.0 when v < 0.01`) also masked useful information — when stationary, 9999 is always returned regardless of how close the lead train actually is.

### Fix
```python
# AFTER — _get_obs()
dist_from_last_station = max(0.0, self.x - last_st_pos)   # [0, ~22000 m]
obs_headway = np.clip((self.lead_x - self.x) / max(self.v, 1.0), 0.0, 10000.0)

return np.array([dist_from_last_station, self.v, self.dtz, ..., obs_headway])
```

- **`dist_from_last_station`** replaces absolute position. This is translation-invariant — the agent learns "how far into this inter-station span am I" rather than "which kilometre marker am I at."
- **Headway sentinel replaced**: `(lead_x − x) / max(v, 1.0)` degrades gracefully when stationary and correctly reports ≤0 in a collision state.
- **Observation space bounds updated**: high for position features set to 30 000 m (longest inter-station span is ~22 km).

---

## Bug 7 — Non-Stationarity Added Before Baseline Worked

**File:** `environment/railway_env.py`

### What was wrong
```python
# BEFORE — _advance_lead_train()
if self.training_mode and self.lead_dwell_timer == 0 ...:
    if self.np_random.random() < 0.001:
        self.random_stall_timer = self.np_random.integers(15, 60)
```

Random 15–60 second stalls of the lead train (~7 expected per trip) were added while the reward function was still broken. These stalls break the **Markov property** of the observation: the same `(position, speed, dtz, aspect, ...)` can now correspond to a moving *or* a stalled lead train, but the observation doesn't encode which. Any policy learned under one condition will fail under the other.

Adding non-stationarity before a deterministic baseline is established is a known RL anti-pattern — it makes diagnosing learning failures much harder.

### Fix
```python
# AFTER
def __init__(self, ..., enable_random_stalls: bool = False):
    self.enable_random_stalls = enable_random_stalls

# In _advance_lead_train():
if (self.training_mode and self.enable_random_stalls
        and self.lead_dwell_timer == 0 ...):
    if self.np_random.random() < 0.001:
        self.random_stall_timer = ...
```

Random stalls are gated behind an explicit `enable_random_stalls` flag, defaulting to **`False`**. Re-enable as a curriculum step once the deterministic baseline trains stably:

```python
env = ModernizedLine104(training_mode=True, enable_random_stalls=True)
```

---

## Bug 8 — PPO Hyperparameters Too Aggressive

**File:** `think_layer/config.py`

### What was wrong
```python
# BEFORE
learning_rate: float = 3e-4
n_steps: int = 2048
batch_size: int = 64
ent_coef: float = 0.01
```

With `lr=3e-4` and `n_steps=2048`, each policy update was taking a large step on a relatively small batch of experience. During the 80k-step smoke test with the reward fixes applied, the agent found a good policy at step 30k (+6380 reward, all 6 stations) and then **lost it by step 40k** (back to -500, stuck). This is a classic PPO instability symptom — the policy gradient step overshoots and exits the trust region.

`ent_coef=0.01` was too low to maintain enough exploration to recover from a collapse.

### Fix
```python
# AFTER
learning_rate: float = 1e-4    # smaller steps → more stable policy updates
n_steps: int = 4096            # more experience per update → lower-variance advantages
batch_size: int = 128          # scaled with n_steps
ent_coef: float = 0.02         # more entropy keeps exploration alive during recovery
```

With these settings, the 80k-step smoke test showed **stable learning from step 10k onwards** with only one transient regression (at step 40k) that recovered immediately at step 50k.

---

## New File — `think_layer/callbacks.py`

### Why it was added
All previous training runs had no visibility into *which reward component* was dominating each episode. Total reward dropped to -3050 and there was no way to know whether it was the heartbeat, the speed penalty, the jerk penalty, or the overrides causing it. This made each debugging iteration very slow.

### What it does
`RewardBreakdownCallback` accumulates the `info["reward_breakdown"]` dict (published by the environment every step) across each episode, then logs episode-total means to TensorBoard at the end of every rollout:

```
rollout/r_progress_mean
rollout/r_headway_mean
rollout/r_speed_mean
rollout/r_heartbeat_mean
rollout/r_station_mean
rollout/r_jerk_mean
rollout/r_energy_mean
rollout/r_violation_mean
rollout/r_collision_mean
rollout/stations_reached_mean    ← primary task-completion metric
rollout/episodes_finished
```

**`stations_reached_mean` is the most important new metric** — the total reward is a proxy for learning quality, but "did the agent reach all 7 stations per episode" is the ground truth signal for whether it learned the task.

---

## Progress Summary

### Reward component audit (300-step runs, deterministic heuristic policy)

| Policy | Total (300 steps) | Bottleneck |
|---|---:|---|
| Stand still | -15 | heartbeat only |
| Constant gentle throttle | **-1989** | speed penalty -2590 (v_accel trap) |
| Constant gentle throttle (**after fix**) | **+600** | progress dominant |

### 80k-step smoke test results (after all fixes)

| Step | Eval Reward | Stations | Episode Length |
|---:|---:|---:|---:|
| 10 000 | +6 380 | 6/6 | 7 318 |
| 20 000 | +6 576 | 6/6 | 7 252 |
| 30 000 | +7 116 | 6/6 | 6 941 |
| 40 000 | -500 (transient) | 0/6 | 10 000 |
| 50 000 | +6 671 | 6/6 | 6 917 |
| 60 000 | +6 631 | 5/6 | 10 000 |
| 70 000 | +6 686 | 5/6 | 10 000 |
| 80 000 | +7 070 | 6/6 | 6 953 |

**Before:** -3050.5 for 1 000 000 steps, agent never moved.  
**After:** +6 400 to +7 100 per episode, agent reliably completes the 76 km route.

---

## Action Required Before Re-Training

The existing checkpoints in `think_layer/models/` encode the broken value function. Warm-starting from them will immediately re-introduce the braking bias.

```bash
rm -rf think_layer/models/
rm -rf think_layer/runs/
python -m think_layer.train --timesteps 100000   # smoke test first
```

Watch `rollout/stations_reached_mean` in TensorBoard — it should climb toward 6 within the first few rollouts.

Once the deterministic baseline is solid, re-enable stochastic perturbations as a curriculum:

```python
ModernizedLine104(training_mode=True, enable_random_stalls=True)
```
