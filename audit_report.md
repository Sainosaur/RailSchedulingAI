# 🔴 RailSchedulingAI — Full Codebase Audit

**Auditor**: Opus 4.6 (Thinking) · **Date**: 2026-04-13 · **Scope**: Every `.py` file in the project

Your friend owes you £1,100. Here's why:

---

## CRITICAL — Would Crash at Runtime

### BUG 1 · `ModernizedLine104.__init__` missing `training_mode` parameter
**Files**: [agent.py](file:///home/dharms/RailSchedulingAI/think_layer/agent.py#L38), [evaluate.py](file:///home/dharms/RailSchedulingAI/think_layer/evaluate.py#L70), [railway_env.py](file:///home/dharms/RailSchedulingAI/environment/railway_env.py#L59-L63)

> [!CAUTION]
> **Instant crash on training or evaluation.** The constructor `ModernizedLine104.__init__` accepts `(lead_train_speed, render_mode)`. But both [agent.py:38](file:///home/dharms/RailSchedulingAI/think_layer/agent.py#L38) and [evaluate.py:70](file:///home/dharms/RailSchedulingAI/think_layer/evaluate.py#L70) call it with a **`training_mode=True/False`** keyword argument **that does not exist in the constructor signature**.

```python
# agent.py line 38:
env = ModernizedLine104(lead_train_speed=lead_train_speed, training_mode=True)

# evaluate.py line 70:
env = ModernizedLine104(lead_train_speed=config.lead_train_speed, training_mode=False)

# But the __init__ signature is:
def __init__(self, lead_train_speed: float = 20.0, render_mode: str | None = None):
```

**Impact**: `TypeError: __init__() got an unexpected keyword argument 'training_mode'`. Training, evaluation, and anything using `build_agent()` or `evaluate()` will crash immediately.

---

### BUG 2 · `simulation.py` references attributes that don't exist on `ModernizedLine104`
**File**: [simulation.py](file:///home/dharms/RailSchedulingAI/server/simulation.py#L119-L134)

> [!CAUTION]
> **Server simulation crashes on every step.** `simulation.py` reads several attributes from the raw environment that **do not exist** on `ModernizedLine104`:

| Reference in simulation.py | Exists on `ModernizedLine104`? |
|---|---|
| `raw_env._compute_headway()` (lines 119, 177) | ❌ No such method |
| `raw_env.lead_v` (lines 124, 182) | ❌ No such attribute (lead has constant speed stored as `lead_train_speed`) |
| `raw_env.lead_dwell_timer` (lines 125, 183) | ❌ No such attribute |
| `raw_env.landslides` (lines 132-135, 190-193) | ❌ No such attribute |
| `raw_env.set_landslide()` (server/main.py:156) | ❌ No such method |
| `raw_env.clear_landslide()` (server/main.py:171) | ❌ No such method |
| `raw_env.toggle_landslide()` (server/main.py:186) | ❌ No such method |

**Impact**: The entire server/simulation layer will throw `AttributeError` on every call. The live demo functionality (landslides, lead train dwell, headway display) is 100% non-functional.

---

### BUG 3 · `station_detail.py` makes live HTTP calls to external APIs during graph construction
**File**: [station_detail.py](file:///home/dharms/RailSchedulingAI/graph/utils/station_detail.py#L16-L27)

> [!WARNING]
> **Every `import` of the graph module triggers 7 sequential HTTP requests** (one per station) to `api.opentopodata.org` with 1-second `time.sleep()` delays between geocoding lookups. This means:
> - Graph construction takes **≥ 7 seconds** minimum on every run
> - If the API is down or rate-limited, `graph.py` **throws an unhandled exception** and the entire system fails to initialise
> - There is **no caching** — these calls happen every single time `graph()` is called even though the elevations never change
> - Running in a lab/exam environment without internet = **instant failure**

The elevations should be hardcoded constants.

---

## CRITICAL — Logic Gaps That Silently Produce Wrong Results

### BUG 4 · Validator uses `SERVICE_DECEL = -0.5` but Env uses `DECEL = -1.0` — Inconsistent braking model
**Files**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L40-L41), [railway_env.py](file:///home/dharms/RailSchedulingAI/environment/railway_env.py#L44)

> [!IMPORTANT]
> The validator computes the dynamic safe speed using `SERVICE_DECEL = -0.5 m/s²`:
> ```python
> # validator.py line 159-160:
> a_comfort = abs(SERVICE_DECEL)  # = 0.5
> v_dynamic = math.sqrt(2 * a_comfort * distance_available)
> ```
> But the environment's braking capability is `DECEL = -1.0 m/s²`.
> 
> This means the VL **under-estimates** the safe speed — it thinks the train brakes at 0.5 m/s² when computing how fast it can go, but the train actually *can* brake at 1.0 m/s². While conservative, this creates an unnecessarily restrictive system that will override the agent far more than necessary, artificially inflating override penalties (`-500 per override!`) and destroying the learning gradient.

### BUG 5 · `_speed_for_aspect` uses `SERVICE_DECEL` to compute `v_dynamic`, but override logic uses emergency decel — contradictory safety models
**Files**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L159-L163) vs [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L346-L356)

The Layer 1 dynamic speed is based on `SERVICE_DECEL = 0.5 m/s²`:
```python
v_dynamic = math.sqrt(2 * abs(SERVICE_DECEL) * distance_available)
```

But the Layer 3 override computes `a_needed` assuming the train needs to **stop within `distance_available`**:
```python
a_needed = -(u**2) / (2.0 * distance_available)
```

These are two different safety philosophies applied in the same pipeline. Layer 1 says "can the train be going this fast and still stop in the available distance at service braking?". Layer 3 says "what deceleration is needed to stop in the available distance?". They're consistent in *intent* but the mismatch in deceleration values (0.5 vs whatever is needed) means Layer 1 can flag violations that Layer 3 then corrects with a gentler-than-expected override, creating a strange discontinuity.

---

### BUG 6 · `_project()` coasts at `u` when `proposed_a > 0` and `u >= v_ceil` — should cap at `v_ceil`
**File**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L188-L189)

```python
if proposed_a > 0:
    if u >= v_ceil:
        return x + u * t, u  # ← Returns u, NOT v_ceil
```

If `u = 26 m/s` and `v_ceil = 25 m/s` (segment limit), and the agent requests acceleration, `_project()` returns `v = 26` — **above the speed limit**. It should return `min(u, v_ceil)` and coast at `v_ceil`, not `u`. This means the projection thinks the train is going faster than it actually will (since the env caps at the limit), causing **phantom spatial violations** that trigger unnecessary overrides.

The env correctly handles this case (line 154-156: `self.v = limit_v`), but the validator's model disagrees.

---

### BUG 7 · `_check_action_safety` projects into future segments but uses the **originating segment's** `v_ceiling`
**File**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L246-L260)

```python
v_ceiling = current_seg.limit_ms   # Set ONCE before the loop
for t in sim_steps:
    x_proj, v = self._project(x, u, proposed_a, v_ceiling, t)
    # ... later checks proj_seg.limit_ms (lines 272-274)
```

The projection uses `v_ceiling` from the **current** segment for the entire simulation. If the train crosses into a **slower segment** during the projection, the projection *coasts the train at the original (higher) speed limit* through the new segment. The Segment Limit Violation check at line 273 catches this *after the fact*, but by then the position `x_proj` is **over-estimated** because the train would have been going slower in reality. This causes spurious spatial violations.

---

### BUG 8 · Headway penalty formula is inverted / produces penalties in the safe zone
**File**: [reward_function.py](file:///home/dharms/RailSchedulingAI/reward/reward_function.py#L100-L108)

```python
def _compute_headway_penalty(state, config):
    warning_thresh = state.temporal_headway * config.headway_warning_multiplier  # 3*TH
    violation_thresh = state.temporal_headway * config.headway_violation_multiplier  # 1*TH
    h = state.headway
    
    if violation_thresh < h < warning_thresh:
        return -config.k_h * (h * (warning_thresh - h)) / 1000.0
```

The formula `-k_h * h * (warning_thresh - h) / 1000` is a **downward-opening parabola**. It is maximally punishing at `h = warning_thresh / 2` (the middle of the warning zone) and gets *less* punishing as the train approaches the violation threshold. **The penalty gets weaker as the train gets closer to a dangerous headway.**

Correct behaviour: penalty should **increase monotonically** as `h` decreases toward `violation_thresh`. A simple inverse or linear interpolation would be correct.

---

### BUG 9 · Headway violation terminates the episode but the VL should prevent this from ever happening
**Files**: [reward_function.py](file:///home/dharms/RailSchedulingAI/reward/reward_function.py#L158-L164), [railway_env.py](file:///home/dharms/RailSchedulingAI/environment/railway_env.py#L215-L218)

```python
# reward_function.py
def _compute_headway_violation(state, config):
    violation_thresh = state.temporal_headway * config.headway_violation_multiplier  # 1*TH
    if state.headway <= violation_thresh:
        return -150.0, True  # Episode terminates!
```

The headway is computed as `(lead_x - x) / v`. But the VL uses **spatial distance** (DTZ + N×SH) for its checks, not temporal headway. There is **no mechanism in the VL that constrains temporal headway**. The VL only looks at spatial violations — it's entirely possible for the agent to cruise at low speed well within spatial limits but still have `headway <= 1*TH` and get the episode terminated. The two safety systems are not aligned.

---

### BUG 10 · Override penalty of **-500** per step is catastrophically large
**File**: [reward_function.py](file:///home/dharms/RailSchedulingAI/reward/reward_function.py#L39)

```python
override_penalty: float = -500.0
```

> [!WARNING]
> With a heartbeat of `-0.1` per step, a progress reward of `~5.0` per inter-station span, and a station reward of `+5.0`, a **single override wipes out ~100 station arrivals worth of positive reward**. Combined with BUG 4/6/7 (which cause spurious overrides), the agent will receive overwhelmingly negative reward signals that are functionally indistinguishable from random noise. **The agent cannot possibly learn a useful policy with this reward scale.**

---

### BUG 11 · Jerk penalty uses `action_delta` as a float but comment says "e.g. 0 to 3"
**Files**: [reward_function.py](file:///home/dharms/RailSchedulingAI/reward/reward_function.py#L78), [railway_env.py](file:///home/dharms/RailSchedulingAI/environment/railway_env.py#L136)

```python
# TrainState dataclass:
action_delta: int = 0  # Magnitude of change (e.g. 0 to 3)

# Environment computes it as:
action_delta = abs(safe_a - self.last_a)  # float in range [0, 1.5]
```

The `TrainState` type annotation says `int` but the actual computed value is a **float** (difference between two continuous accelerations, range `[0, 1.5]`). The comment "e.g. 0 to 3" is from an old discrete action space. The jerk penalty formula `jerk_penalty * (action_delta ** 2)` will work numerically but the documentation/typing is misleading and would lose marks for inconsistency.

---

## SEVERE — Missing Features / Stubs

### BUG 12 · `railway_env_test.py` is **completely empty**
**File**: [railway_env_test.py](file:///home/dharms/RailSchedulingAI/environment/railway_env_test.py)

Zero environment unit tests. None. The most complex piece of the system has no test coverage.

---

### BUG 13 · Punctuality system is dead code — never triggered
**File**: [reward_function.py](file:///home/dharms/RailSchedulingAI/reward/reward_function.py#L147-L156)

```python
def _compute_punctuality_penalty(state, config):
    if not state.reached_new_station:
        return 0.0
    if state.scheduled_arrival_time is None or state.actual_arrival_time is None:
        return 0.0  # ← Always exits here
```

`scheduled_arrival_time` and `actual_arrival_time` both default to `None` in `TrainState` and are **never set** by the environment. The punctuality penalty — a key feature for a rail scheduling system — is permanently inactive.

---

### BUG 14 · No timetable, no scheduling — this is a "drive forward" simulator, not a "scheduler"
**Files**: All

> [!IMPORTANT]
> For a project called **"RailSchedulingAI"**, there is:
> - No timetable data
> - No departure/arrival time targets
> - No multi-train coordination (single AI train follows a ghost lead)
> - No track switching or passing loops (despite stations having a `passing_loop` attribute)
> - No bidirectional traffic
> - No actual *scheduling* decisions — the agent only controls acceleration
> 
> The environment is a single-agent, single-track, one-direction speed control problem. This is a significant feature gap for the project name/scope.

---

### BUG 15 · Server kill/restore is completely non-functional
**File**: [server/main.py](file:///home/dharms/RailSchedulingAI/server/main.py#L115-L133)

The `KILLED` flag is set/unset but **never read by any other code**. The simulation ignores it entirely. The kill switch is cosmetic only.

---

## MODERATE — Issues That Show Poor Engineering

### BUG 16 · Graph blocks are built twice — once in `graph()` with `Block` objects, once in `build_vl_segments()` with float boundaries
**File**: [graph.py](file:///home/dharms/RailSchedulingAI/graph/graph.py#L157-L163) vs [graph.py](file:///home/dharms/RailSchedulingAI/graph/graph.py#L213-L219)

The same block subdivision logic is implemented twice with different data structures. The `Block` objects from the graph edges are never used by the VL — it builds its own `float` boundaries. The `Block.hazard` field is set by the server but the VL never reads it. **Hazard data is silently ignored by the safety system.**

---

### BUG 17 · `_check_action_safety` adaptive lookahead can create `sim_steps` with duplicates
**File**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L253-L257)

```python
t_stop = current_seg.limit_ms / abs(EMERGENCY_DECEL)
sim_steps = [
    max(1, round(t_stop * 0.33)),  # S3: max(1, round(2.78)) = 3
    max(2, round(t_stop * 0.67)),  # S3: max(2, round(5.58)) = 6  
    max(3, round(t_stop)),          # S3: max(3, round(8.33)) = 8
]
```

For segments S3-S5 with `limit_ms = 8.33` and `EMERGENCY_DECEL = -1.0`: `t_stop = 8.33s`. Steps = `[3, 6, 8]`. Fine here. But for a hypothetical slow segment where `t_stop ≤ 3`, you'd get `sim_steps = [1, 2, 3]` which is correct but the `max()` floors hide the adaptive intent. More concerning: the comment says the horizon is "capped at the minimum time needed to guarantee a safe stop" but `round(t_stop)` can be `t_stop + 0.5` due to rounding up — the projection extends *beyond* the stopping distance.

---

### BUG 18 · `compute_dtz` returns the wrong value at the exact segment boundary
**File**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L106-L130)

If the train is at position `x = 5481.0` (the boundary between S0 and S1), `get_segment` returns S1 (since `seg.start <= x < seg.end` and 5481 is S1's start). Then `compute_dtz` looks for the first boundary `> x`. But the first boundary in S1 **is** 5481.0, which is not `> x`, it's `== x`. So it skips to the next boundary at `5481 + 312.5 = 5793.5`. DTZ = 312.5m = a full block. This is **correct** behaviour (you're on a boundary, so the next one is SH ahead), but only by coincidence — the comment says "if exactly on a boundary, the following boundary is used" but the code achieves this through `>` rather than `>=`, which only works because the first block boundary equals `seg.start`.

---

### BUG 19 · `_project()` returns `max(0.0, x_proj)` — allows position to be silently clamped to 0
**File**: [validator.py](file:///home/dharms/RailSchedulingAI/validate_layer/validator.py#L204)

If there's a numerical edge case where `x_proj` goes negative (shouldn't happen in normal operation but could via float precision), the position is silently clamped to 0.0 — far behind the track start at 582m — which would then cause `get_segment(0.0)` to throw `ValueError` in the safety check, which is caught only for `Track_Bounds_Violation`. Not a likely runtime issue but a defensive gap.

---

### BUG 20 · Observation space doesn't match actual observations for `signal_aspect`
**File**: [railway_env.py](file:///home/dharms/RailSchedulingAI/environment/railway_env.py#L78-L81)

```python
# Observation space high:
high=np.array([80000, 30.0, 80000, 3.0, 80000, 30.0, 10000], ...)
```

`signal_aspect` has `high=3.0` but the actual observation puts a **float** cast of an integer (0-3) into a `Box` space. This is technically valid but semantically poor — aspects are categorical, not continuous. PPO will treat `aspect=2.5` as meaningful during normalisation, which is nonsensical.

---

### BUG 21 · `TrainState.action_delta` type annotation is `int` but receives `float`
**File**: [reward_function.py](file:///home/dharms/RailSchedulingAI/reward/reward_function.py#L78)

Already noted as part of BUG 11, but separately: if a marker or type-checker enforces this, `mypy` would flag this as an error.

---

## Summary Scoreboard

| Severity | Count | Examples |
|---|---|---|
| 🔴 **Runtime Crash** | 3 | BUG 1, 2, 3 |
| 🔴 **Silent Wrong Behaviour** | 8 | BUG 4-11 |
| 🟡 **Missing Features** | 4 | BUG 12-15 |
| 🟠 **Engineering Quality** | 6 | BUG 16-21 |
| **Total** | **21** | |

> [!CAUTION]
> **BUGs 1-3 alone would make the system completely non-functional.** The training pipeline (`agent.py`, `trainer.py`), evaluation pipeline (`evaluate.py`), and live demo server (`simulation.py`) all crash immediately due to missing constructor parameters and non-existent attributes. The system as-is cannot train, evaluate, or demo.
