"""
reward/reward_function.py
Railway RL Reward Function — Simplified

Design principles:
  1. Dense progress signal dominates: moving > stationary, always
  2. Outcome-based rewards, not input-based (reward speed, not exact accel value)
  3. Minimal components to reduce reward-hacking surface
  4. Emergency braking: single flat penalty regardless of aspect
  5. Headway sweet-spot: bonus for staying in 3-4 SH bracket
  6. Slight exponential station milestones (progressive rewards)
  7. No patience/dwell rewards — agent doesn't gain from sitting
"""

from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any


@dataclass
class TrainState:
    """All variables required to compute the reward at one timestep."""
    current_position: float
    previous_position: float
    last_station_position: float
    next_station_position: float
    current_speed: float
    speed_limit: float
    headway: float  # temporal headway in seconds
    temporal_headway: float
    spatial_headway: float
    reached_new_station: bool
    scheduled_arrival_time: Optional[float] = None
    actual_arrival_time: Optional[float] = None
    collision: bool = False
    safety_overridden: bool = False
    action_delta: float = 0.0
    applied_traction: float = 0.0
    distance_to_occupied: float = 99999.0
    is_dwelling: bool = False
    station_index: int = 0
    signal_aspect: int = 3
    applied_acceleration: float = 0.0
    proposed_acceleration: float = 0.0
    current_time: float = 0.0
    next_scheduled_arrival_time: float = 0.0


@dataclass
class RewardOutput:
    """Structured reward breakdown for debugging."""
    r_progress: float = 0.0
    r_overspeed: float = 0.0
    r_headway: float = 0.0
    r_station: float = 0.0
    r_punctuality: float = 0.0
    r_jerk: float = 0.0
    r_shaping: float = 0.0  # PBRS shaping term: γφ(s') - φ(s), applied in railway_env
    r_total: float = 0.0

    def to_dict(self):
        return asdict(self)


# ── Potential-Based Reward Shaping ───────────────────────────────────────────
# γ must match PPO's gamma in config.py exactly.
PBRS_GAMMA = 0.999  # matches gamma in config.py


def potential(state: TrainState) -> float:
    """
    Potential function φ(s) for PBRS shaping.
    Shaping bonus each step: PBRS_GAMMA * φ(s') - φ(s)

    Simplified to 2 components: progress + speed.
    Both bounded, smooth, monotonic — no sharp transitions.
    """

    # ── Component 1: Segment progress ────────────────────────────────────────
    # How far through current inter-station segment.
    # Bounded [0.0, 5.0].
    total_span = max(state.next_station_position - state.last_station_position, 1.0)
    progress_ratio = (state.current_position - state.last_station_position) / total_span
    progress_ratio = max(0.0, min(1.0, progress_ratio))
    phi_progress = progress_ratio * 5.0

    # ── Component 2: Speed profile adherence ─────────────────────────────────
    # Being at speed limit = 3.0, stopped = 0.0.
    # Bounded [0.0, 3.0].
    speed_ratio = min(state.current_speed / max(state.speed_limit, 1.0), 1.0)
    phi_speed = speed_ratio * 3.0

    return phi_progress + phi_speed


def compute_reward(
    state: TrainState,
    info: Dict[str, Any],
    weights: Optional[Dict[str, float]] = None,
    detailed_logs: bool = False,
) -> RewardOutput:
    # Extract info
    train_aspect = info.get("train_aspect", 3)
    station_aspect = info.get("station_aspect", 3)
    x_lead_zone_start = info.get("x_lead_zone_start", 99999.0)
    violations = info.get("violations", {})
    lead_train_stalled = info.get("lead_train_stalled", False)
    lead_train_held = info.get("lead_train_held", False)

    # Resolve weights — default to 1.0 for all components if not provided
    _w = weights or {}
    w_progress     = _w.get("progress",     1.0)
    w_overspeed    = _w.get("overspeed",    1.0)
    w_headway      = _w.get("headway",      1.0)
    w_station      = _w.get("station",      1.0)
    w_punctuality  = _w.get("punctuality",  1.0)
    w_jerk         = _w.get("jerk",         1.0)

    sh = state.spatial_headway
    dt = 1.0
    u = state.current_speed
    a = state.applied_acceleration
    x_ai_zone_end = info.get("x_ai_zone_end", state.current_position)
    x_diff = x_lead_zone_start - x_ai_zone_end  # gap between AI and lead train

    out = RewardOutput()

    # ═══════════════════════════════════════════════════════════════════════════
    # 1. PROGRESS — dense, every step
    # ═══════════════════════════════════════════════════════════════════════════
    # Reward = speed / max_line_speed, capped at 1.0
    # Stopped → 0.0 (this IS the penalty for not moving, no separate step cost)
    # At speed limit → ~1.0
    # Budget: ~1.0/step × 9500 moving steps = ~9,500 over perfect episode
    MAX_LINE_SPEED_MS = 25.0  # 90 km/h — fastest segment on line 104
    if u > 0.01:
        out.r_progress = min(1.0, u / MAX_LINE_SPEED_MS)
    else:
        out.r_progress = 0.0

    # Suspend progress reward if lead train stalled/held (not AI's fault)
    if lead_train_stalled or lead_train_held:
        out.r_progress = max(0.0, out.r_progress)

    # ═══════════════════════════════════════════════════════════════════════════
    # 2. OVERSPEED & EMERGENCY BRAKING — penalty only
    # ═══════════════════════════════════════════════════════════════════════════
    # Two violations: exceeding speed limit, or using emergency braking
    # Flat penalties regardless of signal aspect — simplest possible
    # Budget: ~-500 total (if agent hits ~100 violations per episode)

    # Emergency braking penalty (a <= -1.0 is emergency decel)
    if a <= -1.0:
        out.r_overspeed = -5.0

    # Overspeed penalty (above segment speed limit)
    elif u > state.speed_limit + 0.5:
        overspeed_amount = u - state.speed_limit
        out.r_overspeed = max(-5.0, -overspeed_amount * 2.0)

    # Smooth approach bonus: reward smooth service braking when approaching
    # station or red signal. Deceleration in [-0.6, -0.3] range = service braking.
    # Agent discovers optimal ~0.5 m/s/s naturally.
    elif station_aspect == 0 and -0.6 <= a <= -0.3 and u > 0.5:
        out.r_overspeed = 1.0  # mild bonus for smooth approach

    # Red signal: must be braking or stopped
    elif station_aspect == 0 and u > 0.1 and a > -0.1:
        out.r_overspeed = -3.0  # moving at red without braking

    elif train_aspect == 0 and u > 0.1 and a > -0.1:
        out.r_overspeed = -3.0  # moving at red lead signal without braking

    # ═══════════════════════════════════════════════════════════════════════════
    # 3. HEADWAY SWEET-SPOT — bonus for staying in bracket
    # ═══════════════════════════════════════════════════════════════════════════
    # Reward staying 3-4 SH behind lead train (max throughput + safety)
    # Only when moving (no bonus for sitting in the sweet spot stopped)
    # Budget: +0.5/step × ~3000 steps in bracket = +1,500
    if u > 0.5 and sh > 0:
        if 3 * sh <= x_diff <= 4 * sh:
            out.r_headway = 0.5   # in sweet spot — bonus
        # No penalty for being outside — just no bonus

    # ═══════════════════════════════════════════════════════════════════════════
    # 4. STATION MILESTONES — sparse, progressive
    # ═══════════════════════════════════════════════════════════════════════════
    # Slight exponential weighting — later stations worth more
    # Total budget: ~3,500 across 6 stations
    STATION_REWARDS = {
        1: 200.0,    # Rabka-Zdrój
        2: 300.0,    # Mszana Dolna
        3: 400.0,    # Tymbark
        4: 500.0,    # Limanowa
        5: 600.0,    # Marcinkowice
        6: 1500.0,   # Nowy Sącz (terminus)
    }
    if state.reached_new_station:
        out.r_station = STATION_REWARDS.get(state.station_index, 0.0)

    # ═══════════════════════════════════════════════════════════════════════════
    # 5. PUNCTUALITY — sparse, on arrival
    # ═══════════════════════════════════════════════════════════════════════════
    # Budget: +200/station × 6 = +1,200 total for on-time arrivals
    if state.reached_new_station:
        if state.scheduled_arrival_time is not None and state.actual_arrival_time is not None:
            deviation = abs(state.scheduled_arrival_time - state.actual_arrival_time)
            if deviation <= 60.0:
                out.r_punctuality = 200.0
            else:
                out.r_punctuality = -100.0

    # ═══════════════════════════════════════════════════════════════════════════
    # 6. JERK — penalty only, smooth control is baseline expectation
    # ═══════════════════════════════════════════════════════════════════════════
    # Budget: ~-250 total (if agent has ~500 jerk events)
    jerk = abs(state.applied_acceleration - info.get("previous_a", 0.0)) / dt
    if jerk > 1.0:
        out.r_jerk = -0.5

    # ═══════════════════════════════════════════════════════════════════════════
    # SUM
    # ═══════════════════════════════════════════════════════════════════════════
    out.r_total = (
        w_progress    * out.r_progress +
        w_overspeed   * out.r_overspeed +
        w_headway     * out.r_headway +
        w_station     * out.r_station +
        w_punctuality * out.r_punctuality +
        w_jerk        * out.r_jerk
    )

    # ── Speed Mode ────────────────────────────────────────────────────────────
    if detailed_logs:
        info["reward_breakdown"] = out.to_dict()

    return out
"""
BUDGET SUMMARY (perfect episode ~10,000 steps):
  r_progress:    +1.0/step × 9500 steps  = +9,500   (dominant signal)
  r_overspeed:   -5.0/event × ~50 events = -250     (penalty only)
  r_headway:     +0.5/step × ~3000 steps = +1,500   (bonus for bracket)
  r_station:     progressive totaling     = +3,500   (sparse milestones)
  r_punctuality: +200 × 6 stations        = +1,200   (sparse on-time)
  r_jerk:        -0.5/event × ~500 events = -250     (penalty only)
  PBRS shaping:  net ~0 over episode      = ~0       (dense gradient)
  ──────────────────────────────────────────────────
  TOTAL PERFECT EPISODE                   ≈ +15,200
  TOTAL STATIONARY EPISODE (never moves)  ≈ 0        (no exploit)
"""