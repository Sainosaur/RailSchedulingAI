"""
think_layer/callbacks.py

Reward-component tracking callback. The environment publishes a
per-step `info["reward_breakdown"]` dict; this callback accumulates the
components across each episode and logs the episode totals to TensorBoard
under `rollout/r_<component>`, plus a `rollout/stations_reached` side
metric that reports true task progress (the main reward is a proxy).
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class RewardBreakdownCallback(BaseCallback):
    """Log summed reward components and stations reached per episode."""

    def __init__(self, verbose: int = 0) -> None:
        super().__init__(verbose)
        self._running: list[dict[str, float]] = []
        self._stations: list[int] = []
        self._last_stations_flag: list[bool] = []
        self._done_buffer: dict[str, list[float]] = defaultdict(list)
        self._stations_buffer: list[int] = []

    def _init_callback(self) -> None:
        n = self.training_env.num_envs
        self._running = [defaultdict(float) for _ in range(n)]
        self._stations = [0] * n

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [False] * len(infos))
        for i, info in enumerate(infos):
            breakdown: dict[str, float] | None = info.get("reward_breakdown")
            if breakdown:
                for k, v in breakdown.items():
                    self._running[i][k] += float(v)
                if breakdown.get("station", 0.0) > 0.0:
                    self._stations[i] += 1

            if dones[i]:
                for k, v in self._running[i].items():
                    self._done_buffer[k].append(v)
                self._stations_buffer.append(self._stations[i])
                self._running[i] = defaultdict(float)
                self._stations[i] = 0
        return True

    def _on_rollout_end(self) -> None:
        # Flush the buffered episode stats to TensorBoard.
        if not self._stations_buffer:
            return
        for k, vals in self._done_buffer.items():
            if vals:
                self.logger.record(f"rollout/r_{k}_mean", float(np.mean(vals)))
        self.logger.record(
            "rollout/stations_reached_mean",
            float(np.mean(self._stations_buffer)),
        )
        self.logger.record(
            "rollout/episodes_finished",
            int(len(self._stations_buffer)),
        )
        self._done_buffer.clear()
        self._stations_buffer.clear()
