import asyncio
import sys
from pathlib import Path
import numpy as np
# Ensure the environment can be imported correctly regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from environment.railway_env import ModernizedLine104
from environment.timetable import STATION_NAMES


class SimulationRunner:
    """
    Runs the trained PPO agent asynchronously in a background loop,
    broadcasting the state of the env over WebSocket via a callback.
    """

    def __init__(self, broadcast_callback=None):
        self.broadcast_callback = broadcast_callback
        self.model = None
        self.venv = None

        self.is_running = False
        self.steps_per_second = 10.0  # 10x simulation speed by default
        self._task = None
        self._current_obs = None

    def load_model(self, lead_train_speed=25.0):
        print(f"Loading simulation model (Lead Speed = {lead_train_speed} m/s)")

        # 1. Thunk to create the bare environment
        def make_env():
            return ModernizedLine104(lead_train_speed=lead_train_speed)

        # 2. Vectorise
        self.venv = DummyVecEnv([make_env])

        # 3. Load Normalisation Stats
        vecnorm_path = PROJECT_ROOT / "think_layer" / "models" / "final_vecnormalize.pkl"
        if vecnorm_path.exists():
            # Workaround for numpy 2.x pickle loaded in numpy 1.x (and vice versa)
            import sys
            import importlib
            import numpy

            # 1. Defensive imports of internal modules to satisfy Pyright/static analysis
            try:
                import numpy.core.multiarray as ncm  # type: ignore
                import numpy.core.numeric as ncn  # type: ignore
                # Use importlib to avoid direct static reference to internal module
                nr_pickle = importlib.import_module("numpy.random._pickle")
            except ImportError:
                # Fallback if names differ or modules are moved
                ncm = getattr(getattr(numpy, "core", None), "multiarray", None)
                ncn = getattr(getattr(numpy, "core", None), "numeric", None)
                nr_pickle = getattr(numpy.random, "_pickle", None)

            # 2. Core aliases (numpy 2.x renamed core to _core)
            if hasattr(numpy, "core"):
                sys.modules.setdefault("numpy._core", numpy.core)
                if ncn:
                    sys.modules.setdefault("numpy._core.numeric", ncn)
                if ncm:
                    sys.modules.setdefault("numpy._core.multiarray", ncm)

            # 3. Monkeypatch __bit_generator_ctor to handle BitGenerator classes
            if nr_pickle and not hasattr(nr_pickle, "_patched"):
                orig_ctor = getattr(nr_pickle, "__bit_generator_ctor", None)
                if orig_ctor:

                    def patched_ctor(bit_generator_name):
                        if not isinstance(bit_generator_name, str):
                            if hasattr(bit_generator_name, "__name__"):
                                bit_generator_name = bit_generator_name.__name__
                        return orig_ctor(bit_generator_name)

                    setattr(nr_pickle, "__bit_generator_ctor", patched_ctor)
                    setattr(nr_pickle, "_patched", True)

            self.venv = VecNormalize.load(str(vecnorm_path), self.venv)
            self.venv.training = False
            self.venv.norm_reward = False
        else:
            print(
                f"WARNING: VecNormalize not found at {vecnorm_path}. Running without norm."
            )

        # 4. Load PPO Checkpoint
        model_path = PROJECT_ROOT / "think_layer" / "models" / "final_model.zip"
        if model_path.exists():
            self.model = PPO.load(str(model_path), env=self.venv, device="cpu")
        else:
            print(
                f"ERROR: Model not found at {model_path}. You need to train it first!"
            )

        self._current_obs = self.venv.reset()

    async def start(self):
        """Starts the background simulation loop."""
        if self.is_running:
            return

        if self.model is None or self.venv is None:
            self.load_model()

        self.is_running = True
        self._task = asyncio.create_task(self._sim_loop())

    async def stop(self):
        """Pauses the simulation."""
        self.is_running = False
        if self._task and not self._task.done():
            await self._task
        self._task = None

    async def reset(self):
        """Resets the environment back to the start."""
        if self.venv:
            self._current_obs = self.venv.reset()
            # Broadcast the initial state
            await self._broadcast_step(force_done=False)

    def set_speed(self, steps_per_second: float):
        """Updates playback speed."""
        self.steps_per_second = max(0.1, steps_per_second)

    async def _broadcast_step(self, force_done=False):
        """Extracts the state dict and pushes it to the websocket callback."""
        if not self.broadcast_callback or self._current_obs is None:
            return

        # Vectorised environments return arrays. Get index 0.
        # VecNormalize hides the raw properties. So we have to unwrap it to access lead_x etc.
        raw_env = self.venv.envs[0]
        while hasattr(raw_env, "env"):
            raw_env = raw_env.env

        # We step the environment elsewhere. This just reads the current state.
        # To get the info dict, we actually need to capture it during `venv.step()`.
        # However, for `ModernizedLine104`, all positional variables are accessible direct from the object.

        # Read direct from raw_env for UN-NORMALISED raw stats
        segment = raw_env.vl.get_segment(raw_env.x)
        progress = (raw_env.x - segment.start) / (segment.end - segment.start)
        next_st_idx = min(raw_env.last_station_idx + 1, len(raw_env.STATIONS) - 1)
        next_st_pos = raw_env.STATIONS[next_st_idx]

        signal_aspect = raw_env._get_signal_aspect()
        if signal_aspect == 3:
            signal = "green"
        elif signal_aspect == 2:
            signal = "double-amber"
        elif signal_aspect == 1:
            signal = "amber"
        else:
            signal = "red"

        lead_segment = raw_env.vl.get_segment(
            min(raw_env.lead_train.x, raw_env.TRACK_END)
        )
        lead_progress = (raw_env.lead_train.x - lead_segment.start) / (
            lead_segment.end - lead_segment.start
        )

        state = {
            "type": "sim_update",
            "step": raw_env.step_count,
            "time": raw_env.time,
            "ai": {
                "position_m": float(raw_env.x),
                "progress": float(progress),
                "speed_ms": float(raw_env.v),
                "speed_kmh": float(raw_env.v) * 3.6,
                "acceleration": float(raw_env.last_a),
                "dtz": float(raw_env.dtz),
                "signal": signal,
                "dist_to_next_station": float(next_st_pos - raw_env.x),
                "dist_to_obstruction": float(raw_env._dist_to_nearest_occupied()),
                "speed_limit_ms": float(segment.limit_ms),
                "headway": float(
                    (raw_env.lead_train.x - raw_env.x) / raw_env.v
                    if raw_env.v > 0.01
                    else 9999.0
                ),
                "segment_id": segment.id,
                "approaching_station": STATION_NAMES[next_st_idx],
                "dwell_timer": int(raw_env.ai_dwell_timer),
                "authority_ranges": {
                    "red": [0.0, float(segment.spatial_headway)],
                    "yellow": [
                        float(segment.spatial_headway),
                        float(2 * segment.spatial_headway),
                    ],
                    "double_yellow": [
                        float(2 * segment.spatial_headway),
                        float(3 * segment.spatial_headway),
                    ],
                    "green": [float(3 * segment.spatial_headway), 9999.9],
                },
            },
            "lead": {
                "position_m": float(raw_env.lead_train.x),
                "progress": float(lead_progress),
                "speed_ms": float(raw_env.lead_train.v),
                "speed_kmh": float(raw_env.lead_train.v) * 3.6,
                "signal": "green",
                "segment": lead_segment.id,
                "dwell_timer": int(raw_env.lead_train.dwell_timer),
                "stalled": raw_env.lead_train.stalled,
                "held": raw_env.lead_train.held,
            },
            "override": {
                "active": False,
                "proposed_a": 0.0,
                "safe_a": 0.0,
            },
            "stations_visited": list(raw_env.visited_stations),
            "hazards": [{"start": s, "end": e} for s, e in raw_env.active_hazards],
            "done": bool(force_done),
            "timetable": raw_env.timetable.to_dict(),
            "punctuality": raw_env.get_punctuality_status(),
            "reward": {
                "total": 0.0,
                "breakdown": {},
            },
        }

        await self.broadcast_callback(state)

    async def _sim_loop(self):
        """The core async loop pacing the RL environment steps."""
        try:
            while self.is_running:
                if self._current_obs is None:
                    self._current_obs = self.venv.reset()

                # SB3 inference
                if self.model is not None:
                    action, _ = self.model.predict(
                        self._current_obs, deterministic=True
                    )
                else:
                    # Fallback: Just coast (0 acceleration) if no model exists
                    action = [np.array([0.0], dtype=np.float32)]

                self._current_obs, rewards, dones, infos = self.venv.step(action)

                info = infos[0]
                done = dones[0]

                # Reconstruct the payload to include dynamic info from the step step
                raw_env = self.venv.envs[0]
                while hasattr(raw_env, "env"):
                    raw_env = raw_env.env

                segment = raw_env.vl.get_segment(raw_env.x)
                progress = (raw_env.x - segment.start) / (segment.end - segment.start)
                next_st_idx = min(
                    raw_env.last_station_idx + 1, len(raw_env.STATIONS) - 1
                )
                next_st_pos = raw_env.STATIONS[next_st_idx]

                signal_aspect = info.get("aspect", raw_env._get_signal_aspect())
                if signal_aspect == 3:
                    signal = "green"
                elif signal_aspect == 2:
                    signal = "double-amber"
                elif signal_aspect == 1:
                    signal = "amber"
                else:
                    signal = "red"

                lead_segment = raw_env.vl.get_segment(
                    min(raw_env.lead_train.x, raw_env.TRACK_END)
                )
                lead_progress = (raw_env.lead_train.x - lead_segment.start) / (
                    lead_segment.end - lead_segment.start
                )

                state = {
                    "type": "sim_update",
                    "step": raw_env.step_count,
                    "time": info.get("time", 0.0),
                    "ai": {
                        "position_m": float(raw_env.x),
                        "progress": float(progress),
                        "speed_ms": float(raw_env.v),
                        "speed_kmh": float(raw_env.v) * 3.6,
                        "acceleration": float(raw_env.last_a),
                        "dtz": float(raw_env.dtz),
                        "signal": signal,
                        "dist_to_next_station": float(next_st_pos - raw_env.x),
                        "dist_to_obstruction": float(
                            raw_env._dist_to_nearest_occupied()
                        ),
                        "speed_limit_ms": float(segment.limit_ms),
                        "headway": float(
                            (raw_env.lead_train.x - raw_env.x) / raw_env.v
                            if raw_env.v > 0.01
                            else 9999.0
                        ),
                        "segment_id": info.get("segment", segment.id),
                        "approaching_station": STATION_NAMES[next_st_idx],
                        "dwell_timer": int(raw_env.ai_dwell_timer),
                        "authority_ranges": {
                            "red": [0.0, float(segment.spatial_headway)],
                            "yellow": [
                                float(segment.spatial_headway),
                                float(2 * segment.spatial_headway),
                            ],
                            "double_yellow": [
                                float(2 * segment.spatial_headway),
                                float(3 * segment.spatial_headway),
                            ],
                            "green": [float(3 * segment.spatial_headway), 9999.9],
                        },
                    },
                    "lead": {
                        "position_m": float(raw_env.lead_train.x),
                        "progress": float(lead_progress),
                        "speed_ms": float(raw_env.lead_train.v),
                        "speed_kmh": float(raw_env.lead_train.v) * 3.6,
                        "signal": "green",
                        "segment": lead_segment.id,
                        "dwell_timer": int(raw_env.lead_train.dwell_timer),
                        "stalled": raw_env.lead_train.stalled,
                        "held": raw_env.lead_train.held,
                    },
                    "override": {
                        "active": info.get("overridden", False),
                        "proposed_a": float(info.get("proposed_a", 0.0)),
                        "safe_a": float(info.get("safe_a", 0.0)),
                    },
                    "reward": {
                        "total": float(rewards[0]),
                        "breakdown": info.get("reward_breakdown", {}),
                    },
                    "stations_visited": list(raw_env.visited_stations),
                    "hazards": [
                        {"start": s, "end": e} for s, e in raw_env.active_hazards
                    ],
                    "done": bool(done),
                    "timetable": raw_env.timetable.to_dict(),
                    "punctuality": raw_env.get_punctuality_status(),
                }

                if self.broadcast_callback:
                    try:
                        await self.broadcast_callback(state)
                    except Exception as e:
                        print(f"Broadcast failed: {e}")

                if done:
                    self.is_running = False  # Auto pause when episode is finished

                # Throttle the simulation
                await asyncio.sleep(1.0 / self.steps_per_second)
        except Exception as e:
            print(f"SIMULATION ABORTED DUE TO ERROR: {e}")
            import traceback

            traceback.print_exc()
            self.is_running = False
