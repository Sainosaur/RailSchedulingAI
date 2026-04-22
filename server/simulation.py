import sys
import asyncio
from pathlib import Path

# Ensure the environment can be imported correctly regardless of cwd
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.railway_env import ModernizedLine104


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

    def load_model(self, lead_train_speed=20.0):
        print(f"Loading simulation model (Lead Speed = {lead_train_speed} m/s)")
        
        # 1. Thunk to create the bare environment
        def make_env():
            return ModernizedLine104(lead_train_speed=lead_train_speed)
        
        # 2. Vectorise
        self.venv = DummyVecEnv([make_env])
        
        # 3. Load Normalisation Stats
        vecnorm_path = PROJECT_ROOT / "think_layer" / "models" / "best_vecnormalize.pkl"
        if vecnorm_path.exists():
            self.venv = VecNormalize.load(str(vecnorm_path), self.venv)
            self.venv.training = False     
            self.venv.norm_reward = False  
        else:
            print(f"WARNING: VecNormalize not found at {vecnorm_path}. Running without norm.")
            
        # 4. Load PPO Checkpoint
        model_path = PROJECT_ROOT / "think_layer" / "models" / "best_model.zip"
        if model_path.exists():
            self.model = PPO.load(str(model_path))
        else:
            print(f"ERROR: Model not found at {model_path}. You need to train it first!")
            
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
            
        lead_segment = raw_env.vl.get_segment(min(raw_env.lead_x, raw_env.TRACK_END))
        lead_progress = (raw_env.lead_x - lead_segment.start) / (lead_segment.end - lead_segment.start)
        
        state = {
            "type": "sim_update",
            "step": raw_env.step_count,
            "time": raw_env.time,
            "ai": {
                "position_m": float(raw_env.x),
                "progress": float(progress),
                "speed_ms": float(raw_env.v),
                "speed_kmh": float(raw_env.v) * 3.6,
                "dtz": float(raw_env.dtz),
                "signal": signal,
                "dist_to_next_station": float(next_st_pos - raw_env.x),
                "dist_to_obstruction": float(raw_env._dist_to_nearest_occupied()),
                "speed_limit_ms": float(segment.limit_ms),
                "headway": float((raw_env.lead_x - raw_env.x) / raw_env.v if raw_env.v > 0.01 else 9999.0),
                "segment": segment.id,
                "dwell_timer": int(raw_env.ai_dwell_timer),
            },
            "lead": {
                "position_m": float(raw_env.lead_x),
                "progress": float(lead_progress),
                "speed_ms": float(raw_env.lead_v),
                "speed_kmh": float(raw_env.lead_v) * 3.6,
                "signal": "green",
                "segment": lead_segment.id,
                "dwell_timer": int(raw_env.lead_dwell_timer),
                "stalled": raw_env.lead_stalled,
                "held": raw_env.lead_held,
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
        }
        
        await self.broadcast_callback(state)

    async def _sim_loop(self):
        """The core async loop pacing the RL environment steps."""
        try:
            while self.is_running:
                if self._current_obs is None:
                    self._current_obs = self.venv.reset()
                    
                # SB3 inference
                action, _ = self.model.predict(self._current_obs, deterministic=True)
                self._current_obs, rewards, dones, infos = self.venv.step(action)
                
                info = infos[0]
                done = dones[0]
                
                # Reconstruct the payload to include dynamic info from the step step
                raw_env = self.venv.envs[0]
                while hasattr(raw_env, "env"):
                    raw_env = raw_env.env
                    
                segment = raw_env.vl.get_segment(raw_env.x)
                progress = (raw_env.x - segment.start) / (segment.end - segment.start)
                next_st_idx = min(raw_env.last_station_idx + 1, len(raw_env.STATIONS) - 1)
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
                
                lead_segment = raw_env.vl.get_segment(min(raw_env.lead_x, raw_env.TRACK_END))
                lead_progress = (raw_env.lead_x - lead_segment.start) / (lead_segment.end - lead_segment.start)
                
                state = {
                    "type": "sim_update",
                    "step": raw_env.step_count,
                    "time": info.get("time", 0.0),
                    "ai": {
                        "position_m": float(raw_env.x),
                        "progress": float(progress),
                        "speed_ms": float(raw_env.v),
                        "speed_kmh": float(raw_env.v) * 3.6,
                        "dtz": float(raw_env.dtz),
                        "signal": signal,
                        "dist_to_next_station": float(next_st_pos - raw_env.x),
                        "dist_to_obstruction": float(raw_env._dist_to_nearest_occupied()),
                        "speed_limit_ms": float(segment.limit_ms),
                        "headway": float((raw_env.lead_x - raw_env.x) / raw_env.v if raw_env.v > 0.01 else 9999.0),
                        "segment": info.get("segment", segment.id),
                        "dwell_timer": int(raw_env.ai_dwell_timer),
                    },
                    "lead": {
                        "position_m": float(raw_env.lead_x),
                        "progress": float(lead_progress),
                        "speed_ms": float(raw_env.lead_v),
                        "speed_kmh": float(raw_env.lead_v) * 3.6,
                        "signal": "green",
                        "segment": lead_segment.id,
                        "dwell_timer": int(raw_env.lead_dwell_timer),
                        "stalled": raw_env.lead_stalled,
                        "held": raw_env.lead_held,
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
                    "hazards": [{"start": s, "end": e} for s, e in raw_env.active_hazards],
                    "done": bool(done),
                    "timetable": raw_env.timetable.to_dict(),
                    "punctuality": raw_env.get_punctuality_status(),
                }
                
                if self.broadcast_callback:
                    await self.broadcast_callback(state)
                    
                if done:
                    self.is_running = False  # Auto pause when episode is finished
                    
                # Throttle the simulation
                await asyncio.sleep(1.0 / self.steps_per_second)
        except Exception as e:
            print(f"SIMULATION ABORTED DUE TO ERROR: {e}")
            import traceback
            traceback.print_exc()
            self.is_running = False
