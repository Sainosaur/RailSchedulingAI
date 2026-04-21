import sys
import time
from pathlib import Path

# Ensure project-root imports work
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from environment.railway_env import ModernizedLine104
from think_layer.config import TrainConfig
import os

def watch_live():
    config = TrainConfig()
    model_path = os.path.join(config.model_dir, "best_model.zip")
    vecnorm_path = os.path.join(config.model_dir, "best_vecnormalize.pkl")
    
    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}. You might need to train first.")
        return

    def make_env():
        return ModernizedLine104(lead_train_speed=20.0, training_mode=False)

    venv = DummyVecEnv([make_env])
    
    if os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        
    model = PPO.load(model_path)
    
    print("Starting Live AI Watcher...")
    print("Press Ctrl+C to stop.\n")
    
    obs = venv.reset()
    done = False
    
    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, dones, infos = venv.step(action)
            
            info = infos[0]
            raw_reward = reward[0]
            
            # Unwrap env
            base_env = venv.envs[0]
            while hasattr(base_env, "env"):
                base_env = base_env.env
                
            seg = base_env.vl.get_segment(base_env.x)
            
            print(f"Time: {info.get('time', 0):6.1f}s | Pos: {base_env.x:7.1f}m | Vel: {base_env.v:5.2f}m/s (Limit: {seg.limit_ms:4.1f}) | Next Station: {base_env.STATIONS[min(base_env.last_station_idx + 1, len(base_env.STATIONS) - 1)]:7.1f}m")
            print(f"  ↳ Proposed Accel: {info.get('proposed_a', 0):+5.2f} | Safe Accel: {info.get('safe_a', 0):+5.2f} | Override: {info.get('overridden', False)}")
            
            # Format non-zero rewards
            breakdown = info.get("reward_breakdown", {})
            active_rewards = [f"{k}={v:+.2f}" for k, v in breakdown.items() if abs(v) > 0.001]
            print(f"  ↳ Total Reward: {raw_reward:+6.2f} | Breakdown: {', '.join(active_rewards)}")
            print("-" * 80)
            
            time.sleep(0.5) # Sleep for half a second so it's readable
            done = dones[0]
            
    except KeyboardInterrupt:
        print("\nWatcher stopped by user.")

if __name__ == "__main__":
    watch_live()
