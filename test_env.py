import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    from environment.railway_env import ModernizedLine104

    env = ModernizedLine104()
    print("Environment created successfully!")
    obs, info = env.reset()
    print("Reset successful!")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    print("Step successful!")
except Exception as e:
    import traceback

    traceback.print_exc()
