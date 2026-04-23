import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve()))

try:
    from environment.railway_env import ModernizedLine104
    from server.simulation import SimulationRunner

    print("Imports successful!")
except Exception as e:
    import traceback

    import traceback
    traceback.print_exc()
