import numpy
import sys

print(f"NumPy Version: {numpy.__version__}")

try:
    import numpy.core.multiarray
    import numpy.core.numeric
    import numpy.random._pickle
    print("Direct imports succeeded")
except ImportError as e:
    print(f"Import failed: {e}")

# Try the monkeypatching logic
try:
    sys.modules.setdefault("numpy._core", numpy.core)
    sys.modules.setdefault("numpy._core.numeric", numpy.core.numeric)
    sys.modules.setdefault("numpy._core.multiarray", numpy.core.multiarray)
    
    if not hasattr(numpy.random._pickle, "_patched"):
        orig_ctor = getattr(numpy.random._pickle, "__bit_generator_ctor")
        def patched_ctor(bit_generator_name):
            if not isinstance(bit_generator_name, str):
                if hasattr(bit_generator_name, "__name__"):
                    bit_generator_name = bit_generator_name.__name__
            return orig_ctor(bit_generator_name)
        setattr(numpy.random._pickle, "__bit_generator_ctor", patched_ctor)
        setattr(numpy.random._pickle, "_patched", True)
    print("Monkeypatch succeeded")
except Exception as e:
    print(f"Logic failed: {e}")
