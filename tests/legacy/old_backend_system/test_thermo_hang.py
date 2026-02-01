"""
Test Thermo Library behavior on inorganic compounds
"""
import time
import sys

print("Loading thermo...")
start = time.time()
try:
    import thermo
    print(f"Import took {time.time() - start:.2f}s")
except ImportError:
    print("Thermo not installed")
    sys.exit(0)

print("\nQuerying 'Fe2O3' in thermo (This mimics the hang)...")
start = time.time()
try:
    c = thermo.Chemical('Fe2O3')
    print(f"Result: {c}")
    print(f"Time: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error/Not Found after {time.time() - start:.2f}s: {e}")

print("\nQuerying 'Water' in thermo...")
start = time.time()
try:
    c = thermo.Chemical('Water')
    print(f"Result: {c.name}, Density: {c.rho}")
    print(f"Time: {time.time() - start:.2f}s")
except Exception as e:
    print(f"Error: {e}")
