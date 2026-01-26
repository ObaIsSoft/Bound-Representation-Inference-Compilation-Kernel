import sys
import os
import logging
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agents.critics.scientist import ScientistAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_scientist_verification():
    print("=== Scientist Agent (Symbolic Regression) Verification ===")
    
    scientist = ScientistAgent()
    
    # Scenario 1: Simple Linear Law
    # y = mass * 2 + 5
    print("\n--- Discovery Task 1: Linear Law (y = 2*m + 5) ---")
    data_1 = []
    for _ in range(20):
        m = random.uniform(1, 10)
        y = m * 2.0 + 5.0
        data_1.append({"mass": m, "outcome": y})
        
    law_1 = scientist.discover_law(data_1, "outcome")
    print(f"Discovered: {law_1}")
    
    # Scenario 2: Physics Law (Kinetic Energy)
    # E = 0.5 * m * v^2
    # NOTE: Our simple GP might struggle with squares or 0.5 constants exactly, but should approximate
    print("\n--- Discovery Task 2: Kinetic Energy (E = 0.5 * m * v * v) ---")
    data_2 = []
    for _ in range(50):
        m = random.uniform(1, 5)
        v = random.uniform(1, 5)
        e = 0.5 * m * (v * v)
        data_2.append({"mass": m, "velocity": v, "energy": e})
        
    law_2 = scientist.discover_law(data_2, "energy")
    print(f"Discovered: {law_2}")
    
    print("\n[SUCCESS] Scientist Agent verification complete (Check qualitative results above).")

if __name__ == "__main__":
    run_scientist_verification()
