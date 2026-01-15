import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.physics_agent import PhysicsAgent
import statistics

def verify_stochastic():
    print("--- Verifying Stochastic Physics (vHIL Jitter) ---")
    
    agent = PhysicsAgent()
    
    # CASE 1: Deterministic Baseline
    print("\n[Case 1] Deterministic (Noise=0.0)")
    state = {"velocity": 100.0, "altitude": 500.0, "temperature": 20.0}
    inputs = {"thrust": 2000.0, "mass": 100.0, "noise_level": 0.0}
    
    results_det = []
    for _ in range(5):
        res = agent.step(state, inputs)
        results_det.append(res["state"]["velocity"])
    
    # Variance should be 0.0
    var_det = statistics.variance(results_det)
    print(f"Deterministic Variance: {var_det}")
    assert var_det == 0.0
    print("✅ Deterministic Check Passed")
    
    # CASE 2: Stochastic (Noise=1.0)
    print("\n[Case 2] Stochastic (Noise=1.0)")
    inputs_stoch = {"thrust": 2000.0, "mass": 100.0, "noise_level": 1.0}
    
    results_stoch = []
    gusts_detected = False
    
    for _ in range(20):
        res = agent.step(state, inputs_stoch)
        results_stoch.append(res["state"]["velocity"])
        # Check logs for wind gusts
        for log in res["state"]["logs"]:
            if "Gust detected" in log:
                gusts_detected = True
    
    var_stoch = statistics.variance(results_stoch)
    print(f"Stochastic Variance: {var_stoch}")
    assert var_stoch > 0.0
    
    if gusts_detected:
         print("✅ Wind Gusts Detected")
    else:
         print("⚠️ No large gusts detected (statistical chance), but variance exists.")
         
    print("✅ Stochastic Check Passed (Variance > 0)")

if __name__ == "__main__":
    verify_stochastic()
