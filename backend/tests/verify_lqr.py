import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.control_agent import ControlAgent

def verify_lqr():
    print("--- Verifying LQR Control Synthesis ---")
    
    agent = ControlAgent()
    
    # CASE 1: Baseline (Low Inertia, Balanced Costs)
    print("\n[Case 1] Small Drone")
    res1 = agent.run({
        "inertia_tensor": [0.01, 0.01, 0.02],
        "q_error_cost": 100.0,
        "r_effort_cost": 1.0
    })
    gains1 = res1["gain_matrix"]["roll"]
    print(f"J=0.01, Q=100 -> Kp={gains1['kp']}, Kd={gains1['kd']}")
    
    # CASE 2: Heavy Inertia (Should increase Kd to dampen)
    print("\n[Case 2] Heavy Drone (J x 10)")
    res2 = agent.run({
        "inertia_tensor": [0.1, 0.1, 0.2],
        "q_error_cost": 100.0,
        "r_effort_cost": 1.0
    })
    gains2 = res2["gain_matrix"]["roll"]
    print(f"J=0.1, Q=100 -> Kp={gains2['kp']}, Kd={gains2['kd']}")
    
    # Check: Kp should be same (defined by Q/R), Kd should increase
    assert gains1['kp'] == gains2['kp']
    assert gains2['kd'] > gains1['kd']
    print("✅ Physics Check Passed: Higher Inertia = Higher Damping Gain")
    
    # CASE 3: High Performance (High Q)
    print("\n[Case 3] Aggressive Tune (Q x 100)")
    res3 = agent.run({
        "inertia_tensor": [0.01, 0.01, 0.02],
        "q_error_cost": 10000.0, # Want precise control
        "r_effort_cost": 1.0
    })
    gains3 = res3["gain_matrix"]["roll"]
    print(f"J=0.01, Q=10000 -> Kp={gains3['kp']}, Kd={gains3['kd']}")
    
    # Check: Kp should increase significantly
    assert gains3['kp'] > gains1['kp']
    print("✅ Performance Check Passed: Higher Cost = Higher Stiffness Gain")
    
    print("\nLQR Synthesis Verified Successfully.")

if __name__ == "__main__":
    verify_lqr()
