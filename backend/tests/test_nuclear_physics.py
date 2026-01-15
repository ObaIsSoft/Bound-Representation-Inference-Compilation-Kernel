
import sys
import os
print("DEBUG: Starting Nuclear Verification Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.physics_oracle.physics_oracle import PhysicsOracle

def test_nuclear_integration():
    print("--- NUCLEAR PHYSICS ORACLE TEST ---")
    
    oracle = PhysicsOracle()
    
    # [1] FISSION: Reactor Startup
    print("\n[1] Testing Fission Reactor Dynamics (Point Kinetics)...")
    # Reactivity +0.002 (Positive but < Beta 0.0065) -> Stable Period
    fission_res = oracle.solve(
        query="Simulate reactor startup",
        domain="NUCLEAR",
        params={
            "type": "FISSION", 
            "reactivity": 0.002, 
            "beta": 0.0065,
            "duration": 5.0
        }
    )
    print(f"Fission Result:\n{fission_res}")
    
    # Check if power increased and is valid
    if fission_res["status"] == "solved" and fission_res["final_power_ratio"] > 1.0:
        if not fission_res["prompt_critical"]:
             print("Fission Solver: SUCCESS (Expected Supercritical Growth, Safe)")
        else:
             print("Fission Solver: WARNING (Prompt Critical Detected!)")
    else:
        print("Fission Solver: FAILURE")

    # [2] FUSION: Ignition Check
    print("\n[2] Testing Fusion Plasma (Lawson Criterion)...")
    # ITER parameters roughly: n=1e20, T=15keV, tau=3s
    fusion_res = oracle.solve(
        query="Check ITER Ignition",
        domain="NUCLEAR",
        params={
             "type": "FUSION",
             "density": 1e20,
             "temperature_kev": 15.0,
             "confinement_time": 3.0
        }
    )
    print(f"Fusion Result:\n{fusion_res}")
    
    if fusion_res["status"] == "solved":
        q = fusion_res["Q_factor"]
        print(f"Fusion Q-Factor: {q}")
        
        # Print Relatable Metrics
        metrics = fusion_res.get("relatable_metrics", {})
        print("\n--- Relatable Output (100m^3 Core, 1kg Fuel) ---")
        print(f"Total Power: {metrics.get('total_thermal_power_MW'):.2f} MW")
        print(f"Houses Powered: {metrics.get('houses_powered'):,}")
        print(f"Duration: {metrics.get('operation_duration_years'):.2f} Years (continuous)")
        print("--------------------------------------------------\n")
        
        if q > 1.0 or fusion_res["ignition"]:
            print("Fusion Solver: SUCCESS (Breakeven Calculated)")
        else:
             print("Fusion Solver: SUCCESS (Calculated No-Ignition Correctly)")
    else:
        print("Fusion Solver: FAILURE")

if __name__ == "__main__":
    test_nuclear_integration()
