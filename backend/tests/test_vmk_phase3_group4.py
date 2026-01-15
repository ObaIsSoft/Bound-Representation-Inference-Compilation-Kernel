
import sys
import os
print("DEBUG: Starting Phase 3 Group 4 Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.verification_agent import VerificationAgent
from agents.validator_agent import ValidatorAgent

def test_phase3_group4_integration():
    print("--- VMK PHASE 3 (GROUP 4) V&V TEST ---")
    
    # 1. Verification Agent (G-Code Safety)
    print("\n[1] Testing VerificationAgent.verify_code_safety...")
    verifier = VerificationAgent()
    
    # Simulate G-Code with a crash
    # Rapid move from outside [-50,0,0] to opposite side [50,0,0]
    # Through a 20x20x20 block at [0,0,0].
    # Stock Dims: [20,20,20].
    # Path goes right through the middle.
    
    gcode = [
        {"tool_id": "drill", "op": "RAPID", "path": [[-50,0,0], [50,0,0]]}
    ]
    
    safety_check = verifier.verify_code_safety(gcode, stock_dims=[20,20,20])
    print(f"Safety Check: {safety_check}")
    
    if not safety_check["verified"] and safety_check["collision_count"] > 0:
        print("Safety Verification: SUCCESS (Detected Crash)")
    else:
        print("Safety Verification: FAILURE")
        
        
    # 2. Validator Agent (Fidelity Check)
    print("\n[2] Testing ValidatorAgent.validate_simulation_fidelity...")
    validator = ValidatorAgent()
    
    sim_result = {"thrust_n": 100.0, "drag_n": 10.0}
    ground_truth = {"thrust_n": 105.0, "drag_n": 9.5} # 5.0 and 0.5 diff
    
    # Tolerance 2.0 -> Should FAIL (Diff 5.0 > 2.0)
    check_strict = validator.validate_simulation_fidelity(sim_result, ground_truth, tolerance=2.0)
    print(f"Strict Check (Tol=2.0): {check_strict['validated']} (Avg Err={check_strict['avg_error']})")
    
    # Tolerance 10.0 -> Should PASS
    check_loose = validator.validate_simulation_fidelity(sim_result, ground_truth, tolerance=10.0)
    print(f"Loose Check (Tol=10.0): {check_loose['validated']}")
    
    if not check_strict["validated"] and check_loose["validated"]:
        print("Fidelity Validation: SUCCESS")
    else:
        print("Fidelity Validation: FAILURE")

if __name__ == "__main__":
    test_phase3_group4_integration()
