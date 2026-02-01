import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from agents.surrogate.pinn_model import MultiPhysicsPINN
from agents.critics.adversarial import RedTeamAgent
from agents.evolution import PrimitiveType

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_adversarial_verification():
    print("=== Adversarial Validation (Red Team) Verification ===")
    
    # 1. Setup Oracle and Adversary
    pinn = MultiPhysicsPINN()
    red_team = RedTeamAgent(pinn)
    
    # 2. Mock a "Weak" Design (Disconnected or thin)
    weak_nodes = [
        {"id": "1", "type": "CUBE", "params": {"width": {"value": 0.1}, "height": {"value": 10.0}, "depth": {"value": 0.1}}, "transform": [0,0,0, 0,0,0]}
    ]
    
    # 3. Mock a "Strong" Design (Solid block)
    strong_nodes = [
         {"id": "s1", "type": "CUBE", "params": {"width": {"value": 2.0}, "height": {"value": 2.0}, "depth": {"value": 2.0}}, "transform": [0,0,0, 0,0,0]},
         {"id": "s2", "type": "CUBE", "params": {"width": {"value": 2.0}, "height": {"value": 2.0}, "depth": {"value": 2.0}}, "transform": [1,0,0, 0,0,0]}
    ]
    
    constraints = {
        "max_weight": {"val": 50.0},
        "ambient_temp": {"val": 300.0}
    }
    
    print("\n--- Testing WEAK Design (Thin Beam) ---")
    weak_result = red_team.stress_test(weak_nodes, constraints, trials=200) # More trials for coverage
    print(f"Failure Rate: {weak_result['failure_rate']*100:.1f}%")
    print(f"Robust: {weak_result['is_robust']}")
    if weak_result['worst_scenario']:
         print(f"Worst Scenario: {weak_result['worst_scenario']['type']} (Mult: {weak_result['worst_scenario']['multiplier']:.1f})")
    
    print("\n--- Testing STRONG Design (Solid Block) ---")
    strong_result = red_team.stress_test(strong_nodes, constraints, trials=200)
    print(f"Failure Rate: {strong_result['failure_rate']*100:.1f}%")
    print(f"Robust: {strong_result['is_robust']}")
    if strong_result['worst_scenario']:
         print(f"Worst Scenario: {strong_result['worst_scenario']['type']} (Mult: {strong_result['worst_scenario']['multiplier']:.1f})")
    
    # 4. Validation
    if weak_result['failure_rate'] > strong_result['failure_rate']:
        print("\n[SUCCESS] Red Team correctly identified the weaker design.")
    else:
        print("\n[FAILED] Red Team logic inconclusive.")

if __name__ == "__main__":
    run_adversarial_verification()
