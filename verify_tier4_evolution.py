
import sys
import os
import json
import logging

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), "backend"))

from agents.compliance_agent import ComplianceAgent
from agents.control_agent import ControlAgent
from agents.electronics_agent import ElectronicsAgent
from agents.mep_agent import MepAgent
from agents.network_agent import NetworkAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Tier4Verification")

def verify_compliance():
    print("\n--- ComplianceAgent (Dynamic Rules) ---")
    agent = ComplianceAgent()
    
    # Test 1: Violation (Overweight)
    params_heavy = {
        "regime": "AERIAL",
        "design_params": {"mass_kg": 30.0, "is_night_operation": False}
    }
    res_heavy = agent.run(params_heavy)
    print(f"Heavy Test (30kg): {res_heavy['status']} (Expected: non_compliant)")
    if res_heavy["status"] == "non_compliant":
        print("✅ Correctly identified mass violation.")
    else:
        print("❌ Failed to identify mass violation.")

    # Test 2: Pass
    params_light = {
        "regime": "AERIAL",
        "design_params": {"mass_kg": 10.0, "is_night_operation": False}
    }
    res_light = agent.run(params_light)
    print(f"Light Test (10kg): {res_light['status']} (Expected: compliant)")
    if res_light["status"] == "compliant":
        print("✅ Correctly passed valid design.")
    else:
        print("❌ Failed to pass valid design.")
        print(f"FAILED RULES: {res_light.get('failed_rules')}")
        print(f"LOGS: {res_light.get('logs')}")

def verify_control():
    print("\n--- ControlAgent (RL + SysID) ---")
    agent = ControlAgent()
    
    # Mock Flight History for SysID
    history = [
        {"error": 0.0},
        {"error": 0.5}, # Drift
        {"error": 1.2}  # Increasing error -> Wind?
    ]
    
    params = {
        "control_mode": "RL",
        "flight_history": history,
        "state_vec": [0,0,10, 0,0,0], 
        "target_vec": [0,0,10],
    }
    
    res = agent.run(params)
    print(f"Status: {res.get('status')}")
    print(f"Method: {res.get('method')}")
    print(f"Disturbance Est: {res.get('estimated_disturbance')}")
    print(f"LOGS: {res.get('logs')}")
    
    if res.get("status") == "success" and "RL" in res.get("method", ""):
        print("✅ PPO Policy Inference successful.")
    else:
        print("❌ Failed to run RL policy (Check if pickle exists).")
        # Print warning if any
        if "message" in res: print(f"Message: {res['message']}")

def verify_electronics():
    print("\n--- ElectronicsAgent (Generative Topology GA) ---")
    agent = ElectronicsAgent()
    
    reqs = {
        "v_in": 12.0,
        "v_out": 3.3,
        "min_efficiency": 0.90,
        "pop_size": 10,  # Small for speed
        "generations": 3
    }
    
    res = agent.evolve_topology(reqs)
    
    print(f"Method: {res.get('method')}")
    print(f"Best Fitness: {res.get('best_fitness')}")
    print(f"Generations: {res.get('generations_run')}")
    print(f"LOGS: {res.get('logs')}")
    
    if res.get("best_fitness", 0) > 0:
        print("✅ GA generated a valid topology.")
    else:
        print("❌ GA failed to produce results.")

def verify_mep():
    print("\n--- MepAgent (3D A* Routing) ---")
    agent = MepAgent()
    
    params = {
        "structure_geometry": [
            {"position": [5,5,5]}, # Obstacle
            {"position": [5,5,6]}
        ],
        "systems": [
            {"id": "pipe_1", "start": [0,0,0], "end": [10,10,10]}
        ]
    }
    
    res = agent.run(params)
    routes = res.get("routes", [])
    print(f"Routes generated: {len(routes)}")
    if len(routes) > 0:
        print(f"Path Length: {routes[0]['length']}")
        print("✅ A* found a path.")
    else:
        print("❌ No path found.")

def verify_network():
    print("\n--- NetworkAgent (Latency Prediction) ---")
    agent = NetworkAgent()
    
    params = {
        "traffic_flows": [
            {"src": "Cam1", "dst": "CPU", "mbps": 50.0}, # High load
            {"src": "Sensor1", "dst": "CPU", "mbps": 0.1} # Low load
        ]
    }
    
    res = agent.run(params)
    preds = res.get("flow_predictions", [])
    
    print(f"Predictions: {len(preds)}")
    for p in preds:
        print(f"  {p['flow_id']}: {p['predicted_latency_ms']}ms ({p['status']})")
        
    if len(preds) == 2:
        print("✅ Latency model ran successfully.")
    else:
        print("❌ Failed to generate predictions.")

if __name__ == "__main__":
    print("=== STARTING TIER 4 VERIFICATION ===")
    verify_compliance()
    verify_control()
    verify_electronics()
    verify_mep()
    verify_network()
    print("\n=== VERIFICATION COMPLETE ===")
