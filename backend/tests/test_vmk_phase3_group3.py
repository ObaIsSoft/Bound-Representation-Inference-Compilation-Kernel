
import sys
import os
import math
print("DEBUG: Starting Phase 3 Group 3 Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.manifold_agent import ManifoldAgent
from agents.geometry_agent import GeometryAgent

def test_phase3_group3_integration():
    print("--- VMK PHASE 3 (GROUP 3) INTEGRATION TEST ---")
    
    # 1. Manifold Agent (Repair)
    print("\n[1] Testing ManifoldAgent.repair_topology...")
    manifold = ManifoldAgent()
    
    # Simulate broken mesh defined by history
    history = [
        {"tool_id": "repair_tool", "radius": 5.0, "path": [[0,0,0], [10,0,0]]}
    ]
    
    # Run Repair
    repair_res = manifold.repair_topology(history, stock_dims=[40,40,40])
    print(f"Repair Result: {repair_res}")
    
    if repair_res["status"] == "repaired" and repair_res["is_watertight"]:
        print("Manifold Repair: SUCCESS")
    else:
        print("Manifold Repair: FAILURE")
        
    # 2. Geometry Agent (Safe Boolean)
    print("\n[2] Testing GeometryAgent.perform_safe_boolean...")
    geo_agent = GeometryAgent()
    
    # Define Base and Tool for Difference
    base = {"id": "block", "radius": 20.0, "path": [[0,0,0],[0,0,0]]} # Essentially Stock
    tool = {"id": "drill", "radius": 5.0,  "path": [[0,0,0],[0,0,10]]}
    
    bool_res = geo_agent.perform_safe_boolean(base, tool, operation="DIFFERENCE")
    print(f"Boolean Result: {bool_res}")
    
    # We expect Success
    if bool_res["success"] and bool_res["is_valid"]:
        print("Safe Boolean: SUCCESS")
    else:
        print("Safe Boolean: FAILURE")

if __name__ == "__main__":
    test_phase3_group3_integration()
