import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import requests
from agents.geometry_agent import GeometryAgent
from agents.physics_agent import PhysicsAgent
from agents.mitigation_agent import MitigationAgent

def test_integration():
    print("--- Testing Agent Wiring ---")

    # 1. Chat API (Mock Request)
    # We can't easily curl the running server from here without knowing port/state, 
    # but we can check the endpoint function exists if we imported main...
    # instead let's trust the endpoint creation for now and test the logic paths.

    # 2. Manifold in Geometry
    print("\n[Geometry] Testing Manifold Integration...")
    geo = GeometryAgent()
    # Generative mode triggers the stub which returns a simple mesh
    res = geo.run({}, "Make a messy mesh") 
    print(f"Manifold Validation Key Present: {'manifold_validation' in res}")
    if 'manifold_validation' in res:
        print(f"Validation Result: {res['manifold_validation']}")

    # 3. Multi-Mode in Physics
    print("\n[Physics] Testing Multi-Mode Integration...")
    phy = PhysicsAgent()
    # Mismatch: AERIAL regime but GROUND params
    env = {"regime": "AERIAL", "gravity": 9.81}
    bad_res = phy.run(
        environment=env, 
        geometry_tree=[{"mass_kg": 10}], 
        design_params={"mode": "GROUND_VEHICLE"} # Should trigger mismatch if logic works
    )
    print("Result validation flags:", bad_res.get("validation_flags"))
    
    # 4. Mitigation Logic (Direct test since Orchestrator needs full graph/async)
    print("\n[Mitigation] Testing Logic Path...")
    # Simulate what Orchestrator does
    flags = {"physics_safe": False, "reasons": ["Stress 300 > Yield 200"]}
    if not flags["physics_safe"]:
        print("Triggering Mitigation...")
        mit = MitigationAgent()
        fixes = mit.run({"errors": flags["reasons"], "physics_data": {"max_stress_mpa": 300, "yield_strength_mpa": 200}})
        print(f"Proposed Fixes: {len(fixes.get('fixes', []))}")
        print(fixes.get('fixes')[0]['description'])

if __name__ == "__main__":
    test_integration()
