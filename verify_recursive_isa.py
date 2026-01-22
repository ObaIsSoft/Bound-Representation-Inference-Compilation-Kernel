import asyncio
from typing import Dict, Any
from core.system_registry import SYSTEM_RESOLVER, ModularISA

async def verify_recursive_isa():
    print("--- Verifying Recursive ISA & Aggregation ---")
    
    # 1. Setup Test Hierarchy
    # Root -> Leg (sub) -> Foot (sub)
    root = SYSTEM_RESOLVER.root
    
    # Reset for clean test
    root.exports = {"mass": 0.0, "power_draw": 0.0, "cost": 0.0}
    root.sub_pods = {} 
    
    leg = ModularISA(name="Test_Leg", parent_id=root.id, constraints={"local_mass": 10.0, "local_cost": 100.0})
    foot = ModularISA(name="Test_Foot", parent_id=leg.id, constraints={"local_mass": 5.0, "local_cost": 50.0})
    
    root.sub_pods["leg"] = leg
    leg.sub_pods["foot"] = foot
    
    print(f"Hierarchy Created: Root -> Leg -> Foot")
    
    # 2. Simulate Agent Execution (Bottom-Up Population)
    # Ideally agents run and write to exports. We simulate this manually first.
    # Foot Execution
    foot.exports["mass"] = foot.constraints["local_mass"]
    foot.exports["cost"] = foot.constraints["local_cost"]
    
    # Leg Execution (Pre-aggregation, just local)
    leg.exports["mass"] = leg.constraints["local_mass"] 
    # Note: leg.exports["cost"] usually assumes sum of children + local. 
    # But agents might write ONLY local computed mass if they don't do recursion themselves.
    # However, converge_up handles the summing.
    
    # 3. Test Converge Up
    print("Running converge_up(root)...")
    await SYSTEM_RESOLVER.converge_up(root)
    
    # 4. Verify Aggregation
    print("\n--- Results ---")
    print(f"Foot Mass: {foot.exports.get('mass')} (Expected 5.0)")
    print(f"Leg Mass: {leg.exports.get('mass')} (Expected 15.0 -> 10 local + 5 foot)")
    print(f"Root Mass: {root.exports.get('mass')} (Expected 15.0 -> 0 local + 15 leg)")
    
    print(f"Foot Cost: {foot.exports.get('cost')} (Expected 50.0)")
    print(f"Leg Cost: {leg.exports.get('cost')} (Expected 150.0 -> 100 local + 50 foot)")
    
    assert foot.exports["mass"] == 5.0
    assert leg.exports["mass"] == 15.0
    assert root.exports["mass"] == 15.0
    
    assert leg.exports["cost"] == 150.0
    
    print("\n[SUCCESS] Aggregation Logic Verified.")

    # 5. Access Control Test
    print("\n--- Testing Access Control ---")
    try:
        # Parent accessing Child Constraints (Should Fail)
        SYSTEM_RESOLVER.validate_access(accessor=root, target=leg, property_type="constraints")
        print("[FAIL] Prop access violation not caught!")
    except PermissionError as e:
        print(f"[PASS] Caught expected violation: {e}")

    try:
        # Parent accessing Child Exports (Should Pass)
        SYSTEM_RESOLVER.validate_access(accessor=root, target=leg, property_type="exports")
        print("[PASS] Access to exports allowed.")
    except PermissionError:
        print("[FAIL] Valid access blocked!")

if __name__ == "__main__":
    asyncio.run(verify_recursive_isa())
