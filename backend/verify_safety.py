import asyncio
import sys
import os

# Add backend to path
sys.path.append(os.getcwd())

from agent_registry import registry

async def verify_safety():
    """
    Verify SafetyAgent functionality with new materials.
    """
    print("🚀 Starting Safety Verification...")
    
    # Get agent from registry
    try:
        safety_agent = registry.get_agent("SafetyAgent")
        print("✅ SafetyAgent loaded from registry")
    except Exception as e:
        print(f"❌ Failed to load SafetyAgent: {e}")
        return

    # Test Case 1: Simple Single Material (Copper)
    print("\n🧪 Test Case 1: Single Material (Copper)")
    try:
        result = await safety_agent.run({
            "materials": ["Copper"],
            "physics_metrics": {
                "max_stress_mpa": 50,    # Low stress
                "max_temp_c": 100,       # Moderate temp
                "mass_kg": 5.0
            },
            "application_type": "industrial"
        })
        print(f"Result: {result.get('status')} (Score: {result.get('safety_score')})")
        
        if result.get("status") == "safe":
             print("✅ Standard Check PASSED")
        else:
             print(f"⚠️ Standard Check Warning: {result.get('hazards')}")
             
    except Exception as e:
        print(f"❌ Standard Check CRASHED: {e}")

    # Test Case 2: Multi-Material Assembly (Copper + Steel)
    print("\n🧪 Test Case 2: Multi-Material Assembly (Copper + Steel)")
    try:
        # Scenario: 
        # Copper part (limit ~200 MPa yield)
        # Steel part (limit ~200 MPa yield)
        # Load: 150 MPa (Safe for both)
        result = await safety_agent.run({
            "materials": ["Copper", "Steel"],
            "physics_metrics": {
                "max_stress_mpa": 150, 
                "max_temp_c": 200      
            },
            "application_type": "aerospace"
        })
        
        print(f"Result: {result.get('status')}")
        print(f"Hazards: {result.get('hazards')}")
        
        # We expect it to run without crashing.
        # It might flag hazards if 150MPa is too close to Copper yield.
        if result:
            print("✅ Multi-Material Check COMPLETED (No Crash)")
        
    except Exception as e:
        print(f"❌ Multi-Material Check CRASHED: {e}")

if __name__ == "__main__":
    asyncio.run(verify_safety())
