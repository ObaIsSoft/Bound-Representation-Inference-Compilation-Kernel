
import sys
import os
import time

# Fix Imports
sys.path.append(os.getcwd())
sys.path.append(os.path.join(os.getcwd(), "backend"))

def run_integrity_check():
    print("--- BRICK OS SYSTEM INTEGRITY CHECK ---")
    results = {}
    
    # 1. VERIFY CRITICS
    print("\n[1] Verifying Adversarial Critics...")
    try:
        from backend.agents.critics.SurrogateCritic import SurrogateCritic
        from backend.agents.critics.PhysicsCritic import PhysicsCritic
        
        # Instantiate
        sc = SurrogateCritic(window_size=10)
        pc = PhysicsCritic(window_size=10)
        
        # Test Observation Method
        sc.observe(input_state={"test": 1}, prediction={"gate_value": 0.5}, validation_result={"verified": True})
        
        results["Critics"] = "✅ Active"
        print("  - SurrogateCritic: OK")
        print("  - PhysicsCritic: OK")
    except ImportError as e:
        results["Critics"] = f"❌ ImportError: {e}"
        print(f"  - FAILED: {e}")
    except Exception as e:
         results["Critics"] = f"❌ Runtime Error: {e}"
         print(f"  - FAILED: {e}")

    # 2. VERIFY ARES MIDDLEWARE (Validation)
    print("\n[2] Verifying ARES Middleware...")
    try:
        from backend.ares import AresMiddleware, AresUnitError
        ares = AresMiddleware()
        # Test valid input
        ares.validate_unit({"mass": {"value": 10, "unit": "kg"}}, "mass")
        results["Ares"] = "✅ Active"
        print("  - AresMiddleware: OK")
    except Exception as e:
        results["Ares"] = f"❌ Failed: {e}"
        print(f"  - FAILED: {e}")

    # 3. VERIFY NEURAL SURROGATES (MaterialNet)
    print("\n[3] Verifying Neural Surrogates...")
    try:
        from backend.models.material_net import MaterialNet
        net = MaterialNet()
        # Check if weights loaded (MaterialNet has W1, W2 etc)
        if hasattr(net, 'W1'):
            results["MaterialNet"] = "✅ Loaded"
            print("  - MaterialNet (Weights): OK")
        else:
             results["MaterialNet"] = "⚠️ Uninitialized"
    except Exception as e:
        results["MaterialNet"] = f"❌ Failed: {e}"
        print(f"  - FAILED: {e}")

    # 4. VERIFY DATABASE (Supabase)
    print("\n[4] Verifying Database Connection...")
    try:
        from backend.database.supabase_client import SupabaseClient
        db = SupabaseClient()
        if db.client:
            results["Database"] = "✅ Connected"
            print("  - Supabase: OK")
        else:
            results["Database"] = "❌ Client None"
    except Exception as e:
        results["Database"] = f"❌ Failed: {e}"
        print(f"  - FAILED: {e}")

    # 5. VERIFY PHYSICS KERNEL
    print("\n[5] Verifying Physics Kernel...")
    try:
        from backend.physics.kernel import get_physics_kernel
        pk = get_physics_kernel()
        results["PhysicsKernel"] = "✅ Loaded"
        print("  - PhysicsKernel: OK")
    except Exception as e:
        results["PhysicsKernel"] = f"❌ Failed: {e}"
        print(f"  - FAILED: {e}")

    print("\n--- SUMMARY ---")
    for k, v in results.items():
        print(f"{k}: {v}")

if __name__ == "__main__":
    run_integrity_check()
