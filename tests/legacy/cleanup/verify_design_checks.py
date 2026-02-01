
import logging
import sys
import os

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from physics.validation.constraint_checker import ConstraintChecker

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def verify_design_checks():
    checker = ConstraintChecker()
    
    tests = [
        # --- Astrophysics ---
        {
            "name": "Stable Orbit",
            "check": lambda: checker.check_orbital_stability(velocity=7000, escape_velocity=11000),
            "expect_valid": True
        },
        {
            "name": "Unbound Orbit (Escape)",
            "check": lambda: checker.check_orbital_stability(velocity=12000, escape_velocity=11000),
            "expect_valid": False
        },
        
        # --- Construction ---
        {
            "name": "Beam (Safe Deflection)",
            "check": lambda: checker.check_deflection(deflection=0.01, length=5.0, limit_ratio=360), # 0.01 < 5/360=0.0138
            "expect_valid": True
        },
        {
            "name": "Beam (Excessive Deflection)",
            "check": lambda: checker.check_deflection(deflection=0.02, length=5.0, limit_ratio=360),
            "expect_valid": False
        },
        {
            "name": "Column (Safe Load)",
            "check": lambda: checker.check_buckling(load=5000, critical_load=10000),
            "expect_valid": True
        },
        {
            "name": "Column (Buckling Failure)",
            "check": lambda: checker.check_buckling(load=12000, critical_load=10000),
            "expect_valid": False
        },
        
        # --- Quantum ---
        {
            "name": "Valid Uncertainty",
            "check": lambda: checker.check_uncertainty(delta_x=1e-10, delta_p=1e-23), # product 1e-33 > hbar/2 ~ 5e-35
            "expect_valid": True
        },
        {
            "name": "Heisenberg Violation",
            "check": lambda: checker.check_uncertainty(delta_x=1e-12, delta_p=1e-26), # product 1e-38 < hbar/2
            "expect_valid": False
        }
    ]
    
    passed = 0
    
    print("\n=== Design Verification Checks ===\n")
    
    for test in tests:
        result = test['check']()
        is_valid = result['valid']
        
        status = "PASS ✅" if is_valid == test['expect_valid'] else "FAIL ❌"
        if status == "PASS ✅":
            passed += 1
            
        print(f"{test['name']}: {status}")
        if not is_valid:
            print(f"  Reason: {result.get('reason')}")
            
    print(f"\nSummary: {passed}/{len(tests)} Tests Passed")

if __name__ == "__main__":
    verify_design_checks()
