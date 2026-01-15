
import sys
import os
import json

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from hwc_kernel import HighFidelityKernel, FitClass

def verify_hwc_v3():
    print("\n--- Testing HWC Kernel v3.2.0 ---")
    hwc = HighFidelityKernel()
    
    # 1. Define Test Geometry with Fits
    tree = [
        # Base Chasis
        {
            "id": "chassis",
            "type": "box",
            "params": {"dims": [10.0, 10.0, 2.0]},
            "blend": 0.0
        },
        # Bearing Mount (Precision Fit)
        {
            "id": "bearing_mount_hole",
            "type": "cylinder",
            "params": {"height": 2.0, "radius": 5.0}, # Nominal 5.0mm
            "fit": FitClass.H7_g6, # Should adjust radius
            "blend": 0.0
        }
    ]
    
    # 2. Synthesize
    isa = hwc.synthesize_isa("test_project", [], tree)
    
    # 3. Transpile to GLSL
    glsl = hwc.to_glsl(isa)
    
    # 4. Inspection
    print("\n[GLSL Output Analysis]")
    if "d_0 = min(d_0, d_1);" in glsl:
        print("✓ Dynamic Tree Traversal functioning")
    else:
        print("✗ Dynamic Tree Traversal FAILED")
        
    # Check for Fit Adjustment
    # Nominal 5.0. H7/g6 Hole -> +0.015/+0.0 -> Mid = 5.0 + 0.0075 = 5.0075
    if "5.0075" in glsl:
        print("✓ ISO Fit Applied: 5.0mm -> 5.0075mm (H7/g6 Hole)")
    else:
        print("✗ ISO Fit FAILED (Expected 5.0075)")
        print(glsl)
        
    print("\n[Tolerance Report]")
    print(json.dumps(isa["tolerance_report"], indent=2, default=str))

if __name__ == "__main__":
    verify_hwc_v3()
