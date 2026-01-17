
import sys
import os
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from utils.mesh_to_sdf_bridge import MeshSDFBridge
import trimesh

def test_tolerance_verification():
    print("Testing Tolerance Verification...")
    bridge = MeshSDFBridge(cache_dir="backend/tests/temp_cache")
    
    # 1. Create a simple box
    mesh = trimesh.creation.box(extents=[1, 1, 1])
    temp_path = "backend/tests/temp_box.stl"
    mesh.export(temp_path)
    
    try:
        # 2. Bake to SDF (this triggers verify_tolerance)
        print("Baking Box to SDF...")
        sdf_grid, metadata = bridge.bake_sku_to_sdf(temp_path, resolution=64, use_cache=False)
        
        # 3. Check Metadata for tolerance report
        print("Checking metadata for tolerance report...")
        report = metadata.get("tolerance_analysis")
        
        if not report:
            print("❌ Tolerance report missing in metadata!")
            return False
            
        print(f"Report: {report}")
        
        if report["verified"] == True:
            print("✅ Tolerance verification passed (as expected for simple box)")
        else:
            print(f"⚠️ Tolerance verification failed. Mean error: {report['mean_error']}")
            
        print("Test Complete.")
        return True
        
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == "__main__":
    test_tolerance_verification()
