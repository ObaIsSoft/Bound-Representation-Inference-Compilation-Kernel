import sys
import os
import trimesh
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from agents.geometry_agent import GeometryAgent

def test_safe_boolean():
    print("Initializing GeometryAgent...")
    agent = GeometryAgent()
    
    # Create primitives
    print("Creating Box and Sphere...")
    box = trimesh.creation.box(extents=[10, 10, 10]) # Centered at 0, 0, 0
    sphere = trimesh.creation.icosphere(radius=6.0) # Centered at 0, 0, 0
    
    # 1. Test Standard Path
    print("\n--- Testing perform_mesh_boolean (Auto) ---")
    try:
        result_auto = agent.perform_mesh_boolean(box, sphere, "DIFFERENCE")
        if result_auto and isinstance(result_auto, trimesh.Trimesh):
            print(f"Auto Result: V={len(result_auto.vertices)}, F={len(result_auto.faces)}")
            print(f"Watertight? {result_auto.is_watertight}")
        else:
            print("Auto Failed (Returned None or Invalid)")
    except Exception as e:
        print(f"Auto Exception: {e}")

    # 2. Test SDF Path Explicitly
    print("\n--- Testing _sdf_boolean (VMK/SDF) ---")
    try:
        # We manually call private method to verify logic
        result_sdf = agent._sdf_boolean(box, sphere, "DIFFERENCE")
        if result_sdf and isinstance(result_sdf, trimesh.Trimesh):
            print(f"SDF Result: V={len(result_sdf.vertices)}, F={len(result_sdf.faces)}")
            print(f"Watertight? {result_sdf.is_watertight}")
            
            # Additional Check: Volume should be < Box Volume (1000)
            print(f"Volume: {result_sdf.volume}")
            if result_sdf.volume < 1000 and result_sdf.volume > 0:
                print("Volume Logic: CORRECT (Material Removed)")
            else:
                print(f"Volume Logic: SUSPICIOUS ({result_sdf.volume})")

        else:
            print("SDF Failed (Returned None)")
            
    except Exception as e:
        print(f"SDF Exception: {e}")

if __name__ == "__main__":
    test_safe_boolean()
