
import trimesh
import numpy as np
from backend.geometry.processors.sdf_generator import generate_sdf_volume
import struct

def test_sdf_gen():
    print("Creating Sphere Mesh...")
    mesh = trimesh.creation.icosphere(radius=1.0)
    
    print("Generating SDF Volume (32^3)...")
    sdf_bytes = generate_sdf_volume(mesh, resolution=32, padding=0.2)
    
    print(f"SDF Data Size: {len(sdf_bytes)} bytes")
    
    # Expected size: 32*32*32 * 4 (float32) = 32768 * 4 = 131072
    expected = 32**3 * 4
    if len(sdf_bytes) == expected:
        print("✅ Size matches expected float32 buffer.")
    else:
        print(f"❌ Size Mismatch! Expected {expected}, got {len(sdf_bytes)}")

    # Decode and check center value
    # Center of grid should be inside sphere => negative distance
    # With 32^3, the middle index is roughly 15,15,15
    floats = np.frombuffer(sdf_bytes, dtype=np.float32)
    
    # Check bounds
    min_dist = np.min(floats)
    max_dist = np.max(floats)
    
    print(f"Min Distance: {min_dist} (Should be negative)")
    print(f"Max Distance: {max_dist} (Should be positive)")
    
    if min_dist < 0 and max_dist > 0:
        print("✅ SDF Values span Inside/Outside.")
    else:
        print("❌ SDF Values Invalid (All positive or all negative).")

if __name__ == "__main__":
    test_sdf_gen()
