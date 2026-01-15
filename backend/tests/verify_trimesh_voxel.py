import trimesh
import numpy as np

def create_non_watertight_mesh():
    # A cube with one face missing
    vertices = [
        [0,0,0], [1,0,0], [1,1,0], [0,1,0],
        [0,0,1], [1,0,1], [1,1,1], [0,1,1]
    ]
    # Missing top and bottom faces for sure
    faces = [
        [0,1,5], [0,5,4], # Front
        [1,2,6], [1,6,5], # Right
        [2,3,7], [2,7,6], # Back
        [3,0,4], [3,4,7]  # Left
    ]
    return trimesh.Trimesh(vertices=vertices, faces=faces)

def test_voxel_repair():
    print("Creating broken mesh...")
    mesh = create_non_watertight_mesh()
    print(f"Is watertight? {mesh.is_watertight}")
    
    print("Attempting Voxelization...")
    try:
        # Voxelize
        voxel = trimesh.voxel.creation.voxelize(mesh, pitch=0.1)
        print(f"Voxel grid created. Filled count: {voxel.filled_count}")
        
        # Convert to mesh (Blocky)
        print("Converting to boxes (as_boxes)...")
        repaired = voxel.as_boxes()
        print(f"Repaired Is Watertight? {repaired.is_watertight}")
        print(f"Repaired Faces: {len(repaired.faces)}")
        
        # Check if marching cubes works (expect failure)
        print("Attempting Marching Cubes (marching_cubes)...")
        try:
            smooth = voxel.marching_cubes
            print(f"Marching Cubes Success! Watertight? {smooth.is_watertight}")
        except Exception as e:
            print(f"Marching Cubes Failed: {e}")

    except Exception as e:
        print(f"Voxelization Failed: {e}")

if __name__ == "__main__":
    test_voxel_repair()
