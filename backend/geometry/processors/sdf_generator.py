import numpy as np
import trimesh
import logging
import io
import struct

logger = logging.getLogger(__name__)

def generate_sdf_volume(mesh: trimesh.Trimesh, resolution: int = 64, padding: float = 0.1) -> bytes:
    """
    Computes a 3D Signed Distance Field (SDF) grid for the given mesh.
    
    Args:
        mesh: The input trimesh object.
        resolution: Grid resolution (N x N x N). Warning: O(N^3) complexity.
        padding: Percentage of padding to add around the mesh bounding box.
        
    Returns:
        bytes: Binary float32 buffer of the SDF values (flattened).
    """
    try:
        # Import the new robust voxelizer
        from .mesh_voxelizer import MeshVoxelizer
        
        logger.info(f"Generating SDF for mesh with {len(mesh.vertices)} vertices")
        
        # Initialize robust voxelizer
        voxelizer = MeshVoxelizer(resolution=resolution, padding_factor=padding)
        
        # Run voxelization (using robust winding number by default)
        sdf_grid, bounds = voxelizer.voxelize(
            mesh.vertices,
            mesh.faces,
            use_winding_number=True  # Force robust mode
        )
        
        # Flatten and ensure float32
        sdf_data = sdf_grid.flatten().astype(np.float32)
        
        logger.info(f"SDF Generation Complete. Range: [{sdf_data.min():.3f}, {sdf_data.max():.3f}]")
        
        return sdf_data.tobytes()

    except Exception as e:
        logger.error(f"SDF Generation Failed: {e}", exc_info=True)
        # Fallback to empty bytes or re-raise
        raise e
        
    except Exception as e:
        logger.error(f"SDF Generation Failed: {e}")
        # Return empty bytes or raise
        raise e
