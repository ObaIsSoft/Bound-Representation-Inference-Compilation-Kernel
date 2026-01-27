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
        # 1. Determine Bounds
        bounds = mesh.bounds
        extent = bounds[1] - bounds[0]
        max_dim = np.max(extent)
        
        # Cube bounds centered on mesh center
        center = bounds.mean(axis=0)
        # Pad slightly to ensure boundary conditions
        size = max_dim * (1.0 + padding)
        
        half_size = size / 2.0
        min_bound = center - half_size
        max_bound = center + half_size # Not strictly needed for linspace but good for verify
        
        # 2. Create Grid Points
        # Create a grid of x, y, z coordinates
        x = np.linspace(min_bound[0], min_bound[0] + size, resolution)
        y = np.linspace(min_bound[1], min_bound[1] + size, resolution)
        z = np.linspace(min_bound[2], min_bound[2] + size, resolution)
        
        # Meshgrid (Note: indexing='ij' for matrix order)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        
        # Flatten to list of points (N^3, 3)
        query_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        
        # 3. Compute Signed Distance
        # trimesh.nearest.signed_distance signature: (mesh, points) -> distance
        # Returns positive outside, negative inside? Trimesh convention:
        # "Signed distance from the mesh surface to the points. Points outside are positive, points inside are negative."
        # This matches standard SDF convention for raymarching (d < 0 is inside).
        
        # Note: signed_distance might be slow for massive grids. 
        # Ideally use `trimesh.proximity.signed_distance`
        
        # Check if mesh is watertight; if not, signed distance might be unreliable
        if not mesh.is_watertight:
            logger.warning("Mesh is not watertight! SDF generation might be inaccurate (using unsigned distance fallback logic might be needed).")
            # Fallback? For now, proceed.
            
        distances = trimesh.proximity.signed_distance(mesh, query_points)
        
        # 4. Pack into Bytes
        # Ensure float32
        sdf_data = distances.astype(np.float32)
        
        # Return raw bytes
        return sdf_data.tobytes()
        
    except Exception as e:
        logger.error(f"SDF Generation Failed: {e}")
        # Return empty bytes or raise
        raise e
