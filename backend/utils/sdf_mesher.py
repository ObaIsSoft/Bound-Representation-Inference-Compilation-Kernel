import numpy as np
from skimage import measure
import trimesh
import logging

logger = logging.getLogger(__name__)

def sdf_box(p, b):
    """SDF for a Box of dim b [x, y, z] centered at origin."""
    q = np.abs(p) - b
    return np.linalg.norm(np.maximum(q, 0.0), axis=1) + np.minimum(np.max(q, axis=1), 0.0)

def sdf_sphere(p, r):
    return np.linalg.norm(p, axis=1) - r

def generate_mesh_from_sdf(sdf_func, bounds, resolution=64):
    """
    Generate a mesh from an SDF function using Marching Cubes.
    
    Args:
        sdf_func: Function accepting (N,3) points -> (N,) distances.
        bounds: Tuple ((min_x, min_y, min_z), (max_x, max_y, max_z))
        resolution: Grid resolution (integer).
        
    Returns:
        trimesh.Trimesh object or None if empty.
    """
    
    # 1. Create Grid
    min_b, max_b = bounds
    x = np.linspace(min_b[0], max_b[0], resolution)
    y = np.linspace(min_b[1], max_b[1], resolution)
    z = np.linspace(min_b[2], max_b[2], resolution)
    
    # Meshgrid (Note: 'ij' indexing for marching cubes usually prefers consistent ordering)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Flatten for query
    points = np.column_stack([X.ravel(), Y.ravel(), Z.ravel()])
    
    # 2. Query SDF
    try:
        # Batch processing might be needed for very high res, but 64^3 is small (262k points)
        distances = sdf_func(points)
        vol = distances.reshape(X.shape)
    except Exception as e:
        logger.error(f"SDF Query Failed: {e}")
        return None
        
    # 3. Marching Cubes
    try:
        # level=0.0 is the isosurface
        verts, faces, normals, values = measure.marching_cubes(vol, level=0.0)
        
        # Scale/Transform vertices back to world space
        # verts are in grid coords (0..resolution-1)
        
        # Normalize 0..1
        verts_norm = verts / (resolution - 1)
        
        # Scale to bounds size
        size = np.array(max_b) - np.array(min_b)
        verts_world = (verts_norm * size) + np.array(min_b)
        
        # Create Trimesh
        mesh = trimesh.Trimesh(vertices=verts_world, faces=faces, vertex_normals=normals)
        return mesh
        
    except ValueError:
        # Raised if no surface found (all inside or all outside)
        logger.warning("Marching Cubes found no surface (empty or solid).")
        return None

def simple_test_mesh():
    """Generates a test sphere mesh."""
    def sphere_sdf(p):
        return sdf_sphere(p, 1.0)
    
    bounds = ([-2,-2,-2], [2,2,2])
    return generate_mesh_from_sdf(sphere_sdf, bounds, resolution=32)
