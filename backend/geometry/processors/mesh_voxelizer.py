
import numpy as np
from scipy.spatial import cKDTree
import logging
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

class MeshVoxelizer:
    """
    Convert triangle mesh to signed distance field (SDF).
    
    Uses generalized winding number for robust inside/outside determination,
    even for non-watertight meshes.
    """
    
    def __init__(self, resolution: int = 64, padding_factor: float = 0.1):
        """
        Initialize voxelizer.
        
        Args:
            resolution: Grid resolution (64 = 64x64x64 grid).
            padding_factor: Extra space around mesh (0.1 = 10% padding).
        """
        self.resolution = resolution
        self.padding_factor = padding_factor
        
    def voxelize(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray,
        use_winding_number: bool = True
    ) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Convert mesh to signed distance field.
        """
        logger.info(f"Voxelizing mesh: {len(vertices)} vertices, {len(faces)} faces")
        
        # Step 1: Compute bounds with padding
        min_xyz, max_xyz = self._compute_bounds(vertices)
        
        # Step 2: Create 3D grid of sample points
        grid_points = self._create_grid(min_xyz, max_xyz)
        
        # Step 3: Compute unsigned distance (closest point on surface)
        unsigned_distances = self._compute_unsigned_distance(vertices, faces, grid_points)
        
        # Step 4: Determine sign (inside vs outside)
        if use_winding_number:
            signs = self._compute_sign_winding_number(grid_points, vertices, faces)
        else:
            signs = self._compute_sign_ray_casting(grid_points, vertices, faces)
        
        # Step 5: Combine into signed distance field
        sdf = signs * unsigned_distances
        sdf_grid = sdf.reshape(self.resolution, self.resolution, self.resolution)
        
        return sdf_grid, (min_xyz, max_xyz)
    
    def _compute_bounds(self, vertices: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute bounding box with padding."""
        min_xyz = np.min(vertices, axis=0)
        max_xyz = np.max(vertices, axis=0)
        
        # Add padding
        extent = max_xyz - min_xyz
        padding = self.padding_factor * extent
        
        min_xyz -= padding
        max_xyz += padding
        
        return min_xyz, max_xyz
    
    def _create_grid(self, min_xyz: np.ndarray, max_xyz: np.ndarray) -> np.ndarray:
        """Create uniform 3D grid of sample points."""
        res = self.resolution
        
        x = np.linspace(min_xyz[0], max_xyz[0], res)
        y = np.linspace(min_xyz[1], max_xyz[1], res)
        z = np.linspace(min_xyz[2], max_xyz[2], res)
        
        # Create meshgrid using indexing='ij' for matrix order
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing='ij')
        # Flatten: (N, 3) 
        grid_points = np.stack([grid_x.ravel(), grid_y.ravel(), grid_z.ravel()], axis=1)
        
        return grid_points
    
    def _compute_unsigned_distance(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray,
        grid_points: np.ndarray
    ) -> np.ndarray:
        """
        Compute unsigned distance from grid points to mesh surface.
        Uses KD-tree on uniformly sampled surface points for efficiency.
        """
        # Sample points on mesh surface
        n_samples = min(50000, len(faces) * 10)  # Adaptive sampling limit
        surface_points = self._sample_surface(vertices, faces, n_samples)
        
        # Build spatial index
        tree = cKDTree(surface_points)
        
        # Query nearest surface point for each grid point
        distances, _ = tree.query(grid_points, k=1)
        
        return distances
    
    def _sample_surface(
        self, 
        vertices: np.ndarray, 
        faces: np.ndarray, 
        n_samples: int
    ) -> np.ndarray:
        """Uniformly sample points on triangle mesh surface."""
        triangles = vertices[faces]  # (M, 3, 3)
        
        # Compute triangle areas
        v0, v1, v2 = triangles[:, 0], triangles[:, 1], triangles[:, 2]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        
        # Handle degenerate triangles
        areas = np.maximum(areas, 1e-10)
        
        # Sample triangles proportional to area
        probabilities = areas / areas.sum()
        chosen_faces = np.random.choice(len(faces), size=n_samples, p=probabilities, replace=True)
        
        # Uniform barycentric sampling
        r1 = np.random.random(n_samples)
        r2 = np.random.random(n_samples)
        
        sqrt_r1 = np.sqrt(r1)
        u = 1 - sqrt_r1
        v = sqrt_r1 * (1 - r2)
        w = sqrt_r1 * r2
        
        # Compute positions
        samples = []
        # Optimization: Vectorize this loop
        # Extract chosen triangles
        chosen_tris = triangles[chosen_faces] # (N_samples, 3, 3)
        # Combine barycentric weights
        # p = u*v0 + v*v1 + w*v2
        # u, v, w are (N_samples,)
        # tris are (N_samples, 3, 3)
        samples = (
            chosen_tris[:, 0] * u[:, np.newaxis] +
            chosen_tris[:, 1] * v[:, np.newaxis] +
            chosen_tris[:, 2] * w[:, np.newaxis]
        )
        
        return samples
    
    def _compute_sign_winding_number(
        self,
        grid_points: np.ndarray,
        vertices: np.ndarray,
        faces: np.ndarray
    ) -> np.ndarray:
        """
        Determine inside/outside using generalized winding number.
        Robust to non-watertight meshes.
        """
        triangles = vertices[faces]  # (M, 3, 3)
        n_points = len(grid_points)
        
        # Batch processing to manage memory
        batch_size = 2000 # Smaller batch to prevent OOM
        winding_numbers = np.zeros(n_points)
        
        for i in range(0, n_points, batch_size):
            batch_end = min(i + batch_size, n_points)
            batch_points = grid_points[i:batch_end]
            batch_winding = self._winding_number_batch(batch_points, triangles)
            winding_numbers[i:batch_end] = batch_winding
            
        # Inside if winding number > 0.5 (approx)
        signs = np.where(winding_numbers > 0.5, -1.0, 1.0)
        
        return signs
    
    def _winding_number_batch(
        self,
        points: np.ndarray,
        triangles: np.ndarray
    ) -> np.ndarray:
        """
        Compute winding number for a batch of points.
        Returns array of shape (N_points,)
        """
        # Expand dimensions for broadcasting
        # points: (N, 3) -> (N, 1, 3)
        points_expanded = points[:, np.newaxis, :]
        # triangles: (M, 3, 3) -> (1, M, 3, 3)
        triangles_expanded = triangles[np.newaxis, :, :, :]
        
        # Vectors from points to vertices
        # v0, v1, v2 shape: (N, M, 3)
        v0 = triangles_expanded[:, :, 0, :] - points_expanded
        v1 = triangles_expanded[:, :, 1, :] - points_expanded
        v2 = triangles_expanded[:, :, 2, :] - points_expanded
        
        # Norms
        d0 = np.linalg.norm(v0, axis=2) + 1e-10
        d1 = np.linalg.norm(v1, axis=2) + 1e-10
        d2 = np.linalg.norm(v2, axis=2) + 1e-10
        
        # Normalize vectors manually
        # v0 / d0[..., newaxis]
        v0_norm = v0 / d0[..., np.newaxis]
        v1_norm = v1 / d1[..., np.newaxis]
        v2_norm = v2 / d2[..., np.newaxis]
        
        # Compute Solid Angle (L'Huilier's theorem)
        # Numerator: v0 . (v1 x v2) --> scalar triple product
        cross_v1_v2 = np.cross(v1_norm, v2_norm, axis=2)
        numerator = np.sum(v0_norm * cross_v1_v2, axis=2) # (N, M)
        
        # Denominator: 1 + v0.v1 + v1.v2 + v2.v0
        dot_v0_v1 = np.sum(v0_norm * v1_norm, axis=2)
        dot_v1_v2 = np.sum(v1_norm * v2_norm, axis=2)
        dot_v2_v0 = np.sum(v2_norm * v0_norm, axis=2)
        
        denominator = 1.0 + dot_v0_v1 + dot_v1_v2 + dot_v2_v0
        
        solid_angles = 2.0 * np.arctan2(numerator, denominator + 1e-10) # (N, M)
        
        # Sum over all triangles for each point
        total_angle = np.sum(solid_angles, axis=1) # (N,)
        
        return total_angle / (4.0 * np.pi)

    def _compute_sign_ray_casting(self, grid_points, vertices, faces):
        """Fallback for perfectly watertight meshes (faster)."""
        # (Simplified implementation or just error out since we rely on winding)
        # Using trimesh's ray casting if available is better, but here assuming standalone.
        # For safety in this specific "Robustness" task, we default to Winding Number.
        logger.warning("Ray casting requested but defaulting to Winding Number for safety.")
        return self._compute_sign_winding_number(grid_points, vertices, faces)
