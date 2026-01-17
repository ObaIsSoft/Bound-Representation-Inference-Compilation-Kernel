"""
BRICK OS: Mesh to SDF Bridge
Converts real-world mesh files (STL/OBJ/GLTF) to SDF textures for GPU rendering.
"""

import trimesh
import numpy as np
from skimage import measure
from typing import Tuple, Optional, Dict, Any, List
import hashlib
import json
import os
import logging
import base64
from scipy.ndimage import map_coordinates

logger = logging.getLogger(__name__)


class MeshSDFBridge:
    """
    Converts mesh files to 3D SDF volume textures for seamless integration
    with the SDF kernel (boolean operations, cutaway views, etc.).
    """
    
    def __init__(self, cache_dir: str = "data/sdf_cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
    def bake_sku_to_sdf(
        self,
        mesh_path: str,
        resolution: int = None,
        bounds_padding: float = 0.1,
        use_cache: bool = True
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Converts mesh to 3D SDF volume texture.
        
        Args:
            mesh_path: Path to STL/OBJ/GLTF file
            resolution: Grid resolution (auto-computed if None)
            bounds_padding: Extra space around mesh (0.1 = 10%)
            use_cache: Whether to use cached SDF if available
            
        Returns:
            (sdf_grid, metadata)
            - sdf_grid: 3D numpy array of signed distances (float32)
            - metadata: {bounds, resolution, format, mesh_hash}
        """
        # Check cache first
        if use_cache:
            cache_key = self._get_cache_key(mesh_path, resolution, bounds_padding)
            cached = self._load_from_cache(cache_key)
            if cached is not None:
                logger.info(f"Loaded SDF from cache: {cache_key}")
                return cached
        
        # 1. Load mesh
        logger.info(f"Loading mesh: {mesh_path}")
        mesh = trimesh.load(mesh_path)
        
        # Handle multi-mesh GLTF/scene
        if isinstance(mesh, trimesh.Scene):
            mesh = mesh.dump(concatenate=True)
        
        # 2. Auto-resolution based on complexity
        if resolution is None:
            resolution = self._adaptive_resolution(mesh)
            logger.info(f"Auto-selected resolution: {resolution}")
            
        # 3. Compute SDF grid
        logger.info(f"Computing SDF grid (resolution={resolution})...")
        sdf_grid = self._compute_sdf_grid(mesh, resolution, bounds_padding)
        
        # 4. Generate metadata
        metadata = {
            "bounds": mesh.bounds.tolist(),
            "resolution": int(resolution),
            "format": "float32",
            "mesh_path": mesh_path,
            "face_count": len(mesh.faces),
            "is_watertight": mesh.is_watertight
        }
        
        # Update 3.5: Verify Tolerance
        tolerance_report = self.verify_tolerance(mesh, sdf_grid, resolution, bounds_padding)
        metadata["tolerance_analysis"] = tolerance_report
        metadata["verified"] = tolerance_report["verified"]
        
        # 5. Cache result
        if use_cache:
            self._save_to_cache(cache_key, sdf_grid, metadata)
        
        return sdf_grid, metadata

    def bake_scene_to_atlas(
        self,
        scene_path: str,
        resolution: int = 256,
        padding: float = 0.1
    ) -> Dict[str, Any]:
        """
        Bakes a multi-component scene into a single 3D Texture Atlas.
        
        Args:
            scene_path: Path to GLTF/OBJ scene file
            resolution: Size of the Atlas texture (e.g. 256^3)
            padding: Padding around individual components
            
        Returns:
            Dict containing:
            - texture_data: base64 encoded atlas
            - manifest: List of components with atlas offsets/scales
            - resolution: Atlas resolution
            - glsl: Shader code for multi-component sampling
        """
        logger.info(f"Loading scene for Atlas: {scene_path}")
        scene_or_mesh = trimesh.load(scene_path)
        
        # Normalize to list of (name, mesh, transform)
        components = []
        if isinstance(scene_or_mesh, trimesh.Scene):
            for name, geom in scene_or_mesh.geometry.items():
                # Find transform in graph
                # For simplicity, we assume flat list or baked transforms in geom
                components.append({
                    "id": name,
                    "mesh": geom,
                    "transform": np.eye(4) # geometry is usually already transformed in trimesh scene dump
                })
        else:
             components.append({"id": "main", "mesh": scene_or_mesh, "transform": np.eye(4)})

        logger.info(f"Found {len(components)} components for Atlas packing")

        # 1. Pack Meshes into 3D Atlas
        # We assign each component a sub-region in the NxNxN grid.
        # Simple Strategy: Split grid into uniform slots (e.g. 2x2x2 = 8 slots)
        # Advanced Strategy: Shelf Packing (TODO for optimization)
        
        # For MVP: Simple Grid Packing
        # If we have N components, we need ceil(cbrt(N)) grid cells per axis
        grid_dim = int(np.ceil(len(components)**(1/3)))
        slot_res = resolution // grid_dim
        
        logger.info(f"Atlas Grid: {grid_dim}x{grid_dim}x{grid_dim} (Slot size: {slot_res}^3)")
        
        atlas_grid = np.ones((resolution, resolution, resolution), dtype=np.float32) * 100.0 # Init with large distance
        manifest = []
        
        idx = 0
        for z in range(grid_dim):
            for y in range(grid_dim):
                for x in range(grid_dim):
                    if idx >= len(components):
                        break
                        
                    comp = components[idx]
                    mesh = comp["mesh"]
                    
                    # Offsets in Atlas Grid
                    x_start, y_start, z_start = x * slot_res, y * slot_res, z * slot_res
                    
                    # Compute SDF for this specific component at slot_res
                    # Note: We bake it LOCAL to the mesh (centered)
                    local_sdf = self._compute_sdf_grid(mesh, slot_res, padding)
                    
                    # Write to Atlas
                    atlas_grid[x_start:x_start+slot_res, y_start:y_start+slot_res, z_start:z_start+slot_res] = local_sdf
                    
                    # Record Manifest
                    # UVW Offsets (0-1)
                    uv_offset = [x_start/resolution, y_start/resolution, z_start/resolution]
                    uv_scale = [slot_res/resolution, slot_res/resolution, slot_res/resolution]
                    
                    manifest.append({
                        "id": comp["id"],
                        "atlas_offset": uv_offset,
                        "atlas_scale": uv_scale,
                        "local_bounds": mesh.bounds.tolist(),
                        "sdf_range": [float(local_sdf.min()), float(local_sdf.max())]
                    })
                    
                    # Optional: Verify individual components?
                    # For performance, maybe only verify if requested, or verify the 'main' one.
                    # Let's run a quick verification on the component locally.
                    # Reuse the same validation logic.
                    # tolerance_report = self.verify_tolerance(mesh, local_sdf, slot_res, padding)
                    # manifest[-1]["tolerance_analysis"] = tolerance_report
                    
                    idx += 1
                    
        # 2. Normalize & Encode Atlas
        sdf_min = float(atlas_grid.min())
        sdf_max = float(atlas_grid.max())
        
        # Avoid div by zero
        if abs(sdf_max - sdf_min) < 1e-6:
            normalized = np.zeros_like(atlas_grid)
        else:
            normalized = (atlas_grid - sdf_min) / (sdf_max - sdf_min)
            
        texture_bytes = normalized.astype(np.float32).tobytes()
        texture_b64 = base64.b64encode(texture_bytes).decode('ascii')
        
        # 3. Generate Multi-Component GLSL
        glsl_code = self._generate_atlas_glsl(len(components))
        
        return {
            "texture_data": texture_b64,
            "manifest": manifest,
            "resolution": resolution,
            "sdf_range": [sdf_min, sdf_max],
            "glsl": glsl_code,
            "component_count": len(components)
        }
        
    def _adaptive_resolution(self, mesh: trimesh.Trimesh) -> int:
        """
        Determine grid resolution based on mesh complexity.
        
        Higher polygon count → higher resolution needed to capture detail.
        Memory usage scales as O(resolution^3), so be conservative.
        """
        face_count = len(mesh.faces)
        
        if face_count < 1000:
            return 32  # Low-poly (32KB for float32)
        elif face_count < 10000:
            return 64  # Medium (1MB)
        elif face_count < 100000:
            return 128 # High (8MB)
        else:
            return 256 # Very high (64MB) - only for complex meshes
            
    def _compute_sdf_grid(
        self,
        mesh: trimesh.Trimesh,
        resolution: int,
        padding: float
    ) -> np.ndarray:
        """
        Generate 3D SDF grid using trimesh.proximity.signed_distance.
        
        Returns:
            3D numpy array where negative values = inside mesh.
        """
        # Ensure watertight for proper SDF
        if not mesh.is_watertight:
            logger.warning("Mesh not watertight, attempting repair...")
            trimesh.repair.fill_holes(mesh)
            trimesh.repair.fix_normals(mesh)
            
        # Define grid bounds with padding
        bounds = mesh.bounds
        extent = bounds[1] - bounds[0]
        min_p = bounds[0] - extent * padding
        max_p = bounds[1] + extent * padding
        
        # Create uniform grid points
        x = np.linspace(min_p[0], max_p[0], resolution)
        y = np.linspace(min_p[1], max_p[1], resolution)
        z = np.linspace(min_p[2], max_p[2], resolution)
        
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        pts = np.column_stack((X.flatten(), Y.flatten(), Z.flatten()))
        
        # Compute signed distance (negative inside, positive outside)
        logger.info(f"Computing SDF for {len(pts)} points...")
        sdf_flat = trimesh.proximity.signed_distance(mesh, pts)
        sdf_grid = sdf_flat.reshape(resolution, resolution, resolution).astype(np.float32)
        
        logger.info(f"SDF range: [{sdf_grid.min():.3f}, {sdf_grid.max():.3f}]")
        
        return sdf_grid
        
        return sdf_grid

    def verify_tolerance(
        self,
        mesh: trimesh.Trimesh,
        sdf_grid: np.ndarray,
        resolution: int,
        padding: float
    ) -> Dict[str, Any]:
        """
        Verify SDF accuracy by sampling points on the mesh surface.
        A perfect SDF should be exactly 0.0 at all surface points.
        
        Args:
            mesh: Original mesh object
            sdf_grid: Generated SDF volume
            resolution: Grid resolution used
            padding: Padding factor used
            
        Returns:
            Dict containing error metrics (mean, max, percent_within_tolerance)
        """
        # 1. Sample sample points on surface
        SAMPLE_COUNT = 1000
        # trimesh.sample.sample_surface returns (points, face_indices)
        points, _ = trimesh.sample.sample_surface(mesh, SAMPLE_COUNT)
        
        # 2. Map world points to grid coordinates
        bounds = mesh.bounds
        extent = bounds[1] - bounds[0]
        min_p = bounds[0] - extent * padding
        max_p = bounds[1] + extent * padding
        
        # Avoid division by zero for flat meshes
        step = (max_p - min_p)
        step[step < 1e-6] = 1.0 
        
        # Normalized [0, 1]
        norm_pts = (points - min_p) / step
        
        # Grid coords [0, resolution-1]
        # Note: numpy coords are (x, y, z) but map_coordinates expects (dim0, dim1, dim2)
        # matching meshgrid indexing='ij' which is x,y,z
        grid_coords = norm_pts * (resolution - 1)
        
        # 3. Sample SDF at these coordinates
        # map_coordinates expects shape (ndim, n_points) -> (3, 1000)
        sampled_sdf_values = map_coordinates(
            sdf_grid, 
            grid_coords.T, 
            order=1, # Linear interpolation
            mode='nearest'
        )
        
        # 4. Compute Statistics
        # Values should be 0. Real values will be small floats.
        abs_errors = np.abs(sampled_sdf_values)
        
        mean_error = float(np.mean(abs_errors))
        max_error = float(np.max(abs_errors))
        std_error = float(np.std(abs_errors))
        
        # Tolerance check (e.g. within 1% of bounding box diagonal)
        diag = np.linalg.norm(extent)
        tolerance_thresh = diag * 0.01 # 1% tolerance
        passing_count = np.sum(abs_errors < tolerance_thresh)
        pass_rate = float(passing_count / SAMPLE_COUNT)
        
        logger.info(f"Tolerance Validation: Mean Err={mean_error:.4f}, Pass Rate={pass_rate*100:.1f}%")
        
        return {
            "mean_error": mean_error,
            "max_error": max_error,
            "std_error": std_error,
            "tolerance_threshold": tolerance_thresh,
            "pass_rate": pass_rate,
            "verified": (pass_rate > 0.95), # Pass if 95% of points are good
            "sample_count": SAMPLE_COUNT
        }

    def generate_glsl_sampler(
        self,
        sdf_grid: np.ndarray,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate GLSL code for SDF texture sampling.
        
        Returns:
            {
                "glsl": GLSL function code,
                "texture_data": base64-encoded float32 texture,
                "resolution": grid resolution,
                "sdf_range": [min, max],
                "bounds": [[minx,miny,minz], [maxx,maxy,maxz]]
            }
        """
        import base64
        
        # Get SDF range for denormalization
        sdf_min = float(sdf_grid.min())
        sdf_max = float(sdf_grid.max())
        
        # Normalize to [0, 1] for texture storage
        if sdf_max > sdf_min:
            normalized = (sdf_grid - sdf_min) / (sdf_max - sdf_min)
        else:
            normalized = np.zeros_like(sdf_grid)
        
        # Convert to bytes (float32)
        texture_bytes = normalized.astype(np.float32).tobytes()
        texture_b64 = base64.b64encode(texture_bytes).decode('ascii')
        
        # Generate GLSL sampling function
        bounds = metadata["bounds"]
        glsl_code = f"""
// Mesh SDF Sampler (Auto-generated from {os.path.basename(metadata['mesh_path'])})
// Resolution: {metadata['resolution']}^3, Faces: {metadata['face_count']}

uniform sampler3D u_meshSDFTexture;
uniform vec3 u_meshBounds[2];  // [min, max]
uniform float u_meshSDF_min;   // {sdf_min}
uniform float u_meshSDF_max;   // {sdf_max}
float sampleMeshSDF(vec3 p) {{
    // Transform world coords to texture UVW [0,1]
    vec3 uvw = (p - u_meshBounds[0]) / (u_meshBounds[1] - u_meshBounds[0]);
    
    // Bounds check (outside texture = large positive distance)
    if (any(lessThan(uvw, vec3(0.0))) || any(greaterThan(uvw, vec3(1.0)))) {{
        return 1e10;
    }}
    
    // Sample normalized value [0,1]
    float normalized = texture(u_meshSDFTexture, uvw).r;
    
    // Denormalize to actual SDF value
    return normalized * (u_meshSDF_max - u_meshSDF_min) + u_meshSDF_min;
}}
"""
        return {
            "glsl": glsl_code,
            "texture_data": texture_b64,
            "resolution": int(metadata["resolution"]),
            "sdf_range": [sdf_min, sdf_max],
            "bounds": bounds
        }

    def _generate_atlas_glsl(self, component_count: int) -> str:
        """Generates GLSL for iterating through the Atlas Manifest."""
        return f"""
#define MAX_COMPONENTS {max(16, component_count)}

struct ComponentSDF {{
    vec3 atlasOffset;
    vec3 atlasScale;
    vec3 localBounds[2];
    vec2 sdfRange; // min, max
    mat4 transform; // World to Local
}};

uniform sampler3D uAtlasTexture;
uniform ComponentSDF uComponents[MAX_COMPONENTS];
uniform int uComponentCount;

float sampleAtlasSDF(vec3 p) {{
    float d_min = 1e10;
    
    // Loop through all active components
    for(int i=0; i < MAX_COMPONENTS; i++) {{
        if(i >= uComponentCount) break;
        
        // 1. Transform World to Component Local Space
        // Note: In shader, passed matrices are usually WorldToLocal
        vec4 localP_4 = uComponents[i].transform * vec4(p, 1.0);
        vec3 localP = localP_4.xyz; // Assuming rigid scale/rotation
        
        // 2. Check Component Local Bounds
        vec3 bMin = uComponents[i].localBounds[0];
        vec3 bMax = uComponents[i].localBounds[1];
        
        // Optimization: AABB check
        if(any(lessThan(localP, bMin)) || any(greaterThan(localP, bMax))) {{
            // Skip expensive texture sample if far outside
            // But for SDF blending we need distance. 
            // Approximation: distance to box
            vec3 d = max(abs(localP - (bMin+bMax)*0.5) - (bMax-bMin)*0.5, 0.0);
            float distToBox = length(d);
            d_min = min(d_min, distToBox); 
            continue; 
        }}
        
        // 3. Map Local Pos to Atlas UVW
        // Normalized local [0,1]
        vec3 uv_local = (localP - bMin) / (bMax - bMin);
        
        // Map to Atlas UV
        vec3 uv_atlas = uComponents[i].atlasOffset + uv_local * uComponents[i].atlasScale;
        
        // 4. Sample Atlas
        float norm_val = texture(uAtlasTexture, uv_atlas).r;
        float real_dist = norm_val * (uComponents[i].sdfRange.y - uComponents[i].sdfRange.x) + uComponents[i].sdfRange.x;
        
        // 5. Union (min)
        // Using slight smoothing could be nice, but hard min for now
        d_min = min(d_min, real_dist);
    }}
    
    return d_min;
}}
"""
        
    def _get_cache_key(self, mesh_path: str, resolution: int, padding: float) -> str:
        """Generate cache key from mesh file + parameters."""
        # Hash file contents + params
        with open(mesh_path, 'rb') as f:
            file_hash = hashlib.md5(f.read()).hexdigest()[:8]
        
        param_str = f"{resolution}_{padding}"
        return f"{file_hash}_{param_str}"
        
    def _save_to_cache(self, cache_key: str, sdf_grid: np.ndarray, metadata: Dict):
        """Save SDF grid and metadata to cache."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npz")
        meta_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        # Save SDF as compressed numpy
        np.savez_compressed(cache_path, sdf=sdf_grid)
        
        # Save metadata as JSON
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        logger.info(f"Cached SDF to {cache_path}")
        
    def _load_from_cache(self, cache_key: str) -> Optional[Tuple[np.ndarray, Dict]]:
        """Load SDF grid and metadata from cache if exists."""
        cache_path = os.path.join(self.cache_dir, f"{cache_key}.npz")
        meta_path = os.path.join(self.cache_dir, f"{cache_key}.json")
        
        if os.path.exists(cache_path) and os.path.exists(meta_path):
            # Load SDF
            data = np.load(cache_path)
            sdf_grid = data['sdf']
            
            # Load metadata
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
                
            return sdf_grid, metadata
        
        return None
