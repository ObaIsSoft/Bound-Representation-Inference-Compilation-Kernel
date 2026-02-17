import logging
import io
import numpy as np
from typing import List, Dict
import manifold3d
from .base_engine import BaseGeometryEngine, GeometryRequest, GeometryResult
from .processors.sdf_generator import generate_sdf_volume # Phase 17

logger = logging.getLogger(__name__)

class ManifoldEngine(BaseGeometryEngine):
    """
    HOT PATH ENGINE
    Uses Manifold3D (C++ Bindings) for <100ms mesh generation.
    Primary Output: GLB (Binary GLTF) for Frontend.
    """
    
    def build(self, request: GeometryRequest) -> GeometryResult:
        try:
            # 1. Compose Geometry
            final_manifold = self._compose_tree(request.tree)
            
            # 2. Export
            if request.output_format == "glb" or request.output_format == "gltf":
                 # Use Trimesh for export wrapper or manifold's export?
                 # Manifold has limited export, usually returns mesh data.
                 # Let's verify if we need to bridge to Trimesh for generic export.
                 # Manifold3D generic steps: Mesh -> Export
                 
                 mesh = final_manifold.to_mesh()
                 
                 # Optimization: Direct Mesh -> GLB bytes
                 # For now, wrap in minimal Trimesh to perform robust export
                 import trimesh
                 # Manifold3D API uses vert_properties for vertex positions
                 t_mesh = trimesh.Trimesh(
                     vertices=np.array(mesh.vert_properties, dtype=np.float32),
                     faces=np.array(mesh.tri_verts, dtype=np.int32)
                 )
                 
                 # Fix for Manifold3D <-> Trimesh data structures
                 # Manifold 3.0+ usually has specific property accessors
                 
                 out_bytes = t_mesh.export(file_type="glb")
                 return GeometryResult(success=True, payload=out_bytes)
                 
            elif request.output_format == "stl":
                 # TODO: Similar flow
                 pass

            elif request.output_format == "sdf_grid":
                # Phase 17: Full SDF Support
                # Manifold -> Mesh -> SDF Grid
                mesh = final_manifold.to_mesh()
                import trimesh
                # Manifold3D v2+ API uses vert_pos, not vert_properties
                t_mesh = trimesh.Trimesh(
                    vertices=np.array(mesh.vert_pos, dtype=np.float32),
                    faces=np.array(mesh.tri_verts, dtype=np.int32)
                )
                
                # Check for resolution param, default 64
                res = request.parameters.get("resolution", 64)
                
                sdf_bytes = generate_sdf_volume(t_mesh, resolution=res)
                
                # Return payload. 
                # Note: The frontend needs metadata (bounds, resolution) to interpret this.
                # We can prepend metadata or wrap in a structured format.
                # For now, let's assume the payload is just the bytes, and we pass metadata differently?
                # Actually, BaseGeometryEngine returns payload as bytes.
                # Let's verify how to pass bounds. 
                
                # OPTION: Return a JSON-header + Binary body? 
                # Or just return the bytes and let the Agent construct the response with metadata.
                # The GeometryResult has 'payload' (bytes).
                
                # We will return the raw bytes here. The calling agent (GeometryAgent or ManifoldAgent)
                # should handle bounding box calculation if needed, or we attach it to result.extras.
                
                # Let's add 'extras' to GeometryResult? Base class definition check needed.
                # Assuming simple bytes for now.
                return GeometryResult(success=True, payload=sdf_bytes)
                 
            return GeometryResult(success=False, error=f"Unsupported format for Manifold: {request.output_format}")
            
        except Exception as e:
            logger.error(f"Manifold Engine Error: {e}")
            return GeometryResult(success=False, error=str(e))

    def _compose_tree(self, tree: List[Dict]) -> manifold3d.Manifold:
        """
        Recursively builds the Manifold object.
        """
        # Start with empty? Or first item?
        # Manifold.compose?
        
        # Strategy: Accumulate
        # If tree is a list, assumes implicit UNION?
        
        union_result = None
        
        for node in tree:
            shape = self._create_primitive(node)
            
            # Apply Transform
            transform = node.get("transform")
            if transform:
                shape = self._apply_transform(shape, transform)
                
            if union_result is None:
                union_result = shape
            else:
                op = node.get("operation", "UNION").upper()
                if op == "UNION":
                    union_result += shape
                elif op == "SUBTRACT" or op == "DIFFERENCE":
                    union_result -= shape
                elif op == "INTERSECT":
                    union_result ^= shape # Manifold operator for intersect
                    
        if union_result is None:
             # Return empty/cube?
             return manifold3d.Manifold()
        
        # Validate mesh quality (watertight check via trimesh)
        try:
            import trimesh
            mesh = union_result.to_mesh()
            t_mesh = trimesh.Trimesh(
                vertices=np.array(mesh.vert_properties, dtype=np.float32),
                faces=np.array(mesh.tri_verts, dtype=np.int32)
            )
            if not t_mesh.is_watertight:
                logger.warning("Generated mesh is not watertight - may have holes or self-intersections")
                # Optionally: try to repair or raise error
                # For now, log and continue - caller can check metadata
        except Exception as e:
            logger.debug(f"Could not validate watertightness: {e}")
        
        return union_result

    def _create_primitive(self, node: Dict) -> manifold3d.Manifold:
        """
        Create a primitive shape from node parameters.
        
        Raises:
            ValueError: If required parameters are missing
        """
        ptype = node.get("type", "box")
        params = node.get("params", {})
        
        if ptype == "box":
            # Require explicit dimensions - no defaults
            if "length" not in params or "width" not in params or "height" not in params:
                raise ValueError(f"Box primitive requires 'length', 'width', 'height' in params. Got: {params}")
            x = float(params["length"])
            y = float(params["width"])
            z = float(params["height"])
            return manifold3d.Manifold.cube(np.array([x, y, z]), center=True)
            
        elif ptype == "sphere":
            # Require explicit radius - no defaults
            if "radius" not in params:
                raise ValueError(f"Sphere primitive requires 'radius' in params. Got: {params}")
            r = float(params["radius"])
            return manifold3d.Manifold.sphere(r, circular_segments=32)
            
        elif ptype == "cylinder":
            # Require explicit dimensions - no defaults
            if "height" not in params or "radius" not in params:
                raise ValueError(f"Cylinder primitive requires 'height', 'radius' in params. Got: {params}")
            h = float(params["height"])
            r = float(params["radius"])
            return manifold3d.Manifold.cylinder(h, r, r, circular_segments=32)
             
        else:
            raise ValueError(f"Unknown primitive type: {ptype}. Supported: box, sphere, cylinder")
    
    def _apply_transform(self, shape: manifold3d.Manifold, transform: Dict) -> manifold3d.Manifold:
        """
        Apply transform (translate, rotate, scale) to a Manifold shape.
        
        Args:
            shape: Manifold object to transform
            transform: Dict with 'translate', 'rotate', 'scale' keys
            
        Returns:
            Transformed Manifold object
        """
        import numpy as np
        
        # Get transform components
        translate = transform.get("translate", [0, 0, 0])
        rotate = transform.get("rotate", [0, 0, 0])  # Euler angles in degrees
        scale = transform.get("scale", [1, 1, 1])
        
        # Build 4x4 transformation matrix
        # Order: Scale -> Rotate -> Translate
        matrix = np.eye(4, dtype=np.float32)
        
        # Apply scale
        matrix[0, 0] *= scale[0]
        matrix[1, 1] *= scale[1]
        matrix[2, 2] *= scale[2]
        
        # Apply rotation (Euler angles: roll, pitch, yaw in degrees)
        roll, pitch, yaw = np.radians(rotate)
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0, 0],
            [0, np.cos(roll), -np.sin(roll), 0],
            [0, np.sin(roll), np.cos(roll), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch), 0],
            [0, 1, 0, 0],
            [-np.sin(pitch), 0, np.cos(pitch), 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0, 0],
            [np.sin(yaw), np.cos(yaw), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        
        # Combine rotations: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx
        matrix = R @ matrix
        
        # Apply translation
        matrix[0, 3] = translate[0]
        matrix[1, 3] = translate[1]
        matrix[2, 3] = translate[2]
        
        # Apply transform using Manifold's warp() if available, 
        # otherwise export to trimesh, transform, and re-import
        try:
            # Try Manifold 3.0+ warp() method
            return shape.warp(matrix)
        except (AttributeError, TypeError):
            # Fallback: export to mesh, transform with trimesh, re-import
            import trimesh
            mesh = shape.to_mesh()
            t_mesh = trimesh.Trimesh(
                vertices=np.array(mesh.vert_pos, dtype=np.float32),
                faces=np.array(mesh.tri_verts, dtype=np.int32)
            )
            t_mesh.apply_transform(matrix)
            # Convert back to Manifold via mesh
            return manifold3d.Manifold.from_mesh(t_mesh.vertices, t_mesh.faces)
