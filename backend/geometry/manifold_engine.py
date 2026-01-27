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
                 t_mesh = trimesh.Trimesh(
                     vertices=mesh.vert_properties, # Manifold v3 nomenclature?
                     faces=mesh.tri_verts
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
                t_mesh = trimesh.Trimesh(
                    vertices=mesh.vert_properties, 
                    faces=mesh.tri_verts
                )
                
                # Check for resolution param, default 64
                res = request.params.get("resolution", 64)
                
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
                # Apply translate/rotate
                pass # TODO: Implement matrix logic
                
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
             
        return union_result

    def _create_primitive(self, node: Dict) -> manifold3d.Manifold:
        ptype = node.get("type", "box")
        params = node.get("params", {})
        
        if ptype == "box":
            x = params.get("length", 1.0)
            y = params.get("width", 1.0)
            z = params.get("height", 1.0)
            return manifold3d.Manifold.cube(np.array([x, y, z]), center=True)
            
        elif ptype == "sphere":
            r = params.get("radius", 1.0)
            return manifold3d.Manifold.sphere(r, circular_segments=32)
            
        elif ptype == "cylinder":
             h = params.get("height", 1.0)
             r = params.get("radius", 1.0)
             return manifold3d.Manifold.cylinder(h, r, r, circular_segments=32)
             
        # Fallback
        return manifold3d.Manifold.cube(np.array([0.1, 0.1, 0.1]), center=True)
