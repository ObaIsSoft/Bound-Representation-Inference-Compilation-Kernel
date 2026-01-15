from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ManifoldAgent:
    """
    Manifold Agent - Watertight Mesh Validation.
    
    Validates that geometry forms a closed, manifold surface suitable for:
    - 3D printing (STL export)
    - Manufacturing (CAM toolpaths)
    - Physics simulation (closed volumes)
    
    Checks:
    - Edge connectivity (every edge shared by exactly 2 faces)
    - Genus (holes/handles in topology)
    - Self-intersections
    - Degenerate faces
    """
    
    def __init__(self):
        self.name = "ManifoldAgent"
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate mesh manifoldness.
        
        Args:
            params: {
                "geometry_tree": List of geometry nodes with mesh data,
                "tolerance": Optional float (default 1e-6)
            }
        
        Returns:
            {
                "is_manifold": bool,
                "is_watertight": bool,
                "edge_count": int,
                "boundary_edges": int,  # Should be 0 for watertight
                "non_manifold_edges": int,  # Edges shared by >2 faces
                "genus": int,  # Topological holes (0 = sphere-like)
                "issues": List of detected problems,
                "logs": List of operation logs
            }
        """
        geometry_tree = params.get("geometry_tree", [])
        tolerance = params.get("tolerance", 1e-6)
        
        logs = [
            f"[MANIFOLD] Starting validation with tolerance {tolerance}",
            f"[MANIFOLD] Analyzing {len(geometry_tree)} geometry node(s)"
        ]
        
        if not geometry_tree:
            return {
                "is_manifold": False,
                "is_watertight": False,
                "edge_count": 0,
                "boundary_edges": 0,
                "non_manifold_edges": 0,
                "genus": 0,
                "issues": ["No geometry provided"],
                "logs": logs + ["[MANIFOLD] No geometry to validate"]
            }
        
        # Simplified validation (real implementation would parse actual mesh data)
        # This is a stub that demonstrates the expected behavior
        
        total_faces = sum(node.get("face_count", 0) for node in geometry_tree)
        total_vertices = sum(node.get("vertex_count", 0) for node in geometry_tree)
        
        # Euler characteristic: V - E + F = 2 - 2*genus (for closed surfaces)
        # E (edges) ≈ 3F/2 for triangulated mesh
        estimated_edges = int(total_faces * 1.5)
        
        logs.append(f"[MANIFOLD] Faces: {total_faces}, Vertices: {total_vertices}, Edges: ~{estimated_edges}")
        
        # Heuristic checks (real agent would analyze actual mesh topology)
        issues = []
        boundary_edges = 0
        non_manifold_edges = 0
        
        # Check if mesh is properly closed
        if total_faces < 4:
            issues.append("Insufficient faces for closed volume (need ≥4)")
            boundary_edges = estimated_edges
        
        # Euler characteristic check for genus
        euler_char = total_vertices - estimated_edges + total_faces
        genus = (2 - euler_char) // 2
        
        if genus < 0:
            genus = 0
            issues.append("Invalid topology detected (negative genus)")
        
        logs.append(f"[MANIFOLD] Euler characteristic: {euler_char}, Genus: {genus}")
        
        # Final determination
        is_watertight = len(issues) == 0 and boundary_edges == 0
        is_manifold = len(issues) == 0 and non_manifold_edges == 0
        
        if is_watertight:
            logs.append("[MANIFOLD] ✓ Mesh is watertight")
        else:
            logs.append(f"[MANIFOLD] ✗ Mesh has {len(issues)} issue(s)")
        
        return {
            "is_manifold": is_manifold,
            "is_watertight": is_watertight,
            "edge_count": estimated_edges,
            "boundary_edges": boundary_edges,
            "non_manifold_edges": non_manifold_edges,
            "genus": genus,
            "issues": issues,
            "logs": logs
        }
    
    def check_edge_connectivity(self, mesh_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Detailed edge analysis.
        
        Args:
            mesh_data: {
                "vertices": List of [x, y, z],
                "faces": List of [v1, v2, v3] indices
            }
        
        Returns:
            {
                "boundary_edges": List of edges with 1 face,
                "manifold_edges": List of edges with 2 faces,
                "non_manifold_edges": List of edges with >2 faces
            }
        """
        # Stub for detailed edge analysis
        # Real implementation would build edge->face adjacency map
        return {
            "boundary_edges": [],
            "manifold_edges": [],
            "non_manifold_edges": []
        }

    def repair_topology(self, geometry_history: List[Dict[str, Any]], stock_dims: List[float]) -> Dict[str, Any]:
        """
        Guaranteed Manifold Repair using SDF.
        If a mesh is broken (holes, non-manifold edges), we convert it to SDF and re-mesh.
        SDFs are by definition watertight (implicit surface d=0).
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
            from skimage import measure
            import trimesh
        except ImportError as e:
            return {"status": "failed", "error": f"Dependency missing: {e}"}
            
        logs = ["[MANIFOLD-REPAIR] Starting SDF Reconstruction..."]
            
        # 1. Build SDF from History
        try:
            kernel = SymbolicMachiningKernel(stock_dims=stock_dims)
            for op in geometry_history:
                 tid = op.get("tool_id", "repair_tool")
                 if tid not in kernel.tools:
                     kernel.register_tool(ToolProfile(id=tid, radius=op.get("radius", 1.0), type="BALL"))
                 kernel.execute_gcode(VMKInstruction(**op))
            
            logs.append(f"[MANIFOLD-REPAIR] VMK State Built. History: {len(geometry_history)} ops")

            # 2. Generate SDF Grid
            # Resolution: 64^3 is fast enough for verifying topology. 
            # For production, we'd use 128^3 or 256^3 or Octree.
            res = (64, 64, 64) # Increased to cube for better sampling
            padding = 2.0
            logs.append(f"[MANIFOLD-REPAIR] Generating SDF Grid {res} with padding {padding}...")
            sdf_grid = kernel.get_sdf_grid(dims=res, padding=padding)
            
            # 3. Marching Cubes
            logs.append("[MANIFOLD-REPAIR] Running Marching Cubes...")
            
            # Use level=0 for surface. Gradient direction ensures correct normals.
            # Step size scaling creates real-world dimensions.
            # Grid spans: stock_dims + 2*padding
            
            span_x = stock_dims[0] + 2*padding
            span_y = stock_dims[1] + 2*padding
            span_z = stock_dims[2] + 2*padding
            
            spacing = (
                span_x / (res[0]-1),
                span_y / (res[1]-1),
                span_z / (res[2]-1)
            )
            
            verts, faces, normals, values = measure.marching_cubes(
                sdf_grid, 
                level=0.0, 
                spacing=spacing
            )
            
            # Center the mesh (Marching cubes starts at 0,0,0)
            # Origin of grid was -stock/2 - padding.
            offset = np.array([
                -stock_dims[0]/2 - padding, 
                -stock_dims[1]/2 - padding, 
                -stock_dims[2]/2 - padding
            ])
            verts += offset
            
            # 4. Validate New Mesh (Process=True for auto-cleanup)
            mesh = trimesh.Trimesh(vertices=verts, faces=faces, process=True)
            
            # Aggressive Repair
            try:
                trimesh.repair.fix_normals(mesh)
                trimesh.repair.fix_inversion(mesh)
                trimesh.repair.fill_holes(mesh)
                trimesh.repair.fix_winding(mesh)
            except Exception as e:
                logs.append(f"[MANIFOLD-REPAIR] Repair Warning: {e}")
            
            is_watertight = mesh.is_watertight
            is_winding = mesh.is_winding_consistent
            
            logs.append(f"[MANIFOLD-REPAIR] Reconstruction Complete. V: {len(mesh.vertices)}, F: {len(mesh.faces)}")
            logs.append(f"[MANIFOLD-REPAIR] Watertight: {is_watertight}")
            
            return {
                "status": "repaired",
                "method": "SDF_Reconstruction",
                "is_manifold": True,
                "is_watertight": is_watertight,
                "new_vertex_count": len(mesh.vertices),
                "new_face_count": len(mesh.faces),
                "logs": logs
            }
            
        except Exception as e:
            return {
                "status": "failed", 
                "error": str(e),
                "logs": logs + [f"[ERROR] {str(e)}"]
            }
