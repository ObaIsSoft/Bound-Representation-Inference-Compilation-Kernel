
import numpy as np
import math
from typing import List, Dict, Any, Optional, Tuple, Union
try:
    from .openscad_parser import ASTNode, NodeType
except ImportError:
    from backend.agents.openscad_parser import ASTNode, NodeType

class AABB:
    """Axis Aligned Bounding Box."""
    def __init__(self, min_pt=None, max_pt=None):
        self.min_pt = np.array(min_pt if min_pt is not None else [float('inf'), float('inf'), float('inf')])
        self.max_pt = np.array(max_pt if max_pt is not None else [float('-inf'), float('-inf'), float('-inf')])
        
    def expand(self, other: 'AABB'):
        """Union with another AABB."""
        if other.is_empty(): return
        self.min_pt = np.minimum(self.min_pt, other.min_pt)
        self.max_pt = np.maximum(self.max_pt, other.max_pt)
        
    def add_point(self, pt: np.ndarray):
        """Expand to include a point."""
        self.min_pt = np.minimum(self.min_pt, pt)
        self.max_pt = np.maximum(self.max_pt, pt)
        
    def is_empty(self):
        return self.min_pt[0] == float('inf')
        
    def center(self):
        if self.is_empty(): return np.array([0., 0., 0.])
        return (self.min_pt + self.max_pt) / 2.0
    
    def corners(self):
        """Return 8 corners of the box."""
        if self.is_empty(): return []
        x0, y0, z0 = self.min_pt
        x1, y1, z1 = self.max_pt
        return [
            np.array([x0, y0, z0]), np.array([x1, y0, z0]),
            np.array([x0, y1, z0]), np.array([x1, y1, z0]),
            np.array([x0, y0, z1]), np.array([x1, y0, z1]),
            np.array([x0, y1, z1]), np.array([x1, y1, z1])
        ]
        
    def transform(self, matrix: np.ndarray) -> 'AABB':
        """Apply 4x4 matrix to AABB and return new AABB covering the result."""
        if self.is_empty(): return AABB()
        
        corners = self.corners()
        new_box = AABB()
        
        for p in corners:
            # Homogeneous coord
            ph = np.array([p[0], p[1], p[2], 1.0])
            p_trans = matrix @ ph
            new_box.add_point(p_trans[:3])
            
        return new_box


class GeometryEstimator:
    """
    Analytic Geometry Bounds Estimator from AST.
    Avoids CSG compilation.
    """
    def __init__(self):
        self.name = "GeometryEstimator"

    def estimate(self, intent: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Quick feasibility check based on intent keywords and parameters.
        Phase 1 Feasibility.
        """
        # Heuristic check
        impossible = False
        reason = ""
        
        # Check size constraints
        dims = params.get("dimensions", [0,0,0])
        MAX_SIZE = 5000 # mm
        if any(d > MAX_SIZE for d in dims):
            impossible = True
            reason = f"Dimensions exceed max size ({MAX_SIZE}mm)"
            
        return {
            "feasible": not impossible,
            "impossible": impossible,
            "reason": reason,
            "estimated_bounds": {"min": [0,0,0], "max": dims} # Placeholder
        }
    
    def calculate_bounds(self, nodes: List[ASTNode], variables: Dict[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Calculate global bounding box for the AST.
        Returns: bounds dict, center dict
        """
        # Identity matrix
        matrix = np.eye(4)
        
        global_aabb = AABB()
        
        for node in nodes:
            aabb = self._traverse(node, matrix, variables)
            global_aabb.expand(aabb)
            
        if global_aabb.is_empty():
            return {"min": [0,0,0], "max": [0,0,0]}, {"x":0,"y":0,"z":0}
            
        center = global_aabb.center()
        
        return {
            "min": global_aabb.min_pt.tolist(),
            "max": global_aabb.max_pt.tolist()
        }, {
            "x": float(center[0]),
            "y": float(center[1]),
            "z": float(center[2])
        }

    def _traverse(self, node: ASTNode, current_matrix: np.ndarray, variables: Dict[str, Any]) -> AABB:
        """Recursive traversal."""
        
        # 1. Primitives (Base Case)
        if node.node_type == NodeType.PRIMITIVE:
            return self._bounds_primitive(node, current_matrix, variables)
            
        # 2. Transforms calls
        if node.node_type == NodeType.TRANSFORM:
            # Calculate new matrix
            new_matrix = self._apply_transform_matrix(node, current_matrix, variables)
            
            # Recurse on children with new matrix
            result_aabb = AABB()
            for child in node.children:
                child_aabb = self._traverse(child, new_matrix, variables)
                result_aabb.expand(child_aabb)
            return result_aabb
            
        # 3. Booleans
        if node.node_type == NodeType.BOOLEAN:
            # Special logic for Difference/Intersection
            # Heuristic: Difference bounds <= First Child bounds
            # Union bounds = Sum of all child bounds
            
            op = node.name
            
            if op == "difference":
                # Only consider first child?
                # A - B. Bounds are at most bounds(A).
                if not node.children: return AABB()
                return self._traverse(node.children[0], current_matrix, variables)
                
            elif op == "intersection":
                # Bounds are intersection of Intersection(bounds(A), bounds(B)).
                # Conservative: min of volumes? Or just first child.
                # Let's use intersection of AABBs if possible, or just first child.
                if not node.children: return AABB()
                return self._traverse(node.children[0], current_matrix, variables) # Conservative
                
            else:
                # Union, Hull, Minkowski -> Union of bounds
                # (Minkowski is technically A+B, but we approximate as Union for centering? 
                # No, Minkowski is larger. But it's rare. Let's assume Union for now.)
                result_aabb = AABB()
                for child in node.children:
                    child_aabb = self._traverse(child, current_matrix, variables)
                    result_aabb.expand(child_aabb)
                return result_aabb
                
        # 4. Modules / Loops / Conditionals / Others
        # Just traverse children with current matrix
        result_aabb = AABB()
        for child in node.children:
            child_aabb = self._traverse(child, current_matrix, variables)
            result_aabb.expand(child_aabb)
        return result_aabb
        
        
    def _bounds_primitive(self, node: ASTNode, matrix: np.ndarray, variables: Dict[str, Any]) -> AABB:
        """Calculate AABB for primitive in local space, then transform."""
        name = node.name
        params = node.params
        
        local_aabb = AABB()
        
        # Resolve values helpers
        def get_val(key, default=0.0):
            # Param map check
            val = params.get(key)
            # If positional?
            if val is None and '_pos_0' in params:
                 # Heuristic mapping for simple primitives
                 if name == "cube":
                     if key == "size": return params['_pos_0']
                 if name == "sphere":
                     if key == "r": return params['_pos_0']
            
            if val is None: return default
            
            # If val is a variable name string? _evaluate_expression in parser handles most.
            # But if it's still a string here, we might need lookup?
            # Creating a robust resolver? 
            # The parser evaluates simple expressions. 
            # Assuming 'params' already contains numbers or lists.
            return val

        # Handle 'center' param
        center = get_val("center", False)
        
        if name == "cube":
            # cube(size=[x,y,z], center=bool) or cube(size=N)
            size = get_val("size")
            if size is None: size = get_val("_pos_0", 1.0)
            
            if isinstance(size, (int, float)):
                dims = [size, size, size]
            elif isinstance(size, list) and len(size) >= 3:
                dims = size[:3]
            else:
                dims = [1,1,1] # Fallback
                
            if center:
                local_aabb.min_pt = np.array([-dims[0]/2, -dims[1]/2, -dims[2]/2])
                local_aabb.max_pt = np.array([dims[0]/2, dims[1]/2, dims[2]/2])
            else:
                local_aabb.min_pt = np.array([0, 0, 0])
                local_aabb.max_pt = np.array([dims[0], dims[1], dims[2]])
                
        elif name == "sphere":
            # sphere(r=N) or sphere(d=N)
            r = get_val("r")
            if r is None: 
                d = get_val("d")
                if d is not None: r = d/2
                else: r = get_val("_pos_0", 1.0)
                
            local_aabb.min_pt = np.array([-r, -r, -r])
            local_aabb.max_pt = np.array([r, r, r])
            
        elif name == "cylinder":
            # cylinder(h, r/r1/r2, center)
            h = get_val("h")
            if h is None: h = get_val("_pos_0", 1.0) # Pos 0 is usually H? No, usually r is named or unnamed?
            # cylinder(h=10, r=5). cylinder(5, 10)? 
            # Docs: cylinder(h, r|d, ...)
            
            r = get_val("r")
            if r is None:
                r1 = get_val("r1", 1.0)
                r2 = get_val("r2", 1.0)
                r = max(r1, r2)
            if r is None: r = 1.0
            
            if center:
                local_aabb.min_pt = np.array([-r, -r, -h/2])
                local_aabb.max_pt = np.array([r, r, h/2])
            else:
                local_aabb.min_pt = np.array([-r, -r, 0])
                local_aabb.max_pt = np.array([r, r, h])
                
        # Default fallback for others (polyhedron, etc) -> Bounds 0?
        # Or unit box?
        # Safe to assume empty? Better unit box so we see it.
        
        return local_aabb.transform(matrix)

    def _apply_transform_matrix(self, node: ASTNode, parent_matrix: np.ndarray, variables: Dict[str, Any]) -> np.ndarray:
        """Multiply parent matrix by node transform."""
        t_name = node.name
        params = node.params
        
        # Build local 4x4
        local_m = np.eye(4)
        
        def pv(k, default=None):
            # Resolve value helper similar to above
            val = params.get(k)
            if val is None: val = params.get('_pos_0')
            return val if val is not None else default

        if t_name == "translate":
            v = pv("v", [0,0,0])
            if isinstance(v, list) and len(v) >= 3:
                local_m[0, 3] = v[0]
                local_m[1, 3] = v[1]
                local_m[2, 3] = v[2]
                
        elif t_name == "scale":
            v = pv("v", [1,1,1])
            if isinstance(v, list) and len(v) >= 3:
                local_m[0, 0] = v[0]
                local_m[1, 1] = v[1]
                local_m[2, 2] = v[2]
                
        elif t_name == "rotate":
            # Handle [x,y,z] euler
            # or a, v=[x,y,z] ?? 
            # Standard: rotate([x,y,z])
            v = pv("a", [0,0,0]) # Param name is 'a' often
            if v is None: v = params.get("_pos_0", [0,0,0])
            
            if isinstance(v, list) and len(v) >= 3:
                rx, ry, rz = np.radians(v[0]), np.radians(v[1]), np.radians(v[2])
                
                # Rx
                mx = np.eye(4)
                mx[1,1] = np.cos(rx); mx[1,2] = -np.sin(rx)
                mx[2,1] = np.sin(rx); mx[2,2] = np.cos(rx)
                
                # Ry
                my = np.eye(4)
                my[0,0] = np.cos(ry); my[0,2] = np.sin(ry)
                my[2,0] = -np.sin(ry); my[2,2] = np.cos(ry)
                
                # Rz
                mz = np.eye(4)
                mz[0,0] = np.cos(rz); mz[0,1] = -np.sin(rz)
                mz[1,0] = np.sin(rz); mz[1,1] = np.cos(rz)
                
                # Order X Y Z? SCAD def usually XYZ apply order.
                # Actually SCAD rotates around axes.
                local_m = mz @ my @ mx
                
        # mirror, multmatrix support can be added. 
        # For now, translate/rotate/scale covers 99% of "centering" issues.
        
        return parent_matrix @ local_m
