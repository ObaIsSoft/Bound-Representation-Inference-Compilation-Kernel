
import numpy as np
from typing import List, Dict, Any, Union, Optional, Tuple
from enum import Enum
from pydantic import BaseModel
import hashlib
import json

class OpType(str, Enum):
    UNION = "union"
    SUBTRACT = "subtract"
    INTERSECT = "intersect"
    SMOOTH_UNION = "smin"

class FitClass(str, Enum):
    """ISO 286-2 Fit Class Primitives"""
    H7_g6 = "CLEARANCE_LOCATIONAL" # Sliding fits
    H7_h6 = "CLEARANCE_SLIDING"    # Precision sliding
    H7_p6 = "INTERFERENCE_PRESS"   # Permanent assembly
    H7_s6 = "INTERFERENCE_HEAVY"   # High-torque press
    NONE = "none"

class Dimension(BaseModel):
    """A Stochastic Physical Dimension"""
    nominal: float # mm
    upper_tol: float # mm
    lower_tol: float # mm
    
    def get_range(self) -> Tuple[float, float]:
        return (self.nominal + self.lower_tol, self.nominal + self.upper_tol)

class HighFidelityKernel:
    """
    HWC Kernel v3.2.0: Recursive Precision Synthesis.
    A symbolic engine featuring the Ares Precision Layer for 
    automated Tolerance Stack-up and Fit-Class resolution.
    """
    def __init__(self):
        self.instructions = []
        self.registry = {} # SKU Digital Twin mapping
        self.tolerance_stack = []

    def smin(self, d1: str, d2: str, k: str = "0.1") -> str:
        """Symbolic Smooth-Minimum for organic structural transitions."""
        return f"smin({d1}, {d2}, {k})"

    def primitive(self, p_type: str, params: Dict[str, Any], offset: List[float] = [0, 0, 0]) -> str:
        """
        Generates a symbolic SDF primitive string.
        """
        p_str = "p"
        if offset != [0,0,0]:
             p_str = f"(p - vec3({offset[0]}, {offset[1]}, {offset[2]}))"
        
        if p_type == "box":
            return f"sdBox({p_str}, vec3({params['x']}, {params['y']}, {params['z']}))"
        elif p_type == "torus":
            return f"sdTorus({p_str}, vec2({params['radius']}, {params['thickness']}))"
        elif p_type == "cylinder":
            return f"sdCylinder({p_str}, {params['height']}, {params['radius']})"
        elif p_type == "sphere":
            return f"sdSphere({p_str}, {params['radius']})"
        elif p_type == "capsule":
             return f"sdCapsule({p_str}, vec3({params['a'][0]},{params['a'][1]},{params['a'][2]}), vec3({params['b'][0]},{params['b'][1]},{params['b'][2]}), {params['r']})"
        elif p_type == "structure": # Fallback for unknown structure types
             # Default to sphere for containment volume
             return f"sdSphere({p_str}, {params.get('radius', 0.1)})"
        return "1e10" # Empty/Far away

    def calculate_stackup(self, chain: List[Dimension]) -> Dict[str, float]:
        """
        Ares Layer: Automated Tolerance Stack-up Resolver.
        Uses RSS (Root Sum Squares) for statistical assembly probability.
        """
        nominal_total = sum(d.nominal for d in chain)
        
        # 1. Statistical Variance (3-Sigma / 99.73% confidence)
        # RSS = sqrt(tol1^2 + tol2^2 + ... + tolN^2)
        rss_tol = np.sqrt(sum(((d.upper_tol - d.lower_tol) / 2)**2 for d in chain))
        
        # 2. Worst-Case (Absolute limits)
        worst_case_upper = sum(d.upper_tol for d in chain)
        worst_case_lower = sum(d.lower_tol for d in chain)
        
        return {
            "nominal": nominal_total,
            "rss_tolerance": rss_tol,
            "worst_case_upper": worst_case_upper,
            "worst_case_lower": worst_case_lower,
            "probability_of_failure": 0.0027 if rss_tol > 0.05 else 0.0 # Heuristic
        }

    def mating_interface(self, target_id: str, base_dim: float, fit: FitClass) -> float:
        """
        Determines the precise dimension for mating based on ISO standards.
        Returns the adjusted dimension (float), does NOT return SDF string directly.
        """
        # ISO 286-2 Table Lookups (Mocked)
        fit_deltas = {
            FitClass.H7_g6: {"hole": [0.015, 0.0], "shaft": [-0.005, -0.014]},
            FitClass.H7_p6: {"hole": [0.015, 0.0], "shaft": [0.042, 0.024]},
            FitClass.H7_h6: {"hole": [0.015, 0.0], "shaft": [0.0, -0.009]},
            FitClass.H7_s6: {"hole": [0.015, 0.0], "shaft": [0.053, 0.035]}
        }
        
        delta = fit_deltas.get(fit, {"hole": [0,0], "shaft": [0,0]})
        
        # Determine if Shaft or Hole based on context (default to Hole/Female for receiving part)
        # Or require explicit flag. For now, heuristics:
        is_shaft = "shaft" in target_id.lower() or "male" in target_id.lower()
        
        d_type = "shaft" if is_shaft else "hole"
        vals = delta[d_type]
        
        # Record tolerance
        self.tolerance_stack.append({
            "id": target_id,
            "fit": fit,
            "type": d_type,
            "nominal": base_dim,
            "dim_object": Dimension(nominal=base_dim, upper_tol=vals[0], lower_tol=vals[1]).dict()
        })
        
        # Return mid-tolerance dimension target
        return base_dim + (vals[0] + vals[1]) / 2

    def to_glsl(self, isa: Dict[str, Any]) -> str:
        """
        Transpiles the Precision ISA into a Raymarching map() function.
        Dynamically traverses the render_tree components.
        """
        ops = isa["render_tree"]["operations"]
        if not ops:
            body = "return 100.0;"
        else:
            lines = []
            final_d = "d_0"
            
            for i, op in enumerate(ops):
                p_type = op.get("type", "box")
                params = op.get("params", {}).copy()
                offset = op.get("offset", [0,0,0])
                
                # --- ARES LAYER INTERCEPT ---
                # If component needs a fit, adjust dimensions BEFORE generating SDF
                fit = op.get("fit", FitClass.NONE)
                if fit != FitClass.NONE:
                    # Adjust major dimension (radius for cylinder/hole)
                    if "radius" in params:
                        params["radius"] = self.mating_interface(op.get("id"), params["radius"], fit)
                
                # Param Normalization
                if p_type == "box" and "dims" in params:
                    d = params["dims"]
                    params["x"] = d[0]/2
                    params["y"] = d[1]/2
                    params["z"] = d[2]/2
                
                sdf_str = self.primitive(p_type, params, offset)
                lines.append(f"    float d_{i} = {sdf_str};")
                
                if i > 0:
                    blend = op.get("blend", 0.0)
                    if blend > 0:
                        lines.append(f"    d_0 = smin(d_0, d_{i}, {blend});")
                    else:
                        lines.append(f"    d_0 = min(d_0, d_{i});")
            
            body = "\n".join(lines) + "\n    return d_0;"

        primitives = """
// SDF Primitives
float sdSphere( vec3 p, float s ) {
  return length(p)-s;
}
float sdBox( vec3 p, vec3 b ) {
  vec3 q = abs(p) - b;
  return length(max(q,0.0)) + min(max(q.x,max(q.y,q.z)),0.0);
}
float sdTorus( vec3 p, vec2 t ) {
  vec2 q = vec2(length(p.xz)-t.x,p.y);
  return length(q)-t.y;
}
float sdCylinder( vec3 p, float h, float r ) {
  vec2 d = abs(vec2(length(p.xz),p.y)) - vec2(r,h);
  return min(max(d.x,d.y),0.0) + length(max(d,0.0));
}
float sdCapsule( vec3 p, vec3 a, vec3 b, float r ) {
  vec3 pa = p - a, ba = b - a;
  float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
  return length( pa - ba*h ) - r;
}
// Smooth Min (Polynomial)
float smin( float a, float b, float k ) {
    float h = max( k-abs(a-b), 0.0 )/k;
    return min( a, b ) - h*h*k*(1.0/4.0);
}
"""
        return f"""
{primitives}

float map(vec3 p) {{
{body}
}}
"""

    def synthesize_isa(self, project_id: str, components: List[Dict], ops: List[Dict]) -> Dict[str, Any]:
        """Final compilation of the Physical State."""
        isa = {
            "project_id": project_id,
            "fidelity": "PRECISION_L4_STOCHASTIC",
            "state_hash": None,
            "tolerance_report": self.tolerance_stack,
            "render_tree": {"components": components, "operations": ops}
        }
        isa["state_hash"] = hash(str(isa))
        return isa
