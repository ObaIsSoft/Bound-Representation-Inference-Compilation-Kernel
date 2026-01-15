from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class ToleranceAgent:
    """
    Tolerance Agent - ISO Fit Analysis.
    
    Analyzes dimensional tolerances for:
    - Clearance fits (loose)
    - Transition fits (press/slip)
    - Interference fits (press)
    - GD&T (Geometric Dimensioning & Tolerancing)
    """
    
    def __init__(self):
        self.name = "ToleranceAgent"
        # ISO 286 hole basis tolerances (simplified)
        self.hole_basis_fits = {
            "H7/g6": {"type": "clearance", "min_clear": 0.01, "max_clear": 0.04},
            "H7/h6": {"type": "transition", "min_clear": -0.01, "max_clear": 0.01},
            "H7/p6": {"type": "interference", "min_clear": -0.04, "max_clear": -0.01},
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze fits and tolerances.
        
        Args:
            params: {
                "components": List of components with dimensions,
                "nominal_diameter": Optional float (mm),
                "fit_type": Optional str (clearance/transition/interference),
                "manufacturing_process": Optional str
            }
        
        Returns:
            {
                "fits": List of fit analyses,
                "warnings": List of tolerance issues,
                "recommendations": List of suggestions,
                "logs": List of operation logs
            }
        """
        components = params.get("components", [])
        nominal_diam = params.get("nominal_diameter", 10.0)
        fit_type = params.get("fit_type", "clearance")
        process = params.get("manufacturing_process", "CNC")
        
        logs = [
            f"[TOLERANCE] Analyzing {len(components)} component(s)",
            f"[TOLERANCE] Nominal diameter: {nominal_diam} mm",
            f"[TOLERANCE] Desired fit: {fit_type}"
        ]
        
        # Process tolerance analysis
        fits = []
        warnings = []
        recommendations = []
        
        # Standard ISO fit selection based on nominal diameter
        if nominal_diam <= 3:
            tol_class = "H7/g6"  # Looser for small parts
        elif nominal_diam <= 50:
            tol_class = "H7/h6"  # Standard
        else:
            tol_class = "H8/h7"  # Looser for large parts
        
        fit_data = self.hole_basis_fits.get("H7/g6", {})
        
        fits.append({
            "tolerance_class": tol_class,
            "fit_type": fit_data.get("type", "clearance"),
            "min_clearance_mm": fit_data.get("min_clear", 0.01),
            "max_clearance_mm": fit_data.get("max_clear", 0.04),
            "nominal_diameter_mm": nominal_diam
        })
        
        # Manufacturing process considerations
        if process == "3D_PRINT":
            warnings.append("3D printing typically achieves ±0.2mm tolerance")
            recommendations.append("Increase clearances by +0.3mm for reliable assembly")
        elif process == "CNC":
            logs.append("[TOLERANCE] CNC machining can achieve ±0.01mm")
        
        # Check for tight tolerances
        if fit_data.get("max_clear", 0) < 0.01:
            warnings.append("Very tight tolerance - consider manufacturing capability")
        
        logs.append(f"[TOLERANCE] Recommended: {tol_class} ({fit_data.get('type', 'N/A')})")
        logs.append(f"[TOLERANCE] Generated {len(fits)} fit recommendation(s)")
        
        return {
            "fits": fits,
            "warnings": warnings,
            "recommendations": recommendations,
            "logs": logs
        }

    def verify_fit_precision(self, fit_spec: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verify fit precision using Virtual Machining Kernel (VMK).
        Ensures Zero Tolerance for interference in clearance fits.
        
        Args:
            fit_spec: {
                "type": "clearance",
                "hole_path": [[x,y,z], ...],
                "shaft_path": [[x,y,z], ...],
                "nominal_gap": 0.05
            }
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"verified": False, "error": "VMK not available"}

        # Initialize Kernel
        kernel = SymbolicMachiningKernel(stock_dims=[100, 100, 100])
        
        # Simulate Hole (Subtraction)
        # For a clearance check, we care if the SURFACE of the shaft hits the SURFACE of the hole.
        # We can simulate the hole cut, then query points on the shaft surface.
        
        # 1. Cut the Hole
        hole_tool_r = fit_spec.get("hole_radius", 5.0)
        tool_hole = ToolProfile(id="t_hole", radius=hole_tool_r, type="BALL")
        kernel.register_tool(tool_hole)
        kernel.execute_gcode(VMKInstruction(tool_id="t_hole", path=fit_spec["hole_path"]))
        
        # 2. Check Shaft Surface Points
        # Shaft is 'nominal_gap' smaller. 
        # We query points that represent the shaft's outer boundary.
        # If SDF < 0, it means the shaft is INSIDE the stock (Collision/Interference).
        # If SDF > 0, it means the shaft is in AIR (Clearance).
        
        shaft_pts = fit_spec.get("shaft_surface_points", [])
        min_clearance = float('inf')
        
        deviations = []
        for pt in shaft_pts:
            sdf = kernel.get_sdf(np.array(pt))
            min_clearance = min(min_clearance, sdf)
            
            # Zero Tolerance Check (allow 1nm floating point noise)
            if sdf < -1e-6: 
                deviations.append(f"Point {pt} interferes by {abs(sdf):.6f}mm")

        success = len(deviations) == 0 and min_clearance > 0
        
        return {
            "verified": success,
            "min_clearance_mm": min_clearance,
            "deviations": deviations,
            "zero_tolerance_pass": success
        }
