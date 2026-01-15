from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class StructuralAgent:
    """
    Structural Analysis Agent.
    Estimates stress, strain, and safety factors.
    """
    def __init__(self):
        self.name = "StructuralAgent"
        
        # Initialize Oracles for structural analysis
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            from agents.materials_oracle.materials_oracle import MaterialsOracle
            self.physics_oracle = PhysicsOracle()
            self.materials_oracle = MaterialsOracle()
            self.has_oracles = True
        except ImportError:
            self.physics_oracle = None
            self.materials_oracle = None
            self.has_oracles = False

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate stress under load.
        """
        logger.info(f"{self.name} starting structural analysis...")
        
        # Robust Param Extraction
        mass_kg = float(params.get("mass_kg", 10.0))
        g_force = float(params.get("g_loading", 3.0)) 
        cross_section_mm2 = float(params.get("cross_section_mm2", 100.0))
        length_m = float(params.get("length_m", 1.0)) 

        # Material Properties (Input from MaterialAgent chain)
        mat_props = params.get("material_properties", {})
        
        # Fallbacks (Log Warnings if used)
        if "yield_strength" in mat_props:
            yield_strength_mpa = float(mat_props["yield_strength"]) / 1e6 # Pa to MPa
        else:
            yield_strength_mpa = float(params.get("yield_strength_mpa", 276.0))
            if not mat_props: logger.warning("No material_props found in payload. Using fallback Yield Strength.")

        if "elastic_modulus" in mat_props:
             # DB might store as Youngs Modulus in Pa or GPa. Assumed Pa usually.
             elastic_modulus_gpa = float(mat_props["elastic_modulus"]) / 1e9 
        else:
             elastic_modulus_gpa = float(params.get("elastic_modulus_gpa", 69.0))

        logs = []
        
        # 1. Stress Analysis (Axial)
        force_n = mass_kg * (g_force * 9.81)
        stress_mpa = force_n / max(cross_section_mm2, 0.1) # Avoid div/0
        
        fos_yield = yield_strength_mpa / max(stress_mpa, 0.001)
        
        logs.append(f"Load Case: {g_force}G on {mass_kg}kg ({force_n:.1f}N)")
        logs.append(f"Axial Stress: {stress_mpa:.1f} MPa (Yield: {yield_strength_mpa} MPa)")
        
        # 2. Buckling Analysis (Euler)
        # P_cr = (pi^2 * E * I) / (K * L)^2
        # Assume solid circular rod for Inertia (I) if not provided
        # I = (pi * r^4) / 4
        # Area = pi * r^2 -> r = sqrt(Area/pi)
        
        # Derived Radius (mm)
        radius_mm = (cross_section_mm2 / 3.14159) ** 0.5
        radius_m = radius_mm / 1000.0
        
        # Moment of Inertia (m^4)
        I = (3.14159 * (radius_m ** 4)) / 4
        
        # Critical Load (N)
        pi = 3.14159
        E_pa = elastic_modulus_gpa * 1e9
        L = length_m
        K = 1.0 # Pinned-Pinned default
        
        try:
            critical_load_n = (pi**2 * E_pa * I) / ((K * L)**2)
        except ZeroDivisionError:
            critical_load_n = 0
            
        fos_buckling = critical_load_n / max(force_n, 0.001)
        
        logs.append(f"Buckling Critical Load: {critical_load_n:.1f}N")
        logs.append(f"Buckling FoS: {fos_buckling:.2f}")

        # 3. Verdict
        # Overall FoS is the min of Yield and Buckling
        overall_fos = min(fos_yield, fos_buckling)
        
        status = "safe"
        if overall_fos < 1.0: 
            status = "failure"
            if fos_buckling < 1.0: logs.append("FAILURE MODE: Buckling Instability")
            elif fos_yield < 1.0: logs.append("FAILURE MODE: Yield Stress Exceeded")
            
        elif overall_fos < 1.5: 
            status = "marginal"
        
        return {
            "status": status,
            "max_stress_mpa": round(stress_mpa, 2),
            "safety_factor": round(overall_fos, 2),
            "yield_fos": round(fos_yield, 2),
            "buckling_fos": round(fos_buckling, 2),
            "load_n": round(force_n, 2),
            "logs": logs
        }

    def detect_stress_risers(self, geometry_history: List[Dict[str, Any]], critical_radius: float = 1.0) -> List[str]:
        """
        Uses VMK to find sharp corners (Radius < critical_radius) which act as stress concentrators.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return []
            
        # Replay Geometry
        kernel = SymbolicMachiningKernel(stock_dims=[100,100,100]) # Generic bounds for analysis
        for op in geometry_history:
             # Auto-register tools (Simplified)
             tid = op.get("tool_id")
             if tid and tid not in kernel.tools:
                 kernel.register_tool(ToolProfile(id=tid, radius=op.get("radius", 1.0), type="BALL"))
             kernel.execute_gcode(VMKInstruction(**op))
             
        # Scan Surface for Curvature
        # Heuristic: Sample points. If SDF changes rapidly (high 2nd derivative) -> Sharp.
        # But SDF of sharp corner is continuous. 
        # Actually, sharp corner = Gradient Discontinuity.
        # But Sampled SDF smooths it? No, Exact SDF preserves it.
        
        # Simplified Check for MVP:
        # Check if we have "Square" cuts? 
        # A Square Cut (Endmill) leaves a radius = tool_radius.
        # If tool_radius < critical_radius, it's a riser.
        
        risers = []
        
        # Analytical Check (Meta-Analysis of History)
        # Verify that all Subtractive Tools have Radius >= Critical Radius
        for op in geometry_history:
            r = op.get("radius", 0.0)
            # If it's a cutting move (subtract)
            # And radius is small.
            if r > 0 and r < critical_radius:
                risers.append(f"Sharp Corner Detected: Tool '{op.get('tool_id')}' Radius {r}mm < Limit {critical_radius}mm")
                
        # Future: True Geometric Curvature scan using kernel.get_sdf gradient analysis.
        
        return risers

    def analyze_stress_oracle(self, params: dict) -> dict:
        """Analyze structural stress using Physics Oracle (MECHANICS)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.physics_oracle.solve(
            query="Structural stress analysis",
            domain="MECHANICS",
            params=params
        )
    
    def analyze_material_properties_oracle(self, params: dict) -> dict:
        """Analyze material properties using Materials Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Material property analysis",
            domain="MECHANICAL",
            params=params
        )
    
    def predict_failure_oracle(self, params: dict) -> dict:
        """Predict structural failure using Materials Oracle (FAILURE domain)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Failure prediction",
            domain="FAILURE",
            params=params
        )
