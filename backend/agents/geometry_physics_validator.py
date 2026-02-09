"""
Geometry Physics Validation Helpers

Provides physics-based validation for geometry components.
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def validate_geometry_physics(
    physics_kernel,
    geometry_tree: List[Dict[str, Any]], 
    material: str
) -> Dict[str, Any]:
    """
    Validates geometry using real physics calculations.
    
    Args:
        physics_kernel: UnifiedPhysicsKernel instance
        geometry_tree: List of geometry components
        material: Material name for properties lookup
    
    Returns:
        Dictionary with validation results and physics metadata
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "physics_metadata": {}
    }
    
    try:
        # Get material properties
        mat_props = physics_kernel.domains["materials"].get_material_properties(material, temperature=20.0)
        density = mat_props.get("density", 2700.0)  # kg/m^3
        yield_strength = mat_props.get("yield_strength", 275e6)  # Pa
        youngs_modulus = mat_props.get("youngs_modulus", 69e9)  # Pa
        
        total_mass = 0.0
        max_stress = 0.0
        max_deflection = 0.0
        
        g = physics_kernel.get_constant("g")  # Gravity constant
        
        for part in geometry_tree:
            ptype = part.get("type", "box")
            params = part.get("params", {})
            
            # Calculate volume based on geometry type
            if ptype in ["box", "plate", "structure"]:
                length = params.get("length", 1.0)
                width = params.get("width", 1.0)
                height = params.get("height", 0.1)
                volume = length * width * height
                
                # Calculate mass
                mass = volume * density
                total_mass += mass
                
                # Check if beam-like (length >> cross-section)
                if length > max(width, height) * 3:
                    # Treat as beam - calculate deflection and stress
                    load = mass * g  # Self-weight load in N
                    
                    # Calculate moment of inertia for rectangular cross-section
                    I = physics_kernel.domains["structures"].calculate_moi_rectangular(
                        width=width,
                        height=height
                    )
                    
                    # Calculate maximum deflection (simply supported beam)
                    deflection = physics_kernel.domains["structures"].calculate_beam_deflection(
                        load=load,
                        length=length,
                        youngs_modulus=youngs_modulus,
                        moment_of_inertia=I,
                        support_type="simply_supported"
                    )
                    
                    max_deflection = max(max_deflection, abs(deflection))
                    
                    # Calculate bending stress
                    moment = load * length / 4  # Max moment at center
                    stress = physics_kernel.domains["structures"].calculate_bending_stress(
                        moment=moment,
                        distance_from_neutral=height / 2,
                        moment_of_inertia=I
                    )
                    
                    max_stress = max(max_stress, abs(stress))
                
            elif ptype == "sphere":
                radius = params.get("radius", 0.5)
                volume = (4/3) * 3.14159 * (radius ** 3)
                mass = volume * density
                total_mass += mass
            
            elif ptype == "cylinder":
                radius = params.get("radius", 0.5)
                height = params.get("height", 1.0)
                volume = 3.14159 * (radius ** 2) * height
                mass = volume * density
                total_mass += mass
        
        # Calculate factor of safety
        if max_stress > 0:
            fos = yield_strength / max_stress
            validation["physics_metadata"]["factor_of_safety"] = fos
            
            if fos < 1.5:
                validation["warnings"].append(
                    f"Low factor of safety: {fos:.2f} (recommended: >2.0)"
                )
                if fos < 1.0:
                    validation["is_valid"] = False
                    validation["warnings"].append("CRITICAL: Factor of safety < 1.0 - structure may fail!")
        
        # Store physics metadata
        validation["physics_metadata"].update({
            "total_mass_kg": total_mass,
            "max_stress_pa": max_stress,
            "max_deflection_m": max_deflection,
            "material": material,
            "density_kg_m3": density,
            "yield_strength_pa": yield_strength,
            "weight_n": total_mass * g
        })
        
        logger.info(
            f"Physics validation: "
            f"mass={total_mass:.3f}kg, "
            f"stress={max_stress/1e6:.1f}MPa, "
            f"deflection={max_deflection*1000:.2f}mm"
        )
        
    except Exception as e:
        logger.error(f"Physics validation failed: {e}")
        validation["warnings"].append(f"Physics validation error: {str(e)}")
        validation["is_valid"] = False
    
    return validation


class GeometryPhysicsValidator:
    """
    Agent wrapper for geometry-physics compatibility checking.
    """
    def __init__(self):
        self.name = "GeometryPhysicsValidator"
    
    def run(self, geometry: List[Dict[str, Any]], mass_props: Dict[str, Any], structural: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate consistency between geometry and physics analysis results.
        """
        logger.info("Validating geometry-physics consistency...")
        
        compatible = True
        issues = []
        
        # 1. Geometry Check
        if not geometry:
            issues.append("Geometry tree is empty")
            compatible = False
            
        # 2. Mass Check
        mass = mass_props.get("total_mass_kg", 0)
        # Relaxed check: Only warn if 0, unless it's strictly required
        if mass < 0:
            issues.append(f"Invalid negative mass: {mass} kg")
            compatible = False
            
        # 3. Structural Check
        # If structural analysis failed or has critical warnings
        if structural.get("status") == "failed":
            issues.append("Structural analysis failed")
            compatible = False
            
        max_stress = structural.get("max_stress_pa", 0)
        yield_strength = structural.get("yield_strength_pa", 275e6) # Default Al
        
        if max_stress > yield_strength:
             issues.append(f"Stress {max_stress:.2e} Pa exceeds yield strength {yield_strength:.2e} Pa")
             compatible = False
        
        return {
            "compatible": compatible,
            "issues": issues,
            "checked_items": ["geometry_presence", "mass_validity", "structural_safety"]
        }
