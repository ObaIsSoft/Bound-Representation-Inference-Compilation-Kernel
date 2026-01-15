from typing import Dict, Any, List
import logging
import math
from isa import PhysicalValue, Unit, create_physical_value

logger = logging.getLogger(__name__)

class MassPropertiesAgent:
    """
    Mass Properties Agent.
    Calculates Mass, Center of Gravity (CoG), and Inertia based on geometry and material.
    """
    def __init__(self):
        self.name = "MassPropertiesAgent"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute mass properties analysis.
        Expected params:
        - volume_cm3: float (Geometry volume)
        - material_density: float (g/cm3)
        - bounding_box: [Lx, Ly, Lz] (cm) - Optional, for inertia estimation
        """
        logger.info(f"{self.name} calculating mass stats...")
        
        # Inputs
        volume_cm3 = params.get("volume_cm3", 1000.0) # Default 10x10x10 cube
        density_g_cm3 = params.get("material_density", 2.7) # Default Al-6061
        bbox = params.get("bounding_box", [10.0, 10.0, 10.0]) # Lx, Ly, Lz in cm
        
        # 1. Calculate Mass
        mass_g = volume_cm3 * density_g_cm3
        mass_kg = mass_g / 1000.0
        
        # 2. Estimate Inertia Tensor (Cuboid Approximation)
        # Ixx = m/12 * (y^2 + z^2)
        # Iyy = m/12 * (x^2 + z^2)
        # Izz = m/12 * (x^2 + y^2)
        lx, ly, lz = bbox[0]/100.0, bbox[1]/100.0, bbox[2]/100.0 # Convert cm to m for Inertia output
        
        ixx = (mass_kg / 12.0) * (ly**2 + lz**2)
        iyy = (mass_kg / 12.0) * (lx**2 + lz**2)
        izz = (mass_kg / 12.0) * (lx**2 + ly**2)
        
        # 3. Center of Gravity (CoG)
        # For a single component, we assume CoG is geometric center (0,0,0) relative to body
        # In a real assembly, this would aggregate children.
        cg = [0.0, 0.0, 0.0]
        
        # Create PhysicalValues
        pv_mass = create_physical_value(mass_kg, Unit.KILOGRAMS, source=self.name)
        
        logs = [
            f"Volume: {volume_cm3:.2f} cm³, Density: {density_g_cm3:.2f} g/cm³",
            f"Calculated Mass: {mass_kg:.4f} kg",
            f"Inertia (diag): [{ixx:.4f}, {iyy:.4f}, {izz:.4f}] kg·m²"
        ]

        return {
            "status": "success",
            "mass": pv_mass.to_dict(),
            "inertia_tensor": [ixx, iyy, izz],
            "center_of_gravity": cg,
            "logs": logs
        }
