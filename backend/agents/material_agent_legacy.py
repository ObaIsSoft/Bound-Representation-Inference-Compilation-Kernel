from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class MaterialAgent:
    """
    The 'Librarian' of Matter.
    Provides rigorous, temperature-dependent material properties.
    
    Uses Supabase ONLY for material data.
    """
    
    def __init__(self):
        self._initialized = False
        
        # Initialize Physics Kernel (Real Physics!)
        try:
            from physics import get_physics_kernel
            self.physics = get_physics_kernel()
        except ImportError:
            self.physics = None
            logger.warning("Physics kernel not available")
        
        # Initialize Oracles for advanced calculations
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            from agents.chemistry_oracle.chemistry_oracle import ChemistryOracle
            self.physics_oracle = PhysicsOracle()
            self.chemistry_oracle = ChemistryOracle()
            self.has_oracles = True
        except ImportError:
            self.physics_oracle = None
            self.chemistry_oracle = None
            self.has_oracles = False
    
    async def initialize(self):
        """Initialize the agent"""
        if self._initialized:
            return
        
        # Initialize Supabase service
        from backend.services import supabase_service
        await supabase_service.initialize()
        
        self._initialized = True
    
    async def run(self, material_name: str, temperature: float = 20.0) -> Dict[str, Any]:
        """
        Query Supabase for material properties and apply thermal degradation.
        
        Args:
            material_name: Name of material (e.g., "Aluminum 6061")
            temperature: Operating temperature in Celsius
            
        Returns:
            Material properties dictionary with thermal adjustments
        """
        await self.initialize()
        
        from backend.services import supabase_service
        
        # Query Supabase for material
        try:
            material_data = await supabase_service.get_material(material_name)
        except ValueError as e:
            logger.error(f"Material '{material_name}' not found: {e}")
            return {
                "name": material_name,
                "properties": {},
                "error": f"Material not found in database: {e}",
                "feasible": False
            }
        except RuntimeError as e:
            logger.error(f"Supabase not initialized: {e}")
            return {
                "name": material_name,
                "properties": {},
                "error": "Database service unavailable",
                "feasible": False
            }
        
        # Extract properties from database
        props = {
            "name": material_data.get("name", material_name),
            "density": material_data.get("density_kg_m3"),
            "yield_strength": material_data.get("yield_strength_mpa"),
            "ultimate_strength": material_data.get("ultimate_strength_mpa"),
            "elastic_modulus": material_data.get("elastic_modulus_gpa"),
            "melting_point": material_data.get("melting_point_c"),
            "max_temp": material_data.get("max_operating_temp_c"),
            "thermal_expansion": material_data.get("thermal_expansion_um_m_k"),
            "thermal_conductivity": material_data.get("thermal_conductivity_w_m_k"),
            "cost_per_kg_usd": material_data.get("cost_per_kg_usd"),
        }
        
        # Apply thermal degradation if temperature provided
        if temperature != 20.0 and props["yield_strength"]:
            props["yield_strength_original"] = props["yield_strength"]
            props["yield_strength"] = self._apply_thermal_degradation(
                props["yield_strength"], 
                temperature,
                props.get("melting_point")
            )
        
        # Check feasibility against max operating temp
        feasible = True
        warnings = []
        
        if props.get("max_temp") and temperature > props["max_temp"]:
            feasible = False
            warnings.append(f"Temperature {temperature}C exceeds max operating temp {props['max_temp']}C")
        
        return {
            "name": props["name"],
            "properties": props,
            "temperature_c": temperature,
            "feasible": feasible,
            "warnings": warnings,
            "source": "supabase"
        }
    
    def _apply_thermal_degradation(
        self, 
        strength: float, 
        temperature: float,
        melting_point: float = None
    ) -> float:
        """
        Apply thermal degradation to material strength.
        
        Simplified model: Linear degradation from 100% at 20C to 0% at 80% of melting point.
        """
        if not melting_point:
            # Conservative estimate: assume 500C melting point
            melting_point = 500.0
        
        # Critical temperature is 80% of melting point
        critical_temp = melting_point * 0.8
        
        if temperature <= 20.0:
            return strength
        
        if temperature >= critical_temp:
            return strength * 0.1  # 10% strength at critical temp
        
        # Linear interpolation
        degradation = (temperature - 20.0) / (critical_temp - 20.0)
        return strength * (1.0 - degradation * 0.9)
