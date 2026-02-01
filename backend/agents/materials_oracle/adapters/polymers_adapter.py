
import math
from typing import Dict, Any, List

class PolymersAdapter:
    """
    Simulates Polymer Synthesis and Performance.
    Handles 'Molecular Engineering' for:
    - Ballistics (High Strain Rate)
    - Space Environments (Outgassing, Radiation)
    - Optical Transparency
    """
    
    def __init__(self):
        try:
            from materials.materials_db import MaterialsDatabase
        except ImportError:
            from materials.materials_db import MaterialsDatabase
        self.db = MaterialsDatabase()
        # Initial load attempt
        self.known_monomers = self._load_monomers()

    def _load_monomers(self) -> Dict[str, Any]:
        """Fetch monomers via MaterialsDatabase."""
        try:
            return self.db.get_monomers()
        except Exception:
            return {}

    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point.
        params can specify:
        - type: 'SYNTHESIS', 'BALLISTIC', 'SPACE'
        """
        sim_type = params.get("type", "SYNTHESIS")
        
        if sim_type == "SYNTHESIS":
            return self._synthesize(params)
        elif sim_type == "BALLISTIC":
            return self._ballistic_test(params)
        elif sim_type == "SPACE":
            return self._space_qualification(params)
        else:
            return {"error": f"Unknown simulation type: {sim_type}"}

    def _synthesize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predicts properties from monomers and processing.
        """
        monomer = params.get("monomer", "ETHYLENE").upper()
        chain_length = params.get("chain_length", 1000)
        cross_link_density = params.get("cross_link_density", 0.0) # 0 to 1
        alignment = params.get("chain_alignment", 0.5) # 0 (Random) to 1 (Ordered Fiber)
        
        # Safe lookup with fallback to Ethylene if unknown (but preferable to fail or warn?)
        # For robustness, we fallback to a default structure if not found.
        # But here we rely on DB properties.
        default_props = {"mw": 28.05, "stiffness": 1.0, "density_g_cc": 0.95, "base_strength_mpa": 30.0, "alignment_bonus": 3.0}
        base_props = self.known_monomers.get(monomer, default_props)
        
        # 1. Molecular Weight
        mw_polymer = base_props.get("mw", 28.0) * chain_length
        
        # 2. Tensile Strength (MPa) depends on Alignment and Cross-linking
        base_strength = base_props.get("base_strength_mpa", 30.0)
        
        # Alignment Factor: Exponential bonus for fiber spinning
        align_factor = 1.0 + (alignment * 100.0) 
        
        # Apply material specific bonus (e.g. Aramid stacking, UHMWPE crystallization)
        # Bonus loaded from DB
        align_bonus = base_props.get("alignment_bonus", 1.0)
        align_factor *= align_bonus
        
        tensile_strength = base_strength * align_factor * (1.0 + cross_link_density)
        
        # 3. Young's Modulus (GPa)
        base_modulus = base_props.get("stiffness", 1.0) # GPa
        modulus = base_modulus * (1.0 + alignment * 2.0)
        
        # 4. Density
        density = base_props.get("density_g_cc", 1.0)
        
        results = {
            "molecular_weight_kda": mw_polymer / 1000.0,
            "tensile_strength_mpa": tensile_strength,
            "youngs_modulus_gpa": modulus,
            "density_g_cc": density,
            "classification": self._classify(tensile_strength, modulus)
        }
        
        return {
            "status": "synthesized",
            "properties": results
        }
        
    def _ballistic_test(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simulates High-Velocity Impact.
        """
        thickness_mm = params.get("thickness_mm", 10.0)
        material_props = params.get("material_properties", {})
        projectile = params.get("projectile", "50_BMG")
        
        # Projectile Data
        try:
            threats = self.db.get_ballistic_threats()
        except:
            threats = {}
        
        threat = threats.get(projectile)
        if not threat:
            # Fallback if specific threat not found in DB
             threat = {"mass_g": 46.0, "velocity_mps": 900} # Default 50 BMG-ish
             
        m_kg = threat["mass_g"] / 1000.0
        v = threat["velocity_mps"]
        ke_joules = 0.5 * m_kg * (v ** 2)
        
        # Material Resistance - Simplified Energy Absorption
        strength = material_props.get("tensile_strength_mpa", 500.0)
        density = material_props.get("density_g_cc", 1.0) # g/cc = 1000 kg/m3
        
        # Heuristic for Specific Absorption (E_spec) derived from Tensile strength
        e_spec_j_kg = (strength / 3000.0) * 5000.0
        if e_spec_j_kg < 50: e_spec_j_kg = 50.0
        
        # Mass of Armor Plate Area
        rho_kg_m3 = density * 1000.0
        dist_m = thickness_mm / 1000.0
        
        areal_density = rho_kg_m3 * dist_m # kg/m2
        
        # Stopping Power Calculation
        eff_factor = strength * 0.1 # Heuristic tuning
        energy_stopped = areal_density * eff_factor
        
        stopped = energy_stopped >= ke_joules
        
        return {
            "threat": projectile,
            "impact_energy_j": int(ke_joules),
            "energy_stopped_j": int(energy_stopped),
            "result": "STOPPED" if stopped else "PENETRATION",
            "safety_margin": round(energy_stopped / ke_joules, 2) if ke_joules > 0 else 0,
            "thickness_mm": thickness_mm,
            "areal_density_kg_m2": round(areal_density, 2)
        }

    def _space_qualification(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Checks Suitability for Vacuum/Radiation using DB properties.
        """
        material_props = params.get("material_properties", {})
        monomer_name = params.get("monomer_name", "ETHYLENE").upper()
        
        # Fetch properties from known monomers
        props = self.known_monomers.get(monomer_name, {})
        
        # Outgassing (TML/CVCM)
        # Default to False (Unsafe) if unknown, unless we want to be permissive.
        # "outgassing_compliant" should be 1 (True) or 0 (False).
        is_safe = bool(props.get("outgassing_compliant", False))
        
        issues = []
        if not is_safe:
            issues.append(f"High Outgassing Risk: {monomer_name}")
            
        # Radiation Shielding (GCR/SPE)
        shielding_score = props.get("radiation_shielding_score", 50)
            
        return {
            "space_qualified": is_safe,
            "outgassing_check": "PASS" if is_safe else "FAIL",
            "radiation_shielding_score": shielding_score,
            "issues": issues
        }

    def _classify(self, strength, modulus):
        if strength > 1000: return "HIGH_PERFORMANCE_FIBER"
        if strength > 100: return "ENGINEERING_PLASTIC"
        return "COMMODITY_PLASTIC"
