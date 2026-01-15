"""
Materials Science Adapter
Handles phase transitions, crystal structures, and composites.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MaterialsAdapter:
    """
    Materials Science Solver
    Domains: Phase Transitions, Crystals, Composites
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "PHASE").upper()
        
        logger.info(f"[MATERIALS] Solving {sim_type}...")
        
        if sim_type == "PHASE":
            return self._solve_phase(params)
        elif sim_type == "CRYSTAL":
            return self._solve_crystal(params)
        elif sim_type == "COMPOSITE":
            return self._solve_composite(params)
        else:
            return {"status": "error", "message": f"Unknown materials type: {sim_type}"}
    
    def _solve_phase(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Phase transition calculations"""
        material = params.get("material", "WATER").upper()
        
        # Common materials database
        materials_db = {
            "WATER": {"melt_k": 273.15, "boil_k": 373.15, "latent_heat_jkg": 2.26e6},
            "IRON": {"melt_k": 1811, "boil_k": 3134, "latent_heat_jkg": 6.09e6},
            "ALUMINUM": {"melt_k": 933, "boil_k": 2792, "latent_heat_jkg": 10.7e6}
        }
        
        props = materials_db.get(material, materials_db["WATER"])
        temperature = params.get("temperature_k", 300.0)
        
        # Determine phase
        if temperature < props["melt_k"]:
            phase = "Solid"
        elif temperature < props["boil_k"]:
            phase = "Liquid"
        else:
            phase = "Gas"
        
        # Energy for phase change
        mass = params.get("mass_kg", 1.0)
        energy_required = mass * props["latent_heat_jkg"]
        
        return {
            "status": "solved",
            "method": "Phase Diagram",
            "current_phase": phase,
            "melting_point_k": props["melt_k"],
            "boiling_point_k": props["boil_k"],
            "latent_heat_jkg": props["latent_heat_jkg"],
            "energy_for_transition_j": float(energy_required)
        }
    
    def _solve_crystal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Crystal structure analysis"""
        lattice_type = params.get("lattice", "FCC").upper()
        lattice_param = params.get("lattice_parameter_m", 4.05e-10)  # Aluminum
        
        # Atomic packing factor
        apf_values = {
            "SC": 0.52,   # Simple Cubic
            "BCC": 0.68,  # Body-Centered Cubic
            "FCC": 0.74,  # Face-Centered Cubic
            "HCP": 0.74   # Hexagonal Close-Packed
        }
        
        apf = apf_values.get(lattice_type, 0.74)
        
        # Number of atoms per unit cell
        atoms_per_cell = {
            "SC": 1,
            "BCC": 2,
            "FCC": 4,
            "HCP": 6
        }
        
        n_atoms = atoms_per_cell.get(lattice_type, 4)
        
        # Unit cell volume
        volume = lattice_param**3
        
        return {
            "status": "solved",
            "method": "Crystallography",
            "lattice_type": lattice_type,
            "atomic_packing_factor": float(apf),
            "atoms_per_cell": n_atoms,
            "unit_cell_volume_m3": float(volume)
        }
    
    def _solve_composite(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Composite material properties (Rule of Mixtures)"""
        # Matrix properties
        E_matrix = params.get("matrix_modulus_pa", 3.5e9)  # Epoxy
        rho_matrix = params.get("matrix_density_kgm3", 1200)
        
        # Fiber properties
        E_fiber = params.get("fiber_modulus_pa", 230e9)  # Carbon fiber
        rho_fiber = params.get("fiber_density_kgm3", 1800)
        
        # Volume fraction
        V_f = params.get("fiber_volume_fraction", 0.6)
        V_m = 1 - V_f
        
        # Rule of mixtures (parallel)
        E_composite = E_fiber * V_f + E_matrix * V_m
        rho_composite = rho_fiber * V_f + rho_matrix * V_m
        
        # Specific stiffness
        specific_stiffness = E_composite / rho_composite
        
        return {
            "status": "solved",
            "method": "Rule of Mixtures",
            "composite_modulus_pa": float(E_composite),
            "composite_density_kgm3": float(rho_composite),
            "specific_stiffness_nm2kg": float(specific_stiffness),
            "strength_improvement": float(E_composite / E_matrix)
        }
