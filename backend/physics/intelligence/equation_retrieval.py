"""
Equation Retrieval

LLM-based intelligent equation selection.
"""

import logging
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class EquationRetrieval:
    """
    Uses LLM to intelligently retrieve relevant physics equations.
    """
    
    def __init__(self, llm_provider=None):
        """
        Initialize equation retrieval with LLM.
        
        Args:
            llm_provider: LLM provider for equation selection
        """
        self.llm = llm_provider
    
    def retrieve(self, query: str) -> Dict[str, Any]:
        """
        Retrieve relevant equation for a physics problem.
        
        Args:
            query: Natural language physics question
        
        Returns:
            Equation information (domain, equation, variables)
        """
        if not self.llm:
            logger.warning("No LLM provider, falling back to keyword matching")
            return self._keyword_match(query)
        
        prompt = f"""
Physics Problem: {query}

Return the relevant physics equation(s) in this JSON format:
{{
    "domain": "mechanics|thermodynamics|electromagnetism|fluids|...",
    "equation": "F = m * a",
    "equation_name": "Newton's Second Law",
    "variables": {{
        "F": {{"name": "force", "unit": "N"}},
        "m": {{"name": "mass", "unit": "kg"}},
        "a": {{"name": "acceleration", "unit": "m/s^2"}}
    }},
    "library_function": "domain.calculate_force(mass, acceleration)"
}}
"""
        
        try:
            response = self.llm.generate(prompt)
            return json.loads(response)
        except Exception as e:
            logger.error(f"LLM equation retrieval failed: {e}")
            return self._keyword_match(query)
    
    def _keyword_match(self, query: str) -> Dict[str, Any]:
        """
        Fallback keyword-based equation matching.
        
        Args:
            query: Physics query
        
        Returns:
            Best-guess equation information
        """
        query_lower = query.lower()
        
        # Simple keyword matching
        if "stress" in query_lower:
            return {
                "domain": "structures",
                "equation": "σ = F/A",
                "equation_name": "Normal Stress",
                "variables": {
                    "σ": {"name": "stress", "unit": "Pa"},
                    "F": {"name": "force", "unit": "N"},
                    "A": {"name": "area", "unit": "m^2"}
                },
                "library_function": "structures.calculate_stress(force, area)"
            }
        elif "drag" in query_lower or "aerodynamic" in query_lower:
            return {
                "domain": "fluids",
                "equation": "F_D = 0.5 * ρ * v^2 * C_D * A",
                "equation_name": "Drag Force",
                "variables": {
                    "F_D": {"name": "drag_force", "unit": "N"},
                    "ρ": {"name": "density", "unit": "kg/m^3"},
                    "v": {"name": "velocity", "unit": "m/s"},
                    "C_D": {"name": "drag_coefficient", "unit": "dimensionless"},
                    "A": {"name": "area", "unit": "m^2"}
                },
                "library_function": "fluids.calculate_drag_force(velocity, density, area, drag_coefficient)"
            }
        elif "heat" in query_lower or "thermal" in query_lower:
            return {
                "domain": "thermodynamics",
                "equation": "Q = k * A * ΔT / d",
                "equation_name": "Heat Conduction",
                "variables": {
                    "Q": {"name": "heat_transfer_rate", "unit": "W"},
                    "k": {"name": "thermal_conductivity", "unit": "W/m⋅K"},
                    "A": {"name": "area", "unit": "m^2"},
                    "ΔT": {"name": "temperature_difference", "unit": "K"},
                    "d": {"name": "thickness", "unit": "m"}
                },
                "library_function": "thermodynamics.calculate_heat_conduction(...)"
            }
        # --- Expanded Domains ---
        elif any(k in query_lower for k in ["gravity", "orbit", "planet", "escape velocity"]):
            if "escape" in query_lower:
                return {
                    "domain": "mechanics",
                    "equation": "v_e = sqrt(2GM/r)",
                    "equation_name": "Escape Velocity",
                    "variables": {
                        "v_e": {"name": "escape_velocity", "unit": "m/s"},
                        "M": {"name": "mass", "unit": "kg"},
                        "r": {"name": "radius", "unit": "m"}
                    },
                    "library_function": "mechanics.calculate_escape_velocity(mass, radius)"
                }
            elif "orbit" in query_lower:
                return {
                    "domain": "mechanics",
                    "equation": "T = 2π * sqrt(r^3/GM)",
                    "equation_name": "Orbital Period",
                    "variables": {
                        "T": {"name": "orbital_period", "unit": "s"},
                        "M": {"name": "mass", "unit": "kg"},
                        "r": {"name": "radius", "unit": "m"}
                    },
                    "library_function": "mechanics.calculate_orbital_period(mass, radius)"
                }
            else:
                return {
                    "domain": "mechanics",
                    "equation": "F = G * m1 * m2 / r^2",
                    "equation_name": "Gravitational Force",
                    "variables": {
                        "F": {"name": "gravitational_force", "unit": "N"},
                        "m1": {"name": "mass_1", "unit": "kg"},
                        "m2": {"name": "mass_2", "unit": "kg"},
                        "r": {"name": "distance", "unit": "m"}
                    },
                    "library_function": "mechanics.calculate_gravitational_force(mass_1, mass_2, distance)"
                }
                
        elif any(k in query_lower for k in ["beam", "deflection", "buckling"]):
            if "buckling" in query_lower:
                 return {
                    "domain": "structures",
                    "equation": "P_cr = π^2 * E * I / L^2",
                    "equation_name": "Euler Buckling Load",
                    "variables": {
                        "P_cr": {"name": "buckling_load", "unit": "N"},
                        "E": {"name": "youngs_modulus", "unit": "Pa"},
                        "I": {"name": "moment_of_inertia", "unit": "m^4"},
                        "L": {"name": "length", "unit": "m"}
                    },
                    "library_function": "structures.calculate_buckling_load(youngs_modulus, moment_of_inertia, length)"
                }
            else:
                 return {
                    "domain": "structures",
                    "equation": "δ = F * L^3 / (48 * E * I)",
                    "equation_name": "Beam Deflection (Simply Supported)",
                    "variables": {
                        "δ": {"name": "deflection", "unit": "m"},
                        "F": {"name": "force", "unit": "N"},
                        "L": {"name": "length", "unit": "m"},
                        "E": {"name": "youngs_modulus", "unit": "Pa"},
                        "I": {"name": "moment_of_inertia", "unit": "m^4"}
                    },
                    "library_function": "structures.calculate_beam_deflection(force, length, youngs_modulus, moment_of_inertia)"
                }
        
        elif any(k in query_lower for k in ["photon", "quantum", "wavelength", "uncertainty"]):
            if "wavelength" in query_lower and "mass" in query_lower:
                return {
                    "domain": "quantum",
                    "equation": "λ = h / p",
                    "equation_name": "De Broglie Wavelength",
                    "variables": {
                        "λ": {"name": "wavelength", "unit": "m"},
                        "m": {"name": "mass", "unit": "kg"},
                        "v": {"name": "velocity", "unit": "m/s"}
                    },
                    "library_function": "quantum.calculate_de_broglie_wavelength(mass, velocity)"
                }
            elif "uncertainty" in query_lower:
                 return {
                    "domain": "quantum",
                    "equation": "Δx * Δp >= ħ/2",
                    "equation_name": "Heisenberg Uncertainty",
                    "variables": {
                        "Δp": {"name": "uncertainty_momentum", "unit": "kg m/s"},
                        "Δx": {"name": "delta_x", "unit": "m"}
                    },
                    "library_function": "quantum.calculate_uncertainty_momentum(delta_x)"
                }
            else:
                return {
                    "domain": "quantum",
                    "equation": "E = h * f",
                    "equation_name": "Photon Energy",
                    "variables": {
                        "E": {"name": "energy", "unit": "J"},
                        "f": {"name": "frequency", "unit": "Hz"}
                    },
                    "library_function": "quantum.calculate_photon_energy(frequency)"
                }

        elif any(k in query_lower for k in ["neuron", "neural", "nerve", "bio", "nernst"]):
            return {
                "domain": "electromagnetism",
                "equation": "V = (RT/zF) * ln([out]/[in])",
                "equation_name": "Nernst Potential",
                "variables": {
                    "V": {"name": "membrane_potential", "unit": "V"},
                    "T": {"name": "temperature", "unit": "K"},
                    "z": {"name": "ion_charge", "unit": "elementary_charge"},
                    "cout": {"name": "concentration_out", "unit": "mM"},
                    "cin": {"name": "concentration_in", "unit": "mM"}
                },
                "library_function": "electromagnetism.calculate_nernst_potential(temperature, ion_charge, concentration_out, concentration_in)"
            }
        else:
            return {
                "domain": "unknown",
                "equation": "Unknown",
                "equation_name": "Unknown",
                "variables": {},
                "library_function": None
            }
