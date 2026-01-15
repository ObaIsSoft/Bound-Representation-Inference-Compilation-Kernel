"""
Biomaterials Adapter
Handles biocompatibility, degradation in biological environments, and tissue interactions.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class BiomaterialsAdapter:
    """Biomaterials Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "DEGRADATION").upper()
        
        if sim_type == "DEGRADATION":
            return self._solve_degradation(params)
        elif sim_type == "BIOCOMPATIBILITY":
            return self._solve_biocompatibility(params)
        elif sim_type == "DRUG_RELEASE":
            return self._solve_drug_release(params)
        else:
            return {"status": "error", "message": f"Unknown biomaterial type: {sim_type}"}
    
    def _solve_degradation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Biodegradation kinetics (first-order)
        M(t) = M₀ exp(-kt)
        """
        M0 = params.get("initial_mass_kg", 1.0)
        k = params.get("degradation_rate_per_day", 0.01)
        t = params.get("time_days", 30)
        
        # Mass remaining
        M_t = M0 * np.exp(-k * t)
        
        # Half-life
        t_half = np.log(2) / k
        
        # Percent degraded
        percent_degraded = (1 - M_t/M0) * 100
        
        return {
            "status": "solved",
            "method": "Biodegradation (First-Order)",
            "mass_remaining_kg": float(M_t),
            "percent_degraded": float(percent_degraded),
            "half_life_days": float(t_half),
            "degradation_rate_per_day": k
        }
    
    def _solve_biocompatibility(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Biocompatibility assessment based on cell viability
        ISO 10993 standards
        """
        cell_viability_percent = params.get("cell_viability_percent", 85)
        
        # ISO 10993 criteria
        if cell_viability_percent >= 80:
            biocompatibility = "Non-cytotoxic"
            grade = "Excellent"
        elif cell_viability_percent >= 60:
            biocompatibility = "Slightly cytotoxic"
            grade = "Acceptable"
        elif cell_viability_percent >= 40:
            biocompatibility = "Moderately cytotoxic"
            grade = "Poor"
        else:
            biocompatibility = "Severely cytotoxic"
            grade = "Unacceptable"
        
        return {
            "status": "solved",
            "method": "ISO 10993 Biocompatibility",
            "cell_viability_percent": cell_viability_percent,
            "biocompatibility": biocompatibility,
            "grade": grade
        }
    
    def _solve_drug_release(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Higuchi model for drug release
        Q = √(D(2C₀ - Cs)Cs·t)
        """
        D = params.get("diffusion_coefficient_m2_s", 1e-12)
        C0 = params.get("initial_concentration_kg_m3", 100)
        Cs = params.get("solubility_kg_m3", 10)
        t = params.get("time_s", 3600)
        
        # Higuchi equation
        Q = np.sqrt(D * (2*C0 - Cs) * Cs * t)
        
        return {
            "status": "solved",
            "method": "Higuchi Drug Release Model",
            "cumulative_release_kg_m2": float(Q),
            "time_s": t
        }
