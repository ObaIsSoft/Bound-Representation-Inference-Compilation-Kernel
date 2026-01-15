"""Nanomaterials Adapter"""
import numpy as np
from typing import Dict, Any

class NanomaterialsAdapter:
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "MELTING_POINT").upper()
        
        if sim_type == "MELTING_POINT":
            # T_m(r) = T_m,bulk(1 - α/r)
            T_m_bulk = params.get("bulk_melting_point_k", 1358)  # Gold
            alpha = params.get("alpha_nm", 1.0)
            r = params.get("particle_radius_nm", 5)
            T_m = T_m_bulk * (1 - alpha / r)
            depression = T_m_bulk - T_m
            return {"status": "solved", "method": "Melting Point Depression", "melting_point_k": float(T_m), "depression_k": float(depression)}
        
        elif sim_type == "QUANTUM_CONFINEMENT":
            # E_g,eff = E_g,bulk + ħ²π²/(2μr²)
            E_g_bulk = params.get("bulk_band_gap_ev", 1.12)  # Silicon
            h_bar = 1.054571817e-34
            m_e = 9.10938356e-31
            r = params.get("particle_radius_nm", 5) * 1e-9
            delta_E = (h_bar**2 * np.pi**2) / (2 * m_e * r**2) / 1.602176634e-19  # Convert to eV
            E_g_eff = E_g_bulk + delta_E
            return {"status": "solved", "method": "Quantum Confinement", "effective_band_gap_ev": float(E_g_eff), "confinement_energy_ev": float(delta_E)}
        
        return {"status": "error", "message": "Unknown nano type"}
