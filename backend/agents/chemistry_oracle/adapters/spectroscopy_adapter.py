"""
Spectroscopy Adapter
Handles Beer-Lambert law, IR vibrational frequencies, NMR, and UV-Vis spectroscopy.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class SpectroscopyAdapter:
    """
    Spectroscopy Solver
    Domains: Beer-Lambert, IR, NMR, UV-Vis
    """
    
    # Physical constants
    H = 6.62607015e-34  # Planck constant (J·s)
    C = 299792458  # Speed of light (m/s)
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "BEER_LAMBERT").upper()
        
        logger.info(f"[SPECTROSCOPY] Solving {sim_type}...")
        
        if sim_type == "BEER_LAMBERT":
            return self._solve_beer_lambert(params)
        elif sim_type == "IR":
            return self._solve_ir_frequency(params)
        elif sim_type == "NMR":
            return self._solve_nmr(params)
        elif sim_type == "UV_VIS":
            return self._solve_uv_vis(params)
        else:
            return {"status": "error", "message": f"Unknown spectroscopy type: {sim_type}"}
    
    def _solve_beer_lambert(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Beer-Lambert Law: A = εcl
        A = absorbance, ε = molar absorptivity, c = concentration, l = path length
        """
        if "absorbance" in params:
            # Calculate concentration from absorbance
            A = params["absorbance"]
            epsilon = params.get("molar_absorptivity", 1000)  # M⁻¹cm⁻¹
            l = params.get("path_length_cm", 1.0)
            
            c = A / (epsilon * l)
            
            # Transmittance: T = 10^(-A)
            T = 10**(-A)
            T_percent = T * 100
            
            return {
                "status": "solved",
                "method": "Beer-Lambert Law",
                "concentration_M": float(c),
                "absorbance": A,
                "transmittance": float(T),
                "transmittance_percent": float(T_percent),
                "molar_absorptivity": epsilon
            }
        
        elif "concentration" in params:
            # Calculate absorbance from concentration
            c = params["concentration"]
            epsilon = params.get("molar_absorptivity", 1000)
            l = params.get("path_length_cm", 1.0)
            
            A = epsilon * c * l
            T = 10**(-A)
            T_percent = T * 100
            
            return {
                "status": "solved",
                "method": "Beer-Lambert Law",
                "absorbance": float(A),
                "concentration_M": c,
                "transmittance": float(T),
                "transmittance_percent": float(T_percent)
            }
        
        else:
            return {"status": "error", "message": "Need either absorbance or concentration"}
    
    def _solve_ir_frequency(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        IR Vibrational Frequency (Hooke's Law)
        ν = (1/2π)√(k/μ)
        """
        k = params.get("force_constant_n_m", 500)  # N/m
        m1 = params.get("mass1_amu", 12)  # Carbon
        m2 = params.get("mass2_amu", 16)  # Oxygen
        
        # Reduced mass: μ = m1*m2/(m1+m2)
        # Convert AMU to kg: 1 amu = 1.66054e-27 kg
        amu_to_kg = 1.66054e-27
        m1_kg = m1 * amu_to_kg
        m2_kg = m2 * amu_to_kg
        mu = (m1_kg * m2_kg) / (m1_kg + m2_kg)
        
        # Frequency (Hz)
        nu_hz = (1 / (2 * np.pi)) * np.sqrt(k / mu)
        
        # Wavenumber (cm⁻¹): ṽ = ν/c
        nu_cm_inv = nu_hz / (self.C * 100)  # Convert to cm⁻¹
        
        # Wavelength (μm)
        wavelength_um = 1e4 / nu_cm_inv
        
        return {
            "status": "solved",
            "method": "IR Vibrational Frequency (Hooke's Law)",
            "frequency_hz": float(nu_hz),
            "wavenumber_cm_inv": float(nu_cm_inv),
            "wavelength_um": float(wavelength_um),
            "reduced_mass_amu": float(mu / amu_to_kg)
        }
    
    def _solve_nmr(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        NMR Chemical Shift
        δ = (ν_sample - ν_reference) / ν_reference × 10⁶
        """
        nu_sample = params.get("sample_frequency_hz", 400.1e6)
        nu_reference = params.get("reference_frequency_hz", 400.0e6)  # TMS
        
        # Chemical shift (ppm)
        delta = ((nu_sample - nu_reference) / nu_reference) * 1e6
        
        # Coupling constant (if provided)
        J = params.get("coupling_constant_hz", None)
        
        result = {
            "status": "solved",
            "method": "NMR Chemical Shift",
            "chemical_shift_ppm": float(delta),
            "sample_frequency_hz": nu_sample,
            "reference_frequency_hz": nu_reference
        }
        
        if J is not None:
            result["coupling_constant_hz"] = J
        
        return result
    
    def _solve_uv_vis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        UV-Vis Spectroscopy
        E = hν = hc/λ
        """
        if "wavelength_nm" in params:
            # Calculate energy from wavelength
            wavelength_nm = params["wavelength_nm"]
            wavelength_m = wavelength_nm * 1e-9
            
            # Energy (J)
            E_j = (self.H * self.C) / wavelength_m
            
            # Energy (eV)
            E_ev = E_j / 1.602176634e-19
            
            # Frequency
            nu = self.C / wavelength_m
            
            return {
                "status": "solved",
                "method": "UV-Vis (Planck)",
                "wavelength_nm": wavelength_nm,
                "energy_j": float(E_j),
                "energy_ev": float(E_ev),
                "frequency_hz": float(nu)
            }
        
        elif "energy_ev" in params:
            # Calculate wavelength from energy
            E_ev = params["energy_ev"]
            E_j = E_ev * 1.602176634e-19
            
            wavelength_m = (self.H * self.C) / E_j
            wavelength_nm = wavelength_m * 1e9
            
            nu = self.C / wavelength_m
            
            return {
                "status": "solved",
                "method": "UV-Vis (Planck)",
                "energy_ev": E_ev,
                "wavelength_nm": float(wavelength_nm),
                "frequency_hz": float(nu)
            }
        
        else:
            return {"status": "error", "message": "Need either wavelength_nm or energy_ev"}
