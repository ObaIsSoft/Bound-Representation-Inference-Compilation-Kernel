"""
Electromagnetism Adapter
Handles Maxwell's equations, antenna design, and EMI analysis.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ElectromagnetismAdapter:
    """
    Electromagnetic Field Solver
    Domains: Maxwell's Equations, Antenna Design, EMI
    """
    
    # Physical constants
    C = 299792458  # Speed of light (m/s)
    MU_0 = 4 * np.pi * 1e-7  # Permeability of free space (H/m)
    EPSILON_0 = 8.854187817e-12  # Permittivity of free space (F/m)
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for EM simulations.
        """
        sim_type = params.get("type", "FIELD").upper()
        
        logger.info(f"[ELECTROMAGNETISM] Solving {sim_type}...")
        
        if sim_type == "FIELD":
            return self._solve_field(params)
        elif sim_type == "ANTENNA":
            return self._solve_antenna(params)
        elif sim_type == "EMI":
            return self._solve_emi(params)
        elif sim_type == "WAVE":
            return self._solve_wave(params)
        else:
            return {"status": "error", "message": f"Unknown EM type: {sim_type}"}
    
    def _solve_field(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Electric and Magnetic Field Calculations
        """
        # Point charge electric field: E = kQ/r²
        charge = params.get("charge_c", 1e-6)  # 1 μC
        distance = params.get("distance_m", 1.0)
        
        k = 1 / (4 * np.pi * self.EPSILON_0)  # Coulomb's constant
        E_field = k * charge / (distance**2)
        
        # Current-carrying wire magnetic field: B = μ₀I/(2πr)
        current = params.get("current_a", 1.0)
        B_field = (self.MU_0 * current) / (2 * np.pi * distance)
        
        # Poynting vector (energy flux): S = E × B / μ₀
        S = (E_field * B_field) / self.MU_0
        
        return {
            "status": "solved",
            "method": "Maxwell's Equations",
            "electric_field_vm": float(E_field),
            "magnetic_field_t": float(B_field),
            "poynting_vector_wm2": float(S),
            "field_energy_density_jm3": float(0.5 * self.EPSILON_0 * E_field**2)
        }
    
    def _solve_antenna(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Antenna Design: Dipole antenna calculations
        """
        # Antenna parameters
        frequency = params.get("frequency_hz", 1e9)  # 1 GHz default
        power = params.get("power_w", 100.0)
        distance = params.get("distance_m", 1000.0)
        
        # Wavelength
        wavelength = self.C / frequency
        
        # Optimal dipole length (λ/2)
        dipole_length = wavelength / 2
        
        # Radiation resistance (dipole)
        R_rad = 73  # Ohms (half-wave dipole)
        
        # Directivity (dipole)
        D = 1.64  # Linear (2.15 dBi)
        
        # Effective Isotropic Radiated Power
        EIRP = power * D
        
        # Free space path loss: FSPL = (4πd/λ)²
        FSPL = (4 * np.pi * distance / wavelength)**2
        FSPL_dB = 10 * np.log10(FSPL)
        
        # Received power (Friis equation)
        P_rx = EIRP / FSPL
        
        # Gain (dBi)
        gain_dbi = 10 * np.log10(D)
        
        return {
            "status": "solved",
            "method": "Dipole Antenna Theory",
            "wavelength_m": float(wavelength),
            "optimal_length_m": float(dipole_length),
            "radiation_resistance_ohm": float(R_rad),
            "directivity": float(D),
            "gain_dbi": float(gain_dbi),
            "eirp_w": float(EIRP),
            "path_loss_db": float(FSPL_dB),
            "received_power_w": float(P_rx)
        }
    
    def _solve_emi(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Electromagnetic Interference Analysis
        """
        # Shield parameters
        thickness = params.get("shield_thickness_m", 0.001)  # 1mm
        conductivity = params.get("conductivity_sm", 5.96e7)  # Copper
        frequency = params.get("frequency_hz", 1e6)  # 1 MHz
        
        # Skin depth: δ = √(2/(ωμσ))
        omega = 2 * np.pi * frequency
        skin_depth = np.sqrt(2 / (omega * self.MU_0 * conductivity))
        
        # Shielding effectiveness (dB): SE = 20 log₁₀(e^(t/δ))
        # Simplified: SE ≈ 8.69 * (t/δ) for t >> δ
        if thickness > 3 * skin_depth:
            SE = 8.69 * (thickness / skin_depth)
        else:
            SE = 20 * np.log10(np.exp(thickness / skin_depth))
        
        # Attenuation factor
        attenuation = 10**(-SE/20)
        
        return {
            "status": "solved",
            "method": "Shielding Effectiveness",
            "skin_depth_m": float(skin_depth),
            "shielding_effectiveness_db": float(SE),
            "attenuation_factor": float(attenuation),
            "shield_quality": "Excellent" if SE > 60 else "Good" if SE > 40 else "Moderate"
        }
    
    def _solve_wave(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Electromagnetic Wave Propagation
        """
        frequency = params.get("frequency_hz", 1e9)
        distance = params.get("distance_m", 1000.0)
        power = params.get("power_w", 100.0)
        
        # Wavelength and wave number
        wavelength = self.C / frequency
        k = 2 * np.pi / wavelength
        
        # Angular frequency
        omega = 2 * np.pi * frequency
        
        # Characteristic impedance of free space
        Z_0 = np.sqrt(self.MU_0 / self.EPSILON_0)  # ~377 Ohms
        
        # Power density at distance (inverse square law)
        power_density = power / (4 * np.pi * distance**2)
        
        # Electric field amplitude
        E_0 = np.sqrt(2 * Z_0 * power_density)
        
        # Magnetic field amplitude
        H_0 = E_0 / Z_0
        
        return {
            "status": "solved",
            "method": "Wave Propagation",
            "wavelength_m": float(wavelength),
            "wave_number_radm": float(k),
            "angular_frequency_rads": float(omega),
            "impedance_ohm": float(Z_0),
            "power_density_wm2": float(power_density),
            "electric_field_vm": float(E_0),
            "magnetic_field_am": float(H_0)
        }
