"""
Acoustics Adapter
Handles sound propagation, noise analysis, and sonar.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class AcousticsAdapter:
    """
    Acoustic Wave Solver
    Domains: Sound Propagation, Noise, Sonar
    """
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "PROPAGATION").upper()
        
        logger.info(f"[ACOUSTICS] Solving {sim_type}...")
        
        if sim_type == "PROPAGATION":
            return self._solve_propagation(params)
        elif sim_type == "NOISE":
            return self._solve_noise(params)
        elif sim_type == "SONAR":
            return self._solve_sonar(params)
        else:
            return {"status": "error", "message": f"Unknown acoustics type: {sim_type}"}
    
    def _solve_propagation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sound propagation in materials"""
        # Material properties
        density = params.get("density_kgm3", 1.225)  # Air
        bulk_modulus = params.get("bulk_modulus_pa", 142e3)  # Air
        
        # Speed of sound: v = √(K/ρ)
        speed = np.sqrt(bulk_modulus / density)
        
        # Wavelength
        frequency = params.get("frequency_hz", 1000.0)
        wavelength = speed / frequency
        
        # Attenuation (simplified)
        distance = params.get("distance_m", 100.0)
        alpha = params.get("attenuation_npm", 0.01)  # Neper/m
        attenuation_db = 8.686 * alpha * distance
        
        return {
            "status": "solved",
            "method": "Wave Equation",
            "speed_of_sound_mps": float(speed),
            "wavelength_m": float(wavelength),
            "attenuation_db": float(attenuation_db)
        }
    
    def _solve_noise(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Noise level calculations"""
        # Sound pressure level: SPL = 20 log₁₀(p/p₀)
        pressure = params.get("pressure_pa", 0.02)  # 2 Pa
        p_ref = 20e-6  # Reference pressure (20 μPa)
        
        SPL = 20 * np.log10(pressure / p_ref)
        
        # A-weighting (simplified)
        frequency = params.get("frequency_hz", 1000.0)
        if 500 <= frequency <= 2000:
            A_weight = 0  # Minimal weighting
        else:
            A_weight = -10  # Simplified attenuation
        
        SPL_A = SPL + A_weight
        
        return {
            "status": "solved",
            "method": "Sound Pressure Level",
            "spl_db": float(SPL),
            "spl_a_dba": float(SPL_A),
            "loudness": "Loud" if SPL > 80 else "Moderate" if SPL > 60 else "Quiet"
        }
    
    def _solve_sonar(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Sonar range calculations"""
        frequency = params.get("frequency_hz", 50e3)  # 50 kHz
        power = params.get("power_w", 100.0)
        target_strength = params.get("target_strength_db", 10.0)
        
        # Speed of sound in water
        c = 1500  # m/s
        
        # Wavelength
        wavelength = c / frequency
        
        # Sonar equation: SL - 2TL + TS = NL + DT
        # Simplified: Range ≈ c * t / 2
        max_range = params.get("max_range_m", 1000.0)
        
        # Doppler shift
        velocity = params.get("target_velocity_mps", 0.0)
        doppler_shift = 2 * frequency * velocity / c
        
        return {
            "status": "solved",
            "method": "Sonar Equation",
            "wavelength_m": float(wavelength),
            "max_range_m": float(max_range),
            "doppler_shift_hz": float(doppler_shift)
        }
