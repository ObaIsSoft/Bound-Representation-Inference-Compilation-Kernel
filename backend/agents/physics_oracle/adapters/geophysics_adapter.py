"""
Geophysics Adapter
Handles seismic analysis, gravitational fields, and magnetic fields.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class GeophysicsAdapter:
    """
    Geophysics Solver
    Domains: Seismic, Gravity, Magnetism
    """
    
    # Earth parameters
    G = 6.67430e-11  # Gravitational constant
    EARTH_MASS = 5.972e24  # kg
    EARTH_RADIUS = 6.371e6  # m
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Main entry point"""
        sim_type = params.get("type", "SEISMIC").upper()
        
        logger.info(f"[GEOPHYSICS] Solving {sim_type}...")
        
        if sim_type == "SEISMIC":
            return self._solve_seismic(params)
        elif sim_type == "GRAVITY":
            return self._solve_gravity(params)
        elif sim_type == "MAGNETIC":
            return self._solve_magnetic(params)
        else:
            return {"status": "error", "message": f"Unknown geophysics type: {sim_type}"}
    
    def _solve_seismic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Seismic wave propagation"""
        # Wave velocities in Earth's crust
        v_p = params.get("p_wave_velocity_mps", 6000.0)  # P-wave
        v_s = params.get("s_wave_velocity_mps", 3500.0)  # S-wave
        
        # Earthquake parameters
        magnitude = params.get("magnitude", 5.0)  # Richter scale
        distance = params.get("distance_m", 100e3)  # 100 km
        
        # Energy release: log₁₀(E) = 4.8 + 1.5*M
        energy_j = 10**(4.8 + 1.5 * magnitude)
        
        # Travel time
        t_p = distance / v_p
        t_s = distance / v_s
        
        # S-P interval
        sp_interval = t_s - t_p
        
        # Ground motion (simplified)
        amplitude = energy_j / (4 * np.pi * distance**2)
        
        return {
            "status": "solved",
            "method": "Seismic Wave Theory",
            "energy_release_j": float(energy_j),
            "p_wave_arrival_s": float(t_p),
            "s_wave_arrival_s": float(t_s),
            "sp_interval_s": float(sp_interval),
            "ground_motion_amplitude": float(amplitude),
            "severity": "Major" if magnitude > 7 else "Moderate" if magnitude > 5 else "Minor"
        }
    
    def _solve_gravity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Gravitational field calculations"""
        # Location parameters
        latitude = params.get("latitude_deg", 45.0)
        altitude = params.get("altitude_m", 0.0)
        
        # Distance from Earth's center
        r = self.EARTH_RADIUS + altitude
        
        # Gravitational acceleration: g = GM/r²
        g = self.G * self.EARTH_MASS / r**2
        
        # Centrifugal correction (due to Earth's rotation)
        omega = 7.2921e-5  # Earth's angular velocity (rad/s)
        lat_rad = np.radians(latitude)
        centrifugal = omega**2 * r * np.cos(lat_rad)**2
        
        # Effective gravity
        g_eff = g - centrifugal
        
        # Gravity anomaly (deviation from standard 9.81 m/s²)
        anomaly = g_eff - 9.81
        
        return {
            "status": "solved",
            "method": "Newton's Law of Gravitation",
            "gravitational_acceleration_mps2": float(g),
            "effective_gravity_mps2": float(g_eff),
            "gravity_anomaly_mgal": float(anomaly * 1e5),  # milligals
            "altitude_correction_mps2": float(g - 9.81)
        }
    
    def _solve_magnetic(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Magnetic field calculations"""
        # Location parameters
        latitude = params.get("latitude_deg", 45.0)
        longitude = params.get("longitude_deg", 0.0)
        
        # Simplified dipole model
        # Field strength varies with latitude
        lat_rad = np.radians(latitude)
        
        # Total field intensity (approximate)
        # F = M/r³ * √(1 + 3sin²λ)
        # where M is magnetic moment, λ is latitude
        F_equator = 30000e-9  # 30,000 nT at equator
        F = F_equator * np.sqrt(1 + 3 * np.sin(lat_rad)**2)
        
        # Inclination (dip angle)
        inclination = np.degrees(np.arctan(2 * np.tan(lat_rad)))
        
        # Declination (simplified, varies by location)
        declination = params.get("declination_deg", 0.0)
        
        return {
            "status": "solved",
            "method": "Dipole Model",
            "total_field_nt": float(F * 1e9),  # Convert to nT
            "inclination_deg": float(inclination),
            "declination_deg": float(declination),
            "horizontal_component_nt": float(F * np.cos(np.radians(inclination)) * 1e9),
            "vertical_component_nt": float(F * np.sin(np.radians(inclination)) * 1e9)
        }
