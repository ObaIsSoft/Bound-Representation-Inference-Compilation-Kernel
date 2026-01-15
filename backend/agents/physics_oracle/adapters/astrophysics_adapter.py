
import logging
import numpy as np
from typing import Dict, Any

logger = logging.getLogger(__name__)

class AstrophysicsAdapter:
    """
    Astrophysics & Space Mission Oracle.
    1. Orbital Mechanics: Keplerian Propagator & Hohmann Transfers.
    2. Rocket Science: Tsiolkovsky Equation (Fuel Sizing).
    """
    
    def __init__(self):
        self.name = "Astrophysics-Solver"
        
        # Constants
        self.G = 6.674e-11
        self.M_earth = 5.972e24
        self.M_sun = 1.989e30
        self.g0 = 9.81
        
        # Astronomical Units (m)
        self.AU = 1.496e11
        
    def run_simulation(self, params: dict) -> dict:
        """
        Run Astrophysics Calculation.
        Params:
            - type: 'ORBIT', 'TRANSFER', 'ROCKET'
        """
        sim_type = params.get("type", "UNKNOWN").upper()
        
        if sim_type == "ORBIT":
            return self._solve_orbit(params)
        elif sim_type == "TRANSFER":
            return self._solve_transfer(params)
        elif sim_type == "ROCKET":
            return self._solve_rocket(params)
        else:
            return {"status": "error", "message": f"Unknown astrophysics simulation type: {sim_type}"}

    def _solve_orbit(self, params):
        """
        Keplerian 2-Body Orbit Solver.
        Calculates Period and Velocity at altitude.
        """
        logger.info("[ASTRO] Solving Keplerian Orbit...")
        
        body = params.get("central_body", "EARTH").upper()
        mu = self.G * self.M_sun if body == "SUN" else self.G * self.M_earth
        
        # Inputs: Altitude (km) or Radius (m)
        alt_km = params.get("altitude_km", 35786) # GEO default
        R_body = 6371000 if body == "EARTH" else 696340000
        r = R_body + alt_km * 1000
        
        # Circular Velocity v = sqrt(mu/r)
        v = np.sqrt(mu / r)
        
        # Period T = 2*pi * sqrt(r^3/mu)
        T = 2 * np.pi * np.sqrt(r**3 / mu)
        
        return {
            "status": "solved",
            "method": "Kepler 2-Body",
            "orbit_radius_km": r / 1000,
            "period_seconds": T,
            "period_hours": T / 3600,
            "velocity_km_s": v / 1000
        }

    def _solve_transfer(self, params):
        """
        Hohmann Transfer Calculator.
        Calculates Delta V required between two circular orbits.
        """
        logger.info("[ASTRO] Calculating Hohmann Transfer...")
        
        body = params.get("central_body", "SUN").upper()
        mu = self.G * self.M_sun if body == "SUN" else self.G * self.M_earth
        
        # Radii (e.g. Earth to Mars)
        # Default Earth(1 AU) -> Mars(1.524 AU)
        r1 = params.get("r1_au", 1.0) * self.AU
        r2 = params.get("r2_au", 1.524) * self.AU
        
        # 1. Delta V1 (Departure burn)
        # v1 = sqrt(mu/r1)
        # v_transfer_periap = sqrt(mu/r1) * sqrt(2*r2 / (r1+r2))
        t1 = np.sqrt(2 * r2 / (r1 + r2))
        dv1 = np.sqrt(mu / r1) * (t1 - 1)
        
        # 2. Delta V2 (Arrival burn)
        # v2 = sqrt(mu/r2)
        # v_transfer_apoap = sqrt(mu/r2) * sqrt(2*r1 / (r1+r2))
        t2 = np.sqrt(2 * r1 / (r1 + r2))
        dv2 = np.sqrt(mu / r2) * (1 - t2)
        
        total_dv = abs(dv1) + abs(dv2)
        
        # Transfer Time = 0.5 * Period of Transfer Orbit
        a_transfer = (r1 + r2) / 2
        T_transfer = 2 * np.pi * np.sqrt(a_transfer**3 / mu)
        time_seconds = 0.5 * T_transfer
        
        return {
            "status": "solved",
            "method": "Hohmann Transfer",
            "origin_au": r1 / self.AU,
            "destination_au": r2 / self.AU,
            "delta_v1_km_s": abs(dv1) / 1000,
            "delta_v2_km_s": abs(dv2) / 1000,
            "total_delta_v_km_s": total_dv / 1000,
            "transfer_time_days": time_seconds / (3600 * 24)
        }

    def _solve_rocket(self, params):
        """
        Tsiolkovsky Rocket Equation.
        Calculates Mass Ratio and Fuel Mass.
        dV = Isp * g0 * ln(m0 / mf)
        """
        logger.info("[ASTRO] Solving Rocket Equation...")
        
        dv = params.get("delta_v_km_s", 5.7) * 1000 # m/s (approx Earth-Mars)
        isp = params.get("isp_s", 450) # Chemical Isp. Nuclear ~900-3000
        dry_mass = params.get("dry_mass_kg", 10000) # Spaceship stats
        
        ve = isp * self.g0
        
        # Mass Ratio MR = m0 / mf = exp(dV / ve)
        mr = np.exp(dv / ve)
        
        # mf = dry_mass + payload (Assume dry_mass is final mass)
        mf = dry_mass
        m0 = mf * mr
        fuel_mass = m0 - mf
        
        return {
            "status": "solved",
            "method": "Tsiolkovsky Rocket Eq",
            "input_delta_v_m_s": dv,
            "input_isp_s": isp,
            "required_mass_ratio": mr,
            "dry_mass_kg": dry_mass,
            "fuel_mass_kg": fuel_mass,
            "total_launch_mass_kg": m0,
            "fuel_fraction_percent": (fuel_mass / m0) * 100
        }
