from typing import Dict, Any, List
import logging
import math # Added for math.pow

logger = logging.getLogger(__name__)

class ThermalAgent:
    """
    Thermal Analysis Agent.
    Calculates equilibrium temperatures and heat dissipation requirements.
    """
    CONSTANTS = {
        "SIGMA": 5.67e-8, # Stefan-Boltzmann Constant
        "C_P": 900,       # Specific Heat Capacity (J/kgK) - Aluminum default
    }

    def __init__(self):
        self.name = "ThermalAgent"
        
        # Initialize Oracles for thermal analysis
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            from agents.materials_oracle.materials_oracle import MaterialsOracle
            self.physics_oracle = PhysicsOracle()
            self.materials_oracle = MaterialsOracle()
            self.has_oracles = True
        except ImportError:
            self.physics_oracle = None
            self.materials_oracle = None
            self.has_oracles = False

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculates equilibrium temperature.
        Supports Convection (Air) and Radiation (Vacuum).
        """
        logger.info(f"Running Thermal Analysis on: {payload}")
        
        # 1. Robust Param Extraction
        power_w = float(payload.get("power_watts", 0))
        surface_area = float(payload.get("surface_area", 0.1)) # Default 0.1m^2
        emissivity = float(payload.get("emissivity", 0.9)) # Anodized Aluminum
        ambient_temp = float(payload.get("ambient_temp", 25.0)) # Celsius
        
        # Environment Detection
        # If heat_transfer_coeff is 0 or low, assume vacuum/radiation dominated
        h = float(payload.get("heat_transfer_coeff", 0))
        if h == 0:
            # Check if environment implies air? If not provided, assume natural convection (h=5-10)
            # UNLESS 'environment' explicitly says 'SPACE'
            env_type = payload.get("environment_type", "GROUND")
            if "SPACE" in env_type or "VACUUM" in env_type:
                mode = "RADIATION"
                h = 0
            else:
                mode = "CONVECTION"
                h = 10.0 # Default natural convection
        else:
            mode = "CONVECTION"

        # 2. Solver
        # Q_in = Power
        # Q_out_conv = h * A * (T_eq - T_amb)
        # Q_out_rad = sigma * epsilon * A * (T_eq^4 - T_amb^4) (Kelvin)
        
        logs = []
        logs.append(f"Mode: {mode}")
        logs.append(f"Dissipating {power_w}W over {surface_area}m²")

        if mode == "CONVECTION":
            # Q = hA(dT) -> dT = Q / hA
            try:
                delta_t = power_w / (h * surface_area)
            except ZeroDivisionError:
                delta_t = 0
                logs.append("Error: Zero surface area.")
                
            final_temp = ambient_temp + delta_t
            logs.append(f"Convection h={h} W/m²K")

        elif mode == "RADIATION":
            # Q = sigma * eps * A * (T^4 - T_amb^4)
            # T^4 = (Q / (sigma * eps * A)) + T_amb^4
            
            # Convert to Kelvin
            T_amb_k = ambient_temp + 273.15
            sigma = self.CONSTANTS["SIGMA"]
            
            try:
                t4_term = power_w / (sigma * emissivity * surface_area)
                final_temp_k = math.pow(t4_term + math.pow(T_amb_k, 4), 0.25)
                final_temp = final_temp_k - 273.15
            except ZeroDivisionError:
                 final_temp = 9999 # Meltdown
                 logs.append("Error: Zero area/emissivity in vacuum.")
            
            logs.append(f"Radiation Only (Vacuum)")

        # 3. Verdict
        status = "nominal"
        if final_temp > 100: status = "warning"
        if final_temp > 150: status = "critical"
        
        logs.append(f"Equilibrium: {final_temp:.1f}°C ({status})")

        return {
            "status": status,
            "equilibrium_temp_c": round(final_temp, 2),
            "delta_t": round(final_temp - ambient_temp, 2),
            "heat_load_w": power_w,
            "logs": logs
        }

    def analyze_heat_transfer_oracle(self, params: dict) -> dict:
        """Analyze heat transfer using Physics Oracle (THERMODYNAMICS)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.physics_oracle.solve(
            query="Heat transfer analysis",
            domain="THERMODYNAMICS",
            params=params
        )
    
    def analyze_thermal_properties_oracle(self, params: dict) -> dict:
        """Analyze thermal material properties using Materials Oracle"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.materials_oracle.solve(
            query="Thermal properties",
            domain="THERMAL",
            params=params
        )
