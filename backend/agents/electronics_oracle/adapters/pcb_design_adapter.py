"""
PCB Design Adapter
Handles trace impedance, current capacity, via design, and thermal management.
"""

import numpy as np
from typing import Dict, Any

class PCBDesignAdapter:
    """PCB Design Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "TRACE_IMPEDANCE").upper()
        
        if sim_type == "TRACE_IMPEDANCE":
            return self._solve_trace_impedance(params)
        elif sim_type == "CURRENT_CAPACITY":
            return self._solve_current_capacity(params)
        elif sim_type == "VIA":
            return self._solve_via(params)
        elif sim_type == "THERMAL":
            return self._solve_thermal(params)
        else:
            return {"status": "error", "message": f"Unknown PCB type: {sim_type}"}
    
    def _solve_trace_impedance(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Microstrip impedance: Z₀ = 87/√(εr+1.41) × ln(5.98h/(0.8w+t))
        """
        trace_type = params.get("trace_type", "MICROSTRIP").upper()
        
        if trace_type == "MICROSTRIP":
            er = params.get("dielectric_constant", 4.5)  # FR-4
            h = params.get("dielectric_height_mil", 10)
            w = params.get("trace_width_mil", 10)
            t = params.get("trace_thickness_mil", 1.4)  # 1 oz copper
            
            # Microstrip formula
            Z0 = (87 / np.sqrt(er + 1.41)) * np.log(5.98 * h / (0.8 * w + t))
            
            return {
                "status": "solved",
                "method": "Microstrip Impedance",
                "impedance_ohm": float(Z0),
                "trace_width_mil": w,
                "dielectric_height_mil": h
            }
        
        elif trace_type == "STRIPLINE":
            er = params.get("dielectric_constant", 4.5)
            b = params.get("dielectric_spacing_mil", 20)
            w = params.get("trace_width_mil", 10)
            t = params.get("trace_thickness_mil", 1.4)
            
            # Stripline formula (simplified)
            Z0 = (60 / np.sqrt(er)) * np.log(4 * b / (0.67 * np.pi * (w + t)))
            
            return {
                "status": "solved",
                "method": "Stripline Impedance",
                "impedance_ohm": float(Z0),
                "trace_width_mil": w
            }
        
        return {"status": "error", "message": f"Unknown trace type: {trace_type}"}
    
    def _solve_current_capacity(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        IPC-2221: I = k × ΔT^0.44 × A^0.725
        """
        delta_T = params.get("temperature_rise_c", 10)
        
        # Trace dimensions
        width_mil = params.get("trace_width_mil", 10)
        thickness_mil = params.get("trace_thickness_mil", 1.4)
        
        # Cross-sectional area
        A = width_mil * thickness_mil  # mil²
        
        # IPC-2221 constants
        # External layers: k = 0.048, internal: k = 0.024
        layer = params.get("layer", "EXTERNAL").upper()
        k = 0.048 if layer == "EXTERNAL" else 0.024
        
        # Current capacity
        I = k * (delta_T ** 0.44) * (A ** 0.725)
        
        # Resistance per inch
        rho = 0.688  # Ω·mil²/inch for copper at 20°C
        R_per_inch = rho / A
        
        # Power dissipation per inch
        P_per_inch = I**2 * R_per_inch
        
        return {
            "status": "solved",
            "method": "IPC-2221 Current Capacity",
            "current_capacity_a": float(I),
            "resistance_per_inch_ohm": float(R_per_inch),
            "power_per_inch_w": float(P_per_inch),
            "temperature_rise_c": delta_T
        }
    
    def _solve_via(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Via resistance and current capacity
        """
        diameter_mil = params.get("via_diameter_mil", 10)
        plating_thickness_mil = params.get("plating_thickness_mil", 1.0)
        board_thickness_mil = params.get("board_thickness_mil", 62)
        
        # Via barrel area
        r_outer = diameter_mil / 2
        r_inner = r_outer - plating_thickness_mil
        A_barrel = np.pi * (r_outer**2 - r_inner**2)
        
        # Resistance
        rho = 0.688  # Ω·mil²/inch for copper
        length_inch = board_thickness_mil / 1000
        R = rho * length_inch / A_barrel
        
        # Current capacity (simplified)
        I_max = 1.0 * A_barrel  # Rough estimate: 1A per mil²
        
        return {
            "status": "solved",
            "method": "Via Design",
            "resistance_ohm": float(R),
            "current_capacity_a": float(I_max),
            "barrel_area_mil2": float(A_barrel)
        }
    
    def _solve_thermal(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thermal via array for heat dissipation
        """
        power_w = params.get("power_dissipation_w", 5.0)
        delta_T = params.get("max_temperature_rise_c", 20)
        
        # Thermal resistance of single via (typical)
        R_th_via = params.get("via_thermal_resistance_c_w", 50)
        
        # Number of vias needed
        R_th_required = delta_T / power_w
        num_vias = int(np.ceil(R_th_via / R_th_required))
        
        # Actual thermal resistance with via array
        R_th_actual = R_th_via / num_vias
        
        # Actual temperature rise
        delta_T_actual = power_w * R_th_actual
        
        return {
            "status": "solved",
            "method": "Thermal Via Array",
            "vias_required": num_vias,
            "thermal_resistance_c_w": float(R_th_actual),
            "temperature_rise_c": float(delta_T_actual)
        }
