"""
Sensors & Transducers Adapter
Handles temperature sensors, strain gauges, optical sensors, and pressure sensors.
"""

import numpy as np
from typing import Dict, Any

class SensorsAdapter:
    """Sensors & Transducers Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "THERMOCOUPLE").upper()
        
        if sim_type == "THERMOCOUPLE":
            return self._solve_thermocouple(params)
        elif sim_type == "RTD":
            return self._solve_rtd(params)
        elif sim_type == "THERMISTOR":
            return self._solve_thermistor(params)
        elif sim_type == "STRAIN_GAUGE":
            return self._solve_strain_gauge(params)
        elif sim_type == "PHOTODIODE":
            return self._solve_photodiode(params)
        elif sim_type == "PRESSURE":
            return self._solve_pressure(params)
        else:
            return {"status": "error", "message": f"Unknown sensor type: {sim_type}"}
    
    def _solve_thermocouple(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thermocouple: V = α(T₁ - T₂)
        """
        alpha = params.get("seebeck_coefficient_uv_k", 40)  # Type K: ~40 μV/K
        T1 = params.get("hot_junction_k", 373)
        T2 = params.get("cold_junction_k", 273)
        
        # Voltage output
        V = alpha * (T1 - T2)  # μV
        
        # Temperature measurement
        T_measured = T1
        
        return {
            "status": "solved",
            "method": "Thermocouple",
            "output_voltage_uv": float(V),
            "temperature_k": float(T_measured),
            "temperature_c": float(T_measured - 273.15)
        }
    
    def _solve_rtd(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        RTD (Resistance Temperature Detector): R(T) = R₀[1 + α(T - T₀)]
        """
        R0 = params.get("resistance_at_0c_ohm", 100)  # Pt100
        alpha = params.get("temp_coefficient_k", 0.00385)  # Platinum
        T = params.get("temperature_c", 25)
        T0 = 0
        
        # Resistance at temperature T
        R = R0 * (1 + alpha * (T - T0))
        
        # Reverse calculation (if resistance given)
        if params.get("measured_resistance_ohm"):
            R_meas = params["measured_resistance_ohm"]
            T_calc = T0 + (R_meas / R0 - 1) / alpha
            return {
                "status": "solved",
                "method": "RTD (Resistance Temperature Detector)",
                "temperature_c": float(T_calc),
                "resistance_ohm": R_meas
            }
        
        return {
            "status": "solved",
            "method": "RTD (Resistance Temperature Detector)",
            "resistance_ohm": float(R),
            "temperature_c": T
        }
    
    def _solve_thermistor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Thermistor (Steinhart-Hart): 1/T = A + B·ln(R) + C·(ln(R))³
        """
        R = params.get("resistance_ohm", 10000)
        
        # Steinhart-Hart coefficients (typical NTC)
        A = params.get("coeff_a", 0.001129148)
        B = params.get("coeff_b", 0.000234125)
        C = params.get("coeff_c", 0.0000000876741)
        
        # Temperature calculation
        ln_R = np.log(R)
        T_inv = A + B * ln_R + C * (ln_R ** 3)
        T_k = 1 / T_inv
        T_c = T_k - 273.15
        
        return {
            "status": "solved",
            "method": "Thermistor (Steinhart-Hart)",
            "temperature_k": float(T_k),
            "temperature_c": float(T_c),
            "resistance_ohm": R
        }
    
    def _solve_strain_gauge(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Strain Gauge: ΔR/R = GF × ε
        Wheatstone bridge output
        """
        GF = params.get("gauge_factor", 2.0)
        strain = params.get("strain", 1000e-6)  # 1000 microstrain
        R = params.get("nominal_resistance_ohm", 120)
        
        # Resistance change
        delta_R = GF * strain * R
        R_strained = R + delta_R
        
        # Wheatstone bridge (quarter bridge)
        Vex = params.get("excitation_voltage_v", 5.0)
        Vout = (Vex / 4) * GF * strain
        
        return {
            "status": "solved",
            "method": "Strain Gauge (Wheatstone Bridge)",
            "strain": strain,
            "resistance_change_ohm": float(delta_R),
            "output_voltage_v": float(Vout),
            "microstrain": strain * 1e6
        }
    
    def _solve_photodiode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Photodiode: I = R × P
        R = responsivity (A/W)
        """
        R = params.get("responsivity_a_w", 0.5)  # Typical silicon
        P = params.get("optical_power_w", 1e-3)  # 1 mW
        
        # Photocurrent
        I = R * P
        
        # Quantum efficiency
        wavelength_nm = params.get("wavelength_nm", 850)
        h = 6.626e-34  # Planck's constant
        c = 3e8  # Speed of light
        q = 1.602e-19  # Electron charge
        
        QE = (R * h * c) / (q * wavelength_nm * 1e-9)
        
        return {
            "status": "solved",
            "method": "Photodiode",
            "photocurrent_a": float(I),
            "quantum_efficiency": float(QE),
            "responsivity_a_w": R
        }
    
    def _solve_pressure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Pressure sensor (piezoresistive)
        """
        sensitivity = params.get("sensitivity_mv_v_bar", 10)  # mV/V/bar
        Vex = params.get("excitation_voltage_v", 5.0)
        pressure_bar = params.get("pressure_bar", 1.0)
        
        # Output voltage
        Vout = (sensitivity / 1000) * Vex * pressure_bar
        
        # Convert to PSI
        pressure_psi = pressure_bar * 14.5038
        
        return {
            "status": "solved",
            "method": "Pressure Sensor (Piezoresistive)",
            "output_voltage_v": float(Vout),
            "pressure_bar": pressure_bar,
            "pressure_psi": float(pressure_psi)
        }
