"""
Control Systems Adapter
Handles transfer functions, PID control, stability analysis, and frequency response.
"""

import numpy as np
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class ControlSystemsAdapter:
    """Control Systems Solver"""
    
    def run_simulation(self, params: Dict[str, Any]) -> Dict[str, Any]:
        sim_type = params.get("type", "PID").upper()
        
        if sim_type == "PID":
            return self._solve_pid(params)
        elif sim_type == "TRANSFER_FUNCTION":
            return self._solve_transfer_function(params)
        elif sim_type == "STABILITY":
            return self._solve_stability(params)
        elif sim_type == "BODE":
            return self._solve_bode(params)
        else:
            return {"status": "error", "message": f"Unknown control type: {sim_type}"}
    
    def _solve_pid(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        PID Controller: u(t) = Kp·e(t) + Ki·∫e(t)dt + Kd·de(t)/dt
        """
        Kp = params.get("proportional_gain", 1.0)
        Ki = params.get("integral_gain", 0.1)
        Kd = params.get("derivative_gain", 0.01)
        
        error = params.get("error", 1.0)
        error_integral = params.get("error_integral", 0.0)
        error_derivative = params.get("error_derivative", 0.0)
        
        # Control output
        u = Kp * error + Ki * error_integral + Kd * error_derivative
        
        # Ziegler-Nichols tuning (if requested)
        if params.get("auto_tune", False):
            Ku = params.get("ultimate_gain", 2.0)
            Tu = params.get("ultimate_period_s", 1.0)
            
            # Ziegler-Nichols PID
            Kp_zn = 0.6 * Ku
            Ki_zn = 1.2 * Ku / Tu
            Kd_zn = 0.075 * Ku * Tu
            
            return {
                "status": "solved",
                "method": "PID Controller (Ziegler-Nichols)",
                "control_output": float(u),
                "tuned_kp": float(Kp_zn),
                "tuned_ki": float(Ki_zn),
                "tuned_kd": float(Kd_zn)
            }
        
        return {
            "status": "solved",
            "method": "PID Controller",
            "control_output": float(u),
            "kp": Kp,
            "ki": Ki,
            "kd": Kd
        }
    
    def _solve_transfer_function(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Transfer Function: G(s) = N(s)/D(s)
        Poles and zeros analysis
        """
        # Numerator and denominator coefficients
        num = params.get("numerator", [1])
        den = params.get("denominator", [1, 1])
        
        # Find poles (roots of denominator)
        poles = np.roots(den)
        
        # Find zeros (roots of numerator)
        zeros = np.roots(num) if len(num) > 1 else []
        
        # DC gain
        dc_gain = num[-1] / den[-1] if den[-1] != 0 else float('inf')
        
        # Stability check (all poles in left half-plane)
        stable = all(np.real(pole) < 0 for pole in poles)
        
        return {
            "status": "solved",
            "method": "Transfer Function Analysis",
            "poles": [complex(p) for p in poles],
            "zeros": [complex(z) for z in zeros],
            "dc_gain": float(dc_gain),
            "stable": stable
        }
    
    def _solve_stability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Routh-Hurwitz stability criterion
        """
        # Characteristic equation coefficients
        coeffs = params.get("characteristic_equation", [1, 3, 2])
        
        # Build Routh array
        n = len(coeffs)
        routh = np.zeros((n, (n+1)//2))
        
        # First two rows
        routh[0, :] = coeffs[0::2]
        routh[1, :len(coeffs[1::2])] = coeffs[1::2]
        
        # Remaining rows
        for i in range(2, n):
            for j in range((n+1)//2 - 1):
                if routh[i-1, 0] != 0:
                    routh[i, j] = (routh[i-1, 0] * routh[i-2, j+1] - routh[i-2, 0] * routh[i-1, j+1]) / routh[i-1, 0]
        
        # Check first column for sign changes
        first_col = routh[:, 0]
        sign_changes = sum(1 for i in range(len(first_col)-1) if first_col[i] * first_col[i+1] < 0)
        
        stable = sign_changes == 0
        
        return {
            "status": "solved",
            "method": "Routh-Hurwitz Stability",
            "stable": stable,
            "sign_changes": int(sign_changes),
            "unstable_poles": int(sign_changes)
        }
    
    def _solve_bode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Bode plot analysis - gain and phase margins
        """
        # System parameters
        wc = params.get("crossover_frequency_rad_s", 10)  # Gain crossover
        phase_at_wc = params.get("phase_at_crossover_deg", -150)
        
        # Phase margin: PM = 180° + phase(wc)
        phase_margin = 180 + phase_at_wc
        
        # Gain margin (simplified)
        w_180 = params.get("phase_180_frequency_rad_s", 20)
        gain_at_w180_db = params.get("gain_at_180_db", -10)
        gain_margin = -gain_at_w180_db
        
        # Stability assessment
        stable = phase_margin > 0 and gain_margin > 0
        
        return {
            "status": "solved",
            "method": "Bode Plot Analysis",
            "phase_margin_deg": float(phase_margin),
            "gain_margin_db": float(gain_margin),
            "crossover_frequency_rad_s": wc,
            "stable": stable
        }
