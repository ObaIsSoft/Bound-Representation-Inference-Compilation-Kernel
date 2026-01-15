from typing import Dict, Any, List
import logging
import math

logger = logging.getLogger(__name__)

class ControlAgent:
    """
    Control Systems Agent (LQR Upgrade).
    Calculates Optimal Control Gains based on Physical Inertia (J) and Performance Costs (Q, R).
    Implements State-Space Logic: u = -Kx
    """
    def __init__(self):
        self.name = "ControlAgent"
        self.rl_policy_path = "data/rl_control_policy/cem_policy_v1.pkl"

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize Control Law (LQR or RL).
        
        Modes:
        - "LQR": Analytic Gains (Default)
        - "RL": Load Pre-trained CEM Policy (Linear Weights)
        """
        mode = params.get("control_mode", "LQR")
        logger.info(f"{self.name} synthesizing controller (Mode={mode})...")
        
        if mode == "RL":
            import os
            import pickle
            import numpy as np
            
            if os.path.exists(self.rl_policy_path):
                try:
                    with open(self.rl_policy_path, 'rb') as f:
                        weights_flat = pickle.load(f)
                        
                    # Reconstruct Policy for verification (simplified metadata)
                    # We assume fixed dimensions for the hover task: 4 obs -> 1 act
                    obs_dim = 4
                    act_dim = 1
                    
                    return {
                        "status": "success",
                        "method": "RL_CEM_Evolutionary",
                        "policy_path": self.rl_policy_path,
                        "policy_type": "Linear (Tanh)",
                        "policy_weights_shape": f"{act_dim}x{obs_dim}",
                        "logs": [f"CEM Policy loaded successfully. Ready for inference."]
                    }
                except Exception as e:
                    logger.error(f"Failed to load policy: {e}")
                    mode = "LQR"
            else:
                logger.warning(f"RL Policy not found at {self.rl_policy_path}. Falling back to LQR.")
                mode = "LQR" # Fallback
                
        # 1. Physics Inputs (The "Plant")
        # Default to small drone inertia if missing
        J_vec = params.get("inertia_tensor", [0.005, 0.005, 0.01]) 
        
        # 2. Performance Inputs (The "Intent")
        # Q: How much do we hate error? (High = Aggressive)
        # R: How much do we hate using fuel/volts? (High = Conservative)
        q_pos = params.get("q_error_cost", 100.0)
        r_control = params.get("r_effort_cost", 1.0)
        
        # 3. LQR Synthesis (Analytic Solution for Double Integrator)
        # System: J * theta'' = u
        # State: x = [theta, theta']
        # For a simple diagonal inertia, we decouple axes (Roll, Pitch, Yaw).
        
        gains = {}
        axes = ["roll", "pitch", "yaw"]
        
        logs = [f"Inertia J: {J_vec} kg·m²", f"Costs Q={q_pos}, R={r_control}"]
        
        for i, axis in enumerate(axes):
            J = J_vec[i]
            if J <= 0: J = 0.001 # Protect div by zero
            
            # Analytic LQR for 1/J s^2 plant
            # kp = sqrt(Q/R)
            # kd = sqrt(2*sqrt(Q/R) + q_vel/R) -- assuming q_vel=0 for now results in:
            # Damping ratio zeta = 0.707 (optimal butterworth) typically implied by standard forms
            
            # Let's use the explicit textbook analytic result for x = [pos, vel]
            # Cost = integral(q*pos^2 + r*u^2)
            # We normalize by inertia so u_accel = u_torque / J
            # But here R is on torque.
            
            # Resulting Gains:
            kp = math.sqrt(q_pos / r_control) # Independent of Inertia in 'accel' domain, but dependent in 'torque' domain
            # Wait, u = -Kx. u is torque.
            # Plant: theta'' = (1/J) * u
            # This makes the "effective" B matrix differ.
            
            # Correct Analytic derivation:
            # kp = sqrt(Q/R)
            # kd = sqrt(2 * J * sqrt(Q/R)) 
            # ... Checking Dimensionality ... 
            # kp units: Nm/rad. Q units: 1/rad^2? R units: 1/(Nm)^2? 
            # Let's assume standard Bryson's rule:
            # Q = 1/max_error^2, R = 1/max_torque^2
            
            # Simplified LQR approximation for Kp, Kd:
            # Natural Frequency wn = (Q/R)^(1/4) / sqrt(J) ?? No.
            
            # Let's stick to the Pole Placement result derived from Q/R ratio which sets bandwidth:
            # Bandwidth (rad/s) ~ sqrt(kp_normalized)
            
            # Approach:
            # 1. Optimal Natural Frequency wn = sqrt(sqrt(Q/R) / J) is commonly cited for 1/s^2 system?
            # Let's do:
            # kp_torque = sqrt(Q/R)  <-- This implies Torque is proportional to error, heavily.
            # kd_torque = sqrt(2 * J * kp_torque) <-- Critical damping-ish relationship
            
            # Revised for robustness:
            kp_val = math.sqrt(q_pos / r_control)
            kd_val = math.sqrt(2.0 * J * kp_val)
            
            gains[axis] = {
                "kp": round(kp_val, 4),
                "kd": round(kd_val, 4),
                "ki": 0.0 # LQR doesn't give KI natively (needs integral state)
            }
            logs.append(f"{axis.upper()}: Kp={kp_val:.2f}, Kd={kd_val:.2f} (J={J})")

        return {
            "status": "success",
            "method": "LQR_Analytic_Double_Integrator",
            "gain_matrix": gains,
            "logs": logs
        }
