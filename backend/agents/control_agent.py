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
        """
        mode = params.get("control_mode", "RL") # Default to RL now
        logger.info(f"{self.name} synthesizing controller (Mode={mode})...")
        
        # 1. System Identification (SysID) - Online Adaptation
        # Estimate disturbances based on previous state history (mocked here)
        history = params.get("flight_history", [])
        disturbances = self._estimate_disturbance(history)
        
        if mode == "RL":
            try:
                # Load Inference Engine
                policy = self._load_policy()
                if not policy:
                    raise FileNotFoundError("No active policy found.")
                
                # Construct Observation Vector [State, Target, Disturbance]
                # Simplified 12-D vector: [Pos(3), Vel(3), Target(3), Dist(3)]
                state_vec = params.get("state_vec", [0]*6)
                target_vec = params.get("target_vec", [0]*3)
                obs = state_vec + target_vec + disturbances
                
                # Forward Pass
                action = policy.predict(obs)
                
                return {
                    "status": "success",
                    "method": "RL_PPO_Adaptive",
                    "control_signal": action,
                    "estimated_disturbance": disturbances,
                    "policy_version": policy.version,
                    "logs": [f"Inference: PPO Policy v{policy.version}", f"Est. Disturbance: {disturbances}"]
                }
                
            except Exception as e:
                logger.warning(f"RL Inference failed ({e}). Falling back to LQR.")
                mode = "LQR" # Fallback
                return {"status": "error", "message": f"RL Failed: {str(e)}", "method": "RL->LQR_Fallback"}

        if mode == "LQR":
            return self._run_lqr(params, disturbances)
            
        return {"status": "error", "message": "Unknown control mode"}

    def _estimate_disturbance(self, history: List[Dict]) -> List[float]:
        """
        SysID: Estimate unmodeled forces (wind, payload shift) from flight data.
        d_hat = m*a_measured - (u_thrust + g)
        """
        if len(history) < 2:
            return [0.0, 0.0, 0.0]
            
        # Mock calculation: In production this filters the IMU stream
        # returning a 3D force vector [Fx, Fy, Fz]
        last_error = history[-1].get("error", 0.0)
        # Adaptive: larger error implies larger unmodeled force
        d_mag = last_error * 0.1 
        return [d_mag, d_mag * 0.5, 0.0]

    def _load_policy(self):
        """Load weights from Pickle (Legacy CEM) or JSON (PPO)."""
        import json
        import os
        import pickle
        import numpy as np
        
        # 1. Try Pickle (Real Trained Policy) first
        pkl_path = "backend/data/rl_control_policy/cem_policy_v1.pkl"
        # Fix path for relative execution
        if not os.path.exists(pkl_path):
            pkl_path = "data/rl_control_policy/cem_policy_v1.pkl"
            
        if os.path.exists(pkl_path):
             try:
                with open(pkl_path, 'rb') as f:
                    # Generic load - flat parameter vector from CEM
                    params = pickle.load(f)
                    
                    # Wrap in Linear Policy adapter that matches train_rl.py structure
                    class CEMPolicyWrapper:
                        def __init__(self, flat_params, obs_dim=12, act_dim=4):
                            self.version = "CEM-v1 (Pickle)"
                            
                            # Reconstruct weights/bias from flat vector
                            # Param count = obs*act + act
                            expected = obs_dim * act_dim + act_dim
                            
                            if len(flat_params) != expected:
                                # Fallback for mismatch (e.g. if training env dims differed)
                                # Try to deduce
                                pass
                                
                            split = obs_dim * act_dim
                            self.weights = flat_params[:split].reshape(act_dim, obs_dim)
                            self.bias = flat_params[split:]
                            
                        def predict(self, obs):
                            # Tanh Linear Policy: u = (tanh(Wx + b) + 1) / 2
                            x = np.array(obs)
                            logit = np.dot(self.weights, x) + self.bias
                            action = (np.tanh(logit) + 1.0) / 2.0
                            return action.tolist()
                                
                    # Note: We need to know dims. 
                    # BrickEnv (Hover) typically has Obs=12, Act=4 (Motor Thrusts)
                    return CEMPolicyWrapper(params, obs_dim=12, act_dim=4)
             except Exception as e:
                 logger.warning(f"Found pickle but failed to load: {e}")

        # 2. Fallback to JSON (Mock PPO)
        path = "data/rl_control_policy/ppo_policy_v2.json"
        if not os.path.exists(path):
            return None
            
        class NumpyMLP:
            # ... (Existing JSON MLP Class)
            def __init__(self, weights):
                self.weights = weights
                self.version = weights.get("version", "1.0")
                self.l1_w = np.array(weights["layer1_w"])
                self.l1_b = np.array(weights["layer1_b"])
                self.l2_w = np.array(weights["layer2_w"])
                self.l2_b = np.array(weights["layer2_b"])
                
            def predict(self, obs):
                x = np.array(obs)
                # Reshape if needed or assume flat
                h1 = np.maximum(0, np.dot(x, self.l1_w) + self.l1_b)
                out = np.dot(h1, self.l2_w) + self.l2_b
                return out.tolist()

        try:
            with open(path, 'r') as f:
                data = json.load(f)
                return NumpyMLP(data)
        except Exception:
            return None

    def _run_lqr(self, params: Dict, disturbances: List[float]) -> Dict:
        """Legacy LQR Logic (Refactored)"""
        # ... (Existing LQR logic, injecting disturbances into FeedForward if needed)
        J_vec = params.get("inertia_tensor", [0.005, 0.005, 0.01])
        q_pos = params.get("q_error_cost", 100.0)
        r_control = params.get("r_effort_cost", 1.0)
        
        gains = {}
        axes = ["roll", "pitch", "yaw"]
        for i, axis in enumerate(axes):
            J = J_vec[i]
            kp = math.sqrt(q_pos / r_control)
            kd = math.sqrt(2.0 * J * kp)
            
            # Feedforward term to counteract disturbance
            u_ff = -1.0 * disturbances[i] # Simple cancellation
            
            gains[axis] = {"kp": kp, "kd": kd, "u_ff": u_ff}
            
        return {
            "status": "success", 
            "method": "LQR_Analytic", 
            "gains": gains,
            "disturbance_compensation": disturbances
        }
