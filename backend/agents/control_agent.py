"""
Production Control Agent - Optimal Control Synthesis

Features:
- LQR controller with automatic gain scheduling
- RL policy loading with dimension adaptation
- Multi-format policy support (Pickle, JSON, ONNX)
- Disturbance estimation from flight history
- Adaptive control with online parameter estimation
- Safe fallback chains
"""

from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass
from enum import Enum
import json
import logging
import math
import os
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)


class ControlMode(Enum):
    """Control synthesis modes."""
    LQR = "LQR"
    RL = "RL"
    ADAPTIVE = "ADAPTIVE"
    MPC = "MPC"


@dataclass
class ControlGains:
    """Controller gains."""
    kp: float
    ki: float = 0.0
    kd: float
    u_ff: float = 0.0


@dataclass
class PolicyConfig:
    """RL Policy configuration."""
    obs_dim: int = 12
    act_dim: int = 4
    hidden_dim: int = 64
    activation: str = "tanh"


class LinearPolicy:
    """Linear policy wrapper for flat parameter vectors."""
    
    def __init__(self, weights: np.ndarray, bias: np.ndarray, 
                 version: str = "linear-v1"):
        self.weights = weights
        self.bias = bias
        self.version = version
        self.obs_dim = weights.shape[1]
        self.act_dim = weights.shape[0]
    
    def predict(self, obs: List[float]) -> List[float]:
        """Forward pass through linear layer."""
        x = np.array(obs)
        
        # Handle dimension mismatch
        if len(x) != self.obs_dim:
            x = self._adapt_observation(x, self.obs_dim)
        
        logit = np.dot(self.weights, x) + self.bias
        action = (np.tanh(logit) + 1.0) / 2.0
        return action.tolist()
    
    def _adapt_observation(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        """Adapt observation to target dimension."""
        current_dim = len(obs)
        
        if current_dim < target_dim:
            # Pad with zeros
            return np.pad(obs, (0, target_dim - current_dim), mode='constant')
        elif current_dim > target_dim:
            # Truncate or project
            return obs[:target_dim]
        return obs


class NumpyMLP:
    """MLP policy using numpy (no torch dependency)."""
    
    def __init__(self, weights: Dict[str, Any], version: str = "mlp-v1"):
        self.l1_w = np.array(weights["layer1_w"])
        self.l1_b = np.array(weights["layer1_b"])
        self.l2_w = np.array(weights["layer2_w"])
        self.l2_b = np.array(weights["layer2_b"])
        self.version = version
        self.input_dim = self.l1_w.shape[0]
    
    def predict(self, obs: List[float]) -> List[float]:
        """Forward pass through MLP."""
        x = np.array(obs)
        
        # Handle dimension mismatch
        if len(x) != self.input_dim:
            x = self._adapt_observation(x, self.input_dim)
        
        h1 = np.maximum(0, np.dot(x, self.l1_w) + self.l1_b)  # ReLU
        out = np.dot(h1, self.l2_w) + self.l2_b
        return out.tolist()
    
    def _adapt_observation(self, obs: np.ndarray, target_dim: int) -> np.ndarray:
        """Adapt observation to target dimension."""
        current_dim = len(obs)
        
        if current_dim < target_dim:
            # Pad with zeros
            return np.pad(obs, (0, target_dim - current_dim), mode='constant')
        elif current_dim > target_dim:
            # Truncate
            return obs[:target_dim]
        return obs


class AdaptivePolicy:
    """Adaptive policy that learns online from feedback."""
    
    def __init__(self, base_policy: Callable, learning_rate: float = 0.01):
        self.base_policy = base_policy
        self.learning_rate = learning_rate
        self.error_history = []
        self.adaptation_gain = np.zeros(base_policy.act_dim if hasattr(base_policy, 'act_dim') else 4)
    
    def predict(self, obs: List[float], target: Optional[List[float]] = None) -> List[float]:
        """Predict with online adaptation."""
        base_action = self.base_policy.predict(obs)
        
        # Add adaptive correction based on recent errors
        adaptive_correction = self.adaptation_gain * 0.1
        action = np.array(base_action) + adaptive_correction
        
        # Clip to valid range
        return np.clip(action, 0, 1).tolist()
    
    def update(self, error: float):
        """Update adaptation based on error."""
        self.error_history.append(error)
        if len(self.error_history) > 10:
            self.error_history.pop(0)
        
        # Simple gradient descent on error
        mean_error = np.mean(self.error_history)
        self.adaptation_gain += self.learning_rate * mean_error


class ControlAgent:
    """
    Production-grade control synthesis agent.
    
    Synthesizes control laws using:
    - LQR (Linear Quadratic Regulator) with analytic gains
    - RL (Reinforcement Learning) policies from file
    - Adaptive control with online learning
    - Safe fallback chains
    """
    
    # Default LQR parameters
    DEFAULT_INERTIA = [0.005, 0.005, 0.01]  # kg*m^2
    DEFAULT_Q_COST = 100.0  # Position error cost
    DEFAULT_R_COST = 1.0    # Control effort cost
    
    def __init__(self, policy_dir: str = "data/rl_control_policy"):
        self.name = "ControlAgent"
        self.policy_dir = Path(policy_dir)
        self.active_policies: Dict[str, Any] = {}
        self.fallback_chain = [ControlMode.RL, ControlMode.LQR]
        self.adaptive_policies: Dict[str, AdaptivePolicy] = {}
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synthesize control law.
        
        Args:
            params: {
                "control_mode": "RL" | "LQR" | "ADAPTIVE",
                "state_vec": List[float],  # Current state
                "target_vec": List[float],  # Target setpoint
                "flight_history": List[Dict],  # For disturbance estimation
                "inertia_tensor": List[float],  # [Jx, Jy, Jz]
                "q_error_cost": float,  # LQR position cost
                "r_effort_cost": float,  # LQR control cost
                "adaptation_enabled": bool
            }
        """
        mode_str = params.get("control_mode", "RL")
        try:
            mode = ControlMode(mode_str)
        except ValueError:
            return {
                "status": "error",
                "message": f"Unknown control mode: {mode_str}",
                "available_modes": [m.value for m in ControlMode]
            }
        
        logger.info(f"[CONTROL] Synthesizing controller (Mode={mode.value})...")
        
        # Estimate disturbances from history
        history = params.get("flight_history", [])
        disturbances = self._estimate_disturbances(history)
        
        # Try control modes in fallback chain
        for try_mode in self._get_fallback_chain(mode):
            try:
                if try_mode == ControlMode.RL:
                    return self._run_rl_control(params, disturbances)
                elif try_mode == ControlMode.LQR:
                    return self._run_lqr_control(params, disturbances)
                elif try_mode == ControlMode.ADAPTIVE:
                    return self._run_adaptive_control(params, disturbances)
                elif try_mode == ControlMode.MPC:
                    return self._run_mpc_control(params, disturbances)
            except Exception as e:
                logger.warning(f"[CONTROL] {try_mode.value} failed: {e}")
                continue
        
        # All modes failed - return safe fallback
        return {
            "status": "error",
            "message": "All control modes failed",
            "method": "FAILED",
            "control_signal": [0.5, 0.5, 0.5, 0.5]  # Neutral hover
        }
    
    def _get_fallback_chain(self, primary: ControlMode) -> List[ControlMode]:
        """Get control mode fallback chain."""
        chain = [primary]
        for mode in self.fallback_chain:
            if mode not in chain:
                chain.append(mode)
        return chain
    
    def _estimate_disturbances(self, history: List[Dict]) -> List[float]:
        """
        Estimate unmodeled disturbances from flight history.
        Uses simple moving average of recent errors.
        """
        if len(history) < 2:
            return [0.0, 0.0, 0.0]
        
        # Extract recent errors
        recent_errors = [
            h.get("error", 0.0) 
            for h in history[-5:]
        ]
        
        # Estimate disturbance magnitude from error trend
        avg_error = sum(recent_errors) / len(recent_errors)
        
        # Map to 3D disturbance vector [Fx, Fy, Fz]
        # Simple model: larger error -> larger disturbance
        d_mag = avg_error * 0.1
        return [d_mag, d_mag * 0.5, 0.0]
    
    def _run_rl_control(self, params: Dict, disturbances: List[float]) -> Dict[str, Any]:
        """Run RL-based control."""
        policy = self._load_policy()
        
        if not policy:
            raise FileNotFoundError("No RL policy available")
        
        # Construct observation vector
        state_vec = params.get("state_vec", [0.0] * 6)
        target_vec = params.get("target_vec", [0.0] * 3)
        
        # Combine into observation [state, target, disturbance]
        obs = state_vec + target_vec + disturbances
        
        # Forward pass
        action = policy.predict(obs)
        
        return {
            "status": "success",
            "method": "RL_Policy",
            "control_signal": action,
            "observation_dim": len(obs),
            "estimated_disturbance": disturbances,
            "policy_version": getattr(policy, 'version', 'unknown')
        }
    
    def _load_policy(self) -> Optional[Any]:
        """Load RL policy from file with dimension adaptation."""
        # Search paths in order of preference
        search_paths = [
            self.policy_dir / "ppo_policy_v2.json",
            self.policy_dir / "cem_policy_v1.pkl",
            Path("backend") / self.policy_dir / "cem_policy_v1.pkl",
            Path("data/rl_control_policy/ppo_policy_v2.json"),
        ]
        
        for path in search_paths:
            if not path.exists():
                continue
            
            try:
                if path.suffix == ".pkl":
                    return self._load_pickle_policy(path)
                elif path.suffix == ".json":
                    return self._load_json_policy(path)
                elif path.suffix == ".onnx":
                    return self._load_onnx_policy(path)
            except Exception as e:
                logger.warning(f"Failed to load policy from {path}: {e}")
                continue
        
        return None
    
    def _load_pickle_policy(self, path: Path) -> LinearPolicy:
        """Load policy from pickle file."""
        import pickle
        
        with open(path, 'rb') as f:
            flat_params = pickle.load(f)
        
        # Default dimensions for BrickEnv hover task
        obs_dim = 12
        act_dim = 4
        
        # Calculate expected parameter count
        expected_params = obs_dim * act_dim + act_dim
        
        actual_params = len(flat_params) if hasattr(flat_params, '__len__') else flat_params.shape[0]
        
        if actual_params != expected_params:
            logger.warning(
                f"Policy dimension mismatch: expected {expected_params}, got {actual_params}. "
                f"Attempting auto-adaptation..."
            )
            
            # Try to infer dimensions
            # For a linear policy: params = [W_flat, b]
            # So total = obs * act + act = act * (obs + 1)
            
            # Try common configurations
            configs = [
                (12, 4),   # Standard hover: pos(3)+vel(3)+target(3)+dist(3), 4 motors
                (9, 4),    # Without disturbance estimate
                (6, 4),    # Minimal: pos(3)+vel(3), 4 motors
                (8, 4),    # 2D hover: pos(2)+vel(2)+target(2)+dist(2), 4 motors
            ]
            
            for try_obs, try_act in configs:
                try_expected = try_obs * try_act + try_act
                if actual_params == try_expected:
                    obs_dim, act_dim = try_obs, try_act
                    logger.info(f"Auto-detected dimensions: obs={obs_dim}, act={act_dim}")
                    break
            else:
                # Force reshape to default dimensions
                logger.warning(f"Could not auto-detect, forcing reshape to {obs_dim}x{act_dim}")
                if actual_params > expected_params:
                    flat_params = flat_params[:expected_params]
                else:
                    # Pad with small random values
                    padding = np.random.randn(expected_params - actual_params) * 0.01
                    flat_params = np.concatenate([flat_params, padding])
        
        # Reshape parameters
        split_idx = obs_dim * act_dim
        weights = flat_params[:split_idx].reshape(act_dim, obs_dim)
        bias = flat_params[split_idx:]
        
        return LinearPolicy(weights, bias, version=f"CEM-{path.stem}")
    
    def _load_json_policy(self, path: Path) -> NumpyMLP:
        """Load policy from JSON file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        version = data.get("version", path.stem)
        return NumpyMLP(data, version=version)
    
    def _load_onnx_policy(self, path: Path) -> Any:
        """Load policy from ONNX file."""
        try:
            import onnxruntime as ort
            
            session = ort.InferenceSession(str(path))
            
            class ONNXPolicy:
                def __init__(self, session):
                    self.session = session
                    self.version = f"ONNX-{path.stem}"
                    self.input_name = session.get_inputs()[0].name
                
                def predict(self, obs):
                    obs_array = np.array(obs).astype(np.float32).reshape(1, -1)
                    outputs = self.session.run(None, {self.input_name: obs_array})
                    return outputs[0].flatten().tolist()
            
            return ONNXPolicy(session)
        except ImportError:
            logger.warning("ONNX runtime not available")
            return None
    
    def _run_lqr_control(self, params: Dict, disturbances: List[float]) -> Dict[str, Any]:
        """Run LQR control."""
        J_vec = params.get("inertia_tensor", self.DEFAULT_INERTIA)
        q_pos = params.get("q_error_cost", self.DEFAULT_Q_COST)
        r_control = params.get("r_effort_cost", self.DEFAULT_R_COST)
        
        gains = {}
        axes = ["roll", "pitch", "yaw"]
        
        for i, axis in enumerate(axes):
            J = J_vec[i]
            
            # LQR gains for double integrator
            # For system: J * theta_ddot = u
            # Optimal gains: kp = sqrt(q/r), kd = sqrt(2*sqrt(q/r)*J)
            omega = math.sqrt(q_pos / r_control)
            kp = omega
            kd = math.sqrt(2.0 * J * omega)
            
            # Feedforward disturbance compensation
            u_ff = -disturbances[i] if i < len(disturbances) else 0.0
            
            gains[axis] = {
                "kp": round(kp, 4),
                "kd": round(kd, 4),
                "u_ff": round(u_ff, 4)
            }
        
        return {
            "status": "success",
            "method": "LQR_Analytic",
            "gains": gains,
            "natural_frequency": round(omega, 4),
            "disturbance_estimate": disturbances,
            "damping_ratio": 0.707  # LQR gives ~sqrt(2)/2 damping
        }
    
    def _run_adaptive_control(self, params: Dict, disturbances: List[float]) -> Dict[str, Any]:
        """Run adaptive control with online learning."""
        policy_id = params.get("policy_id", "default")
        
        # Get or create adaptive wrapper
        if policy_id not in self.adaptive_policies:
            base_policy = self._load_policy()
            if not base_policy:
                raise RuntimeError("No base policy for adaptation")
            self.adaptive_policies[policy_id] = AdaptivePolicy(base_policy)
        
        adaptive_policy = self.adaptive_policies[policy_id]
        
        # Construct observation
        state_vec = params.get("state_vec", [0.0] * 6)
        target_vec = params.get("target_vec", [0.0] * 3)
        obs = state_vec + target_vec + disturbances
        
        # Predict with adaptation
        action = adaptive_policy.predict(obs)
        
        # Update adaptation if error provided
        error = params.get("tracking_error")
        if error is not None:
            adaptive_policy.update(error)
        
        return {
            "status": "success",
            "method": "Adaptive_RL",
            "control_signal": action,
            "adaptation_error_history": adaptive_policy.error_history[-5:],
            "policy_id": policy_id
        }
    
    def _run_mpc_control(self, params: Dict, disturbances: List[float]) -> Dict[str, Any]:
        """Run Model Predictive Control."""
        # Simplified MPC - in production would use CasADi/ACADO
        horizon = params.get("mpc_horizon", 10)
        dt = params.get("dt", 0.01)
        
        # Current state
        state = np.array(params.get("state_vec", [0.0] * 6))
        target = np.array(params.get("target_vec", [0.0] * 3))
        
        # Simple trajectory optimization (placeholder)
        # In production: solve QP at each step
        error = target - state[:3]
        
        # PD control as MPC approximation
        Kp = 1.0
        Kd = 0.5
        action = Kp * error[:3] + Kd * state[3:6]
        
        return {
            "status": "success",
            "method": "MPC_Simplified",
            "control_signal": action.tolist(),
            "horizon": horizon,
            "predicted_cost": float(np.linalg.norm(error))
        }
    
    def tune_lqr(self, q_cost: float, r_cost: float) -> Dict[str, float]:
        """Tune LQR cost parameters."""
        # Calculate resulting gains
        omega = math.sqrt(q_cost / r_cost)
        
        return {
            "natural_frequency_hz": round(omega / (2 * math.pi), 3),
            "rise_time_sec": round(1.8 / omega, 3),
            "settling_time_sec": round(4.6 / (0.707 * omega), 3),
            "bandwidth_hz": round(omega / (2 * math.pi), 3)
        }


# Convenience functions
def quick_lqr(inertia: List[float] = [0.005, 0.005, 0.01],
              q_cost: float = 100.0,
              r_cost: float = 1.0) -> Dict[str, Any]:
    """Quick LQR gain calculation."""
    agent = ControlAgent()
    return agent.run({
        "control_mode": "LQR",
        "inertia_tensor": inertia,
        "q_error_cost": q_cost,
        "r_effort_cost": r_cost
    })


def quick_rl(state: List[float], target: List[float]) -> Dict[str, Any]:
    """Quick RL control inference."""
    agent = ControlAgent()
    return agent.run({
        "control_mode": "RL",
        "state_vec": state,
        "target_vec": target
    })
