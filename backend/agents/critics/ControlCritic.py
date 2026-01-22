"""
ControlCritic: Validates control actions for stability and safety.

Checks:
1. Lyapunov Stability (energy decreasing)
2. Actuator Saturation (within limits)
3. Energy Efficiency (minimize control effort)
4. State Constraints (position/velocity bounds)
"""

from typing import Dict, Any, List
import numpy as np

class ControlCritic:
    def __init__(self):
        self.name = "ControlCritic"
        
        # Physical limits
        self.MAX_THRUST = 1000.0  # N
        self.MAX_TORQUE = 100.0   # Nm
        self.MAX_VELOCITY = 50.0  # m/s
        self.MAX_POSITION = 1000.0  # m
        
    def critique(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate control action.
        
        Args:
            prediction: {action, state_next, value}
            context: {state_current, dt, constraints}
            
        Returns:
            {valid: bool, violations: List[str], confidence: float}
        """
        violations = []
        
        action = prediction.get("action", np.zeros(3))
        state_current = context.get("state_current", np.zeros(6))
        state_next = prediction.get("state_next", state_current)
        dt = context.get("dt", 0.01)
        
        # 1. Actuator Saturation Check
        thrust = action[0] if len(action) > 0 else 0
        torque_x = action[1] if len(action) > 1 else 0
        torque_y = action[2] if len(action) > 2 else 0
        
        if abs(thrust) > self.MAX_THRUST:
            violations.append(f"Thrust {thrust:.1f}N exceeds limit {self.MAX_THRUST}N")
        
        if abs(torque_x) > self.MAX_TORQUE:
            violations.append(f"Torque X {torque_x:.1f}Nm exceeds limit {self.MAX_TORQUE}Nm")
        
        if abs(torque_y) > self.MAX_TORQUE:
            violations.append(f"Torque Y {torque_y:.1f}Nm exceeds limit {self.MAX_TORQUE}Nm")
        
        # 2. State Constraints Check
        if len(state_next) >= 6:
            pos_x, pos_y, pos_z = state_next[0:3]
            vel_x, vel_y, vel_z = state_next[3:6]
            
            velocity_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            if velocity_mag > self.MAX_VELOCITY:
                violations.append(f"Velocity {velocity_mag:.1f}m/s exceeds limit {self.MAX_VELOCITY}m/s")
            
            position_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
            if position_mag > self.MAX_POSITION:
                violations.append(f"Position {position_mag:.1f}m exceeds limit {self.MAX_POSITION}m")
        
        # 3. Lyapunov Stability Check (simplified)
        # Energy should decrease: E_next < E_current
        if len(state_current) >= 6 and len(state_next) >= 6:
            E_current = self._compute_energy(state_current)
            E_next = self._compute_energy(state_next)
            
            # Allow small energy increase (for control effort)
            if E_next > E_current * 1.1:
                violations.append(f"Energy increasing: {E_current:.2f} → {E_next:.2f} (unstable)")
        
        # 4. Energy Efficiency Check
        control_effort = np.sum(action ** 2)
        if control_effort > 100.0:  # Arbitrary threshold
            violations.append(f"High control effort: {control_effort:.2f}")
        
        # Calculate confidence
        confidence = 1.0 - (len(violations) * 0.25)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "confidence": confidence,
            "critic": self.name
        }
    
    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute total energy (kinetic + potential)"""
        if len(state) < 6:
            return 0.0
        
        pos_x, pos_y, pos_z = state[0:3]
        vel_x, vel_y, vel_z = state[3:6]
        
        # Kinetic energy: 0.5 * m * v^2 (assume m=1)
        KE = 0.5 * (vel_x**2 + vel_y**2 + vel_z**2)
        
        # Potential energy: m * g * h (assume g=9.81)
        PE = 9.81 * pos_z
        
        return KE + PE
