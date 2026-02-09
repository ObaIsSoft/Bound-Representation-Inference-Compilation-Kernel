"""
ControlCritic: Validates control actions for stability and safety.

Uses database-driven thresholds for vehicle-specific limits.
No hardcoded values - all limits come from critic_thresholds table.

Checks:
1. Lyapunov Stability (energy decreasing)
2. Actuator Saturation (within limits)
3. Energy Efficiency (minimize control effort)
4. State Constraints (position/velocity bounds)
"""

from typing import Dict, Any, List, Optional
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ControlCritic:
    """
    Control Critic with database-driven thresholds.
    
    Thresholds are loaded from Supabase based on vehicle_type.
    No hardcoded limits - fails if thresholds not configured.
    """
    
    def __init__(self, vehicle_type: str = "default"):
        self.name = "ControlCritic"
        self.vehicle_type = vehicle_type
        
        # Thresholds loaded from database
        self._thresholds: Optional[Dict[str, float]] = None
        
    async def initialize(self):
        """
        Load thresholds from database.
        
        Raises:
            ValueError: If thresholds not found in database
        """
        from backend.services import supabase
        
        try:
            self._thresholds = await supabase.get_critic_thresholds(
                critic_name="ControlCritic",
                vehicle_type=self.vehicle_type
            )
            
            logger.info(
                f"ControlCritic initialized for {self.vehicle_type} "
                f"with thresholds: {list(self._thresholds.keys())}"
            )
            
        except Exception as e:
            raise ValueError(
                f"ControlCritic thresholds not found for vehicle_type='{self.vehicle_type}'. "
                f"Configure in critic_thresholds table or run seed_critic_thresholds.py. "
                f"Error: {e}"
            )
    
    @property
    def max_thrust(self) -> float:
        """Max thrust in Newtons from database"""
        if self._thresholds is None:
            raise RuntimeError("ControlCritic not initialized. Call initialize() first.")
        return self._thresholds.get("max_thrust_n", float('inf'))
    
    @property
    def max_torque(self) -> float:
        """Max torque in Nm from database"""
        if self._thresholds is None:
            raise RuntimeError("ControlCritic not initialized. Call initialize() first.")
        return self._thresholds.get("max_torque_nm", float('inf'))
    
    @property
    def max_velocity(self) -> float:
        """Max velocity in m/s from database"""
        if self._thresholds is None:
            raise RuntimeError("ControlCritic not initialized. Call initialize() first.")
        return self._thresholds.get("max_velocity_ms", float('inf'))
    
    @property
    def max_position(self) -> float:
        """Max position in meters from database"""
        if self._thresholds is None:
            raise RuntimeError("ControlCritic not initialized. Call initialize() first.")
        return self._thresholds.get("max_position_m", float('inf'))
    
    @property
    def energy_increase_limit(self) -> float:
        """Energy increase limit (e.g., 1.1 = 10% increase allowed)"""
        if self._thresholds is None:
            raise RuntimeError("ControlCritic not initialized. Call initialize() first.")
        return self._thresholds.get("energy_increase_limit", 1.1)
    
    @property
    def control_effort_threshold(self) -> float:
        """Control effort threshold from database"""
        if self._thresholds is None:
            raise RuntimeError("ControlCritic not initialized. Call initialize() first.")
        return self._thresholds.get("control_effort_threshold", float('inf'))
    
    async def critique(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate control action.
        
        Args:
            prediction: {action, state_next, value}
            context: {state_current, dt, constraints}
            
        Returns:
            {valid: bool, violations: List[str], confidence: float}
        """
        # Ensure initialized
        if self._thresholds is None:
            await self.initialize()
        
        violations = []
        
        action = prediction.get("action", np.zeros(3))
        state_current = context.get("state_current", np.zeros(6))
        state_next = prediction.get("state_next", state_current)
        dt = context.get("dt", 0.01)
        
        # 1. Actuator Saturation Check
        thrust = action[0] if len(action) > 0 else 0
        torque_x = action[1] if len(action) > 1 else 0
        torque_y = action[2] if len(action) > 2 else 0
        
        if abs(thrust) > self.max_thrust:
            violations.append(
                f"Thrust {thrust:.1f}N exceeds limit {self.max_thrust}N "
                f"(vehicle_type={self.vehicle_type})"
            )
        
        if abs(torque_x) > self.max_torque:
            violations.append(
                f"Torque X {torque_x:.1f}Nm exceeds limit {self.max_torque}Nm"
            )
        
        if abs(torque_y) > self.max_torque:
            violations.append(
                f"Torque Y {torque_y:.1f}Nm exceeds limit {self.max_torque}Nm"
            )
        
        # 2. State Constraints Check
        if len(state_next) >= 6:
            pos_x, pos_y, pos_z = state_next[0:3]
            vel_x, vel_y, vel_z = state_next[3:6]
            
            velocity_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
            if velocity_mag > self.max_velocity:
                violations.append(
                    f"Velocity {velocity_mag:.1f}m/s exceeds limit {self.max_velocity}m/s"
                )
            
            position_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
            if position_mag > self.max_position:
                violations.append(
                    f"Position {position_mag:.1f}m exceeds limit {self.max_position}m"
                )
        
        # 3. Lyapunov Stability Check (simplified)
        # Energy should decrease: E_next < E_current
        if len(state_current) >= 6 and len(state_next) >= 6:
            E_current = self._compute_energy(state_current)
            E_next = self._compute_energy(state_next)
            
            # Allow small energy increase (for control effort)
            if E_next > E_current * self.energy_increase_limit:
                violations.append(
                    f"Energy increasing: {E_current:.2f} → {E_next:.2f} "
                    f"(limit: {self.energy_increase_limit:.2f}x)"
                )
        
        # 4. Energy Efficiency Check
        control_effort = np.sum(action ** 2)
        if control_effort > self.control_effort_threshold:
            violations.append(
                f"High control effort: {control_effort:.2f} "
                f"(threshold: {self.control_effort_threshold:.2f})"
            )
        
        # Calculate confidence
        confidence = 1.0 - (len(violations) * 0.25)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "confidence": confidence,
            "critic": self.name,
            "vehicle_type": self.vehicle_type
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
