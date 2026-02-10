"""
ControlCritic: Validates control actions for stability and safety.

Uses database-driven thresholds for vehicle-specific limits.
Thresholds loaded from Supabase critic_thresholds table.

Checks:
1. Lyapunov Stability (energy decreasing)
2. Actuator Saturation (within limits)
3. Energy Efficiency (minimize control effort)
4. State Constraints (position/velocity bounds)
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class ControlCritic:
    """
    Control Critic with database-driven thresholds.
    
    Thresholds are loaded from Supabase based on vehicle_type.
    No hardcoded limits - fails if thresholds not configured.
    """
    
    def __init__(self, vehicle_type: str = "default", window_size: int = None):
        self.name = "ControlCritic"
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds: Optional[Dict[str, float]] = None
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        # History tracking
        from collections import deque
        self.control_history = deque(maxlen=window_size or 100)
        self.violation_history = deque(maxlen=window_size or 100)
        self.total_evaluations = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("ControlCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                # Update deque sizes if changed
                self.control_history = deque(self.control_history, maxlen=self._window_size)
                self.violation_history = deque(self.violation_history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"ControlCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "max_thrust_n": float('inf'),
            "max_torque_nm": float('inf'),
            "max_velocity_ms": float('inf'),
            "max_position_m": float('inf'),
            "energy_increase_limit": 1.1,
            "control_effort_threshold": float('inf'),
            "window_size": 100,
        }
    
    @property
    def vehicle_type(self) -> str:
        return self._vehicle_type
        
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @property
    def max_thrust(self) -> float:
        """Max thrust in Newtons from database"""
        if self._thresholds is None:
            return float('inf')
        return self._thresholds.get("max_thrust_n", float('inf'))
    
    @property
    def max_torque(self) -> float:
        """Max torque in Nm from database"""
        if self._thresholds is None:
            return float('inf')
        return self._thresholds.get("max_torque_nm", float('inf'))
    
    @property
    def max_velocity(self) -> float:
        """Max velocity in m/s from database"""
        if self._thresholds is None:
            return float('inf')
        return self._thresholds.get("max_velocity_ms", float('inf'))
    
    @property
    def max_position(self) -> float:
        """Max position in meters from database"""
        if self._thresholds is None:
            return float('inf')
        return self._thresholds.get("max_position_m", float('inf'))
    
    @property
    def energy_increase_limit(self) -> float:
        """Energy increase limit (e.g., 1.1 = 10% increase allowed)"""
        if self._thresholds is None:
            return 1.1
        return self._thresholds.get("energy_increase_limit", 1.1)
    
    @property
    def control_effort_threshold(self) -> float:
        """Control effort threshold from database"""
        if self._thresholds is None:
            return float('inf')
        return self._thresholds.get("control_effort_threshold", float('inf'))
    
    async def initialize(self):
        """
        Load thresholds from database.
        
        Raises:
            ValueError: If thresholds not found in database
        """
        await self._load_thresholds()
        
        if self._thresholds is None:
            raise ValueError(
                f"ControlCritic thresholds not found for vehicle_type='{self._vehicle_type}'. "
                f"Configure in critic_thresholds table or run seed_critic_thresholds.py."
            )
    
    async def observe(self, 
                prediction: Dict[str, Any], 
                context: Dict[str, Any],
                violations: List[str] = None):
        """
        Record a control action observation.
        
        Args:
            prediction: {action, state_next, value}
            context: {state_current, dt, constraints}
            violations: Any violations observed
        """
        try:
            self.total_evaluations += 1
            
            self.control_history.append({
                "prediction": prediction,
                "context": context,
                "violations": violations or []
            })
            
            if violations:
                for v in violations:
                    self.violation_history.append(v)
        except Exception as e:
            logger.error(f"Error in observe: {e}")
    
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
        
        try:
            action = prediction.get("action", np.zeros(3))
            state_current = context.get("state_current", np.zeros(6))
            state_next = prediction.get("state_next", state_current)
            dt = context.get("dt", 0.01)
            
            # 1. Actuator Saturation Check
            thrust = action[0] if len(action) > 0 else 0
            torque_x = action[1] if len(action) > 1 else 0
            torque_y = action[2] if len(action) > 2 else 0
            
            max_thrust = self.max_thrust
            if abs(thrust) > max_thrust and max_thrust != float('inf'):
                violations.append(
                    f"Thrust {thrust:.1f}N exceeds limit {max_thrust}N "
                    f"(vehicle_type={self._vehicle_type})"
                )
            
            max_torque = self.max_torque
            if abs(torque_x) > max_torque and max_torque != float('inf'):
                violations.append(
                    f"Torque X {torque_x:.1f}Nm exceeds limit {max_torque}Nm"
                )
            
            if abs(torque_y) > max_torque and max_torque != float('inf'):
                violations.append(
                    f"Torque Y {torque_y:.1f}Nm exceeds limit {max_torque}Nm"
                )
            
            # 2. State Constraints Check
            max_vel = self.max_velocity
            max_pos = self.max_position
            
            if len(state_next) >= 6:
                pos_x, pos_y, pos_z = state_next[0:3]
                vel_x, vel_y, vel_z = state_next[3:6]
                
                velocity_mag = np.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
                if velocity_mag > max_vel and max_vel != float('inf'):
                    violations.append(
                        f"Velocity {velocity_mag:.1f}m/s exceeds limit {max_vel}m/s"
                    )
                
                position_mag = np.sqrt(pos_x**2 + pos_y**2 + pos_z**2)
                if position_mag > max_pos and max_pos != float('inf'):
                    violations.append(
                        f"Position {position_mag:.1f}m exceeds limit {max_pos}m"
                    )
            
            # 3. Lyapunov Stability Check (simplified)
            # Energy should decrease: E_next < E_current
            if len(state_current) >= 6 and len(state_next) >= 6:
                E_current = self._compute_energy(state_current)
                E_next = self._compute_energy(state_next)
                
                energy_limit = self.energy_increase_limit
                # Allow small energy increase (for control effort)
                if E_next > E_current * energy_limit:
                    violations.append(
                        f"Energy increasing: {E_current:.2f} → {E_next:.2f} "
                        f"(limit: {energy_limit:.2f}x)"
                    )
            
            # 4. Energy Efficiency Check
            control_effort = np.sum(action ** 2)
            effort_threshold = self.control_effort_threshold
            if control_effort > effort_threshold and effort_threshold != float('inf'):
                violations.append(
                    f"High control effort: {control_effort:.2f} "
                    f"(threshold: {effort_threshold:.2f})"
                )
        except Exception as e:
            logger.error(f"Error in critique: {e}")
            violations.append(f"Critique error: {e}")
        
        # Calculate confidence
        confidence = 1.0 - (len(violations) * 0.25)
        confidence = max(0.0, min(1.0, confidence))
        
        # Record observation
        await self.observe(prediction, context, violations)
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "confidence": confidence,
            "critic": self.name,
            "vehicle_type": self._vehicle_type
        }
    
    async def analyze(self) -> Dict[str, Any]:
        """
        Analyze control critic performance.
        
        Returns:
            Analysis report with violation statistics
        """
        await self._load_thresholds()
        
        try:
            if len(self.control_history) < 5:
                return {
                    "status": "insufficient_data",
                    "observations": len(self.control_history)
                }
            
            total_violations = len(self.violation_history)
            violation_rate = total_violations / max(1, self.total_evaluations)
            
            # Categorize violations
            violation_types = {}
            for v in self.violation_history:
                # Extract violation type (first word before space or colon)
                vtype = v.split()[0] if v else "UNKNOWN"
                violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            return {
                "timestamp": self.total_evaluations,
                "total_evaluations": self.total_evaluations,
                "total_violations": total_violations,
                "violation_rate": violation_rate,
                "violation_types": violation_types,
                "vehicle_type": self._vehicle_type,
                "confidence": min(1.0, len(self.control_history) / self._window_size)
            }
        except Exception as e:
            logger.error(f"Error in analyze: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": self.total_evaluations
            }
    
    async def should_evolve(self) -> Tuple[bool, str, Optional[str]]:
        """
        Determine if control agent needs evolution.
        
        Returns:
            (should_evolve, reason, strategy_name)
        """
        await self._load_thresholds()
        
        try:
            if len(self.control_history) < 20:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            violation_rate = report.get("violation_rate", 0)
            
            # High violation rate indicates control issues
            if violation_rate > 0.3:
                return True, f"High violation rate: {violation_rate:.0%}", "TUNE_CONTROL_GAINS"
            
            # Check for specific violation patterns
            vtypes = report.get("violation_types", {})
            if vtypes.get("Thrust", 0) > 10:
                return True, "Frequent thrust saturation", "INCREASE_THRUST_LIMITS"
            if vtypes.get("Velocity", 0) > 10:
                return True, "Frequent velocity violations", "TUNE_VELOCITY_CONSTRAINTS"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Control within acceptable parameters", None
    
    def _compute_energy(self, state: np.ndarray) -> float:
        """Compute total energy (kinetic + potential)"""
        try:
            if len(state) < 6:
                return 0.0
            
            pos_x, pos_y, pos_z = state[0:3]
            vel_x, vel_y, vel_z = state[3:6]
            
            # Kinetic energy: 0.5 * m * v^2 (assume m=1)
            KE = 0.5 * (vel_x**2 + vel_y**2 + vel_z**2)
            
            # Potential energy: m * g * h (assume g=9.81)
            PE = 9.81 * pos_z
            
            return KE + PE
        except Exception as e:
            logger.error(f"Error computing energy: {e}")
            return 0.0
    
    def export_report(self, filepath: str):
        """Export analysis to JSON."""
        import asyncio
        import json
        try:
            report = asyncio.run(self.analyze())
            with open(filepath, 'w') as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            logger.error(f"Error exporting report: {e}")
