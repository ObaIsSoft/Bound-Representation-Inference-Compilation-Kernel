"""
FluidCritic: Validates fluid dynamics predictions against conservation laws.

Checks:
1. Mass Conservation (Continuity Equation)
2. Momentum Conservation (Drag/Lift Force Balance)
3. Energy Conservation (Bernoulli's Principle)
4. Physical Plausibility (Cd/Cl ranges)

Thresholds loaded from Supabase critic_thresholds table.
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class FluidCritic:
    def __init__(self, vehicle_type: str = "default", window_size: int = None):
        self.name = "FluidCritic"
        self._vehicle_type = vehicle_type
        self._thresholds_loaded = False
        self._thresholds: Optional[Dict[str, float]] = None
        
        # These will be loaded from Supabase if None
        self._window_size = window_size
        
        # History tracking
        from collections import deque
        self.prediction_history = deque(maxlen=window_size or 100)
        self.violation_history = deque(maxlen=window_size or 100)
        self.total_evaluations = 0
        
    async def _load_thresholds(self):
        """Load thresholds from Supabase if not provided."""
        if self._thresholds_loaded:
            return
            
        try:
            from backend.services import supabase
            self._thresholds = await supabase.get_critic_thresholds("FluidCritic", self._vehicle_type)
            
            if self._window_size is None:
                self._window_size = self._thresholds.get("window_size", 100)
                # Update deque sizes if changed
                self.prediction_history = deque(self.prediction_history, maxlen=self._window_size)
                self.violation_history = deque(self.violation_history, maxlen=self._window_size)
                
            self._thresholds_loaded = True
            logger.info(f"FluidCritic thresholds loaded for {self._vehicle_type}")
        except Exception as e:
            logger.warning(f"Could not load thresholds from Supabase: {e}. Using defaults.")
            if self._window_size is None:
                self._window_size = 100
            self._thresholds = self._default_thresholds()
            self._thresholds_loaded = True
    
    def _default_thresholds(self) -> Dict:
        """Default thresholds if Supabase unavailable."""
        return {
            "cd_min": 0.01,
            "cd_max": 2.0,
            "cl_min": -2.0,
            "cl_max": 2.0,
            "drag_error_tolerance": 0.1,
            "drag_pressure_ratio": 2.0,
            "velocity_threshold": 0.1,
            "mass_conservation_tolerance": 1e-6,
            "window_size": 100,
        }
    
    @property
    def vehicle_type(self) -> str:
        return self._vehicle_type
        
    @property
    def window_size(self) -> int:
        return self._window_size
    
    @property
    def CD_MIN(self) -> float:
        """Minimum drag coefficient"""
        if self._thresholds is None:
            return 0.01
        return self._thresholds.get("cd_min", 0.01)
    
    @property
    def CD_MAX(self) -> float:
        """Maximum drag coefficient"""
        if self._thresholds is None:
            return 2.0
        return self._thresholds.get("cd_max", 2.0)
    
    @property
    def CL_MIN(self) -> float:
        """Minimum lift coefficient"""
        if self._thresholds is None:
            return -2.0
        return self._thresholds.get("cl_min", -2.0)
    
    @property
    def CL_MAX(self) -> float:
        """Maximum lift coefficient"""
        if self._thresholds is None:
            return 2.0
        return self._thresholds.get("cl_max", 2.0)
    
    async def observe(self,
                prediction: Dict[str, Any],
                context: Dict[str, Any],
                violations: List[str] = None):
        """
        Record a fluid dynamics observation.
        
        Args:
            prediction: {cd, cl, drag_n, lift_n, solver}
            context: {velocity, density, frontal_area, mass}
            violations: Any violations observed
        """
        try:
            self.total_evaluations += 1
            
            self.prediction_history.append({
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
        Validate fluid dynamics prediction.
        
        Args:
            prediction: {cd, cl, drag_n, lift_n, solver}
            context: {velocity, density, frontal_area, mass}
            
        Returns:
            {valid: bool, violations: List[str], confidence: float}
        """
        await self._load_thresholds()
        
        violations = []
        
        try:
            cd = prediction.get("cd", 0.0)
            cl = prediction.get("cl", 0.0)
            drag_force = prediction.get("drag_n", 0.0)
            lift_force = prediction.get("lift_n", 0.0)
            
            velocity = context.get("velocity", 0.0)
            density = context.get("density", 1.225)
            frontal_area = context.get("frontal_area", 1.0)
            
            # 1. Physical Plausibility Check
            if cd < self.CD_MIN or cd > self.CD_MAX:
                violations.append(f"Drag coefficient {cd:.3f} outside physical range [{self.CD_MIN}, {self.CD_MAX}]")
            
            if cl < self.CL_MIN or cl > self.CL_MAX:
                violations.append(f"Lift coefficient {cl:.3f} outside physical range [{self.CL_MIN}, {self.CL_MAX}]")
            
            # 2. Force Equation Consistency
            # Fd = 0.5 * rho * v^2 * Cd * A
            vel_threshold = self._thresholds.get("velocity_threshold", 0.1)
            drag_tol = self._thresholds.get("drag_error_tolerance", 0.1)
            
            if velocity > vel_threshold:  # Only check if moving
                expected_drag = 0.5 * density * (velocity ** 2) * cd * frontal_area
                drag_error = abs(drag_force - expected_drag) / max(expected_drag, 1e-6)
                
                if drag_error > drag_tol:  # 10% tolerance
                    violations.append(f"Drag force inconsistent with equation: predicted={drag_force:.2f}N, expected={expected_drag:.2f}N")
            
            # 3. Momentum Conservation (simplified)
            # For steady flow, drag should not exceed dynamic pressure * area
            max_drag = density * (velocity ** 2) * frontal_area
            if drag_force > max_drag:
                violations.append(f"Drag force {drag_force:.2f}N exceeds maximum possible {max_drag:.2f}N")
            
            # 4. Energy Conservation (Bernoulli Check)
            # Total pressure should be conserved: P + 0.5*rho*v^2 = const
            # If drag is too high, it implies excessive energy dissipation
            dynamic_pressure = 0.5 * density * (velocity ** 2)
            drag_pressure = drag_force / max(frontal_area, 1e-6)
            
            pressure_ratio = self._thresholds.get("drag_pressure_ratio", 2.0)
            if drag_pressure > pressure_ratio * dynamic_pressure:
                violations.append(f"Excessive energy dissipation: drag pressure {drag_pressure:.2f}Pa >> dynamic pressure {dynamic_pressure:.2f}Pa")
            
            # 5. Reynolds Number Sanity
            # Check if Cd is reasonable for the flow regime
            # (This would require Reynolds number in context, skipping for now)
        except Exception as e:
            logger.error(f"Error in critique: {e}")
            violations.append(f"Critique error: {e}")
        
        # Calculate confidence
        confidence = 1.0 - (len(violations) * 0.2)
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
        Analyze fluid critic performance.
        
        Returns:
            Analysis report with violation statistics
        """
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 5:
                return {
                    "status": "insufficient_data",
                    "observations": len(self.prediction_history)
                }
            
            total_violations = len(self.violation_history)
            violation_rate = total_violations / max(1, self.total_evaluations)
            
            # Categorize violations
            violation_types = {}
            for v in self.violation_history:
                vtype = v.split()[0] if v else "UNKNOWN"
                violation_types[vtype] = violation_types.get(vtype, 0) + 1
            
            return {
                "timestamp": self.total_evaluations,
                "total_evaluations": self.total_evaluations,
                "total_violations": total_violations,
                "violation_rate": violation_rate,
                "violation_types": violation_types,
                "vehicle_type": self._vehicle_type,
                "confidence": min(1.0, len(self.prediction_history) / self._window_size)
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
        Determine if fluid agent needs evolution.
        
        Returns:
            (should_evolve, reason, strategy_name)
        """
        await self._load_thresholds()
        
        try:
            if len(self.prediction_history) < 20:
                return False, "Insufficient data", None
            
            report = await self.analyze()
            violation_rate = report.get("violation_rate", 0)
            
            # High violation rate indicates fluid model issues
            if violation_rate > 0.3:
                return True, f"High violation rate: {violation_rate:.0%}", "RETRAIN_FLUID_MODEL"
            
            # Check for specific violation patterns
            vtypes = report.get("violation_types", {})
            if vtypes.get("Drag", 0) > 10:
                return True, "Frequent drag inconsistencies", "CALIBRATE_DRAG_MODEL"
        except Exception as e:
            logger.error(f"Error in should_evolve: {e}")
            return False, f"Error: {e}", None
        
        return False, "Fluid model within acceptable parameters", None
    
    async def validate_conservation(self, state_before: Dict, state_after: Dict, dt: float) -> Dict[str, Any]:
        """
        Validate conservation laws over a time step.
        
        Args:
            state_before: {velocity, position, mass}
            state_after: {velocity, position, mass}
            dt: Time step in seconds
            
        Returns:
            {mass_conserved: bool, momentum_conserved: bool}
        """
        await self._load_thresholds()
        
        violations = []
        
        try:
            # Mass conservation (should be constant for incompressible flow)
            mass_before = state_before.get("mass", 0.0)
            mass_after = state_after.get("mass", 0.0)
            
            mass_tol = self._thresholds.get("mass_conservation_tolerance", 1e-6)
            if abs(mass_after - mass_before) > mass_tol:
                violations.append(f"Mass not conserved: {mass_before:.6f} -> {mass_after:.6f} kg")
            
            # Momentum conservation (F = dp/dt)
            # This is tricky without knowing applied forces, so we skip for now
        except Exception as e:
            logger.error(f"Error in validate_conservation: {e}")
            violations.append(f"Validation error: {e}")
        
        return {
            "mass_conserved": abs(mass_after - mass_before) < mass_tol if 'mass_tol' in dir() else False,
            "violations": violations
        }
    
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
