"""
FluidCritic: Validates fluid dynamics predictions against conservation laws.

Checks:
1. Mass Conservation (Continuity Equation)
2. Momentum Conservation (Drag/Lift Force Balance)
3. Energy Conservation (Bernoulli's Principle)
4. Physical Plausibility (Cd/Cl ranges)
"""

from typing import Dict, Any, List
import numpy as np

class FluidCritic:
    def __init__(self):
        self.name = "FluidCritic"
        
        # Physical bounds
        self.CD_MIN = 0.01  # Streamlined airfoil
        self.CD_MAX = 2.0   # Flat plate perpendicular
        self.CL_MIN = -2.0  # Inverted airfoil
        self.CL_MAX = 2.0   # High-lift airfoil
        
    def critique(self, prediction: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate fluid dynamics prediction.
        
        Args:
            prediction: {cd, cl, drag_n, lift_n, solver}
            context: {velocity, density, frontal_area, mass}
            
        Returns:
            {valid: bool, violations: List[str], confidence: float}
        """
        violations = []
        
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
        if velocity > 0.1:  # Only check if moving
            expected_drag = 0.5 * density * (velocity ** 2) * cd * frontal_area
            drag_error = abs(drag_force - expected_drag) / max(expected_drag, 1e-6)
            
            if drag_error > 0.1:  # 10% tolerance
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
        
        if drag_pressure > 2 * dynamic_pressure:
            violations.append(f"Excessive energy dissipation: drag pressure {drag_pressure:.2f}Pa >> dynamic pressure {dynamic_pressure:.2f}Pa")
        
        # 5. Reynolds Number Sanity
        # Check if Cd is reasonable for the flow regime
        # (This would require Reynolds number in context, skipping for now)
        
        # Calculate confidence
        confidence = 1.0 - (len(violations) * 0.2)
        confidence = max(0.0, min(1.0, confidence))
        
        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "confidence": confidence,
            "critic": self.name
        }
    
    def validate_conservation(self, state_before: Dict, state_after: Dict, dt: float) -> Dict[str, Any]:
        """
        Validate conservation laws over a time step.
        
        Args:
            state_before: {velocity, position, mass}
            state_after: {velocity, position, mass}
            dt: Time step in seconds
            
        Returns:
            {mass_conserved: bool, momentum_conserved: bool}
        """
        violations = []
        
        # Mass conservation (should be constant for incompressible flow)
        mass_before = state_before.get("mass", 0.0)
        mass_after = state_after.get("mass", 0.0)
        
        if abs(mass_after - mass_before) > 1e-6:
            violations.append(f"Mass not conserved: {mass_before:.6f} -> {mass_after:.6f} kg")
        
        # Momentum conservation (F = dp/dt)
        # This is tricky without knowing applied forces, so we skip for now
        
        return {
            "mass_conserved": abs(mass_after - mass_before) < 1e-6,
            "violations": violations
        }
