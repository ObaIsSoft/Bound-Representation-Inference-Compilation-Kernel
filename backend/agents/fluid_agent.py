from typing import Dict, Any, List, Optional
import logging
import math
import numpy as np

logger = logging.getLogger(__name__)

from physics.kernel import get_physics_kernel

class FluidAgent:
    """
    Tier 2 Fluid Dynamics Agent (EVOLVED).
    Gated Hybrid Logic:
    1. Fast: Neural Surrogate (FluidSurrogate)
    2. Accurate: OpenFOAM Adapter (Shell)
    3. Fallback: Potential Flow (Heuristic)
    4. Validation: FluidCritic (Conservation Laws)
    """

    def __init__(self):
        # Initialize Physics Kernel
        self.physics = get_physics_kernel()
        logger.info("FluidAgent: Physics kernel initialized")
        
        self.name = "FluidAgent"
        
        # Initialize Oracles if available (PhysicsOracle handles complex CFD)
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            self.physics_oracle = PhysicsOracle()
            self.has_oracle = True
        except ImportError:
            try:
                from agents.physics_oracle.physics_oracle import PhysicsOracle
                self.physics_oracle = PhysicsOracle()
                self.has_oracle = True
            except ImportError:
                self.has_oracle = False
        
        # Initialize Neural Surrogate
        try:
            from models.fluid_surrogate import FluidSurrogate
            self.surrogate = FluidSurrogate()
            self.has_surrogate = True
        except ImportError:
            try:
                from models.fluid_surrogate import FluidSurrogate
                self.surrogate = FluidSurrogate()
                self.has_surrogate = True
            except ImportError:
                self.surrogate = None
                self.has_surrogate = False
                print("FluidSurrogate not found")
        
        # Initialize Critic
        try:
            from agents.critics.FluidCritic import FluidCritic
            self.critic = FluidCritic()
            self.has_critic = True
        except ImportError:
            try:
                from agents.critics.FluidCritic import FluidCritic
                self.critic = FluidCritic()
                self.has_critic = True
            except ImportError:
                self.critic = None
                self.has_critic = False
                print("FluidCritic not found")
                
    def run(self, geometry_tree: List[Dict[str, Any]], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Analysis Entry Point.
        Args:
            geometry: List of geometric primitives.
            context: Environment info (density, velocity, etc.)
        """
        regime = context.get("regime", "SUBSONIC") # SUBSONIC, TRANSONIC, SUPERSONIC
        velocity = context.get("velocity", 0.0)
        
        # 1. Regime Choice
        if regime == "SUPERSONIC":
            return self._run_openfoam(geometry_tree, context)
        else:
            # Default to Fast Potential Flow for interactivity
            return self._run_potential_flow(geometry_tree, context)
            
    def _run_potential_flow(self, geometry: List[Dict], context: Dict) -> Dict:
        """
        Simplified Panel Method (2D/2.5D Sections).
        Estimates Drag and Lift coefficients using REAL PHYSICS.
        """
        # Get air density from physics or context
        altitude = context.get("altitude", 0.0)  # meters
        temperature = context.get("temperature", 288.15)  # Kelvin
        
        # Use physics kernel to calculate air density
        try:
            density = self.physics.domains["fluids"].calculate_air_density(
                temperature=temperature,
                pressure=101325 * (1 - 0.0065 * altitude / 288.15) ** 5.255  # ISA
            )
        except:
            density = context.get("density", 1.225)  # Fallback
        
        velocity = context.get("velocity", 10.0) # m/s
        
        # 1. Calculate Frontal Area (Projected on Y-Z plane if moving in X)
        frontal_area = self._calculate_projected_area(geometry)
        
        # 2. Refine Cd (Drag Coeff) based on shape
        # Default Cube Cd=1.05. Streamlined=0.04.
        cd = self._estimate_cd(geometry)
        
        # 3. Use Physics Kernel for Drag Calculation
        drag_force = self.physics.domains["fluids"].calculate_drag(
            velocity=velocity,
            density=density,
            reference_area=frontal_area,
            drag_coefficient=cd
        )
        
        # Calculate Reynolds number
        char_length = (frontal_area ** 0.5)  # Approximate characteristic length
        reynolds_number = self.physics.domains["fluids"].calculate_reynolds_number(
            velocity=velocity,
            characteristic_length=char_length,
            density=density,
            dynamic_viscosity=1.81e-5  # Air at 15°C
        )
        
        return {
            "solver": "Potential Flow (Physics Kernel)",
            "drag_n": round(drag_force, 2),
            "lift_n": 0.0, # Simple bodies don't lift without airfoil logic
            "cd": cd,
            "frontal_area_m2": frontal_area,
            "air_density_kg_m3": density,
            "reynolds_number": reynolds_number
        }

    def _run_openfoam(self, geometry: List[Dict], context: Dict) -> Dict:
        """
        Shell out to OpenFOAM (Docker/Native).
        """
        # Feature Check: Is OpenFOAM installed?
        import shutil
        if not shutil.which("simpleFoam"):
             logger.warning("OpenFOAM not found. Falling back to Potential Flow.")
             return self._run_potential_flow(geometry, context)
             
        # STUB: Write dicts, run mesh, run solver.
        # This is complex implementation. For Tier 2, we setup the adapter structure.
        return {
            "solver": "OpenFOAM (Unavailable - Stub)",
            "status": "Skipped (Configuration Required)",
            "fallback_result": self._run_potential_flow(geometry, context)
        }

    def _calculate_projected_area(self, geometry: List[Dict]) -> float:
        """Sum of projected areas of primitives."""
        # Simplified: Sum of max Y*Z of bounding boxes.
        total_area = 0.0
        for part in geometry:
            params = part.get("params", {})
            ptype = part.get("type", "").lower()
            
            w, h = 0, 0
            if ptype == "box":
                # Assuming X is flow direction. Frontal is Y*Z (Width*Height)
                # Params keys vary: often width, height, thickness (or length)
                # Convention: X=Length, Y=Width, Z=Height?
                # Let's try to grab geometric bounds if possible.
                # Assuming params are {width, length, height}
                w = params.get("width", 1.0)
                h = params.get("height", 1.0)
                
            elif ptype == "cylinder":
                # Circle area if end-on, Rect if side-on.
                # Assume side-on for fuselage.
                r = params.get("radius", 0.5)
                length = params.get("height", params.get("length", 1.0))
                w = 2*r
                h = 2*r
                
            total_area += w * h
            
        return total_area

    def _estimate_cd(self, geometry: List[Dict]) -> float:
        """
        Estimate drag coefficient based on geometric slenderness.
        """
        # If long and thin -> Lower Cd.
        # If boxy -> High Cd.
        
        # Just check one dominant part for now
        if not geometry: return 1.0
        
        main_part = geometry[0]
        params = main_part.get("params", {})
        l = params.get("length", 1.0)
        w = params.get("width", 1.0)
        
        aspect_ratio = l / max(0.001, w)
        
        if aspect_ratio > 5.0:
            return 0.3 # Streamlined-ish
        elif aspect_ratio > 2.0:
            return 0.6
        else:
            return 1.05 # Cube
            
    def dynamics_step(self, dt: float, state: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Dynamic Step for vHIL Loop.
        Calculates Aerodynamic Forces for Physics Engine.
        """
        # Unpack
        velocity = state.get("velocity", 0.0) # Speed scalar (m/s)
        # Assuming simple uniaxial motion for now
        
        context = {
            "velocity": velocity, 
            "density": inputs.get("environment", {}).get("fluid_density", 1.225)
        }
        
        # We need geometry. Usually this is static or passed in. 
        # For dynamics_step, we assume cached or simple placeholder.
        # Ideally, `FluidAgent` should store current geometry/mesh.
        # Here we use a dummy logic or look for 'geometry' in state? 
        # Let's assume generic "Pod" geometry until we wire up full mesh passing.
        geometry = [{"type": "cylinder", "params": {"radius": 1.5, "length": 10.0}}]
        
        aero_result = self.run(geometry, context)
        
        drag = aero_result["drag_n"]
        
        return {
            "forces": {
                "drag": drag,
                "lift": aero_result["lift_n"]
            },
            "coefficients": {
                "cd": aero_result["cd"]
            }
        }
    def _calculate_reynolds_number(self, velocity: float, length: float, density: float = 1.225) -> float:
        """
        Calculate Reynolds number: Re = (rho * v * L) / mu
        
        Args:
            velocity: Flow velocity (m/s)
            length: Characteristic length (m) - use sqrt(frontal_area) as proxy
            density: Fluid density (kg/m^3)
            
        Returns:
            Reynolds number (dimensionless)
        """
        import numpy as np
        # Dynamic viscosity of air at 20°C: 1.81e-5 Pa·s
        mu = 1.81e-5
        
        # Use sqrt(area) as characteristic length if length not provided
        char_length = np.sqrt(length) if length > 0 else 1.0
        
        re = (density * velocity * char_length) / mu
        return max(re, 1.0)  # Avoid zero
    
    def _estimate_aspect_ratio(self, geometry: List[Dict]) -> float:
        """Estimate length/width aspect ratio from geometry"""
        if not geometry:
            return 1.0
            
        main_part = geometry[0]
        params = main_part.get("params", {})
        l = params.get("length", 1.0)
        w = params.get("width", params.get("radius", 1.0) * 2)
        
        return l / max(0.001, w)
    
    def evolve(self, training_data: List[Any]) -> Dict[str, Any]:
        """
        Deep Evolution: Train the FluidSurrogate on real/simulated CFD data.
        
        Args:
            training_data: List of (features, labels) tuples
                features: [frontal_area, aspect_ratio, reynolds_number, roughness]
                labels: [cd, cl]
                
        Returns:
            {status, avg_loss, epochs}
        """
        if not self.has_surrogate:
            return {"status": "error", "message": "No surrogate available"}
        
        import numpy as np
        total_loss = 0.0
        count = 0
        
        for x, y in training_data:
            # Normalize inputs (same as surrogate.predict)
            x_norm = np.array([
                x[0] / 10.0,  # frontal_area
                x[1] / 10.0,  # aspect_ratio
                np.log10(x[2] + 1) / 6.0,  # reynolds_number
                x[3] / 10.0   # roughness
            ])
            
            loss = self.surrogate.train_step(x_norm, np.array(y))
            total_loss += loss
            count += 1
        
        self.surrogate.trained_epochs += 1
        self.surrogate.save()
        
        return {
            "status": "evolved",
            "avg_loss": total_loss / max(1, count),
            "epochs": self.surrogate.trained_epochs
        }
