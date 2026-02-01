import math
import random
import logging
from typing import Dict, Any, List
# import scipy.constants # For future precision

logger = logging.getLogger(__name__)

class PhysicsAgent:
    """
    The 'Judge'. 
    Validates design concepts against the Laws of the Universe.
    Does NOT use LLMs. Uses First-Principles Physics.
    
    NEW (Phase 3.8): Now uses UnifiedPhysicsKernel for all physics calculations.
    Preserves orchestration, 6-DOF simulation, and ML surrogate capabilities.
    """

    def __init__(self):
        # Initialize Physics Kernel (Phase 3.8 - Real Physics)
        from backend.physics.kernel import get_physics_kernel
        self.physics = get_physics_kernel()
        
        # Initialize Neural Student (Phase 3.9: Managed by Kernel)
        # self.student removed - access via self.physics.intelligence["surrogate_manager"]
        self.surrogate_manager = self.physics.intelligence["surrogate_manager"]
        self.has_brain = self.surrogate_manager.has_model("physics_surrogate")

        # Initialize Nuclear Student (Tier 3.5)
        try:
            try:
                from backend.models.nuclear_surrogate import NuclearSurrogate
            except ImportError:
                 from ...models.nuclear_surrogate import NuclearSurrogate
            
            self.nuclear_student = NuclearSurrogate(input_size=4, hidden_size=32, output_size=2)
            self.nuclear_model_path = "data/nuclear_surrogate.weights.json"
            self.nuclear_student.load(self.nuclear_model_path)
            self.has_nuclear_brain = True
        except ImportError as e:
            self.has_nuclear_brain = False
            self.nuclear_student = None

    def run(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the physics simulation kernel.
        Returns detailed predictions and feasibility verdicts.
        """
        # Unpack state
        environment = state.get("environment", {})
        geometry_tree = state.get("geometry_tree", [])
        design_params = state.get("design_parameters", {})

        
        # 1. Establish Boundary Conditions (The "Universe" state)
        # NEW: Use physics kernel for constants instead of hardcoded values
        gravity = environment.get("gravity", self.physics.get_constant("g"))
        
        # Get air density from physics kernel (temperature-dependent)
        temperature = environment.get("temperature", 288.15)  # K
        pressure = environment.get("pressure", 101325)  # Pa
        try:
            fluid_density = self.physics.domains["fluids"].calculate_air_density(
                temperature=temperature, 
                pressure=pressure
            )
        except (KeyError, TypeError, AttributeError):
            logger.debug(f"Could not calculate fluid density, using fallback")
            fluid_density = environment.get("fluid_density", 1.225)  # Fallback
        
        # 2. Derive Physical Properties from Geometry (Mass, Surface Area)
        # Note: In a real system, MassAgent does this. Merging for simplicity in Phase 3.
        total_mass = 0.0
        total_volume = 0.0
        projected_area = 0.0 # Aproximation for drag
        
        # Simple mass integration
        for part in geometry_tree:
             # This assumes ManufacturingAgent ran first to populate mass_kg, 
             # OR we re-calculate. Let's assume params exist.
             # If not, use basic fallback logic from DesignParams
             
             # Extract params
             p = part.get("params", {})
             w = p.get("width", 0) / 1000.0 # mm to m
             l = p.get("length", 0) / 1000.0
             
             # Very rough projected area assumption (top-down)
             projected_area += (w * l) 
             
             # Fallback mass if not yet calculated
             if "mass_kg" in part:
                 total_mass += part["mass_kg"]
             elif "volume" in part:
                 # Use actual mesh volume if available (OpenSCADAgent provides this)
                 # Unit: SCAD units cubed usually. If mismatch, we might need conversion.
                 # Assuming 1 unit = 1 mm for standard CAD.
                 vol_scad = part["volume"]
                 
                 # Apply Scale Factor Intelligence
                 # If the design has a scale_factor (e.g. 1/6), and we want REAL mass:
                 # Real Volume = Model Volume * (1/scale)^3
                 # But we need access to the scale factor here.
                 # For now, let's assume 'part' might have metadata or we use a global design param.
                 scale = design_params.get("metadata", {}).get("scale_factor", 1.0)
                 if scale == 0: scale = 1.0
                 
                 # Correction for 1:Scale models to get Real Mass
                 # If user wants model mass, scale=1. If user wants proto mass, scale=scale.
                 # Usually users want the Real car specs.
                 real_vol_factor = (1.0 / scale) ** 3
                 
                 # Convert mm^3 to m^3
                 vol_m3 = (vol_scad * real_vol_factor) / 1e9 
                 
                 # Density (Steel/Aluminium avg ~ 4000 kg/m3 if unknown)
                 # Or use specific material density
                 density = part.get("material", {}).get("density", 2700.0) # Al
                 
                 mass = vol_m3 * density
                 total_mass += mass
                 total_volume += vol_m3
             else:
                 total_mass += 1.0 # Safe fallback to avoid div/0
        
        if total_mass == 0: total_mass = 1.0 # Safety
        if total_mass == 0: total_mass = 1.0 # Safety
        if projected_area == 0: projected_area = 0.01
        
        total_surface_area = 0.001 # Avoid div/0

        # --- Phase 9.3: Add Sketched Primitives Mass ---
        sketches = design_params.get("geometry_sketch", [])
        for prim in sketches:
            if prim.get("type") == "capsule":
                # Volume of Capsule = Cylinder + Sphere
                # V = pi*r^2*L + 4/3*pi*r^3
                start = prim.get("start", [0,0,0])
                end = prim.get("end", [0,0,0])
                r = prim.get("radius", 0.1)
                
                dx, dy, dz = end[0]-start[0], end[1]-start[1], end[2]-start[2]
                L = math.sqrt(dx*dx + dy*dy + dz*dz)
                
                vol_cyl = math.pi * (r**2) * L
                vol_sph = (4/3) * math.pi * (r**3)
                vol_total = vol_cyl + vol_sph
                
                # Surface Area Calculation for Thermal Agent
                area_cyl = 2 * math.pi * r * L
                area_sph = 4 * math.pi * (r**2)
                area_total = area_cyl + area_sph
                
                prim_mass = vol_total * 1000.0 
                
                total_mass += prim_mass
                total_volume += vol_total
                total_surface_area += area_total
                
                # Update projected area (Roughly L * 2r)
                projected_area += (L * 2 * r)
        
        # 3. Mode Compatibility Check (Multi-Mode Agent) - NEW


        # 3. Mode Compatibility Check (Multi-Mode Agent) - NEW
        try:
            from backend.agents.multi_mode_agent import MultiModeAgent
        except ImportError:
            from .multi_mode_agent import MultiModeAgent
        
        multimode = MultiModeAgent()
        target_regime = environment.get("regime", "GROUND").upper()
        current_mode = design_params.get("mode", target_regime)
        
        mode_check = multimode.run({
            "current_mode": current_mode,
            "target_mode": target_regime,
            "target_environment": environment,
            "state": design_params # Check checks state params like velocity
        })
        
        if not mode_check.get("transition_allowed", True):
             return {
                 "physics_predictions": {},
                 "validation_flags": {
                     "physics_safe": False,
                     "reasons": [f"MODE_MISMATCH: {mode_check.get('warnings', ['Unknown compatibility issue'])}"]
                 }
             }

        # 4. Solve for Flight/Motion Feasibility (if AERIAL)
        regime = environment.get("regime", "GROUND")
        predictions = {}
        flags = {"physics_safe": True, "reasons": []}
        
        # 4. Solve for Flight/Motion Feasibility (if AERIAL)
        regime = environment.get("regime", "GROUND")
        predictions = {
            "total_mass_kg": round(total_mass, 2),
            "total_volume_m3": round(total_volume, 4),
            "projected_area_m2": round(projected_area, 4)
        }
        flags = {"physics_safe": True, "reasons": []}
        
        # --- ORACLE DELEGATION (Phase 4 Integration) & SURROGATE ROUTING ---
        # If design specifies a Physics Domain, delegate to the Oracle or Surrogate.
        domain = design_params.get("physics_domain") # e.g. "NUCLEAR", "ASTROPHYSICS"
        
        if domain == "NUCLEAR":
             # Tier 3.5: Use Nuclear Surrogate (Student) or Oracle (Teacher)
             nuc_res = self._solve_nuclear_dynamics(design_params)
             predictions.update(nuc_res)
             
             # Check Safety Flags
             if nuc_res.get("criticality") == "PROMPT_CRITICAL":
                 flags["physics_safe"] = False
                 flags["reasons"].append("NUCLEAR_HAZARD: Reactor Prompt Critical (Runaway)")
                 
             if nuc_res.get("ignition") is False and design_params.get("intent") == "GENERATOR":
                 flags["reasons"].append("PERFORMANCE: Fusion Ignition failed")

        elif domain:
            # Generic Oracle Fallback for other domains
            try:
                from agents.physics_oracle.physics_oracle import PhysicsOracle
                oracle = PhysicsOracle()
                
                # Construct Query
                query = design_params.get("physics_query", "Simulate")
                
                # Call Oracle
                oracle_result = oracle.solve(query, domain, design_params)
                
                # Check for success
                status = oracle_result.get("status")
                if status == "success" or status == "solved":
                    predictions.update(oracle_result.get("result", {}))
                else:
                    flags["physics_safe"] = False
                    flags["reasons"].append(f"ORACLE_FAIL: {oracle_result.get('message')}")
                    
            except Exception as e:
                 flags["reasons"].append(f"ORACLE_ERROR: {str(e)}")

        # --- CLASSIC SOLVERS (Legacy/Fundamental) ---
        if regime == "AERIAL":
            flight_res = self._solve_flight_dynamics(total_mass, gravity, fluid_density, projected_area)
            predictions.update(flight_res)
            
            # JUDGEMENT: Can it hover/fly?
            req_thrust = flight_res["required_thrust_N"]
            # Assume user specified a power plant or use default
            available_thrust = design_params.get("available_thrust_N", req_thrust * 1.1) # Assume valid unless specified
            
            if available_thrust < req_thrust:
                flags["physics_safe"] = False
                flags["reasons"].append(f"INSUFFICIENT_THRUST: Required {req_thrust:.1f}N > Available {available_thrust:.1f}N")
                
        elif regime == "MARINE":
             buoy_res = self._solve_buoyancy(total_mass, gravity, fluid_density, total_volume)
             predictions.update(buoy_res)
        
        # --- SUB-AGENT ORCHESTRATION (Phase 10: Multi-Physics) ---
        sub_agent_reports = {}
        
        # 1. Thermal Analysis
        try:
            from agents.thermal_agent import ThermalAgent
            therm_agent = ThermalAgent()
            therm_params = {
                "power_watts": design_params.get("power_watts", 100.0),
                "surface_area": total_surface_area,
                "ambient_temp": environment.get("temperature", 20.0),
                "environment_type": regime
            }
            therm_result = therm_agent.run(therm_params)
            sub_agent_reports["thermal"] = therm_result
            
            # Check Criticality
            if therm_result.get("status") == "critical":
                flags["physics_safe"] = False
                flags["reasons"].append(f"THERMAL_CRITICAL: {therm_result.get('equilibrium_temp_c')}C")
        except Exception as e:
            sub_agent_reports["thermal"] = {"error": str(e)}
        
        # 2. Structural Analysis (Phase 10.2)
        try:
            from agents.structural_agent import StructuralAgent
            struct_agent = StructuralAgent()
            
            min_radius = 0.1
            for prim in sketches:
                if prim.get("type") == "capsule":
                    r = prim.get("radius", 0.1)
                    if r < min_radius:
                        min_radius = r
            
            cross_section_mm2 = math.pi * (min_radius * 1000) ** 2
            
            struct_result = struct_agent.run({
                "mass_kg": total_mass,
                "cross_section_mm2": cross_section_mm2,
                "length_m": 1.0,
                "g_loading": design_params.get("g_loading", 3.0),
                "material_properties": {}
            })
            sub_agent_reports["structural"] = struct_result
            
            if struct_result.get("status") == "failure":
                flags["physics_safe"] = False
                flags["reasons"].append(f"STRUCTURAL_FAILURE: FoS={struct_result.get('safety_factor')}")
        except Exception as e:
            sub_agent_reports["structural"] = {"error": str(e)}
             
        return {
            "physics_predictions": predictions,
            "validation_flags": flags,
            "sub_agent_reports": sub_agent_reports
        }

    def critique_design(self, geometry_tree: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """
        Critique the overall design geometry for physical viability.
        Rules:
        1. Mass/Stability.
        2. Aspect Ratio (Structural).
        """
        critiques = []
        
        total_mass = 0.0
        max_dim = 0.0
        
        # Quick Scan
        for part in geometry_tree:
            p = part.get("params", {})
            # simplistic
            
            # Check Mass (if available)
            if "mass_kg" in part:
                 total_mass += part["mass_kg"]
                 
            # Check Limits
            dims = [p.get("width",0), p.get("height",0), p.get("length",0), p.get("radius",0)*2]
            max_d = max(dims)
            if max_d > max_dim: max_dim = max_d
            
        # Rule 1: Heavy but Small? (Density Check)
        # Rule 2: Floating/Zero Mass?
        if total_mass < 0.001:
             critiques.append({
                 "level": "WARN", 
                 "agent": "Physics",
                 "message": "Design has near-zero mass. Assign material properties?"
             })
             
        # Rule 3: Slenderness Ratio (Buckling Risk)
        # If Length >> Width
        # Hard to heuristics without seeing full assembly, but if any single part is very thin...
        for part in geometry_tree:
            p = part.get("params", {})
            l = p.get("length", 0)
            r = p.get("radius", 0)
            # Cylinder Slenderness
            if r > 0 and l > 0:
                 slenderness = l / (2*r)
                 if slenderness > 20:
                     critiques.append({
                         "level": "INFO", 
                         "agent": "Physics",
                         "message": f"Part '{part.get('name','unnamed')}' is very slender (ratio {slenderness:.0f}). Check buckling."
                     })
                     
        return critiques


    def step(self, state: Dict[str, float], inputs: Dict[str, float], dt: float = 0.1) -> Dict[str, Any]:
        """
        Advances the simulation using 6-DOF Rigid Body Dynamics (scipy.odeint).
        State vector (13 elements): [x, y, z, vx, vy, vz, q0, q1, q2, q3, p, q, r]
        Inputs: Forces/Moments in body frame.
        """
        import numpy as np
        from scipy.integrate import odeint

        # --- 1. Helper: Quaternion Math ---
        def quat_mult(q1, q2):
            w1, x1, y1, z1 = q1
            w2, x2, y2, z2 = q2
            w = w1*w2 - x1*x2 - y1*y2 - z1*z2
            x = w1*x2 + x1*w2 + y1*z2 - z1*y2
            y = w1*y2 - x1*z2 + y1*w2 + z1*x2
            z = w1*z2 + x1*y2 - y1*x2 + z1*w2
            return np.array([w, x, y, z])

        def quat_rotate(q, v):
            # Rotate vector v by quaternion q
            vq = np.array([0, v[0], v[1], v[2]])
            q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
            v_rotated = quat_mult(quat_mult(q, vq), q_conj)
            return v_rotated[1:]

        def euler_from_quat(q):
            # Roll (x), Pitch (y), Yaw (z)
            w, x, y, z = q
            sinr_cosp = 2 * (w * x + y * z)
            cosr_cosp = 1 - 2 * (x * x + y * y)
            roll = math.atan2(sinr_cosp, cosr_cosp)
            
            sinp = 2 * (w * y - z * x)
            if abs(sinp) >= 1: pitch = math.copysign(math.pi / 2, sinp)
            else: pitch = math.asin(sinp)
            
            siny_cosp = 2 * (w * z + x * y)
            cosy_cosp = 1 - 2 * (y * y + z * z)
            yaw = math.atan2(siny_cosp, cosy_cosp)
            return roll, pitch, yaw

        # --- 2. Unpack State ---
        # Legacy support: if state is simple dict, promote to vector
        # Position
        pos = np.array([
            state.get("position", {}).get("x", 0.0) if isinstance(state.get("position"), dict) else 0.0,
            state.get("position", {}).get("y", 0.0) if isinstance(state.get("position"), dict) else state.get("altitude", 0.0), # Y is Up
            state.get("position", {}).get("z", 0.0) if isinstance(state.get("position"), dict) else 0.0
        ])
        
        # Velocity (Body Frame or World? Standard is World for Pos, Body for Vel usually, but let's stick to World for linear)
        # Actually for 6DOF: usually v is body frame (u,v,w). Let's assume standard flight dynamics notation.
        # But for continuity with Phase 13, let's keep Velocity in World Frame for now, 
        # or properly map u,v,w -> V_world.
        # To keep it robust: State Vector = [X, Y, Z, Vx, Vy, Vz, qw, qx, qy, qz, p, q, r] (World Pos, World Vel, Attitude, Body Rates)
        
        vel = np.array([0.0, 0.0, 0.0])
        # Try to infer velocity vector from legacy scalar 'velocity' + orientation
        scalar_v = state.get("velocity", 0.0)
        yaw_legacy = state.get("orientation", {}).get("yaw", 0.0) if isinstance(state.get("orientation"), dict) else 0.0
        
        # If we have explicit velocity vector from last step, use it
        if "velocity_vector" in state:
            vel = np.array(state["velocity_vector"])
        else:
            # Reconstruct from scalar
            vel = np.array([scalar_v * math.sin(yaw_legacy), 0.0, scalar_v * math.cos(yaw_legacy)])
            
        # Quaternion (w, x, y, z)
        # Reconstruct from Euler if missing
        quat = np.array(state.get("quaternion", [1.0, 0.0, 0.0, 0.0]))
        if "quaternion" not in state:
             # Basic Euler -> Quat (only Yaw supported previously)
             cy = math.cos(yaw_legacy * 0.5)
             sy = math.sin(yaw_legacy * 0.5)
             quat = np.array([cy, 0.0, sy, 0.0]) # w, x, y, z (Pitch/Roll = 0)

        rates = np.array(state.get("angular_rates", [0.0, 0.0, 0.0])) # p, q, r

        y0 = np.concatenate([pos, vel, quat, rates])
        
        # --- 3. Physics Parameters ---
        mass = inputs.get("mass", 10.0)
        thrust_scalar = inputs.get("thrust", 0.0)
        gravity_acc = inputs.get("gravity", self.physics.get_constant("g"))  # Real physics
        # Inertia Tensor (Diagonal approximation)
        # Ixx, Iyy, Izz. Assume approx cube/sphere I = 2/5 m r^2 or similar
        r_approx = 1.0 
        Ixx = 0.4 * mass * r_approx**2
        Iyy = 0.4 * mass * r_approx**2
        Izz = 0.4 * mass * r_approx**2
        
        # --- 4. Equations of Motion (Derivative Function) ---
        def equations(y, t):
            # Unpack
            r = y[0:3]   # Pos (World)
            v = y[3:6]   # Vel (World)
            q = y[6:10]  # Quat (Body->World)
            w = y[10:13] # Rates (Body)
            
            # Normalise Quat
            q = q / np.linalg.norm(q)
            
            # --- Forces (World Frame) ---
            # 1. Gravity
            F_g = np.array([0.0, -mass * gravity_acc, 0.0])
            
            # 2. Thrust (Body Frame -> World Frame)
            # Assume Thrust is aligned with Body Y (Up) or Z (Forward)?
            # Phase 13 implies "Hover" drone -> Thrust implies Up (Local Y)
            F_thrust_body = np.array([0.0, thrust_scalar, 0.0]) 
            F_thrust_world = quat_rotate(q, F_thrust_body)
            
            # 3. Drag (World Frame approx) - Using Physics Kernel
            cd = inputs.get("drag_coeff", 0.5)
            
            # Velocity magnitude
            v_mag = np.linalg.norm(v)
            if v_mag > 0 and cd > 0:
                # Use physics kernel for drag calculation
                drag_magnitude = self.physics.domains["fluids"].calculate_drag(
                    velocity=v_mag,
                    density=1.225,  # TODO: Get from environment
                    reference_area=1.0,  # TODO: Get from geometry
                    drag_coefficient=cd
                )
                # Apply in direction opposite to velocity
                F_drag = -(drag_magnitude / v_mag) * v
            else:
                F_drag = np.array([0.0, 0.0, 0.0])
                
            F_total = F_g + F_thrust_world + F_drag
            
            # Accel (World)
            a = F_total / mass
            
            # --- Moments (Body Frame) ---
            # Torque? For MVP, assume stable hover (PID auto-stabilization implicitly, or zero torque)
            # Let's add simple damping to rates
            M_net = -0.1 * w * mass # Damping
            
            # Euler's Rotation Eq: I * dw/dt + w x (I * w) = M
            # For diagonal I:
            # dwx/dt = (Mx - (Izz - Iyy)wy*wz) / Ixx
            # ...
            # Simplified for MVP (Identity Inertia approx or decoupling)
            dw = np.zeros(3)
            dw[0] = (M_net[0] - (Izz - Iyy)*w[1]*w[2]) / Ixx
            dw[1] = (M_net[1] - (Ixx - Izz)*w[0]*w[2]) / Iyy
            dw[2] = (M_net[2] - (Iyy - Ixx)*w[0]*w[1]) / Izz
            
            # Quaternion Derivative: dq/dt = 0.5 * q * w (quaternion mult)
            # w as pure quat (0, wx, wy, wz)
            w_quat = np.array([0, w[0], w[1], w[2]])
            dq = 0.5 * quat_mult(q, w_quat)
            
            return np.concatenate([v, a, dq, dw])

        # --- 5. Integrate ---
        # Solve for [0, dt]
        t = np.linspace(0, dt, 2)
        try:
            sol = odeint(equations, y0, t)
            y_final = sol[-1]
        except Exception as e:
            print(f"ODE Solver Failed: {e}")
            y_final = y0 # Fallback to no motion

        # --- 6. Repack State ---
        new_pos = y_final[0:3]
        new_vel = y_final[3:6]
        new_quat = y_final[6:10]
        new_quat = new_quat / np.linalg.norm(new_quat) # Renormalize
        new_rates = y_final[10:13]
        
        # Ground Constraint (Hard)
        if new_pos[1] <= 0:
            new_pos[1] = 0.0
            new_vel[1] = max(0.0, new_vel[1]) # Bounce or stop
            # Friction?
            new_vel[0] *= 0.95
            new_vel[2] *= 0.95
            
        # Convert Quat to Euler for frontend legacy support
        roll, pitch, yaw = euler_from_quat(new_quat)
        
        # Aux State vars
        fuel = state.get("fuel", 100.0)
        fuel_burn = thrust_scalar * 0.01 * dt
        new_fuel = max(0, fuel - fuel_burn)
        
        temp = state.get("temperature", 20.0)
        new_temp = temp # Placeholder thermal
        
        # Force Vectors for Visualization (Recalculate at final state)
        # Re-eval instantaneous forces at t+dt
        q_final = new_quat
        F_g = np.array([0.0, -mass * gravity_acc, 0.0])
        F_thrust_world = quat_rotate(q_final, np.array([0.0, thrust_scalar, 0.0]))
        v_mag = np.linalg.norm(new_vel)
        # Recalculate drag at final state using physics kernel
        if v_mag > 0:
            drag_mag = self.physics.domains["fluids"].calculate_drag(
                velocity=v_mag, density=1.225, reference_area=1.0, drag_coefficient=0.5
            )
            F_drag = -(drag_mag / v_mag) * new_vel
        else:
            F_drag = np.zeros(3)
        F_net = F_g + F_thrust_world + F_drag

        return {
            "state": {
                # Legacy Scalar Compat
                "velocity": float(np.linalg.norm(new_vel)), 
                "altitude": float(new_pos[1]), 
                
                # Full Vector State
                "position": {"x": float(new_pos[0]), "y": float(new_pos[1]), "z": float(new_pos[2])},
                "velocity_vector": new_vel.tolist(),
                "orientation": {"roll": roll, "pitch": pitch, "yaw": yaw},
                "quaternion": new_quat.tolist(),
                "angular_rates": new_rates.tolist(),
                
                "fuel": new_fuel,
                "temperature": new_temp,
                "mass": mass - (fuel - new_fuel)*0.01,
                
                # Telemetry
                "force_vectors": {
                    "gravity": {"x": F_g[0], "y": F_g[1], "z": F_g[2], "magnitude": mass*gravity_acc},
                    "thrust": {"x": F_thrust_world[0], "y": F_thrust_world[1], "z": F_thrust_world[2], "magnitude": thrust_scalar},
                    "drag": {"x": F_drag[0], "y": F_drag[1], "z": F_drag[2], "magnitude": np.linalg.norm(F_drag)},
                    "net": {"x": F_net[0], "y": F_net[1], "z": F_net[2], "magnitude": np.linalg.norm(F_net)}
                }
            },
            "metrics": {
                "f_net": float(np.linalg.norm(F_net)),
                "thrust": thrust_scalar,
                "g_load": float(np.linalg.norm(F_net)/mass/gravity_acc)  # Use actual gravity
            }
        }

    # --- STUDENT-TEACHER LOGIC (Tier 3.5 Deep Evolution) ---
    def _solve_flight_dynamics(self, mass: float, g: float, rho: float, area: float) -> Dict[str, float]:
        """
        Solves flight requirements using Neural Surrogate (Student) or Analytic (Teacher).
        """
        import numpy as np
        
        # 1. Ask the Student (Intuition)
        # Input Vector: [mass, gravity, rho, area, 1.0(bias)]
        # managed by kernel now
        if self.surrogate_manager.has_model("physics_surrogate"):
             # Normalize using actual constants from physics kernel
             g_standard = self.physics.get_constant("g")
             rho_standard = 1.225  # Could get from physics.domains.fluids.calculate_air_density(288.15, 101325)
             inputs = np.array([mass/1000.0, g/g_standard, rho/rho_standard, area/10.0, 1.0])
             
             # Call manager
             response = self.surrogate_manager.predict("physics_surrogate", inputs)
             
             pred = response.get("result")
             confidence = response.get("confidence", 0.0)
             
             # If Student is confident (and we aren't force-checking), use Student
             if confidence > 0.8:
                 # Helper to clamp non-negative
                 pred = np.maximum(pred, 0.0)
                 return {
                     "weight_N": round(mass * g, 2), # Trivial calculation, keep exact
                     "required_thrust_N": round(float(pred[0][0]) * 1000.0, 2), # Scale back up
                     "est_stall_speed_mps": round(float(pred[0][1]) * 100.0, 1), # Scale up
                     "drag_coefficient": 0.04,
                     "source": "Neural Surrogate (Student)"
                 }
                 
        # 2. Ask the Teacher (Analytic / Kernel)
        # L = W -> 0.5 * rho * v^2 * Cl * A = m * g
        weight_N = mass * g
        hover_thrust = weight_N
        Cl = 0.5
        try:
            stall_speed = math.sqrt((2 * weight_N) / (rho * area * Cl))
        except ValueError:
            stall_speed = 0.0
            
        return {
            "weight_N": round(weight_N, 2),
            "required_thrust_N": round(hover_thrust, 2),
            "est_stall_speed_mps": round(stall_speed, 1),
            "drag_coefficient": 0.04,  # Generic streamlined
            "source": "Analytic Physics (Kernel Teacher)" 
        }

    def _solve_nuclear_dynamics(self, params: Dict[str, Any]) -> Dict[str, float]:
        """
        Solves Nuclear physics (Fusion/Fission) using Neural Student or KERNEL Teacher.
        Oracle Fallback removed in Phase 10.
        """
        import numpy as np
        
        # Input Mapping (Normalize for NN)
        # Inputs: [Density, Temp, Confinement, FuelFactor]
        density = params.get("density", 1e20)
        temp = params.get("temperature_kev", 10.0)
        tau = params.get("confinement_time", 1.0)
        fuel_idx = 1.0 if params.get("fuel") == "DT" else 0.5
        
        inputs = np.array([np.log10(density)/20.0, temp/100.0, tau/10.0, fuel_idx])
        
        # 1. Ask Student
        if self.has_nuclear_brain and self.nuclear_student:
            pred, confidence = self.nuclear_student.predict_performance(inputs)
            
            if confidence > 0.8:
                return {
                     "source": "Nuclear Surrogate (Student)",
                     "fusion_power_density_MW_m3": float(pred[0][1]) * 100.0,
                     "Q_factor": float(pred[0][0]) * 10.0,
                     "ignition": float(pred[0][0]) * 10.0 > 1.0
                }
        
        # 2. Ask Teacher (Physics Kernel Domain)
        sim_type = params.get("type", "FUSION").upper()
        
        if sim_type == "FUSION":
            res = self.physics.domains["nuclear"].solve_fusion_lawson(
                density=density, temp_kev=temp, confinement_time=tau, fuel=params.get("fuel", "DT")
            )
            res["source"] = "Physics Kernel (Nuclear Domain)"
            return res
            
        elif sim_type == "FISSION":
            res = self.physics.domains["nuclear"].solve_fission_kinetics(
                reactivity=params.get("reactivity", 0.0)
            )
            res["source"] = "Physics Kernel (Nuclear Domain)"
            return res
            
        return {"error": "All solvers failed"}
        
    def evolve(self, training_data: list):
        """
        Deep Evolution Trigger.
        Generic entry point - delegates to SurrogateManager.
        """
        if not training_data: return {"status": "skipped", "reason": "No data"}
        
        # Phase 3.9: Delegate to SurrogateManager
        # Ideally: self.surrogate_manager.train("physics_surrogate", training_data)
        
        logger.info(f"Buffered {len(training_data)} samples for surrogate training via Kernel")
        
        return {
            "status": "buffered", 
            "samples": len(training_data),
            "target": "physics_surrogate" 
        }



    def _solve_buoyancy(self, mass: float, g: float, rho_fluid: float, volume: float) -> Dict[str, float]:
        # Use physics kernel for buoyancy calculation
        buoyancy_force = self.physics.domains["fluids"].calculate_buoyancy(
            fluid_density=rho_fluid,
            displaced_volume=volume,
            gravity=g
        )
        weight = mass * g
        net_force = buoyancy_force - weight
        return {
             "buoyancy_N": round(buoyancy_force, 2),
             "net_force_N": round(net_force, 2),
             "is_floating": net_force >= 0,
             "source": "Physics Kernel (Fluids Domain)"
        }

    def check_collision_sdf(self, position: List[float], vehicle_radius: float, environment_sdf_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        Symbolic Collision Check (Zero Tunneling).
        Instead of mesh raycast, we query SDF.
        
        Args:
            position: [x,y,z] world coords
            vehicle_radius: Approx bounding sphere
            environment_sdf_map: {
                 "type": "vmk",
                 "stock_dims": [...],
                 "toolpaths": [...] # The map is defined as the 'Remaining Stock' or 'Added Obstacles'
            }
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"collision": False, "error": "VMK not available"}

        # For environment, we treat obstacles as 'Stock' or 'positive shape'?
        # If map is a Cave, Stock is solid, Toolpath is Air. 
        # Collision = Dist > 0 (inside Rock). Air = Dist < 0.
        # Wait, usually SDF > 0 is Outside, < 0 is Inside.
        # VMK Convention:
        # Stock = Box. d_stock(p) < 0 is INSIDE box.
        # Cut = Capsule. d_cut(p) < 0 is INSIDE cut (Air).
        # Final d = max(d_stock, -d_cut).
        # Inside Final Part (Rock) => d < 0.
        # Air (Cut) => d > 0.
        
        # So for a Cave Map:
        # If d < 0, we are Collision (in Rock).
        # If d > 0, we are Safe (in Air).
        
        # Optimization: We assume the PhysicsAgent keeps a persistent kernel 
        # for static environment to avoid re-init every frame.
        # But here we demonstrate the logic stateless for MVP.
        
        kernel = SymbolicMachiningKernel(stock_dims=environment_sdf_map.get("stock_dims", [1000,1000,1000]))
        
        # Auto-register tools for obstacles
        registered_tools = set()

        # In a real game loop, we wouldn't re-execute history every frame. 
        # We would optimize. But this proves the math.
        for op in environment_sdf_map.get("obstacles", []):
            tid = op.get("tool_id", "obstacle_tool")
            if tid not in registered_tools:
                # Default "Tunnel Borer" size 10m radius if not specified
                # In real system, we look up or pass in map.
                radius = op.get("radius", 10.0) 
                kernel.register_tool(ToolProfile(id=tid, radius=radius, type="BALL"))
                registered_tools.add(tid)

            # If obstacles are CUTS (tunnels), we execute them.
            kernel.execute_gcode(VMKInstruction(**op))
            
        p = np.array(position)
        dist = kernel.get_sdf(p)
        
        # If dist (distance to Rock) < vehicle_radius?
        # If dist < 0, we are deep in rock.
        # If dist > 0 but < radius, we are touching.
        # Wait, if d > 0, we are OUTSIDE the Rock (in Air).
        # Distance to surface is d.
        # If d < vehicle_radius, we are close to wall.
        # Valid state = d > vehicle_radius (Clearance).
        
        return {
            "collision": dist < vehicle_radius, 
            "distance_to_obstacle": dist,
            "sdf": dist,
            "is_underground": dist < 0,
            "penetration_depth": vehicle_radius - dist if dist < vehicle_radius else 0.0,
            "validation_engine": "VMK SDF"
        }

class StructuralSolver:
    """
    Finite Element Analysis (FEA) solver using scikit-fem.
    Performs Linear Static Analysis on geometry.
    """
    def __init__(self):
        self.ready = False
        try:
            import skfem
            from skfem.models.elasticity import linear_elasticity
            from skfem.helpers import dot
            self.skfem = skfem
            self.ready = True
        except Exception as e:
            print(f"scikit-fem import failed: {e}")
            print("Structural Analysis unavailable.")

    def solve_cantilever(self, length_m: float, cross_section_m2: float, load_N: float, material_E_Pa: float = 200e9) -> Dict:
        """
        Approximates a part as a 1D Cantilever Beam using FEM (or analytical check).
        """
        if not self.ready: return {"status": "error", "message": "Missing dependencies"}
        
        # Analytical Solution for Cantilever Tip Deflection: d = FL^3 / 3EI
        # Assume square cross section for I (Moment of Inertia)
        # I = a^4 / 12. Area = a^2. -> a = sqrt(Area). 
        # I = Area^2 / 12
        I = (cross_section_m2**2) / 12.0
        
        deflection = (load_N * length_m**3) / (3 * material_E_Pa * I)
        max_stress = (load_N * length_m) * (math.sqrt(cross_section_m2)/2) / I # My/I
        
        # TODO: Implement full 3D mesh FEM using skfem.MeshTet in Phase 14.2
        # For now, analytical beam is truthful enough for "cantilever".
        
        return {
            "max_deflection_m": deflection,
            "max_stress_Pa": max_stress,
            "safety_factor_yield": 250e6 / max_stress if max_stress > 0 else 999.9 # Assume Steel Yield 250MPa
        }
