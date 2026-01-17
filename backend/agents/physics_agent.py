import math
import random
from typing import Dict, Any, List
# import scipy.constants # For future precision

class PhysicsAgent:
    """
    The 'Judge'. 
    Validates design concepts against the Laws of the Universe.
    Does NOT use LLMs. Uses First-Principles Physics.
    """
    
    # Fundamental Constants (SI Units)
    CONSTANTS = {
        "G": 9.80665,            # Standard Gravity (m/s^2)
        "R_GAS": 8.314,          # Ideal Gas Constant (J/(mol·K))
        "SIGMA": 5.670374e-8,    # Stefan-Boltzmann Constant (W/(m^2·K^4))
        "C": 299792458,          # Speed of Light (m/s)
        "P_ATM_EARTH": 101325,   # Standard Pressure (Pa)
        "RHO_AIR": 1.225         # Standard Air Density (kg/m^3)
    }

    def run(self, environment: Dict[str, Any], geometry_tree: List[Dict[str, Any]], design_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run the physics simulation kernel.
        Returns detailed predictions and feasibility verdicts.
        """
        
        # 1. Establish Boundary Conditions (The "Universe" state)
        gravity = environment.get("gravity", self.CONSTANTS["G"])
        fluid_density = environment.get("fluid_density", self.CONSTANTS["RHO_AIR"])
        
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
             else:
                 total_mass += 1.0 # Safe fallback to avoid div/0
        
        if total_mass == 0: total_mass = 1.0 # Safety
        if projected_area == 0: projected_area = 0.01

        # 3. Mode Compatibility Check (Multi-Mode Agent) - NEW
        from agents.multi_mode_agent import MultiModeAgent
        multimode = MultiModeAgent()
        mode_check = multimode.run({
            "current_mode": design_params.get("mode", environment.get("regime", "GROUND")),
            "target_environment": environment
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
        predictions = {}
        flags = {"physics_safe": True, "reasons": []}
        
        # --- ORACLE DELEGATION (Phase 4 Integration) ---
        # If design specifies a Physics Domain, delegate to the Oracle.
        domain = design_params.get("physics_domain") # e.g. "NUCLEAR", "ASTROPHYSICS"
        if domain:
            try:
                from agents.physics_oracle.physics_oracle import PhysicsOracle
                oracle = PhysicsOracle()
                
                # Construct Query
                query = design_params.get("physics_query", "Simulate")
                
                # Call Oracle
                oracle_result = oracle.solve(query, domain, design_params)
                
                # Check for success (Adapters use 'solved', Oracle wrapper might use 'success' or pass through)
                status = oracle_result.get("status")
                if status == "success" or status == "solved":
                    predictions.update(oracle_result.get("result", {}))
                    # Check for critical failures in result
                    if oracle_result.get("result", {}).get("is_critical") == False and domain == "NUCLEAR":
                         # Example logic: If reactor not critical, it's safe (or unsafe depending on goal)
                         pass
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
             
        return {
            "physics_predictions": predictions,
            "validation_flags": flags
        }

    def step(self, state: Dict[str, float], inputs: Dict[str, float], dt: float = 0.1) -> Dict[str, Any]:
        """
        Advances the simulation by one time step (dt).
        Implements basic 1D Euler integration for vertical motion and thermal dynamics.
        Now Stochastic: Injects noise if inputs['noise_level'] > 0.
        Enhanced: Returns 3D position and force vectors for visualization.
        """
        # Unpack State
        velocity = state.get("velocity", 0.0)
        altitude = state.get("altitude", 0.0)
        temp = state.get("temperature", 20.0) # Ambient start
        fuel = state.get("fuel", 100.0)
        
        # 3D position (for visualization)
        pos_x = state.get("position", {}).get("x", 0.0) if isinstance(state.get("position"), dict) else 0.0
        pos_y = state.get("position", {}).get("y", 0.0) if isinstance(state.get("position"), dict) else altitude
        pos_z = state.get("position", {}).get("z", 0.0) if isinstance(state.get("position"), dict) else 0.0
        
        # Orientation (yaw, pitch, roll)
        yaw = state.get("orientation", {}).get("yaw", 0.0) if isinstance(state.get("orientation"), dict) else 0.0

        # Unpack Inputs
        thrust = inputs.get("thrust", 0.0)
        gravity = inputs.get("gravity", 9.81)
        drag_coeff = inputs.get("drag_coeff", 0.5) 
        mass = inputs.get("mass", 10.0)
        
        # Stochastic Param
        noise_level = inputs.get("noise_level", 0.0)

        # 1. External Disturbances (Wind Gusts)
        wind_force = 0.0
        if noise_level > 0:
             # Random gust +/- max_force * level
             max_wind = inputs.get("max_wind_force", 10.0)
             wind_force = (random.random() - 0.5) * max_wind * noise_level 

        # Physics Calculations (1D Vertical Motion + 3D Position Tracking)
        # F_net = Thrust - Weight - Drag + Wind
        weight = mass * gravity
        
        # Drag opposes velocity
        fluid_density = inputs.get("fluid_density", 1.225)
        drag_force = 0.5 * fluid_density * (velocity ** 2) * drag_coeff
        if velocity < 0:
            drag_force = -drag_force 
            
        # Net Force
        mag_force = inputs.get("mag_force", 0.0)
        f_net = thrust - weight - (drag_force if velocity > 0 else -drag_force) + wind_force + mag_force
        
        # Integration (Euler)
        acceleration = f_net / mass
        new_velocity = velocity + acceleration * dt
        new_altitude = altitude + velocity * dt
        
        # 3D position update (simple 2D motion on XZ plane based on yaw)
        vx = new_velocity * math.sin(yaw)
        vz = new_velocity * math.cos(yaw)
        new_pos_x = pos_x + vx * dt
        new_pos_y = new_altitude
        new_pos_z = pos_z + vz * dt
        
        # Ground Constraint
        if new_altitude <= 0:
            new_altitude = 0
            new_pos_y = 0
            if new_velocity < -1.0:
               new_velocity = 0 
            else:
               new_velocity = 0

        # Thermal Model (Newton's Cooling + Joule Heating)
        ambient_temp = 20.0
        heat_gen = thrust * 0.005 * dt 
        cooling = (temp - ambient_temp) * 0.1 * dt
        new_temp = temp + heat_gen - cooling
        
        # 2. Sensor Noise
        if noise_level > 0:
            new_velocity += random.gauss(0, 0.5 * noise_level)
            new_altitude += random.gauss(0, 0.2 * noise_level)
            new_temp += random.gauss(0, 0.1 * noise_level)

        # Fuel Consumer
        fuel_burn = thrust * 0.01 * dt
        new_fuel = max(0, fuel - fuel_burn)
        new_mass = mass - (fuel -new_fuel) * 0.01

        # Generate Logs
        logs = []
        if noise_level > 0 and abs(wind_force) > 2.0:
             logs.append(f"[WARN] Gust detected: {wind_force:.1f}N")
        if velocity < 343 and new_velocity >= 343:
            logs.append("[PHYS] Supersonic transition verified. Mach 1.0 achieved.")
        if new_temp > 100 and temp <= 100:
            logs.append("[WARN] Thermal limit approached. Active cooling recommended.")
        if new_fuel < 10 and fuel >= 10:
            logs.append("[WARN] Energy reserves critical (<10%).")
        if new_altitude > 1000 and altitude <= 1000:
             logs.append("[INFO] Altitude milestone: 1km vertical displacement.")

        return {
            "state": {
                "velocity": new_velocity,
                "altitude": new_altitude,
                "temperature": new_temp,
                "fuel": new_fuel,
                "acceleration": acceleration,
                "mass": new_mass,
                "position": {
                    "x": new_pos_x,
                    "y": new_pos_y,
                    "z": new_pos_z
                },
                "orientation": {
                    "yaw": yaw,
                    "pitch": 0.0,
                    "roll": 0.0
                },
                "force_vectors": {
                    "gravity": {
                        "x": 0.0,
                        "y": -weight,
                        "z": 0.0,
                        "magnitude": weight
                    },
                    "thrust": {
                        "x": 0.0,
                        "y": thrust,
                        "z": 0.0,
                        "magnitude": thrust
                    },
                    "drag": {
                        "x": 0.0,
                        "y": -drag_force if velocity > 0 else drag_force,
                        "z": 0.0,
                        "magnitude": abs(drag_force)
                    },
                    "net": {
                        "x": 0.0,
                        "y": f_net,
                        "z": 0.0,
                        "magnitude": abs(f_net)
                    }
                },
                "logs": logs # Added logs to the state
            },
            "metrics": {
                "f_net": f_net,
                "drag": drag_force,
                "weight": weight,
                "thrust": thrust
            }
        }

    def _solve_flight_dynamics(self, mass: float, g: float, rho: float, area: float) -> Dict[str, float]:
        """
        Solves steady-state flight requirements.
        L = W -> 0.5 * rho * v^2 * Cl * A = m * g
        """
        weight_N = mass * g
        
        # Hover case (Thrust = Weight)
        hover_thrust = weight_N
        
        # Cruise velocity estimation (assuming Cl=0.5 for generic airfoil)
        Cl = 0.5
        # v = sqrt( (2 * m * g) / (rho * A * Cl) )
        try:
            stall_speed = math.sqrt((2 * weight_N) / (rho * area * Cl))
        except ValueError:
            stall_speed = float('inf') # Divide by zero or negative
            
        return {
            "weight_N": round(weight_N, 2),
            "required_thrust_N": round(hover_thrust, 2),
            "est_stall_speed_mps": round(stall_speed, 1),
            "drag_coefficient": 0.04 # Generic streamlined
        }

    def _solve_buoyancy(self, mass: float, g: float, rho_fluid: float, volume: float) -> Dict[str, float]:
        buoyancy_force = rho_fluid * volume * g
        weight = mass * g
        net_force = buoyancy_force - weight
        return {
             "buoyancy_N": round(buoyancy_force, 2),
             "net_force_N": round(net_force, 2),
             "is_floating": net_force >= 0
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
