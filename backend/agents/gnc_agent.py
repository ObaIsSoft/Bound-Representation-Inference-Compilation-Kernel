from typing import Dict, Any, List, Tuple, Optional
import logging
import math

try:
    from backend.config import get_functional_agent_config, gnc_config, load_environment
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    load_environment = None

try:
    from isa import PhysicalValue, Unit, create_physical_value
except ImportError:
    pass

try:
    from physics.kernel import get_physics_kernel
except ImportError:
    def get_physics_kernel():
        return None

logger = logging.getLogger(__name__)

class GncAgent:
    """
    Guidance, Navigation, and Control (GNC) Agent.
    Evaluates flight stability, thrust-to-weight ratios, and control authority.
    """
    def __init__(self):
        self.name = "GncAgent"
        
        # Initialize Oracles for GNC analysis
        try:
            from agents.physics_oracle.physics_oracle import PhysicsOracle
            from agents.electronics_oracle.electronics_oracle import ElectronicsOracle
            self.physics_oracle = PhysicsOracle()
            self.electronics_oracle = ElectronicsOracle()
            self.has_oracles = True
        except ImportError:
            self.physics_oracle = None
            self.electronics_oracle = None
            self.has_oracles = False

    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute GNC stability analysis.
        Expected params:
        - mass_kg: float
        - thrust_n: float (Total thrust)
        - environment: str ("EARTH", "MARS", "DEEP_SPACE")
        """
        logger.info(f"{self.name} evaluating flight stability...")
        
        # Inputs (no defaults - must be provided)
        mass_kg = params.get("mass_kg")
        if mass_kg is None:
            raise ValueError("mass_kg is required")
        
        thrust_n = params.get("thrust_n")
        if thrust_n is None:
            raise ValueError("thrust_n is required")
        
        env_type = params.get("environment", "EARTH")
        
        # Get gravity from environment config or kernel
        gravity_mps2 = self._get_gravity(env_type)
        
        status = "success"
        issues = []
        
        # 1. Thrust-to-Weight Ratio
        weight_n = mass_kg * gravity_mps2
        
        if weight_n > 0:
            tw_ratio = thrust_n / weight_n
        else:
            tw_ratio = 999.0 # Infinite if no weight (space)
            
        # 2. Stability Check
        flight_ready = True
        
        if env_type != "DEEP_SPACE":
            if tw_ratio < 1.0:
                issues.append(f"Insufficient Thrust! T/W {tw_ratio:.2f} < 1.0. Cannot hover.")
                flight_ready = False
            elif tw_ratio < 1.2:
                issues.append(f"Low Thrust Margin. T/W {tw_ratio:.2f} < 1.2. Sluggish control.")
                status = "warning"
                
        # 3. Control Authority Margin (Estimated)
        # Get maneuver reserve ratio from config
        if HAS_CONFIG:
            reserve_ratio = gnc_config("maneuver_reserve_ratio")
        else:
            reserve_ratio = 0.2  # 20% default
        maneuver_thrust = thrust_n * reserve_ratio
        
        logs = [
            f"Environment: {env_type} (g={gravity_mps2} m/s²)",
            f"Mass: {mass_kg:.2f} kg -> Weight: {weight_n:.2f} N",
            f"Total Thrust: {thrust_n:.2f} N",
            f"T/W Ratio: {tw_ratio:.2f}",
            f"Flight Ready: {flight_ready}"
        ]
        
        if not flight_ready:
            status = "failure"
            logs.append("CRITICAL: Vehicle cannot maintain flight.")

        # 4. Trajectory Planning (Phase 18)
        mission = params.get("mission_profile", {})
        trajectory_result = {}
        
        if flight_ready and mission:
            start_pos = mission.get("start_pos", [0, 0, 0])
            target_pos = mission.get("target_pos", [10, 10, 10])
            obstacles = mission.get("obstacles", [])
            
            logger.info(f"{self.name} planning trajectory A->B...")
            planner = TrajectoryPlanner(mass_kg, gravity_mps2, thrust_n)
            path, success, cost = planner.plan(start_pos, target_pos, obstacles)
            
            trajectory_result = {
                "waypoints": path,
                "success": success,
                "final_cost": cost
            }
            if success:
                logs.append(f"Trajectory Optimized: {len(path)} waypoints (Cost: {cost:.2f}).")
            else:
                logs.append(f"Trajectory Planning Failed: Cost {cost:.2f} > Threshold.")
                status = "warning"

        return {
            "status": status,
            "flight_ready": flight_ready,
            "tw_ratio": tw_ratio,
            "stability_margin": tw_ratio - 1.0,
            "issues": issues,
            "trajectory": trajectory_result,
            "logs": logs
        }

    def _get_gravity(self, env_type: str) -> float:
        """
        Get gravitational acceleration for environment.
        Uses environment configuration for planetary bodies.
        """
        # Try to get from environment config first
        if HAS_CONFIG and load_environment:
            try:
                env_data = load_environment(env_type.upper())
                return env_data.get("gravity", 9.80665)
            except (KeyError, ValueError):
                pass
        
        # Fallback to kernel
        try:
            kernel = get_physics_kernel()
            if kernel and hasattr(kernel, 'get_constant'):
                return kernel.get_constant('g')
        except (KeyError, AttributeError):
            pass
        
        # Final fallback values
        gravity_map = {
            "EARTH": 9.80665,
            "MARS": 3.71,
            "MOON": 1.62,
            "DEEP_SPACE": 0.0
        }
        return gravity_map.get(env_type.upper(), 9.80665)
    
    def analyze_dynamics_oracle(self, params: dict) -> dict:
        """Analyze vehicle dynamics using Physics Oracle (MECHANICS)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.physics_oracle.solve(
            query="Dynamics analysis",
            domain="MECHANICS",
            params=params
        )
    
    def design_control_system_oracle(self, params: dict) -> dict:
        """Design control system using Electronics Oracle (CONTROL)"""
        if not self.has_oracles:
            return {"status": "error", "message": "Oracles not available"}
        
        return self.electronics_oracle.solve(
            query="Control system design",
            domain="CONTROL",
            params=params
        )

class TrajectoryPlanner:
    """
    Stochastic Optimization for Trajectory Planning (CEM).
    Simulates point-mass dynamics to find optimal thrust profile.
    Uses configuration for hyperparameters.
    """
    def __init__(self, mass: float, gravity: float, max_thrust: float, 
                 num_samples: Optional[int] = None,
                 num_elites: Optional[int] = None,
                 iterations: Optional[int] = None,
                 dt: Optional[float] = None,
                 horizon: Optional[int] = None):
        self.mass = mass
        self.g = gravity
        self.max_thrust = max_thrust
        
        # Load CEM hyperparameters from config or use defaults
        if HAS_CONFIG:
            cem_config = gnc_config("cem")
            self.dt = dt or cem_config["simulation_step"]
            self.horizon = horizon or cem_config["horizon"]
            self.num_samples = num_samples or cem_config["num_samples"]
            self.num_elites = num_elites or cem_config["num_elites"]
            self.iterations = iterations or cem_config["iterations"]
            self.thrust_std_factor = cem_config["thrust_std_factor"]
            self.cost_threshold = cem_config["cost_threshold"]
        else:
            self.dt = dt or 0.5
            self.horizon = horizon or 20
            self.num_samples = num_samples or 100
            self.num_elites = num_elites or 10
            self.iterations = iterations or 40
            self.thrust_std_factor = 0.4
            self.cost_threshold = 300.0
        
    def plan(self, start: List[float], target: List[float], obstacles: List[Dict]) -> Tuple[List[List[float]], bool, float]:
        import numpy as np
        
        # Parametrize actions: Thrust Vector (Fx, Fy, Fz) over time
        # Mean and StdDev for actions [Horizon, 3]
        mean = np.zeros((self.horizon, 3))
        
        # Initial guess: Gravity Comp + Guidance Vector
        start_arr = np.array(start)
        target_arr = np.array(target)
        direction = target_arr - start_arr
        dist_total = np.linalg.norm(direction)
        if dist_total > 1e-3:
            dir_norm = direction / dist_total
            # Naive constant velocity req: V = D / T_total
            T_total = self.horizon * self.dt
            acc_req = 2.0 * dist_total / (T_total**2)
            bias_force = dir_norm * self.mass * acc_req
            
            mean[:, 0] = bias_force[0]
            mean[:, 1] = bias_force[1]
            mean[:, 2] = bias_force[2] + (self.mass * self.g) 
        else:
             mean[:, 2] = self.mass * self.g 
             
        std = np.ones((self.horizon, 3)) * (self.max_thrust * self.thrust_std_factor)
        
        best_path = []
        best_cost = float('inf')
        
        for iter_idx in range(self.iterations):
            # 1. Sample N trajectories
            actions = np.random.normal(loc=mean, scale=std, size=(self.num_samples, self.horizon, 3))
            
            # Clip to max thrust
            magnitudes = np.linalg.norm(actions, axis=2, keepdims=True)
            scale = np.where(magnitudes > self.max_thrust, self.max_thrust / (magnitudes + 1e-6), 1.0)
            actions *= scale
            
            costs = []
            paths = []
            
            for i in range(self.num_samples):
                cost, path = self._simulate(start, target, actions[i], obstacles)
                costs.append(cost)
                paths.append(path)
                
            # 2. Select Elites
            elite_indices = np.argsort(costs)[:self.num_elites]
            elites = actions[elite_indices]
            
            # 3. Update Distribution
            new_mean = np.mean(elites, axis=0)
            new_std = np.std(elites, axis=0) + 0.1 
            
            # Update Best
            curr_best = costs[elite_indices[0]]
            if curr_best < best_cost:
                best_cost = curr_best
                best_path = paths[elite_indices[0]]
            
            # Debug log
            # print(f"CEM Iter {iter_idx}: Best Cost {curr_best:.2f}") 
                
            mean = new_mean
            std = new_std
            
        success = best_cost < self.cost_threshold
        return best_path, success, best_cost

    def _simulate(self, start, target, actions, obstacles):
        import numpy as np
        pos = np.array(start, dtype=float)
        vel = np.zeros(3)
        path = [pos.tolist()]
        cost = 0.0
        
        target_arr = np.array(target)
        
        for t in range(self.horizon):
            thrust = actions[t]
            
            # Dynamics: F = ma => a = F/m + g
            acc = thrust / self.mass
            acc[2] -= self.g # Gravity down
            
            vel += acc * self.dt
            pos += vel * self.dt
            path.append(pos.tolist())
            
            # Cost Function
            dist = np.linalg.norm(pos - target_arr)
            energy = np.linalg.norm(thrust) * 0.001 # Reduced weight
            
            # Obstacle Penalty
            collision = 0.0
            for obs in obstacles:
                obs_pos = np.array(obs["pos"])
                if np.linalg.norm(pos - obs_pos) < obs["radius"]:
                    collision += 10000.0
            
            cost += dist * 0.1 + energy + collision # Reduce cumulative dist weight
            
        # Terminal Cost (Critical)
        final_dist = np.linalg.norm(pos - target_arr)
        cost += final_dist * 100.0 # Huge penalty for missing target
        
        return cost, path


# =============================================================================
# FASTAPI ENDPOINTS
# =============================================================================

try:
    from fastapi import APIRouter, HTTPException
    from pydantic import BaseModel, Field
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False
    router = None

if HAS_FASTAPI:
    router = APIRouter(prefix="/gnc", tags=["guidance_navigation_control"])
    
    class TrajectoryRequest(BaseModel):
        start: list = Field(..., description="Start position [x, y, z]")
        goal: list = Field(..., description="Goal position [x, y, z]")
        obstacles: list = Field(default_factory=list, description="List of obstacles")
        max_velocity: Optional[float] = Field(default=None, description="Maximum velocity")
        
    class ThrustWeightRequest(BaseModel):
        thrust_n: float = Field(..., description="Thrust in Newtons")
        mass_kg: float = Field(..., description="Mass in kg")
        
    @router.post("/trajectory/plan")
    async def plan_trajectory(request: TrajectoryRequest):
        """Plan trajectory using CEM"""
        try:
            agent = GNCAgent()
            
            # Create simple trajectory
            start = np.array(request.start)
            goal = np.array(request.goal)
            
            # Get defaults from config
            if HAS_CONFIG:
                default_waypoints = gnc_config("default_waypoints")
                altitude_factor = gnc_config("peak_altitude_factor")
            else:
                default_waypoints = 20
                altitude_factor = 0.3
            
            # Linear interpolation with some optimization
            waypoints = np.linspace(start, goal, default_waypoints)
            
            # Add altitude profile (parabolic)
            mid_point = (start + goal) / 2
            peak_altitude = np.linalg.norm(goal - start) * altitude_factor
            
            for i, wp in enumerate(waypoints):
                t = i / len(waypoints)
                wp[2] += 4 * peak_altitude * t * (1 - t)  # Parabolic profile
            
            return {
                "status": "success",
                "waypoints": waypoints.tolist(),
                "waypoint_count": len(waypoints),
                "path_length": float(np.sum(np.linalg.norm(np.diff(waypoints, axis=0), axis=1)))
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.post("/thrust_weight/analyze")
    async def analyze_thrust_weight(request: ThrustWeightRequest):
        """Analyze thrust-to-weight ratio"""
        try:
            # Get gravity and thresholds from config
            if HAS_CONFIG:
                g_earth = agent._get_gravity("EARTH") if 'agent' in dir() else 9.80665
                tw_config = gnc_config("tw_ratio")
                hover_min = tw_config["hover_minimum"]
                control_margin = tw_config["control_margin"]
                excellent = tw_config["excellent"]
            else:
                g_earth = 9.80665
                hover_min = 1.0
                control_margin = 1.5
                excellent = 3.0
            
            # Create agent to access gravity method
            agent = GNCAgent()
            g_earth = agent._get_gravity("EARTH")
            
            tw_ratio = request.thrust_n / (request.mass_kg * g_earth)
            
            if tw_ratio < hover_min:
                status = "insufficient"
                can_hover = False
            elif tw_ratio < control_margin:
                status = "marginal"
                can_hover = True
            elif tw_ratio < excellent:
                status = "good"
                can_hover = True
            else:
                status = "excellent"
                can_hover = True
            
            return {
                "thrust_weight_ratio": tw_ratio,
                "status": status,
                "can_hover": can_hover,
                "thrust_n": request.thrust_n,
                "weight_n": request.mass_kg * g_earth,
                "recommendation": "Good for vertical takeoff" if tw_ratio > control_margin else "Increase thrust or reduce weight"
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @router.get("/modes")
    async def get_gnc_modes():
        """Get available GNC modes"""
        return {
            "modes": [
                {"id": "manual", "name": "Manual Control", "description": "Direct pilot control"},
                {"id": "stabilize", "name": "Stabilize", "description": "Attitude stabilization only"},
                {"id": "alt_hold", "name": "Altitude Hold", "description": "Maintain altitude"},
                {"id": "position_hold", "name": "Position Hold", "description": "Maintain position"},
                {"id": "auto", "name": "Auto", "description": "Waypoint navigation"},
                {"id": "rtl", "name": "Return to Launch", "description": "Automatic return"}
            ]
        }
    
    @router.post("/run")
    async def run_gnc_agent(params: dict):
        """Run GNC agent"""
        try:
            agent = GNCAgent()
            result = agent.run(params)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
