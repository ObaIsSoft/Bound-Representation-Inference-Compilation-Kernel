from typing import Dict, Any, List, Tuple
import logging
import math
from isa import PhysicalValue, Unit, create_physical_value

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
        
        # Inputs
        mass_kg = params.get("mass_kg", 1.0)
        thrust_n = params.get("thrust_n", 0.0)
        env_type = params.get("environment", "EARTH")
        
        # Gravity map
        gravity_mps2 = {
            "EARTH": 9.81,
            "MARS": 3.71,
            "MOON": 1.62,
            "DEEP_SPACE": 0.0
        }.get(env_type, 9.81)
        
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
        # Assume 20% of thrust is reserved for maneuvering
        maneuver_thrust = thrust_n * 0.2
        
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
    """
    def __init__(self, mass: float, gravity: float, max_thrust: float):
        self.mass = mass
        self.g = gravity
        self.max_thrust = max_thrust
        self.dt = 0.5 # Simulation step
        self.horizon = 20 # Steps to look ahead
        
        # CEM Hyperparams
        self.num_samples = 100
        self.num_elites = 10
        self.iterations = 40
        
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
             
        std = np.ones((self.horizon, 3)) * (self.max_thrust * 0.4)
        
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
            
        success = best_cost < 300.0 # Threshold (Approx 2-3m error allowed)
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
