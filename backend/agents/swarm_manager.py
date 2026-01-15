from typing import Dict, Any, List, Type
import logging
from agents.environment_agent import EnvironmentAgent
from agents.von_neumann_agent import VonNeumannAgent
from agents.construction_agent import ConstructionAgent

logger = logging.getLogger(__name__)

class SwarmManager:
    """
    Orchestrates the Swarm Simulation Loop.
    Manages Agent Registry, Environment State, and Time-Stepping.
    """
    def __init__(self):
        self.env_agent = EnvironmentAgent()
        self.agents: List[Any] = []
        self.resources: List[Dict] = []
        self.pheromones: Dict[str, float] = {}
        self.tick_count = 0
        
    def init_simulation(self, config: Dict[str, Any], environment: Dict[str, Any] = None, geometry_tree: List[Dict[str, Any]] = None):
        """
        Initialize the swarm with seed agents, resources, and tasks.
        """
        self.env_agent = EnvironmentAgent()
        self.gravity = environment.get("gravity", 9.81) if environment else 9.81
        self.construction_targets = geometry_tree if geometry_tree else []
        
        # Task Queue for Distributed Compute
        self.task_queue = []
        task_count = config.get("task_count", 0)
        if task_count > 0:
            logger.info(f"Swarm: Generating {task_count} compute tasks...")
            for i in range(task_count):
                self.task_queue.append({
                    "id": f"task_{i}",
                    "type": "COMPUTE_HASH",
                    "effort": 10.0, # Energy required to complete
                    "status": "pending"
                })

        # 1. Setup Environment
        self.resources = self.env_agent.init_swarm_resources(
            width=200.0, height=200.0, 
            count=config.get("resource_count", 20)
        )
        self.pheromones = {}
        
        # 2. Setup Seed Population
        agent_names = config.get("agent_types", ["VonNeumannAgent"])
        count = config.get("initial_pop", 2)
        
        for i in range(count):
            ag_type_name = agent_names[i % len(agent_names)]
            
            if ag_type_name == "VonNeumannAgent":
                agent = VonNeumannAgent(initial_energy=200.0)
                self.agents.append(agent)
            elif ag_type_name == "ConstructionAgent":
                agent = ConstructionAgent(initial_energy=250.0)
                self.agents.append(agent)
                
        logger.info(f"Swarm Initialized: {len(self.agents)} Agents, {len(self.task_queue)} Tasks")

    def run_simulation(self, ticks: int = 50) -> Dict[str, Any]:
        """
        Run the simulation for N ticks.
        Returns final metrics.
        """
        for _ in range(ticks):
            self.run_tick()
            
            # Stop if all tasks done? 
            # if not self.task_queue: break
            
        return self._capture_metrics()

    def run_tick(self):
        """
        Execute one simulation step.
        """
        self.tick_count += 1
        
        # Snapshot state for agents
        env_state = {
            "resources": self.resources,
            "pheromones": self.pheromones,
            "tasks": self.task_queue, # Shared Queue
            "targets": self.construction_targets
        }
        
        new_babies = []
        dead_ids = set()
        
        # A. Agent Decisions
        for agent in self.agents:
            # Pass shared state (Agents modify tasks/resources in place? 
            # Ideally they return requests, but for Python sim we pass Ref for speed)
            # To be safe, we let them modify 'tasks' via request?
            # For simplicity in V1: Agents modify the task object directly if they 'work' on it.
            res = agent.run(env_state)
            
            # Handle Task Completion Request
            if completed_task_id := res.get("completed_task_id"):
                # Remove from queue
                original_len = len(self.task_queue)
                self.task_queue = [t for t in self.task_queue if t["id"] != completed_task_id]
                if len(self.task_queue) < original_len:
                    # Grant Energy Bonus for Task?
                    agent.energy += 20.0 

            
            # Harvest
            if req := res.get("harvest_request"):
                harvested = self.env_agent.consume_resource(self.resources, res["pos"], req["amount"])
                if harvested > 0 and hasattr(agent, "energy"):
                   agent.energy += harvested
                   # Deposite Pheromone (Stigmergy)
                   self.pheromones[req["target_id"]] = self.pheromones.get(req["target_id"], 0.0) + 1.0

            # Replicate
            if child_config := res.get("child"):
                c_gen = child_config["genetics"]
                c_eng = child_config["energy_grant"]
                baby = None
                
                # Dynamic instantiation based on type string
                if child_config["type"] == "VonNeumannAgent":
                    baby = VonNeumannAgent(genetics=c_gen, initial_energy=c_eng)
                elif child_config["type"] == "ConstructionAgent":
                    baby = ConstructionAgent(genetics=c_gen, initial_energy=c_eng)
                
                if baby:
                    baby.pos = list(res["pos"])
                    new_babies.append(baby)
            
            # Death
            if res.get("status") == "dead":
                dead_ids.add(agent.id)
                
        # B. Update Lists
        self.agents = [a for a in self.agents if a.id not in dead_ids] + new_babies
        
        # C. Environment Entropy
        self.pheromones = self.env_agent.update_pheromones(self.pheromones)
        
        # D. Phase 3 VMK Safety Check (Collision Detection)
        collisions = self.check_swarm_collisions(self.agents)
        if collisions:
            logger.warning(f"Swarm Safety Audit: {len(collisions)} Collisions Detected!")
            # In future: Revert moves or damage agents.
        
    def check_swarm_collisions(self, active_agents: List[Any], margin: float = 0.1) -> List[str]:
        """
        Use VMK Math to detect agent-agent overlaps.
        O(N^2) for MVP.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return []

        # We can use a kernel to represent one agent, then probe with others.
        # But simpler: Use the kernel's helper math for analytical shapes?
        # Or instantiate a kernel for the 'Universe'.
        
        # Let's verify pairwise. 
        # A collision exists if Distance(A, B) < Radius(A) + Radius(B).
        # We use VMK to support future complex shapes.
        
        collisions = []
        n = len(active_agents)
        
        if n < 2: return []
        
        # Shared Kernel for math (static helper)
        # We don't really need a Stock for this, just the SDF functions.
        # But we must satisfy the class init.
        kernel = SymbolicMachiningKernel(stock_dims=[1,1,1]) 

        for i in range(n):
            for j in range(i + 1, n):
                a1 = active_agents[i]
                a2 = active_agents[j]
                
                # Assume spherical for Phase 3 (Ball Tool)
                # Future: Could be Box, V-Bit, etc.
                p1 = np.array(a1.pos + [0.0]) # 2D to 3D
                p2 = np.array(a2.pos + [0.0])
                
                # Get radii (using harvest efficiency as proxy for size in this simulation)
                r1 = getattr(a1.genetics, "harvest_efficiency", 1.0) * 5.0
                r2 = getattr(a2.genetics, "harvest_efficiency", 1.0) * 5.0
                
                dist = np.linalg.norm(p1 - p2)
                overlap = dist - (r1 + r2 + margin)
                
                # print(f"DEBUG: Check {a1.id} vs {a2.id}. Dist={dist:.2f}, R1={r1}, R2={r2}, Overlap={overlap:.2f}")

                if overlap < 0:
                    collisions.append(f"{a1.id} <-> {a2.id} (Overlap {abs(overlap):.3f})")

        return collisions

    def _capture_metrics(self) -> Dict[str, Any]:
        """
        Aggregate swarm stats.
        """
        total_energy = sum([a.energy for a in self.agents if hasattr(a, 'energy')])
        gens = [a.genetics.generation for a in self.agents if hasattr(a, 'genetics')]
        structures = sum([a.structures_built for a in self.agents if hasattr(a, 'structures_built')])
        
        return {
            "ticks": self.tick_count,
            "population": len(self.agents),
            "biomass_energy": total_energy,
            "structures_built": structures,
            "max_generation": max(gens) if gens else 0,
            "max_generation": max(gens) if gens else 0,
            "resources_remaining": len([r for r in self.resources if r["amount"] > 1.0]),
            "tasks_remaining": len(self.task_queue)
        }
