from typing import Dict, Any, List, Optional
import logging
import uuid
import math
import random
from agents.replicator_mixin import ReplicatorMixin

logger = logging.getLogger(__name__)

class VonNeumannAgent(ReplicatorMixin):
    """
    Von Neumann Probe Prototype.
    A self-replicating agent that harvests resources and spawns children.
    """
    
    def __init__(self, agent_id: Optional[str] = None, initial_energy: float = 100.0, genetics: Optional[Dict] = None):
        super().__init__()
        self.id = agent_id or f"vn_{uuid.uuid4().hex[:8]}"
        self.energy = initial_energy
        self.pos = [0.0, 0.0] # x, y
        
        # Load Genetics if provided (Child)
        if genetics:
            self.genetics.generation = genetics.get("generation", 0)
            self.genetics.parent_ids = genetics.get("parent_ids", [])
            self.genetics.lineage_id = genetics.get("lineage_id", "")
            # Hyperparams
            self.genetics.metabolism_rate = genetics.get("metabolism_rate", 1.0)
            self.genetics.risk_tolerance = genetics.get("risk_tolerance", 0.5)
            self.genetics.harvest_efficiency = genetics.get("harvest_efficiency", 1.0)
            self.genetics.replication_threshold = genetics.get("replication_threshold", 500.0)
            self.genetics.mutation_rate = genetics.get("mutation_rate", 0.05)

    def run(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one tick of lifecycle.
        """
        logs = []
        action = "idle"
        status = "alive"
        child = None
        harvest_request = None
        
        # 1. Entropy (Cost of Living)
        cost_of_living = 1.0 * self.genetics.metabolism_rate
        self.energy -= cost_of_living
        
        if self.energy <= 0:
            return {
                "id": self.id,
                "status": "dead", 
                "logs": [f"{self.id} died of energy depletion."]
            }
            
        # 2. Perception
        resources = environment_state.get("resources", [])
        pheromones = environment_state.get("pheromones", {})
        
        # 3. Decision Logic
        # A. Check Reproduction
        child = None
        if self.energy > self.genetics.replication_threshold:
            # Attempt Replication
            # Cost = 60% of threshold (Child gets half, 10% lost)
            cost = self.genetics.replication_threshold * 0.6 
            child_config = self.reproduce(self.energy, cost)
            
            if child_config:
                self.energy -= cost
                action = "replicated"
                logs.append(f"{self.id} spawned child (Gen {self.genetics.generation + 1})")
                child = child_config
                
        # B. Task Processing (Compute Swarm)
        # If queue exists, prioritizing work over random harvesting
        tasks = environment_state.get("tasks", [])
        completed_task_id = None
        
        if tasks and self.energy > 50.0:
            # Pick first available task
            task = tasks[0]
            action = "computing"
            
            # Consume Energy to Work
            effort = 5.0
            self.energy -= effort
            
            # "Complete" the task (Simplification: 1 tick = done for small tasks)
            # In real logic, we'd decrement task['effort'].
            completed_task_id = task["id"]
            logs.append(f"{self.id} completed {task['id']}")
            
        # C. Foraging / Harvesting (Fallback)
        elif resources:
            # Find nearest resource
            target = None
            min_dist = float('inf')
            
            for res in resources:
                dist = math.sqrt((res["x"] - self.pos[0])**2 + (res["y"] - self.pos[1])**2)
                
                # Pheromone Avoidance (Stigmergy)
                ph_level = pheromones.get(res["id"], 0.0)
                avoidance = ph_level * (1.0 - self.genetics.risk_tolerance)
                
                score = dist + (avoidance * 100.0)
                
                if score < min_dist:
                    min_dist = score
                    target = res
            
            if target:
                # Move towards target
                step_size = 5.0
                dx = target["x"] - self.pos[0]
                dy = target["y"] - self.pos[1]
                mag = math.sqrt(dx*dx + dy*dy)
                
                if mag > 0:
                    self.pos[0] += (dx/mag) * step_size
                    self.pos[1] += (dy/mag) * step_size
                    
                # Harvest
                if min_dist < 10.0:
                    action = "harvesting"
                    harvest_request = {
                        "target_id": target["id"],
                        "amount": 10.0 * self.genetics.harvest_efficiency
                    }
                else:
                    action = "moving"
            else:
                harvest_request = None
        
        # D. Replication Check (End of Cycle)
        if self.energy > self.genetics.replication_threshold:
             cost = self.genetics.replication_threshold * 0.6
             child_config = self.reproduce(self.energy, cost)
             
             if child_config:
                 # Phase 3 VMK Validation: Verify Phenotype Fidelity
                 try:
                     # Instantiate temp child to check genetics->phenotype mapping
                     temp_child = VonNeumannAgent(genetics=child_config["genetics"])
                     check = self.verify_replication_fidelity(temp_child, tolerance=5.0) # 5.0 unit (radius) tolerance
                     
                     if check["verified"]:
                         self.energy -= cost
                         child = child_config
                         logs.append(f"{self.id} replicated (Gen {self.genetics.generation + 1}). Drift={check['drift']:.3f}")
                     else:
                         # Abort cancerous mutation
                         self.energy -= (cost * 0.1) # Wasted effort
                         logs.append(f"{self.id} replication ABORTED. Cancerous Drift {check['drift']:.2f} > Limit.")
                 except Exception as e:
                     logs.append(f"Replication Check Failed: {e}")

        return {
            "id": self.id,
            "status": status,
            "energy": self.energy,
            "pos": self.pos,
            "action": action,
            "child": child, 
            "harvest_request": harvest_request,
            "completed_task_id": completed_task_id, # New Output
            "logs": logs
        }
