from typing import Dict, Any, List, Optional
import logging
import uuid
import math
from agents.replicator_mixin import ReplicatorMixin

logger = logging.getLogger(__name__)

class ConstructionAgent(ReplicatorMixin):
    """
    Builder Agent.
    Interprets Pheromones as Construction Sites.
    Consumes ORE to build STRUCTURES.
    """
    
    def __init__(self, agent_id: Optional[str] = None, initial_energy: float = 150.0, genetics: Optional[Dict] = None):
        super().__init__()
        self.id = agent_id or f"builder_{uuid.uuid4().hex[:8]}"
        self.energy = initial_energy
        self.ore_storage = 0.0
        self.pos = [0.0, 0.0]
        self.structures_built = 0
        
        # Load Genetics
        if genetics:
            self.genetics.generation = genetics.get("generation", 0)
            self.genetics.parent_ids = genetics.get("parent_ids", [])
            # Builder-specific defaults?
            self.genetics.metabolism_rate = genetics.get("metabolism_rate", 1.2) # Heavier
            self.genetics.replication_threshold = genetics.get("replication_threshold", 600.0)

    def run(self, environment_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Lifecycle: Forage -> Harvest -> Build -> Replicate.
        """
        logs = []
        status = "alive"
        action = "idle"
        child = None
        harvest_request = None
        
        # 1. Entropy
        self.energy -= (1.0 * self.genetics.metabolism_rate)
        if self.energy <= 0:
            return {"id": self.id, "status": "dead"}

        # 2. Perception
        resources = environment_state.get("resources", [])
        pheromones = environment_state.get("pheromones", {})
        
        # 3. Logic
        
        # A. Construction (Priority)
        # If holding ORE and near High Pheromone (Worksite), Build.
        if self.ore_storage >= 50.0:
            # Look for worksite (High Pheromone)
            best_site = None
            max_ph = 0.0
            
            # Identify Construction Site
            # If templates exist, pick one that needs building
            targets = environment_state.get("targets", [])
            my_target = targets[0] if targets else None # Simple: Everyone targets first item for now
            
            if my_target:
                # Move to target location (Placeholder: Assume target is at 0,0 or random)
                # In real sim, parts have coordinates.
                # For now, if we have ore, we just "Complete" the target item.
                self.ore_storage -= 50.0
                self.structures_built += 1
                action = "building"
                
                # Signal Completion of a specific part
                logs.append(f"{self.id} built Part [{my_target.get('id', 'unknown')}]")
                
                # Remove target from shared state? 
                # Ideally return request "build_complete_id".
                
            elif self._check_local_pheromone(pheromones, self.pos) > 5.0:
                # Build generic structure if no target but high pheromone
                self.ore_storage -= 50.0
                self.structures_built += 1
                action = "building"
                logs.append(f"{self.id} built Generic Structure at {self.pos}")
            
            my_ph = self._check_local_pheromone(pheromones, self.pos)
            if my_ph > 5.0:
                # Build here!
                self.ore_storage -= 50.0
                self.structures_built += 1
                action = "building"
                logs.append(f"{self.id} built Structure at {self.pos}")
            else:
                # Move to Pheromone Peak
                # Simple gradient ascent (simulated)
                # Just find max pheromone ID and go there
                pass # Movement logic below

        # B. Harvesting (If need ORE or Energy)
        # Scan for ORE
        target = None
        min_dist = float('inf')
        
        for res in resources:
            dx = res["x"] - self.pos[0]
            dy = res["y"] - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Builders ATTRACTED to Pheromones (Stigmergy cooperation)
            ph_level = pheromones.get(res["id"], 0.0)
            attraction = ph_level * 10.0 
            
            score = dist - attraction # Lower is better
            
            if score < min_dist:
                min_dist = score
                target = res
                
        if target:
            # Move
            step = 3.0 # Slower than Scout
            dx = target["x"] - self.pos[0]
            dy = target["y"] - self.pos[1]
            mag = math.sqrt(dx*dx + dy*dy)
            if mag > 0:
                self.pos[0] += (dx/mag) * step
                self.pos[1] += (dy/mag) * step
            
            # Harvest
            if mag < 10.0:
                harvest_request = {
                    "target_id": target["id"],
                    "amount": 10.0 # Standard scoop
                }
                action = "harvesting"
                # If ORE, store it. If Energy, eat it.
                # Simplified: All resources give Energy, but we count it as "Ore" if type matches
                # In EnvAgent, we treat everything as generic "amount" returned.
                # Let's assume Env returns generic "harvested".
                # Logic: If target.type == ORE -> self.ore += harvested
                # We need type checking. Handled in Manager? Or assume generic for now.
                if target["type"] == "ORE":
                    # We will handle the split in Manager or assume generic energy for now
                    # For Phase 21 proof, let's treat everything as Energy but count "harvested" as potential ore
                    self.ore_storage += 5.0 # Simulation
                
        # C. Replication (Low priority)
        if self.energy > self.genetics.replication_threshold:
            cost = self.genetics.replication_threshold * 0.7
            child_config = self.reproduce(self.energy, cost)
            if child_config:
                self.energy -= cost
                child = child_config
                action = "replicated"

        return {
            "id": self.id,
            "status": status,
            "energy": self.energy,
            "pos": self.pos,
            "action": action,
            "child": child,
            "harvest_request": harvest_request,
            "structures": self.structures_built # Metric
        }

    def _check_local_pheromone(self, grid: Dict[str, float], pos: List[float]) -> float:
        # Simplified: Sum pheromones of closest resource IDs? 
        # Since grid is ID->Float, we don't have spatial grid without Env.
        # Returning 0 for now as placeholder unless we link ID to Pos.
        # For prototype, assume we find max global (Basic Swarm).
        vals = list(grid.values())
        return max(vals) if vals else 0.0
