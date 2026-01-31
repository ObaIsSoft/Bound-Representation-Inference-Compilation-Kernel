from typing import Dict, Any, List, Optional
import logging
import uuid
import math
from backend.agents.replicator_mixin import ReplicatorMixin

logger = logging.getLogger(__name__)

class ConstructionAgent(ReplicatorMixin):
    """
    Builder Agent.
    Interprets Pheromones as Construction Sites.
    Consumes ORE to build STRUCTURES.
    """
    
    def __init__(self, agent_id: Optional[str] = None, initial_energy: float = None, genetics: Optional[Dict] = None):
        super().__init__()
        
        # Load Config
        try:
            from backend.config.swarm_config import CONSTRUCTION_DEFAULTS, PHEROMONE_THRESHOLDS, GENETIC_DEFAULTS
            self.defaults = CONSTRUCTION_DEFAULTS
            self.thresholds = PHEROMONE_THRESHOLDS
            self.genetic_defaults = GENETIC_DEFAULTS
        except ImportError:
            logger.warning("Could not import swarm_config. Using hardcoded fallbacks.")
            self.defaults = {"initial_energy": 150.0, "metabolism_rate": 1.2, "replication_threshold": 600.0, "build_cost_ore": 50.0}
            self.thresholds = {"build_trigger": 5.0}
            self.genetic_defaults = {}

        self.id = agent_id or f"builder_{uuid.uuid4().hex[:8]}"
        self.energy = initial_energy if initial_energy is not None else self.defaults["initial_energy"]
        self.ore_storage = 0.0
        self.pos = [0.0, 0.0]
        self.structures_built = 0
        
        # Load Genetics
        if genetics:
            self.genetics.generation = genetics.get("generation", 0)
            self.genetics.parent_ids = genetics.get("parent_ids", [])
            # Builder-specific defaults
            self.genetics.metabolism_rate = genetics.get("metabolism_rate", self.defaults["metabolism_rate"]) 
            self.genetics.replication_threshold = genetics.get("replication_threshold", self.defaults["replication_threshold"])
        else:
            self.genetics.metabolism_rate = self.defaults["metabolism_rate"]
            self.genetics.replication_threshold = self.defaults["replication_threshold"]

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
        # 3. Logic
        
        # A. Construction (Priority)
        # If holding ORE and near High Pheromone (Worksite), Build.
        build_cost = self.defaults.get("build_cost_ore", 50.0)
        build_trigger = self.thresholds.get("build_trigger", 5.0)
        
        if self.ore_storage >= build_cost:
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
                self.ore_storage -= build_cost
                self.structures_built += 1
                action = "building"
                
                # Signal Completion of a specific part
                logs.append(f"{self.id} built Part [{my_target.get('id', 'unknown')}]")
                
                # Remove target from shared state? 
                # Ideally return request "build_complete_id".
                
            elif self._check_local_pheromone(pheromones, self.pos) > build_trigger:
                # Build generic structure if no target but high pheromone
                self.ore_storage -= build_cost
                self.structures_built += 1
                action = "building"
                logs.append(f"{self.id} built Generic Structure at {self.pos}")
            
            my_ph = self._check_local_pheromone(pheromones, self.pos)
            if my_ph > build_trigger:
                # Build here!
                self.ore_storage -= build_cost
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
        
        attract_mult = self.thresholds.get("harvest_attraction_multiplier", 10.0)
        
        for res in resources:
            dx = res["x"] - self.pos[0]
            dy = res["y"] - self.pos[1]
            dist = math.sqrt(dx*dx + dy*dy)
            
            # Builders ATTRACTED to Pheromones (Stigmergy cooperation)
            ph_level = pheromones.get(res["id"], 0.0)
            attraction = ph_level * attract_mult
            
            score = dist - attraction # Lower is better
            
            if score < min_dist:
                min_dist = score
                target = res
                
        if target:
            # Move
            step = self.defaults.get("movement_step", 3.0) # Slower than Scout
            dx = target["x"] - self.pos[0]
            dy = target["y"] - self.pos[1]
            mag = math.sqrt(dx*dx + dy*dy)
            if mag > 0:
                self.pos[0] += (dx/mag) * step
                self.pos[1] += (dy/mag) * step
            
            # Harvest
            if mag < 10.0:
                harvest_amt = self.defaults.get("harvest_amount", 10.0)
                harvest_request = {
                    "target_id": target["id"],
                    "amount": harvest_amt
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
                    self.ore_storage += harvest_amt * 0.5 # Simulation efficiency loss
                
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
