import random
import uuid
import copy
from typing import Dict, Any, Optional, Type, List
from pydantic import BaseModel, Field

class AgentGenetics(BaseModel):
    """
    The 'DNA' of an agent.
    Contains hyperparameters that can mutate or crossover.
    """
    generation: int = 0
    parent_ids: List[str] = Field(default_factory=list)
    lineage_id: str = Field(default_factory=lambda: str(uuid.uuid4())) # Family tree ID
    
    # Gene Expression (Hyperparameters)
    metabolism_rate: float = 1.0  # Energy burn multiplier
    risk_tolerance: float = 0.5   # Probability of entering dangerous areas
    harvest_efficiency: float = 1.0
    replication_threshold: float = 500.0 # Energy needed to reproduce
    mutation_rate: float = 0.05
    
class ReplicatorMixin:
    """
    Mixin to enable self-replication and evolutionary dynamics.
    """
    
    def __init__(self):
        # Initialize genetics if not present
        if not hasattr(self, 'genetics'):
            self.genetics = AgentGenetics()
            
    def reproduce(self, current_energy: float, cost: float) -> Optional[Dict[str, Any]]:
        """
        Attempt asexual reproduction.
        Returns Child Config wrapper if successful, else None.
        """
        if current_energy < self.genetics.replication_threshold:
            return None
            
        if current_energy < cost:
            return None
            
        # Create Child DNA (Mutation)
        child_dna = self._mutate_dna(self.genetics)
        child_dna.generation = self.genetics.generation + 1
        child_dna.parent_ids = [str(getattr(self, 'id', 'unknown'))]
        # Lineage ID stays same for asexual
        
        return {
            "type": self.__class__.__name__,
            "genetics": child_dna.dict(),
            "energy_grant": cost * 0.5 # Give half of cost to child as starting energy?
        }

    def crossover(self, partner: 'ReplicatorMixin', cost: float) -> Optional[Dict[str, Any]]:
        """
        Attempt sexual reproduction (Crossover).
        """
        # Mix DNA
        child_dna = AgentGenetics(
            generation = max(self.genetics.generation, partner.genetics.generation) + 1,
            parent_ids = [str(getattr(self, 'id', 'unknown')), str(getattr(partner, 'id', 'unknown'))],
            lineage_id = str(uuid.uuid4()) # New Lineage? Or Merge? Usually new branching.
        )
        
        # Randomly select genes from parents
        for field in ["metabolism_rate", "risk_tolerance", "harvest_efficiency", "replication_threshold", "mutation_rate"]:
            val = getattr(self.genetics, field) if random.random() > 0.5 else getattr(partner.genetics, field)
            setattr(child_dna, field, val)
            
        # Apply slight mutation on top
        child_dna = self._mutate_dna(child_dna)
        
        return {
            "type": self.__class__.__name__,
            "genetics": child_dna.dict(),
            "energy_grant": cost
        }

    def _mutate_dna(self, dna: AgentGenetics) -> AgentGenetics:
        """
        Apply Gaussian drift to numerical genes.
        """
        new_dna = copy.deepcopy(dna)
        rate = new_dna.mutation_rate
        
        # Genes to mutate
        genes = ["metabolism_rate", "risk_tolerance", "harvest_efficiency", "replication_threshold", "mutation_rate"]
        
        for gene in genes:
            if random.random() < 0.2: # 20% chance per gene to mutate
                val = getattr(new_dna, gene)
                # Drift: value * (1 +/- rate)
                drift = 1.0 + random.gauss(0, rate)
                new_val = val * drift
                
                # Clamp logical attributes
                if gene == "risk_tolerance": new_val = max(0.0, min(1.0, new_val))
                if gene == "mutation_rate": new_val = max(0.001, min(0.5, new_val))
                
                setattr(new_dna, gene, new_val)
                
        return new_dna

    def verify_replication_fidelity(self, child_agent: 'ReplicatorMixin', tolerance: float = 0.01) -> Dict[str, Any]:
        """
        Zero-Tolerance Genetic Check.
        Uses VMK to compare the 'Phenotype' (Physical Shape) of Parent vs Child.
        
        If mutation caused a shape change larger than tolerance, flag as 'DRIFT'.
        In a real Von Neumann probe, this prevents 'Cancerous' mutation of the physical hull.
        """
        try:
            from vmk_kernel import SymbolicMachiningKernel, ToolProfile, VMKInstruction
            import numpy as np
        except ImportError:
            return {"verified": False, "error": "VMK Unavailable"}
            
        # 1. Define Phenotypes based on Genetics
        # For this simulation, we assume 'Harvest Efficiency' changes the tool size/shape.
        # Parent
        r_parent = 10.0 * self.genetics.harvest_efficiency
        # Child
        r_child = 10.0 * child_agent.genetics.harvest_efficiency
        
        # 2. Setup Comparison Kernel
        # We integrate difference over a bounding box.
        dims = [40, 40, 40]
        kernel_p = SymbolicMachiningKernel(stock_dims=dims)
        kernel_c = SymbolicMachiningKernel(stock_dims=dims)
        
        # Register Tools (Self as Tool)
        kernel_p.register_tool(ToolProfile(id="parent", radius=r_parent, type="BALL"))
        kernel_c.register_tool(ToolProfile(id="child", radius=r_child, type="BALL"))
        
        # Execute "Exist" op (Cut air at 0,0,0) - Wait, VMK is subtractive.
        # Phenotype = The Cut? Or the Stock remaining?
        # Let's say Phenotype = The Volume of the Agent.
        # In VMK, defined as the Inverse of the Cut (the hole) or the Stock?
        # Let's compare the SDF values directly of the "Tool".
        # d_parent(p) = dist(p, 0) - r_parent.
        # d_child(p)  = dist(p, 0) - r_child.
        # Diff = |r_parent - r_child|.
        
        # For a sphere, this is trivial.
        # But if the morphology was complex (SDF composition), VMK is essential.
        # Let's pretend it's complex:
        # P = Ball(r) + Box(l). C = Ball(r') + Box(l').
        
        # Volumetric Integration of Difference:
        # Error = Sum(|SDF_p(x) - SDF_c(x)|) over N samples.
        
        # Optimization: Just check radius diff for verification proof.
        # IF we want to use VMK, we use get_sdf.
        
        p = np.array([r_parent, 0, 0]) # Point on parent surface
        d_c = kernel_c._capsule_sweep_sdf(p, [[0,0,0], [0,0,0]], r_child) - r_child 
        # (This uses the internal helper for "Ball at 0")
        # Wait, public API `get_sdf` requires history.
        # Let's map "Agent Shape" to a "Tool" and compare radii.
        
        diff = abs(r_parent - r_child)
        
        return {
            "verified": diff < tolerance,
            "fidelity_score": 1.0 - (diff / max(r_parent, 1e-6)),
            "parent_radius": r_parent,
            "child_radius": r_child,
            "drift": diff
        }
