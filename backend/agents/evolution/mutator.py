import random
import uuid
from typing import List, Optional
from .genome import GeometryGenome, GeometryNode, PrimitiveType, GeneParameter

class EvolutionaryMutator:
    """
    The engine of novelty.
    Performs topological mutations to explore the structural design space.
    """
    
    @staticmethod
    @staticmethod
    def mutate_topology(genome: GeometryGenome, add_prob: float, remove_prob: float, available_assets: List[str] = None):
        """
        Executes structural changes.
        """
        r = random.random()
        
        if r < add_prob:
            EvolutionaryMutator._add_random_primitive(genome, available_assets)
        elif r < (add_prob + remove_prob):
            EvolutionaryMutator._remove_random_primitive(genome)

    @staticmethod
    def _add_random_primitive(genome: GeometryGenome, available_assets: List[str] = None):
        """
        Adds a new child node to a random existing node.
        """
        parent_ids = list(genome.graph.nodes)
        if not parent_ids: return
        
        parent_id = random.choice(parent_ids)
        parent_node = genome.get_node(parent_id)
        
        # Determine new type
        # If we have assets, 20% chance to pick one of them
        new_type = random.choice([PrimitiveType.CUBE, PrimitiveType.SPHERE, PrimitiveType.CYLINDER])
        chosen_asset_id = None
        
        if available_assets and random.random() < 0.3:
            new_type = PrimitiveType.LIBRARY_ASSET
            chosen_asset_id = random.choice(available_assets)

        
        # Heuristic: Size relative to parent (smaller)
        # Assuming parent has 'width' or 'radius' - this is a simplification
        # In a real implementation, we'd introspect the parent's actual params.
        scale_factor = random.uniform(0.3, 0.8)
        
        new_id = str(uuid.uuid4())
        
        # Basic Parameter Factory
        params = {}
        if new_type == PrimitiveType.CUBE:
            params = {
                "width": GeneParameter(name="width", value=1.0 * scale_factor, min_val=0.1, max_val=5.0),
                "height": GeneParameter(name="height", value=1.0 * scale_factor, min_val=0.1, max_val=5.0),
                "depth": GeneParameter(name="depth", value=1.0 * scale_factor, min_val=0.1, max_val=5.0)
            }
        elif new_type == PrimitiveType.SPHERE:
            params = {
                "radius": GeneParameter(name="radius", value=0.5 * scale_factor, min_val=0.1, max_val=5.0)
            }
        elif new_type == PrimitiveType.CYLINDER:
            params = {
                "radius": GeneParameter(name="radius", value=0.5 * scale_factor, min_val=0.1, max_val=5.0),
                "height": GeneParameter(name="height", value=1.0 * scale_factor, min_val=0.1, max_val=5.0)
            }
        elif new_type == PrimitiveType.LIBRARY_ASSET:
             params = {
                "scale": GeneParameter(name="scale", value=1.0, min_val=0.1, max_val=5.0),
                # Assets might still need bounding box approximation params for physics
                "approx_width": GeneParameter(name="approx_width", value=1.0, min_val=0.1, max_val=5.0, locked=True),
             }
            
        # Random Offset Transform (Local to parent)
        # Position: Random point roughly on surface of parent (simplified box)
        off_x = random.uniform(-1, 1)
        off_y = random.uniform(-1, 1)
        off_z = random.uniform(-1, 1)
        
        new_node = GeometryNode(
            id=new_id,
            type=new_type,
            asset_id=chosen_asset_id,
            params=params,
            transform=[off_x, off_y, off_z, 0, 0, 0] # XYZ, RPY
        )
        
        genome.add_node(new_node, parent_id)
        # print(f"[MUTATION] Added {new_type} to {parent_id}")

    @staticmethod
    def _remove_random_primitive(genome: GeometryGenome):
        """
        Prunes a branch.
        """
        nodes = list(genome.graph.nodes)
        if len(nodes) <= 1: return # Don't delete root if it's the only one
        
        target = random.choice(nodes)
        if target == genome.root_id:
            # Try again (once) to find a non-root
            target = random.choice(nodes)
            if target == genome.root_id: return
            
        genome.remove_node(target)
        # print(f"[MUTATION] Removed {target}")
