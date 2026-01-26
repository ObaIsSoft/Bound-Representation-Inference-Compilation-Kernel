import random
import uuid
import copy
import networkx as nx
from .genome import GeometryGenome, GeometryNode

class EvolutionaryCrossover:
    """
    The engine of recombination.
    Swaps sub-structures between two distinct genomes.
    """
    
    @staticmethod
    def crossover(parent_a: GeometryGenome, parent_b: GeometryGenome) -> GeometryGenome:
        """
        Performs subtree crossover.
        Returns a new child genome that is primarily A with a branch from B.
        """
        # 1. Clone Parent A as the base for the Child
        child = parent_a.clone()
        
        # 2. Select graft points
        # Target in A: Where we will remove a branch and attach B's branch
        nodes_a = list(child.graph.nodes)
        if len(nodes_a) <= 1: return child # A is only root, can't easily swap
        
        target_id_a = random.choice(nodes_a)
        if target_id_a == child.root_id:
             # Try to pick a non-root if possible, else we are replacing the whole tree content
             target_id_a = random.choice(nodes_a)
        
        # Source in B: The branch we want to steal
        nodes_b = list(parent_b.graph.nodes)
        source_id_b = random.choice(nodes_b)
        
        # 3. Identify functionality
        # If we are replacing the root of A (unlikely but possible), strict rules apply.
        # Let's handle the common case: restructuring a limb.
        
        if target_id_a == child.root_id:
            # Replacing root is dangerous for validity. Skip or implement logic to merge roots.
            # Strategy: Don't replace root. Replace a child of root.
            successors = list(child.graph.successors(child.root_id))
            if successors:
                target_id_a = random.choice(successors)
            else:
                # A is empty root. Just append B's branch to it.
                EvolutionaryCrossover._graft_subtree(child, parent_b, child.root_id, source_id_b)
                return child
        
        # 4. Perform the Swap
        # Get parent of the target in A (to re-attach later)
        parents_of_target = list(child.graph.predecessors(target_id_a))
        if not parents_of_target: return child # Should not happen for non-root
        attachment_point_id = parents_of_target[0]
        
        # Prune A's branch
        # Get all descendants
        descendants_a = nx.descendants(child.graph, target_id_a)
        nodes_to_remove = descendants_a.union({target_id_a})
        child.graph.remove_nodes_from(nodes_to_remove)
        
        # Graft B's branch
        EvolutionaryCrossover._graft_subtree(child, parent_b, attachment_point_id, source_id_b)
        
        return child
        
    @staticmethod
    def _graft_subtree(destination_genome: GeometryGenome, source_genome: GeometryGenome, 
                      dest_parent_id: str, source_node_id: str):
        """
        Copies a subtree from Source to Destination, attaching it to dest_parent_id.
        Regenerates UUIDs to ensure uniqueness.
        """
        # 1. Get subgraph from source (node + descendants)
        descendants = nx.descendants(source_genome.graph, source_node_id)
        subgraph_nodes = descendants.union({source_node_id})
        
        # Map old IDs to new IDs
        id_map = {old: str(uuid.uuid4()) for old in subgraph_nodes}
        
        # 2. Copy nodes
        for old_id in subgraph_nodes:
            new_id = id_map[old_id]
            old_node = source_genome.get_node(old_id)
            
            # Deep copy data
            new_node = copy.deepcopy(old_node)
            new_node.id = new_id
            
            destination_genome.graph.add_node(new_id, data=new_node)
            
        # 3. Copy edges (internal to subtree)
        for u, v in source_genome.graph.edges(subgraph_nodes):
             if u in id_map and v in id_map:
                 destination_genome.graph.add_edge(id_map[u], id_map[v])
                 
        # 4. Create attachment edge
        new_root_id = id_map[source_node_id]
        if dest_parent_id in destination_genome.graph:
            destination_genome.graph.add_edge(dest_parent_id, new_root_id)
