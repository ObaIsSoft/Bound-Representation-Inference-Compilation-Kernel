import networkx as nx
import uuid
import random
import copy
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum
from pydantic import BaseModel, Field

class PrimitiveType(str, Enum):
    CUBE = "CUBE"
    SPHERE = "SPHERE"
    CYLINDER = "CYLINDER"
    CONNECTOR = "CONNECTOR" # Structural joint
    VOID = "VOID" # Negative space (subtraction)
    LIBRARY_ASSET = "LIBRARY_ASSET" # External complex module (e.g. Motor, Wing)

class GeneParameter(BaseModel):
    name: str
    value: float
    min_val: float
    max_val: float
    locked: bool = False # Locked constraints cannot be mutated

class GeometryNode(BaseModel):
    id: str
    type: PrimitiveType
    asset_id: Optional[str] = None # Reference to external library ID (if type=LIBRARY_ASSET)
    params: Dict[str, GeneParameter]
    transform: List[float] = Field(default_factory=lambda: [0,0,0, 0,0,0]) # Pos(3) + Rot(3)

class GeometryGenome:
    """
    The DNA of a BRICK Design.
    Represents a Directed Acyclic Graph (DAG) of geometric primitives.
    
    Capabilities:
    - Topological Mutation: Add/Remove nodes.
    - Parametric Mutation: Tweak continuous values.
    - Crossover: Serialize/Deserialize for breeding.
    """
    def __init__(self, root_id: Optional[str] = None, seed_params: Optional[Dict[str, Any]] = None, seed_type: PrimitiveType = PrimitiveType.CUBE):
        self.graph = nx.DiGraph()
        self.root_id = root_id or str(uuid.uuid4())
        
        # Initialize with a root node (Hub) if empty
        if not root_id:
            defaults = seed_params or {}
            min_v = 0.1
            max_v = 10.0
            
            # Dynamic Parameter Factory based on Type
            params = {}
            if seed_type == PrimitiveType.CUBE:
                params = {
                    "width": GeneParameter(name="width", value=defaults.get("width", 1.0), min_val=min_v, max_val=max_v),
                    "height": GeneParameter(name="height", value=defaults.get("height", 1.0), min_val=min_v, max_val=max_v),
                    "depth": GeneParameter(name="depth", value=defaults.get("depth", 1.0), min_val=min_v, max_val=max_v)
                }
            elif seed_type == PrimitiveType.SPHERE:
                 params = {
                    "radius": GeneParameter(name="radius", value=defaults.get("radius", 1.0), min_val=min_v, max_val=max_v)
                 }
            elif seed_type == PrimitiveType.CYLINDER:
                 params = {
                    "radius": GeneParameter(name="radius", value=defaults.get("radius", 0.5), min_val=min_v, max_val=max_v),
                    "height": GeneParameter(name="height", value=defaults.get("height", 2.0), min_val=min_v, max_val=max_v)
                 }
            
            # Default to Cube if undetermined or complex
            if not params:
                 params = {
                    "width": GeneParameter(name="width", value=1.0, min_val=min_v, max_val=max_v),
                    "height": GeneParameter(name="height", value=1.0, min_val=min_v, max_val=max_v),
                    "depth": GeneParameter(name="depth", value=1.0, min_val=min_v, max_val=max_v)
                }
            
            root_node = GeometryNode(
                id=self.root_id,
                type=seed_type,
                params=params
            )
            self.add_node(root_node)

    def add_node(self, node: GeometryNode, parent_id: Optional[str] = None):
        """Adds a primitive gene to the genome."""
        self.graph.add_node(node.id, data=node)
        if parent_id:
            if parent_id not in self.graph:
                raise ValueError(f"Parent {parent_id} does not exist.")
            self.graph.add_edge(parent_id, node.id)

    def remove_node(self, node_id: str):
        """
        Removes a gene. 
        If it has children, they are either pruned or re-attached to the grandparent.
        Current strategy: Prune subtree (drastic mutation).
        """
        if node_id == self.root_id:
            raise ValueError("Cannot remove root node.")
        
        if node_id in self.graph:
            # Get descendants to remove entire branch
            descendants = nx.descendants(self.graph, node_id)
            self.graph.remove_nodes_from(descendants)
            self.graph.remove_node(node_id)

    def get_node(self, node_id: str) -> GeometryNode:
        return self.graph.nodes[node_id]['data']

    def mutate_parameter(self, mutation_rate: float = 0.1, strength: float = 0.2):
        """
        Gaussian mutation of continuous parameters.
        Iterates through all nodes and mutates unlocked params.
        """
        for node_id in self.graph.nodes:
            node = self.get_node(node_id)
            for param_name, gene in node.params.items():
                if gene.locked:
                    continue
                
                if random.random() < mutation_rate:
                    # Gaussian perturbation
                    delta = random.gauss(0, strength) * gene.value
                    new_val = gene.value + delta
                    
                    # Clamp
                    gene.value = max(gene.min_val, min(gene.max_val, new_val))

    def clone(self) -> 'GeometryGenome':
        """Deep copy for reproduction."""
        new_genome = GeometryGenome(root_id=self.root_id)
        new_genome.graph = self.graph.copy()
        # Deep copy the node data objects
        for n in new_genome.graph.nodes:
            original_data = self.graph.nodes[n]['data']
            new_genome.graph.nodes[n]['data'] = copy.deepcopy(original_data)
        return new_genome

    def to_json(self) -> Dict[str, Any]:
        """Serialization for Network Transport / Storage."""
        nodes = []
        edges = []
        for n_id in self.graph.nodes:
            node = self.get_node(n_id)
            nodes.append(node.dict())
        
        for u, v in self.graph.edges:
            edges.append({"source": u, "target": v})
            
        return {"root_id": self.root_id, "nodes": nodes, "edges": edges}

    @staticmethod
    def from_json(data: Dict[str, Any]) -> 'GeometryGenome':
        """Deserialization."""
        genome = GeometryGenome(root_id=data['root_id'])
        genome.graph.clear()
        
        for node_data in data['nodes']:
            # Reconstruct GeneParameter objects first
            params = {}
            for k, v in node_data['params'].items():
                params[k] = GeneParameter(**v)
            
            node_data['params'] = params
            node = GeometryNode(**node_data)
            genome.graph.add_node(node.id, data=node)
            
        for edge in data['edges']:
            genome.graph.add_edge(edge['source'], edge['target'])
            
        return genome
