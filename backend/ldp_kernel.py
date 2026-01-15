import networkx as nx
import logging
from typing import Dict, Any, List, Callable, Optional, Set
from dataclasses import dataclass, field

# Configure Logging
logger = logging.getLogger("LDP_KERNEL")

@dataclass(frozen=True)
class RequirementNode:
    """
    A Symbolic Node in the Dependency Graph.
    Decoupled from domain-specific physics to ensure architectural broadness.
    """
    id: str
    domain: str  # Metadata for agentic filtering (e.g., 'THERMAL', 'STRUCTURAL')
    input_keys: List[str]  # Keys required from the global state for resolution
    resolver: Callable[[Dict[str, Any]], Any]  # Logical transformation function
    geometry_handler: Optional[str] = None  # Compilation target for SynthesisKernel
    metadata: Dict[str, Any] = field(default_factory=dict)

class LogicalDependencyParser:
    """
    HWC Core: The Universal Constraint Satisfier.
    Resolves the converged physical state of a machine from an arbitrary DAG of dependencies.
    """
    def __init__(self):
        self.graph = nx.DiGraph()
        self.registry: Dict[str, RequirementNode] = {}
        self.state: Dict[str, Any] = {}
        self.stale_nodes: Set[str] = set()
        self.logger = logger

    def register_requirement(self, node: RequirementNode):
        """
        Injects a requirement into the symbolic registry.
        Automatically maps edges based on defined input_keys.
        """
        self.registry[node.id] = node
        self.graph.add_node(node.id)
        
        # Connect to parent dependencies based on input requirements
        for key in node.input_keys:
            self.graph.add_edge(key, node.id)
        
        self.stale_nodes.add(node.id)

    def inject_input(self, key: str, value: Any, is_locked: bool = True):
        """
        External entry point for User Intent or Sensor Telemetry.
        Triggers downstream staleness propagation.
        """
        self.state[key] = value
        if key in self.graph:
            try:
                descendants = nx.descendants(self.graph, key)
                self.stale_nodes.update(descendants)
            except nx.NetworkXError:
                pass # Node might not be in graph yet if just an input var

    async def resolve(self, max_iterations: int = 5) -> List[Dict[str, Any]]:
        """
        Topological Resolution Engine (Async).
        Walks the graph and executes resolvers to converge the Physical State.
        Returns a list of instructions for the SynthesisKernel.
        """
        try:
            # Check for circular dependencies
            order = list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            self.logger.error("Cyclic dependency detected in Physical ISA.")
            raise ValueError("CYCLIC_DEPENDENCY_ERROR: Use iterative convergence resolver.")

        compilation_buffer = []

        # Convergence loop
        for i in range(max_iterations):
            has_changed = False
            
            for node_id in order:
                if node_id not in self.registry:
                    continue
                
                node = self.registry[node_id]
                
                # Check if all inputs are present in state
                if not all(k in self.state for k in node.input_keys):
                    continue

                # Execute Symbolic Resolution
                input_context = {k: self.state[k] for k in node.input_keys}
                previous_val = self.state.get(node_id)
                
                try:
                    # Support both async and sync resolvers
                    import inspect
                    if inspect.iscoroutinefunction(node.resolver):
                        new_val = await node.resolver(input_context)
                    else:
                        new_val = node.resolver(input_context)
                except Exception as e:
                    self.logger.error(f"Resolver failed for {node_id}: {e}")
                    continue
                
                if new_val != previous_val:
                    self.state[node_id] = new_val
                    has_changed = True
                    
                    # Prepare Physical Instruction for the Kernel
                    if node.geometry_handler:
                        compilation_buffer.append({
                            "id": node_id,
                            "handler": node.geometry_handler,
                            "params": new_val,
                            "domain": node.domain,
                            "iteration": i
                        })

            if not has_changed:
                self.logger.info(f"Convergence achieved at iteration {i}")
                break
        
        # Return unique instructions (latest state)
        unique_instr = {}
        for instr in compilation_buffer:
            unique_instr[instr['id']] = instr
            
        return list(unique_instr.values())

    def get_node_impact(self, node_id: str) -> List[str]:
        """Returns the 'Trace Path' (Children affected by this node)."""
        return list(nx.descendants(self.graph, node_id))

    def get_node_influence(self, node_id: str) -> List[str]:
        """Returns the 'Probe Path' (Parents affecting this node)."""
        return list(nx.ancestors(self.graph, node_id))
