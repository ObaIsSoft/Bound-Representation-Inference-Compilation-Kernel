import networkx as nx
from typing import List, Dict, Set

class DependencyGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_parameter(self, param_name: str):
        """Register a parameter node."""
        if not self.graph.has_node(param_name):
            self.graph.add_node(param_name)

    def add_dependency(self, upstream: str, downstream: str):
        """
        Define that 'downstream' depends on 'upstream'.
        Change in upstream -> invalidates downstream.
        """
        self.add_parameter(upstream)
        self.add_parameter(downstream)
        self.graph.add_edge(upstream, downstream)

    def update_parameter(self, param_name: str) -> List[str]:
        """
        Mark a parameter as changed and return all dependent parameters 
        that need re-evaluation (downstream descendants).
        """
        if not self.graph.has_node(param_name):
            return []
        
        # Get all descendants (recursive)
        descendants = nx.descendants(self.graph, param_name)
        # Return as list, sorted topologically if possible, or just set
        # Ideally we want execution order, so let's stick to topological sort of subgraph
        subgraph = self.graph.subgraph(descendants | {param_name})
        try:
            order = list(nx.topological_sort(subgraph))
            # Remove the param itself from invalidation list (it's the source)
            if param_name in order:
                order.remove(param_name)
            return order
        except nx.NetworkXUnfeasible:
            # Cycle detected
            return list(descendants)

    def get_execution_order(self) -> List[str]:
        """Return full topological sort of the graph."""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXUnfeasible:
            return []

def create_default_dependency_graph() -> DependencyGraph:
    """Initialize the graph with standard physical relationships."""
    dg = DependencyGraph()
    
    # Aero/Drone relationships
    dg.add_dependency("rotor_diameter", "lift_force")
    dg.add_dependency("rotor_diameter", "drag_coefficient")
    dg.add_dependency("rotor_diameter", "power_requirement")
    dg.add_dependency("rotor_diameter", "thrust")
    
    dg.add_dependency("velocity", "lift_force")
    dg.add_dependency("velocity", "drag_force")
    dg.add_dependency("velocity", "power_requirement")
    
    dg.add_dependency("total_mass", "power_requirement")
    dg.add_dependency("total_mass", "center_of_gravity")
    dg.add_dependency("total_mass", "stability_margin")
    
    dg.add_dependency("center_of_gravity", "stability_margin")
    dg.add_dependency("center_of_gravity", "control_authority")
    
    return dg
