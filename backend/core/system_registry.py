
from core.hierarchical_resolver import ModularISA, HierarchicalResolver
from typing import Dict, Optional

class RecursiveISARegistry:
    """
    Manages the lifecycle of ModularISA pods.
    Supports dynamic creation, hierarchy changes, and recursion.
    Replaces static/singleton "Sesame" demo.
    """
    def __init__(self):
        # Default Root (Empty Project)
        self.root = ModularISA(name="Project_Root", constraints={"mass_budget": 100.0})
        self.resolver = HierarchicalResolver(self.root)
        self.registry: Dict[str, ModularISA] = {self.root.id: self.root}

    def create_pod(self, name: str, parent_id: Optional[str] = None, constraints: Dict = None) -> ModularISA:
        """Dynamically creates a new Pod and links it to the hierarchy."""
        new_pod = ModularISA(name=name, constraints=constraints or {})
        
        # Link to Parent
        if parent_id:
            parent = self.registry.get(parent_id)
            if not parent:
                # If parent not found, attach to root as fallback or error?
                # Attaching to root for safety
                parent = self.root
            
            new_pod.parent_id = parent.id
            # Auto-generate key name from pod name (slugify)
            key = name.lower().replace(" ", "_")
            parent.sub_pods[key] = new_pod
        else:
            # If no parent specified, assumes it's a new separate root?
            # For now, we only support one project root active.
            pass

        self.registry[new_pod.id] = new_pod
        return new_pod

    def get_pod(self, pod_id: str) -> Optional[ModularISA]:
        return self.registry.get(pod_id)

    def delete_pod(self, pod_id: str) -> bool:
        """Removes a pod and its subtree."""
        if pod_id == self.root.id:
            return False # Cannot delete root
            
        pod = self.registry.get(pod_id)
        if not pod:
            return False

        # Unlink from parent
        if pod.parent_id:
            parent = self.registry.get(pod.parent_id)
            if parent:
                # Find key
                keys_to_remove = [k for k, v in parent.sub_pods.items() if v.id == pod_id]
                for k in keys_to_remove:
                    del parent.sub_pods[k]

        # Recursively remove from registry
        self._recursive_cleanup(pod)
        return True

    def _recursive_cleanup(self, pod: ModularISA):
        if pod.id in self.registry:
            del self.registry[pod.id]
        for sub in pod.sub_pods.values():
            self._recursive_cleanup(sub)

    def get_resolver(self):
        return self.resolver

# Singleton Instance
_REGISTRY = RecursiveISARegistry()

def get_system_registry():
    return _REGISTRY

def get_system_resolver():
    return _REGISTRY.get_resolver()
