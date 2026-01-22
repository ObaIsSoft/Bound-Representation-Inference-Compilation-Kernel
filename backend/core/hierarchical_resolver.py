import uuid
import asyncio
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field

class ModularISA(BaseModel):
    """
    The Recursive Hardware Unit. 
    Acts like a 'File' in Fusion but functions as a 'Scoped Compiler'.
    """
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    parent_id: Optional[str] = None
    
    # Internal Logic (Private State)
    constraints: Dict[str, Any] = {}
    
    # Nested Sub-Branches (The 'Component' tree)
    sub_pods: Dict[str, 'ModularISA'] = {}
    
    # The 'Public' interface exposed to parents (The 'Footprint')
    exports: Dict[str, float] = {
        "mass": 0.0,
        "power_draw": 0.0,
        "bounding_volume": 0.0
    }
    
    # Dirty Flag for Lazy Evaluation
    is_dirty: bool = False

class HierarchicalResolver:
    """
    Resolves the 'Up-to-Down' and 'Down-to-Up' propagation.
    Ensures that a change in a 'Leg' correctly affects the 'Main Body'.
    """
    def __init__(self, root: ModularISA):
        self.root = root

    async def propagate_down(self, pod: ModularISA, shared_constraints: Dict[str, Any]):
        """
        Top-Down: Passing 'Global Intent' to sub-elements.
        Only propagates if constraints actually change to save compute.
        """
        changed = False
        for key, val in shared_constraints.items():
            if key in pod.constraints and pod.constraints[key] != val:
                pod.constraints[key] = val
                changed = True
                print(f"[LDP]: Propagated {key} to {pod.name}")
        
        if changed:
            pod.is_dirty = True
        
        # Recurse into sub-components (Legs -> Feet -> Toes)
        for sub in pod.sub_pods.values():
            await self.propagate_down(sub, shared_constraints)

    async def converge_up(self, pod: ModularISA):
        """
        Bottom-Up: Aggregating 'Physical Truth' from sub-elements.
        Bubbles up mass/power/thermal accumulation.
        """
        # If not dirty and no children are dirty, we might skip (optimization),
        # but for now we re-sum to be safe.
        
        total_mass = 0.0
        total_power = 0.0
        total_cost = 0.0
        
        for sub in pod.sub_pods.values():
            # Recursively ensure children are converged first
            await self.converge_up(sub)
            # LEGAL ACCESS: Parent reads Child's EXPORTS
            total_mass += sub.exports.get("mass", 0.0)
            total_power += sub.exports.get("power_draw", 0.0)
            total_cost += sub.exports.get("cost", 0.0)
        
        # Add local mass of the 'Chassis' of this pod
        local_mass = pod.constraints.get("local_mass", 0.0)
        local_power = pod.constraints.get("local_power", 0.0)
        local_cost = pod.constraints.get("local_cost", 0.0) # E.g. fasteners unique to this level
        
        # Update Exports
        pod.exports["mass"] = total_mass + local_mass
        pod.exports["power_draw"] = total_power + local_power
        pod.exports["cost"] = total_cost + local_cost
        pod.is_dirty = False
        
        if pod.parent_id:
            print(f"[Ares]: {pod.name} converged. Bubbling {pod.exports['mass']}kg to parent.")

    def validate_access(self, accessor: ModularISA, target: ModularISA, property_type: str) -> bool:
        """
        Enforces the '140 IQ' Encapsulation Rules.
        
        Rules:
        1. Self-Access: Use whatever you want.
        2. Parent-to-Child: Can ONLY read 'exports'. Cannot read 'constraints' (Private State).
        3. Child-to-Parent: Can read 'shared_constraints' (Environmental Context).
        4. Sibling-to-Sibling: PROHIBITED. Must query via Parent (The 'Bus').
        """
        if accessor.id == target.id:
            return True
            
        # Parent accessing Child
        if target.parent_id == accessor.id:
            if property_type == 'exports':
                return True
            if property_type == 'constraints':
                raise PermissionError(f"[ScopeViolation] {accessor.name} cannot access PRIVATE constraints of child {target.name}. Use EXPORTS.")
                
        # Child accessing Parent (Context)
        if accessor.parent_id == target.id:
            if property_type == 'constraints':
                # Actually, children inherit context, they don't read parent's raw memory.
                # But for now, let's allow context reading.
                return True 
                
        # Default Deny
        raise PermissionError(f"[ScopeViolation] {accessor.name} cannot access {target.name}. components are decoupled.")

    def get_pod_by_path(self, path: str) -> Optional[ModularISA]:
        """
        Resolves a path like 'legs/front_left' to a Pod instance.
        """
        parts = path.strip("/").split("/")
        current = self.root
        
        for part in parts:
            if not part: continue
            if part in current.sub_pods:
                current = current.sub_pods[part]
            else:
                return None
        return current
