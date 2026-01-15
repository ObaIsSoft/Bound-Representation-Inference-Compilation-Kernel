from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class MepAgent:
    """
    MEP Agent - Mechanical, Electrical, Plumbing Systems Routing.
    
    Responsible for:
    - Routing HVAC ducts, pipes, and electrical conduits.
    - Clash detection between systems and structure.
    - Load validation (electrical, hydraulic, thermal).
    - Optimizing routing paths for efficiency and cost.
    """
    
    def __init__(self):
        self.name = "MepAgent"
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Route MEP systems and check for clashes.
        
        Args:
            params: {
                "structure_geometry": List of geometry nodes (walls, floors),
                "systems": List of system requests (e.g., {"type": "hvac", "start": [x,y,z], "end": [x,y,z]}),
                "constraints": Dict of routing constraints
            }
        
        Returns:
            {
                "routes": List of generated routes,
                "clashes": List of detected collisions,
                "efficiency_score": float,
                "logs": List of operation logs
            }
        """
        structure = params.get("structure_geometry", [])
        systems = params.get("systems", [])
        constraints = params.get("constraints", {})
        
        logs = [
            f"[MEP] Processing {len(systems)} system request(s)",
            f"[MEP] Analyzing structural context with {len(structure)} elements"
        ]
        
        routes = []
        clashes = []
        
        # Mock routing logic
        for i, system in enumerate(systems):
            sys_type = system.get("type", "generic")
            start = system.get("start", [0,0,0])
            end = system.get("end", [10,10,10])
            
            # Simple direct path (mock)
            route = {
                "system_id": f"{sys_type}_{i}",
                "path": [start, end],
                "length": self._calculate_distance(start, end)
            }
            routes.append(route)
            
            # Mock clash detection
            if i % 3 == 2: # detected clash mock logic
                clashes.append({
                     "type": "structural_clash",
                     "location": self._midpoint(start, end),
                     "severity": "high",
                     "description": f"System {sys_type}_{i} intersects logical beam"
                })
        
        efficiency = 0.85 if len(systems) > 0 else 1.0
        
        logs.append(f"[MEP] Generated {len(routes)} route(s)")
        if clashes:
            logs.append(f"[MEP] ⚠ Detected {len(clashes)} clash(es)")
        else:
            logs.append(f"[MEP] ✓ No clashes detected")
            
        return {
            "routes": routes,
            "clashes": clashes,
            "efficiency_score": efficiency,
            "logs": logs
        }

    def _calculate_distance(self, p1, p2):
        return ((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)**0.5

    def _midpoint(self, p1, p2):
        return [(p1[0]+p2[0])/2, (p1[1]+p2[1])/2, (p1[2]+p2[2])/2]
