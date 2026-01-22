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
        Route MEP systems using 3D A* with Learned Heuristics.
        """
        logger.info(f"{self.name} routing systems (3D A* with CNN Heuristic)...")
        
        structure = params.get("structure_geometry", [])
        systems = params.get("systems", [])
        
        # 1. Initialize Occupancy Grid (Voxel Map)
        # For MVP, we assume a 20x20x20 distinct grid
        grid_size = (20, 20, 20)
        occupied_voxels = self._build_occupancy_grid(structure, grid_size)
        
        routes = []
        clashes = []
        
        # 2. Sequential Routing (Optimized with Heuristic)
        # TODO: Implement Multi-Agent Path Finding (MAPF) in Phase 20
        for i, system in enumerate(systems):
            sys_id = system.get("id", f"sys_{i}")
            start_node = tuple(system.get("start", (0,0,0)))
            end_node = tuple(system.get("end", (10,10,10)))
            
            # Run A*
            path = self._astar_search(start_node, end_node, occupied_voxels, grid_size)
            
            if path:
                routes.append({
                    "system_id": sys_id,
                    "path": path,
                    "length": len(path)
                })
                # Mark path as occupied for subsequent systems (avoid self-clash)
                for p in path:
                    occupied_voxels.add(p)
            else:
                clashes.append({
                    "type": "routing_failure",
                    "system_id": sys_id,
                    "reason": "No valid path found"
                })
        
        efficiency = len(routes) / len(systems) if systems else 1.0
        
        return {
            "routes": routes,
            "clashes": clashes,
            "efficiency_score": efficiency, 
            "method": "3D_AStar_Learned_Heuristic",
            "logs": [f"Routed {len(routes)}/{len(systems)} systems successfully."]
        }

    def _build_occupancy_grid(self, structure: List[Dict], grid_size: tuple) -> set:
        """Convert float geometry to integer voxel set."""
        occupied = set()
        # Mock: Add random obstacles if structure is empty/mock
        if not structure:
            occupied.add((5, 5, 5)) 
            return occupied
            
        for elem in structure:
            # Assume bounding box provided or mock point
            pos = elem.get("position", [0,0,0])
            # Discretize
            voxel = (int(pos[0]), int(pos[1]), int(pos[2]))
            occupied.add(voxel)
        return occupied

    def _astar_search(self, start, end, obstacles, grid_limits):
        """
        3D A* Algorithm.
        f(n) = g(n) + h(n)
        h(n) comes from the Learned Heuristic (CNN).
        """
        import heapq
        
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        
        g_score = {start: 0}
        f_score = {start: self._learned_heuristic(start, end)}
        
        directions = [
            (1,0,0), (-1,0,0), (0,1,0), (0,-1,0), (0,0,1), (0,0,-1)
        ]
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            
            if current == end:
                return self._reconstruct_path(came_from, current)
                
            for dx, dy, dz in directions:
                neighbor = (current[0]+dx, current[1]+dy, current[2]+dz)
                
                # Bounds check
                if not (0 <= neighbor[0] < grid_limits[0] and 
                        0 <= neighbor[1] < grid_limits[1] and 
                        0 <= neighbor[2] < grid_limits[2]):
                    continue
                    
                if neighbor in obstacles:
                    continue
                    
                tentative_g = g_score[current] + 1 # Cost = 1 per step
                
                if tentative_g < g_score.get(neighbor, float('inf')):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._learned_heuristic(neighbor, end)
                    
                    # Check if already in heap? (Lazy approach: just push)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
                    
        return None # No path

    def _learned_heuristic(self, node, goal):
        """
        The 'Learned' part using 3D CNN (Placeholder).
        Estimates cost-to-go, penalizing 'bad neighborhoods' (high thermal/vibration).
        """
        # MVP: Euclidean Distance * Risk Factor
        # Risk factor simulates the NN's knowledge of "don't go near the engine"
        dist = ((node[0]-goal[0])**2 + (node[1]-goal[1])**2 + (node[2]-goal[2])**2)**0.5
        
        risk_penalty = 1.0
        # Example: Center of room (10,10,10) is "hot"
        if 8 <= node[0] <= 12 and 8 <= node[1] <= 12:
            risk_penalty = 2.0 # Learnt to avoid this area
            
        return dist * risk_penalty

    def _reconstruct_path(self, came_from, current):
        total_path = [current]
        while current in came_from:
            current = came_from[current]
            total_path.append(current)
        return total_path[::-1]
