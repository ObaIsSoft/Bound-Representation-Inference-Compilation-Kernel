from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

class TopologicalAgent:
    """
    Topological Agent - Terrain Classification & Mode Recommendation.
    
    Analyzes environment topology to:
    - Classify terrain type
    - Recommend operational mode
    - Identify traversability constraints
    - Suggest vehicle configuration
    """
    
    def __init__(self):
        self.name = "TopologicalAgent"
        self.terrain_types = {
            "flat": {"slope": (0, 5), "roughness": (0, 0.1)},
            "gentle_hills": {"slope": (5, 15), "roughness": (0.1, 0.3)},
            "mountainous": {"slope": (15, 45), "roughness": (0.3, 0.7)},
            "extreme": {"slope": (45, 90), "roughness": (0.7, 1.0)},
        }
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify terrain and recommend mode.
        
        Args:
            params: {
                "elevation_data": Optional heightmap or point cloud,
                "current_position": Optional [x, y, z],
                "scan_radius": Optional float (meters),
                "preferences": Optional Dict (speed/safety trade-off)
            }
        
        Returns:
            {
                "terrain_type": str,
                "recommended_mode": str (AERIAL/GROUND/MARINE),
                "traversability": float (0-1),
                "hazards": List of detected hazards,
                "slope_max": float (degrees),
                "roughness": float (0-1),
                "logs": List of operation logs
            }
        """
        elevation_data = params.get("elevation_data", [])
        current_pos = params.get("current_position", [0, 0, 0])
        scan_radius = params.get("scan_radius", 50.0)
        preferences = params.get("preferences", {})
        
        logs = [
            f"[TOPOLOGICAL] Analyzing terrain around ({current_pos[0]:.1f}, {current_pos[1]:.1f})",
            f"[TOPOLOGICAL] Scan radius: {scan_radius} m"
        ]
        
        # Simplified terrain analysis (real implementation would process heightmap)
        if not elevation_data:
            # Default flat terrain assumption
            slope_max = 2.0
            roughness = 0.05
            terrain_type = "flat"
            logs.append("[TOPOLOGICAL] No elevation data, assuming flat terrain")
        else:
            # Analyze elevation data
            slope_max, roughness = self._analyze_elevation(elevation_data)
            terrain_type = self._classify_terrain(slope_max, roughness)
            logs.append(f"[TOPOLOGICAL] Detected terrain: {terrain_type}")
        
        # Recommend operational mode
        mode_recommendation = self._recommend_mode(
            terrain_type, slope_max, roughness, preferences
        )
        
        # Calculate traversability
        traversability = self._calculate_traversability(
            terrain_type, slope_max, roughness, mode_recommendation["mode"]
       )
        
        # Detect hazards
        hazards = self._detect_hazards(slope_max, roughness, terrain_type)
        
        logs.append(f"[TOPOLOGICAL] Recommended mode: {mode_recommendation['mode']}")
        logs.append(f"[TOPOLOGICAL] Traversability: {traversability:.0%}")
        logs.append(f"[TOPOLOGICAL] Hazards: {len(hazards)}")
        
        return {
            "terrain_type": terrain_type,
            "recommended_mode": mode_recommendation["mode"],
            "mode_rationale": mode_recommendation["rationale"],
            "traversability": traversability,
            "hazards": hazards,
            "slope_max": slope_max,
            "roughness": roughness,
            "logs": logs
        }
    
    def _analyze_elevation(self, elevation_data: List) -> tuple:
        """
        Analyze elevation data to extract slope and roughness.
        
        Returns: (slope_max_degrees, roughness_0_to_1)
        """
        # Stub - real implementation would process actual heightmap
        # For now, return moderate values
        slope_max = 12.0  # degrees
        roughness = 0.25  # 0-1
        return slope_max, roughness
    
    def _classify_terrain(self, slope: float, roughness: float) -> str:
        """Classify terrain based on slope and roughness."""
        for terrain_name, ranges in self.terrain_types.items():
            slope_range = ranges["slope"]
            rough_range = ranges["roughness"]
            
            if (slope_range[0] <= slope <= slope_range[1] and
                rough_range[0] <= roughness <= rough_range[1]):
                return terrain_name
        
        return "extreme"
    
    def _recommend_mode(self, terrain: str, slope: float, roughness: float,
                       preferences: Dict) -> Dict:
        """Recommend operational mode based on terrain."""
        prefer_speed = preferences.get("prefer_speed", False)
        prefer_safety = preferences.get("prefer_safety", True)
        
        # Mode selection logic
        if terrain == "flat":
            mode = "GROUND" if not prefer_speed else "AERIAL"
            rationale = "Flat terrain optimal for ground travel" if mode == "GROUND" else "Aerial mode for speed"
            
        elif terrain == "gentle_hills":
            mode = "GROUND"
            rationale = "Gentle slopes navigable by ground vehicle"
            
        elif terrain == "mountainous":
            mode = "AERIAL"
            rationale = "Steep slopes require aerial navigation"
            
        elif terrain == "extreme":
            mode = "AERIAL"
            rationale = "Extreme terrain impassable by ground"
            
        else:
            mode = "AERIAL"
            rationale = "Default to aerial for unknown terrain"
        
        # Override for marine environments (detected by negative elevation)
        # This would be detected in real elevation data
        
        return {"mode": mode, "rationale": rationale}
    
    def _calculate_traversability(self, terrain: str, slope: float,
                                  roughness: float, mode: str) -> float:
        """
        Calculate traversability score (0-1).
        
        1.0 = Easy
        0.5 = Moderate
        0.0 = Impassable
        """
        if mode == "AERIAL":
            # Aerial mode mostly terrain-independent
            return 0.95 - (roughness * 0.2)  # Slight penalty for turbulence
        
        elif mode == "GROUND":
            # Ground mode heavily dependent on terrain
            slope_penalty = min(slope / 45.0, 1.0)  # 45° = impassable
            rough_penalty = roughness
            
            traversability = 1.0 - (slope_penalty * 0.6 + rough_penalty * 0.4)
            return max(0.0, min(1.0, traversability))
        
        elif mode == "MARINE":
            # Marine mode depends on water depth and currents
            return 0.85  # Placeholder
        
        else:
            return 0.5  # Unknown mode
    
    def _detect_hazards(self, slope: float, roughness: float, terrain: str) -> List[str]:
        """Detect potential hazards in terrain."""
        hazards = []
        
        if slope > 30:
            hazards.append("Steep slopes (rollover risk)")
        if slope > 45:
            hazards.append("Cliff/precipice detected")
        if roughness > 0.6:
            hazards.append("Extremely rough terrain (damage risk)")
        if terrain == "extreme":
            hazards.append("Navigation difficulty: EXTREME")
        
        return hazards
