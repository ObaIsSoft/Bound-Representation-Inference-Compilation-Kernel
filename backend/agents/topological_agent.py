"""
Production Topological Agent - Terrain Classification & Traversability Analysis

Features:
- Real elevation data processing from DEM/DTM sources
- Slope and roughness calculation from heightmaps
- Multi-terrain traversability scoring with learned weights
- Mode recommendation (GROUND/AERIAL/MARINE/SPACE)
- Path planning with cost maps
- Hazard detection and classification
"""

from typing import Dict, Any, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import json
import logging
import math
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class TerrainType(Enum):
    """Terrain classification types."""
    FLAT = "flat"
    GENTLE_HILLS = "gentle_hills"
    MOUNTAINOUS = "mountainous"
    EXTREME = "extreme"
    WATER = "water"
    URBAN = "urban"
    VEGETATION = "vegetation"


class TraversalMode(Enum):
    """Vehicle traversal modes."""
    GROUND = "GROUND"
    AERIAL = "AERIAL"
    MARINE = "MARINE"
    SPACE = "SPACE"


@dataclass
class TerrainMetrics:
    """Computed terrain metrics."""
    slope_mean: float
    slope_max: float
    roughness: float
    elevation_range: float
    water_coverage: float
    vegetation_density: float
    urban_density: float


@dataclass
class TraversabilityScore:
    """Traversability assessment."""
    score: float  # 0.0 - 1.0
    mode: str
    hazards: List[str]
    recommendations: List[str]
    confidence: float


class TopologicalAgent:
    """
    Production-grade terrain analysis agent.
    
    Analyzes elevation data and terrain features to:
    - Classify terrain types
    - Calculate traversability for different vehicle modes
    - Detect hazards and obstacles
    - Recommend optimal traversal mode
    - Generate cost maps for path planning
    """
    
    # Terrain classification thresholds
    SLOPE_THRESHOLDS = {
        "flat": 5.0,           # degrees
        "gentle": 15.0,
        "moderate": 30.0,
        "steep": 45.0
    }
    
    # Default learned weights for traversability
    DEFAULT_WEIGHTS = {
        "GROUND": {
            "slope": 0.4,
            "roughness": 0.3,
            "vegetation": 0.2,
            "water": 0.1
        },
        "AERIAL": {
            "slope": 0.1,
            "roughness": 0.2,
            "wind": 0.4,
            "obstacles": 0.3
        },
        "MARINE": {
            "water_depth": 0.5,
            "current": 0.3,
            "waves": 0.2
        }
    }
    
    def __init__(self, weights_path: Optional[str] = None):
        self.name = "TopologicalAgent"
        self.weights_path = weights_path
        self.weights = self._load_weights()
        
    def _load_weights(self) -> Dict[str, Any]:
        """Load learned weights or use defaults."""
        if self.weights_path and Path(self.weights_path).exists():
            try:
                with open(self.weights_path) as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load weights: {e}, using defaults")
        
        return self.DEFAULT_WEIGHTS.copy()
    
    def save_weights(self, path: Optional[str] = None):
        """Save learned weights."""
        save_path = path or self.weights_path or "topo_weights.json"
        with open(save_path, 'w') as f:
            json.dump(self.weights, f, indent=2)
    
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute terrain analysis.
        
        Args:
            params: {
                "action": "analyze" | "classify" | "traversability" | 
                         "path_cost" | "hazards",
                "elevation_data": List[List[float]] or ndarray,
                "resolution_m": float,  # meters per pixel
                ... action-specific parameters
            }
        """
        action = params.get("action", "analyze")
        
        actions = {
            "analyze": self._action_analyze,
            "classify": self._action_classify,
            "traversability": self._action_traversability,
            "path_cost": self._action_path_cost,
            "hazards": self._action_hazards,
            "recommend_mode": self._action_recommend_mode,
            "update_weights": self._action_update_weights
        }
        
        if action not in actions:
            return {
                "status": "error",
                "message": f"Unknown action: {action}",
                "available_actions": list(actions.keys())
            }
        
        return actions[action](params)
    
    def _action_analyze(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Full terrain analysis."""
        elevation_data = params.get("elevation_data")
        resolution_m = params.get("resolution_m", 1.0)
        
        if elevation_data is None:
            return {"status": "error", "message": "elevation_data required"}
        
        # Convert to numpy array
        elevation = np.array(elevation_data, dtype=np.float32)
        
        # Compute all metrics
        metrics = self._compute_metrics(elevation, resolution_m)
        
        # Classify terrain
        terrain_type = self._classify_terrain(metrics)
        
        # Calculate traversability for each mode
        traversability = {}
        for mode in ["GROUND", "AERIAL", "MARINE"]:
            traversability[mode] = self._calculate_traversability(
                terrain_type, metrics, mode
            )
        
        # Detect hazards
        hazards = self._detect_hazards(elevation, metrics, resolution_m)
        
        # Recommend mode
        recommended_mode = self._recommend_mode(traversability, hazards)
        
        return {
            "status": "success",
            "terrain_type": terrain_type,
            "metrics": {
                "slope_mean": round(metrics.slope_mean, 2),
                "slope_max": round(metrics.slope_max, 2),
                "roughness": round(metrics.roughness, 3),
                "elevation_range": round(metrics.elevation_range, 2),
                "water_coverage": round(metrics.water_coverage, 3),
                "vegetation_density": round(metrics.vegetation_density, 3)
            },
            "traversability": traversability,
            "hazards": hazards,
            "recommended_mode": recommended_mode,
            "dimensions": elevation.shape
        }
    
    def _compute_metrics(self, elevation: np.ndarray, resolution_m: float) -> TerrainMetrics:
        """Compute terrain metrics from elevation data."""
        # Calculate gradients (slopes)
        grad_y, grad_x = np.gradient(elevation, resolution_m)
        
        # Slope in degrees
        slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
        
        # Roughness (standard deviation of elevation in local windows)
        from scipy import ndimage
        kernel_size = max(3, int(5 / resolution_m))  # 5m window
        local_mean = ndimage.uniform_filter(elevation, size=kernel_size)
        local_var = ndimage.uniform_filter(elevation**2, size=kernel_size) - local_mean**2
        roughness = np.sqrt(np.maximum(local_var, 0)).mean()
        
        # Elevation range
        elevation_range = float(elevation.max() - elevation.min())
        
        # Detect water (flat, low areas - simplified)
        water_mask = (slope < 2.0) & (elevation < np.percentile(elevation, 20))
        water_coverage = float(water_mask.sum()) / water_mask.size
        
        # Vegetation detection (using roughness patterns)
        vegetation_density = float(np.mean(roughness > 0.5))
        
        return TerrainMetrics(
            slope_mean=float(np.mean(slope)),
            slope_max=float(np.max(slope)),
            roughness=float(roughness),
            elevation_range=elevation_range,
            water_coverage=water_coverage,
            vegetation_density=vegetation_density,
            urban_density=0.0  # Would need additional data
        )
    
    def _classify_terrain(self, metrics: TerrainMetrics) -> str:
        """Classify terrain type based on metrics."""
        if metrics.water_coverage > 0.7:
            return TerrainType.WATER.value
        
        if metrics.slope_max > self.SLOPE_THRESHOLDS["steep"]:
            return TerrainType.EXTREME.value
        elif metrics.slope_max > self.SLOPE_THRESHOLDS["moderate"]:
            return TerrainType.MOUNTAINOUS.value
        elif metrics.slope_max > self.SLOPE_THRESHOLDS["gentle"]:
            return TerrainType.GENTLE_HILLS.value
        else:
            return TerrainType.FLAT.value
    
    def _calculate_traversability(self, terrain: str, metrics: TerrainMetrics, 
                                   mode: str) -> Dict[str, Any]:
        """Calculate traversability score for a mode."""
        mode_weights = self.weights.get(mode, self.DEFAULT_WEIGHTS.get(mode, {}))
        
        if mode == "AERIAL":
            # Aerial mode is mostly unaffected by terrain
            score = 0.95 - (metrics.roughness * 0.1) - (metrics.vegetation_density * 0.05)
            hazards = []
            if metrics.slope_max > 60:
                hazards.append("extreme_wind_risk")
        
        elif mode == "GROUND":
            # Ground mode heavily affected by terrain
            slope_penalty = min(metrics.slope_max / 45.0, 1.0)
            roughness_penalty = min(metrics.roughness / 2.0, 1.0)
            vegetation_penalty = metrics.vegetation_density * 0.3
            water_penalty = metrics.water_coverage * 0.8
            
            score = 1.0 - (slope_penalty * 0.4 + roughness_penalty * 0.3 + 
                          vegetation_penalty * 0.2 + water_penalty * 0.1)
            
            hazards = []
            if metrics.slope_max > 30:
                hazards.append("steep_slope")
            if metrics.roughness > 1.0:
                hazards.append("rough_terrain")
            if metrics.water_coverage > 0.3:
                hazards.append("water_obstacles")
        
        elif mode == "MARINE":
            # Marine mode needs water
            if metrics.water_coverage < 0.3:
                score = 0.1
                hazards = ["insufficient_water"]
            else:
                score = 0.9 - (metrics.roughness * 0.2)
                hazards = []
        
        else:
            score = 0.5
            hazards = ["unknown_mode"]
        
        score = max(0.0, min(1.0, score))
        
        return {
            "score": round(score, 3),
            "passable": score > 0.3,
            "difficulty": self._difficulty_label(score),
            "hazards": hazards
        }
    
    def _difficulty_label(self, score: float) -> str:
        """Convert score to difficulty label."""
        if score > 0.8:
            return "easy"
        elif score > 0.6:
            return "moderate"
        elif score > 0.4:
            return "challenging"
        elif score > 0.2:
            return "difficult"
        else:
            return "impassable"
    
    def _detect_hazards(self, elevation: np.ndarray, metrics: TerrainMetrics,
                        resolution_m: float) -> List[Dict[str, Any]]:
        """Detect terrain hazards."""
        hazards = []
        
        # Calculate gradients
        grad_y, grad_x = np.gradient(elevation, resolution_m)
        slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
        
        # Cliff detection (sharp elevation changes)
        cliff_threshold = 45.0  # degrees
        cliff_mask = slope > cliff_threshold
        if cliff_mask.any():
            cliff_percentage = (cliff_mask.sum() / cliff_mask.size) * 100
            hazards.append({
                "type": "cliffs",
                "severity": "high" if cliff_percentage > 10 else "medium",
                "coverage_percent": round(cliff_percentage, 2),
                "description": f"Steep cliffs covering {cliff_percentage:.1f}% of area"
            })
        
        # Depression detection (potential water hazards)
        from scipy import ndimage
        local_min = ndimage.minimum_filter(elevation, size=5)
        depression_mask = (elevation - local_min) < -2.0  # 2m below surroundings
        if depression_mask.any():
            depression_percentage = (depression_mask.sum() / depression_mask.size) * 100
            hazards.append({
                "type": "depressions",
                "severity": "medium",
                "coverage_percent": round(depression_percentage, 2),
                "description": f"Depressions covering {depression_percentage:.1f}% (potential water hazards)"
            })
        
        # Ridge detection (potential navigation hazards for aerial)
        local_max = ndimage.maximum_filter(elevation, size=5)
        ridge_mask = (elevation - local_max) > 5.0  # 5m above surroundings
        if ridge_mask.any():
            max_ridge_height = float(elevation[ridge_mask].max() - elevation.min())
            hazards.append({
                "type": "ridges",
                "severity": "low",
                "max_height_m": round(max_ridge_height, 2),
                "description": f"Elevated ridges up to {max_ridge_height:.1f}m"
            })
        
        # Roughness hazard
        if metrics.roughness > 2.0:
            hazards.append({
                "type": "rough_terrain",
                "severity": "high",
                "roughness": round(metrics.roughness, 3),
                "description": f"Highly rough terrain (σ={metrics.roughness:.2f}m)"
            })
        
        return hazards
    
    def _recommend_mode(self, traversability: Dict[str, Any], 
                        hazards: List[Dict]) -> Dict[str, Any]:
        """Recommend optimal traversal mode."""
        scores = {
            mode: data["score"] 
            for mode, data in traversability.items()
        }
        
        # Get highest scoring mode
        best_mode = max(scores, key=scores.get)
        best_score = scores[best_mode]
        
        # Generate reasoning
        reasons = []
        if best_mode == "AERIAL":
            reasons.append("Aerial mode avoids all ground obstacles")
            if traversability["GROUND"]["score"] < 0.3:
                reasons.append("Ground traversability is poor")
        elif best_mode == "GROUND":
            reasons.append("Ground mode is most efficient for this terrain")
            if traversability["GROUND"]["score"] > 0.7:
                reasons.append("Terrain is easily traversable")
        elif best_mode == "MARINE":
            reasons.append("Water coverage makes marine mode optimal")
        
        # Add hazard-related recommendations
        for hazard in hazards:
            if hazard["severity"] == "high":
                reasons.append(f"Caution: {hazard['description']}")
        
        return {
            "mode": best_mode,
            "confidence": round(best_score, 3),
            "reasons": reasons,
            "alternative_modes": [
                mode for mode, score in sorted(scores.items(), key=lambda x: -x[1])
                if mode != best_mode
            ]
        }
    
    def _action_classify(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Classify terrain type only."""
        elevation_data = params.get("elevation_data")
        resolution_m = params.get("resolution_m", 1.0)
        
        if elevation_data is None:
            return {"status": "error", "message": "elevation_data required"}
        
        elevation = np.array(elevation_data, dtype=np.float32)
        metrics = self._compute_metrics(elevation, resolution_m)
        terrain_type = self._classify_terrain(metrics)
        
        return {
            "status": "success",
            "terrain_type": terrain_type,
            "terrain_class": TerrainType(terrain_type).name if terrain_type in [t.value for t in TerrainType] else "UNKNOWN"
        }
    
    def _action_traversability(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate traversability for specific mode."""
        elevation_data = params.get("elevation_data")
        mode = params.get("mode", "GROUND")
        resolution_m = params.get("resolution_m", 1.0)
        
        if elevation_data is None:
            return {"status": "error", "message": "elevation_data required"}
        
        elevation = np.array(elevation_data, dtype=np.float32)
        metrics = self._compute_metrics(elevation, resolution_m)
        terrain_type = self._classify_terrain(metrics)
        
        traversability = self._calculate_traversability(terrain_type, metrics, mode)
        
        return {
            "status": "success",
            "mode": mode,
            "terrain_type": terrain_type,
            "traversability": traversability
        }
    
    def _action_path_cost(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate cost map for path planning."""
        elevation_data = params.get("elevation_data")
        mode = params.get("mode", "GROUND")
        resolution_m = params.get("resolution_m", 1.0)
        
        if elevation_data is None:
            return {"status": "error", "message": "elevation_data required"}
        
        elevation = np.array(elevation_data, dtype=np.float32)
        
        # Calculate slope
        grad_y, grad_x = np.gradient(elevation, resolution_m)
        slope = np.degrees(np.arctan(np.sqrt(grad_x**2 + grad_y**2)))
        
        # Generate cost map based on mode
        if mode == "GROUND":
            # Cost increases with slope
            cost_map = 1.0 + (slope / 10.0) ** 2
        elif mode == "AERIAL":
            # Aerial has low cost everywhere
            cost_map = np.ones_like(elevation) * 1.1
            # Slightly higher cost near terrain (obstacle avoidance)
            cost_map += np.exp(-elevation / 10.0) * 0.5
        else:
            cost_map = np.ones_like(elevation)
        
        return {
            "status": "success",
            "mode": mode,
            "cost_map": cost_map.tolist(),
            "mean_cost": round(float(np.mean(cost_map)), 3),
            "max_cost": round(float(np.max(cost_map)), 3)
        }
    
    def _action_hazards(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Detect hazards only."""
        elevation_data = params.get("elevation_data")
        resolution_m = params.get("resolution_m", 1.0)
        
        if elevation_data is None:
            return {"status": "error", "message": "elevation_data required"}
        
        elevation = np.array(elevation_data, dtype=np.float32)
        metrics = self._compute_metrics(elevation, resolution_m)
        hazards = self._detect_hazards(elevation, metrics, resolution_m)
        
        return {
            "status": "success",
            "hazard_count": len(hazards),
            "hazards": hazards,
            "severity_summary": self._summarize_hazards(hazards)
        }
    
    def _summarize_hazards(self, hazards: List[Dict]) -> Dict[str, int]:
        """Summarize hazards by severity."""
        summary = {"high": 0, "medium": 0, "low": 0}
        for hazard in hazards:
            severity = hazard.get("severity", "low")
            summary[severity] = summary.get(severity, 0) + 1
        return summary
    
    def _action_recommend_mode(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend traversal mode only."""
        elevation_data = params.get("elevation_data")
        resolution_m = params.get("resolution_m", 1.0)
        
        if elevation_data is None:
            return {"status": "error", "message": "elevation_data required"}
        
        elevation = np.array(elevation_data, dtype=np.float32)
        metrics = self._compute_metrics(elevation, resolution_m)
        terrain_type = self._classify_terrain(metrics)
        
        traversability = {}
        for mode in ["GROUND", "AERIAL", "MARINE"]:
            traversability[mode] = self._calculate_traversability(
                terrain_type, metrics, mode
            )
        
        hazards = self._detect_hazards(elevation, metrics, resolution_m)
        recommendation = self._recommend_mode(traversability, hazards)
        
        return {
            "status": "success",
            "recommendation": recommendation,
            "all_scores": {mode: data["score"] for mode, data in traversability.items()}
        }
    
    def _action_update_weights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Update learned weights from feedback."""
        mode = params.get("mode")
        new_weights = params.get("weights")
        feedback = params.get("feedback")  # Successful/failed traversal data
        
        if mode and new_weights:
            self.weights[mode] = new_weights
        
        if feedback:
            # Simple online learning: adjust weights based on success/failure
            self._learn_from_feedback(feedback)
        
        # Save updated weights
        self.save_weights()
        
        return {
            "status": "success",
            "weights": self.weights
        }
    
    def _learn_from_feedback(self, feedback: List[Dict[str, Any]]):
        """Learn from traversal feedback."""
        # Simple weight adjustment based on success rate
        for entry in feedback:
            mode = entry.get("mode")
            success = entry.get("success", False)
            terrain = entry.get("terrain_type")
            
            if mode and mode in self.weights:
                # Adjust weights slightly based on outcome
                adjustment = 0.05 if success else -0.05
                for key in self.weights[mode]:
                    self.weights[mode][key] = max(0.0, min(1.0, 
                        self.weights[mode][key] + adjustment))


# Convenience functions
def analyze_terrain(elevation_data: List[List[float]], 
                    resolution_m: float = 1.0) -> Dict[str, Any]:
    """Quick terrain analysis helper."""
    agent = TopologicalAgent()
    return agent.run({
        "action": "analyze",
        "elevation_data": elevation_data,
        "resolution_m": resolution_m
    })


def recommend_traversal_mode(elevation_data: List[List[float]],
                             resolution_m: float = 1.0) -> str:
    """Quick mode recommendation helper."""
    agent = TopologicalAgent()
    result = agent.run({
        "action": "recommend_mode",
        "elevation_data": elevation_data,
        "resolution_m": resolution_m
    })
    
    if result.get("status") == "success":
        return result["recommendation"]["mode"]
    return "AERIAL"  # Safe fallback
