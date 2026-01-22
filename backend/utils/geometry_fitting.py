import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import minimize

def fit_stroke_to_primitive(points: List[List[float]]) -> Dict[str, Any]:
    """
    Fits a 3D geometric primitive to a cloud of points (stroke).
    
    Supported Primitives:
    - Cylinder / Capsule (Linear fit)
    
    Algorithm:
    1. PCA to find principal axis (direction).
    2. Project points onto axis to find start/end.
    3. Radius is mean distance from axis.
    """
    pts = np.array(points)
    if len(pts) < 2:
        return None
        
    # 1. Center the data
    center = np.mean(pts, axis=0)
    centered = pts - center
    
    # 2. PCA for direction (SVD)
    # The first principal component is the direction of maximum variance (the stroke line)
    U, S, Vt = np.linalg.svd(centered)
    direction = Vt[0] # Primary axis
    
    # 3. Project points onto this axis
    # projections = dot(point - center, direction)
    projections = np.dot(centered, direction)
    
    min_proj = np.min(projections)
    max_proj = np.max(projections)
    
    # 4. Define Start and End points
    start_point = center + direction * min_proj
    end_point = center + direction * max_proj
    
    # 5. Calculate Radius (Mean distance from line)
    # Distance from line defined by center and direction:
    # d = || (p - center) - ((p - center) . direction) * direction ||
    # We can use the cross product method: d = || (p-a) x n || / ||n|| (where n is unit vec)
    
    # Vector from center to points
    vecs = pts - center
    # Cross product with direction
    cross_prods = np.cross(vecs, direction)
    distances = np.linalg.norm(cross_prods, axis=1)
    
    avg_radius = np.mean(distances)
    
    # Heuristic: If radius is very small (straight line), clamp it
    if avg_radius < 0.05: avg_radius = 0.05
    
    return {
        "type": "capsule", # Capsule is safer for SDFs than raw cylinder
        "start": start_point.tolist(),
        "end": end_point.tolist(),
        "radius": float(avg_radius),
        "length": float(np.linalg.norm(end_point - start_point))
    }
