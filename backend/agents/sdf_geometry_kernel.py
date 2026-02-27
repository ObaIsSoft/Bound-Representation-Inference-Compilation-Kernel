"""
SDF Geometry Kernel - Primary Signed Distance Field Geometry System

Makes SDF the primary representation instead of derived from mesh.
Provides:
- Native SDF primitives (sphere, box, cylinder, cone, torus, etc.)
- SDF boolean operations (union, subtract, intersect, smooth variants)
- SDF transforms (translate, rotate, scale, mirror)
- SDF to mesh conversion (marching cubes)
- Direct GLSL shader code generation for frontend rendering
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from functools import lru_cache

logger = logging.getLogger(__name__)


# =============================================================================
# SDF PRIMITIVE TYPES
# =============================================================================

class SDFPrimitiveType(Enum):
    SPHERE = auto()
    BOX = auto()
    ROUNDED_BOX = auto()
    BOX_FRAME = auto()
    CYLINDER = auto()
    CAPSULE = auto()
    CONE = auto()
    TORUS = auto()
    PLANE = auto()
    ELLIPSOID = auto()
    HEXAGONAL_PRISM = auto()
    TRIANGULAR_PRISM = auto()
    CAPPED_CONE = auto()
    CAPPED_CYLINDER = auto()
    SOLID_ANGLE = auto()
    OCTAHEDRON = auto()
    PYRAMID = auto()


# =============================================================================
# SDF TRANSFORM
# =============================================================================

@dataclass
class SDFTransform:
    """3D transformation for SDF evaluation"""
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation: np.ndarray = field(default_factory=lambda: np.eye(3))  # 3x3 matrix
    scale: float = 1.0
    
    def apply(self, p: np.ndarray) -> np.ndarray:
        """Apply inverse transform to point for SDF evaluation"""
        # Scale
        if self.scale != 1.0:
            p = p / self.scale
        # Rotation (inverse is transpose for orthonormal matrix)
        p = self.rotation.T @ p
        # Translation
        p = p - self.translation
        return p
    
    def transform_distance(self, d: float) -> float:
        """Transform distance back to world space"""
        return d * self.scale
    
    @staticmethod
    def from_euler_angles(tx: float = 0, ty: float = 0, tz: float = 0,
                          rx: float = 0, ry: float = 0, rz: float = 0,
                          scale: float = 1.0) -> 'SDFTransform':
        """Create transform from euler angles (degrees)"""
        # Convert to radians
        rx, ry, rz = np.radians([rx, ry, rz])
        
        # Rotation matrices
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(rx), -np.sin(rx)],
            [0, np.sin(rx), np.cos(rx)]
        ])
        Ry = np.array([
            [np.cos(ry), 0, np.sin(ry)],
            [0, 1, 0],
            [-np.sin(ry), 0, np.cos(ry)]
        ])
        Rz = np.array([
            [np.cos(rz), -np.sin(rz), 0],
            [np.sin(rz), np.cos(rz), 0],
            [0, 0, 1]
        ])
        
        R = Rz @ Ry @ Rx
        
        return SDFTransform(
            translation=np.array([tx, ty, tz]),
            rotation=R,
            scale=scale
        )


# =============================================================================
# SDF NODE BASE CLASS
# =============================================================================

@dataclass
class SDFNode:
    """Base class for SDF geometry nodes"""
    name: str = ""
    transform: SDFTransform = field(default_factory=SDFTransform)
    material_id: str = "default"
    
    def __post_init__(self):
        if not self.name:
            self.name = f"SDF_{id(self)}"
    
    def evaluate(self, p: np.ndarray) -> float:
        """Evaluate SDF at point p (world space)"""
        # Transform to local space
        p_local = self.transform.apply(p)
        # Evaluate primitive
        d_local = self._evaluate_local(p_local)
        # Transform distance back
        return self.transform.transform_distance(d_local)
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        """Override in subclasses - evaluate in local space"""
        raise NotImplementedError
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return bounding box (min, max)"""
        raise NotImplementedError
    
    def to_glsl(self) -> str:
        """Generate GLSL code for this node"""
        raise NotImplementedError


# =============================================================================
# SDF PRIMITIVES
# =============================================================================

@dataclass
class SDFSphere(SDFNode):
    """Sphere primitive"""
    radius: float = 1.0
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        return np.linalg.norm(p) - self.radius
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        r = self.radius * self.transform.scale
        t = self.transform.translation
        return (t - r, t + r)
    
    def to_glsl(self) -> str:
        return f"length(p - vec3({self.transform.translation[0]:.6f}, {self.transform.translation[1]:.6f}, {self.transform.translation[2]:.6f})) - {self.radius:.6f}"


@dataclass
class SDFBox(SDFNode):
    """Box primitive (centered at origin)"""
    size: np.ndarray = field(default_factory=lambda: np.ones(3))
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        q = np.abs(p) - self.size / 2
        return np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0)
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        s = self.size * self.transform.scale / 2
        t = self.transform.translation
        return (t - s, t + s)
    
    def to_glsl(self) -> str:
        tx, ty, tz = self.transform.translation
        sx, sy, sz = self.size / 2
        return f"sdBox(p - vec3({tx:.6f}, {ty:.6f}, {tz:.6f}), vec3({sx:.6f}, {sy:.6f}, {sz:.6f}))"


@dataclass
class SDFRoundedBox(SDFNode):
    """Rounded box primitive"""
    size: np.ndarray = field(default_factory=lambda: np.ones(3))
    radius: float = 0.1
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        q = np.abs(p) - self.size / 2 + self.radius
        return np.linalg.norm(np.maximum(q, 0)) + min(np.max(q), 0) - self.radius
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        s = (self.size / 2 + self.radius) * self.transform.scale
        t = self.transform.translation
        return (t - s, t + s)
    
    def to_glsl(self) -> str:
        tx, ty, tz = self.transform.translation
        sx, sy, sz = self.size / 2 - self.radius
        return f"sdRoundBox(p - vec3({tx:.6f}, {ty:.6f}, {tz:.6f}), vec3({sx:.6f}, {sy:.6f}, {sz:.6f}), {self.radius:.6f})"


@dataclass
class SDFCylinder(SDFNode):
    """Cylinder primitive (oriented along Y axis)"""
    radius: float = 1.0
    height: float = 2.0
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        d = np.array([np.linalg.norm(p[[0, 2]]) - self.radius, abs(p[1]) - self.height / 2])
        return min(np.max(d), 0.0) + np.linalg.norm(np.maximum(d, 0))
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        s = np.array([self.radius, self.height / 2, self.radius]) * self.transform.scale
        t = self.transform.translation
        return (t - s, t + s)
    
    def to_glsl(self) -> str:
        tx, ty, tz = self.transform.translation
        return f"sdCylinder(p - vec3({tx:.6f}, {ty:.6f}, {tz:.6f}), vec2({self.radius:.6f}, {self.height/2:.6f}))"


@dataclass
class SDFCapsule(SDFNode):
    """Capsule primitive (line segment with radius)"""
    a: np.ndarray = field(default_factory=lambda: np.array([0, -0.5, 0]))
    b: np.ndarray = field(default_factory=lambda: np.array([0, 0.5, 0]))
    radius: float = 0.2
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        pa = p - self.a
        ba = self.b - self.a
        h = np.clip(np.dot(pa, ba) / np.dot(ba, ba), 0.0, 1.0)
        return np.linalg.norm(pa - ba * h) - self.radius
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        # Capsule bounds
        a_world = self.transform.rotation @ self.a * self.transform.scale + self.transform.translation
        b_world = self.transform.rotation @ self.b * self.transform.scale + self.transform.translation
        min_pt = np.minimum(a_world, b_world) - self.radius * self.transform.scale
        max_pt = np.maximum(a_world, b_world) + self.radius * self.transform.scale
        return (min_pt, max_pt)
    
    def to_glsl(self) -> str:
        tx, ty, tz = self.transform.translation
        ax, ay, az = self.a
        bx, by, bz = self.b
        return f"sdCapsule(p - vec3({tx:.6f}, {ty:.6f}, {tz:.6f}), vec3({ax:.6f}, {ay:.6f}, {az:.6f}), vec3({bx:.6f}, {by:.6f}, {bz:.6f}), {self.radius:.6f})"


@dataclass
class SDFCone(SDFNode):
    """Cone primitive (oriented along Y axis, apex at top)"""
    angle: float = 0.5  # Radians
    height: float = 2.0
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        # Cone SDF
        c = np.array([np.sin(self.angle), np.cos(self.angle)])
        q = np.array([np.linalg.norm(p[[0, 2]]), p[1]])
        w = np.array([c[1], -c[0]])
        
        # Project onto cone surface
        a = q - c * np.clip(np.dot(q, c), 0.0, self.height)
        b = q - c * np.array([np.dot(q, c) / np.dot(c, c), self.height])
        
        k = np.sign(c[1])
        d = min(np.dot(a, a), np.dot(b, b))
        s = max(k * (q[0] * c[1] - q[1] * c[0]), k * (q[1] - self.height))
        
        return np.sqrt(d) * np.sign(s)
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        r = self.height * np.tan(self.angle) * self.transform.scale
        h = self.height * self.transform.scale
        t = self.transform.translation
        return (t - np.array([r, h/2, r]), t + np.array([r, h/2, r]))
    
    def to_glsl(self) -> str:
        tx, ty, tz = self.transform.translation
        return f"sdCone(p - vec3({tx:.6f}, {ty:.6f}, {tz:.6f}), vec2({np.sin(self.angle):.6f}, {np.cos(self.angle):.6f}), {self.height:.6f})"


@dataclass
class SDFTorus(SDFNode):
    """Torus primitive"""
    major_radius: float = 1.0  # Distance from center to tube center
    minor_radius: float = 0.3  # Tube radius
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        q = np.array([np.linalg.norm(p[[0, 2]]) - self.major_radius, p[1]])
        return np.linalg.norm(q) - self.minor_radius
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        R = (self.major_radius + self.minor_radius) * self.transform.scale
        r = self.minor_radius * self.transform.scale
        t = self.transform.translation
        return (t - np.array([R, r, R]), t + np.array([R, r, R]))
    
    def to_glsl(self) -> str:
        tx, ty, tz = self.transform.translation
        return f"sdTorus(p - vec3({tx:.6f}, {ty:.6f}, {tz:.6f}), vec2({self.major_radius:.6f}, {self.minor_radius:.6f}))"


@dataclass
class SDFPlane(SDFNode):
    """Infinite plane primitive"""
    normal: np.ndarray = field(default_factory=lambda: np.array([0, 1, 0]))
    distance: float = 0.0
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        return np.dot(p, self.normal) - self.distance
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        # Infinite bounds
        return (np.array([-1e10, -1e10, -1e10]), np.array([1e10, 1e10, 1e10]))
    
    def to_glsl(self) -> str:
        nx, ny, nz = self.normal
        return f"dot(p, vec3({nx:.6f}, {ny:.6f}, {nz:.6f})) - {self.distance:.6f}"


# =============================================================================
# SDF BOOLEAN OPERATIONS
# =============================================================================

class SDFBooleanType(Enum):
    UNION = auto()
    SUBTRACT = auto()
    INTERSECT = auto()
    SMOOTH_UNION = auto()
    SMOOTH_SUBTRACT = auto()
    SMOOTH_INTERSECT = auto()


@dataclass
class SDFBoolean(SDFNode):
    """Boolean operation between two SDF nodes"""
    left: SDFNode = None
    right: SDFNode = None
    op_type: SDFBooleanType = SDFBooleanType.UNION
    smoothness: float = 0.1  # For smooth operations
    
    def _evaluate_local(self, p: np.ndarray) -> float:
        d1 = self.left.evaluate(p)
        d2 = self.right.evaluate(p)
        
        if self.op_type == SDFBooleanType.UNION:
            return min(d1, d2)
        elif self.op_type == SDFBooleanType.SUBTRACT:
            return max(-d1, d2)
        elif self.op_type == SDFBooleanType.INTERSECT:
            return max(d1, d2)
        elif self.op_type == SDFBooleanType.SMOOTH_UNION:
            return self._smooth_min(d1, d2, self.smoothness)
        elif self.op_type == SDFBooleanType.SMOOTH_SUBTRACT:
            return self._smooth_max(-d1, d2, self.smoothness)
        elif self.op_type == SDFBooleanType.SMOOTH_INTERSECT:
            return self._smooth_max(d1, d2, self.smoothness)
        
        return min(d1, d2)
    
    def _smooth_min(self, a: float, b: float, k: float) -> float:
        """Polynomial smooth minimum"""
        h = max(k - abs(a - b), 0.0) / k
        return min(a, b) - h * h * k * 0.25
    
    def _smooth_max(self, a: float, b: float, k: float) -> float:
        """Polynomial smooth maximum"""
        h = max(k - abs(a - b), 0.0) / k
        return max(a, b) + h * h * k * 0.25
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        if self.op_type in [SDFBooleanType.UNION, SDFBooleanType.SMOOTH_UNION]:
            # Union bounds = bounds of both
            b1_min, b1_max = self.left.bounds()
            b2_min, b2_max = self.right.bounds()
            return (np.minimum(b1_min, b2_min), np.maximum(b1_max, b2_max))
        elif self.op_type in [SDFBooleanType.INTERSECT, SDFBooleanType.SMOOTH_INTERSECT]:
            # Intersection bounds = overlap
            b1_min, b1_max = self.left.bounds()
            b2_min, b2_max = self.right.bounds()
            return (np.maximum(b1_min, b2_min), np.minimum(b1_max, b2_max))
        else:
            # Subtract bounds = left operand
            return self.left.bounds()
    
    def to_glsl(self) -> str:
        left_code = self.left.to_glsl()
        right_code = self.right.to_glsl()
        
        if self.op_type == SDFBooleanType.UNION:
            return f"min({left_code}, {right_code})"
        elif self.op_type == SDFBooleanType.SUBTRACT:
            return f"max(-({left_code}), {right_code})"
        elif self.op_type == SDFBooleanType.INTERSECT:
            return f"max({left_code}, {right_code})"
        elif self.op_type == SDFBooleanType.SMOOTH_UNION:
            return f"opSmoothUnion({left_code}, {right_code}, {self.smoothness:.6f})"
        elif self.op_type == SDFBooleanType.SMOOTH_SUBTRACT:
            return f"opSmoothSubtraction({left_code}, {right_code}, {self.smoothness:.6f})"
        elif self.op_type == SDFBooleanType.SMOOTH_INTERSECT:
            return f"opSmoothIntersection({left_code}, {right_code}, {self.smoothness:.6f})"
        
        return f"min({left_code}, {right_code})"


# =============================================================================
# SDF SCENE - Collection of nodes
# =============================================================================

class SDFScene:
    """Collection of SDF nodes with operations"""
    
    def __init__(self, name: str = "Scene"):
        self.name = name
        self.root: Optional[SDFNode] = None
        self.nodes: List[SDFNode] = []
    
    def add(self, node: SDFNode) -> 'SDFScene':
        """Add node to scene (union)"""
        if self.root is None:
            self.root = node
        else:
            self.root = SDFBoolean(
                left=self.root,
                right=node,
                op_type=SDFBooleanType.UNION
            )
        self.nodes.append(node)
        return self
    
    def subtract(self, node: SDFNode) -> 'SDFScene':
        """Subtract node from scene"""
        if self.root is None:
            self.root = SDFBox(size=np.zeros(3))  # Empty
        self.root = SDFBoolean(
            left=self.root,
            right=node,
            op_type=SDFBooleanType.SUBTRACT
        )
        self.nodes.append(node)
        return self
    
    def intersect(self, node: SDFNode) -> 'SDFScene':
        """Intersect scene with node"""
        if self.root is None:
            self.root = node
        else:
            self.root = SDFBoolean(
                left=self.root,
                right=node,
                op_type=SDFBooleanType.INTERSECT
            )
        self.nodes.append(node)
        return self
    
    def evaluate(self, p: np.ndarray) -> float:
        """Evaluate entire scene at point"""
        if self.root is None:
            return 1e10  # Far away
        return self.root.evaluate(p)
    
    def evaluate_grid(self, resolution: int = 64, 
                      bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None) -> np.ndarray:
        """Evaluate SDF on a 3D grid"""
        if bounds is None:
            bounds = self.bounds()
        
        min_b, max_b = bounds
        x = np.linspace(min_b[0], max_b[0], resolution)
        y = np.linspace(min_b[1], max_b[1], resolution)
        z = np.linspace(min_b[2], max_b[2], resolution)
        
        grid = np.zeros((resolution, resolution, resolution))
        
        for i, xi in enumerate(x):
            for j, yj in enumerate(y):
                for k, zk in enumerate(z):
                    p = np.array([xi, yj, zk])
                    grid[i, j, k] = self.evaluate(p)
        
        return grid
    
    def bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get scene bounds"""
        if self.root is None:
            return (np.zeros(3), np.ones(3))
        return self.root.bounds()
    
    def to_glsl(self) -> str:
        """Generate complete GLSL scene function"""
        if self.root is None:
            return "return 1e10;"
        
        scene_func = self.root.to_glsl()
        
        return f"""
// SDF Primitive Functions
float sdSphere(vec3 p, float r) {{
    return length(p) - r;
}}

float sdBox(vec3 p, vec3 b) {{
    vec3 d = abs(p) - b;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0));
}}

float sdRoundBox(vec3 p, vec3 b, float r) {{
    vec3 d = abs(p) - b + r;
    return min(max(d.x, max(d.y, d.z)), 0.0) + length(max(d, 0.0)) - r;
}}

float sdCylinder(vec3 p, vec2 h) {{
    vec2 d = abs(vec2(length(p.xz), p.y)) - h;
    return min(max(d.x, d.y), 0.0) + length(max(d, 0.0));
}}

float sdCapsule(vec3 p, vec3 a, vec3 b, float r) {{
    vec3 pa = p - a;
    vec3 ba = b - a;
    float h = clamp(dot(pa, ba) / dot(ba, ba), 0.0, 1.0);
    return length(pa - ba * h) - r;
}}

float sdCone(vec3 p, vec2 c, float h) {{
    float q = length(p.xz);
    return max(dot(c.xy, vec2(q, p.y)), -h - p.y);
}}

float sdTorus(vec3 p, vec2 t) {{
    vec2 q = vec2(length(p.xz) - t.x, p.y);
    return length(q) - t.y;
}}

// Boolean Operations
float opSmoothUnion(float d1, float d2, float k) {{
    float h = clamp(0.5 + 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) - k * h * (1.0 - h);
}}

float opSmoothSubtraction(float d1, float d2, float k) {{
    float h = clamp(0.5 - 0.5 * (d2 + d1) / k, 0.0, 1.0);
    return mix(d2, -d1, h) + k * h * (1.0 - h);
}}

float opSmoothIntersection(float d1, float d2, float k) {{
    float h = clamp(0.5 - 0.5 * (d2 - d1) / k, 0.0, 1.0);
    return mix(d2, d1, h) + k * h * (1.0 - h);
}}

// Scene SDF
float sceneSDF(vec3 p) {{
    return {scene_func};
}}
"""


# =============================================================================
# SDF TO MESH CONVERSION (Marching Cubes)
# =============================================================================

class SDFToMeshConverter:
    """Convert SDF to mesh using marching cubes"""
    
    def __init__(self):
        pass
    
    def convert(self, scene: SDFScene, resolution: int = 128,
                bounds: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                padding: float = 0.1) -> Dict[str, Any]:
        """
        Convert SDF scene to mesh using marching cubes.
        
        Returns:
            Dict with 'vertices', 'faces', 'normals'
        """
        try:
            from skimage import measure
        except ImportError:
            logger.error("scikit-image required for marching cubes")
            return None
        
        # Get bounds with padding
        if bounds is None:
            bounds = scene.bounds()
        
        min_b, max_b = bounds
        extent = max_b - min_b
        min_p = min_b - extent * padding
        max_p = max_b + extent * padding
        
        # Create grid
        x = np.linspace(min_p[0], max_p[0], resolution)
        y = np.linspace(min_p[1], max_p[1], resolution)
        z = np.linspace(min_p[2], max_p[2], resolution)
        
        # Evaluate SDF on grid
        logger.info(f"Evaluating SDF on {resolution}³ grid...")
        grid = np.zeros((resolution, resolution, resolution))
        
        for i in range(resolution):
            for j in range(resolution):
                for k in range(resolution):
                    p = np.array([x[i], y[j], z[k]])
                    grid[i, j, k] = scene.evaluate(p)
        
        # Run marching cubes
        logger.info("Running marching cubes...")
        verts, faces, normals, _ = measure.marching_cubes(
            grid, level=0, spacing=[(max_p[0] - min_p[0]) / resolution] * 3
        )
        
        # Transform vertices to world space
        verts = verts + min_p
        
        return {
            'vertices': verts,
            'faces': faces,
            'normals': normals,
            'vertex_count': len(verts),
            'face_count': len(faces)
        }


# =============================================================================
# HIGH-LEVEL API (Similar to geometry_api.py)
# =============================================================================

class SDFGeometryAPI:
    """High-level API for SDF geometry creation"""
    
    def __init__(self):
        self.scene = SDFScene()
    
    # Primitives
    def sphere(self, radius: float = 1.0, 
               center: Tuple[float, float, float] = (0, 0, 0)) -> SDFSphere:
        """Create sphere"""
        transform = SDFTransform(translation=np.array(center))
        return SDFSphere(radius=radius, transform=transform, name="Sphere")
    
    def box(self, length: float = 1.0, width: float = 1.0, height: float = 1.0,
            center: Tuple[float, float, float] = (0, 0, 0)) -> SDFBox:
        """Create box"""
        transform = SDFTransform(translation=np.array(center))
        return SDFBox(size=np.array([length, width, height]), 
                     transform=transform, name="Box")
    
    def cylinder(self, radius: float = 1.0, height: float = 2.0,
                 center: Tuple[float, float, float] = (0, 0, 0)) -> SDFCylinder:
        """Create cylinder"""
        transform = SDFTransform(translation=np.array(center))
        return SDFCylinder(radius=radius, height=height, 
                          transform=transform, name="Cylinder")
    
    def torus(self, major_radius: float = 1.0, minor_radius: float = 0.3,
              center: Tuple[float, float, float] = (0, 0, 0)) -> SDFTorus:
        """Create torus"""
        transform = SDFTransform(translation=np.array(center))
        return SDFTorus(major_radius=major_radius, minor_radius=minor_radius,
                       transform=transform, name="Torus")
    
    def cone(self, angle: float = 30.0, height: float = 2.0,
             center: Tuple[float, float, float] = (0, 0, 0)) -> SDFCone:
        """Create cone (angle in degrees)"""
        transform = SDFTransform(translation=np.array(center))
        return SDFCone(angle=np.radians(angle), height=height,
                      transform=transform, name="Cone")
    
    # Operations
    def union(self, *nodes: SDFNode) -> SDFNode:
        """Union multiple nodes"""
        if len(nodes) == 0:
            return None
        if len(nodes) == 1:
            return nodes[0]
        
        result = nodes[0]
        for node in nodes[1:]:
            result = SDFBoolean(left=result, right=node, 
                               op_type=SDFBooleanType.UNION)
        return result
    
    def subtract(self, left: SDFNode, right: SDFNode) -> SDFNode:
        """Subtract right from left"""
        return SDFBoolean(left=left, right=right, 
                         op_type=SDFBooleanType.SUBTRACT)
    
    def intersect(self, left: SDFNode, right: SDFNode) -> SDFNode:
        """Intersect left and right"""
        return SDFBoolean(left=left, right=right, 
                         op_type=SDFBooleanType.INTERSECT)
    
    def smooth_union(self, left: SDFNode, right: SDFNode, 
                     k: float = 0.1) -> SDFNode:
        """Smooth union"""
        return SDFBoolean(left=left, right=right, 
                         op_type=SDFBooleanType.SMOOTH_UNION, 
                         smoothness=k)


# =============================================================================
# TEST
# =============================================================================

def test_sdf_kernel():
    """Test the SDF geometry kernel"""
    print("="*70)
    print("SDF GEOMETRY KERNEL TEST")
    print("="*70)
    
    # 1. Primitives
    print("\n1. Testing primitives...")
    api = SDFGeometryAPI()
    
    sphere = api.sphere(radius=1.0, center=(0, 0, 0))
    print(f"   Sphere at origin, r=1.0")
    print(f"   SDF at (0,0,0): {sphere.evaluate(np.array([0, 0, 0])):.3f}")
    print(f"   SDF at (2,0,0): {sphere.evaluate(np.array([2, 0, 0])):.3f}")
    print(f"   SDF at (0.5,0,0): {sphere.evaluate(np.array([0.5, 0, 0])):.3f}")
    
    box = api.box(length=2.0, width=2.0, height=2.0)
    print(f"\n   Box 2x2x2 at origin")
    print(f"   SDF at (0,0,0): {box.evaluate(np.array([0, 0, 0])):.3f}")
    print(f"   SDF at (2,0,0): {box.evaluate(np.array([2, 0, 0])):.3f}")
    
    # 2. Boolean operations
    print("\n2. Testing boolean operations...")
    
    sphere2 = api.sphere(radius=0.8, center=(1, 0, 0))
    union = api.union(sphere, sphere2)
    print(f"   Union of two spheres")
    print(f"   SDF at (0,0,0): {union.evaluate(np.array([0, 0, 0])):.3f}")
    print(f"   SDF at (1,0,0): {union.evaluate(np.array([1, 0, 0])):.3f}")
    
    subtract = api.subtract(sphere, sphere2)
    print(f"\n   Subtract small sphere from large")
    print(f"   SDF at (0,0,0): {subtract.evaluate(np.array([0, 0, 0])):.3f}")
    
    # 3. Scene
    print("\n3. Testing scene...")
    scene = SDFScene("Test")
    scene.add(api.sphere(radius=1.0, center=(-1, 0, 0)))
    scene.add(api.sphere(radius=1.0, center=(1, 0, 0)))
    scene.subtract(api.sphere(radius=0.5, center=(0, 0, 0)))
    
    print(f"   Scene: Two spheres with hole")
    print(f"   SDF at (0,0,0): {scene.evaluate(np.array([0, 0, 0])):.3f}")
    print(f"   SDF at (2,0,0): {scene.evaluate(np.array([2, 0, 0])):.3f}")
    
    # 4. GLSL Generation
    print("\n4. Testing GLSL generation...")
    scene2 = SDFScene("GLSL_Test")
    scene2.add(api.sphere(radius=1.0))
    scene2.add(api.box(length=1.0, width=1.0, height=1.0, center=(1.5, 0, 0)))
    
    glsl = scene2.to_glsl()
    print(f"   Generated GLSL code: {len(glsl)} characters")
    print(f"   (First 200 chars): {glsl[:200]}...")
    
    # 5. Mesh conversion (if skimage available)
    print("\n5. Testing mesh conversion...")
    try:
        from skimage import measure
        converter = SDFToMeshConverter()
        mesh = converter.convert(scene2, resolution=32)
        if mesh:
            print(f"   Mesh vertices: {mesh['vertex_count']}")
            print(f"   Mesh faces: {mesh['face_count']}")
    except ImportError:
        print("   (skimage not available, skipping)")
    
    print("\n" + "="*70)
    print("SDF KERNEL TEST COMPLETE")
    print("="*70)
    
    return True


if __name__ == "__main__":
    test_sdf_kernel()
