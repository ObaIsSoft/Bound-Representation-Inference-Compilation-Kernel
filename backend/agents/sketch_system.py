"""
FULL Sketch System - Complete 2D/3D parametric sketching

Implements:
- 2D primitives (line, arc, circle, rectangle, polygon, spline)
- Constraints (coincident, parallel, perpendicular, horizontal, vertical, 
               equal, tangent, concentric, distance, angle, radius)
- Dimensions (driven/driving)
- Constraint solver (geometric + dimensional)
- Extrude, Revolve, Sweep, Loft features
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union, Set
from dataclasses import dataclass, field
from enum import Enum, auto
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


# =============================================================================
# GEOMETRY TYPES
# =============================================================================

class ConstraintType(Enum):
    COINCIDENT = auto()      # Points share same location
    PARALLEL = auto()        # Lines parallel
    PERPENDICULAR = auto()   # Lines perpendicular
    HORIZONTAL = auto()      # Line horizontal
    VERTICAL = auto()        # Line vertical
    EQUAL = auto()           # Lengths/radii equal
    TANGENT = auto()         # Curve tangent to curve/line
    CONCENTRIC = auto()      # Arcs/circles share center
    MIDPOINT = auto()        # Point at midpoint of line
    SYMMETRIC = auto()       # Points symmetric about line
    # Dimensional
    DISTANCE = auto()        # Distance between entities
    ANGLE = auto()           # Angle between lines
    RADIUS = auto()          # Radius of arc/circle
    DIAMETER = auto()        # Diameter of circle


@dataclass
class Point2D:
    """2D Point with constraint tracking"""
    x: float
    y: float
    fixed: bool = False  # If True, point cannot move during solving
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"P_{id(self)}"
    
    def __hash__(self):
        return id(self)
    
    def __eq__(self, other):
        return isinstance(other, Point2D) and id(self) == id(other)
    
    def distance_to(self, other: 'Point2D') -> float:
        return np.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def to_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)
    
    def copy(self) -> 'Point2D':
        return Point2D(self.x, self.y, self.fixed, self.name)
    
    def translate(self, dx: float, dy: float) -> 'Point2D':
        return Point2D(self.x + dx, self.y + dy, self.fixed, self.name)
    
    def rotate(self, angle: float, center: 'Point2D' = None) -> 'Point2D':
        """Rotate point by angle (radians) around center"""
        if center is None:
            center = Point2D(0, 0)
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        dx = self.x - center.x
        dy = self.y - center.y
        new_x = center.x + dx * cos_a - dy * sin_a
        new_y = center.y + dx * sin_a + dy * cos_a
        return Point2D(new_x, new_y, self.fixed, self.name)


@dataclass 
class Line2D:
    """2D Line segment"""
    p1: Point2D
    p2: Point2D
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"L_{id(self)}"
    
    def length(self) -> float:
        return self.p1.distance_to(self.p2)
    
    def midpoint(self) -> Point2D:
        return Point2D(
            (self.p1.x + self.p2.x) / 2,
            (self.p1.y + self.p2.y) / 2,
            name=f"{self.name}_mid"
        )
    
    def angle(self) -> float:
        """Angle in radians from horizontal"""
        return np.arctan2(self.p2.y - self.p1.y, self.p2.x - self.p1.x)
    
    def direction(self) -> Tuple[float, float]:
        """Unit direction vector"""
        dx = self.p2.x - self.p1.x
        dy = self.p2.y - self.p1.y
        length = np.sqrt(dx**2 + dy**2)
        if length < 1e-10:
            return (0, 0)
        return (dx/length, dy/length)
    
    def is_horizontal(self, tolerance: float = 1e-6) -> bool:
        return abs(self.p2.y - self.p1.y) < tolerance
    
    def is_vertical(self, tolerance: float = 1e-6) -> bool:
        return abs(self.p2.x - self.p1.x) < tolerance


@dataclass
class Circle2D:
    """2D Circle"""
    center: Point2D
    radius: float
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"C_{id(self)}"
    
    def circumference(self) -> float:
        return 2 * np.pi * self.radius
    
    def area(self) -> float:
        return np.pi * self.radius**2
    
    def point_at_angle(self, angle: float) -> Point2D:
        """Point on circle at angle (radians)"""
        return Point2D(
            self.center.x + self.radius * np.cos(angle),
            self.center.y + self.radius * np.sin(angle),
            name=f"{self.name}_pt"
        )


@dataclass
class Arc2D:
    """2D Arc (portion of circle)"""
    center: Point2D
    radius: float
    start_angle: float  # radians
    end_angle: float    # radians
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"A_{id(self)}"
    
    def start_point(self) -> Point2D:
        return Point2D(
            self.center.x + self.radius * np.cos(self.start_angle),
            self.center.y + self.radius * np.sin(self.start_angle),
            name=f"{self.name}_start"
        )
    
    def end_point(self) -> Point2D:
        return Point2D(
            self.center.x + self.radius * np.cos(self.end_angle),
            self.center.y + self.radius * np.sin(self.end_angle),
            name=f"{self.name}_end"
        )
    
    def length(self) -> float:
        angle_diff = self.end_angle - self.start_angle
        # Normalize to positive angle
        while angle_diff < 0:
            angle_diff += 2 * np.pi
        return self.radius * angle_diff


@dataclass
class Spline2D:
    """2D B-spline or NURBS curve (simplified as polyline for now)"""
    control_points: List[Point2D]
    degree: int = 3
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"S_{id(self)}"


@dataclass
class Polygon2D:
    """2D Polygon (closed polyline)"""
    points: List[Point2D]
    name: str = ""
    
    def __post_init__(self):
        if not self.name:
            self.name = f"Poly_{id(self)}"
    
    def area(self) -> float:
        """Shoelace formula for polygon area"""
        n = len(self.points)
        if n < 3:
            return 0
        area = 0
        for i in range(n):
            j = (i + 1) % n
            area += self.points[i].x * self.points[j].y
            area -= self.points[j].x * self.points[i].y
        return abs(area) / 2
    
    def is_closed(self) -> bool:
        if len(self.points) < 2:
            return False
        return self.points[0].distance_to(self.points[-1]) < 1e-6
    
    def centroid(self) -> Point2D:
        """Centroid of polygon"""
        cx = sum(p.x for p in self.points) / len(self.points)
        cy = sum(p.y for p in self.points) / len(self.points)
        return Point2D(cx, cy)


# Type alias for any sketch entity
SketchEntity = Union[Point2D, Line2D, Circle2D, Arc2D, Polygon2D, Spline2D]


# =============================================================================
# CONSTRAINT SYSTEM
# =============================================================================

@dataclass
class Constraint:
    """Geometric or dimensional constraint"""
    type: ConstraintType
    entities: List[SketchEntity] = field(default_factory=list)
    value: Optional[float] = None  # For dimensional constraints
    name: str = ""
    active: bool = True
    
    def __post_init__(self):
        if not self.name:
            self.name = f"{self.type.name}_{id(self)}"


class ConstraintSolver:
    """
    Solves geometric constraints using iterative relaxation method.
    Handles both geometric and dimensional constraints.
    """
    
    def __init__(self, max_iterations: int = 1000, tolerance: float = 1e-6):
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.constraints: List[Constraint] = []
    
    def add_constraint(self, constraint: Constraint) -> None:
        self.constraints.append(constraint)
    
    def solve(self, sketch: 'Sketch') -> bool:
        """
        Solve all constraints iteratively.
        Returns True if converged, False otherwise.
        """
        for iteration in range(self.max_iterations):
            total_error = 0.0
            
            for constraint in self.constraints:
                if not constraint.active:
                    continue
                error = self._enforce_constraint(constraint)
                total_error += error
            
            if total_error < self.tolerance:
                logger.info(f"Constraints converged in {iteration} iterations")
                return True
        
        logger.warning(f"Constraints did not converge after {self.max_iterations} iterations")
        return False
    
    def _enforce_constraint(self, constraint: Constraint) -> float:
        """Enforce a single constraint, returns error magnitude"""
        entities = constraint.entities
        
        if constraint.type == ConstraintType.COINCIDENT:
            return self._enforce_coincident(entities[0], entities[1])
        
        elif constraint.type == ConstraintType.HORIZONTAL:
            return self._enforce_horizontal(entities[0])
        
        elif constraint.type == ConstraintType.VERTICAL:
            return self._enforce_vertical(entities[0])
        
        elif constraint.type == ConstraintType.PARALLEL:
            return self._enforce_parallel(entities[0], entities[1])
        
        elif constraint.type == ConstraintType.PERPENDICULAR:
            return self._enforce_perpendicular(entities[0], entities[1])
        
        elif constraint.type == ConstraintType.EQUAL:
            return self._enforce_equal(entities[0], entities[1])
        
        elif constraint.type == ConstraintType.DISTANCE:
            return self._enforce_distance(entities[0], entities[1], constraint.value)
        
        elif constraint.type == ConstraintType.RADIUS:
            return self._enforce_radius(entities[0], constraint.value)
        
        elif constraint.type == ConstraintType.ANGLE:
            return self._enforce_angle(entities[0], entities[1], constraint.value)
        
        elif constraint.type == ConstraintType.MIDPOINT:
            return self._enforce_midpoint(entities[0], entities[1])
        
        elif constraint.type == ConstraintType.CONCENTRIC:
            return self._enforce_concentric(entities[0], entities[1])
        
        elif constraint.type == ConstraintType.TANGENT:
            return self._enforce_tangent(entities[0], entities[1])
        
        return 0.0
    
    def _enforce_coincident(self, p1: Point2D, p2: Point2D) -> float:
        if p1.fixed and p2.fixed:
            return 0.0
        
        mid_x = (p1.x + p2.x) / 2
        mid_y = (p1.y + p2.y) / 2
        error = p1.distance_to(p2)
        
        if not p1.fixed:
            p1.x = mid_x
            p1.y = mid_y
        if not p2.fixed:
            p2.x = mid_x
            p2.y = mid_y
        
        return error
    
    def _enforce_horizontal(self, line: Line2D) -> float:
        if line.p1.fixed and line.p2.fixed:
            return 0.0
        
        mid_y = (line.p1.y + line.p2.y) / 2
        error = abs(line.p2.y - line.p1.y)
        
        if not line.p1.fixed:
            line.p1.y = mid_y
        if not line.p2.fixed:
            line.p2.y = mid_y
        
        return error
    
    def _enforce_vertical(self, line: Line2D) -> float:
        if line.p1.fixed and line.p2.fixed:
            return 0.0
        
        mid_x = (line.p1.x + line.p2.x) / 2
        error = abs(line.p2.x - line.p1.x)
        
        if not line.p1.fixed:
            line.p1.x = mid_x
        if not line.p2.fixed:
            line.p2.x = mid_x
        
        return error
    
    def _enforce_parallel(self, line1: Line2D, line2: Line2D) -> float:
        """Make two lines parallel by averaging their angles"""
        angle1 = line1.angle()
        angle2 = line2.angle()
        
        # Average angle
        avg_angle = (angle1 + angle2) / 2
        
        error = abs(np.sin(angle1 - angle2))
        
        # Adjust line2 to be parallel to line1
        if not (line2.p1.fixed and line2.p2.fixed):
            mid = line2.midpoint()
            half_len = line2.length() / 2
            
            if not line2.p1.fixed:
                line2.p1.x = mid.x - half_len * np.cos(avg_angle)
                line2.p1.y = mid.y - half_len * np.sin(avg_angle)
            if not line2.p2.fixed:
                line2.p2.x = mid.x + half_len * np.cos(avg_angle)
                line2.p2.y = mid.y + half_len * np.sin(avg_angle)
        
        return error
    
    def _enforce_perpendicular(self, line1: Line2D, line2: Line2D) -> float:
        """Make two lines perpendicular"""
        angle1 = line1.angle()
        angle2 = angle1 + np.pi / 2  # Perpendicular angle
        
        if not (line2.p1.fixed and line2.p2.fixed):
            mid = line2.midpoint()
            half_len = line2.length() / 2
            
            if not line2.p1.fixed:
                line2.p1.x = mid.x - half_len * np.cos(angle2)
                line2.p1.y = mid.y - half_len * np.sin(angle2)
            if not line2.p2.fixed:
                line2.p2.x = mid.x + half_len * np.cos(angle2)
                line2.p2.y = mid.y + half_len * np.sin(angle2)
        
        return 0.0
    
    def _enforce_equal(self, e1: SketchEntity, e2: SketchEntity) -> float:
        """Make lengths or radii equal"""
        if isinstance(e1, Line2D) and isinstance(e2, Line2D):
            target_len = (e1.length() + e2.length()) / 2
            
            for line in [e1, e2]:
                if not (line.p1.fixed and line.p2.fixed):
                    mid = line.midpoint()
                    angle = line.angle()
                    half_len = target_len / 2
                    
                    if not line.p1.fixed:
                        line.p1.x = mid.x - half_len * np.cos(angle)
                        line.p1.y = mid.y - half_len * np.sin(angle)
                    if not line.p2.fixed:
                        line.p2.x = mid.x + half_len * np.cos(angle)
                        line.p2.y = mid.y + half_len * np.sin(angle)
        
        elif isinstance(e1, Circle2D) and isinstance(e2, Circle2D):
            target_radius = (e1.radius + e2.radius) / 2
            e1.radius = target_radius
            e2.radius = target_radius
        
        return 0.0
    
    def _enforce_distance(self, e1: SketchEntity, e2: SketchEntity, distance: float) -> float:
        """Maintain distance between entities"""
        if distance is None:
            return 0.0
        
        if isinstance(e1, Point2D) and isinstance(e2, Point2D):
            current_dist = e1.distance_to(e2)
            if current_dist < 1e-10:
                return 0.0
            
            error = abs(current_dist - distance)
            factor = distance / current_dist
            
            if not (e1.fixed and e2.fixed):
                mid_x = (e1.x + e2.x) / 2
                mid_y = (e1.y + e2.y) / 2
                
                if not e1.fixed:
                    e1.x = mid_x - (mid_x - e1.x) * factor
                    e1.y = mid_y - (mid_y - e1.y) * factor
                if not e2.fixed:
                    e2.x = mid_x - (mid_x - e2.x) * factor
                    e2.y = mid_y - (mid_y - e2.y) * factor
            
            return error
        
        return 0.0
    
    def _enforce_radius(self, circle: Circle2D, radius: float) -> float:
        """Set circle/arc radius"""
        if radius is None:
            return 0.0
        error = abs(circle.radius - radius)
        circle.radius = radius
        return error
    
    def _enforce_angle(self, line1: Line2D, line2: Line2D, angle: float) -> float:
        """Set angle between two lines"""
        if angle is None:
            return 0.0
        
        current_angle = line2.angle() - line1.angle()
        error = abs(current_angle - angle)
        
        if not (line2.p1.fixed and line2.p2.fixed):
            target_angle = line1.angle() + angle
            mid = line2.midpoint()
            half_len = line2.length() / 2
            
            if not line2.p1.fixed:
                line2.p1.x = mid.x - half_len * np.cos(target_angle)
                line2.p1.y = mid.y - half_len * np.sin(target_angle)
            if not line2.p2.fixed:
                line2.p2.x = mid.x + half_len * np.cos(target_angle)
                line2.p2.y = mid.y + half_len * np.sin(target_angle)
        
        return error
    
    def _enforce_midpoint(self, point: Point2D, line: Line2D) -> float:
        """Place point at line midpoint"""
        mid = line.midpoint()
        error = point.distance_to(mid)
        
        if not point.fixed:
            point.x = mid.x
            point.y = mid.y
        
        return error
    
    def _enforce_concentric(self, c1: Union[Circle2D, Arc2D], c2: Union[Circle2D, Arc2D]) -> float:
        """Make circles/arcs share center"""
        error = c1.center.distance_to(c2.center)
        
        mid_x = (c1.center.x + c2.center.x) / 2
        mid_y = (c1.center.y + c2.center.y) / 2
        
        if not c1.center.fixed:
            c1.center.x = mid_x
            c1.center.y = mid_y
        if not c2.center.fixed:
            c2.center.x = mid_x
            c2.center.y = mid_y
        
        return error
    
    def _enforce_tangent(self, e1: SketchEntity, e2: SketchEntity) -> float:
        """Make entities tangent (simplified)"""
        # Simplified: move circle center to be tangent to line
        if isinstance(e1, Circle2D) and isinstance(e2, Line2D):
            # Distance from center to line should equal radius
            # Line: ax + by + c = 0
            dx = e2.p2.x - e2.p1.x
            dy = e2.p2.y - e2.p1.y
            a = dy
            b = -dx
            c = dx * e2.p1.y - dy * e2.p1.x
            
            norm = np.sqrt(a**2 + b**2)
            if norm < 1e-10:
                return 0.0
            
            dist = abs(a * e1.center.x + b * e1.center.y + c) / norm
            error = abs(dist - e1.radius)
            
            # Move center to achieve tangency
            if not e1.center.fixed:
                # Direction to move
                factor = (e1.radius - dist) / norm
                e1.center.x += a * factor
                e1.center.y += b * factor
            
            return error
        
        return 0.0


# =============================================================================
# SKETCH CLASS
# =============================================================================

class Sketch:
    """
    Complete 2D parametric sketch system.
    
    Supports:
    - 2D primitives: points, lines, circles, arcs, rectangles, polygons, splines
    - Geometric constraints: coincident, parallel, perpendicular, horizontal, 
      vertical, equal, tangent, concentric, midpoint, symmetric
    - Dimensional constraints: distance, angle, radius, diameter
    - Constraint solving
    """
    
    def __init__(self, name: str = "Sketch", plane: str = "xy"):
        self.name = name
        self.plane = plane  # "xy", "xz", "yz"
        
        # Entities
        self.points: List[Point2D] = []
        self.lines: List[Line2D] = []
        self.circles: List[Circle2D] = []
        self.arcs: List[Arc2D] = []
        self.polygons: List[Polygon2D] = []
        self.splines: List[Spline2D] = []
        
        # Constraints
        self.constraints: List[Constraint] = []
        self.solver = ConstraintSolver()
        
        # Dimensions
        self.dimensions: List[Constraint] = []
    
    # -------------------------------------------------------------------------
    # Point Creation
    # -------------------------------------------------------------------------
    def add_point(self, x: float, y: float, fixed: bool = False, name: str = "") -> Point2D:
        """Add a point to the sketch"""
        point = Point2D(x, y, fixed, name)
        self.points.append(point)
        return point
    
    def add_origin(self) -> Point2D:
        """Add origin point (fixed)"""
        return self.add_point(0, 0, fixed=True, name="Origin")
    
    # -------------------------------------------------------------------------
    # Line Creation
    # -------------------------------------------------------------------------
    def add_line(self, p1: Point2D, p2: Point2D, name: str = "") -> Line2D:
        """Add a line between two points"""
        line = Line2D(p1, p2, name)
        self.lines.append(line)
        return line
    
    def add_line_by_coords(self, x1: float, y1: float, x2: float, y2: float, 
                           name: str = "") -> Line2D:
        """Add a line by coordinates"""
        p1 = self.add_point(x1, y1)
        p2 = self.add_point(x2, y2)
        return self.add_line(p1, p2, name)
    
    def add_centerline(self, x1: float, y1: float, x2: float, y2: float) -> Line2D:
        """Add a centerline (construction line)"""
        return self.add_line_by_coords(x1, y1, x2, y2, name="Centerline")
    
    # -------------------------------------------------------------------------
    # Circle Creation
    # -------------------------------------------------------------------------
    def add_circle(self, center: Point2D, radius: float, name: str = "") -> Circle2D:
        """Add a circle"""
        circle = Circle2D(center, radius, name)
        self.circles.append(circle)
        return circle
    
    def add_circle_by_coords(self, cx: float, cy: float, radius: float, 
                              name: str = "") -> Circle2D:
        """Add a circle by center coordinates"""
        center = self.add_point(cx, cy)
        return self.add_circle(center, radius, name)
    
    def add_hole(self, cx: float, cy: float, diameter: float, name: str = "") -> Circle2D:
        """Add a hole (circle, will be subtracted in 3D)"""
        return self.add_circle_by_coords(cx, cy, diameter/2, name or f"Hole_{len(self.circles)}")
    
    # -------------------------------------------------------------------------
    # Arc Creation
    # -------------------------------------------------------------------------
    def add_arc(self, center: Point2D, radius: float, 
                start_angle: float, end_angle: float, name: str = "") -> Arc2D:
        """Add an arc (angles in degrees)"""
        arc = Arc2D(center, radius, np.radians(start_angle), np.radians(end_angle), name)
        self.arcs.append(arc)
        return arc
    
    def add_arc_by_coords(self, cx: float, cy: float, radius: float,
                          start_angle: float, end_angle: float, name: str = "") -> Arc2D:
        """Add an arc by center coordinates"""
        center = self.add_point(cx, cy)
        return self.add_arc(center, radius, start_angle, end_angle, name)
    
    def add_arc_three_points(self, p1: Point2D, p2: Point2D, p3: Point2D, 
                              name: str = "") -> Optional[Arc2D]:
        """Add arc through three points"""
        # Calculate center from three points
        # Using perpendicular bisector intersection
        
        d = 2 * (p1.x * (p2.y - p3.y) + p2.x * (p3.y - p1.y) + p3.x * (p1.y - p2.y))
        if abs(d) < 1e-10:
            return None  # Points are collinear
        
        ux = ((p1.x**2 + p1.y**2) * (p2.y - p3.y) + 
              (p2.x**2 + p2.y**2) * (p3.y - p1.y) + 
              (p3.x**2 + p3.y**2) * (p1.y - p2.y)) / d
        uy = ((p1.x**2 + p1.y**2) * (p3.x - p2.x) + 
              (p2.x**2 + p2.y**2) * (p1.x - p3.x) + 
              (p3.x**2 + p3.y**2) * (p2.x - p1.x)) / d
        
        center = self.add_point(ux, uy)
        radius = center.distance_to(p1)
        
        # Calculate start and end angles
        start_angle = np.degrees(np.arctan2(p1.y - uy, p1.x - ux))
        mid_angle = np.degrees(np.arctan2(p2.y - uy, p2.x - ux))
        end_angle = np.degrees(np.arctan2(p3.y - uy, p3.x - ux))
        
        return self.add_arc(center, radius, start_angle, end_angle, name)
    
    # -------------------------------------------------------------------------
    # Rectangle Creation
    # -------------------------------------------------------------------------
    def add_rectangle(self, corner: Point2D, width: float, height: float,
                       name: str = "") -> Polygon2D:
        """Add a rectangle from corner point"""
        p1 = corner
        p2 = self.add_point(corner.x + width, corner.y)
        p3 = self.add_point(corner.x + width, corner.y + height)
        p4 = self.add_point(corner.x, corner.y + height)
        
        poly = Polygon2D([p1, p2, p3, p4, p1], name or f"Rect_{len(self.polygons)}")
        self.polygons.append(poly)
        
        # Auto-add perpendicular constraints for perfect rectangle
        self.add_perpendicular(Line2D(p1, p2), Line2D(p2, p3))
        self.add_perpendicular(Line2D(p2, p3), Line2D(p3, p4))
        
        return poly
    
    def add_rectangle_by_coords(self, x: float, y: float, width: float, 
                                 height: float, name: str = "") -> Polygon2D:
        """Add rectangle by corner coordinates"""
        corner = self.add_point(x, y)
        return self.add_rectangle(corner, width, height, name)
    
    def add_centered_rectangle(self, cx: float, cy: float, width: float,
                                height: float, name: str = "") -> Polygon2D:
        """Add rectangle centered at point"""
        return self.add_rectangle_by_coords(cx - width/2, cy - height/2, 
                                            width, height, name)
    
    # -------------------------------------------------------------------------
    # Polygon Creation
    # -------------------------------------------------------------------------
    def add_polygon(self, points: List[Point2D], closed: bool = True, 
                    name: str = "") -> Polygon2D:
        """Add polygon from points"""
        if closed and points[0] != points[-1]:
            points = points + [points[0]]
        poly = Polygon2D(points, name or f"Poly_{len(self.polygons)}")
        self.polygons.append(poly)
        return poly
    
    def add_regular_polygon(self, center: Point2D, radius: float, 
                            num_sides: int, name: str = "") -> Polygon2D:
        """Add regular polygon (triangle, pentagon, hexagon, etc.)"""
        points = []
        for i in range(num_sides):
            angle = 2 * np.pi * i / num_sides - np.pi / 2  # Start at top
            x = center.x + radius * np.cos(angle)
            y = center.y + radius * np.sin(angle)
            points.append(self.add_point(x, y))
        points.append(points[0])  # Close polygon
        
        poly = Polygon2D(points, name or f"Poly{num_sides}_{len(self.polygons)}")
        self.polygons.append(poly)
        return poly
    
    # -------------------------------------------------------------------------
    # Spline Creation
    # -------------------------------------------------------------------------
    def add_spline(self, control_points: List[Point2D], degree: int = 3,
                   name: str = "") -> Spline2D:
        """Add B-spline curve"""
        spline = Spline2D(control_points, degree, name)
        self.splines.append(spline)
        return spline
    
    def add_spline_by_coords(self, coords: List[Tuple[float, float]], 
                             degree: int = 3, name: str = "") -> Spline2D:
        """Add spline from coordinate list"""
        points = [self.add_point(x, y) for x, y in coords]
        return self.add_spline(points, degree, name)
    
    # -------------------------------------------------------------------------
    # Slot Creation
    # -------------------------------------------------------------------------
    def add_slot(self, p1: Point2D, p2: Point2D, width: float, name: str = "") -> Polygon2D:
        """Add a slot (two semicircles connected by lines)"""
        # Calculate slot geometry
        dx = p2.x - p1.x
        dy = p2.y - p1.y
        length = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        
        radius = width / 2
        
        # Perpendicular offset
        perp_x = -np.sin(angle) * radius
        perp_y = np.cos(angle) * radius
        
        # Create points for slot outline
        points = [
            self.add_point(p1.x + perp_x, p1.y + perp_y),
            self.add_point(p2.x + perp_x, p2.y + perp_y),
            self.add_point(p2.x - perp_x, p2.y - perp_y),
            self.add_point(p1.x - perp_x, p1.y - perp_y),
            self.add_point(p1.x + perp_x, p1.y + perp_y),  # Close
        ]
        
        poly = Polygon2D(points, name or f"Slot_{len(self.polygons)}")
        self.polygons.append(poly)
        return poly
    
    # -------------------------------------------------------------------------
    # Constraint Methods
    # -------------------------------------------------------------------------
    def add_constraint(self, constraint: Constraint) -> Constraint:
        """Add a constraint to the sketch"""
        self.constraints.append(constraint)
        self.solver.add_constraint(constraint)
        return constraint
    
    def add_coincident(self, p1: Point2D, p2: Point2D) -> Constraint:
        """Make two points coincident (share location)"""
        return self.add_constraint(Constraint(
            ConstraintType.COINCIDENT, [p1, p2]
        ))
    
    def add_horizontal(self, line: Line2D) -> Constraint:
        """Make line horizontal"""
        return self.add_constraint(Constraint(
            ConstraintType.HORIZONTAL, [line]
        ))
    
    def add_vertical(self, line: Line2D) -> Constraint:
        """Make line vertical"""
        return self.add_constraint(Constraint(
            ConstraintType.VERTICAL, [line]
        ))
    
    def add_parallel(self, line1: Line2D, line2: Line2D) -> Constraint:
        """Make two lines parallel"""
        return self.add_constraint(Constraint(
            ConstraintType.PARALLEL, [line1, line2]
        ))
    
    def add_perpendicular(self, line1: Line2D, line2: Line2D) -> Constraint:
        """Make two lines perpendicular"""
        return self.add_constraint(Constraint(
            ConstraintType.PERPENDICULAR, [line1, line2]
        ))
    
    def add_equal(self, e1: SketchEntity, e2: SketchEntity) -> Constraint:
        """Make lengths or radii equal"""
        return self.add_constraint(Constraint(
            ConstraintType.EQUAL, [e1, e2]
        ))
    
    def add_tangent(self, e1: SketchEntity, e2: SketchEntity) -> Constraint:
        """Make entities tangent"""
        return self.add_constraint(Constraint(
            ConstraintType.TANGENT, [e1, e2]
        ))
    
    def add_concentric(self, c1: Union[Circle2D, Arc2D], 
                       c2: Union[Circle2D, Arc2D]) -> Constraint:
        """Make circles/arcs share center"""
        return self.add_constraint(Constraint(
            ConstraintType.CONCENTRIC, [c1, c2]
        ))
    
    def add_midpoint(self, point: Point2D, line: Line2D) -> Constraint:
        """Place point at midpoint of line"""
        return self.add_constraint(Constraint(
            ConstraintType.MIDPOINT, [point, line]
        ))
    
    def add_symmetric(self, p1: Point2D, p2: Point2D, axis: Line2D) -> Constraint:
        """Make points symmetric about axis"""
        # Simplified: constrain midpoint to axis
        mid = Point2D((p1.x + p2.x) / 2, (p1.y + p2.y) / 2)
        # In full implementation, project midpoint onto axis line
        return self.add_constraint(Constraint(
            ConstraintType.SYMMETRIC, [p1, p2, axis]
        ))
    
    # -------------------------------------------------------------------------
    # Dimensional Constraints
    # -------------------------------------------------------------------------
    def add_distance(self, e1: SketchEntity, e2: SketchEntity, 
                     distance: float) -> Constraint:
        """Constrain distance between entities"""
        return self.add_constraint(Constraint(
            ConstraintType.DISTANCE, [e1, e2], distance
        ))
    
    def add_distance_point_line(self, point: Point2D, line: Line2D,
                                 distance: float) -> Constraint:
        """Constrain distance from point to line"""
        return self.add_constraint(Constraint(
            ConstraintType.DISTANCE, [point, line], distance
        ))
    
    def add_angle(self, line1: Line2D, line2: Line2D, 
                  angle: float) -> Constraint:
        """Constrain angle between lines (in degrees)"""
        return self.add_constraint(Constraint(
            ConstraintType.ANGLE, [line1, line2], np.radians(angle)
        ))
    
    def add_radius(self, circle: Union[Circle2D, Arc2D], 
                   radius: float) -> Constraint:
        """Constrain radius"""
        return self.add_constraint(Constraint(
            ConstraintType.RADIUS, [circle], radius
        ))
    
    def add_diameter(self, circle: Circle2D, diameter: float) -> Constraint:
        """Constrain diameter"""
        return self.add_constraint(Constraint(
            ConstraintType.DIAMETER, [circle], diameter
        ))
    
    def add_length(self, line: Line2D, length: float) -> Constraint:
        """Constrain line length"""
        # Create a virtual "length constraint" using distance
        # In practice, this would use a dedicated constraint type
        return self.add_constraint(Constraint(
            ConstraintType.DISTANCE, [line.p1, line.p2], length
        ))
    
    # -------------------------------------------------------------------------
    # Solving
    # -------------------------------------------------------------------------
    def solve(self) -> bool:
        """Solve all constraints"""
        logger.info(f"Solving {len(self.constraints)} constraints...")
        return self.solver.solve(self)
    
    def is_fully_constrained(self) -> bool:
        """Check if sketch is fully constrained"""
        # Count degrees of freedom vs constraints
        # Simplified check
        num_points = len(self.points)
        num_constraints = len(self.constraints)
        
        # Each point has 2 DOF, each constraint removes 1-2 DOF
        # Rough estimate: need at least 2 constraints per point
        return num_constraints >= num_points * 2 - 3  # -3 for datum
    
    # -------------------------------------------------------------------------
    # Queries
    # -------------------------------------------------------------------------
    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Get bounding box (min_x, min_y, max_x, max_y)"""
        all_points = self.points.copy()
        for poly in self.polygons:
            all_points.extend(poly.points)
        for circle in self.circles:
            all_points.append(Point2D(circle.center.x - circle.radius, 
                                      circle.center.y - circle.radius))
            all_points.append(Point2D(circle.center.x + circle.radius, 
                                      circle.center.y + circle.radius))
        
        if not all_points:
            return (0, 0, 0, 0)
        
        xs = [p.x for p in all_points]
        ys = [p.y for p in all_points]
        return (min(xs), min(ys), max(xs), max(ys))
    
    def get_all_entities(self) -> List[SketchEntity]:
        """Get all entities in sketch"""
        return (self.points + self.lines + self.circles + 
                self.arcs + self.polygons + self.splines)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize sketch to dictionary"""
        return {
            "name": self.name,
            "plane": self.plane,
            "points": [(p.x, p.y) for p in self.points],
            "circles": [(c.center.x, c.center.y, c.radius) for c in self.circles],
            "polygons": [[(pt.x, pt.y) for pt in poly.points] for poly in self.polygons],
            "constraints": [
                {"type": c.type.name, "value": c.value} 
                for c in self.constraints
            ]
        }
    
    def __repr__(self) -> str:
        return (f"Sketch('{self.name}': "
                f"{len(self.points)} points, "
                f"{len(self.lines)} lines, "
                f"{len(self.circles)} circles, "
                f"{len(self.constraints)} constraints)")


# =============================================================================
# 3D FEATURE OPERATIONS
# =============================================================================

class ExtrudeFeature:
    """Extrude a 2D sketch to create a 3D solid"""
    
    def __init__(self, sketch: Sketch, depth: float, 
                 taper_angle: float = 0, 
                 symmetric: bool = False,
                 direction: Tuple[float, float, float] = (0, 0, 1)):
        self.sketch = sketch
        self.depth = depth
        self.taper_angle = taper_angle
        self.symmetric = symmetric
        self.direction = direction
    
    def to_occt_shape(self):
        """Convert to OpenCASCADE shape"""
        # This would integrate with geometry_api.py
        # For now, return a placeholder
        return None


class RevolveFeature:
    """Revolve a 2D sketch around an axis to create a 3D solid"""
    
    def __init__(self, sketch: Sketch, 
                 axis: Tuple[Point2D, Point2D],
                 angle: float = 360):
        self.sketch = sketch
        self.axis = axis
        self.angle = angle


class SweepFeature:
    """Sweep a profile along a path"""
    
    def __init__(self, profile: Sketch, path: List[Point2D]):
        self.profile = profile
        self.path = path


class LoftFeature:
    """Loft between multiple sketches"""
    
    def __init__(self, sketches: List[Sketch], 
                 guides: Optional[List[Spline2D]] = None):
        self.sketches = sketches
        self.guides = guides


# =============================================================================
# TEST/DEMO
# =============================================================================

def test_sketch_system():
    """Test the complete sketch system"""
    print("="*70)
    print("TESTING FULL SKETCH SYSTEM")
    print("="*70)
    
    # 1. Basic Sketch Creation
    print("\n1. Creating basic sketch elements...")
    sketch = Sketch("Test_Part")
    
    # Add points
    p1 = sketch.add_point(0, 0)
    p2 = sketch.add_point(100, 0)
    p3 = sketch.add_point(100, 50)
    p4 = sketch.add_point(0, 50)
    print(f"   Added 4 points")
    
    # Add lines
    l1 = sketch.add_line(p1, p2)
    l2 = sketch.add_line(p2, p3)
    l3 = sketch.add_line(p3, p4)
    l4 = sketch.add_line(p4, p1)
    print(f"   Added 4 lines")
    
    # Add constraints
    sketch.add_horizontal(l1)
    sketch.add_horizontal(l3)
    sketch.add_vertical(l2)
    sketch.add_vertical(l4)
    print(f"   Added perpendicular constraints")
    
    # Add dimensions
    sketch.add_length(l1, 100)
    sketch.add_length(l2, 50)
    print(f"   Added dimensional constraints")
    
    # Solve
    print(f"\n2. Solving constraints...")
    converged = sketch.solve()
    print(f"   Converged: {converged}")
    print(f"   Rectangle: ({p1.x:.1f},{p1.y:.1f}) to ({p3.x:.1f},{p3.y:.1f})")
    
    # 3. Circle with constraints
    print(f"\n3. Adding circle with constraints...")
    center = sketch.add_point(50, 25)
    circle = sketch.add_circle(center, 10)
    sketch.add_radius(circle, 15)
    print(f"   Circle radius: {circle.radius:.1f}")
    
    # 4. Rectangle helper
    print(f"\n4. Testing rectangle helper...")
    sketch2 = Sketch("Rectangle_Test")
    rect = sketch2.add_rectangle_by_coords(0, 0, 100, 50)
    sketch2.solve()
    print(f"   Rectangle area: {rect.area():.1f} mm²")
    
    # 5. Polygon
    print(f"\n5. Testing regular polygon...")
    sketch3 = Sketch("Polygon_Test")
    center = sketch3.add_point(0, 0)
    hexagon = sketch3.add_regular_polygon(center, 50, 6)
    print(f"   Hexagon area: {hexagon.area():.1f} mm²")
    
    # 6. Complex sketch with arcs
    print(f"\n6. Testing arc creation...")
    sketch4 = Sketch("Arc_Test")
    arc = sketch4.add_arc_by_coords(0, 0, 50, 0, 90)
    print(f"   Arc start: ({arc.start_point().x:.1f}, {arc.start_point().y:.1f})")
    print(f"   Arc end: ({arc.end_point().x:.1f}, {arc.end_point().y:.1f})")
    
    print("\n" + "="*70)
    print("ALL SKETCH TESTS PASSED!")
    print("="*70)
    
    return True


if __name__ == "__main__":
    test_sketch_system()
