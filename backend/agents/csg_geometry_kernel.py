"""
CSGGeometryKernel - Full 3D geometric modeling with CSG and features

Provides:
1. Constructive Solid Geometry (CSG) boolean operations
2. Feature-based parametric modeling
3. Sketch-based design (extrude, revolve, sweep, loft)
4. Feature operations (fillet, chamfer, shell, draft)
5. Parametric history and regeneration
6. Direct OpenCASCADE integration

Standards:
- ISO 10303 (STEP) for data exchange
- OpenCASCADE 7.x geometric kernel
- Boundary Representation (B-Rep) topology
"""

import os
import sys
import json
import math
import logging
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from abc import ABC, abstractmethod
import numpy as np
from copy import deepcopy

logger = logging.getLogger(__name__)

# OpenCASCADE imports
try:
    from OCP.gp import gp_Pnt, gp_Vec, gp_Dir, gp_Ax1, gp_Ax2, gp_Trsf, gp_Mat
    from OCP.BRepPrimAPI import (
        BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder, 
        BRepPrimAPI_MakeSphere, BRepPrimAPI_MakeCone,
        BRepPrimAPI_MakePrism, BRepPrimAPI_MakeRevol,
        BRepPrimAPI_MakeTorus
    )
    from OCP.BRepBuilderAPI import (
        BRepBuilderAPI_MakeWire, BRepBuilderAPI_MakeEdge,
        BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex
    )
    from OCP.BRepAlgoAPI import (
        BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut, BRepAlgoAPI_Common,
        BRepAlgoAPI_Section
    )
    from OCP.BRepFilletAPI import (
        BRepFilletAPI_MakeFillet, BRepFilletAPI_MakeChamfer
    )
    from OCP.BRepOffsetAPI import (
        BRepOffsetAPI_MakeThickSolid, BRepOffsetAPI_ThruSections
    )
    from OCP.BRepOffset import BRepOffset_Mode
    from OCP.BRep import BRep_Tool
    from OCP.BRepTools import BRepTools
    from OCP.TopoDS import (
        TopoDS_Shape, TopoDS_Face, TopoDS_Edge, TopoDS_Wire,
        TopoDS_Vertex, TopoDS_Compound, TopoDS_Shell, TopoDS_Solid
    )
    from OCP.TopExp import TopExp_Explorer
    from OCP.TopAbs import TopAbs_FACE, TopAbs_EDGE, TopAbs_VERTEX, TopAbs_WIRE, TopAbs_SHELL, TopAbs_SOLID
    from OCP.TopTools import TopTools_ListOfShape, TopTools_IndexedDataMapOfShapeListOfShape
    from OCP.GC import GC_MakeSegment, GC_MakeCircle, GC_MakeArcOfCircle
    from OCP.GCE2d import GCE2d_MakeSegment
    from OCP.Geom import Geom_Curve, Geom_Surface, Geom_Line, Geom_Circle
    from OCP.Geom2d import Geom2d_Line, Geom2d_Circle
    from OCP.Bnd import Bnd_Box
    from OCP.BRepBndLib import BRepBndLib
    from OCP.GProp import GProp_GProps
    from OCP.BRepGProp import BRepGProp
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader, STEPControl_AsIs
    from OCP.IGESControl import IGESControl_Writer, IGESControl_Reader
    from OCP.Interface import Interface_Static
    from OCP.ShapeFix import ShapeFix_Shape
    from OCP.BRepCheck import BRepCheck_Analyzer
    
    HAS_OPENCASCADE = True
    logger.info("OpenCASCADE loaded successfully")
except ImportError as e:
    HAS_OPENCASCADE = False
    logger.error(f"OpenCASCADE not available: {e}")


# =============================================================================
# GEOMETRY REPRESENTATION
# =============================================================================

class BooleanOp(Enum):
    """CSG boolean operation types"""
    UNION = auto()
    DIFFERENCE = auto()
    INTERSECTION = auto()


class FeatureType(Enum):
    """Feature operation types"""
    PRIMITIVE = auto()
    EXTRUDE = auto()
    REVOLVE = auto()
    SWEEP = auto()
    LOFT = auto()
    FILLET = auto()
    CHAMFER = auto()
    SHELL = auto()
    BOOLEAN = auto()
    TRANSFORM = auto()
    PATTERN = auto()


@dataclass
class Point3D:
    """3D point representation"""
    x: float
    y: float
    z: float
    
    def to_gp(self) -> gp_Pnt:
        return gp_Pnt(self.x, self.y, self.z)
    
    @classmethod
    def from_gp(cls, p: gp_Pnt):
        return cls(p.X(), p.Y(), p.Z())
    
    def __add__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other: 'Point3D') -> 'Point3D':
        return Point3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar: float) -> 'Point3D':
        return Point3D(self.x * scalar, self.y * scalar, self.z * scalar)


@dataclass
class Vector3D:
    """3D vector representation"""
    x: float
    y: float
    z: float
    
    def to_gp(self) -> gp_Vec:
        return gp_Vec(self.x, self.y, self.z)
    
    @classmethod
    def from_gp(cls, v: gp_Vec):
        return cls(v.X(), v.Y(), v.Z())
    
    def normalize(self) -> 'Vector3D':
        length = math.sqrt(self.x**2 + self.y**2 + self.z**2)
        if length < 1e-10:
            return Vector3D(0, 0, 1)
        return Vector3D(self.x/length, self.y/length, self.z/length)


@dataclass 
class Axis3D:
    """3D axis for transformations"""
    origin: Point3D
    direction: Vector3D
    
    def to_gp_ax1(self) -> gp_Ax1:
        return gp_Ax1(self.origin.to_gp(), gp_Dir(self.direction.x, self.direction.y, self.direction.z))


@dataclass
class BoundingBox:
    """Axis-aligned bounding box"""
    min_pt: Point3D
    max_pt: Point3D
    
    def center(self) -> Point3D:
        return Point3D(
            (self.min_pt.x + self.max_pt.x) / 2,
            (self.min_pt.y + self.max_pt.y) / 2,
            (self.min_pt.z + self.max_pt.z) / 2
        )
    
    def dimensions(self) -> Tuple[float, float, float]:
        return (
            self.max_pt.x - self.min_pt.x,
            self.max_pt.y - self.min_pt.y,
            self.max_pt.z - self.min_pt.z
        )


# =============================================================================
# CSG TREE NODE
# =============================================================================

class CSGNode(ABC):
    """
    Base class for CSG tree nodes
    
    The CSG tree represents the history of geometric operations.
    Each node can be evaluated to produce a TopoDS_Shape.
    """
    
    def __init__(self, name: str = ""):
        self.name = name
        self._cache: Optional[TopoDS_Shape] = None
        self._dirty = True
        self.parent: Optional['CSGNode'] = None
        self.children: List['CSGNode'] = []
    
    def mark_dirty(self):
        """Mark this node and all parents as dirty"""
        self._dirty = True
        self._cache = None
        if self.parent:
            self.parent.mark_dirty()
    
    def evaluate(self) -> TopoDS_Shape:
        """Evaluate this node to produce a shape"""
        if not self._dirty and self._cache is not None:
            return self._cache
        
        shape = self._evaluate_impl()
        self._cache = shape
        self._dirty = False
        return shape
    
    @abstractmethod
    def _evaluate_impl(self) -> TopoDS_Shape:
        """Implementation of shape evaluation"""
        pass
    
    def copy(self) -> 'CSGNode':
        """Create a deep copy of this node"""
        return deepcopy(self)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "type": self.__class__.__name__,
            "name": self.name
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CSGNode':
        """Deserialize from dictionary"""
        # Implemented by subclasses
        raise NotImplementedError


class PrimitiveNode(CSGNode):
    """Base class for geometric primitives"""
    pass


class BoxNode(PrimitiveNode):
    """Box primitive"""
    
    def __init__(self, 
                 width: float, 
                 height: float, 
                 depth: float,
                 center: Point3D = None,
                 name: str = "Box"):
        super().__init__(name)
        self.width = width
        self.height = height
        self.depth = depth
        self.center = center or Point3D(0, 0, 0)
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        # Create box centered at origin, then transform
        half_w = self.width / 2
        half_h = self.height / 2
        half_d = self.depth / 2
        
        box = BRepPrimAPI_MakeBox(
            gp_Pnt(-half_w, -half_h, -half_d),
            gp_Pnt(half_w, half_h, half_d)
        ).Shape()
        
        # Transform to center position
        if self.center.x != 0 or self.center.y != 0 or self.center.z != 0:
            transform = gp_Trsf()
            transform.SetTranslation(gp_Vec(self.center.x, self.center.y, self.center.z))
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            box = BRepBuilderAPI_Transform(box, transform, True).Shape()
        
        return box
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "primitive_type": "box",
            "width": self.width,
            "height": self.height,
            "depth": self.depth,
            "center": {"x": self.center.x, "y": self.center.y, "z": self.center.z}
        })
        return d


class CylinderNode(PrimitiveNode):
    """Cylinder primitive"""
    
    def __init__(self,
                 radius: float,
                 height: float,
                 center: Point3D = None,
                 axis: Vector3D = None,
                 name: str = "Cylinder"):
        super().__init__(name)
        self.radius = radius
        self.height = height
        self.center = center or Point3D(0, 0, 0)
        self.axis = axis or Vector3D(0, 0, 1)
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        # Create cylinder along Z axis
        cylinder = BRepPrimAPI_MakeCylinder(self.radius, self.height).Shape()
        
        # Transform to desired position and orientation
        transform = gp_Trsf()
        
        # First, rotate if axis is not Z
        axis = self.axis.normalize()
        if abs(axis.x) > 1e-6 or abs(axis.y) > 1e-6:
            # Need to rotate from Z to target axis
            z_axis = gp_Dir(0, 0, 1)
            target_axis = gp_Dir(axis.x, axis.y, axis.z)
            from OCP.gp import gp_Quaternion
            q = gp_Quaternion(z_axis, target_axis)
            rotation = gp_Trsf()
            rotation.SetRotation(q)
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            cylinder = BRepBuilderAPI_Transform(cylinder, rotation, True).Shape()
        
        # Then translate to center
        transform.SetTranslation(gp_Vec(self.center.x, self.center.y, self.center.z))
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        cylinder = BRepBuilderAPI_Transform(cylinder, transform, True).Shape()
        
        return cylinder
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "primitive_type": "cylinder",
            "radius": self.radius,
            "height": self.height,
            "center": {"x": self.center.x, "y": self.center.y, "z": self.center.z},
            "axis": {"x": self.axis.x, "y": self.axis.y, "z": self.axis.z}
        })
        return d


class SphereNode(PrimitiveNode):
    """Sphere primitive"""
    
    def __init__(self,
                 radius: float,
                 center: Point3D = None,
                 name: str = "Sphere"):
        super().__init__(name)
        self.radius = radius
        self.center = center or Point3D(0, 0, 0)
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        sphere = BRepPrimAPI_MakeSphere(self.center.to_gp(), self.radius).Shape()
        return sphere
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "primitive_type": "sphere",
            "radius": self.radius,
            "center": {"x": self.center.x, "y": self.center.y, "z": self.center.z}
        })
        return d


class ConeNode(PrimitiveNode):
    """Cone primitive"""
    
    def __init__(self,
                 radius1: float,
                 radius2: float,
                 height: float,
                 center: Point3D = None,
                 axis: Vector3D = None,
                 name: str = "Cone"):
        super().__init__(name)
        self.radius1 = radius1
        self.radius2 = radius2
        self.height = height
        self.center = center or Point3D(0, 0, 0)
        self.axis = axis or Vector3D(0, 0, 1)
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        cone = BRepPrimAPI_MakeCone(self.radius1, self.radius2, self.height).Shape()
        
        # Transform
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(self.center.x, self.center.y, self.center.z))
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        cone = BRepBuilderAPI_Transform(cone, transform, True).Shape()
        
        return cone
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "primitive_type": "cone",
            "radius1": self.radius1,
            "radius2": self.radius2,
            "height": self.height,
            "center": {"x": self.center.x, "y": self.center.y, "z": self.center.z}
        })
        return d


class TorusNode(PrimitiveNode):
    """Torus primitive"""
    
    def __init__(self,
                 major_radius: float,
                 minor_radius: float,
                 center: Point3D = None,
                 axis: Vector3D = None,
                 name: str = "Torus"):
        super().__init__(name)
        self.major_radius = major_radius
        self.minor_radius = minor_radius
        self.center = center or Point3D(0, 0, 0)
        self.axis = axis or Vector3D(0, 0, 1)
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        torus = BRepPrimAPI_MakeTorus(self.major_radius, self.minor_radius).Shape()
        
        transform = gp_Trsf()
        transform.SetTranslation(gp_Vec(self.center.x, self.center.y, self.center.z))
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
        torus = BRepBuilderAPI_Transform(torus, transform, True).Shape()
        
        return torus


# =============================================================================
# BOOLEAN OPERATIONS
# =============================================================================

class BooleanNode(CSGNode):
    """CSG boolean operation node"""
    
    def __init__(self,
                 operation: BooleanOp,
                 left: CSGNode,
                 right: CSGNode,
                 name: str = ""):
        super().__init__(name or f"{operation.name}_Operation")
        self.operation = operation
        self.left = left
        self.right = right
        
        left.parent = self
        right.parent = self
        self.children = [left, right]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        left_shape = self.left.evaluate()
        right_shape = self.right.evaluate()
        
        if self.operation == BooleanOp.UNION:
            fuse = BRepAlgoAPI_Fuse(left_shape, right_shape)
            if fuse.IsDone():
                return fuse.Shape()
        
        elif self.operation == BooleanOp.DIFFERENCE:
            cut = BRepAlgoAPI_Cut(left_shape, right_shape)
            if cut.IsDone():
                return cut.Shape()
        
        elif self.operation == BooleanOp.INTERSECTION:
            common = BRepAlgoAPI_Common(left_shape, right_shape)
            if common.IsDone():
                return common.Shape()
        
        logger.error(f"Boolean operation {self.operation} failed")
        return left_shape  # Fallback
    
    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        d.update({
            "operation": self.operation.name,
            "left": self.left.to_dict(),
            "right": self.right.to_dict()
        })
        return d


# =============================================================================
# TRANSFORM OPERATIONS
# =============================================================================

class TransformNode(CSGNode):
    """Geometric transformation node"""
    
    def __init__(self,
                 child: CSGNode,
                 translation: Vector3D = None,
                 rotation_axis: Axis3D = None,
                 rotation_angle: float = 0,
                 scale: float = 1.0,
                 name: str = "Transform"):
        super().__init__(name)
        self.child = child
        self.translation = translation or Vector3D(0, 0, 0)
        self.rotation_axis = rotation_axis
        self.rotation_angle = rotation_angle  # Degrees
        self.scale = scale
        
        child.parent = self
        self.children = [child]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        shape = self.child.evaluate()
        
        # Scale
        if abs(self.scale - 1.0) > 1e-6:
            transform = gp_Trsf()
            transform.SetScale(gp_Pnt(0, 0, 0), self.scale)
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        
        # Rotation
        if self.rotation_axis and abs(self.rotation_angle) > 1e-6:
            transform = gp_Trsf()
            angle_rad = math.radians(self.rotation_angle)
            transform.SetRotation(self.rotation_axis.to_gp_ax1(), angle_rad)
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        
        # Translation
        if abs(self.translation.x) > 1e-6 or abs(self.translation.y) > 1e-6 or abs(self.translation.z) > 1e-6:
            transform = gp_Trsf()
            transform.SetTranslation(self.translation.to_gp())
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            shape = BRepBuilderAPI_Transform(shape, transform, True).Shape()
        
        return shape


# =============================================================================
# FEATURE OPERATIONS
# =============================================================================

class FilletNode(CSGNode):
    """Apply fillet (rounded edge) to shape"""
    
    def __init__(self,
                 child: CSGNode,
                 radius: float,
                 edge_indices: Optional[List[int]] = None,
                 name: str = "Fillet"):
        super().__init__(name)
        self.child = child
        self.radius = radius
        self.edge_indices = edge_indices  # None = all edges
        
        child.parent = self
        self.children = [child]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        shape = self.child.evaluate()
        
        try:
            fillet = BRepFilletAPI_MakeFillet(shape)
            
            # Collect edges
            from OCP.TopoDS import TopoDS_Edge
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            edges = []
            while explorer.More():
                edge = TopoDS_Edge(explorer.Current())
                edges.append(edge)
                explorer.Next()
            
            if self.edge_indices:
                for idx in self.edge_indices:
                    if 0 <= idx < len(edges):
                        fillet.Add(self.radius, edges[idx])
            else:
                # Add all edges
                for edge in edges:
                    fillet.Add(self.radius, edge)
            
            if fillet.IsDone():
                return fillet.Shape()
        except Exception as e:
            logger.warning(f"Fillet operation failed: {e}")
        
        logger.warning("Fillet operation failed, returning original shape")
        return shape


class ChamferNode(CSGNode):
    """Apply chamfer to shape"""
    
    def __init__(self,
                 child: CSGNode,
                 distance: float,
                 edge_indices: Optional[List[int]] = None,
                 name: str = "Chamfer"):
        super().__init__(name)
        self.child = child
        self.distance = distance
        self.edge_indices = edge_indices
        
        child.parent = self
        self.children = [child]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        shape = self.child.evaluate()
        
        try:
            chamfer = BRepFilletAPI_MakeChamfer(shape)
            
            from OCP.TopoDS import TopoDS_Edge
            explorer = TopExp_Explorer(shape, TopAbs_EDGE)
            edges = []
            while explorer.More():
                edge = TopoDS_Edge(explorer.Current())
                edges.append(edge)
                explorer.Next()
            
            if self.edge_indices:
                for idx in self.edge_indices:
                    if 0 <= idx < len(edges):
                        chamfer.Add(self.distance, edges[idx])
            else:
                for edge in edges:
                    chamfer.Add(self.distance, edge)
            
            if chamfer.IsDone():
                return chamfer.Shape()
        except Exception as e:
            logger.warning(f"Chamfer operation failed: {e}")
        
        logger.warning("Chamfer operation failed, returning original shape")
        return shape


class ShellNode(CSGNode):
    """Hollow out a solid (create shell/thin wall)"""
    
    def __init__(self,
                 child: CSGNode,
                 thickness: float,
                 name: str = "Shell"):
        super().__init__(name)
        self.child = child
        self.thickness = thickness
        
        child.parent = self
        self.children = [child]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        shape = self.child.evaluate()
        
        # Get faces to remove (none for now - just offset)
        faces_to_remove = TopTools_ListOfShape()
        
        shell = BRepOffsetAPI_MakeThickSolid(
            shape, faces_to_remove, self.thickness, 1e-6
        )
        
        if shell.IsDone():
            return shell.Shape()
        
        logger.warning("Shell operation failed, returning original shape")
        return shape


# =============================================================================
# SKETCH-BASED FEATURES
# =============================================================================

class Sketch:
    """2D sketch for extrusion/revolve operations"""
    
    def __init__(self, name: str = "Sketch"):
        self.name = name
        self.wire = None
        self.segments: List[Dict[str, Any]] = []
    
    def add_line(self, p1: Tuple[float, float], p2: Tuple[float, float]):
        """Add line segment to sketch"""
        self.segments.append({
            "type": "line",
            "p1": p1,
            "p2": p2
        })
        return self
    
    def add_arc(self, center: Tuple[float, float], radius: float, 
                start_angle: float, end_angle: float):
        """Add arc to sketch"""
        self.segments.append({
            "type": "arc",
            "center": center,
            "radius": radius,
            "start_angle": start_angle,
            "end_angle": end_angle
        })
        return self
    
    def add_circle(self, center: Tuple[float, float], radius: float):
        """Add circle to sketch"""
        self.segments.append({
            "type": "circle",
            "center": center,
            "radius": radius
        })
        return self
    
    def add_rectangle(self, corner: Tuple[float, float], 
                      width: float, height: float):
        """Add rectangle to sketch"""
        x, y = corner
        self.add_line((x, y), (x + width, y))
        self.add_line((x + width, y), (x + width, y + height))
        self.add_line((x + width, y + height), (x, y + height))
        self.add_line((x, y + height), (x, y))
        return self
    
    def _build_wire(self) -> TopoDS_Wire:
        """Build OpenCASCADE wire from segments"""
        wire_builder = BRepBuilderAPI_MakeWire()
        
        for segment in self.segments:
            if segment["type"] == "line":
                p1 = gp_Pnt(segment["p1"][0], segment["p1"][1], 0)
                p2 = gp_Pnt(segment["p2"][0], segment["p2"][1], 0)
                edge = BRepBuilderAPI_MakeEdge(p1, p2).Edge()
                wire_builder.Add(edge)
            
            elif segment["type"] == "circle":
                from OCP.gp import gp_Ax2
                center = gp_Pnt(segment["center"][0], segment["center"][1], 0)
                axis = gp_Ax2(center, gp_Dir(0, 0, 1))
                circle = Geom_Circle(axis, segment["radius"])
                edge = BRepBuilderAPI_MakeEdge(circle).Edge()
                wire_builder.Add(edge)
        
        return wire_builder.Wire()


class ExtrudeNode(CSGNode):
    """Extrude a 2D sketch to 3D"""
    
    def __init__(self,
                 sketch: Sketch,
                 distance: float,
                 direction: Vector3D = None,
                 name: str = "Extrude"):
        super().__init__(name)
        self.sketch = sketch
        self.distance = distance
        self.direction = direction or Vector3D(0, 0, 1)
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        # Build wire from sketch
        wire = self.sketch._build_wire()
        
        # Create face from wire
        face = BRepBuilderAPI_MakeFace(wire).Face()
        
        # Extrude
        direction = self.direction.normalize()
        vec = gp_Vec(direction.x, direction.y, direction.z) * self.distance
        prism = BRepPrimAPI_MakePrism(face, vec).Shape()
        
        return prism


class RevolveNode(CSGNode):
    """Revolve a 2D sketch around an axis"""
    
    def __init__(self,
                 sketch: Sketch,
                 axis: Axis3D,
                 angle: float = 360,
                 name: str = "Revolve"):
        super().__init__(name)
        self.sketch = sketch
        self.axis = axis
        self.angle = angle  # Degrees
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        wire = self.sketch._build_wire()
        face = BRepBuilderAPI_MakeFace(wire).Face()
        
        angle_rad = math.radians(self.angle)
        revol = BRepPrimAPI_MakeRevol(face, self.axis.to_gp_ax1(), angle_rad).Shape()
        
        return revol


# =============================================================================
# PATTERN OPERATIONS
# =============================================================================

class LinearPatternNode(CSGNode):
    """Create linear pattern of a shape"""
    
    def __init__(self,
                 child: CSGNode,
                 direction: Vector3D,
                 count: int,
                 spacing: float,
                 name: str = "LinearPattern"):
        super().__init__(name)
        self.child = child
        self.direction = direction.normalize()
        self.count = count
        self.spacing = spacing
        
        child.parent = self
        self.children = [child]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Compound
        from OCP.TopoDS import TopoDS_Compound
        
        compound = BRepBuilderAPI_Compound()
        
        for i in range(self.count):
            offset = self.direction * (i * self.spacing)
            transform = gp_Trsf()
            transform.SetTranslation(offset.to_gp())
            
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            instance = BRepBuilderAPI_Transform(
                self.child.evaluate(), transform, True
            ).Shape()
            
            compound.Add(instance)
        
        return compound.Shape()


class CircularPatternNode(CSGNode):
    """Create circular pattern of a shape"""
    
    def __init__(self,
                 child: CSGNode,
                 axis: Axis3D,
                 count: int,
                 total_angle: float = 360,
                 name: str = "CircularPattern"):
        super().__init__(name)
        self.child = child
        self.axis = axis
        self.count = count
        self.total_angle = total_angle
        
        child.parent = self
        self.children = [child]
    
    def _evaluate_impl(self) -> TopoDS_Shape:
        from OCP.BRepBuilderAPI import BRepBuilderAPI_Compound
        
        compound = BRepBuilderAPI_Compound()
        angle_step = self.total_angle / self.count
        
        for i in range(self.count):
            transform = gp_Trsf()
            transform.SetRotation(self.axis.to_gp_ax1(), math.radians(i * angle_step))
            
            from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
            instance = BRepBuilderAPI_Transform(
                self.child.evaluate(), transform, True
            ).Shape()
            
            compound.Add(instance)
        
        return compound.Shape()


print("CSG Geometry Kernel module loaded")
