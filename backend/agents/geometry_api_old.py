"""
GeometryAPI - Fluent interface for complex 3D modeling

High-level API that wraps CSG kernel for intuitive shape creation.
Similar to OpenSCAD/CadQuery but integrated with BRICK OS.

Examples:
    # Bracket with holes
    bracket = (Box(100, 50, 10) 
               - Cylinder(5, 15).move(20, 15, -2.5)
               - Cylinder(5, 15).move(80, 15, -2.5)
               - Cylinder(5, 15).move(20, 35, -2.5)
               - Cylinder(5, 15).move(80, 35, -2.5))
    
    # Flanged pipe
    pipe = (Cylinder(20, 100).move(0, 0, 0)
            + Cylinder(35, 5).move(0, 0, -2.5)  # Bottom flange
            + Cylinder(35, 5).move(0, 0, 97.5))  # Top flange
    
    # Parametric gear (simplified)
    gear = Cylinder(50, 10).shell(2)
"""

import math
from typing import List, Tuple, Optional, Union, Dict, Any
from pathlib import Path
import numpy as np

from csg_geometry_kernel import (
    CSGNode, PrimitiveNode, BoxNode, CylinderNode, SphereNode, 
    ConeNode, TorusNode, BooleanNode, BooleanOp,
    TransformNode, FilletNode, ChamferNode, ShellNode,
    ExtrudeNode, RevolveNode, LinearPatternNode, CircularPatternNode,
    Sketch, Point3D, Vector3D, Axis3D, BoundingBox
)

# Import OpenCASCADE for export
try:
    from OCP.TopoDS import TopoDS_Shape
    from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
    from OCP.IGESControl import IGESControl_Writer
    from OCP.BRepMesh import BRepMesh_IncrementalMesh
    HAS_OPENCASCADE = True
except ImportError:
    HAS_OPENCASCADE = False

# Import meshing engine
try:
    from meshing_engine import MeshingEngine, MeshingParameters, ElementType
    HAS_MESHING = True
except ImportError:
    HAS_MESHING = False


class Shape:
    """
    High-level shape object with fluent API
    
    Wraps a CSGNode and provides intuitive operators:
    +  : Union
    -  : Difference (cut)
    &  : Intersection
    """
    
    def __init__(self, node: CSGNode, name: str = ""):
        self._node = node
        self.name = name or node.name
    
    @property
    def node(self) -> CSGNode:
        return self._node
    
    def evaluate(self):
        """Evaluate the shape to OpenCASCADE TopoDS_Shape"""
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        return self._node.evaluate()
    
    # =================================================================
    # BOOLEAN OPERATIONS
    # =================================================================
    
    def __add__(self, other: 'Shape') -> 'Shape':
        """Union: shape1 + shape2"""
        return Shape(BooleanNode(BooleanOp.UNION, self._node, other._node))
    
    def __sub__(self, other: 'Shape') -> 'Shape':
        """Difference: shape1 - shape2"""
        return Shape(BooleanNode(BooleanOp.DIFFERENCE, self._node, other._node))
    
    def __and__(self, other: 'Shape') -> 'Shape':
        """Intersection: shape1 & shape2"""
        return Shape(BooleanNode(BooleanOp.INTERSECTION, self._node, other._node))
    
    def union(self, *others: 'Shape') -> 'Shape':
        """Union with multiple shapes"""
        result = self
        for other in others:
            result = result + other
        return result
    
    def cut(self, *others: 'Shape') -> 'Shape':
        """Cut multiple shapes from this shape"""
        result = self
        for other in others:
            result = result - other
        return result
    
    def intersect(self, other: 'Shape') -> 'Shape':
        """Intersection with another shape"""
        return self & other
    
    # =================================================================
    # TRANSFORMATIONS
    # =================================================================
    
    def move(self, x: float = 0, y: float = 0, z: float = 0) -> 'Shape':
        """Translate shape"""
        return Shape(TransformNode(
            self._node,
            translation=Vector3D(x, y, z)
        ))
    
    def translate(self, x: float = 0, y: float = 0, z: float = 0) -> 'Shape':
        """Alias for move()"""
        return self.move(x, y, z)
    
    def rotate(self, angle: float, axis: Tuple[float, float, float] = (0, 0, 1),
               origin: Tuple[float, float, float] = (0, 0, 0)) -> 'Shape':
        """
        Rotate shape around axis
        
        Args:
            angle: Rotation angle in degrees
            axis: Rotation axis direction (x, y, z)
            origin: Pivot point
        """
        return Shape(TransformNode(
            self._node,
            rotation_axis=Axis3D(
                Point3D(*origin),
                Vector3D(*axis)
            ),
            rotation_angle=angle
        ))
    
    def rotate_x(self, angle: float) -> 'Shape':
        """Rotate around X axis"""
        return self.rotate(angle, axis=(1, 0, 0))
    
    def rotate_y(self, angle: float) -> 'Shape':
        """Rotate around Y axis"""
        return self.rotate(angle, axis=(0, 1, 0))
    
    def rotate_z(self, angle: float) -> 'Shape':
        """Rotate around Z axis"""
        return self.rotate(angle, axis=(0, 0, 1))
    
    def scale(self, factor: float) -> 'Shape':
        """Scale shape uniformly"""
        return Shape(TransformNode(self._node, scale=factor))
    
    def mirror(self, plane: str = 'xy') -> 'Shape':
        """
        Mirror shape across plane
        
        Args:
            plane: 'xy', 'yz', or 'xz'
        """
        # Create mirror transformation
        if plane == 'xy':
            # Mirror across XY plane (flip Z)
            transform = TransformNode(self._node, scale=1.0)
            # Need to implement proper mirroring
            return Shape(transform)  # Simplified
        elif plane == 'yz':
            return Shape(transform)
        elif plane == 'xz':
            return Shape(transform)
        else:
            raise ValueError(f"Unknown plane: {plane}")
    
    # =================================================================
    # FEATURE OPERATIONS
    # =================================================================
    
    def fillet(self, radius: float, edges: Optional[List[int]] = None) -> 'Shape':
        """
        Apply fillet (rounded edges)
        
        Args:
            radius: Fillet radius
            edges: List of edge indices (None = all edges)
        """
        return Shape(FilletNode(self._node, radius, edges))
    
    def chamfer(self, distance: float, edges: Optional[List[int]] = None) -> 'Shape':
        """
        Apply chamfer (beveled edges)
        
        Args:
            distance: Chamfer distance
            edges: List of edge indices (None = all edges)
        """
        return Shape(ChamferNode(self._node, distance, edges))
    
    def shell(self, thickness: float) -> 'Shape':
        """
        Hollow out shape (create thin wall)
        
        Args:
            thickness: Wall thickness
        """
        return Shape(ShellNode(self._node, thickness))
    
    # =================================================================
    # PATTERN OPERATIONS
    # =================================================================
    
    def linear_pattern(self, direction: Tuple[float, float, float],
                       count: int, spacing: float) -> 'Shape':
        """
        Create linear pattern
        
        Args:
            direction: Pattern direction
            count: Number of instances
            spacing: Distance between instances
        """
        return Shape(LinearPatternNode(
            self._node,
            Vector3D(*direction),
            count,
            spacing
        ))
    
    def circular_pattern(self, axis_point: Tuple[float, float, float],
                         axis_direction: Tuple[float, float, float],
                         count: int, total_angle: float = 360) -> 'Shape':
        """
        Create circular pattern
        
        Args:
            axis_point: Point on rotation axis
            axis_direction: Rotation axis direction
            count: Number of instances
            total_angle: Total sweep angle in degrees
        """
        return Shape(CircularPatternNode(
            self._node,
            Axis3D(Point3D(*axis_point), Vector3D(*axis_direction)),
            count,
            total_angle
        ))
    
    # =================================================================
    # ANALYSIS
    # =================================================================
    
    def bounds(self) -> BoundingBox:
        """Get bounding box"""
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        
        from OCP.BRepBndLib import BRepBndLib
        from OCP.Bnd import Bnd_Box
        
        shape = self.evaluate()
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        return BoundingBox(
            Point3D(xmin, ymin, zmin),
            Point3D(xmax, ymax, zmax)
        )
    
    def volume(self) -> float:
        """Calculate volume"""
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        
        shape = self.evaluate()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        return props.Mass()
    
    def center_of_mass(self) -> Point3D:
        """Calculate center of mass"""
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        
        shape = self.evaluate()
        props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, props)
        com = props.CentreOfMass()
        return Point3D(com.X(), com.Y(), com.Z())
    
    # =================================================================
    # EXPORT
    # =================================================================
    
    def export_step(self, filepath: Union[str, Path]) -> bool:
        """Export to STEP format"""
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        
        shape = self.evaluate()
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        return writer.Write(str(filepath))
    
    def export_iges(self, filepath: Union[str, Path]) -> bool:
        """Export to IGES format"""
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        
        shape = self.evaluate()
        writer = IGESControl_Writer()
        writer.AddShape(shape)
        return writer.Write(str(filepath))
    
    def mesh(self, max_element_size: float = 0.1, 
             element_type: str = "tet"):
        """
        Generate analysis mesh
        
        Args:
            max_element_size: Maximum element size
            element_type: 'tet', 'hex', or 'tri'
            
        Returns:
            Mesh object from meshing_engine
        """
        if not HAS_MESHING:
            raise RuntimeError("Meshing engine not available")
        
        # Export to temporary STEP file
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.step', delete=False) as f:
            temp_path = f.name
        
        try:
            self.export_step(temp_path)
            
            # Mesh using meshing engine
            with MeshingEngine() as engine:
                elem_type = ElementType.TET4 if element_type == "tet" else ElementType.HEX8
                params = MeshingParameters(
                    element_type=elem_type,
                    max_element_size=max_element_size
                )
                mesh = engine.generate_mesh_from_step(temp_path, params)
                return mesh
        finally:
            import os
            os.unlink(temp_path)
    
    def to_csg_tree(self) -> Dict[str, Any]:
        """Export CSG tree as dictionary"""
        return self._node.to_dict()
    
    def __repr__(self) -> str:
        return f"Shape({self.name})"


# =============================================================================
# PRIMITIVE FACTORY FUNCTIONS
# =============================================================================

def Box(width: float, height: float, depth: float,
        center: Tuple[float, float, float] = (0, 0, 0),
        name: str = "Box") -> Shape:
    """Create box primitive"""
    return Shape(BoxNode(width, height, depth, Point3D(*center), name))


def Cylinder(radius: float, height: float,
             center: Tuple[float, float, float] = (0, 0, 0),
             axis: Tuple[float, float, float] = (0, 0, 1),
             name: str = "Cylinder") -> Shape:
    """Create cylinder primitive"""
    return Shape(CylinderNode(radius, height, Point3D(*center), Vector3D(*axis), name))


def Sphere(radius: float,
           center: Tuple[float, float, float] = (0, 0, 0),
           name: str = "Sphere") -> Shape:
    """Create sphere primitive"""
    return Shape(SphereNode(radius, Point3D(*center), name))


def Cone(radius1: float, radius2: float, height: float,
         center: Tuple[float, float, float] = (0, 0, 0),
         axis: Tuple[float, float, float] = (0, 0, 1),
         name: str = "Cone") -> Shape:
    """Create cone primitive"""
    return Shape(ConeNode(radius1, radius2, height, Point3D(*center), Vector3D(*axis), name))


def Torus(major_radius: float, minor_radius: float,
          center: Tuple[float, float, float] = (0, 0, 0),
          axis: Tuple[float, float, float] = (0, 0, 1),
          name: str = "Torus") -> Shape:
    """Create torus primitive"""
    return Shape(TorusNode(major_radius, minor_radius, Point3D(*center), Vector3D(*axis), name))


# =============================================================================
# SKETCH-BASED FEATURES
# =============================================================================

def Extrude(sketch: Sketch, distance: float,
            direction: Tuple[float, float, float] = (0, 0, 1)) -> Shape:
    """Extrude a 2D sketch"""
    return Shape(ExtrudeNode(sketch, distance, Vector3D(*direction)))


def Revolve(sketch: Sketch, 
            axis_point: Tuple[float, float, float],
            axis_direction: Tuple[float, float, float],
            angle: float = 360) -> Shape:
    """Revolve a 2D sketch around an axis"""
    axis = Axis3D(Point3D(*axis_point), Vector3D(*axis_direction))
    return Shape(RevolveNode(sketch, axis, angle))


# =============================================================================
# COMPLEX SHAPE EXAMPLES
# =============================================================================

def create_bracket_with_holes(length: float = 100, 
                               width: float = 50, 
                               thickness: float = 10,
                               hole_diameter: float = 10,
                               hole_spacing: float = 60) -> Shape:
    """
    Create a mounting bracket with mounting holes
    
    Args:
        length: Overall length
        width: Overall width
        thickness: Material thickness
        hole_diameter: Diameter of mounting holes
        hole_spacing: Distance between hole centers
    """
    # Base plate
    base = Box(length, width, thickness)
    
    # Create holes
    hole_radius = hole_diameter / 2
    hole_depth = thickness + 1  # Ensure it cuts through
    
    # Position holes at corners
    offset_x = (length - hole_spacing) / 2
    offset_y = width / 4
    
    hole1 = Cylinder(hole_radius, hole_depth).move(offset_x, offset_y, -0.5)
    hole2 = Cylinder(hole_radius, hole_depth).move(offset_x + hole_spacing, offset_y, -0.5)
    hole3 = Cylinder(hole_radius, hole_depth).move(offset_x, offset_y + width/2, -0.5)
    hole4 = Cylinder(hole_radius, hole_depth).move(offset_x + hole_spacing, offset_y + width/2, -0.5)
    
    # Cut holes from base
    bracket = base - hole1 - hole2 - hole3 - hole4
    
    # Add fillets for strength
    bracket = bracket.fillet(2.0)
    
    return bracket


def create_flanged_pipe(outer_diameter: float = 50,
                        inner_diameter: float = 40,
                        length: float = 200,
                        flange_diameter: float = 80,
                        flange_thickness: float = 10) -> Shape:
    """
    Create a flanged pipe (pipe with flanges at both ends)
    
    Args:
        outer_diameter: Pipe outer diameter
        inner_diameter: Pipe inner diameter (for hollow pipe)
        length: Pipe length
        flange_diameter: Flange outer diameter
        flange_thickness: Flange thickness
    """
    outer_radius = outer_diameter / 2
    inner_radius = inner_diameter / 2
    flange_radius = flange_diameter / 2
    
    # Outer pipe
    pipe_outer = Cylinder(outer_radius, length)
    
    # Inner hole (for hollow pipe)
    pipe_inner = Cylinder(inner_radius, length + 2)
    
    # Pipe = outer - inner
    pipe = pipe_outer - pipe_inner.move(0, 0, -1)
    
    # Bottom flange
    flange1 = Cylinder(flange_radius, flange_thickness).move(0, 0, -flange_thickness)
    flange1_hole = Cylinder(inner_radius, flange_thickness + 1).move(0, 0, -flange_thickness - 0.5)
    flange1 = flange1 - flange1_hole
    
    # Top flange
    flange2 = Cylinder(flange_radius, flange_thickness).move(0, 0, length)
    flange2_hole = Cylinder(inner_radius, flange_thickness + 1).move(0, 0, length - 0.5)
    flange2 = flange2 - flange2_hole
    
    # Combine all
    flanged_pipe = pipe + flange1 + flange2
    
    return flanged_pipe


def create_gear_blank(outer_diameter: float = 100,
                      bore_diameter: float = 20,
                      thickness: float = 20,
                      hub_diameter: float = 40,
                      hub_thickness: float = 10) -> Shape:
    """
    Create a gear blank (cylinder with bore and hub)
    
    Args:
        outer_diameter: Gear outer diameter
        bore_diameter: Center hole diameter
        thickness: Main gear thickness
        hub_diameter: Hub diameter
        hub_thickness: Hub extension thickness
    """
    outer_radius = outer_diameter / 2
    bore_radius = bore_diameter / 2
    hub_radius = hub_diameter / 2
    
    # Main gear disk
    gear = Cylinder(outer_radius, thickness)
    
    # Bore (center hole)
    bore = Cylinder(bore_radius, thickness + 2)
    gear = gear - bore.move(0, 0, -1)
    
    # Hub (protruding cylinder on one side)
    hub = Cylinder(hub_radius, thickness + hub_thickness).move(0, 0, 0)
    
    # Combine (gear + hub, but bore goes through both)
    gear_with_hub = gear + hub
    
    return gear_with_hub


def create_box_with_fillets(length: float = 100,
                            width: float = 50,
                            height: float = 30,
                            fillet_radius: float = 5) -> Shape:
    """Create a box with all edges filleted"""
    box = Box(length, width, height)
    return box.fillet(fillet_radius)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("="*70)
    print("BRICK OS Geometry API Demo")
    print("="*70)
    
    if not HAS_OPENCASCADE:
        print("\nERROR: OpenCASCADE not available!")
        print("Install with: pip install cadquery-ocp")
        exit(1)
    
    # Example 1: Bracket with holes
    print("\n1. Creating bracket with holes...")
    bracket = create_bracket_with_holes(
        length=100, width=50, thickness=10,
        hole_diameter=8, hole_spacing=60
    )
    print(f"   Volume: {bracket.volume():.2f} mm³")
    print(f"   Center of mass: {bracket.center_of_mass()}")
    
    # Export bracket
    bracket.export_step("demo_bracket.step")
    print("   Exported to demo_bracket.step")
    
    # Example 2: Flanged pipe
    print("\n2. Creating flanged pipe...")
    pipe = create_flanged_pipe(
        outer_diameter=50, inner_diameter=40, length=150,
        flange_diameter=80, flange_thickness=10
    )
    print(f"   Volume: {pipe.volume():.2f} mm³")
    pipe.export_step("demo_pipe.step")
    print("   Exported to demo_pipe.step")
    
    # Example 3: Complex boolean (not just primitives)
    print("\n3. Creating complex shape (CSG operations)...")
    complex_shape = (
        Box(100, 100, 50)  # Base
        - Cylinder(15, 60).move(25, 25, -5)  # Hole 1
        - Cylinder(15, 60).move(75, 25, -5)  # Hole 2
        - Cylinder(15, 60).move(25, 75, -5)  # Hole 3
        - Cylinder(15, 60).move(75, 75, -5)  # Hole 4
        + Cylinder(10, 80).move(50, 50, 0)   # Boss in center
    )
    complex_shape = complex_shape.fillet(3)  # Round all edges
    print(f"   Volume: {complex_shape.volume():.2f} mm³")
    complex_shape.export_step("demo_complex.step")
    print("   Exported to demo_complex.step")
    
    # Example 4: Using sketch
    print("\n4. Creating extruded sketch...")
    sketch = Sketch()
    sketch.add_rectangle((0, 0), 50, 30)  # 50x30 rectangle
    
    extruded = Extrude(sketch, distance=20)
    print(f"   Volume: {extruded.volume():.2f} mm³")
    extruded.export_step("demo_extruded.step")
    print("   Exported to demo_extruded.step")
    
    print("\n" + "="*70)
    print("All demos completed successfully!")
    print("="*70)
