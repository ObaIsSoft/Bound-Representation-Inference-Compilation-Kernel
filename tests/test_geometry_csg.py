"""
Test CSG Geometry System

Tests:
1. Primitive creation (Box, Cylinder, Sphere, Cone, Torus)
2. Boolean operations (union, difference, intersection)
3. Transformations (translate, rotate, scale)
4. Feature operations (fillet, chamfer, shell)
5. Sketch-based features (extrude, revolve)
6. Pattern operations (linear, circular)
7. Complex assemblies
"""

import pytest
import sys
from pathlib import Path
import math

sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "agents"))

from geometry_api import (
    Shape, Box, Cylinder, Sphere, Cone, Torus,
    Sketch, Extrude, Revolve,
    create_bracket_with_holes, create_flanged_pipe, create_gear_blank
)
from csg_geometry_kernel import (
    Point3D, Vector3D, BooleanOp
)

# Check OpenCASCADE availability
try:
    from OCP.gp import gp_Pnt
    HAS_OPENCASCADE = True
except ImportError:
    HAS_OPENCASCADE = False


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestPrimitives:
    """Test geometric primitives"""
    
    def test_box_creation(self):
        """Create box and verify properties"""
        box = Box(10, 5, 3)
        
        # Check volume
        volume = box.volume()
        assert abs(volume - 150) < 0.1  # 10 * 5 * 3 = 150
        
        # Check center of mass
        com = box.center_of_mass()
        assert abs(com.x) < 0.01
        assert abs(com.y) < 0.01
        assert abs(com.z) < 0.01
    
    def test_cylinder_creation(self):
        """Create cylinder and verify properties"""
        cyl = Cylinder(radius=5, height=20)
        
        volume = cyl.volume()
        expected = math.pi * 25 * 20  # πr²h
        assert abs(volume - expected) < 0.1
    
    def test_sphere_creation(self):
        """Create sphere and verify properties"""
        sphere = Sphere(radius=5)
        
        volume = sphere.volume()
        expected = 4/3 * math.pi * 125  # 4/3 πr³
        assert abs(volume - expected) < 0.1
    
    def test_cone_creation(self):
        """Create cone and verify properties"""
        cone = Cone(radius1=10, radius2=5, height=15)
        
        volume = cone.volume()
        # Frustum volume: (πh/3)(r1² + r1*r2 + r2²)
        expected = (math.pi * 15 / 3) * (100 + 50 + 25)
        assert abs(volume - expected) < 0.1
    
    def test_torus_creation(self):
        """Create torus and verify properties"""
        torus = Torus(major_radius=10, minor_radius=3)
        
        volume = torus.volume()
        # Torus volume: 2π²Rr²
        expected = 2 * math.pi**2 * 10 * 9
        assert abs(volume - expected) < 1.0  # Higher tolerance


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestBooleanOperations:
    """Test CSG boolean operations"""
    
    def test_union(self):
        """Test union operation"""
        box1 = Box(10, 10, 5).move(-5, 0, 0)
        box2 = Box(10, 10, 5).move(5, 0, 0)
        
        combined = box1 + box2
        
        # Volume should be approximately sum minus overlap
        vol = combined.volume()
        assert vol > 800  # Should be more than single box (500)
        assert vol < 1000  # But less than two non-overlapping (1000)
    
    def test_difference(self):
        """Test difference (cut) operation"""
        box = Box(10, 10, 10)
        hole = Cylinder(radius=2, height=12)
        
        result = box - hole.move(0, 0, -1)
        
        # Volume should be reduced by hole
        original_vol = box.volume()
        result_vol = result.volume()
        
        assert result_vol < original_vol
        # Hole volume ≈ π * 4 * 10 = 125.6
        assert abs((original_vol - result_vol) - 125.6) < 10
    
    def test_intersection(self):
        """Test intersection operation"""
        box = Box(10, 10, 10)
        cylinder = Cylinder(radius=4, height=20)
        
        result = box & cylinder
        
        # Result should be smaller than both inputs
        assert result.volume() < box.volume()
        assert result.volume() < cylinder.volume()
    
    def test_multiple_cuts(self):
        """Multiple holes in a plate"""
        plate = Box(100, 50, 10)
        
        holes = []
        for i in range(4):
            x = 20 + (i % 2) * 60
            y = 15 + (i // 2) * 20
            holes.append(Cylinder(radius=5, height=12).move(x, y, -1))
        
        # Cut all holes
        result = plate
        for hole in holes:
            result = result - hole
        
        assert result.volume() < plate.volume()
    
    def test_complex_boolean(self):
        """Complex boolean: box - cylinder + sphere"""
        base = Box(50, 50, 30)
        hole = Cylinder(radius=10, height=35).move(0, 0, -2.5)
        boss = Sphere(radius=15).move(25, 0, 15)
        
        result = base - hole + boss
        
        vol = result.volume()
        assert vol > 0


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestTransformations:
    """Test geometric transformations"""
    
    def test_translation(self):
        """Test move/translate operation"""
        box = Box(10, 10, 10).move(5, 10, 15)
        
        com = box.center_of_mass()
        assert abs(com.x - 5) < 0.01
        assert abs(com.y - 10) < 0.01
        assert abs(com.z - 15) < 0.01
    
    def test_rotation(self):
        """Test rotation operation"""
        # Create a tall cylinder
        cyl = Cylinder(radius=5, height=50)
        
        # Rotate 90 degrees around X axis
        rotated = cyl.rotate_x(90)
        
        # Bounding box should be different
        bounds = rotated.bounds()
        # After 90° rotation around X, Z becomes Y and Y becomes -Z
        dim_x, dim_y, dim_z = bounds.dimensions()
        
        # Volume should be unchanged
        assert abs(rotated.volume() - cyl.volume()) < 0.1
    
    def test_scale(self):
        """Test scale operation"""
        box = Box(10, 10, 10)
        scaled = box.scale(2)
        
        # Volume should scale by 2³ = 8
        assert abs(scaled.volume() - 8 * box.volume()) < 0.1


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestFeatures:
    """Test feature operations"""
    
    def test_fillet(self):
        """Test fillet operation"""
        box = Box(50, 50, 30)
        filleted = box.fillet(radius=5)
        
        # Filleted shape should have slightly less volume
        # (corners are rounded off)
        assert filleted.volume() < box.volume()
        assert filleted.volume() > 0.9 * box.volume()  # Not too much removed
    
    def test_chamfer(self):
        """Test chamfer operation"""
        box = Box(50, 50, 30)
        chamfered = box.chamfer(distance=3)
        
        # Chamfered shape should have less volume
        assert chamfered.volume() < box.volume()
    
    def test_shell(self):
        """Test shell operation (hollow out)"""
        box = Box(50, 50, 30)
        shelled = box.shell(thickness=2)
        
        # Shelled shape should have less volume
        assert shelled.volume() < box.volume()
        # But still significant volume
        assert shelled.volume() > 0.5 * box.volume()


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestSketches:
    """Test sketch-based features"""
    
    def test_sketch_rectangle(self):
        """Create sketch with rectangle"""
        sketch = Sketch()
        sketch.add_rectangle((0, 0), 50, 30)
        
        # Extrude
        result = Extrude(sketch, distance=20)
        
        # Volume should be approximately 50 * 30 * 20
        expected = 50 * 30 * 20
        assert abs(result.volume() - expected) < 100  # Tolerance for meshing
    
    def test_sketch_circle(self):
        """Create sketch with circle"""
        sketch = Sketch()
        sketch.add_circle((0, 0), radius=20)
        
        result = Extrude(sketch, distance=10)
        
        # Volume should be approximately π * 400 * 10
        expected = math.pi * 400 * 10
        assert abs(result.volume() - expected) < 50


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestPatternOperations:
    """Test pattern operations"""
    
    def test_linear_pattern(self):
        """Test linear pattern"""
        box = Box(10, 10, 5)
        pattern = box.linear_pattern(direction=(1, 0, 0), count=3, spacing=15)
        
        # Volume should be approximately 3x single box
        assert abs(pattern.volume() - 3 * box.volume()) < 50
    
    def test_circular_pattern(self):
        """Test circular pattern"""
        box = Box(10, 10, 5).move(20, 0, 0)
        pattern = box.circular_pattern(
            axis_point=(0, 0, 0),
            axis_direction=(0, 0, 1),
            count=4,
            total_angle=360
        )
        
        # Volume should be approximately 4x single box
        assert abs(pattern.volume() - 4 * box.volume()) < 100


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestComplexShapes:
    """Test complex pre-defined shapes"""
    
    def test_bracket_with_holes(self):
        """Create bracket with mounting holes"""
        bracket = create_bracket_with_holes(
            length=100, width=50, thickness=10,
            hole_diameter=8, hole_spacing=60
        )
        
        # Volume should be less than solid plate
        solid_volume = 100 * 50 * 10
        assert bracket.volume() < solid_volume
        assert bracket.volume() > 0.8 * solid_volume  # Not too many holes
    
    def test_flanged_pipe(self):
        """Create flanged pipe"""
        pipe = create_flanged_pipe(
            outer_diameter=50, inner_diameter=40, length=150,
            flange_diameter=80, flange_thickness=10
        )
        
        volume = pipe.volume()
        assert volume > 0
        
        # Volume should be less than solid cylinder
        solid_volume = math.pi * 25**2 * 170  # Including flanges
        assert volume < solid_volume
    
    def test_gear_blank(self):
        """Create gear blank"""
        gear = create_gear_blank(
            outer_diameter=100, bore_diameter=20,
            thickness=20, hub_diameter=40, hub_thickness=10
        )
        
        volume = gear.volume()
        assert volume > 0


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestExport:
    """Test export functionality"""
    
    def test_step_export(self, tmp_path):
        """Export to STEP format"""
        box = Box(10, 10, 10)
        
        filepath = tmp_path / "test_box.step"
        success = box.export_step(filepath)
        
        assert success
        assert filepath.exists()
        assert filepath.stat().st_size > 0
    
    def test_iges_export(self, tmp_path):
        """Export to IGES format"""
        cyl = Cylinder(radius=5, height=20)
        
        filepath = tmp_path / "test_cyl.iges"
        success = cyl.export_iges(filepath)
        
        assert success
        assert filepath.exists()


@pytest.mark.skipif(not HAS_OPENCASCADE, reason="OpenCASCADE not available")
class TestCSGTree:
    """Test CSG tree serialization"""
    
    def test_csg_tree_export(self):
        """Export CSG tree to dictionary"""
        box = Box(10, 10, 10)
        hole = Cylinder(radius=2, height=12)
        result = box - hole
        
        tree = result.to_csg_tree()
        
        assert "type" in tree
        assert tree["type"] == "BooleanNode"
        assert "operation" in tree
        assert tree["operation"] == "DIFFERENCE"
    
    def test_complex_csg_tree(self):
        """Complex CSG tree structure"""
        base = Box(100, 100, 50)
        hole1 = Cylinder(radius=10, height=60).move(-30, -30, -5)
        hole2 = Cylinder(radius=10, height=60).move(30, 30, -5)
        boss = Cylinder(radius=20, height=70).move(0, 0, -10)
        
        result = base - hole1 - hole2 + boss
        
        tree = result.to_csg_tree()
        
        # Should have nested boolean operations
        assert tree["type"] == "BooleanNode"
        assert "left" in tree
        assert "right" in tree


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
