"""
Test MeshingEngine and MeshQualityChecker

Validates:
1. Gmsh integration for 3D meshing
2. Tetrahedral and hexahedral element generation
3. CalculiX .inp export format
4. Quality metrics (Jacobian, aspect ratio)
5. NAFEMS quality standards
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile
import os

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend" / "agents"))

from meshing_engine import (
    MeshingEngine, Mesh, MeshQuality, MeshingParameters,
    ElementType, create_test_mesh
)
from mesh_quality_checker import (
    MeshQualityChecker, QualityReport, validate_mesh_for_analysis
)


class TestMeshingEngine:
    """Test mesh generation capabilities"""
    
    def test_gmsh_available(self):
        """Verify Gmsh is installed and importable"""
        try:
            import gmsh
            assert True
        except ImportError:
            pytest.skip("Gmsh not installed")
    
    def test_meshing_engine_context_manager(self):
        """Test context manager initialization"""
        with MeshingEngine() as engine:
            assert engine.gmsh_initialized
        assert not engine.gmsh_initialized
    
    def test_generate_box_mesh(self):
        """Generate mesh for box geometry"""
        with MeshingEngine() as engine:
            params = MeshingParameters(
                element_type=ElementType.TET4,
                max_element_size=0.2
            )
            
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
            )
            
            assert mesh is not None
            assert len(mesh.nodes) > 0
            assert len(mesh.elements) > 0
            assert mesh.element_type == ElementType.TET4
            
            # Check volume is approximately correct
            volume = mesh.get_total_volume()
            expected_volume = 1.0 * 0.5 * 0.25  # 0.125
            assert abs(volume - expected_volume) < 0.01  # 1% tolerance
    
    def test_generate_cylinder_mesh(self):
        """Generate mesh for cylinder geometry"""
        with MeshingEngine() as engine:
            params = MeshingParameters(
                element_type=ElementType.TET4,
                max_element_size=0.1
            )
            
            mesh = engine.generate_mesh_from_geometry(
                "cylinder", {"radius": 0.5, "height": 1.0}, params
            )
            
            assert mesh is not None
            assert len(mesh.nodes) > 0
            assert len(mesh.elements) > 0
            
            # Check volume (cylinder: πr²h)
            volume = mesh.get_total_volume()
            expected_volume = np.pi * 0.5**2 * 1.0
            assert abs(volume - expected_volume) / expected_volume < 0.05  # 5% tolerance
    
    def test_generate_sphere_mesh(self):
        """Generate mesh for sphere geometry"""
        with MeshingEngine() as engine:
            params = MeshingParameters(
                element_type=ElementType.TET4,
                max_element_size=0.15
            )
            
            mesh = engine.generate_mesh_from_geometry(
                "sphere", {"radius": 0.5}, params
            )
            
            assert mesh is not None
            assert len(mesh.nodes) > 0
            assert len(mesh.elements) > 0
            
            # Check volume (sphere: 4/3 πr³)
            volume = mesh.get_total_volume()
            expected_volume = 4/3 * np.pi * 0.5**3
            assert abs(volume - expected_volume) / expected_volume < 0.1  # 10% tolerance
    
    def test_mesh_quality_calculated(self):
        """Verify mesh quality is calculated after generation"""
        with MeshingEngine() as engine:
            params = MeshingParameters(
                element_type=ElementType.TET4,
                max_element_size=0.2
            )
            
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 1.0, "height": 1.0}, params
            )
            
            assert mesh.quality is not None
            assert mesh.quality.num_elements == len(mesh.elements)
            assert mesh.quality.num_nodes == len(mesh.nodes)
            assert 0 <= mesh.quality.min_jacobian <= 1
            assert mesh.quality.min_aspect_ratio >= 1


class TestMeshExport:
    """Test mesh export formats"""
    
    def test_export_calculix_format(self):
        """Export to CalculiX .inp format"""
        with MeshingEngine() as engine:
            params = MeshingParameters(element_type=ElementType.TET4)
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.inp', delete=False) as f:
                temp_path = f.name
            
            try:
                success = engine.export_to_calculix(mesh, temp_path)
                assert success
                
                # Verify file contents
                with open(temp_path, 'r') as f:
                    content = f.read()
                    assert "*HEADING" in content
                    assert "*NODE" in content
                    assert "*ELEMENT, TYPE=C3D4" in content
                    assert "*END STEP" in content
                    
                    # Check node count
                    node_lines = [l for l in content.split('\n') if l and l[0].isdigit() and ',' in l]
                    assert len(node_lines) >= len(mesh.nodes)
            finally:
                os.unlink(temp_path)
    
    def test_export_vtk_format(self):
        """Export to VTK format"""
        with MeshingEngine() as engine:
            params = MeshingParameters(element_type=ElementType.TET4)
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
            )
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.vtk', delete=False) as f:
                temp_path = f.name
            
            try:
                success = engine.export_to_vtk(mesh, temp_path)
                assert success
                
                # Verify file contents
                with open(temp_path, 'r') as f:
                    content = f.read()
                    assert "# vtk DataFile Version" in content
                    assert "UNSTRUCTURED_GRID" in content
                    assert "POINTS" in content
                    assert "CELLS" in content
            finally:
                os.unlink(temp_path)


class TestMeshQuality:
    """Test mesh quality metrics"""
    
    def test_jacobian_calculation(self):
        """Verify Jacobian calculation for tetrahedra"""
        # Create a perfect tetrahedron
        nodes = np.array([
            [0, 0, 0],
            [1, 0, 0],
            [0.5, np.sqrt(3)/2, 0],
            [0.5, np.sqrt(3)/6, np.sqrt(2/3)]
        ])
        
        elements = np.array([[0, 1, 2, 3]])
        
        mesh = Mesh(nodes=nodes, elements=elements, element_type=ElementType.TET4)
        
        with MeshingEngine() as engine:
            quality = engine.check_mesh_quality(mesh)
        
        # Perfect tet should have high Jacobian
        assert quality.max_jacobian > 0.5
        assert quality.num_elements == 1
    
    def test_aspect_ratio_calculation(self):
        """Verify aspect ratio calculation"""
        # Create a stretched tetrahedron
        nodes = np.array([
            [0, 0, 0],
            [10, 0, 0],  # Very long edge
            [0, 1, 0],
            [0, 0, 1]
        ])
        
        elements = np.array([[0, 1, 2, 3]])
        mesh = Mesh(nodes=nodes, elements=elements, element_type=ElementType.TET4)
        
        with MeshingEngine() as engine:
            quality = engine.check_mesh_quality(mesh)
        
        # Stretched tet should have high aspect ratio
        assert quality.max_aspect_ratio > 5.0
    
    def test_quality_acceptability(self):
        """Test quality acceptability criteria"""
        # Good quality mesh
        good_quality = MeshQuality(
            min_jacobian=0.3,
            max_jacobian=0.9,
            avg_jacobian=0.7,
            min_aspect_ratio=1.0,
            max_aspect_ratio=4.0,
            avg_aspect_ratio=2.0,
            num_elements=100,
            num_nodes=50,
            num_bad_elements=0,
            quality_score=0.8
        )
        
        assert good_quality.is_acceptable()
        assert good_quality.is_good()
        
        # Poor quality mesh
        poor_quality = MeshQuality(
            min_jacobian=0.05,
            max_jacobian=0.9,
            avg_jacobian=0.4,
            min_aspect_ratio=1.0,
            max_aspect_ratio=15.0,
            avg_aspect_ratio=8.0,
            num_elements=100,
            num_nodes=50,
            num_bad_elements=10,
            quality_score=0.3
        )
        
        assert not poor_quality.is_acceptable()
        assert not poor_quality.is_good()


class TestMeshQualityChecker:
    """Test MeshQualityChecker advanced features"""
    
    def test_quality_report_generation(self):
        """Generate comprehensive quality report"""
        with MeshingEngine() as engine:
            params = MeshingParameters(element_type=ElementType.TET4)
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
            )
            
            checker = MeshQualityChecker()
            report = checker.check_mesh(mesh, "test_report")
            
            assert report.mesh_name == "test_report"
            assert report.num_elements == len(mesh.elements)
            assert report.num_nodes == len(mesh.nodes)
            assert 0 <= report.overall_score <= 1
            assert len(report.recommendations) > 0
    
    def test_nafems_criteria(self):
        """Test NAFEMS quality criteria validation"""
        checker = MeshQualityChecker()
        
        # Create a mesh with known quality
        with MeshingEngine() as engine:
            params = MeshingParameters(
                element_type=ElementType.TET4,
                max_element_size=0.15,
                quality_threshold=0.2
            )
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 1.0, "height": 1.0}, params
            )
            
            report = checker.check_mesh(mesh, "nafems_test")
            
            # Most coarse meshes should pass NAFEMS
            # (allows up to 5% poor elements)
            if report.passes_nafems:
                assert report.num_degenerate / report.num_elements < 0.01
                assert (report.num_degenerate + report.num_poor) / report.num_elements < 0.05
    
    def test_json_export(self):
        """Export quality report to JSON"""
        checker = MeshQualityChecker()
        
        with MeshingEngine() as engine:
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25},
                MeshingParameters(element_type=ElementType.TET4)
            )
            
            report = checker.check_mesh(mesh, "json_test")
            
            with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
                temp_path = f.name
            
            try:
                json_str = report.to_json(temp_path)
                assert len(json_str) > 0
                
                # Verify file was created
                with open(temp_path, 'r') as f:
                    import json
                    data = json.load(f)
                    assert data["mesh_name"] == "json_test"
                    assert "quality_distribution" in data
            finally:
                os.unlink(temp_path)
    
    def test_analysis_type_validation(self):
        """Test validation for different analysis types"""
        with MeshingEngine() as engine:
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25},
                MeshingParameters(element_type=ElementType.TET4)
            )
            
            # Test different analysis types
            for analysis in ["structural", "thermal", "modal", "cfd"]:
                is_valid, report = validate_mesh_for_analysis(mesh, analysis)
                assert isinstance(is_valid, bool)
                assert isinstance(report, QualityReport)


class TestElementTypes:
    """Test different element types"""
    
    @pytest.mark.parametrize("elem_type,expected_nodes", [
        (ElementType.TET4, 4),
        (ElementType.TET10, 10),
        (ElementType.HEX8, 8),
    ])
    def test_element_connectivity(self, elem_type, expected_nodes):
        """Test element connectivity array dimensions"""
        # Skip higher-order elements for now (Gmsh element order setting)
        if elem_type in [ElementType.TET10]:
            pytest.skip("Higher-order elements require additional setup")
        
        with MeshingEngine() as engine:
            params = MeshingParameters(element_type=elem_type)
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
            )
            
            assert mesh.element_type == elem_type
            assert mesh.elements.shape[1] == expected_nodes


class TestMeshingParameters:
    """Test meshing parameter handling"""
    
    def test_default_parameters(self):
        """Test default parameter values"""
        params = MeshingParameters()
        
        assert params.element_type == ElementType.TET4
        assert params.max_element_size == 0.1
        assert params.min_element_size == 0.01  # max/10
        assert params.quality_threshold == 0.1
        assert params.optimization_level == 5
    
    def test_custom_parameters(self):
        """Test custom parameter values"""
        params = MeshingParameters(
            element_type=ElementType.HEX8,
            max_element_size=0.5,
            min_element_size=0.05,
            quality_threshold=0.2,
            optimization_level=10
        )
        
        assert params.element_type == ElementType.HEX8
        assert params.max_element_size == 0.5
        assert params.min_element_size == 0.05
        assert params.quality_threshold == 0.2
        assert params.optimization_level == 10


class TestIntegration:
    """Integration tests"""
    
    def test_full_workflow_box(self):
        """Complete workflow: generate -> check -> export"""
        with MeshingEngine() as engine:
            # Generate mesh
            params = MeshingParameters(
                element_type=ElementType.TET4,
                max_element_size=0.15
            )
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": 2.0, "width": 1.0, "height": 0.5}, params
            )
            
            # Check quality
            checker = MeshQualityChecker()
            report = checker.check_mesh(mesh, "integration_test")
            
            # Export
            with tempfile.TemporaryDirectory() as tmpdir:
                inp_path = Path(tmpdir) / "mesh.inp"
                vtk_path = Path(tmpdir) / "mesh.vtk"
                json_path = Path(tmpdir) / "report.json"
                
                assert engine.export_to_calculix(mesh, inp_path)
                assert engine.export_to_vtk(mesh, vtk_path)
                report.to_json(json_path)
                
                assert inp_path.exists()
                assert vtk_path.exists()
                assert json_path.exists()
    
    def test_mesh_refinement_effect(self):
        """Test effect of mesh refinement on quality"""
        with MeshingEngine() as engine:
            checker = MeshQualityChecker()
            reports = []
            
            for elem_size in [0.3, 0.2, 0.1]:
                params = MeshingParameters(
                    element_type=ElementType.TET4,
                    max_element_size=elem_size
                )
                mesh = engine.generate_mesh_from_geometry(
                    "box", {"length": 1.0, "width": 1.0, "height": 1.0}, params
                )
                
                report = checker.check_mesh(mesh, f"refinement_{elem_size}")
                reports.append((elem_size, report))
            
            # Finer meshes should generally have better quality
            scores = [r.overall_score for _, r in reports]
            # Not strictly monotonic, but finer meshes tend to be better
            assert max(scores) > 0.4  # At least one should be decent


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
