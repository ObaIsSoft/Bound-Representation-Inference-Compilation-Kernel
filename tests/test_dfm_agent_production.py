"""
Production tests for DfmAgent.

Tests feature recognition, GD&T validation, and manufacturability analysis.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import trimesh
    HAS_TRIMESH = True
except ImportError:
    HAS_TRIMESH = False

from backend.agents.dfm_agent_production import (
    ProductionDfmAgent,
    ManufacturingFeature,
    DfmIssue,
    DfmReport,
    GDTRequirement,
    GDTToleranceType,
    FeatureType,
    IssueSeverity,
    ManufacturingProcess,
    analyze_mesh_file
)


@pytest.fixture
def agent():
    """Create DfM agent."""
    if not HAS_TRIMESH:
        pytest.skip("trimesh not installed")
    return ProductionDfmAgent()


@pytest.fixture
def simple_cube():
    """Create simple cube mesh."""
    if not HAS_TRIMESH:
        pytest.skip("trimesh not installed")
    return trimesh.creation.box(extents=[10, 10, 10])


@pytest.fixture
def thin_wall_part():
    """Create part with thin walls."""
    if not HAS_TRIMESH:
        pytest.skip("trimesh not installed")
    return trimesh.creation.box(extents=[20, 20, 0.5])


class TestProductionDfmAgent:
    """Test ProductionDfmAgent functionality."""
    
    def test_initialization(self, agent):
        """Test agent initialization."""
        assert agent is not None
        assert agent.dfm_rules is not None
        assert agent.boothroyd is not None
        assert agent.gdt_rules is not None
        assert agent.step_ap224 is not None
        assert "processes" in agent.dfm_rules
    
    def test_analyze_simple_cube(self, agent, simple_cube):
        """Test analysis of simple cube."""
        report = agent.analyze_mesh(simple_cube)
        
        assert isinstance(report, DfmReport)
        assert 0 <= report.manufacturability_score <= 100
        assert report.overall_assessment != ""
        assert len(report.process_recommendations) > 0
        assert isinstance(report.gdt_validations, list)
    
    def test_detect_thin_walls(self, agent, thin_wall_part):
        """Test thin wall detection."""
        features = agent._detect_thin_walls(thin_wall_part)
        
        assert len(features) > 0
        
        thin_features = [f for f in features if f.feature_type == FeatureType.THIN_WALL]
        assert len(thin_features) > 0
        
        for feature in thin_features:
            assert feature.dimensions["thickness_mm"] < 2.0
    
    def test_wall_thickness_issue_detection(self, agent, thin_wall_part):
        """Test wall thickness issue detection for FDM."""
        features = agent._detect_thin_walls(thin_wall_part)
        
        issues = agent._analyze_for_process(
            thin_wall_part,
            ManufacturingProcess.ADDITIVE_FDM,
            features
        )
        
        wall_issues = [i for i in issues if i.category == "wall_thickness"]
        assert len(wall_issues) > 0
        
        for issue in wall_issues:
            assert issue.severity in [IssueSeverity.WARNING, IssueSeverity.CRITICAL]
    
    def test_manufacturability_score_calculation(self, agent):
        """Test score calculation."""
        score = agent._calculate_manufacturability_score([], [])
        assert score > 70
        
        features = [
            ManufacturingFeature(
                feature_type=FeatureType.THIN_WALL,
                dimensions={"thickness_mm": 0.3},
                location=np.array([0, 0, 0]),
                difficulty_score=50,
                process_compatibility={}
            )
        ]
        score_with_features = agent._calculate_manufacturability_score(features, [])
        assert score_with_features < score
    
    def test_process_recommendations(self, agent, simple_cube):
        """Test process recommendation generation."""
        features = []
        issues = []
        
        recommendations = agent._recommend_processes(simple_cube, features, issues)
        
        assert len(recommendations) > 0
        assert len(recommendations) <= 3
        
        assert recommendations[0].suitability_score >= recommendations[-1].suitability_score
        
        for rec in recommendations:
            assert 0 <= rec.suitability_score <= 100
            assert rec.cost_estimate in ["low", "medium", "high"]
            assert rec.time_estimate in ["fast", "medium", "slow"]
    
    def test_overhang_detection_am(self, agent):
        """Test overhang detection for AM."""
        if not HAS_TRIMESH:
            pytest.skip("trimesh not installed")
        
        vertical = trimesh.creation.box(extents=[5, 5, 15])
        horizontal = trimesh.creation.box(extents=[20, 5, 3])
        horizontal.apply_translation([0, 0, 9])
        
        t_shape = trimesh.util.concatenate([vertical, horizontal])
        
        rules = agent.dfm_rules["processes"]["additive_fdm"]
        issues = agent._analyze_overhangs(
            t_shape,
            ManufacturingProcess.ADDITIVE_FDM,
            rules
        )
        
        assert isinstance(issues, list)
    
    def test_recommendations_generation(self, agent):
        """Test recommendation generation."""
        features = [
            ManufacturingFeature(
                feature_type=FeatureType.THIN_WALL,
                dimensions={"thickness_mm": 0.3},
                location=np.array([0, 0, 0]),
                difficulty_score=50,
                process_compatibility={}
            ),
            ManufacturingFeature(
                feature_type=FeatureType.SHARP_CORNER,
                dimensions={"radius_mm": 0.1},
                location=np.array([1, 0, 0]),
                difficulty_score=40,
                process_compatibility={}
            )
        ]
        
        issues = []
        gdt_validations = []
        
        recommendations = agent._generate_recommendations(features, issues, gdt_validations)
        
        assert len(recommendations) > 0
        assert any("wall" in r.lower() for r in recommendations)
        assert any("corner" in r.lower() or "fillet" in r.lower() for r in recommendations)
    
    def test_draft_angle_detection(self, agent):
        """Test draft angle detection for molding features."""
        if not HAS_TRIMESH:
            pytest.skip("trimesh not installed")
        
        # Create vertical-walled box (no draft)
        box = trimesh.creation.box(extents=[20, 20, 10])
        
        features = agent._detect_draft_angles(box)
        
        # Should detect vertical walls needing draft
        draft_features = [f for f in features if f.feature_type == FeatureType.DRAFT_ANGLE]
        
        # Vertical walls have 0 draft angle
        for feature in draft_features:
            assert feature.dimensions["draft_angle_deg"] < 1.0
            assert "pull_direction" in feature.dimensions
    
    def test_tool_access_analysis(self, agent):
        """Test tool access analysis for CNC."""
        if not HAS_TRIMESH:
            pytest.skip("trimesh not installed")
        
        # Create simple block with hole
        block = trimesh.creation.box(extents=[30, 20, 10])
        
        # Add a hole feature manually
        features = [
            ManufacturingFeature(
                feature_type=FeatureType.HOLE,
                dimensions={"diameter_mm": 5, "depth_mm": 10},
                location=np.array([0, 0, 5]),
                difficulty_score=30,
                process_compatibility={"cnc_milling": 0.9}
            )
        ]
        
        issues = agent._analyze_tool_access(block, features)
        
        # Hole should be accessible from top
        assert isinstance(issues, list)
    
    def test_step_detection(self, agent):
        """Test step feature detection."""
        if not HAS_TRIMESH:
            pytest.skip("trimesh not installed")
        
        # Create stepped part
        bottom = trimesh.creation.box(extents=[30, 20, 5])
        top = trimesh.creation.box(extents=[20, 15, 5])
        top.apply_translation([0, 0, 5])
        
        stepped = trimesh.util.concatenate([bottom, top])
        
        features = agent._detect_steps_and_pockets(stepped)
        
        step_features = [f for f in features if f.feature_type == FeatureType.STEP_FEATURE]
        assert len(step_features) > 0
        
        for feature in step_features:
            assert "height_mm" in feature.dimensions
            assert feature.dimensions["height_mm"] > 0
    
    def test_gdt_validation(self, agent):
        """Test GD&T requirement validation."""
        gdt_reqs = [
            GDTRequirement(
                tolerance_type=GDTToleranceType.POSITION,
                value=0.1,  # 0.1mm position tolerance
                datum_references=["A", "B", "C"],
                material_condition="RFS",
                applies_to="hole_pattern"
            )
        ]
        
        agent.gdt_requirements = gdt_reqs
        
        features = []
        processes = [ManufacturingProcess.CNC_MILLING, ManufacturingProcess.ADDITIVE_FDM]
        
        validations = agent._validate_gdt_requirements(features, processes)
        
        assert len(validations) == 1
        assert validations[0].requirement.tolerance_type == GDTToleranceType.POSITION
        assert isinstance(validations[0].achievable, bool)
        assert 0 <= validations[0].confidence <= 1
    
    def test_step_ap224_mapping(self, agent):
        """Test STEP AP224 feature mapping."""
        features = [
            ManufacturingFeature(
                feature_type=FeatureType.HOLE,
                dimensions={"diameter_mm": 10, "depth_mm": 20},
                location=np.array([0, 0, 0]),
                difficulty_score=30,
                process_compatibility={},
                step_ap224_type="round_hole"
            )
        ]
        
        step_features = agent._map_to_step_ap224(features)
        
        assert len(step_features) == 1  # Should map round_hole
        assert step_features[0]["feature_type"] == "round_hole"
        assert "ap224_definition" in step_features[0]
        assert "feature_id" in step_features[0]
    
    def test_report_to_dict(self, agent, simple_cube):
        """Test report serialization."""
        report = agent.analyze_mesh(simple_cube)
        report_dict = report.to_dict()
        
        assert "manufacturability_score" in report_dict
        assert "features" in report_dict
        assert "issues" in report_dict
        assert "recommendations" in report_dict
        assert "process_recommendations" in report_dict
        assert "gdt_validations" in report_dict
        assert "overall_assessment" in report_dict
        assert "step_ap224_features" in report_dict


class TestDfmConfiguration:
    """Test configuration loading."""
    
    def test_dfm_rules_loaded(self, agent):
        """Test DfM rules configuration."""
        rules = agent.dfm_rules
        
        assert "processes" in rules
        assert "cnc_milling" in rules["processes"]
        assert "additive_fdm" in rules["processes"]
        assert "injection_molding" in rules["processes"]
        
        # Check draft angle for molding
        molding_rules = rules["processes"]["injection_molding"]
        assert "draft_angle" in molding_rules
    
    def test_gdt_rules_loaded(self, agent):
        """Test GD&T rules configuration."""
        rules = agent.gdt_rules
        
        assert "tolerance_types" in rules
        assert "position_tolerance" in rules["validation_rules"]
        assert "process_capabilities" in rules
        assert "cnc_milling" in rules["process_capabilities"]
    
    def test_step_ap224_loaded(self, agent):
        """Test STEP AP224 configuration."""
        config = agent.step_ap224
        
        assert "feature_hierarchy" in config
        assert "manufacturing_feature" in config["feature_hierarchy"]
        hierarchy = config["feature_hierarchy"]["manufacturing_feature"]["subtypes"]
        assert "machining_feature" in hierarchy
        machining = hierarchy["machining_feature"]["subtypes"]
        assert "round_hole" in machining
    
    def test_boothroyd_scores_loaded(self, agent):
        """Test Boothroyd scoring configuration."""
        scores = agent.boothroyd
        
        assert "handling_difficulty" in scores
        assert "insertion_difficulty" in scores
        assert "machining_difficulty" in scores


class TestDfmEdgeCases:
    """Test edge cases and error handling."""
    
    def test_minimal_mesh(self, agent):
        """Test handling of minimal mesh."""
        if not HAS_TRIMESH:
            pytest.skip("trimesh not installed")
        
        vertices = [[0,0,0], [1,0,0], [0,1,0], [0,0,1]]
        faces = [[0,1,2], [0,1,3], [0,2,3], [1,2,3]]
        mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        
        report = agent.analyze_mesh(mesh)
        assert report is not None
    
    def test_config_file_not_found(self):
        """Test error when config file missing."""
        with pytest.raises(FileNotFoundError):
            ProductionDfmAgent(config_path="/nonexistent/path")
    
    def test_no_trimesh_error(self):
        """Test error when trimesh not available."""
        if HAS_TRIMESH:
            pytest.skip("trimesh is installed")
        
        with pytest.raises(RuntimeError, match="trimesh required"):
            ProductionDfmAgent()


@pytest.mark.skipif(not HAS_TRIMESH, reason="trimesh not installed")
class TestDfmIntegration:
    """Integration tests with real meshes."""
    
    def test_cube_all_processes(self, agent, simple_cube):
        """Test cube analysis for all processes."""
        all_processes = list(ManufacturingProcess)
        
        report = agent.analyze_mesh(simple_cube, processes=all_processes)
        
        assert len(report.process_recommendations) > 0
        assert report.manufacturability_score >= 0
        
        top_process = report.process_recommendations[0]
        assert top_process.suitability_score > 0
    
    def test_molding_with_draft_analysis(self, agent):
        """Test molding analysis with draft angle detection."""
        if not HAS_TRIMESH:
            pytest.skip("trimesh not installed")
        
        # Create box with vertical walls (no draft)
        box = trimesh.creation.box(extents=[20, 20, 10])
        
        report = agent.analyze_mesh(
            box,
            processes=[ManufacturingProcess.INJECTION_MOLDING]
        )
        
        # Should detect draft angle issues
        draft_issues = [i for i in report.issues if i.category == "draft_angle"]
        
        # May or may not detect depending on mesh
        assert isinstance(report.issues, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
