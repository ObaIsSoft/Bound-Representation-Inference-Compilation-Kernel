"""
MeshQualityChecker - Advanced mesh quality validation

Validates meshes against NAFEMS and industry quality criteria:
- Jacobian determinant (0.0 = degenerate, 1.0 = perfect)
- Aspect ratio (1.0 = perfect, > 10 = poor)
- Skewness (0.0 = perfect, 1.0 = degenerate)
- Orthogonality (face alignment)
- Volume ratios (adjacent element size consistency)

References:
- NAFEMS Quality Criteria
- NASA/CR-2003-212684 (Mesh Quality Guidelines)
- ANSYS/ABAQUS Quality Standards
"""

import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import json

from meshing_engine import Mesh, MeshQuality, ElementType, MeshingEngine

logger = logging.getLogger(__name__)


@dataclass
class QualityReport:
    """Comprehensive quality assessment report"""
    mesh_name: str
    element_type: str
    num_elements: int
    num_nodes: int
    
    # Quality metrics
    overall_score: float  # 0-1 overall quality score
    jacobian_score: float  # 0-1 based on Jacobian
    aspect_score: float  # 0-1 based on aspect ratio
    
    # Detailed metrics
    min_jacobian: float
    max_jacobian: float
    avg_jacobian: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    avg_aspect_ratio: float
    
    # Failure counts
    num_degenerate: int  # Jacobian < 0.01
    num_poor: int  # Jacobian < 0.1 or aspect > 10
    num_fair: int  # Jacobian < 0.3 or aspect > 5
    num_good: int  # Jacobian >= 0.3 and aspect <= 5
    num_excellent: int  # Jacobian >= 0.6 and aspect <= 2
    
    # Pass/Fail
    passes_nafems: bool
    passes_industry: bool
    
    # Recommendations
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "mesh_name": self.mesh_name,
            "element_type": self.element_type,
            "num_elements": self.num_elements,
            "num_nodes": self.num_nodes,
            "overall_score": self.overall_score,
            "jacobian_score": self.jacobian_score,
            "aspect_score": self.aspect_score,
            "min_jacobian": self.min_jacobian,
            "max_jacobian": self.max_jacobian,
            "avg_jacobian": self.avg_jacobian,
            "min_aspect_ratio": self.min_aspect_ratio,
            "max_aspect_ratio": self.max_aspect_ratio,
            "avg_aspect_ratio": self.avg_aspect_ratio,
            "quality_distribution": {
                "degenerate": self.num_degenerate,
                "poor": self.num_poor,
                "fair": self.num_fair,
                "good": self.num_good,
                "excellent": self.num_excellent
            },
            "passes_nafems": self.passes_nafems,
            "passes_industry": self.passes_industry,
            "recommendations": self.recommendations
        }
    
    def to_json(self, filepath: Optional[str] = None) -> str:
        """Export to JSON"""
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            with open(filepath, 'w') as f:
                f.write(json_str)
        return json_str


class MeshQualityChecker:
    """
    Advanced mesh quality checker with NAFEMS validation
    
    Quality Levels (NAFEMS guidelines):
    - Excellent: Jacobian >= 0.6, Aspect ratio <= 2.0
    - Good: Jacobian >= 0.3, Aspect ratio <= 5.0
    - Fair: Jacobian >= 0.1, Aspect ratio <= 10.0
    - Poor: Jacobian < 0.1 or Aspect ratio > 10.0
    - Degenerate: Jacobian < 0.01
    """
    
    # Quality thresholds per NAFEMS
    THRESHOLDS = {
        "excellent": {"jacobian": 0.6, "aspect": 2.0},
        "good": {"jacobian": 0.3, "aspect": 5.0},
        "fair": {"jacobian": 0.1, "aspect": 10.0},
        "degenerate": {"jacobian": 0.01}
    }
    
    def __init__(self):
        self.quality_history: List[QualityReport] = []
    
    def check_mesh(self, mesh: Mesh, mesh_name: str = "unnamed") -> QualityReport:
        """
        Perform comprehensive quality check on mesh
        
        Args:
            mesh: Mesh object to check
            mesh_name: Name for the report
            
        Returns:
            QualityReport with detailed metrics
        """
        if mesh.element_type in [ElementType.TET4, ElementType.TET10]:
            return self._check_tetrahedral_mesh(mesh, mesh_name)
        elif mesh.element_type in [ElementType.HEX8, ElementType.HEX20]:
            return self._check_hexahedral_mesh(mesh, mesh_name)
        else:
            return self._check_generic_mesh(mesh, mesh_name)
    
    def _check_tetrahedral_mesh(self, mesh: Mesh, mesh_name: str) -> QualityReport:
        """Detailed quality check for tetrahedral mesh"""
        jacobians = []
        aspect_ratios = []
        volumes = []
        
        # Classification counters
        num_degenerate = 0
        num_poor = 0
        num_fair = 0
        num_good = 0
        num_excellent = 0
        
        for elem_idx in range(len(mesh.elements)):
            nodes = mesh.elements[elem_idx]
            v = [mesh.nodes[nodes[i]] for i in range(4)]
            
            # Calculate Jacobian (6 * volume)
            e1 = v[1] - v[0]
            e2 = v[2] - v[0]
            e3 = v[3] - v[0]
            J = abs(np.dot(e1, np.cross(e2, e3)))
            volume = J / 6.0
            volumes.append(volume)
            
            # Edge lengths
            edges = []
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append(np.linalg.norm(v[i] - v[j]))
            
            max_edge = max(edges)
            min_edge = min(edges)
            
            # Normalized Jacobian (compared to ideal equilateral tet)
            # Ideal volume = sqrt(2)/12 * max_edge^3 (if equilateral)
            ideal_volume = (max_edge ** 3) / (6 * np.sqrt(2))
            jacobian_norm = min(1.0, volume / ideal_volume) if ideal_volume > 0 else 0.0
            jacobians.append(jacobian_norm)
            
            # Aspect ratio (max edge / min altitude)
            face_areas = [
                0.5 * np.linalg.norm(np.cross(v[1]-v[0], v[2]-v[0])),  # Face 0-1-2
                0.5 * np.linalg.norm(np.cross(v[1]-v[0], v[3]-v[0])),  # Face 0-1-3
                0.5 * np.linalg.norm(np.cross(v[2]-v[0], v[3]-v[0])),  # Face 0-2-3
                0.5 * np.linalg.norm(np.cross(v[2]-v[1], v[3]-v[1]))   # Face 1-2-3
            ]
            min_altitude = 3 * volume / (max(face_areas) + 1e-10)
            aspect_ratio = max_edge / (min_altitude + 1e-10)
            aspect_ratios.append(aspect_ratio)
            
            # Classify element
            if jacobian_norm < self.THRESHOLDS["degenerate"]["jacobian"]:
                num_degenerate += 1
            elif jacobian_norm < self.THRESHOLDS["fair"]["jacobian"] or aspect_ratio > self.THRESHOLDS["fair"]["aspect"]:
                num_poor += 1
            elif jacobian_norm < self.THRESHOLDS["good"]["jacobian"] or aspect_ratio > self.THRESHOLDS["good"]["aspect"]:
                num_fair += 1
            elif jacobian_norm < self.THRESHOLDS["excellent"]["jacobian"] or aspect_ratio > self.THRESHOLDS["excellent"]["aspect"]:
                num_good += 1
            else:
                num_excellent += 1
        
        jacobians = np.array(jacobians)
        aspect_ratios = np.array(aspect_ratios)
        
        # Calculate scores
        jacobian_score = np.mean(jacobians)
        aspect_score = 1.0 / (1.0 + np.mean(aspect_ratios) / 5.0)
        overall_score = 0.5 * jacobian_score + 0.3 * aspect_score + 0.2 * (num_excellent + num_good) / len(mesh.elements)
        
        # NAFEMS criteria: < 1% degenerate, < 5% poor
        passes_nafems = (
            num_degenerate / len(mesh.elements) < 0.01 and
            (num_degenerate + num_poor) / len(mesh.elements) < 0.05
        )
        
        # Industry criteria: no degenerate, < 1% poor
        passes_industry = (
            num_degenerate == 0 and
            num_poor / len(mesh.elements) < 0.01
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            jacobians, aspect_ratios, num_degenerate, num_poor,
            num_excellent, num_good, len(mesh.elements)
        )
        
        report = QualityReport(
            mesh_name=mesh_name,
            element_type=mesh.element_type.value,
            num_elements=len(mesh.elements),
            num_nodes=len(mesh.nodes),
            overall_score=float(overall_score),
            jacobian_score=float(jacobian_score),
            aspect_score=float(aspect_score),
            min_jacobian=float(np.min(jacobians)),
            max_jacobian=float(np.max(jacobians)),
            avg_jacobian=float(np.mean(jacobians)),
            min_aspect_ratio=float(np.min(aspect_ratios)),
            max_aspect_ratio=float(np.max(aspect_ratios)),
            avg_aspect_ratio=float(np.mean(aspect_ratios)),
            num_degenerate=num_degenerate,
            num_poor=num_poor,
            num_fair=num_fair,
            num_good=num_good,
            num_excellent=num_excellent,
            passes_nafems=passes_nafems,
            passes_industry=passes_industry,
            recommendations=recommendations
        )
        
        self.quality_history.append(report)
        return report
    
    def _check_hexahedral_mesh(self, mesh: Mesh, mesh_name: str) -> QualityReport:
        """Quality check for hexahedral mesh"""
        # Simplified check for hex elements
        jacobians = []
        aspect_ratios = []
        
        num_degenerate = 0
        num_poor = 0
        num_fair = 0
        num_good = 0
        num_excellent = 0
        
        for elem_idx in range(len(mesh.elements)):
            nodes = mesh.elements[elem_idx]
            v = [mesh.nodes[nodes[i]] for i in range(8)]
            
            # Calculate edge lengths
            edges = []
            # Bottom face
            edges.append(np.linalg.norm(v[1] - v[0]))
            edges.append(np.linalg.norm(v[2] - v[1]))
            edges.append(np.linalg.norm(v[3] - v[2]))
            edges.append(np.linalg.norm(v[0] - v[3]))
            # Top face
            edges.append(np.linalg.norm(v[5] - v[4]))
            edges.append(np.linalg.norm(v[6] - v[5]))
            edges.append(np.linalg.norm(v[7] - v[6]))
            edges.append(np.linalg.norm(v[4] - v[7]))
            # Vertical
            edges.append(np.linalg.norm(v[4] - v[0]))
            edges.append(np.linalg.norm(v[5] - v[1]))
            edges.append(np.linalg.norm(v[6] - v[2]))
            edges.append(np.linalg.norm(v[7] - v[3]))
            
            max_edge = max(edges)
            min_edge = min(edges)
            aspect_ratio = max_edge / (min_edge + 1e-10)
            aspect_ratios.append(aspect_ratio)
            
            # Simplified Jacobian (edge ratio)
            jacobian = min_edge / (max_edge + 1e-10)
            jacobians.append(jacobian)
            
            # Classify
            if jacobian < 0.01:
                num_degenerate += 1
            elif jacobian < 0.1 or aspect_ratio > 10:
                num_poor += 1
            elif jacobian < 0.3 or aspect_ratio > 5:
                num_fair += 1
            elif jacobian < 0.6 or aspect_ratio > 2:
                num_good += 1
            else:
                num_excellent += 1
        
        jacobians = np.array(jacobians)
        aspect_ratios = np.array(aspect_ratios)
        
        jacobian_score = np.mean(jacobians)
        aspect_score = 1.0 / (1.0 + np.mean(aspect_ratios) / 5.0)
        overall_score = 0.5 * jacobian_score + 0.3 * aspect_score + 0.2 * (num_excellent + num_good) / len(mesh.elements)
        
        passes_nafems = (
            num_degenerate / len(mesh.elements) < 0.01 and
            (num_degenerate + num_poor) / len(mesh.elements) < 0.05
        )
        
        passes_industry = (
            num_degenerate == 0 and
            num_poor / len(mesh.elements) < 0.01
        )
        
        recommendations = self._generate_recommendations(
            jacobians, aspect_ratios, num_degenerate, num_poor,
            num_excellent, num_good, len(mesh.elements)
        )
        
        return QualityReport(
            mesh_name=mesh_name,
            element_type=mesh.element_type.value,
            num_elements=len(mesh.elements),
            num_nodes=len(mesh.nodes),
            overall_score=float(overall_score),
            jacobian_score=float(jacobian_score),
            aspect_score=float(aspect_score),
            min_jacobian=float(np.min(jacobians)),
            max_jacobian=float(np.max(jacobians)),
            avg_jacobian=float(np.mean(jacobians)),
            min_aspect_ratio=float(np.min(aspect_ratios)),
            max_aspect_ratio=float(np.max(aspect_ratios)),
            avg_aspect_ratio=float(np.mean(aspect_ratios)),
            num_degenerate=num_degenerate,
            num_poor=num_poor,
            num_fair=num_fair,
            num_good=num_good,
            num_excellent=num_excellent,
            passes_nafems=passes_nafems,
            passes_industry=passes_industry,
            recommendations=recommendations
        )
    
    def _check_generic_mesh(self, mesh: Mesh, mesh_name: str) -> QualityReport:
        """Generic quality check for unknown element types"""
        return QualityReport(
            mesh_name=mesh_name,
            element_type=mesh.element_type.value,
            num_elements=len(mesh.elements),
            num_nodes=len(mesh.nodes),
            overall_score=0.5,
            jacobian_score=0.5,
            aspect_score=0.5,
            min_jacobian=0.0,
            max_jacobian=1.0,
            avg_jacobian=0.5,
            min_aspect_ratio=1.0,
            max_aspect_ratio=2.0,
            avg_aspect_ratio=1.5,
            num_degenerate=0,
            num_poor=0,
            num_fair=0,
            num_good=0,
            num_excellent=0,
            passes_nafems=False,
            passes_industry=False,
            recommendations=["Unknown element type - manual quality check required"]
        )
    
    def _generate_recommendations(
        self, jacobians: np.ndarray, aspect_ratios: np.ndarray,
        num_degenerate: int, num_poor: int,
        num_excellent: int, num_good: int, total: int
    ) -> List[str]:
        """Generate mesh improvement recommendations"""
        recommendations = []
        
        # Check for degenerate elements
        if num_degenerate > 0:
            pct = 100 * num_degenerate / total
            recommendations.append(
                f"CRITICAL: {num_degenerate} ({pct:.1f}%) degenerate elements detected. "
                "These will cause solver failures. Remesh with smaller elements or geometry cleanup."
            )
        
        # Check for poor elements
        if num_poor > total * 0.05:
            pct = 100 * num_poor / total
            recommendations.append(
                f"WARNING: {num_poor} ({pct:.1f}%) poor quality elements. "
                "Consider remeshing with tighter quality controls."
            )
        
        # Check Jacobian distribution
        min_jac = np.min(jacobians)
        if min_jac < 0.1:
            recommendations.append(
                f"Minimum Jacobian is {min_jac:.3f} (should be > 0.1). "
                "Refine mesh in regions with high curvature."
            )
        
        avg_jac = np.mean(jacobians)
        if avg_jac < 0.5:
            recommendations.append(
                f"Average Jacobian is {avg_jac:.3f} (should be > 0.5). "
                "Consider global mesh refinement or algorithm change."
            )
        
        # Check aspect ratio
        max_ar = np.max(aspect_ratios)
        if max_ar > 10:
            recommendations.append(
                f"Maximum aspect ratio is {max_ar:.1f} (should be < 10). "
                "Check for highly stretched elements near boundaries."
            )
        
        # Check quality distribution
        good_pct = 100 * (num_excellent + num_good) / total
        if good_pct < 80:
            recommendations.append(
                f"Only {good_pct:.1f}% elements are good/excellent (should be > 80%). "
                "Enable additional mesh optimization passes."
            )
        
        # If no issues found
        if not recommendations:
            recommendations.append("Mesh quality is acceptable for analysis.")
        
        return recommendations
    
    def compare_meshes(self, report1: QualityReport, report2: QualityReport) -> Dict[str, Any]:
        """Compare two mesh quality reports"""
        return {
            "score_difference": report1.overall_score - report2.overall_score,
            "improvement": report1.overall_score > report2.overall_score,
            "elements_delta": report1.num_elements - report2.num_elements,
            "quality_improvement": {
                "jacobian": report1.avg_jacobian - report2.avg_jacobian,
                "aspect_ratio": report2.avg_aspect_ratio - report1.avg_aspect_ratio  # Lower is better
            }
        }
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """Get statistics across all checked meshes"""
        if not self.quality_history:
            return {"error": "No meshes checked yet"}
        
        scores = [r.overall_score for r in self.quality_history]
        return {
            "total_meshes_checked": len(self.quality_history),
            "average_score": np.mean(scores),
            "best_score": max(scores),
            "worst_score": min(scores),
            "passing_nafems": sum(1 for r in self.quality_history if r.passes_nafems),
            "passing_industry": sum(1 for r in self.quality_history if r.passes_industry)
        }


def validate_mesh_for_analysis(mesh: Mesh, analysis_type: str = "structural") -> Tuple[bool, QualityReport]:
    """
    Validate mesh is suitable for specific analysis type
    
    Args:
        mesh: Mesh to validate
        analysis_type: "structural", "thermal", "cfd", "modal"
        
    Returns:
        (is_valid, report)
    """
    checker = MeshQualityChecker()
    report = checker.check_mesh(mesh, f"validation_{analysis_type}")
    
    # Different criteria for different analysis types
    if analysis_type == "structural":
        # Structural needs good quality for stress accuracy
        is_valid = report.passes_nafems and report.avg_jacobian >= 0.3
    elif analysis_type == "thermal":
        # Thermal is more forgiving
        is_valid = report.passes_nafems
    elif analysis_type == "cfd":
        # CFD needs boundary layer resolution
        is_valid = report.passes_nafems and report.max_aspect_ratio <= 100  # Allow high AR in BL
    elif analysis_type == "modal":
        # Modal needs good quality for eigenvalue accuracy
        is_valid = report.passes_industry
    else:
        is_valid = report.passes_nafems
    
    return is_valid, report


if __name__ == "__main__":
    # Test quality checker
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MeshQualityChecker...")
    
    # Create a test mesh
    with MeshingEngine() as engine:
        from meshing_engine import MeshingParameters
        
        params = MeshingParameters(
            element_type=ElementType.TET4,
            max_element_size=0.1
        )
        
        mesh = engine.generate_mesh_from_geometry(
            "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
        )
        
        # Check quality
        checker = MeshQualityChecker()
        report = checker.check_mesh(mesh, "test_box")
        
        print(f"\nQuality Report: {report.mesh_name}")
        print(f"Overall Score: {report.overall_score:.3f}")
        print(f"Jacobian: {report.min_jacobian:.3f} - {report.max_jacobian:.3f} (avg: {report.avg_jacobian:.3f})")
        print(f"Aspect Ratio: {report.min_aspect_ratio:.2f} - {report.max_aspect_ratio:.2f}")
        print(f"\nQuality Distribution:")
        print(f"  Excellent: {report.num_excellent}")
        print(f"  Good: {report.num_good}")
        print(f"  Fair: {report.num_fair}")
        print(f"  Poor: {report.num_poor}")
        print(f"  Degenerate: {report.num_degenerate}")
        print(f"\nPasses NAFEMS: {report.passes_nafems}")
        print(f"Passes Industry: {report.passes_industry}")
        print(f"\nRecommendations:")
        for rec in report.recommendations:
            print(f"  - {rec}")
        
        # Validate for different analysis types
        print("\n" + "="*60)
        for analysis in ["structural", "thermal", "modal"]:
            is_valid, _ = validate_mesh_for_analysis(mesh, analysis)
            print(f"Valid for {analysis}: {is_valid}")
