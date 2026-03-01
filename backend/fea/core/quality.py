"""
FIX-203: Mesh Quality Metrics

Comprehensive mesh quality assessment including:
- Element quality metrics (aspect ratio, skewness, jacobian)
- Mesh statistics and statistics
- Quality histograms
- Pass/fail criteria
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try to import mesh processing libraries
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False

try:
    import meshio
    MESHIO_AVAILABLE = True
except ImportError:
    MESHIO_AVAILABLE = False


class QualityMetric(Enum):
    """Types of mesh quality metrics"""
    ASPECT_RATIO = "aspect_ratio"
    SKEWNESS = "skewness"
    JACOBIAN = "jacobian"
    MIN_ANGLE = "min_angle"
    MAX_ANGLE = "max_angle"
    VOLUME = "volume"
    EDGE_RATIO = "edge_ratio"


@dataclass
class ElementQuality:
    """Quality metrics for a single element"""
    element_id: int
    element_type: str
    aspect_ratio: float
    skewness: float
    min_angle: float
    max_angle: float
    jacobian: float
    volume: float
    passed: bool


@dataclass
class MeshQualityReport:
    """Complete quality report for a mesh"""
    mesh_file: Path
    num_elements: int
    num_nodes: int
    element_type: str
    
    # Overall quality metrics
    min_quality: float
    max_quality: float
    mean_quality: float
    std_quality: float
    
    # Pass/fail statistics
    elements_passed: int
    elements_failed: int
    pass_rate: float
    
    # Quality by metric
    aspect_ratio_stats: Dict[str, float]
    skewness_stats: Dict[str, float]
    jacobian_stats: Dict[str, float]
    angle_stats: Dict[str, float]
    
    # Failed elements
    failed_elements: List[int] = field(default_factory=list)
    
    # Recommendations
    recommendations: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert report to dictionary"""
        return {
            "mesh_file": str(self.mesh_file),
            "num_elements": self.num_elements,
            "num_nodes": self.num_nodes,
            "element_type": self.element_type,
            "quality_summary": {
                "min": self.min_quality,
                "max": self.max_quality,
                "mean": self.mean_quality,
                "std": self.std_quality
            },
            "pass_fail": {
                "passed": self.elements_passed,
                "failed": self.elements_failed,
                "pass_rate": self.pass_rate
            },
            "statistics": {
                "aspect_ratio": self.aspect_ratio_stats,
                "skewness": self.skewness_stats,
                "jacobian": self.jacobian_stats,
                "angles": self.angle_stats
            },
            "recommendations": self.recommendations
        }


class MeshQuality:
    """
    Mesh quality assessment for FEA meshes.
    
    FIX-203: Implements mesh quality metrics.
    
    Quality thresholds (industry standards):
    - Aspect ratio: < 10 (ideally < 3)
    - Skewness: < 0.5 (ideally < 0.25)
    - Jacobian: > 0.1 (ideally > 0.6)
    - Min angle: > 15° (ideally > 30°)
    - Max angle: < 165° (ideally < 120°)
    """
    
    # Quality thresholds
    THRESHOLDS = {
        QualityMetric.ASPECT_RATIO: {"max": 10.0, "ideal": 3.0},
        QualityMetric.SKEWNESS: {"max": 0.5, "ideal": 0.25},
        QualityMetric.JACOBIAN: {"min": 0.1, "ideal": 0.6},
        QualityMetric.MIN_ANGLE: {"min": 15.0, "ideal": 30.0},  # degrees
        QualityMetric.MAX_ANGLE: {"max": 165.0, "ideal": 120.0},  # degrees
    }
    
    def __init__(self):
        self._element_qualities: List[ElementQuality] = []
    
    def analyze_mesh(self, mesh_file: Path) -> MeshQualityReport:
        """
        Analyze mesh quality from file.
        
        Args:
            mesh_file: Path to mesh file (.msh, .inp, .vtk, etc.)
            
        Returns:
            MeshQualityReport with detailed quality metrics
        """
        mesh_file = Path(mesh_file)
        if not mesh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {mesh_file}")
        
        # Try different methods based on available libraries
        if GMSH_AVAILABLE and mesh_file.suffix in ['.msh', '.msh2', '.msh4']:
            return self._analyze_with_gmsh(mesh_file)
        elif MESHIO_AVAILABLE:
            return self._analyze_with_meshio(mesh_file)
        else:
            raise RuntimeError(
                "No mesh processing library available. "
                "Install gmsh or meshio."
            )
    
    def _analyze_with_gmsh(self, mesh_file: Path) -> MeshQualityReport:
        """Analyze mesh using Gmsh API"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)
        
        try:
            gmsh.merge(str(mesh_file))
            
            # Get mesh info
            nodes = gmsh.model.mesh.getNodes()
            elements = gmsh.model.mesh.getElements()
            
            num_nodes = len(nodes[0])
            
            # Get all volume elements
            element_tags = []
            element_types = []
            
            for dim, tags, type_nodes in zip(elements[0], elements[1], elements[2]):
                if dim == 3:  # Volume elements
                    element_tags.extend(tags)
                    element_types.extend([gmsh.model.mesh.getElementProperties(
                        gmsh.model.mesh.getType(dim, len(type_nodes) // len(tags))
                    )[0]] * len(tags))
            
            num_elements = len(element_tags)
            
            if num_elements == 0:
                logger.warning("No volume elements found in mesh")
                return self._create_empty_report(mesh_file)
            
            # Calculate quality metrics for each element
            qualities = []
            
            for tag in element_tags:
                quality = self._calculate_element_quality_gmsh(tag)
                if quality:
                    qualities.append(quality)
            
            self._element_qualities = qualities
            
            return self._create_report(mesh_file, qualities, num_nodes, 
                                       "tetra10" if qualities else "unknown")
            
        finally:
            gmsh.finalize()
    
    def _analyze_with_meshio(self, mesh_file: Path) -> MeshQualityReport:
        """Analyze mesh using meshio"""
        mesh = meshio.read(str(mesh_file))
        
        # Get cells (elements)
        if not mesh.cells:
            logger.warning("No cells found in mesh")
            return self._create_empty_report(mesh_file)
        
        # Use first cell block
        cells = mesh.cells[0]
        cell_type = cells.type
        connectivity = cells.data
        
        num_elements = len(connectivity)
        num_nodes = len(mesh.points)
        
        # Calculate quality for each element
        qualities = []
        
        for i, element_nodes in enumerate(connectivity):
            coords = mesh.points[element_nodes]
            quality = self._calculate_element_quality_coords(coords, cell_type, i)
            if quality:
                qualities.append(quality)
        
        self._element_qualities = qualities
        
        return self._create_report(mesh_file, qualities, num_nodes, cell_type)
    
    def _calculate_element_quality_gmsh(self, element_tag: int) -> Optional[ElementQuality]:
        """Calculate quality metrics for a single element using Gmsh"""
        try:
            # Get element nodes
            element_type, node_tags, _ = gmsh.model.mesh.getElementsByTag([element_tag])
            
            if not element_type:
                return None
            
            # Get node coordinates
            coords = []
            for node_tag in node_tags[0]:
                _, coord, _ = gmsh.model.mesh.getNode(node_tag)
                coords.append(coord)
            
            coords = np.array(coords)
            
            # Calculate metrics based on element type
            elem_type_name = gmsh.model.mesh.getElementProperties(element_type[0])[0]
            
            return self._calculate_element_quality_coords(coords, elem_type_name, element_tag)
            
        except Exception as e:
            logger.debug(f"Error calculating quality for element {element_tag}: {e}")
            return None
    
    def _calculate_element_quality_coords(
        self, 
        coords: np.ndarray, 
        element_type: str,
        element_id: int
    ) -> Optional[ElementQuality]:
        """Calculate quality from node coordinates"""
        
        # Calculate edge lengths
        edges = self._get_element_edges(coords)
        edge_lengths = [np.linalg.norm(e[1] - e[0]) for e in edges]
        
        if not edge_lengths or min(edge_lengths) == 0:
            return None
        
        # Aspect ratio
        aspect_ratio = max(edge_lengths) / min(edge_lengths)
        
        # Volume (for 3D elements)
        volume = self._calculate_volume(coords)
        
        # Calculate angles (for surface elements)
        angles = self._calculate_angles(coords)
        min_angle = min(angles) if angles else 0
        max_angle = max(angles) if angles else 180
        
        # Skewness (0 = perfect, 1 = degenerate)
        # Simplified calculation
        skewness = self._calculate_skewness(coords)
        
        # Jacobian (simplified for tetrahedra)
        jacobian = self._calculate_jacobian(coords)
        
        # Check if element passes quality criteria
        passed = self._check_element_quality(
            aspect_ratio, skewness, jacobian, min_angle, max_angle
        )
        
        return ElementQuality(
            element_id=element_id,
            element_type=element_type,
            aspect_ratio=aspect_ratio,
            skewness=skewness,
            min_angle=min_angle,
            max_angle=max_angle,
            jacobian=jacobian,
            volume=volume,
            passed=passed
        )
    
    def _get_element_edges(self, coords: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get edges of element from coordinates"""
        n = len(coords)
        edges = []
        
        # Simple edge detection for common elements
        if n == 4:  # Tetrahedron
            edge_indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        elif n == 8:  # Hexahedron
            edge_indices = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), 
                           (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
        elif n == 10:  # Quadratic tetrahedron
            # Use corner nodes only
            edge_indices = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
        else:
            # Generic: connect all to all
            edge_indices = [(i, j) for i in range(n) for j in range(i+1, n)]
        
        for i, j in edge_indices:
            edges.append((coords[i], coords[j]))
        
        return edges
    
    def _calculate_volume(self, coords: np.ndarray) -> float:
        """Calculate element volume"""
        if len(coords) == 4:  # Tetrahedron
            # V = |det(v1, v2, v3)| / 6
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]
            return abs(np.dot(v1, np.cross(v2, v3))) / 6.0
        elif len(coords) == 8:  # Hexahedron
            # Approximate as sum of tetrahedra
            return 0.0  # Simplified
        else:
            return 0.0
    
    def _calculate_angles(self, coords: np.ndarray) -> List[float]:
        """Calculate angles between edges in degrees"""
        angles = []
        
        if len(coords) >= 3:
            # Calculate angles at each node
            for i in range(len(coords)):
                neighbors = [(i-1) % len(coords), (i+1) % len(coords)]
                v1 = coords[neighbors[0]] - coords[i]
                v2 = coords[neighbors[1]] - coords[i]
                
                norm1 = np.linalg.norm(v1)
                norm2 = np.linalg.norm(v2)
                
                if norm1 > 0 and norm2 > 0:
                    cos_angle = np.dot(v1, v2) / (norm1 * norm2)
                    cos_angle = np.clip(cos_angle, -1, 1)
                    angle = np.degrees(np.arccos(cos_angle))
                    angles.append(angle)
        
        return angles if angles else [60.0, 60.0, 60.0]
    
    def _calculate_skewness(self, coords: np.ndarray) -> float:
        """Calculate element skewness (simplified)"""
        # Skewness = (θ_max - θ_e) / (180 - θ_e) for 2D
        # where θ_e is equiangular angle
        
        angles = self._calculate_angles(coords)
        if not angles:
            return 0.5
        
        # For tetrahedron, ideal angle is 60°
        theta_e = 60.0
        theta_max = max(abs(a - theta_e) for a in angles)
        
        skewness = theta_max / (180.0 - theta_e)
        return min(skewness, 1.0)
    
    def _calculate_jacobian(self, coords: np.ndarray) -> float:
        """Calculate minimum Jacobian (simplified)"""
        if len(coords) == 4:  # Tetrahedron
            v1 = coords[1] - coords[0]
            v2 = coords[2] - coords[0]
            v3 = coords[3] - coords[0]
            jac = abs(np.dot(v1, np.cross(v2, v3))) / 6.0
            # Normalize by edge length
            avg_edge = np.mean([np.linalg.norm(v) for v in [v1, v2, v3]])
            if avg_edge > 0:
                jac /= avg_edge ** 3
            return min(jac, 1.0)
        
        return 0.5  # Default
    
    def _check_element_quality(
        self,
        aspect_ratio: float,
        skewness: float,
        jacobian: float,
        min_angle: float,
        max_angle: float
    ) -> bool:
        """Check if element passes quality criteria"""
        thresholds = self.THRESHOLDS
        
        checks = [
            aspect_ratio <= thresholds[QualityMetric.ASPECT_RATIO]["max"],
            skewness <= thresholds[QualityMetric.SKEWNESS]["max"],
            jacobian >= thresholds[QualityMetric.JACOBIAN]["min"],
            min_angle >= thresholds[QualityMetric.MIN_ANGLE]["min"],
            max_angle <= thresholds[QualityMetric.MAX_ANGLE]["max"]
        ]
        
        return all(checks)
    
    def _create_report(
        self,
        mesh_file: Path,
        qualities: List[ElementQuality],
        num_nodes: int,
        element_type: str
    ) -> MeshQualityReport:
        """Create quality report from element qualities"""
        
        if not qualities:
            return self._create_empty_report(mesh_file)
        
        num_elements = len(qualities)
        passed = sum(1 for q in qualities if q.passed)
        failed = num_elements - passed
        pass_rate = passed / num_elements if num_elements > 0 else 0
        
        # Calculate statistics
        aspect_ratios = [q.aspect_ratio for q in qualities]
        skewnesses = [q.skewness for q in qualities]
        jacobians = [q.jacobian for q in qualities]
        min_angles = [q.min_angle for q in qualities]
        max_angles = [q.max_angle for q in qualities]
        
        # Overall quality score (simplified)
        quality_scores = [
            1.0 / q.aspect_ratio * (1 - q.skewness) * q.jacobian
            for q in qualities
        ]
        
        # Generate recommendations
        recommendations = []
        
        if pass_rate < 0.95:
            recommendations.append(
                f"Pass rate ({pass_rate:.1%}) below 95%. "
                "Consider remeshing with smaller element size."
            )
        
        if np.mean(aspect_ratios) > 5:
            recommendations.append(
                "High average aspect ratio detected. "
                "Check for distorted elements near boundaries."
            )
        
        if np.mean(skewnesses) > 0.3:
            recommendations.append(
                "High skewness detected. "
                "Consider using different meshing algorithm."
            )
        
        if failed > 0:
            failed_ids = [q.element_id for q in qualities if not q.passed][:10]
            recommendations.append(
                f"{failed} elements failed quality check. "
                f"Worst elements: {failed_ids}"
            )
        
        return MeshQualityReport(
            mesh_file=mesh_file,
            num_elements=num_elements,
            num_nodes=num_nodes,
            element_type=element_type,
            min_quality=min(quality_scores),
            max_quality=max(quality_scores),
            mean_quality=np.mean(quality_scores),
            std_quality=np.std(quality_scores),
            elements_passed=passed,
            elements_failed=failed,
            pass_rate=pass_rate,
            aspect_ratio_stats=self._calc_stats(aspect_ratios),
            skewness_stats=self._calc_stats(skewnesses),
            jacobian_stats=self._calc_stats(jacobians),
            angle_stats={
                "min": min(min_angles),
                "max": max(max_angles),
                "mean_min": np.mean(min_angles),
                "mean_max": np.mean(max_angles)
            },
            failed_elements=[q.element_id for q in qualities if not q.passed],
            recommendations=recommendations
        )
    
    def _calc_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistics for a list of values"""
        if not values:
            return {"min": 0, "max": 0, "mean": 0, "std": 0}
        
        return {
            "min": min(values),
            "max": max(values),
            "mean": np.mean(values),
            "std": np.std(values)
        }
    
    def _create_empty_report(self, mesh_file: Path) -> MeshQualityReport:
        """Create empty report for failed analysis"""
        return MeshQualityReport(
            mesh_file=mesh_file,
            num_elements=0,
            num_nodes=0,
            element_type="unknown",
            min_quality=0,
            max_quality=0,
            mean_quality=0,
            std_quality=0,
            elements_passed=0,
            elements_failed=0,
            pass_rate=0,
            aspect_ratio_stats={},
            skewness_stats={},
            jacobian_stats={},
            angle_stats={},
            recommendations=["Could not analyze mesh"]
        )
    
    def check_quality_pass(self, mesh_file: Path, min_pass_rate: float = 0.95) -> bool:
        """
        Quick check if mesh passes quality criteria.
        
        Args:
            mesh_file: Path to mesh file
            min_pass_rate: Minimum acceptable pass rate (default 95%)
            
        Returns:
            True if mesh passes quality check
        """
        report = self.analyze_mesh(mesh_file)
        return report.pass_rate >= min_pass_rate


# Convenience function
def check_mesh_quality(mesh_file: Path) -> Dict[str, Any]:
    """
    Quick mesh quality check.
    
    Args:
        mesh_file: Path to mesh file
        
    Returns:
        Dictionary with quality summary
    """
    analyzer = MeshQuality()
    report = analyzer.analyze_mesh(mesh_file)
    return report.to_dict()
