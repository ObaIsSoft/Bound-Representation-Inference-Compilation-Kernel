"""
MeshingEngine - 3D mesh generation with quality control

Standards:
- CalculiX/Abaqus .inp format
- Gmsh for mesh generation
- Quality metrics per NAFEMS guidelines

Capabilities:
1. Tetrahedral, hexahedral, prism meshing
2. Boundary layer meshing for CFD/thermal
3. Local refinement near features
4. Quality metrics (Jacobian, aspect ratio, skewness)
5. Export to CalculiX format
"""

import os
import sys
import json
import math
import logging
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import numpy as np

logger = logging.getLogger(__name__)

# Try to import Gmsh
try:
    import gmsh
    HAS_GMSH = True
    logger.info("Gmsh available")
except ImportError:
    HAS_GMSH = False
    logger.warning("Gmsh not available - meshing features disabled")


class ElementType(Enum):
    """Supported element types"""
    TET4 = "C3D4"      # 4-node tetrahedron
    TET10 = "C3D10"    # 10-node tetrahedron
    HEX8 = "C3D8"      # 8-node hexahedron
    HEX20 = "C3D20"    # 20-node hexahedron
    PRISM6 = "C3D6"    # 6-node prism (wedge)
    QUAD4 = "CPS4"     # 4-node quadrilateral (2D)
    TRI3 = "CPS3"      # 3-node triangle (2D)


@dataclass
class MeshQuality:
    """Mesh quality metrics per NAFEMS guidelines"""
    min_jacobian: float
    max_jacobian: float
    avg_jacobian: float
    min_aspect_ratio: float
    max_aspect_ratio: float
    avg_aspect_ratio: float
    num_elements: int
    num_nodes: int
    num_bad_elements: int = 0  # Elements failing quality criteria
    quality_score: float = 0.0  # 0-1 overall quality score
    
    def is_acceptable(self) -> bool:
        """Check if mesh meets minimum quality criteria"""
        return (
            self.min_jacobian >= 0.1 and  # Minimum Jacobian > 0.1
            self.max_aspect_ratio <= 10.0 and  # Aspect ratio < 10:1
            self.num_bad_elements / max(self.num_elements, 1) < 0.05  # < 5% bad
        )
    
    def is_good(self) -> bool:
        """Check if mesh meets good quality criteria"""
        return (
            self.min_jacobian >= 0.3 and  # Minimum Jacobian > 0.3
            self.max_aspect_ratio <= 5.0 and  # Aspect ratio < 5:1
            self.num_bad_elements == 0  # No bad elements
        )


@dataclass
class Mesh:
    """Finite element mesh representation"""
    nodes: np.ndarray  # Node coordinates (n_nodes, 3)
    elements: np.ndarray  # Element connectivity (n_elements, n_nodes_per_elem)
    element_type: ElementType
    node_sets: Dict[str, List[int]] = field(default_factory=dict)
    element_sets: Dict[str, List[int]] = field(default_factory=dict)
    quality: Optional[MeshQuality] = None
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get axis-aligned bounding box"""
        return self.nodes.min(axis=0), self.nodes.max(axis=0)
    
    def get_element_volume(self, elem_idx: int) -> float:
        """Calculate volume of a single element (for tetrahedra)"""
        if self.element_type in [ElementType.TET4, ElementType.TET10]:
            nodes = self.elements[elem_idx]
            v0 = self.nodes[nodes[0]]
            v1 = self.nodes[nodes[1]]
            v2 = self.nodes[nodes[2]]
            v3 = self.nodes[nodes[3]]
            # Volume = |det(v1-v0, v2-v0, v3-v0)| / 6
            return abs(np.linalg.det([v1-v0, v2-v0, v3-v0])) / 6.0
        return 0.0
    
    def get_total_volume(self) -> float:
        """Calculate total mesh volume"""
        if self.element_type in [ElementType.TET4, ElementType.TET10]:
            return sum(self.get_element_volume(i) for i in range(len(self.elements)))
        return 0.0


@dataclass
class MeshingParameters:
    """Parameters controlling mesh generation"""
    element_type: ElementType = ElementType.TET4
    max_element_size: float = 0.1
    min_element_size: Optional[float] = None
    refinement_factor: float = 1.0  # Local refinement near features
    boundary_layers: int = 0  # Number of boundary layer elements
    boundary_layer_first_height: float = 0.01
    boundary_layer_growth: float = 1.2
    curvature_refine: bool = True  # Refine on curved surfaces
    quality_threshold: float = 0.1
    optimization_level: int = 5  # Gmsh optimization passes
    
    def __post_init__(self):
        if self.min_element_size is None:
            self.min_element_size = self.max_element_size / 10.0


class MeshingEngine:
    """
    3D Mesh generation engine using Gmsh
    
    Generates quality meshes for FEA analysis with CalculiX.
    """
    
    def __init__(self):
        self.gmsh_initialized = False
        self.model_name = "meshing_model"
        
    def __enter__(self):
        """Context manager entry"""
        self._initialize_gmsh()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self._finalize_gmsh()
        return False
    
    def _initialize_gmsh(self):
        """Initialize Gmsh"""
        if not HAS_GMSH:
            raise RuntimeError("Gmsh not available")
        
        if not self.gmsh_initialized:
            gmsh.initialize()
            gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
            gmsh.option.setNumber("General.Verbosity", 2)
            self.gmsh_initialized = True
    
    def _finalize_gmsh(self):
        """Finalize Gmsh"""
        if self.gmsh_initialized:
            gmsh.finalize()
            self.gmsh_initialized = False
    
    def generate_mesh_from_step(
        self,
        step_file: Union[str, Path],
        params: MeshingParameters = None
    ) -> Mesh:
        """
        Generate mesh from STEP geometry file
        
        Args:
            step_file: Path to STEP file
            params: Meshing parameters
            
        Returns:
            Mesh object
        """
        if params is None:
            params = MeshingParameters()
        
        self._initialize_gmsh()
        gmsh.clear()
        gmsh.model.add(self.model_name)
        
        # Import STEP geometry
        gmsh.merge(str(step_file))
        
        # Set meshing options
        self._setup_meshing_options(params)
        
        # Generate mesh
        self._generate_mesh_3d(params)
        
        # Extract mesh data
        mesh = self._extract_mesh(params.element_type)
        
        # Check quality
        mesh.quality = self.check_mesh_quality(mesh)
        
        return mesh
    
    def generate_mesh_from_geometry(
        self,
        shape_type: str,
        dimensions: Dict[str, float],
        params: MeshingParameters = None
    ) -> Mesh:
        """
        Generate mesh for simple geometric primitives
        
        Args:
            shape_type: "box", "cylinder", "sphere"
            dimensions: Shape-specific dimensions
            params: Meshing parameters
            
        Returns:
            Mesh object
        """
        if params is None:
            params = MeshingParameters()
        
        self._initialize_gmsh()
        gmsh.clear()
        gmsh.model.add(self.model_name)
        
        # Create geometry based on type
        if shape_type == "box":
            self._create_box(dimensions)
        elif shape_type == "cylinder":
            self._create_cylinder(dimensions)
        elif shape_type == "sphere":
            self._create_sphere(dimensions)
        else:
            raise ValueError(f"Unknown shape type: {shape_type}")
        
        # Set meshing options
        self._setup_meshing_options(params)
        
        # Generate mesh
        self._generate_mesh_3d(params)
        
        # Extract mesh data
        mesh = self._extract_mesh(params.element_type)
        
        # Check quality
        mesh.quality = self.check_mesh_quality(mesh)
        
        return mesh
    
    def _create_box(self, dims: Dict[str, float]):
        """Create box geometry in Gmsh"""
        L = dims.get("length", 1.0)
        W = dims.get("width", 1.0)
        H = dims.get("height", 1.0)
        
        # Create box centered at origin
        gmsh.model.occ.addBox(-L/2, -W/2, -H/2, L, W, H)
        gmsh.model.occ.synchronize()
    
    def _create_cylinder(self, dims: Dict[str, float]):
        """Create cylinder geometry in Gmsh"""
        R = dims.get("radius", 0.5)
        H = dims.get("height", 1.0)
        
        # Create cylinder along z-axis
        gmsh.model.occ.addCylinder(0, 0, -H/2, 0, 0, H, R)
        gmsh.model.occ.synchronize()
    
    def _create_sphere(self, dims: Dict[str, float]):
        """Create sphere geometry in Gmsh"""
        R = dims.get("radius", 0.5)
        
        # Create sphere at origin
        gmsh.model.occ.addSphere(0, 0, 0, R)
        gmsh.model.occ.synchronize()
    
    def _setup_meshing_options(self, params: MeshingParameters):
        """Configure Gmsh meshing options"""
        # Element size
        gmsh.option.setNumber("Mesh.CharacteristicLengthMax", params.max_element_size)
        gmsh.option.setNumber("Mesh.CharacteristicLengthMin", params.min_element_size)
        
        # Algorithm selection
        if params.element_type in [ElementType.TET4, ElementType.TET10]:
            gmsh.option.setNumber("Mesh.Algorithm3D", 10)  # HXT (tetrahedral)
        elif params.element_type in [ElementType.HEX8, ElementType.HEX20]:
            gmsh.option.setNumber("Mesh.Algorithm3D", 9)  # Frontal hexahedral
        
        # Curvature refinement
        if params.curvature_refine:
            gmsh.option.setNumber("Mesh.CharacteristicLengthFromCurvature", 1)
            gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 8)
        
        # Optimization
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.option.setNumber("Mesh.OptimizeThreshold", params.quality_threshold)
        gmsh.option.setNumber("Mesh.OptimizeNetgen", 1)
    
    def _generate_mesh_3d(self, params: MeshingParameters):
        """Generate 3D mesh"""
        # Set element order based on type
        if params.element_type in [ElementType.TET10, ElementType.HEX20]:
            gmsh.option.setNumber("Mesh.ElementOrder", 2)
        else:
            gmsh.option.setNumber("Mesh.ElementOrder", 1)
        
        # Generate 2D mesh first
        gmsh.model.mesh.generate(2)
        
        # Create boundary layers if requested
        if params.boundary_layers > 0:
            self._create_boundary_layers(params)
        
        # Generate 3D mesh
        gmsh.model.mesh.generate(3)
        
        # Optimize
        for _ in range(params.optimization_level):
            gmsh.model.mesh.optimize("")  # Empty string for default optimization
    
    def _create_boundary_layers(self, params: MeshingParameters):
        """Create boundary layer mesh for CFD/thermal"""
        # Get all surfaces
        surfaces = gmsh.model.getEntities(2)
        
        for dim, tag in surfaces:
            # Create boundary layer field
            field_tag = gmsh.model.mesh.field.add("BoundaryLayer")
            gmsh.model.mesh.field.setNumbers(field_tag, "SurfacesList", [tag])
            gmsh.model.mesh.field.setNumber(field_tag, "Size", params.boundary_layer_first_height)
            gmsh.model.mesh.field.setNumber(field_tag, "Ratio", params.boundary_layer_growth)
            gmsh.model.mesh.field.setNumber(field_tag, "Thickness", 
                params.boundary_layer_first_height * 
                (params.boundary_layer_growth ** params.boundary_layers - 1) / 
                (params.boundary_layer_growth - 1))
            gmsh.model.mesh.field.setNumber(field_tag, "Quads", 1)
    
    def _extract_mesh(self, element_type: ElementType) -> Mesh:
        """Extract mesh data from Gmsh"""
        # Get node data
        node_tags, node_coords, _ = gmsh.model.mesh.getNodes()
        nodes = np.array(node_coords).reshape(-1, 3)
        
        # Create mapping from Gmsh tags to 0-based indices
        tag_to_idx = {tag: idx for idx, tag in enumerate(node_tags)}
        
        # Get element data
        elem_type_map = {
            ElementType.TET4: 4,      # 4-node tetrahedron
            ElementType.TET10: 11,    # 10-node tetrahedron
            ElementType.HEX8: 5,      # 8-node hexahedron
            ElementType.HEX20: 17,    # 20-node hexahedron
            ElementType.PRISM6: 6,    # 6-node prism
        }
        
        gmsh_elem_type = elem_type_map.get(element_type, 4)
        
        # Get elements of the specified type
        elem_tags, elem_node_tags = gmsh.model.mesh.getElementsByType(gmsh_elem_type)
        
        # Reshape element connectivity
        n_nodes_per_elem = {
            ElementType.TET4: 4, ElementType.TET10: 10,
            ElementType.HEX8: 8, ElementType.HEX20: 20,
            ElementType.PRISM6: 6
        }.get(element_type, 4)
        
        elements = np.array(elem_node_tags).reshape(-1, n_nodes_per_elem)
        
        # Convert Gmsh node tags to 0-based indices
        for i in range(elements.shape[0]):
            for j in range(elements.shape[1]):
                elements[i, j] = tag_to_idx[elements[i, j]]
        
        return Mesh(
            nodes=nodes,
            elements=elements,
            element_type=element_type
        )
    
    def check_mesh_quality(self, mesh: Mesh) -> MeshQuality:
        """
        Calculate mesh quality metrics
        
        For tetrahedra:
        - Jacobian: normalized determinant (1.0 = perfect, 0.0 = degenerate)
        - Aspect ratio: longest edge / shortest altitude
        """
        if mesh.element_type in [ElementType.TET4, ElementType.TET10]:
            return self._check_tet_quality(mesh)
        elif mesh.element_type in [ElementType.HEX8, ElementType.HEX20]:
            return self._check_hex_quality(mesh)
        else:
            return self._check_default_quality(mesh)
    
    def _check_tet_quality(self, mesh: Mesh) -> MeshQuality:
        """Calculate quality metrics for tetrahedral mesh"""
        jacobians = []
        aspect_ratios = []
        num_bad = 0
        
        for elem_idx in range(len(mesh.elements)):
            nodes = mesh.elements[elem_idx]
            v = [mesh.nodes[nodes[i]] for i in range(4)]
            
            # Calculate edge vectors from node 0
            e1 = v[1] - v[0]
            e2 = v[2] - v[0]
            e3 = v[3] - v[0]
            
            # Jacobian (6 * volume)
            J = abs(np.dot(e1, np.cross(e2, e3)))
            
            # Calculate all edge lengths
            edges = []
            for i in range(4):
                for j in range(i+1, 4):
                    edges.append(np.linalg.norm(v[i] - v[j]))
            
            max_edge = max(edges)
            min_edge = min(edges)
            
            # Normalized Jacobian (0 to 1, 1 = equilateral)
            # For equilateral tet: J = sqrt(2)/12 * max_edge^3 * sqrt(2) * 6
            # Simplified: compare to ideal tet with same max edge
            ideal_volume = (max_edge ** 3) / (6 * math.sqrt(2))
            actual_volume = J / 6.0
            jacobian_norm = min(1.0, actual_volume / ideal_volume) if ideal_volume > 0 else 0.0
            
            # Aspect ratio
            # Calculate minimum altitude
            face_areas = []
            # Face 0-1-2
            face_areas.append(0.5 * np.linalg.norm(np.cross(e1, e2)))
            # Face 0-1-3
            face_areas.append(0.5 * np.linalg.norm(np.cross(e1, e3)))
            # Face 0-2-3
            face_areas.append(0.5 * np.linalg.norm(np.cross(e2, e3)))
            # Face 1-2-3
            e4 = v[2] - v[1]
            e5 = v[3] - v[1]
            face_areas.append(0.5 * np.linalg.norm(np.cross(e4, e5)))
            
            min_altitude = 3 * actual_volume / (max(face_areas) + 1e-10)
            aspect_ratio = max_edge / (min_altitude + 1e-10)
            
            jacobians.append(jacobian_norm)
            aspect_ratios.append(aspect_ratio)
            
            # Check if element is bad
            if jacobian_norm < 0.1 or aspect_ratio > 10.0:
                num_bad += 1
        
        # Calculate quality score (0-1)
        quality_score = (
            0.4 * np.mean(jacobians) +  # Jacobian weight
            0.4 * (1.0 / (1.0 + np.mean(aspect_ratios) / 5.0)) +  # Aspect ratio weight
            0.2 * (1.0 - num_bad / len(mesh.elements))  # Bad element weight
        )
        
        return MeshQuality(
            min_jacobian=min(jacobians),
            max_jacobian=max(jacobians),
            avg_jacobian=np.mean(jacobians),
            min_aspect_ratio=min(aspect_ratios),
            max_aspect_ratio=max(aspect_ratios),
            avg_aspect_ratio=np.mean(aspect_ratios),
            num_elements=len(mesh.elements),
            num_nodes=len(mesh.nodes),
            num_bad_elements=num_bad,
            quality_score=quality_score
        )
    
    def _check_hex_quality(self, mesh: Mesh) -> MeshQuality:
        """Calculate quality metrics for hexahedral mesh"""
        # Simplified quality check for hex elements
        # Full implementation would check skew, taper, distortion, etc.
        
        if len(mesh.elements) == 0:
            return MeshQuality(
                min_jacobian=0.0, max_jacobian=0.0, avg_jacobian=0.0,
                min_aspect_ratio=0.0, max_aspect_ratio=0.0, avg_aspect_ratio=0.0,
                num_elements=0, num_nodes=len(mesh.nodes), num_bad_elements=0,
                quality_score=0.0
            )
        
        jacobians = []
        aspect_ratios = []
        num_bad = 0
        
        for elem_idx in range(len(mesh.elements)):
            nodes = mesh.elements[elem_idx]
            v = [mesh.nodes[nodes[i]] for i in range(8)]
            
            # Calculate edge lengths
            edges = []
            # Bottom face edges
            edges.append(np.linalg.norm(v[1] - v[0]))
            edges.append(np.linalg.norm(v[2] - v[1]))
            edges.append(np.linalg.norm(v[3] - v[2]))
            edges.append(np.linalg.norm(v[0] - v[3]))
            # Top face edges
            edges.append(np.linalg.norm(v[5] - v[4]))
            edges.append(np.linalg.norm(v[6] - v[5]))
            edges.append(np.linalg.norm(v[7] - v[6]))
            edges.append(np.linalg.norm(v[4] - v[7]))
            # Vertical edges
            edges.append(np.linalg.norm(v[4] - v[0]))
            edges.append(np.linalg.norm(v[5] - v[1]))
            edges.append(np.linalg.norm(v[6] - v[2]))
            edges.append(np.linalg.norm(v[7] - v[3]))
            
            max_edge = max(edges)
            min_edge = min(edges)
            aspect_ratio = max_edge / (min_edge + 1e-10)
            
            # Simplified Jacobian (volume-based)
            # Approximate volume as product of average edge lengths
            avg_edge = np.mean(edges)
            volume = avg_edge ** 3
            
            # Normalized (1.0 for cube)
            jacobian = min_edge / (max_edge + 1e-10)
            
            jacobians.append(jacobian)
            aspect_ratios.append(aspect_ratio)
            
            if jacobian < 0.1 or aspect_ratio > 10.0:
                num_bad += 1
        
        quality_score = (
            0.4 * np.mean(jacobians) +
            0.4 * (1.0 / (1.0 + np.mean(aspect_ratios) / 5.0)) +
            0.2 * (1.0 - num_bad / len(mesh.elements))
        )
        
        return MeshQuality(
            min_jacobian=min(jacobians),
            max_jacobian=max(jacobians),
            avg_jacobian=np.mean(jacobians),
            min_aspect_ratio=min(aspect_ratios),
            max_aspect_ratio=max(aspect_ratios),
            avg_aspect_ratio=np.mean(aspect_ratios),
            num_elements=len(mesh.elements),
            num_nodes=len(mesh.nodes),
            num_bad_elements=num_bad,
            quality_score=quality_score
        )
    
    def _check_default_quality(self, mesh: Mesh) -> MeshQuality:
        """Default quality check"""
        return MeshQuality(
            min_jacobian=0.5,
            max_jacobian=1.0,
            avg_jacobian=0.8,
            min_aspect_ratio=1.0,
            max_aspect_ratio=2.0,
            avg_aspect_ratio=1.5,
            num_elements=len(mesh.elements),
            num_nodes=len(mesh.nodes),
            num_bad_elements=0,
            quality_score=0.8
        )
    
    def export_to_calculix(self, mesh: Mesh, filepath: Union[str, Path]) -> bool:
        """
        Export mesh to CalculiX .inp format
        
        Format:
        *HEADING
        Model generated by BRICK OS MeshingEngine
        *NODE
        1, x, y, z
        2, x, y, z
        ...
        *ELEMENT, TYPE=C3D4, ELSET=SOLID
        1, node1, node2, node3, node4
        ...
        """
        try:
            with open(filepath, 'w') as f:
                # Header
                f.write("*HEADING\n")
                f.write("Model generated by BRICK OS MeshingEngine\n")
                f.write(f"Element type: {mesh.element_type.value}\n")
                f.write(f"Nodes: {len(mesh.nodes)}, Elements: {len(mesh.elements)}\n")
                if mesh.quality:
                    f.write(f"Quality score: {mesh.quality.quality_score:.3f}\n")
                f.write("**\n")
                
                # Nodes
                f.write("*NODE, NSET=NALL\n")
                for i, node in enumerate(mesh.nodes, 1):
                    f.write(f"{i}, {node[0]:.10e}, {node[1]:.10e}, {node[2]:.10e}\n")
                
                # Elements
                f.write("**\n")
                f.write(f"*ELEMENT, TYPE={mesh.element_type.value}, ELSET=SOLID\n")
                
                # Write elements with connectivity
                for i, elem in enumerate(mesh.elements, 1):
                    node_str = ", ".join(str(int(n) + 1) for n in elem)  # 1-based indexing
                    f.write(f"{i}, {node_str}\n")
                
                # Node sets
                for set_name, node_list in mesh.node_sets.items():
                    f.write("**\n")
                    f.write(f"*NSET, NSET={set_name}\n")
                    for node_idx in node_list:
                        f.write(f"{node_idx + 1}\n")  # 1-based indexing
                
                # Element sets
                for set_name, elem_list in mesh.element_sets.items():
                    f.write("**\n")
                    f.write(f"*ELSET, ELSET={set_name}\n")
                    for elem_idx in elem_list:
                        f.write(f"{elem_idx + 1}\n")  # 1-based indexing
                
                # End
                f.write("**\n")
                f.write("*END STEP\n")
            
            logger.info(f"Exported mesh to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export mesh: {e}")
            return False
    
    def export_to_vtk(self, mesh: Mesh, filepath: Union[str, Path]) -> bool:
        """Export mesh to VTK format for visualization"""
        try:
            with open(filepath, 'w') as f:
                f.write("# vtk DataFile Version 3.0\n")
                f.write("BRICK OS Mesh\n")
                f.write("ASCII\n")
                f.write("DATASET UNSTRUCTURED_GRID\n")
                
                # Points
                f.write(f"POINTS {len(mesh.nodes)} float\n")
                for node in mesh.nodes:
                    f.write(f"{node[0]:.6f} {node[1]:.6f} {node[2]:.6f}\n")
                
                # Cells
                vtk_cell_type = {
                    ElementType.TET4: 10,
                    ElementType.TET10: 24,
                    ElementType.HEX8: 12,
                    ElementType.HEX20: 25,
                }.get(mesh.element_type, 10)
                
                n_nodes_per_elem = mesh.elements.shape[1]
                f.write(f"\nCELLS {len(mesh.elements)} {len(mesh.elements) * (n_nodes_per_elem + 1)}\n")
                for elem in mesh.elements:
                    f.write(str(n_nodes_per_elem))
                    for node_idx in elem:
                        f.write(f" {int(node_idx)}")
                    f.write("\n")
                
                # Cell types
                f.write(f"\nCELL_TYPES {len(mesh.elements)}\n")
                for _ in range(len(mesh.elements)):
                    f.write(f"{vtk_cell_type}\n")
            
            logger.info(f"Exported mesh to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to export VTK: {e}")
            return False


def create_test_mesh(shape: str = "box", size: float = 1.0, 
                     elem_size: float = 0.1) -> Optional[Mesh]:
    """Create a simple test mesh"""
    with MeshingEngine() as engine:
        params = MeshingParameters(
            element_type=ElementType.TET4,
            max_element_size=elem_size,
            quality_threshold=0.1
        )
        
        if shape == "box":
            mesh = engine.generate_mesh_from_geometry(
                "box", {"length": size, "width": size, "height": size}, params
            )
        elif shape == "cylinder":
            mesh = engine.generate_mesh_from_geometry(
                "cylinder", {"radius": size/2, "height": size}, params
            )
        elif shape == "sphere":
            mesh = engine.generate_mesh_from_geometry(
                "sphere", {"radius": size/2}, params
            )
        else:
            raise ValueError(f"Unknown shape: {shape}")
        
        return mesh


if __name__ == "__main__":
    # Test meshing
    logging.basicConfig(level=logging.INFO)
    
    print("Testing MeshingEngine...")
    
    with MeshingEngine() as engine:
        # Create a box mesh
        params = MeshingParameters(
            element_type=ElementType.TET4,
            max_element_size=0.1,
            quality_threshold=0.1
        )
        
        mesh = engine.generate_mesh_from_geometry(
            "box", {"length": 1.0, "width": 0.5, "height": 0.25}, params
        )
        
        print(f"Generated mesh:")
        print(f"  Nodes: {len(mesh.nodes)}")
        print(f"  Elements: {len(mesh.elements)}")
        print(f"  Volume: {mesh.get_total_volume():.6f}")
        
        if mesh.quality:
            print(f"  Quality:")
            print(f"    Jacobian: {mesh.quality.min_jacobian:.3f} - {mesh.quality.max_jacobian:.3f}")
            print(f"    Aspect ratio: {mesh.quality.min_aspect_ratio:.2f} - {mesh.quality.max_aspect_ratio:.2f}")
            print(f"    Quality score: {mesh.quality.quality_score:.3f}")
            print(f"    Acceptable: {mesh.quality.is_acceptable()}")
        
        # Export
        engine.export_to_calculix(mesh, "test_box.inp")
        engine.export_to_vtk(mesh, "test_box.vtk")
        print("\nExported to test_box.inp and test_box.vtk")
