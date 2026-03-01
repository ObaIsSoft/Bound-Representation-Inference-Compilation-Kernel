"""
FIX-202: Gmsh Mesh Generation

Python interface to Gmsh for generating 3D finite element meshes.
Supports various element types and mesh sizing controls.
"""

import os
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

# Try to import gmsh
try:
    import gmsh
    GMSH_AVAILABLE = True
except ImportError:
    GMSH_AVAILABLE = False
    logger.warning("Gmsh Python API not available. Mesh generation will be limited.")


class ElementType(Enum):
    """Finite element types"""
    # 2D elements
    TRIANGLE_3 = "triangle3"
    TRIANGLE_6 = "triangle6"
    QUAD_4 = "quad4"
    QUAD_8 = "quad8"
    # 3D elements
    TETRA_4 = "tetra4"
    TETRA_10 = "tetra10"
    HEXA_8 = "hexa8"
    HEXA_20 = "hexa20"
    PRISM_6 = "prism6"
    PYRAMID_5 = "pyramid5"


@dataclass
class MeshConfig:
    """Configuration for mesh generation"""
    element_type: ElementType = ElementType.TETRA_10
    mesh_size: float = 0.1
    mesh_size_min: Optional[float] = None
    mesh_size_max: Optional[float] = None
    curvature_based_refinement: bool = True
    min_elements_per_curve: int = 10
    optimization_level: int = 5
    smoothing_steps: int = 3
    algorithm_2d: str = "delquad"  # delaunay, delquad, frontal
    algorithm_3d: str = "del3d"    # del3d, frontal, mmg3d
    
    def __post_init__(self):
        if self.mesh_size_min is None:
            self.mesh_size_min = self.mesh_size / 10
        if self.mesh_size_max is None:
            self.mesh_size_max = self.mesh_size * 10


@dataclass
class MeshStatistics:
    """Statistics about a generated mesh"""
    num_nodes: int
    num_elements: int
    num_surface_elements: int
    num_volume_elements: int
    element_type: str
    mesh_size: float
    bounding_box: Tuple[Tuple[float, ...], Tuple[float, ...]]
    file_path: Path


class GmshMesher:
    """
    Gmsh mesh generator for FEA.
    
    FIX-202: Implements mesh generation using Gmsh API.
    
    Usage:
        mesher = GmshMesher(config)
        mesh = mesher.generate_from_step(step_file)
        mesher.export_to_inp(mesh_file, calculix_inp)
    """
    
    def __init__(self, config: Optional[MeshConfig] = None):
        self.config = config or MeshConfig()
        
        if not GMSH_AVAILABLE:
            raise RuntimeError(
                "Gmsh Python API not available. "
                "Install with: pip install gmsh"
            )
        
        # Track mesh history
        self._mesh_history: List[MeshStatistics] = []
        self._current_model: Optional[int] = None
    
    def _initialize_gmsh(self) -> None:
        """Initialize Gmsh library"""
        gmsh.initialize()
        gmsh.option.setNumber("General.Terminal", 0)  # Suppress output
        gmsh.model.add("fea_mesh")
    
    def _finalize_gmsh(self) -> None:
        """Finalize Gmsh library"""
        if gmsh.isInitialized():
            gmsh.finalize()
    
    def generate_from_step(
        self,
        step_file: Path,
        output_mesh: Optional[Path] = None,
        physical_groups: Optional[Dict[str, List[int]]] = None
    ) -> MeshStatistics:
        """
        Generate mesh from STEP geometry file.
        
        Args:
            step_file: Path to .step/.stp file
            output_mesh: Output path for mesh file
            physical_groups: Dict of group name -> dimension tags
                e.g., {"fixed_faces": [(2, 1), (2, 2)]}
                
        Returns:
            MeshStatistics about generated mesh
        """
        step_file = Path(step_file)
        if not step_file.exists():
            raise FileNotFoundError(f"STEP file not found: {step_file}")
        
        try:
            self._initialize_gmsh()
            
            # Import STEP geometry
            logger.info(f"Importing STEP: {step_file}")
            gmsh.model.occ.importShapes(str(step_file))
            gmsh.model.occ.synchronize()
            
            # Set mesh size
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.config.mesh_size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.config.mesh_size_max)
            gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 
                                 1 if self.config.curvature_based_refinement else 0)
            gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 
                                 self.config.min_elements_per_curve)
            
            # Set algorithms
            algo_2d_map = {
                "delaunay": 5,
                "delquad": 8,
                "frontal": 6,
                "bamgc": 7,
                "pack": 9
            }
            algo_3d_map = {
                "del3d": 1,
                "frontal": 4,
                "mmg3d": 10,
                "hxt": 10
            }
            
            gmsh.option.setNumber("Mesh.Algorithm", 
                                 algo_2d_map.get(self.config.algorithm_2d, 8))
            gmsh.option.setNumber("Mesh.Algorithm3D", 
                                 algo_3d_map.get(self.config.algorithm_3d, 10))
            
            # Generate mesh
            logger.info("Generating 2D mesh...")
            gmsh.model.mesh.generate(2)
            
            logger.info("Generating 3D mesh...")
            gmsh.model.mesh.generate(3)
            
            # Optimize mesh
            if self.config.optimization_level > 0:
                logger.info(f"Optimizing mesh (level {self.config.optimization_level})...")
                gmsh.model.mesh.optimize("Netgen")
            
            # Set element order
            if self.config.element_type in [ElementType.TETRA_10, ElementType.HEXA_20]:
                logger.info("Setting second-order elements...")
                gmsh.model.mesh.setOrder(2)
            
            # Create physical groups
            if physical_groups:
                for name, entities in physical_groups.items():
                    # Determine dimension from first entity
                    if entities:
                        dim = entities[0][0] if isinstance(entities[0], tuple) else 3
                        tags = [e[1] if isinstance(e, tuple) else e for e in entities]
                        gmsh.model.addPhysicalGroup(dim, tags, name=name)
            
            # Get statistics
            stats = self._get_mesh_statistics()
            
            # Export mesh
            if output_mesh is None:
                output_mesh = step_file.with_suffix('.msh')
            
            output_mesh = Path(output_mesh)
            output_mesh.parent.mkdir(parents=True, exist_ok=True)
            
            # Export to MSH format (version 2.2 for CalculiX compatibility)
            gmsh.write(str(output_mesh))
            
            stats.file_path = output_mesh
            self._mesh_history.append(stats)
            
            logger.info(f"Mesh saved to: {output_mesh}")
            logger.info(f"Nodes: {stats.num_nodes}, Elements: {stats.num_elements}")
            
            return stats
            
        finally:
            self._finalize_gmsh()
    
    def generate_simple_cube(
        self,
        size: float = 1.0,
        center: Tuple[float, float, float] = (0, 0, 0),
        output_mesh: Optional[Path] = None
    ) -> MeshStatistics:
        """
        Generate a simple cube mesh for testing.
        
        Args:
            size: Cube side length
            center: Cube center coordinates
            output_mesh: Output path
            
        Returns:
            MeshStatistics
        """
        try:
            self._initialize_gmsh()
            
            # Create cube
            cx, cy, cz = center
            half = size / 2
            
            box = gmsh.model.occ.addBox(
                cx - half, cy - half, cz - half,
                size, size, size
            )
            gmsh.model.occ.synchronize()
            
            # Set mesh size
            gmsh.option.setNumber("Mesh.MeshSizeMin", self.config.mesh_size_min)
            gmsh.option.setNumber("Mesh.MeshSizeMax", self.config.mesh_size_max)
            
            # Generate mesh
            gmsh.model.mesh.generate(3)
            
            if self.config.element_type in [ElementType.TETRA_10, ElementType.HEXA_20]:
                gmsh.model.mesh.setOrder(2)
            
            # Get statistics
            stats = self._get_mesh_statistics()
            
            # Export
            if output_mesh is None:
                output_mesh = Path(tempfile.mktemp(suffix='.msh'))
            
            gmsh.write(str(output_mesh))
            stats.file_path = output_mesh
            self._mesh_history.append(stats)
            
            return stats
            
        finally:
            self._finalize_gmsh()
    
    def export_to_inp(
        self,
        msh_file: Path,
        inp_file: Path,
        material_name: str = "Material-1",
        youngs_modulus: float = 210e3,  # MPa
        poisson_ratio: float = 0.3,
        density: float = 7.8e-9  # tonne/mm³ for steel
    ) -> Path:
        """
        Convert Gmsh .msh to CalculiX .inp format.
        
        Args:
            msh_file: Input Gmsh mesh file
            inp_file: Output CalculiX input file
            material_name: Name for material definition
            youngs_modulus: Young's modulus in MPa
            poisson_ratio: Poisson's ratio
            density: Density in tonne/mm³
            
        Returns:
            Path to generated .inp file
        """
        msh_file = Path(msh_file)
        inp_file = Path(inp_file)
        
        if not msh_file.exists():
            raise FileNotFoundError(f"Mesh file not found: {msh_file}")
        
        inp_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Use gmsh to convert to INP format
        try:
            self._initialize_gmsh()
            gmsh.merge(str(msh_file))
            
            # Write in ABAQUS format (compatible with CalculiX)
            gmsh.option.setString("Mesh.Format", "inp")
            gmsh.write(str(inp_file))
            
            logger.info(f"Exported to CalculiX format: {inp_file}")
            return inp_file
            
        finally:
            self._finalize_gmsh()
    
    def _get_mesh_statistics(self) -> MeshStatistics:
        """Get statistics about current mesh"""
        nodes = gmsh.model.mesh.getNodes()
        elements = gmsh.model.mesh.getElements()
        
        num_nodes = len(nodes[0])
        
        # Count elements by dimension
        num_surface = 0
        num_volume = 0
        
        for dim, tags in zip(elements[0], elements[1]):
            if dim == 2:
                num_surface += len(tags)
            elif dim == 3:
                num_volume += len(tags)
        
        # Get bounding box
        entities = gmsh.model.getEntities()
        bbox = gmsh.model.getBoundingBox(
            entities[0][0], entities[0][1]
        ) if entities else (0, 0, 0, 1, 1, 1)
        
        min_corner = (bbox[0], bbox[1], bbox[2])
        max_corner = (bbox[3], bbox[4], bbox[5])
        
        return MeshStatistics(
            num_nodes=num_nodes,
            num_elements=num_surface + num_volume,
            num_surface_elements=num_surface,
            num_volume_elements=num_volume,
            element_type=self.config.element_type.value,
            mesh_size=self.config.mesh_size,
            bounding_box=(min_corner, max_corner),
            file_path=Path(".")
        )
    
    def get_mesh_history(self) -> List[MeshStatistics]:
        """Get history of generated meshes"""
        return self._mesh_history.copy()


# Convenience functions
def generate_mesh_from_step(
    step_file: Path,
    mesh_size: float = 0.1,
    output_mesh: Optional[Path] = None
) -> MeshStatistics:
    """
    Quick mesh generation from STEP file.
    
    Args:
        step_file: Path to STEP file
        mesh_size: Target mesh size
        output_mesh: Output path
        
    Returns:
        MeshStatistics
    """
    config = MeshConfig(mesh_size=mesh_size)
    mesher = GmshMesher(config)
    return mesher.generate_from_step(step_file, output_mesh)


def generate_test_cube(
    size: float = 1.0,
    mesh_size: float = 0.1,
    output_mesh: Optional[Path] = None
) -> MeshStatistics:
    """
    Generate a test cube mesh.
    
    Args:
        size: Cube size
        mesh_size: Target mesh size
        output_mesh: Output path
        
    Returns:
        MeshStatistics
    """
    config = MeshConfig(mesh_size=mesh_size)
    mesher = GmshMesher(config)
    return mesher.generate_simple_cube(size, output_mesh=output_mesh)
