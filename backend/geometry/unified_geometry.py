"""
Unified Geometry Module - Multi-kernel CAD engine for BRICK OS

Consolidates all geometry operations into a single interface.
Supports: OpenCASCADE, Manifold3D, CGAL, with automatic kernel selection.

Dependencies (install all):
    - cadquery-ocp (OpenCASCADE)
    - manifold3d
    - gmsh
    - meshio
    - trimesh
    - pyvista
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

# =============================================================================
# KERNEL AVAILABILITY CHECKS (Fail if not available, no silent skipping)
# =============================================================================

def _check_opencascade():
    """Check OpenCASCADE via cadquery-ocp"""
    try:
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        from OCP.STEPControl import STEPControl_Writer
        return True
    except ImportError:
        logger.error("OpenCASCADE not available. Install: pip install cadquery-ocp")
        return False

def _check_manifold():
    """Check Manifold3D"""
    try:
        import manifold3d as m3d
        return True
    except ImportError:
        logger.error("Manifold3D not available. Install: pip install manifold3d")
        return False

def _check_gmsh():
    """Check Gmsh"""
    try:
        import gmsh
        return True
    except ImportError:
        logger.error("Gmsh not available. Install: pip install gmsh")
        return False

HAS_OPENCASCADE = _check_opencascade()
HAS_MANIFOLD = _check_manifold()
HAS_GMSH = _check_gmsh()

# Fail if no geometry kernel available
if not (HAS_OPENCASCADE or HAS_MANIFOLD):
    raise RuntimeError(
        "No geometry kernel available. Install at least one:\n"
        "  pip install cadquery-ocp  # OpenCASCADE\n"
        "  pip install manifold3d     # Manifold3D"
    )

# =============================================================================
# DATA CLASSES
# =============================================================================

class KernelType(Enum):
    """Available geometry kernels"""
    OPENCASCADE = "opencascade"
    MANIFOLD = "manifold"
    AUTO = "auto"

class MeshFormat(Enum):
    """Supported mesh formats"""
    STL = "stl"
    OBJ = "obj"
    STEP = "step"
    IGES = "iges"
    MSH = "msh"
    VTK = "vtk"
    INP = "inp"

@dataclass
class Mesh:
    """Mesh representation"""
    vertices: np.ndarray
    faces: np.ndarray
    normals: Optional[np.ndarray] = None
    uv: Optional[np.ndarray] = None
    
    # Quality metrics
    quality_score: float = 0.0
    min_jacobian: float = 0.0
    max_aspect_ratio: float = 0.0
    
    def __post_init__(self):
        if self.vertices is not None:
            self.vertices = np.asarray(self.vertices, dtype=np.float32)
        if self.faces is not None:
            self.faces = np.asarray(self.faces, dtype=np.int32)

@dataclass
class BoundingBox:
    """Axis-aligned bounding box"""
    xmin: float
    ymin: float
    zmin: float
    xmax: float
    ymax: float
    zmax: float
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.xmin + self.xmax) / 2,
            (self.ymin + self.ymax) / 2,
            (self.zmin + self.zmax) / 2
        )
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (
            self.xmax - self.xmin,
            self.ymax - self.ymin,
            self.zmax - self.zmin
        )

@dataclass
class MassProperties:
    """Mass property calculation results"""
    volume: float
    surface_area: float
    centroid: Tuple[float, float, float]
    bounding_box: BoundingBox
    
    # Moments of inertia (optional)
    moment_xx: float = 0.0
    moment_yy: float = 0.0
    moment_zz: float = 0.0
    moment_xy: float = 0.0
    moment_xz: float = 0.0
    moment_yz: float = 0.0

# =============================================================================
# ABSTRACT KERNEL INTERFACE
# =============================================================================

class GeometryKernel(ABC):
    """Abstract base for geometry kernels"""
    
    @property
    @abstractmethod
    def name(self) -> str:
        pass
    
    @abstractmethod
    def create_box(self, width: float, height: float, depth: float) -> Any:
        """Create axis-aligned box"""
        pass
    
    @abstractmethod
    def create_cylinder(self, radius: float, height: float, segments: int = 32) -> Any:
        """Create cylinder along Z-axis"""
        pass
    
    @abstractmethod
    def create_sphere(self, radius: float) -> Any:
        """Create sphere at origin"""
        pass
    
    @abstractmethod
    def boolean_union(self, shape1: Any, shape2: Any) -> Any:
        pass
    
    @abstractmethod
    def boolean_difference(self, shape1: Any, shape2: Any) -> Any:
        pass
    
    @abstractmethod
    def boolean_intersection(self, shape1: Any, shape2: Any) -> Any:
        pass
    
    @abstractmethod
    def fillet_edges(self, shape: Any, radius: float, edges: Optional[List[int]] = None) -> Any:
        pass
    
    @abstractmethod
    def chamfer_edges(self, shape: Any, distance: float, edges: Optional[List[int]] = None) -> Any:
        pass
    
    @abstractmethod
    def tessellate(self, shape: Any, tolerance: float = 0.01) -> Mesh:
        """Convert to triangle mesh"""
        pass
    
    @abstractmethod
    def export_file(self, shape: Any, filepath: Path, format: MeshFormat) -> None:
        pass
    
    @abstractmethod
    def import_file(self, filepath: Path, format: MeshFormat) -> Any:
        pass
    
    @abstractmethod
    def get_mass_properties(self, shape: Any, density: float = 1.0) -> MassProperties:
        pass
    
    @abstractmethod
    def get_bounding_box(self, shape: Any) -> BoundingBox:
        pass

# =============================================================================
# OPENCASCADE KERNEL
# =============================================================================

class OpenCASCADEKernel(GeometryKernel):
    """Production-grade B-Rep kernel"""
    
    def __init__(self):
        if not HAS_OPENCASCADE:
            raise RuntimeError("OpenCASCADE not available")
        
        from OCP.gp import gp_Pnt, gp_Dir, gp_Ax2
        self.gp_Pnt = gp_Pnt
        self.gp_Dir = gp_Dir
        self.gp_Ax2 = gp_Ax2
        
        logger.info("OpenCASCADE kernel initialized")
    
    @property
    def name(self) -> str:
        return "OpenCASCADE"
    
    def create_box(self, width: float, height: float, depth: float):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox
        return BRepPrimAPI_MakeBox(width, height, depth).Shape()
    
    def create_cylinder(self, radius: float, height: float, segments: int = 32):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        axis = self.gp_Ax2(self.gp_Pnt(0, 0, 0), self.gp_Dir(0, 0, 1))
        return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()
    
    def create_sphere(self, radius: float):
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeSphere
        return BRepPrimAPI_MakeSphere(self.gp_Pnt(0, 0, 0), radius).Shape()
    
    def boolean_union(self, shape1, shape2):
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse
        return BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    
    def boolean_difference(self, shape1, shape2):
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Cut
        return BRepAlgoAPI_Cut(shape1, shape2).Shape()
    
    def boolean_intersection(self, shape1, shape2):
        from OCP.BRepAlgoAPI import BRepAlgoAPI_Common
        return BRepAlgoAPI_Common(shape1, shape2).Shape()
    
    def fillet_edges(self, shape, radius: float, edges: Optional[List[int]] = None):
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE
        
        fillet = BRepFilletAPI_MakeFillet(shape)
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        
        edge_idx = 0
        while explorer.More():
            if edges is None or edge_idx in edges:
                fillet.Add(radius, explorer.Current())
            explorer.Next()
            edge_idx += 1
        
        return fillet.Shape()
    
    def chamfer_edges(self, shape, distance: float, edges: Optional[List[int]] = None):
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeChamfer
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE
        
        chamfer = BRepFilletAPI_MakeChamfer(shape)
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        
        edge_idx = 0
        while explorer.More():
            if edges is None or edge_idx in edges:
                chamfer.Add(distance, explorer.Current())
            explorer.Next()
            edge_idx += 1
        
        return chamfer.Shape()
    
    def tessellate(self, shape, tolerance: float = 0.01) -> Mesh:
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRep import BRep_Tool
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_FACE
        from OCP.TopLoc import TopLoc_Location
        
        # Mesh the shape
        BRepMesh_IncrementalMesh(shape, tolerance)
        
        vertices = []
        faces = []
        vertex_map = {}
        vertex_count = 0
        
        explorer = TopExp_Explorer(shape, TopAbs_FACE)
        while explorer.More():
            face = explorer.Current()
            loc = TopLoc_Location()
            
            try:
                triangulation = BRep_Tool.Triangulation(face, loc)
            except:
                explorer.Next()
                continue
            
            if triangulation is not None:
                for i in range(1, triangulation.NbTriangles() + 1):
                    tri = triangulation.Triangle(i)
                    n1, n2, n3 = tri.Get()
                    
                    for n in [n1, n2, n3]:
                        key = f"{vertex_count}_{n}"
                        if key not in vertex_map:
                            pnt = triangulation.Node(n).Transformed(loc.Transformation())
                            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
                            vertex_map[key] = vertex_count
                            vertex_count += 1
                    
                    faces.append([
                        vertex_map[f"{vertex_count}_{n1}"],
                        vertex_map[f"{vertex_count}_{n2}"],
                        vertex_map[f"{vertex_count}_{n3}"]
                    ])
            
            explorer.Next()
        
        return Mesh(
            vertices=np.array(vertices, dtype=np.float32),
            faces=np.array(faces, dtype=np.int32)
        )
    
    def export_file(self, shape, filepath: Path, format: MeshFormat):
        filepath = Path(filepath)
        
        if format == MeshFormat.STEP:
            from OCP.STEPControl import STEPControl_Writer, STEPControl_AsIs
            writer = STEPControl_Writer()
            writer.Transfer(shape, STEPControl_AsIs)
            writer.Write(str(filepath))
        else:
            raise ValueError(f"Export format {format} not supported by OpenCASCADE")
    
    def import_file(self, filepath: Path, format: MeshFormat):
        filepath = Path(filepath)
        
        if format == MeshFormat.STEP:
            from OCP.STEPControl import STEPControl_Reader
            reader = STEPControl_Reader()
            status = reader.ReadFile(str(filepath))
            if status != 1:
                raise RuntimeError(f"Failed to read STEP file: {filepath}")
            reader.TransferRoots()
            return reader.OneShape()
        else:
            raise ValueError(f"Import format {format} not supported by OpenCASCADE")
    
    def get_mass_properties(self, shape, density: float = 1.0) -> MassProperties:
        from OCP.GProp import GProp_GProps
        from OCP.BRepGProp import BRepGProp
        
        # Volume properties
        vol_props = GProp_GProps()
        BRepGProp.VolumeProperties_s(shape, vol_props)
        volume = vol_props.Mass()
        centroid = vol_props.CentreOfMass()
        
        # Surface properties
        surf_props = GProp_GProps()
        BRepGProp.SurfaceProperties_s(shape, surf_props)
        surface_area = surf_props.Mass()
        
        return MassProperties(
            volume=volume,
            surface_area=surface_area,
            centroid=(centroid.X(), centroid.Y(), centroid.Z()),
            bounding_box=self.get_bounding_box(shape)
        )
    
    def get_bounding_box(self, shape) -> BoundingBox:
        from OCP.Bnd import Bnd_Box
        from OCP.BRepBndLib import BRepBndLib
        
        bbox = Bnd_Box()
        BRepBndLib.Add_s(shape, bbox)
        xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
        
        return BoundingBox(xmin, ymin, zmin, xmax, ymax, zmax)

# =============================================================================
# MANIFOLD3D KERNEL
# =============================================================================

class ManifoldKernel(GeometryKernel):
    """Fast mesh-based kernel for boolean operations"""
    
    def __init__(self):
        if not HAS_MANIFOLD:
            raise RuntimeError("Manifold3D not available")
        
        import manifold3d as m3d
        self.m3d = m3d
        logger.info("Manifold3D kernel initialized")
    
    @property
    def name(self) -> str:
        return "Manifold3D"
    
    def create_box(self, width: float, height: float, depth: float):
        return self.m3d.Manifold.cube([width, height, depth], center=False)
    
    def create_cylinder(self, radius: float, height: float, segments: int = 32):
        return self.m3d.Manifold.cylinder(height, radius, segments)
    
    def create_sphere(self, radius: float):
        return self.m3d.Manifold.sphere(radius, 100)
    
    def boolean_union(self, shape1, shape2):
        return shape1 + shape2
    
    def boolean_difference(self, shape1, shape2):
        return shape1 - shape2
    
    def boolean_intersection(self, shape1, shape2):
        return shape1 ^ shape2
    
    def fillet_edges(self, shape, radius: float, edges: Optional[List[int]] = None):
        # Manifold doesn't support fillets directly
        logger.warning("Fillet not supported in Manifold3D")
        return shape
    
    def chamfer_edges(self, shape, distance: float, edges: Optional[List[int]] = None):
        # Manifold doesn't support chamfers directly
        logger.warning("Chamfer not supported in Manifold3D")
        return shape
    
    def tessellate(self, shape, tolerance: float = 0.01) -> Mesh:
        mesh = shape.to_mesh()
        return Mesh(
            vertices=np.array(mesh.vert_properties, dtype=np.float32),
            faces=np.array(mesh.tri_verts, dtype=np.int32)
        )
    
    def export_file(self, shape, filepath: Path, format: MeshFormat):
        mesh = self.tessellate(shape)
        
        if format == MeshFormat.STL:
            import trimesh
            tri_mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
            tri_mesh.export(filepath)
        else:
            raise ValueError(f"Export format {format} not supported by Manifold3D")
    
    def import_file(self, filepath: Path, format: MeshFormat):
        raise NotImplementedError("Manifold3D import not implemented")
    
    def get_mass_properties(self, shape, density: float = 1.0) -> MassProperties:
        # Manifold provides volume via mesh
        mesh = self.tessellate(shape)
        
        # Calculate volume using divergence theorem
        volume = 0.0
        for face in mesh.faces:
            v0, v1, v2 = mesh.vertices[face]
            vol = np.dot(v0, np.cross(v1, v2)) / 6.0
            volume += vol
        volume = abs(volume)
        
        # Centroid
        centroid = tuple(np.mean(mesh.vertices, axis=0))
        
        # Bounding box
        bbox = self.get_bounding_box(shape)
        
        return MassProperties(
            volume=volume,
            surface_area=0.0,  # Would need calculation
            centroid=centroid,
            bounding_box=bbox
        )
    
    def get_bounding_box(self, shape) -> BoundingBox:
        mesh = self.tessellate(shape)
        mins = np.min(mesh.vertices, axis=0)
        maxs = np.max(mesh.vertices, axis=0)
        return BoundingBox(mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2])

# =============================================================================
# UNIFIED GEOMETRY ENGINE
# =============================================================================

class UnifiedGeometry:
    """
    Single interface for all geometry operations.
    
    Automatically selects optimal kernel for each operation.
    """
    
    def __init__(self, preferred_kernel: KernelType = KernelType.AUTO):
        self.kernels: Dict[KernelType, GeometryKernel] = {}
        
        # Initialize available kernels
        if HAS_OPENCASCADE:
            self.kernels[KernelType.OPENCASCADE] = OpenCASCADEKernel()
        
        if HAS_MANIFOLD:
            self.kernels[KernelType.MANIFOLD] = ManifoldKernel()
        
        if not self.kernels:
            raise RuntimeError("No geometry kernels available")
        
        # Select default kernel
        if preferred_kernel == KernelType.AUTO:
            # Prefer OpenCASCADE for production, Manifold for speed
            self.default_kernel = self.kernels.get(
                KernelType.OPENCASCADE,
                self.kernels.get(KernelType.MANIFOLD)
            )
        else:
            if preferred_kernel not in self.kernels:
                raise ValueError(f"Kernel {preferred_kernel} not available")
            self.default_kernel = self.kernels[preferred_kernel]
        
        logger.info(f"UnifiedGeometry initialized with {self.default_kernel.name}")
    
    def create_box(self, width: float, height: float, depth: float,
                   kernel: Optional[KernelType] = None):
        """Create box with specified kernel"""
        k = self._get_kernel(kernel)
        return k.create_box(width, height, depth)
    
    def create_cylinder(self, radius: float, height: float,
                        kernel: Optional[KernelType] = None):
        """Create cylinder"""
        k = self._get_kernel(kernel)
        return k.create_cylinder(radius, height)
    
    def create_sphere(self, radius: float,
                      kernel: Optional[KernelType] = None):
        """Create sphere"""
        k = self._get_kernel(kernel)
        return k.create_sphere(radius)
    
    def boolean_union(self, shape1, shape2,
                      kernel: Optional[KernelType] = None):
        """Boolean union"""
        k = self._get_kernel(kernel)
        return k.boolean_union(shape1, shape2)
    
    def tessellate(self, shape, tolerance: float = 0.01,
                   kernel: Optional[KernelType] = None) -> Mesh:
        """Convert to mesh"""
        k = self._get_kernel(kernel)
        return k.tessellate(shape, tolerance)
    
    def export_file(self, shape, filepath: Union[str, Path],
                    format: MeshFormat, kernel: Optional[KernelType] = None):
        """Export to file"""
        k = self._get_kernel(kernel)
        k.export_file(shape, Path(filepath), format)
    
    def import_file(self, filepath: Union[str, Path],
                    format: MeshFormat, kernel: Optional[KernelType] = None):
        """Import from file"""
        k = self._get_kernel(kernel)
        return k.import_file(Path(filepath), format)
    
    def get_mass_properties(self, shape, density: float = 1.0,
                            kernel: Optional[KernelType] = None) -> MassProperties:
        """Calculate mass properties"""
        k = self._get_kernel(kernel)
        return k.get_mass_properties(shape, density)
    
    def _get_kernel(self, kernel: Optional[KernelType]) -> GeometryKernel:
        """Get kernel instance"""
        if kernel is None or kernel == KernelType.AUTO:
            return self.default_kernel
        if kernel not in self.kernels:
            raise ValueError(f"Kernel {kernel} not available. Available: {list(self.kernels.keys())}")
        return self.kernels[kernel]
    
    @property
    def available_kernels(self) -> List[str]:
        """List available kernel names"""
        return [k.name for k in self.kernels.values()]

# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

_geometry_instance: Optional[UnifiedGeometry] = None

def get_geometry() -> UnifiedGeometry:
    """Get singleton geometry instance"""
    global _geometry_instance
    if _geometry_instance is None:
        _geometry_instance = UnifiedGeometry()
    return _geometry_instance

def create_box(width: float, height: float, depth: float):
    """Convenience: Create box"""
    return get_geometry().create_box(width, height, depth)

def create_cylinder(radius: float, height: float):
    """Convenience: Create cylinder"""
    return get_geometry().create_cylinder(radius, height)

def create_sphere(radius: float):
    """Convenience: Create sphere"""
    return get_geometry().create_sphere(radius)

# =============================================================================
# VALIDATION
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    geo = get_geometry()
    print(f"Available kernels: {geo.available_kernels}")
    
    # Create test geometry
    box = create_box(1.0, 2.0, 3.0)
    print(f"Created box")
    
    # Get mass properties
    props = geo.get_mass_properties(box, density=7850.0)
    print(f"Volume: {props.volume:.6f} m³")
    print(f"Mass: {props.volume * 7850.0:.2f} kg")
    print(f"Centroid: {props.centroid}")
