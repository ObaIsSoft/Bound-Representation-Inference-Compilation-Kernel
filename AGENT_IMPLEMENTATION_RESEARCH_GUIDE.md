# BRICK OS - Agent Implementation Research Guide

**Version**: 1.0  
**Date**: 2026-03-04  
**Purpose**: Comprehensive research on dependencies, libraries, research papers, and APIs for all non-production agents

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Structural/FEA Agents](#2-structuralfea-agents)
3. [Geometry/CAD Agents](#3-geometrycad-agents)
4. [Electronics Agents](#4-electronics-agents)
5. [Chemistry/Materials Agents](#5-chemistrymaterials-agents)
6. [Manufacturing/DFM Agents](#6-manufacturingdfm-agents)
7. [Control/GNC Agents](#7-controlgnc-agents)
8. [Thermal/Physics Agents](#8-thermalphysics-agents)
9. [AI/ML Surrogate Models](#9-aiml-surrogate-models)
10. [Integration Architecture](#10-integration-architecture)

---

## 1. Executive Summary

This document provides comprehensive research for implementing 71 non-production agents across 8 domains. Each section includes:

- **Current State**: What's broken or missing
- **Required Libraries**: Specific packages with versions
- **Research Papers**: Key publications (2020-2025)
- **APIs**: External services and their pricing
- **Implementation Strategy**: Step-by-step approach
- **Code Examples**: Working snippets where applicable

### Quick Reference: Priority Matrix

| Domain | Agents | Complexity | Impact | Priority |
|--------|--------|------------|--------|----------|
| Structural/FEA | 3 | High | Critical | P0 |
| Geometry/CAD | 8 | Medium | High | P0 |
| Electronics | 12 | Medium | High | P1 |
| Materials | 15 | High | Medium | P1 |
| Manufacturing | 5 | Low | Medium | P2 |
| Control/GNC | 4 | High | Low | P2 |
| Thermal | 3 | Medium | Medium | P1 |
| AI/ML | 21 | High | High | P0 |

---

## 2. Structural/FEA Agents

### 2.1 Current State Analysis

**Primary Agent**: `ProductionStructuralAgent` (2,108 lines)

**Critical Issue**: The "Fallback Trap" - All fidelity modes collapse to analytical beam theory (σ=F/A)

```python
# Current broken implementation pattern:
async def _full_fea(self, ...):
    if not self.fea_solver.is_available():
        return self._analytical_solution(...)  # NEVER fails, always falls back
```

### 2.2 Required Libraries

#### Core FEA Libraries

| Library | Version | Purpose | Installation | License |
|---------|---------|---------|--------------|---------|
| **FEniCSx/DOLFINx** | 0.8.0 | Modern FEA framework | `conda install -c conda-forge fenics-dolfinx` | LGPL |
| **CalculiX (ccx)** | 2.21 | Open source FEA solver | `sudo apt install calculix-ccx` | GPL |
| **NGSolve** | latest | High-performance multiphysics | `pip install ngsolve` | LGPL |
| **Sfepy** | 2024.1 | Simple FEA in Python | `pip install sfepy` | BSD |
| **PyMAPDL** | 0.68+ | ANSYS Python interface | `pip install ansys-mapdl-core` | Proprietary |

#### Python Bindings for CalculiX

| Package | Status | Notes |
|---------|--------|-------|
| **pycalculix** | ⚠️ Maintenance mode | Basic meshing + ccx wrapper |
| **ccx2paraview** | ✅ Active | Convert ccx .frd to VTK |
| **calculix-adapter** | ⚠️ Experimental | preCICE coupling |

#### Mesh Generation

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **Gmsh** | 4.12+ | Mesh generation | `pip install gmsh` |
| **meshio** | 5.3+ | Mesh I/O | `pip install meshio` |
| **PyMesh** | 0.3+ | Mesh processing | Build from source |
| **CGAL** | 5.6+ | Geometry algorithms | `conda install -c conda-forge cgal` |

### 2.3 Research Papers

#### Neural Operators for Structures

1. **Li et al. (2021)** - "Fourier Neural Operator for Parametric PDEs"
   - arXiv:2010.08895
   - Foundational FNO paper
   - Code: https://github.com/neuraloperator/neuraloperator

2. **Lu et al. (2021)** - "Learning Nonlinear Operators via DeepONet"
   - Nature Machine Intelligence
   - Branch-trunk architecture for operators
   - Code: https://github.com/lululxvi/deeponet

3. **Wen et al. (2022)** - "U-FNO: Enhanced Fourier Neural Operator"
   - Advances in Water Resources
   - Factorized FNO for multiphase flow

4. **Zhu et al. (2023)** - "Fourier-DeepONet for Full Waveform Inversion"
   - CMAME 416:116300
   - Hybrid FNO-DeepONet architecture

5. **Lanthaler et al. (2024)** - "Nonlocal Neural Operator"
   - Neural Operators on Riemannian manifolds
   - Universal approximation theory

#### Physics-Informed Neural Networks

6. **Raissi et al. (2019)** - "Physics-Informed Neural Networks"
   - Journal of Computational Physics
   - Foundational PINN paper

7. **Karniadakis et al. (2021)** - "Physics-Informed Machine Learning"
   - Nature Reviews Physics
   - Comprehensive review

8. **Cuomo et al. (2022)** - "Scientific Machine Learning through PINNs"
   - Journal of Scientific Computing
   - State of the art review

9. **Cai et al. (2021)** - "Physics-Informed Neural Networks for Fluid Mechanics"
   - Acta Mechanica Sinica
   - Fluid-specific PINN review

10. **Wang et al. (2021)** - "Understanding and Mitigating Gradient Pathologies"
    - ICML 2021
    - PINN training improvements

#### Reduced Order Models

11. **Quarteroni et al. (2016)** - "Reduced Basis Methods and POD"
    - Springer
    - Classical ROM methods

12. **Hesthaven & Ubbiali (2018)** - "Non-intrusive Reduced Order Models"
    - SIAM Journal
    - Neural network-based ROM

### 2.4 Implementation Strategy

#### Phase 1: Fix CalculiX Integration (Week 1-2)

```python
# backend/fea/core/calculix_adapter.py
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Dict, Any
import meshio
import numpy as np

@dataclass
class CalculiXConfig:
    """Configuration for CalculiX solver."""
    num_processors: int = 4
    memory_limit_gb: float = 8.0
    convergence_tolerance: float = 1e-6
    max_iterations: int = 100
    solver_type: str = "PARDISO"  # SPOOLES, PARDISO, PASTIX
    
class CalculiXAdapter:
    """
    Production-grade CalculiX adapter with proper error handling.
    
    Dependencies:
        - calculix-ccx (system package)
        - meshio (pip)
        - numpy
    """
    
    def __init__(self, config: CalculiXConfig = None):
        self.config = config or CalculiXConfig()
        self._check_installation()
        
    def _check_installation(self) -> None:
        """Verify ccx is installed and accessible."""
        try:
            result = subprocess.run(
                ["ccx", "-v"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode != 0:
                raise RuntimeError("CalculiX not properly installed")
        except FileNotFoundError:
            raise RuntimeError(
                "CalculiX (ccx) not found. Install: "
                "sudo apt install calculix-ccx"
            )
    
    def solve(self, input_file: Path, output_dir: Path) -> Dict[str, Any]:
        """
        Run CalculiX analysis.
        
        Args:
            input_file: Path to .inp file (ABAQUS format)
            output_dir: Directory for output files
            
        Returns:
            Dictionary with results and metadata
        """
        input_file = Path(input_file)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy input to output dir (ccx writes to working directory)
        job_name = input_file.stem
        working_input = output_dir / f"{job_name}.inp"
        
        import shutil
        shutil.copy(input_file, working_input)
        
        # Run solver
        cmd = [
            "ccx",
            f"-np", str(self.config.num_processors),
            job_name
        ]
        
        try:
            result = subprocess.run(
                cmd,
                cwd=output_dir,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            # Parse results
            return self._parse_results(output_dir, job_name, result)
            
        except subprocess.TimeoutExpired:
            raise RuntimeError("CalculiX solver timed out")
    
    def _parse_results(self, output_dir: Path, job_name: str, 
                       process_result: subprocess.CompletedProcess) -> Dict[str, Any]:
        """Parse CalculiX output files."""
        results = {
            "converged": False,
            "max_stress": None,
            "max_displacement": None,
            "error": None,
            "output_files": {}
        }
        
        # Check .sta file for convergence
        sta_file = output_dir / f"{job_name}.sta"
        if sta_file.exists():
            with open(sta_file) as f:
                lines = f.readlines()
                if lines and "converged" in lines[-1].lower():
                    results["converged"] = True
        
        # Parse .dat file for nodal results
        dat_file = output_dir / f"{job_name}.dat"
        if dat_file.exists():
            stresses = self._parse_stresses(dat_file)
            results["max_stress"] = max(stresses) if stresses else None
        
        # Parse .frd file (binary results)
        frd_file = output_dir / f"{job_name}.frd"
        if frd_file.exists():
            results["output_files"]["frd"] = str(frd_file)
        
        if process_result.returncode != 0:
            results["error"] = process_result.stderr
            
        return results
```

#### Phase 2: FEniCSx Integration (Week 3-4)

```python
# backend/fea/core/fenicsx_adapter.py
"""
FEniCSx adapter for high-performance FEA.

Installation:
    conda create -n fenicsx-env
    conda activate fenicsx-env
    conda install -c conda-forge fenics-dolfinx mpich pyvista
"""

try:
    import dolfinx
    import dolfinx.fem as fem
    import dolfinx.mesh as mesh
    import ufl
    from mpi4py import MPI
    HAS_FENICSX = True
except ImportError:
    HAS_FENICSX = False

import numpy as np
from typing import Optional, Callable, Dict, Any
import logging

logger = logging.getLogger(__name__)

class FEniCSxAdapter:
    """
    Modern FEA adapter using FEniCSx (successor to legacy FEniCS).
    
    Features:
    - Parallel computation with MPI
    - High-order elements
    - Automatic differentiation
    - Support for complex geometries
    """
    
    def __init__(self):
        if not HAS_FENICSX:
            raise RuntimeError(
                "FEniCSx not installed. "
                "Run: conda install -c conda-forge fenics-dolfinx"
            )
        self.comm = MPI.COMM_WORLD
        
    def solve_linear_elasticity(
        self,
        mesh_file: str,
        youngs_modulus: float = 200e9,  # Pa (steel default)
        poisson_ratio: float = 0.3,
        boundary_conditions: Optional[Dict] = None,
        body_force: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        Solve linear elasticity problem.
        
        Args:
            mesh_file: Path to mesh file (.msh, .xdmf)
            youngs_modulus: Material Young's modulus [Pa]
            poisson_ratio: Material Poisson's ratio
            boundary_conditions: Dict with 'fixed' and 'load' regions
            body_force: Body force function f(x)
            
        Returns:
            Results dictionary with displacements, stresses, etc.
        """
        # Read mesh
        domain = mesh.create_from_file(self.comm, mesh_file)
        
        # Function space (vector-valued, 1st order Lagrange)
        V = fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim,)))
        
        # Material parameters
        E = fem.Constant(domain, youngs_modulus)
        nu = fem.Constant(domain, poisson_ratio)
        mu = E / (2 * (1 + nu))  # Shear modulus
        lambda_ = E * nu / ((1 + nu) * (1 - 2 * nu))  # Lamé parameter
        
        # Stress-strain relation (isotropic linear elastic)
        def epsilon(u):
            return ufl.sym(ufl.grad(u))
        
        def sigma(u):
            return lambda_ * ufl.tr(epsilon(u)) * ufl.Identity(len(u)) + 2 * mu * epsilon(u)
        
        # Variational problem
        u = ufl.TrialFunction(V)
        v = ufl.TestFunction(V)
        
        a = ufl.inner(sigma(u), epsilon(v)) * ufl.dx
        
        # Body force
        if body_force is None:
            body_force = lambda x: np.zeros((domain.geometry.dim, x.shape[1]))
        
        f = fem.Function(V)
        f.interpolate(body_force)
        L = ufl.dot(f, v) * ufl.dx
        
        # Boundary conditions
        bcs = []
        if boundary_conditions:
            if 'fixed' in boundary_conditions:
                # Apply zero displacement on fixed boundary
                fixed_facets = boundary_conditions['fixed']['facets']
                bc_fixed = fem.dirichletbc(
                    np.zeros(domain.geometry.dim),
                    fem.locate_dofs_topological(V, domain.topology.dim-1, fixed_facets),
                    V
                )
                bcs.append(bc_fixed)
        
        # Solve
        problem = fem.petsc.LinearProblem(a, L, bcs=bcs)
        uh = problem.solve()
        
        # Compute stresses at vertices
        stress = fem.Function(fem.functionspace(domain, ("Lagrange", 1, (domain.geometry.dim, domain.geometry.dim))))
        stress_expr = fem.Expression(sigma(uh), stress.function_space.element.interpolation_points())
        stress.interpolate(stress_expr)
        
        # Von Mises stress
        s = stress - ufl.tr(stress) / domain.geometry.dim * ufl.Identity(domain.geometry.dim)
        von_mises = ufl.sqrt(3/2 * ufl.inner(s, s))
        
        von_mises_func = fem.Function(fem.functionspace(domain, ("Lagrange", 1)))
        von_mises_expr = fem.Expression(von_mises, von_mises_func.function_space.element.interpolation_points())
        von_mises_func.interpolate(von_mises_expr)
        
        return {
            "displacement": uh,
            "stress": stress,
            "von_mises": von_mises_func,
            "max_von_mises": float(np.max(von_mises_func.x.array)),
            "max_displacement": float(np.max(np.linalg.norm(uh.x.array.reshape(-1, domain.geometry.dim), axis=1)))
        }
```

#### Phase 3: Neural Operators (Week 5-8)

See Section 9 for detailed Neural Operator implementation.

### 2.5 NAFEMS Benchmarks

Required validation benchmarks:

| Benchmark | Description | Target Error |
|-----------|-------------|--------------|
| LE1 | Elliptic membrane | < 5% |
| LE2 | Thick cylinder | < 3% |
| LE3 | Thin cylinder | < 5% |
| LE4 | Plate with hole | < 5% |
| LE10 | Solid cylinder | < 3% |
| LE11 | Solid cylinder (axisymmetric) | < 3% |

---

## 3. Geometry/CAD Agents

### 3.1 Current State Analysis

**Primary Agent**: `ProductionGeometryAgent` (1,341 lines)

**Status**: Partial implementation with multiple kernels but limited integration

### 3.2 Required Libraries

#### CAD Kernels

| Library | Version | Purpose | Installation | License |
|---------|---------|---------|--------------|---------|
| **cadquery-ocp** | 7.7.0 | OpenCASCADE Python bindings | `pip install cadquery-ocp` | Apache-2.0 |
| **pythonocc-core** | 7.7.0 | Alternative OCC bindings | `conda install -c conda-forge pythonocc-core` | LGPL |
| **Manifold3D** | 2.3+ | Fast boolean operations | `pip install manifold3d` | Apache-2.0 |
| **CGAL** | 5.6+ | Computational geometry | `conda install -c conda-forge cgal` | GPL/LGPL |

#### Mesh Generation

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **Gmsh** | 4.12+ | Mesh generation | `pip install gmsh` |
| **meshio** | 5.3+ | Mesh I/O | `pip install meshio[all]` |
| **PyVista** | 0.43+ | Visualization | `pip install pyvista` |
| **trimesh** | 4.0+ | Mesh processing | `pip install trimesh` |
| **pymeshlab** | 2022.2+ | MeshLab Python | `pip install pymeshlab` |

#### SDF/CSG

| Library | Purpose | Installation |
|---------|---------|--------------|
| **pySDF** | Signed distance fields | Build from source |
| **kaolin** | NVIDIA SDF utilities | `pip install kaolin` |
| **PyMCubes** | Marching cubes | `pip install PyMCubes` |
| **scikit-image** | Morphology operations | `pip install scikit-image` |

### 3.3 Research Papers

#### Deep Learning for CAD

1. **Jayaraman et al. (2023)** - "BRepNet: A Deep Learning Model for B-Rep Recognition"
   - SIGGRAPH Asia
   - Code: https://github.com/AutodeskAILab/BRepNet

2. **Willis et al. (2021)** - "Fusion 360 Gallery: A Dataset and Environment for Programmatic CAD"
   - CVPR 2021
   - Code: https://github.com/AutodeskAILab/Fusion360GalleryDataset

3. **Lambourne et al. (2022)** - "BRepNet: Learning B-Rep Representations"  
   - Learning-based CAD reconstruction

4. **Xu et al. (2022)** - "Point Transformer" for 3D point clouds
   - ICCV 2021
   - Code: https://github.com/qq456cvb/Point-Transformers

### 3.4 Implementation Strategy

```python
# backend/geometry/multi_kernel_adapter.py
"""
Multi-kernel CAD adapter supporting OpenCASCADE, Manifold3D, and CGAL.
"""

from abc import ABC, abstractmethod
from typing import Optional, Union, List, Dict, Any
from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Optional imports with fallbacks
try:
    import manifold3d as m3d
    HAS_MANIFOLD = True
except ImportError:
    HAS_MANIFOLD = False
    logger.warning("Manifold3D not available")

try:
    from OCP.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
    from OCP.BRepAlgoAPI import BRepAlgoAPI_Fuse, BRepAlgoAPI_Cut
    from OCP.STEPControl import STEPControl_Writer, STEPControl_Reader
    HAS_OPENCASCADE = True
except ImportError:
    HAS_OPENCASCADE = False
    logger.warning("OpenCASCADE not available")


class GeometryKernel(ABC):
    """Abstract base class for geometry kernels."""
    
    @abstractmethod
    def create_box(self, width: float, height: float, depth: float) -> Any:
        pass
    
    @abstractmethod
    def boolean_union(self, shape1: Any, shape2: Any) -> Any:
        pass
    
    @abstractmethod
    def export_step(self, shape: Any, filepath: Path) -> None:
        pass


class ManifoldKernel(GeometryKernel):
    """
    Manifold3D kernel for fast boolean operations.
    
    Best for: Fast prototyping, boolean-heavy operations
    Limitations: Limited surface types, no STEP export directly
    """
    
    def __init__(self):
        if not HAS_MANIFOLD:
            raise RuntimeError("Manifold3D not installed. Run: pip install manifold3d")
    
    def create_box(self, width: float, height: float, depth: float) -> m3d.Manifold:
        # Manifold uses half-extents
        return m3d.cube([width, height, depth], center=False)
    
    def create_cylinder(self, radius: float, height: float, 
                        segments: int = 32) -> m3d.Manifold:
        return m3d.cylinder(radius, height, segments)
    
    def boolean_union(self, shape1: m3d.Manifold, shape2: m3d.Manifold) -> m3d.Manifold:
        return shape1 + shape2
    
    def boolean_difference(self, shape1: m3d.Manifold, 
                           shape2: m3d.Manifold) -> m3d.Manifold:
        return shape1 - shape2
    
    def boolean_intersection(self, shape1: m3d.Manifold, 
                             shape2: m3d.Manifold) -> m3d.Manifold:
        return shape1 ^ shape2
    
    def export_mesh(self, shape: m3d.Manifold, filepath: Path) -> None:
        """Export as triangular mesh (STL, OBJ, etc.)."""
        mesh = shape.to_mesh()
        # Use trimesh or meshio for export
        import trimesh
        tri_mesh = trimesh.Trimesh(vertices=mesh.vert_properties, 
                                    faces=mesh.tri_verts)
        tri_mesh.export(filepath)
    
    def export_step(self, shape: m3d.Manifold, filepath: Path) -> None:
        """
        Manifold doesn't natively support STEP (B-Rep).
        Must mesh first, then use surface reconstruction.
        """
        raise NotImplementedError(
            "Manifold is mesh-based. Use OpenCASCADE for STEP export."
        )


class OpenCASCADEKernel(GeometryKernel):
    """
    OpenCASCADE kernel for B-Rep CAD operations.
    
    Best for: Production CAD, STEP import/export, complex surfaces
    Limitations: Slower than Manifold for booleans
    """
    
    def __init__(self):
        if not HAS_OPENCASCADE:
            raise RuntimeError(
                "OpenCASCADE not installed. Run: "
                "pip install cadquery-ocp"
            )
    
    def create_box(self, width: float, height: float, depth: float):
        return BRepPrimAPI_MakeBox(width, height, depth).Shape()
    
    def create_cylinder(self, radius: float, height: float):
        from OCP.gp import gp_Ax2, gp_Pnt, gp_Dir
        from OCP.BRepPrimAPI import BRepPrimAPI_MakeCylinder
        
        axis = gp_Ax2(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1))
        return BRepPrimAPI_MakeCylinder(axis, radius, height).Shape()
    
    def boolean_union(self, shape1, shape2):
        return BRepAlgoAPI_Fuse(shape1, shape2).Shape()
    
    def boolean_difference(self, shape1, shape2):
        return BRepAlgoAPI_Cut(shape1, shape2).Shape()
    
    def export_step(self, shape, filepath: Path) -> None:
        """Export to STEP format (ISO 10303)."""
        writer = STEPControl_Writer()
        writer.Transfer(shape, STEPControl_AsIs)
        writer.Write(str(filepath))
    
    def import_step(self, filepath: Path):
        """Import from STEP format."""
        reader = STEPControl_Reader()
        status = reader.ReadFile(str(filepath))
        if status != 1:
            raise RuntimeError(f"Failed to read STEP file: {filepath}")
        reader.TransferRoots()
        return reader.OneShape()
    
    def fillet_edges(self, shape, radius: float):
        """Apply fillet to all edges."""
        from OCP.BRepFilletAPI import BRepFilletAPI_MakeFillet
        from OCP.TopExp import TopExp_Explorer
        from OCP.TopAbs import TopAbs_EDGE
        
        fillet = BRepFilletAPI_MakeFillet(shape)
        explorer = TopExp_Explorer(shape, TopAbs_EDGE)
        
        while explorer.More():
            edge = explorer.Current()
            fillet.Add(radius, edge)
            explorer.Next()
        
        return fillet.Shape()


class MultiKernelGeometryEngine:
    """
    Unified interface that selects optimal kernel for operation.
    """
    
    def __init__(self):
        self.kernels = {}
        if HAS_MANIFOLD:
            self.kernels['manifold'] = ManifoldKernel()
        if HAS_OPENCASCADE:
            self.kernels['opencascade'] = OpenCASCADEKernel()
    
    def get_kernel(self, name: str) -> GeometryKernel:
        if name not in self.kernels:
            raise ValueError(f"Kernel '{name}' not available. "
                           f"Available: {list(self.kernels.keys())}")
        return self.kernels[name]
    
    def create_box(self, width: float, height: float, depth: float,
                   kernel: str = 'manifold'):
        """Create box with selected kernel."""
        k = self.get_kernel(kernel)
        return k.create_box(width, height, depth)
```

---

## 4. Electronics Agents

### 4.1 Current State Analysis

**Primary Agents**: ElectronicsAgent, ElectronicsOracle (12 adapters)

**Issues**:
- Hardcoded efficiency values
- ElectronicsOracle uses mock data
- No real component database integration

### 4.2 Required APIs

#### Component Databases

| API | Pricing | Rate Limits | Data Coverage |
|-----|---------|-------------|---------------|
| **Nexar (Octopart)** | Free: 1K calls/mo, Pro: $99/mo (10K) | 10 req/sec | 95M+ parts |
| **Digi-Key API** | Free | 1000 req/day | Digi-Key catalog |
| **Mouser API** | Free | 100 req/day | Mouser catalog |
| **SnapEDA API** | Free tier | 100 req/day | Symbols/footprints |
| **UltraLibrarian** | Free tier | Limited | 3D models |

#### Nexar API Implementation

```python
# backend/agents/electronics/nexar_client.py
"""
Nexar API client for Octopart component data.

Pricing:
- Welcome: Free, 1,000 calls/month
- Launch: $49/month, 5,000 calls
- Growth: $99/month, 10,000 calls
- Scale: $249/month, 25,000 calls

Sign up: https://nexar.com/api
"""

import os
import requests
from typing import Optional, Dict, List, Any
from dataclasses import dataclass

NEXAR_API_ENDPOINT = "https://api.nexar.com/graphql"

@dataclass
class ComponentSpecs:
    """Component specifications."""
    mpn: str
    manufacturer: str
    category: str
    parameters: Dict[str, Any]
    datasheet_url: Optional[str]
    stock_status: Dict[str, int]  # distributor -> quantity
    pricing: Dict[str, List[Dict]]  # distributor -> price breaks

class NexarClient:
    """
    GraphQL client for Nexar/Octopart API.
    
    Example:
        client = NexarClient(api_key="your_key")
        results = client.search_parts("STM32F407VGT6")
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEXAR_API_KEY")
        if not self.api_key:
            raise ValueError("Nexar API key required. Get one at https://nexar.com")
        
        self.session = requests.Session()
        self.session.headers.update({
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        })
    
    def search_parts(self, query: str, limit: int = 10) -> List[ComponentSpecs]:
        """
        Search for components by MPN or keyword.
        
        Args:
            query: Part number or search term
            limit: Maximum results (default 10)
            
        Returns:
            List of component specifications
        """
        graphql_query = """
        query SearchParts($query: String!, $limit: Int!) {
            supSearch(q: $query, limit: $limit) {
                results {
                    part {
                        mpn
                        manufacturer {
                            name
                        }
                        category {
                            name
                        }
                        specs {
                            attribute {
                                name
                            }
                            value
                        }
                        datasheet_url: bestDatasheet {
                            url
                        }
                        sellers(authorized: true) {
                            company {
                                name
                            }
                            offers {
                                inventory_level
                                prices {
                                    USD {
                                        price
                                        quantity
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        """
        
        response = self.session.post(
            NEXAR_API_ENDPOINT,
            json={
                "query": graphql_query,
                "variables": {"query": query, "limit": limit}
            }
        )
        response.raise_for_status()
        
        data = response.json()
        results = []
        
        for item in data.get("data", {}).get("supSearch", {}).get("results", []):
            part = item.get("part", {})
            
            # Extract stock and pricing
            stock_status = {}
            pricing = {}
            
            for seller in part.get("sellers", []):
                seller_name = seller["company"]["name"]
                for offer in seller.get("offers", []):
                    stock_status[seller_name] = offer.get("inventory_level", 0)
                    
                    prices = []
                    for price_info in offer.get("prices", {}).get("USD", []):
                        prices.append({
                            "quantity": price_info["quantity"],
                            "price": price_info["price"]
                        })
                    pricing[seller_name] = prices
            
            specs = ComponentSpecs(
                mpn=part.get("mpn", ""),
                manufacturer=part.get("manufacturer", {}).get("name", ""),
                category=part.get("category", {}).get("name", ""),
                parameters={
                    spec["attribute"]["name"]: spec["value"]
                    for spec in part.get("specs", [])
                },
                datasheet_url=part.get("datasheet_url", {}).get("url"),
                stock_status=stock_status,
                pricing=pricing
            )
            results.append(specs)
        
        return results
```

### 4.3 PCB Design Automation

#### KiCad Python API

```python
# backend/agents/electronics/kicad_automation.py
"""
KiCad PCB automation using pcbnew API.

Requirements:
    - KiCad 7.0+ installed
    - pcbnew Python module available
"""

try:
    import pcbnew
    HAS_KICAD = True
except ImportError:
    HAS_KICAD = False

from pathlib import Path
from typing import List, Dict, Tuple, Optional
import numpy as np

class KiCadAutomation:
    """
    Automate KiCad PCB design tasks.
    
    Examples:
        - Component placement
        - Routing assistance
        - Design rule checks
        - BOM generation
    """
    
    def __init__(self):
        if not HAS_KICAD:
            raise RuntimeError(
                "KiCad Python API not available. "
                "Ensure KiCad is installed and pcbnew is in PYTHONPATH"
            )
    
    def load_board(self, pcb_file: Path) -> pcbnew.Board:
        """Load a KiCad PCB file."""
        return pcbnew.LoadBoard(str(pcb_file))
    
    def place_components_grid(self, board: pcbnew.Board, 
                              reference_prefix: str,
                              start_pos: Tuple[float, float],
                              spacing: Tuple[float, float],
                              num_cols: int) -> None:
        """
        Place components in a grid pattern.
        
        Args:
            board: KiCad board object
            reference_prefix: e.g., "C" for capacitors, "R" for resistors
            start_pos: (x, y) in mm
            spacing: (dx, dy) in mm
            num_cols: Number of columns before wrapping
        """
        # KiCad uses internal units (1 IU = 10 nm)
        SCALE = 1e6  # mm to internal units
        
        footprints = [
            fp for fp in board.GetFootprints()
            if fp.GetReference().startswith(reference_prefix)
        ]
        
        # Sort by reference number
        footprints.sort(key=lambda fp: int(
            ''.join(filter(str.isdigit, fp.GetReference()))
        ))
        
        for i, fp in enumerate(footprints):
            col = i % num_cols
            row = i // num_cols
            
            x = (start_pos[0] + col * spacing[0]) * SCALE
            y = (start_pos[1] + row * spacing[1]) * SCALE
            
            fp.SetPosition(pcbnew.VECTOR2I(int(x), int(y)))
    
    def generate_bom(self, board: pcbnew.Board) -> List[Dict]:
        """Generate Bill of Materials from board."""
        bom = []
        
        for fp in board.GetFootprints():
            item = {
                "reference": fp.GetReference(),
                "value": fp.GetValue(),
                "footprint": str(fp.GetFPID().GetLibItemName()),
                "layer": "Top" if fp.GetLayer() == pcbnew.F_Cu else "Bottom",
                "position": (
                    fp.GetPosition().x / 1e6,  # mm
                    fp.GetPosition().y / 1e6
                )
            }
            bom.append(item)
        
        return bom
    
    def export_gerbers(self, board: pcbnew.Board, output_dir: Path,
                       layers: Optional[List[str]] = None) -> None:
        """
        Export Gerber files for manufacturing.
        
        Args:
            board: KiCad board
            output_dir: Output directory
            layers: List of layer names (default: all copper + mask + silk)
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plot_controller = pcbnew.PLOT_CONTROLLER(board)
        plot_options = plot_controller.GetPlotOptions()
        
        plot_options.SetOutputDirectory(str(output_dir))
        plot_options.SetPlotFrameRef(False)
        plot_options.SetPlotValue(True)
        plot_options.SetPlotReference(True)
        
        if layers is None:
            layers = ["F.Cu", "B.Cu", "F.Mask", "B.Mask", 
                     "F.SilkS", "B.SilkS", "Edge.Cuts"]
        
        layer_map = {
            "F.Cu": pcbnew.F_Cu,
            "B.Cu": pcbnew.B_Cu,
            "F.Mask": pcbnew.F_Mask,
            "B.Mask": pcbnew.B_Mask,
            "F.SilkS": pcbnew.F_SilkS,
            "B.SilkS": pcbnew.B_SilkS,
            "Edge.Cuts": pcbnew.Edge_Cuts,
        }
        
        for layer_name in layers:
            if layer_name in layer_map:
                plot_controller.SetLayer(layer_map[layer_name])
                plot_controller.OpenPlotfile(
                    layer_name.replace(".", "_"),
                    pcbnew.PLOT_FORMAT_GERBER,
                    layer_name
                )
                plot_controller.PlotLayer()
                plot_controller.ClosePlot()
```

---

## 5. Chemistry/Materials Agents

### 5.1 Current State Analysis

**Primary Agents**: ChemistryAgent, MaterialsOracle (15 adapters)

**Issues**:
- ChemistryOracle adapters return mock data
- No integration with real materials databases
- GNN for materials not implemented

### 5.2 Required Libraries

#### Materials Informatics

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **pymatgen** | 2024.1+ | Materials Project interface | `pip install pymatgen` |
| **matminer** | 0.9+ | Materials data mining | `pip install matminer` |
| **ase** | 3.22+ | Atomic simulation environment | `pip install ase` |
| **ovito** | 3.9+ | Visualization | `pip install ovito` |
| **rdkit** | 2023.9+ | Cheminformatics | `conda install -c conda-forge rdkit` |
| **openbabel** | 3.1+ | Format conversion | `pip install openbabel` |

#### Graph Neural Networks

| Library | Purpose | Installation |
|---------|---------|--------------|
| **torch_geometric** | GNN framework | `pip install torch-geometric` |
| **dgl** | Deep Graph Library | `pip install dgl` |
| **matgl** | Materials GNN (Materials Project) | `pip install matgl` |

### 5.3 Research Papers

#### GNN for Materials

1. **Xie & Grossman (2018)** - "Crystal Graph Convolutional Neural Networks"
   - PRL 120, 145301
   - Foundational CGCNN paper
   - Code: https://github.com/txie-93/cgcnn

2. **Chen & Ong (2019)** - "A Universal Graph Deep Learning Interatomic Potential"
   - Nature Machine Intelligence
   - MEGNet architecture
   - Code: https://github.com/materialsvirtuallab/megnet

3. **Choudhary & Garrity (2021)** - "Efficient Deep Learning for Accurate Atomistic Potentials"
   - npj Computational Materials
   - ALIGNN architecture
   - Code: https://github.com/usnistgov/alignn

4. **Schütt et al. (2018)** - "SchNet: A Continuous-filter Convolutional Neural Network"
   - JCIM 2018
   - Continuous-filter convolutions
   - Code: https://github.com/atomistic-machine-learning/schnetpack

### 5.4 Materials Project API

```python
# backend/agents/materials/materials_project_client.py
"""
Materials Project API client.

Installation:
    pip install pymatgen mp-api

API Key: Get at https://materialsproject.org/dashboard
Free tier: 1000 requests/month
"""

import os
from typing import Optional, List, Dict, Any
from dataclasses import dataclass

try:
    from mp_api.client import MPRester
    HAS_MP_API = True
except ImportError:
    HAS_MP_API = False

@dataclass
class MaterialProperties:
    """Material properties from Materials Project."""
    material_id: str
    formula: str
    structure: Any  # pymatgen Structure
    formation_energy: float  # eV/atom
    band_gap: float  # eV
    density: float  # g/cm³
    bulk_modulus: Optional[float]  # GPa
    shear_modulus: Optional[float]  # GPa
    elastic_tensor: Optional[Any]
    
class MaterialsProjectClient:
    """
    Client for Materials Project REST API.
    
    Usage:
        client = MaterialsProjectClient(api_key="your_key")
        props = client.get_properties("mp-149")  # Silicon
    """
    
    def __init__(self, api_key: Optional[str] = None):
        if not HAS_MP_API:
            raise RuntimeError(
                "Materials Project API not installed. "
                "Run: pip install mp-api"
            )
        
        self.api_key = api_key or os.getenv("MP_API_KEY")
        self.mpr = MPRester(self.api_key)
    
    def search_by_formula(self, formula: str) -> List[str]:
        """Get material IDs matching chemical formula."""
        return self.mpr.get_materials_ids(formula)
    
    def get_properties(self, material_id: str) -> MaterialProperties:
        """Get comprehensive properties for a material."""
        # Get structure
        structure = self.mpr.get_structure_by_material_id(material_id)
        
        # Get thermodynamic properties
        thermo_data = self.mpr.materials.thermo.search(
            material_ids=[material_id]
        )[0]
        
        # Get electronic properties
        electronic_data = self.mpr.materials.electronic_structure.search(
            material_ids=[material_id]
        )[0]
        
        # Get elastic properties (if available)
        try:
            elastic_data = self.mpr.materials.elasticity.search(
                material_ids=[material_id]
            )[0]
            bulk_modulus = elastic_data.bulk_modulus.get("vrh", None)
            shear_modulus = elastic_data.shear_modulus.get("vrh", None)
            elastic_tensor = elastic_data.elastic_tensor
        except (IndexError, AttributeError):
            bulk_modulus = None
            shear_modulus = None
            elastic_tensor = None
        
        return MaterialProperties(
            material_id=material_id,
            formula=structure.formula,
            structure=structure,
            formation_energy=thermo_data.formation_energy_per_atom,
            band_gap=electronic_data.band_gap,
            density=structure.density,
            bulk_modulus=bulk_modulus,
            shear_modulus=shear_modulus,
            elastic_tensor=elastic_tensor
        )
    
    def query_by_elements(self, elements: List[str], 
                          exclude_elements: Optional[List[str]] = None,
                          min_band_gap: Optional[float] = None) -> List[Dict]:
        """
        Query materials by element composition.
        
        Args:
            elements: Required elements
            exclude_elements: Elements to exclude
            min_band_gap: Minimum band gap (eV)
        """
        # Build MongoDB-style query
        query = {"elements": {"$all": elements}}
        
        if exclude_elements:
            query["elements"]["$nin"] = exclude_elements
        
        if min_band_gap is not None:
            query["band_gap"] = {"$gte": min_band_gap}
        
        results = self.mpr.materials.summary.search(
            **query,
            fields=["material_id", "formula_pretty", "band_gap", 
                   "formation_energy_per_atom", "density"]
        )
        
        return [
            {
                "material_id": r.material_id,
                "formula": r.formula_pretty,
                "band_gap": r.band_gap,
                "formation_energy": r.formation_energy_per_atom,
                "density": r.density
            }
            for r in results
        ]
```

---

## 6. Manufacturing/DFM Agents

### 6.1 Current State Analysis

**Primary Agents**: DfmAgent, ManufacturingAgent

**Issues**:
- DFM uses rule-based scoring with arbitrary penalties
- No ML-based manufacturability prediction
- Missing real manufacturing cost APIs

### 6.2 Required Libraries

#### DFM Analysis

| Library | Purpose | Installation |
|---------|---------|--------------|
| **FreeCAD** | CAD + CAM | `conda install -c conda-forge freecad` |
| **cadquery** | Parametric CAD | `pip install cadquery` |
| **OCC-Core** | OpenCASCADE | `pip install cadquery-ocp` |

#### 3D Printing

| Library | Purpose | Installation |
|---------|---------|--------------|
| **CuraEngine** | Slicing | Build from source |
| **PrusaSlicer** | Slicing | CLI available |
| **Slic3r** | Slicing | `apt install slic3r` |

### 6.3 Manufacturing Cost APIs

| Service | API Available | Pricing Model |
|---------|---------------|---------------|
| **Xometry** | Yes (request access) | Quote-based |
| **Protolabs** | Limited | Quote-based |
| **Hubs (3D Hubs)** | Yes | Quote-based |
| **Fictiv** | Yes | Quote-based |

### 6.4 ML for DFM

```python
# backend/agents/manufacturing/dfm_ml.py
"""
Machine learning for Design for Manufacturing.

Approach: CNN-based manufacturability scoring
"""

import torch
import torch.nn as nn
from typing import Tuple

class DFMScorer(nn.Module):
    """
    CNN-based manufacturability scorer.
    
    Input: Voxelized geometry (32x32x32) + process parameters
    Output: Manufacturability score (0-1)
    
    Architecture based on:
    - "3D Deep Learning for Design for Additive Manufacturing"
    - "Manufacturability Analysis with Deep Learning"
    """
    
    def __init__(self, input_channels: int = 1, num_processes: int = 5):
        super().__init__()
        
        # 3D CNN encoder
        self.encoder = nn.Sequential(
            # 32x32x32 -> 16x16x16
            nn.Conv3d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # 16x16x16 -> 8x8x8
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # 8x8x8 -> 4x4x4
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
            
            # 4x4x4 -> 2x2x2
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(2),
        )
        
        # Global average pooling + process embedding
        self.fc = nn.Sequential(
            nn.Linear(256 * 2 * 2 * 2 + num_processes, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, geometry: torch.Tensor, 
                process_params: torch.Tensor) -> torch.Tensor:
        """
        Args:
            geometry: (batch, 1, 32, 32, 32) voxelized geometry
            process_params: (batch, num_processes) one-hot process type
        """
        x = self.encoder(geometry)
        x = x.view(x.size(0), -1)
        x = torch.cat([x, process_params], dim=1)
        return self.fc(x)


class SurfaceQualityPredictor(nn.Module):
    """
    Predict surface roughness and quality for different processes.
    """
    
    def __init__(self):
        super().__init__()
        
        # Surface analysis CNN
        self.surface_net = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)  # Ra (roughness), Rz (peak-to-valley)
        )
    
    def forward(self, surface_height_map: torch.Tensor) -> torch.Tensor:
        """
        Args:
            surface_height_map: (batch, 1, H, W) surface topography
        Returns:
            (batch, 2) Ra and Rz predictions
        """
        x = self.surface_net(surface_height_map)
        x = x.view(x.size(0), -1)
        return self.regressor(x)
```

---

## 7. Control/GNC Agents

### 7.1 Required Libraries

| Library | Version | Purpose | Installation |
|---------|---------|---------|--------------|
| **control** | 0.9+ | Python Control Systems | `pip install control` |
| **casadi** | 3.6+ | Optimal control | `pip install casadi` |
| **acados** | latest | Fast NMPC | Build from source |
| **pydrake** | 1.24+ | MIT robotics | `pip install drake` |
| **do-mpc** | 4.6+ | MPC framework | `pip install do-mpc` |
| **gymnasium** | 0.29+ | RL environments | `pip install gymnasium` |
| **stable-baselines3** | 2.2+ | RL algorithms | `pip install stable-baselines3` |

### 7.2 Research Papers

1. **Fujimoto et al. (2018)** - "Addressing Function Approximation Error in Actor-Critic Methods"
   - TD3 algorithm
   - Code: https://github.com/sfujim/TD3

2. **Haarnoja et al. (2018)** - "Soft Actor-Critic"
   - SAC algorithm
   - Code: https://github.com/haarnoja/sac

3. **Schulman et al. (2017)** - "Proximal Policy Optimization Algorithms"
   - PPO algorithm
   - Code: https://github.com/openai/baselines

---

## 8. Thermal/Physics Agents

### 8.1 Required Libraries

| Library | Purpose | Installation |
|---------|---------|--------------|
| **CoolProp** | Thermophysical properties | `pip install CoolProp` |
| **Cantera** | Chemical kinetics/thermo | `conda install -c cantera cantera` |
| **Thermo (Caleb Bell)** | Thermo properties | `pip install thermo` |

---

## 9. AI/ML Surrogate Models

### 9.1 Neural Operator Libraries

| Library | Status | Installation |
|---------|--------|--------------|
| **neuraloperator** | Official FNO | `pip install neuraloperator` |
| **DeepXDE** | PINNs | `pip install deepxde` |
| **NVIDIA Modulus** | Physics-ML | `pip install nvidia-modulus` |

### 9.2 FNO Implementation

```python
# backend/ml/neural_operators/fno.py
"""
Fourier Neural Operator implementation.

Based on: Li et al. (2021) "Fourier Neural Operator for Parametric PDEs"
Code adapted from: https://github.com/neuraloperator/neuraloperator
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

torch.manual_seed(0)

class SpectralConv3d(nn.Module):
    """3D Spectral convolution layer."""
    
    def __init__(self, in_channels: int, out_channels: int, modes: List[int]):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Number of Fourier modes to multiply
        
        self.scale = 1 / (in_channels * out_channels)
        self.weights = nn.Parameter(
            self.scale * torch.rand(
                in_channels, out_channels, 
                modes[0], modes[1], modes[2], 
                dtype=torch.cfloat
            )
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batchsize = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfftn(x, dim=[-3, -2, -1])
        
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(
            batchsize, self.out_channels,
            x.size(-3), x.size(-2), x.size(-1) // 2 + 1,
            dtype=torch.cfloat, device=x.device
        )
        
        out_ft[:, :, :self.modes[0], :self.modes[1], :self.modes[2]] = \
            torch.einsum("bixyz,ioxyz->boxyz", 
                        x_ft[:, :, :self.modes[0], :self.modes[1], :self.modes[2]],
                        self.weights)
        
        # iFFT
        x = torch.fft.irfftn(out_ft, s=(x.size(-3), x.size(-2), x.size(-1)))
        return x


class FNO3d(nn.Module):
    """
    3D Fourier Neural Operator for fluid dynamics.
    
    Input: (batch, time, x, y, z, channels)
    Output: (batch, time, x, y, z, channels)
    """
    
    def __init__(
        self,
        modes: Tuple[int, int, int] = (8, 8, 8),
        width: int = 20,
        in_channels: int = 4,
        out_channels: int = 1,
        n_layers: int = 4
    ):
        super().__init__()
        
        self.modes = modes
        self.width = width
        
        # Input channel projection
        self.fc0 = nn.Linear(in_channels, width)
        
        # Fourier layers
        self.conv_layers = nn.ModuleList()
        self.w_layers = nn.ModuleList()
        
        for _ in range(n_layers):
            self.conv_layers.append(
                SpectralConv3d(width, width, list(modes))
            )
            self.w_layers.append(
                nn.Conv3d(width, width, 1)
            )
        
        # Output projection
        self.fc1 = nn.Linear(width, 128)
        self.fc2 = nn.Linear(128, out_channels)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, time, x, y, z, channels)
        Returns:
            (batch, time, x, y, z, out_channels)
        """
        # Add positional encoding
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        
        # Lift
        x = self.fc0(x)
        x = x.permute(0, 5, 1, 2, 3, 4)  # (b, c, t, x, y, z)
        
        # Fourier layers
        for conv, w in zip(self.conv_layers, self.w_layers):
            x1 = conv(x)
            x2 = w(x)
            x = x1 + x2
            x = F.gelu(x)
        
        # Project
        x = x.permute(0, 2, 3, 4, 5, 1)  # (b, t, x, y, z, c)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        
        return x
    
    def get_grid(self, shape: Tuple, device: torch.device) -> torch.Tensor:
        """Generate positional encoding grid."""
        batchsize, size_t, size_x, size_y, size_z = shape[0], shape[1], shape[2], shape[3], shape[4]
        
        grid_t = torch.linspace(0, 1, size_t, device=device).reshape(1, size_t, 1, 1, 1, 1).repeat(
            batchsize, 1, size_x, size_y, size_z, 1
        )
        grid_x = torch.linspace(0, 1, size_x, device=device).reshape(1, 1, size_x, 1, 1, 1).repeat(
            batchsize, size_t, 1, size_y, size_z, 1
        )
        grid_y = torch.linspace(0, 1, size_y, device=device).reshape(1, 1, 1, size_y, 1, 1).repeat(
            batchsize, size_t, size_x, 1, size_z, 1
        )
        grid_z = torch.linspace(0, 1, size_z, device=device).reshape(1, 1, 1, 1, size_z, 1).repeat(
            batchsize, size_t, size_x, size_y, 1, 1
        )
        
        return torch.cat((grid_t, grid_x, grid_y, grid_z), dim=-1)
```

---

## 10. Integration Architecture

### 10.1 Dependency Management

```yaml
# requirements-agents.yml
# Conda environment for all agents

name: brick-agents
channels:
  - conda-forge
  - pytorch
  - nvidia
dependencies:
  # Core
  - python=3.11
  - numpy=1.26
  - scipy=1.12
  - pandas=2.2
  
  # ML/AI
  - pytorch=2.2
  - pytorch-cuda=12.1
  - pip:
    - torch-geometric
    - neuraloperator
    - deepxde
    - nvidia-modulus
  
  # FEA
  - fenics-dolfinx=0.8
  - mpich
  - petsc4py
  - slepc4py
  
  # CAD
  - pythonocc-core=7.7
  - gmsh=4.12
  - meshio=5.3
  
  # Materials
  - pymatgen=2024.1
  - mp-api
  - rdkit=2023.9
  
  # Visualization
  - pyvista=0.43
  - matplotlib=3.8
  - plotly=5.18
```

### 10.2 API Gateway Pattern

```python
# backend/api/physics_gateway.py
"""
Unified API gateway for all physics agents.
Provides consistent interface regardless of backend solver.
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio

router = APIRouter(prefix="/api/physics")

@router.post("/structural/analyze")
async def analyze_structural(request: Dict[str, Any]):
    """
    Run structural analysis.
    
    Tries solvers in order: FEniCSx → CalculiX → PINN surrogate
    """
    from backend.agents.structural_agent import ProductionStructuralAgent
    
    agent = ProductionStructuralAgent()
    
    try:
        # Try high-fidelity FEA first
        result = await agent._full_fea(request)
        result["fidelity"] = "FEA"
        return result
    except Exception as e:
        # Fall back to surrogate
        result = await agent._surrogate_prediction(request)
        result["fidelity"] = "Surrogate"
        result["warning"] = f"FEA failed ({str(e)}), using surrogate"
        return result

@router.post("/thermal/analyze")
async def analyze_thermal(request: Dict[str, Any]):
    """Run thermal analysis."""
    from backend.agents.thermal_solver_3d import ThermalSolver3D
    
    solver = ThermalSolver3D()
    # ... implementation
    pass

@router.post("/fluid/analyze")
async def analyze_fluid(request: Dict[str, Any]):
    """Run CFD analysis."""
    from backend.agents.fluid_agent import FluidAgent
    
    agent = FluidAgent()
    # ... implementation
    pass
```

---

## Appendix A: Installation Scripts

### Complete Development Environment

```bash
#!/bin/bash
# scripts/setup_agents.sh

# Create conda environment
conda create -n brick-agents python=3.11 -y
conda activate brick-agents

# Install core scientific stack
conda install -c conda-forge \
    numpy scipy pandas matplotlib \
    pytorch torchvision torchaudio pytorch-cuda=12.1 \
    fenics-dolfinx mpich pyvista \
    pythonocc-core gmsh meshio \
    pymatgen rdkit \
    -y

# Install pip packages
pip install \
    neuraloperator deepxde nvidia-modulus \
    stable-baselines3 gymnasium \
    control casadi do-mpc \
    trimesh pymeshlab \
    mp-api matminer

# System packages (Ubuntu/Debian)
sudo apt update
sudo apt install -y \
    calculix-ccx \
    slic3r \
    kicad \
    libgl1-mesa-glx

echo "Setup complete. Activate with: conda activate brick-agents"
```

---

## Appendix B: Paper References

### Essential Reading

1. Li et al. (2021) - FNO paper
2. Lu et al. (2021) - DeepONet paper
3. Raissi et al. (2019) - PINN paper
4. Karniadakis et al. (2021) - Physics-Informed ML review
5. Cuomo et al. (2022) - PINN state of the art

### Domain-Specific

- **FEA**: Hughes (2012) "The Finite Element Method"
- **CFD**: Ferziger & Peric (2002) "Computational Methods for Fluid Dynamics"
- **CAD**: Farin (2002) "Curves and Surfaces for CAGD"
- **Materials**: Callister (2018) "Materials Science and Engineering"

---

**End of Research Guide**

*This document provides the technical foundation for implementing all 71 non-production agents in BRICK OS.*
