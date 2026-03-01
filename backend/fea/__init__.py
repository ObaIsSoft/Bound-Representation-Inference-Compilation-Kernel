"""
BRICK OS Finite Element Analysis (FEA) Module

Phase 2: FEA Integration (FIX-201 through FIX-208)

This module provides:
- CalculiX solver integration (FIX-201)
- Gmsh mesh generation (FIX-202)
- Mesh quality metrics (FIX-203)
- Boundary condition handling (FIX-204)
- Convergence monitoring (FIX-205)
- FEA input file generators (FIX-206)
- Result parsing (FIX-207)
- Mesh convergence studies (FIX-208)
"""

from .core.solver import CalculiXSolver, SolverConfig
from .core.mesh import GmshMesher, MeshConfig
from .core.quality import MeshQuality
from .bc.boundary_conditions import BoundaryConditionManager
from .post.parser import ResultParser

__version__ = "1.0.0"

__all__ = [
    "CalculiXSolver",
    "SolverConfig",
    "GmshMesher",
    "MeshConfig",
    "MeshQuality",
    "BoundaryConditionManager",
    "ResultParser"
]
