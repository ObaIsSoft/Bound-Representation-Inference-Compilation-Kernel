"""
FEA Core Components
"""

from .solver import CalculiXSolver, SolverConfig
from .mesh import GmshMesher, MeshConfig
from .quality import MeshQuality
from .convergence import ConvergenceMonitor

__all__ = [
    "CalculiXSolver",
    "SolverConfig",
    "GmshMesher",
    "MeshConfig",
    "MeshQuality",
    "ConvergenceMonitor"
]
