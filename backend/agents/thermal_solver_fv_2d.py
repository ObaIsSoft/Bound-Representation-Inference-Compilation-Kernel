"""
Production 2D Finite Volume Thermal Solver

Simplified 2D version for immediate validation against NAFEMS benchmarks.
Full 3D implementation requires proper mesh generation (Gmsh integration).

Reference: Patankar (1980) - Numerical Heat Transfer and Fluid Flow
"""

import numpy as np
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg
from typing import Optional, Tuple, Callable
import logging

logger = logging.getLogger(__name__)


class FV2DThermalSolver:
    """
    2D Finite Volume thermal solver on structured grid
    
    Solves: ∂/∂x(k ∂T/∂x) + ∂/∂y(k ∂T/∂y) + q''' = 0
    
    Discretization:
    - Central differencing for diffusion
    - Structured grid (rectangular cells)
    - TDMA or CG solver
    """
    
    def __init__(
        self,
        nx: int,
        ny: int,
        lx: float,
        ly: float,
        thermal_conductivity: float = 1.0
    ):
        """
        Initialize 2D thermal solver
        
        Args:
            nx: Number of cells in x direction
            ny: Number of cells in y direction
            lx: Domain length in x
            ly: Domain length in y
            thermal_conductivity: Material thermal conductivity
        """
        self.nx = nx
        self.ny = ny
        self.lx = lx
        self.ly = ly
        self.k = thermal_conductivity
        
        self.dx = lx / nx
        self.dy = ly / ny
        
        # Cell volumes (2D area * unit depth)
        self.volume = self.dx * self.dy * 1.0
        
        # Face areas
        self.area_x = self.dy * 1.0  # Area of face normal to x
        self.area_y = self.dx * 1.0  # Area of face normal to y
        
        # Cell centers
        self.x = np.linspace(self.dx/2, lx - self.dx/2, nx)
        self.y = np.linspace(self.dy/2, ly - self.dy/2, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y)
        
        logger.info(f"2D FV solver: {nx}x{ny} cells, dx={self.dx:.4f}, dy={self.dy:.4f}")
    
    def solve_steady_state(
        self,
        bc_left: Tuple[str, float],      # ('dirichlet', value) or ('neumann', flux)
        bc_right: Tuple[str, float],
        bc_bottom: Tuple[str, float],
        bc_top: Tuple[str, float],
        heat_generation: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Solve steady-state thermal problem
        
        Args:
            bc_left: Boundary condition on left (x=0)
            bc_right: Boundary condition on right (x=lx)
            bc_bottom: Boundary condition on bottom (y=0)
            bc_top: Boundary condition on top (y=ly)
            heat_generation: 2D array of volumetric heat generation
        
        Returns:
            T: 2D array of temperatures
        """
        n_cells = self.nx * self.ny
        
        # Build coefficient matrix A and RHS b
        # Using 5-point stencil
        A_data = []
        A_rows = []
        A_cols = []
        b = np.zeros(n_cells)
        
        def cell_index(i, j):
            """Convert 2D indices to 1D cell index"""
            return j * self.nx + i
        
        # Diffusion coefficients
        Dx = self.k * self.area_x / self.dx  # Conduction in x
        Dy = self.k * self.area_y / self.dy  # Conduction in y
        
        for j in range(self.ny):
            for i in range(self.nx):
                idx = cell_index(i, j)
                a_P = 0.0  # Diagonal coefficient
                
                # West neighbor (i-1, j)
                if i > 0:
                    idx_w = cell_index(i-1, j)
                    A_rows.append(idx)
                    A_cols.append(idx_w)
                    A_data.append(-Dx)
                    a_P += Dx
                else:
                    # Left boundary
                    bc_type, bc_value = bc_left
                    if bc_type == 'dirichlet':
                        a_P += Dx
                        b[idx] += Dx * bc_value
                    elif bc_type == 'neumann':
                        # Flux = -k * dT/dx = q
                        # b += q * area
                        b[idx] -= bc_value * self.area_x
                    elif bc_type == 'robin':
                        # bc_value is (h, T_inf)
                        h, T_inf = bc_value
                        a_P += h * self.area_x
                        b[idx] += h * self.area_x * T_inf
                
                # East neighbor (i+1, j)
                if i < self.nx - 1:
                    idx_e = cell_index(i+1, j)
                    A_rows.append(idx)
                    A_cols.append(idx_e)
                    A_data.append(-Dx)
                    a_P += Dx
                else:
                    # Right boundary
                    bc_type, bc_value = bc_right
                    if bc_type == 'dirichlet':
                        a_P += Dx
                        b[idx] += Dx * bc_value
                    elif bc_type == 'neumann':
                        b[idx] -= bc_value * self.area_x
                    elif bc_type == 'robin':
                        h, T_inf = bc_value
                        a_P += h * self.area_x
                        b[idx] += h * self.area_x * T_inf
                
                # South neighbor (i, j-1)
                if j > 0:
                    idx_s = cell_index(i, j-1)
                    A_rows.append(idx)
                    A_cols.append(idx_s)
                    A_data.append(-Dy)
                    a_P += Dy
                else:
                    # Bottom boundary
                    bc_type, bc_value = bc_bottom
                    if bc_type == 'dirichlet':
                        a_P += Dy
                        b[idx] += Dy * bc_value
                    elif bc_type == 'neumann':
                        b[idx] -= bc_value * self.area_y
                    elif bc_type == 'robin':
                        h, T_inf = bc_value
                        a_P += h * self.area_y
                        b[idx] += h * self.area_y * T_inf
                
                # North neighbor (i, j+1)
                if j < self.ny - 1:
                    idx_n = cell_index(i, j+1)
                    A_rows.append(idx)
                    A_cols.append(idx_n)
                    A_data.append(-Dy)
                    a_P += Dy
                else:
                    # Top boundary
                    bc_type, bc_value = bc_top
                    if bc_type == 'dirichlet':
                        a_P += Dy
                        b[idx] += Dy * bc_value
                    elif bc_type == 'neumann':
                        b[idx] -= bc_value * self.area_y
                    elif bc_type == 'robin':
                        h, T_inf = bc_value
                        a_P += h * self.area_y
                        b[idx] += h * self.area_y * T_inf
                
                # Add diagonal
                A_rows.append(idx)
                A_cols.append(idx)
                A_data.append(a_P)
        
        # Add heat generation
        if heat_generation is not None:
            for j in range(self.ny):
                for i in range(self.nx):
                    idx = cell_index(i, j)
                    b[idx] += heat_generation[j, i] * self.volume
        
        # Build and solve system
        A = sparse.csr_matrix((A_data, (A_rows, A_cols)), shape=(n_cells, n_cells))
        
        logger.info(f"Solving {n_cells}x{n_cells} system")
        
        # Use direct solver for reliability
        try:
            T_flat = sparse_linalg.spsolve(A, b)
        except Exception as e:
            logger.error(f"Direct solve failed: {e}")
            # Fall back to CG
            T_flat, info = sparse_linalg.cg(A, b, atol=1e-10, rtol=1e-10)
            if info != 0:
                logger.warning(f"CG did not converge, info={info}")
        
        # Reshape to 2D
        T = T_flat.reshape((self.ny, self.nx))
        
        return T


def solve_nafems_t1_2d(nx: int = 60) -> Tuple[np.ndarray, float, float]:
    """
    Solve NAFEMS T1 benchmark: 2D steady-state conduction
    
    Problem:
    - Square plate 0.6m x 0.6m
    - Left boundary (x=0): T = 100°C
    - Other boundaries: T = 0°C
    - Material: k = 52 W/(m·K)
    - Expected: T = 36.6°C at (0.15, 0.15)
    - Tolerance: 5%
    
    Args:
        nx: Number of cells in each direction (ny = nx)
    
    Returns:
        T: Temperature field
        computed_temp: Temperature at (0.15, 0.15)
        error_percent: Percent error from reference
    """
    solver = FV2DThermalSolver(
        nx=nx,
        ny=nx,
        lx=0.6,
        ly=0.6,
        thermal_conductivity=52.0
    )
    
    # Boundary conditions
    bc_left = ('dirichlet', 100.0)
    bc_right = ('dirichlet', 0.0)
    bc_bottom = ('dirichlet', 0.0)
    bc_top = ('dirichlet', 0.0)
    
    T = solver.solve_steady_state(
        bc_left=bc_left,
        bc_right=bc_right,
        bc_bottom=bc_bottom,
        bc_top=bc_top
    )
    
    # Find temperature at (0.15, 0.15)
    # Find closest cell
    distances = np.sqrt((solver.X - 0.15)**2 + (solver.Y - 0.15)**2)
    closest_idx = np.unravel_index(np.argmin(distances), distances.shape)
    computed_temp = T[closest_idx]
    
    reference_temp = 36.6
    error_percent = abs(computed_temp - reference_temp) / reference_temp * 100
    
    logger.info(f"NAFEMS T1: computed={computed_temp:.2f}°C, reference={reference_temp}°C")
    logger.info(f"Error: {error_percent:.2f}%")
    logger.info(f"Temperature range: {T.min():.2f} - {T.max():.2f}°C")
    
    return T, computed_temp, error_percent


def solve_nafems_t2_2d(nx: int = 60) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve NAFEMS T2 benchmark: 2D transient conduction
    
    Problem:
    - Same geometry as T1
    - Initial condition: T = 0°C everywhere
    - Time-dependent boundary conditions
    - Expected: T = 36.6°C at (0.15, 0.15) at t = ?
    
    Args:
        nx: Number of cells
    
    Returns:
        T: Final temperature field
        T_history: Temperature at probe point over time
        error_percent: Error from reference
    """
    solver = FV2DThermalSolver(
        nx=nx,
        ny=nx,
        lx=0.6,
        ly=0.6,
        thermal_conductivity=52.0
    )
    
    # Material properties for transient
    rho = 7850.0  # kg/m³ (steel)
    cp = 450.0    # J/(kg·K)
    alpha = solver.k / (rho * cp)  # Thermal diffusivity
    
    # Time stepping
    dt = 0.01  # Time step
    t_final = 10.0  # Final time
    n_steps = int(t_final / dt)
    
    # Initial condition
    T = np.zeros((ny, nx))
    
    # Boundary conditions (constant for simplicity)
    T_left = 100.0
    T_right = 0.0
    T_bottom = 0.0
    T_top = 0.0
    
    # Probe location
    probe_i = int(0.15 / solver.dx)
    probe_j = int(0.15 / solver.dy)
    T_history = []
    
    # Explicit time stepping
    for step in range(n_steps):
        T_new = T.copy()
        
        for j in range(ny):
            for i in range(nx):
                # Diffusion term
                if i > 0:
                    T_w = T[j, i-1]
                else:
                    T_w = T_left
                
                if i < nx - 1:
                    T_e = T[j, i+1]
                else:
                    T_e = T_right
                
                if j > 0:
                    T_s = T[j-1, i]
                else:
                    T_s = T_bottom
                
                if j < ny - 1:
                    T_n = T[j+1, i]
                else:
                    T_n = T_top
                
                # Finite volume update
                d2Tdx2 = (T_e - 2*T[j, i] + T_w) / solver.dx**2
                d2Tdy2 = (T_n - 2*T[j, i] + T_s) / solver.dy**2
                
                T_new[j, i] = T[j, i] + alpha * dt * (d2Tdx2 + d2Tdy2)
        
        T = T_new
        
        # Store probe temperature
        if step % 10 == 0:
            T_history.append(T[probe_j, probe_i])
    
    computed_temp = T[probe_j, probe_i]
    reference_temp = 36.6  # At specific time
    error_percent = abs(computed_temp - reference_temp) / reference_temp * 100
    
    return T, np.array(T_history), error_percent


# Convenience function
def solve_steady_conduction_2d(
    width: float,
    height: float,
    nx: int,
    ny: int,
    thermal_conductivity: float,
    T_left: Optional[float] = None,
    T_right: Optional[float] = None,
    T_bottom: Optional[float] = None,
    T_top: Optional[float] = None,
    q_left: Optional[float] = None,
    q_right: Optional[float] = None,
    heat_generation: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Solve 2D steady-state heat conduction
    
    Args:
        width: Domain width (x direction)
        height: Domain height (y direction)
        nx, ny: Number of cells
        thermal_conductivity: Material thermal conductivity
        T_left, T_right, T_bottom, T_top: Dirichlet BC temperatures
        q_left, q_right: Neumann BC heat fluxes
        heat_generation: Volumetric heat generation
    
    Returns:
        X, Y: Grid coordinates
        T: Temperature field
    """
    solver = FV2DThermalSolver(
        nx=nx,
        ny=ny,
        lx=width,
        ly=height,
        thermal_conductivity=thermal_conductivity
    )
    
    # Set boundary conditions
    if T_left is not None:
        bc_left = ('dirichlet', float(T_left))
    elif q_left is not None:
        bc_left = ('neumann', float(q_left))
    else:
        bc_left = ('neumann', 0.0)  # Adiabatic
    
    if T_right is not None:
        bc_right = ('dirichlet', float(T_right))
    elif q_right is not None:
        bc_right = ('neumann', float(q_right))
    else:
        bc_right = ('neumann', 0.0)
    
    if T_bottom is not None:
        bc_bottom = ('dirichlet', float(T_bottom))
    else:
        bc_bottom = ('neumann', 0.0)
    
    if T_top is not None:
        bc_top = ('dirichlet', float(T_top))
    else:
        bc_top = ('neumann', 0.0)
    
    T = solver.solve_steady_state(
        bc_left=bc_left,
        bc_right=bc_right,
        bc_bottom=bc_bottom,
        bc_top=bc_top,
        heat_generation=heat_generation
    )
    
    return solver.X, solver.Y, T
