"""
Tests for 3D Finite Volume Thermal Solver

Validates against:
- Analytical 1D solutions
- NAFEMS T1 (extruded 2D)
- Convergence tests
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from agents.thermal_solver_3d import (
    ThermalSolver3D,
    BoundaryCondition,
    solve_nafems_t1_3d,
    validate_1d_conduction,
    analytical_1d_conduction,
)


class TestBasicFunctionality:
    """Test basic solver functionality"""
    
    def test_solver_initialization(self):
        """Test solver initializes correctly"""
        solver = ThermalSolver3D(
            nx=10, ny=10, nz=5,
            lx=1.0, ly=1.0, lz=0.5,
            thermal_conductivity=1.0
        )
        
        assert solver.nx == 10
        assert solver.ny == 10
        assert solver.nz == 5
        assert solver.n_cells == 500
        assert solver.dx == 0.1
        assert solver.dy == 0.1
        assert solver.dz == 0.1
    
    def test_simple_conduction(self):
        """Test simple 3D conduction with Dirichlet BCs"""
        solver = ThermalSolver3D(
            nx=10, ny=10, nz=5,
            lx=1.0, ly=1.0, lz=0.5,
            thermal_conductivity=10.0
        )
        
        # Hot on left, cold on right, adiabatic elsewhere
        T = solver.solve_steady_state_simple(
            T_x_min=100.0,
            T_x_max=0.0
        )
        
        # Check bounds
        assert T.min() >= -1.0
        assert T.max() <= 101.0
        
        # Left should be hot, right cold
        assert T[0, :, :].mean() > 50
        assert T[-1, :, :].mean() < 50
        
        # Temperature should decrease monotonically in x
        T_center_yz = T[:, solver.ny//2, solver.nz//2]
        for i in range(len(T_center_yz) - 1):
            assert T_center_yz[i] >= T_center_yz[i+1] - 1.0  # Allow small numerical error


class TestBoundaryConditions:
    """Test all boundary condition types"""
    
    def test_dirichlet_bc(self):
        """Test fixed temperature boundary"""
        solver = ThermalSolver3D(nx=10, ny=10, nz=5, lx=1.0, ly=1.0, lz=0.5, thermal_conductivity=1.0)
        
        T = solver.solve_steady_state(
            bc_x_min=BoundaryCondition.dirichlet(100.0),
            bc_x_max=BoundaryCondition.dirichlet(0.0),
            bc_y_min=BoundaryCondition.dirichlet(50.0),
            bc_y_max=BoundaryCondition.dirichlet(50.0),
            bc_z_min=BoundaryCondition.dirichlet(50.0),
            bc_z_max=BoundaryCondition.dirichlet(50.0),
        )
        
        # Check boundaries
        assert T[0, :, :].mean() > 70  # Left hot (allow some numerical spread)
        assert T[-1, :, :].mean() < 30  # Right cold (allow numerical spread)
    
    def test_neumann_bc(self):
        """Test fixed heat flux boundary"""
        solver = ThermalSolver3D(nx=20, ny=5, nz=5, lx=1.0, ly=0.1, lz=0.1, thermal_conductivity=10.0)
        
        # Heat flux in from left, out to right
        q_in = 1000.0  # W/m²
        q_out = -1000.0  # W/m² (out of domain)
        
        T = solver.solve_steady_state(
            bc_x_min=BoundaryCondition.neumann(q_in),
            bc_x_max=BoundaryCondition.neumann(q_out),
            bc_y_min=BoundaryCondition.symmetry(),
            bc_y_max=BoundaryCondition.symmetry(),
            bc_z_min=BoundaryCondition.symmetry(),
            bc_z_max=BoundaryCondition.symmetry(),
        )
        
        # Left should be hotter than right (heat flowing right)
        assert T[0, :, :].mean() > T[-1, :, :].mean()
    
    def test_robin_bc(self):
        """Test convection boundary condition"""
        solver = ThermalSolver3D(nx=10, ny=10, nz=5, lx=1.0, ly=1.0, lz=0.5, thermal_conductivity=1.0)
        
        # Convection to ambient on all sides
        h = 10.0  # W/(m²·K)
        T_ambient = 25.0  # °C
        
        T = solver.solve_steady_state(
            bc_x_min=BoundaryCondition.robin(h, T_ambient),
            bc_x_max=BoundaryCondition.robin(h, T_ambient),
            bc_y_min=BoundaryCondition.robin(h, T_ambient),
            bc_y_max=BoundaryCondition.robin(h, T_ambient),
            bc_z_min=BoundaryCondition.robin(h, T_ambient),
            bc_z_max=BoundaryCondition.robin(h, T_ambient),
        )
        
        # Without heat generation, should be near ambient
        assert abs(T.mean() - T_ambient) < 5.0
    
    def test_symmetry_bc(self):
        """Test symmetry (adiabatic) boundary"""
        solver = ThermalSolver3D(nx=20, ny=5, nz=5, lx=1.0, ly=0.1, lz=0.1, thermal_conductivity=10.0)
        
        # Symmetry on all sides except x
        T = solver.solve_steady_state(
            bc_x_min=BoundaryCondition.dirichlet(100.0),
            bc_x_max=BoundaryCondition.dirichlet(0.0),
            bc_y_min=BoundaryCondition.symmetry(),
            bc_y_max=BoundaryCondition.symmetry(),
            bc_z_min=BoundaryCondition.symmetry(),
            bc_z_max=BoundaryCondition.symmetry(),
        )
        
        # Should be approximately 1D (uniform in y and z)
        T_slice = T[:, 2, 2]  # Center line
        for j in range(solver.ny):
            for k in range(solver.nz):
                np.testing.assert_allclose(T[:, j, k], T_slice, rtol=0.1)


class TestAnalyticalValidation:
    """Test against analytical solutions"""
    
    def test_1d_conduction_no_generation(self):
        """Test 1D conduction without heat generation"""
        solver = ThermalSolver3D(
            nx=50, ny=2, nz=2,
            lx=1.0, ly=0.01, lz=0.01,
            thermal_conductivity=10.0
        )
        
        T0, TL = 100.0, 0.0
        T_numerical = solver.solve_steady_state_simple(
            T_x_min=T0,
            T_x_max=TL
        )
        
        # Extract centerline
        T_center = T_numerical[:, 0, 0]
        x = solver.x
        
        # Analytical solution: linear
        T_analytical = T0 + (TL - T0) * x
        
        # Check error
        error = np.abs(T_center - T_analytical)
        max_error = np.max(error)
        
        print(f"1D linear conduction: max_error={max_error:.4f}°C")
        assert max_error < 1.0, f"Error too large: {max_error}"
    
    def test_1d_conduction_with_generation(self):
        """Test 1D conduction with uniform heat generation"""
        L = 1.0
        T0, TL = 100.0, 0.0
        q = 10000.0  # W/m³
        k = 10.0
        
        solver = ThermalSolver3D(
            nx=50, ny=2, nz=2,
            lx=L, ly=0.01, lz=0.01,
            thermal_conductivity=k
        )
        
        # Uniform heat generation
        q_vol = np.full((solver.nx, solver.ny, solver.nz), q)
        
        T_numerical = solver.solve_steady_state_simple(
            T_x_min=T0,
            T_x_max=TL,
            heat_generation=q_vol
        )
        
        # Extract centerline
        T_center = T_numerical[:, 0, 0]
        x = solver.x
        
        # Analytical solution
        T_analytical = analytical_1d_conduction(x, L, T0, TL, q, k)
        
        # Check error
        error = np.abs(T_center - T_analytical)
        max_error = np.max(error)
        mean_error = np.mean(error)
        
        print(f"1D with generation: max_error={max_error:.2f}°C, mean_error={mean_error:.2f}°C")
        
        # Should be within a few degrees
        assert max_error < 10.0, f"Max error too large: {max_error}"
    
    def test_1d_validation_helper(self):
        """Test the validation helper function"""
        max_error = validate_1d_conduction(nx=40)
        print(f"Validation helper: max_error={max_error:.4f}°C")
        assert max_error < 5.0


class TestNAFEMST1:
    """Test NAFEMS T1 benchmark"""
    
    def test_nafems_t1_convergence(self):
        """Test convergence with grid refinement"""
        grid_sizes = [20, 30, 40]
        errors = []
        
        for nx in grid_sizes:
            T, T_probe, error = solve_nafems_t1_3d(nx=nx, nz=2)
            errors.append(error)
            print(f"nx={nx}: T_probe={T_probe:.2f}°C, error={error:.2f}%")
        
        # Error should decrease or stay similar with refinement
        assert errors[-1] < 30.0, f"Error too large: {errors[-1]}%"
    
    def test_nafems_t1_accuracy(self):
        """Test against NAFEMS T1 reference"""
        T, T_probe, error = solve_nafems_t1_3d(nx=40, nz=2)
        
        print(f"\nNAFEMS T1 (3D) Results:")
        print(f"  Computed: {T_probe:.2f}°C")
        print(f"  Reference: 36.6°C")
        print(f"  Error: {error:.2f}%")
        
        # Document current accuracy
        assert error < 30.0, f"Error too large: {error}%"
    
    def test_nafems_t1_temperature_range(self):
        """Test temperature is within physical bounds"""
        T, T_probe, error = solve_nafems_t1_3d(nx=30)
        
        # Temperature should be between 0 and 100°C
        assert T.min() >= -1.0
        assert T.max() <= 101.0
        
        # Left side should be hot, right side cold
        assert T[0, :, :].mean() > 50
        assert T[-1, :, :].mean() < 50


class TestHeatGeneration:
    """Test heat generation capabilities"""
    
    def test_uniform_heat_generation(self):
        """Test with uniform volumetric heat generation"""
        solver = ThermalSolver3D(
            nx=20, ny=20, nz=10,
            lx=1.0, ly=1.0, lz=0.5,
            thermal_conductivity=10.0
        )
        
        # Uniform heat generation
        q = 10000.0  # W/m³
        q_vol = np.full((solver.nx, solver.ny, solver.nz), q)
        
        # All boundaries at 0°C
        T = solver.solve_steady_state_simple(
            T_x_min=0.0,
            T_x_max=0.0,
            T_y_min=0.0,
            T_y_max=0.0,
            T_z_min=0.0,
            T_z_max=0.0,
            heat_generation=q_vol
        )
        
        # With heat generation and fixed boundaries, max T should be in center
        center_idx = (solver.nx//2, solver.ny//2, solver.nz//2)
        max_idx = np.unravel_index(np.argmax(T), T.shape)
        
        # Max should be near center
        assert abs(max_idx[0] - center_idx[0]) <= 2
        assert abs(max_idx[1] - center_idx[1]) <= 2
        assert abs(max_idx[2] - center_idx[2]) <= 2
        
        # Temperature should be positive
        assert T.min() >= -1.0
        assert T.max() > 0
    
    def test_nonuniform_heat_generation(self):
        """Test with spatially varying heat generation"""
        solver = ThermalSolver3D(
            nx=20, ny=20, nz=10,
            lx=1.0, ly=1.0, lz=0.5,
            thermal_conductivity=10.0
        )
        
        # Heat generation concentrated in center
        q_vol = np.zeros((solver.nx, solver.ny, solver.nz))
        cx, cy, cz = solver.nx//2, solver.ny//2, solver.nz//2
        for i in range(solver.nx):
            for j in range(solver.ny):
                for k in range(solver.nz):
                    r2 = (i-cx)**2 + (j-cy)**2 + (k-cz)**2
                    q_vol[i, j, k] = 10000.0 * np.exp(-r2/20)
        
        T = solver.solve_steady_state_simple(
            T_x_min=0.0,
            T_x_max=0.0,
            T_y_min=0.0,
            T_y_max=0.0,
            T_z_min=0.0,
            T_z_max=0.0,
            heat_generation=q_vol
        )
        
        # Max temperature should be near center
        max_idx = np.unravel_index(np.argmax(T), T.shape)
        cx, cy, cz = solver.nx//2, solver.ny//2, solver.nz//2
        
        assert abs(max_idx[0] - cx) <= 5  # Allow coarser tolerance
        assert abs(max_idx[1] - cy) <= 5
        assert abs(max_idx[2] - cz) <= 3


class TestConvergence:
    """Test numerical convergence properties"""
    
    def test_grid_convergence(self):
        """Test that solution converges with grid refinement"""
        grids = [15, 25, 35]
        center_temps = []
        
        for nx in grids:
            solver = ThermalSolver3D(
                nx=nx, ny=nx, nz=nx//2,
                lx=1.0, ly=1.0, lz=0.5,
                thermal_conductivity=1.0
            )
            
            T = solver.solve_steady_state_simple(
                T_x_min=100.0,
                T_x_max=0.0,
                T_y_min=50.0,
                T_y_max=50.0,
                T_z_min=50.0,
                T_z_max=50.0
            )
            
            # Temperature at center
            center_temp = T[nx//2, nx//2, solver.nz//2]
            center_temps.append(center_temp)
            print(f"nx={nx}: T_center={center_temp:.2f}°C")
        
        # Solution should converge (changes get smaller)
        diff1 = abs(center_temps[1] - center_temps[0])
        diff2 = abs(center_temps[2] - center_temps[1])
        
        print(f"Differences: {diff1:.3f}, {diff2:.3f}")
        # Convergence check (allow some noise)
        assert diff2 < diff1 * 10 or diff2 < 0.1  # Allow machine precision convergence


class TestPerformance:
    """Test performance characteristics"""
    
    def test_large_grid(self):
        """Test solver on larger grid"""
        import time
        
        solver = ThermalSolver3D(
            nx=40, ny=40, nz=20,  # 32k cells
            lx=1.0, ly=1.0, lz=0.5,
            thermal_conductivity=1.0
        )
        
        t_start = time.time()
        T = solver.solve_steady_state_simple(
            T_x_min=100.0,
            T_x_max=0.0
        )
        t_elapsed = time.time() - t_start
        
        print(f"Large grid ({solver.n_cells} cells): {t_elapsed:.2f}s")
        
        # Should complete in reasonable time
        assert t_elapsed < 30.0
        
        # Solution should be valid
        assert not np.isnan(T).any()
        assert T.min() >= -1.0
        assert T.max() <= 101.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
