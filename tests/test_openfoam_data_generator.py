"""
Tests for OpenFOAM Data Generator

Validates synthetic data generation and OpenFOAM integration.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
import sys
import os

# Ensure backend is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.agents.openfoam_data_generator import (
    CFDParameters,
    CFDResult,
    OpenFOAMRunner,
    SyntheticDataGenerator,
    generate_training_data,
    load_training_data
)


class TestCFDParameters:
    """Test CFDParameters dataclass"""
    
    def test_default_creation(self):
        """Test creating CFDParameters with defaults"""
        params = CFDParameters(
            reynolds_number=1000,
            shape_type="sphere",
            characteristic_length=1.0
        )
        
        assert params.reynolds_number == 1000
        assert params.shape_type == "sphere"
        assert params.mach_number == 0.0
        assert params.density == 1.225  # Default air
    
    def test_compute_reynolds(self):
        """Test Reynolds number computation"""
        params = CFDParameters(
            reynolds_number=0,  # Will compute
            velocity=10.0,
            characteristic_length=1.0,
            density=1.225,
            viscosity=1.81e-5,
            shape_type="cylinder"
        )
        
        Re = params.compute_reynolds()
        expected_Re = (1.225 * 10.0 * 1.0) / 1.81e-5
        assert abs(Re - expected_Re) / expected_Re < 0.01
    
    def test_custom_properties(self):
        """Test custom fluid properties"""
        params = CFDParameters(
            reynolds_number=100,
            shape_type="airfoil",
            characteristic_length=1.0,
            density=998.0,  # Water
            viscosity=1.0e-3,
            velocity=5.0
        )
        
        assert params.density == 998.0
        assert params.viscosity == 1.0e-3


class TestOpenFOAMRunner:
    """Test OpenFOAM runner"""
    
    def test_initialization(self):
        """Test runner initialization"""
        runner = OpenFOAMRunner()
        assert hasattr(runner, 'has_openfoam')
        assert hasattr(runner, 'openfoam_cmd')
    
    def test_synthetic_result_generation(self):
        """Test synthetic CFD result generation"""
        runner = OpenFOAMRunner()
        
        params = CFDParameters(
            reynolds_number=100,
            shape_type="sphere",
            characteristic_length=1.0
        )
        
        result = runner._generate_synthetic_result(params)
        
        assert isinstance(result, CFDResult)
        assert result.cd > 0
        assert result.converged
        assert result.params == params
    
    def test_sphere_drag_low_re(self):
        """Test sphere drag at low Reynolds number (Stokes flow)"""
        runner = OpenFOAMRunner()
        
        params = CFDParameters(
            reynolds_number=0.1,
            shape_type="sphere",
            characteristic_length=1.0
        )
        
        result = runner._generate_synthetic_result(params)
        
        # Stokes drag: Cd = 24/Re
        expected_cd = 24 / 0.1
        assert abs(result.cd - expected_cd) / expected_cd < 0.2
    
    def test_sphere_drag_high_re(self):
        """Test sphere drag at high Reynolds number"""
        runner = OpenFOAMRunner()
        
        params = CFDParameters(
            reynolds_number=10000,
            shape_type="sphere",
            characteristic_length=1.0
        )
        
        result = runner._generate_synthetic_result(params)
        
        # Should be around 0.4-0.5 for turbulent regime
        assert 0.3 < result.cd < 0.6
    
    def test_cylinder_drag(self):
        """Test cylinder drag"""
        runner = OpenFOAMRunner()
        
        params = CFDParameters(
            reynolds_number=100,
            shape_type="cylinder",
            characteristic_length=1.0
        )
        
        result = runner._generate_synthetic_result(params)
        
        # Cylinder drag around 1.0-1.5
        assert 0.5 < result.cd < 2.0
    
    def test_flow_fields_generated(self):
        """Test that flow fields are generated"""
        runner = OpenFOAMRunner()
        
        params = CFDParameters(
            reynolds_number=100,
            shape_type="sphere",
            characteristic_length=1.0
        )
        
        result = runner._generate_synthetic_result(params)
        
        assert result.velocity_field is not None
        assert result.pressure_field is not None
        assert result.velocity_field.ndim == 3  # (n, n, 2) for 2D velocity


class TestSyntheticDataGenerator:
    """Test synthetic data generator"""
    
    def test_initialization(self):
        """Test generator initialization"""
        generator = SyntheticDataGenerator(seed=42)
        assert generator.rng is not None
    
    def test_generate_parameter_space(self):
        """Test parameter space generation"""
        generator = SyntheticDataGenerator(seed=42)
        
        params = generator.generate_parameter_space(10)
        
        assert len(params) == 10
        assert all(isinstance(p, CFDParameters) for p in params)
    
    def test_parameter_ranges(self):
        """Test that parameters are in expected ranges"""
        generator = SyntheticDataGenerator(seed=42)
        
        params = generator.generate_parameter_space(100)
        
        for p in params:
            # Reynolds number: 10 to 1e6
            assert 10 <= p.reynolds_number <= 1e6
            # Aspect ratio: 1 to 10
            assert 1.0 <= p.aspect_ratio <= 10.0
            # Porosity: 0 to 0.5
            assert 0.0 <= p.porosity <= 0.5
            # Shape type from allowed list
            assert p.shape_type in ["sphere", "cylinder", "box"]
    
    def test_generate_dataset(self):
        """Test full dataset generation"""
        generator = SyntheticDataGenerator(seed=42)
        
        X, y = generator.generate_dataset(n_samples=50)
        
        assert X.shape == (50, 4)
        assert y.shape == (50, 1)
        assert not np.any(np.isnan(X))
        assert not np.any(np.isnan(y))
    
    def test_dataset_reproducibility(self):
        """Test that same seed gives same parameter space"""
        generator1 = SyntheticDataGenerator(seed=42)
        generator2 = SyntheticDataGenerator(seed=42)
        
        # Parameter space (X) should be identical
        X1, _ = generator1.generate_dataset(n_samples=20)
        X2, _ = generator2.generate_dataset(n_samples=20)
        
        np.testing.assert_array_almost_equal(X1, X2)
        # Note: Cd values (y) have random perturbation, so won't be identical
    
    def test_save_and_load(self):
        """Test saving and loading dataset"""
        generator = SyntheticDataGenerator(seed=42)
        
        X, y = generator.generate_dataset(n_samples=30)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_data.json"
            
            generator.save_dataset(X, y, str(filepath), metadata={"test": True})
            
            assert filepath.exists()
            
            # Load and verify
            X_loaded, y_loaded = load_training_data(str(filepath))
            
            np.testing.assert_array_almost_equal(X, X_loaded)
            np.testing.assert_array_almost_equal(y, y_loaded)


class TestConvenienceFunctions:
    """Test convenience functions"""
    
    def test_generate_training_data(self):
        """Test generate_training_data function"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "training_data.json"
            
            X, y = generate_training_data(
                n_samples=25,
                output_path=str(output_path)
            )
            
            assert X.shape == (25, 4)
            assert y.shape == (25, 1)
            assert output_path.exists()
    
    def test_load_training_data(self):
        """Test load_training_data function"""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "training_data.json"
            
            X_orig, y_orig = generate_training_data(
                n_samples=20,
                output_path=str(output_path)
            )
            
            X_loaded, y_loaded = load_training_data(str(output_path))
            
            np.testing.assert_array_almost_equal(X_orig, X_loaded)
            np.testing.assert_array_almost_equal(y_orig, y_loaded)


class TestCaseDirectoryCreation:
    """Test OpenFOAM case directory structure"""
    
    def test_directory_structure(self):
        """Test that case directories are created correctly"""
        runner = OpenFOAMRunner()
        
        params = CFDParameters(
            reynolds_number=1000,
            shape_type="cylinder",
            characteristic_length=1.0
        )
        
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = runner.create_case_directory(params, Path(tmpdir))
            
            # Check directory exists
            assert case_dir.exists()
            
            # Check standard OpenFOAM structure
            assert (case_dir / "0").exists()
            assert (case_dir / "constant").exists()
            assert (case_dir / "constant" / "polyMesh").exists()
            assert (case_dir / "system").exists()


class TestCFDResult:
    """Test CFDResult dataclass"""
    
    def test_basic_result(self):
        """Test basic CFD result creation"""
        result = CFDResult(cd=1.2)
        
        assert result.cd == 1.2
        assert result.cl == 0.0  # Default
        assert result.converged is True  # Default
    
    def test_result_with_fields(self):
        """Test CFD result with flow fields"""
        velocity = np.random.randn(32, 32, 2)
        pressure = np.random.randn(32, 32)
        
        result = CFDResult(
            cd=0.8,
            velocity_field=velocity,
            pressure_field=pressure
        )
        
        assert result.velocity_field.shape == (32, 32, 2)
        assert result.pressure_field.shape == (32, 32)


class TestPhysicsCorrelations:
    """Test physics correlations in synthetic data"""
    
    def test_reynolds_dependence(self):
        """Test that drag decreases with Re (roughly)"""
        runner = OpenFOAMRunner()
        
        # Compare low and high Re
        params_low = CFDParameters(reynolds_number=10, shape_type="sphere", characteristic_length=1.0)
        params_high = CFDParameters(reynolds_number=10000, shape_type="sphere", characteristic_length=1.0)
        
        result_low = runner._generate_synthetic_result(params_low)
        result_high = runner._generate_synthetic_result(params_high)
        
        # High Re should have lower Cd than low Re (sphere)
        assert result_high.cd < result_low.cd
    
    def test_shape_variation(self):
        """Test that different shapes have different drag"""
        runner = OpenFOAMRunner()
        
        shapes = ["sphere", "cylinder", "box"]
        cds = []
        
        for shape in shapes:
            params = CFDParameters(reynolds_number=1000, shape_type=shape, characteristic_length=1.0)
            result = runner._generate_synthetic_result(params)
            cds.append(result.cd)
        
        # Different shapes should have different drag coefficients
        assert len(set([round(cd, 1) for cd in cds])) > 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
