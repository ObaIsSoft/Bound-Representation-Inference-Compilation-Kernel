"""
Test suite for VMK Process Simulation Extension

Tests G-code parsing, machining physics, and process simulation.
"""

import pytest
import numpy as np
import tempfile
import os

from backend.vmk_process_simulation import (
    GCodeParser, MachiningPhysics, MachiningParameters,
    ProcessSimulator, ProcessMetrics, simulate_machining_process
)
from backend.vmk_kernel import ToolProfile, SymbolicMachiningKernel


class TestGCodeParser:
    """Test G-code parsing functionality"""
    
    def test_parse_simple_linear(self):
        """Test parsing simple G1 linear move"""
        parser = GCodeParser()
        lines = ["G1 X10 Y20 F500"]
        ops = parser.parse_lines(lines)
        
        assert len(ops) == 1
        assert ops[0]['g_code'] == 'G1'
        assert np.allclose(ops[0]['end'], [10, 20, 0])
        assert ops[0]['feed_rate'] == 500
    
    def test_parse_rapid_move(self):
        """Test parsing G0 rapid move"""
        parser = GCodeParser()
        lines = ["G0 X50 Y50"]
        ops = parser.parse_lines(lines)
        
        assert len(ops) == 1
        assert ops[0]['is_rapid'] == True
    
    def test_parse_multiple_moves(self):
        """Test parsing sequence of moves"""
        parser = GCodeParser()
        lines = [
            "G0 X0 Y0 Z10",
            "G1 Z-1 F100",
            "G1 X50 F500",
            "G1 Y50",
        ]
        ops = parser.parse_lines(lines)
        
        assert len(ops) == 4
        # Check positions accumulate
        assert np.allclose(ops[0]['end'], [0, 0, 10])
        assert np.allclose(ops[1]['end'], [0, 0, -1])
        assert np.allclose(ops[2]['end'], [50, 0, -1])
        assert np.allclose(ops[3]['end'], [50, 50, -1])
    
    def test_parse_spindle_and_feed(self):
        """Test parsing S and F commands"""
        parser = GCodeParser()
        lines = [
            "S2000 M3",
            "G1 X10 F500"
        ]
        ops = parser.parse_lines(lines)
        
        assert ops[0]['spindle_speed'] == 2000
        assert ops[0]['feed_rate'] == parser.current_feed  # Should use default
    
    def test_skip_comments(self):
        """Test that comments are skipped"""
        parser = GCodeParser()
        lines = [
            "; This is a comment",
            "(This is also a comment)",
            "G1 X10 ; inline comment",
        ]
        ops = parser.parse_lines(lines)
        
        assert len(ops) == 1
        assert ops[0]['end'][0] == 10


class TestMachiningPhysics:
    """Test machining physics calculations"""
    
    def test_steel_coefficients(self):
        """Test steel material coefficients"""
        physics = MachiningPhysics("steel")
        assert physics.coeffs['Ktc'] == 1500
        assert physics.coeffs['specific_cutting_energy'] == 5.0
    
    def test_aluminum_coefficients(self):
        """Test aluminum material coefficients"""
        physics = MachiningPhysics("aluminum")
        assert physics.coeffs['Ktc'] == 800
        assert physics.coeffs['specific_cutting_energy'] == 1.5
    
    def test_force_calculation_steel(self):
        """Test cutting force calculation for steel"""
        physics = MachiningPhysics("steel")
        params = MachiningParameters(
            feed_rate=500,
            spindle_speed=1000,
            depth_of_cut=2,
            width_of_cut=5,
            tool_radius=5
        )
        
        forces = physics.calculate_forces(params)
        
        assert forces['Ft'] > 0
        assert forces['Fr'] > 0
        assert forces['resultant'] > 0
        assert forces['torque_nm'] > 0
        assert forces['power_kw'] > 0
    
    def test_force_calculation_aluminum(self):
        """Test that aluminum has lower forces than steel"""
        steel_physics = MachiningPhysics("steel")
        alum_physics = MachiningPhysics("aluminum")
        
        params = MachiningParameters(
            feed_rate=500,
            spindle_speed=1000,
            depth_of_cut=2,
            width_of_cut=5,
            tool_radius=5
        )
        
        steel_forces = steel_physics.calculate_forces(params)
        alum_forces = alum_physics.calculate_forces(params)
        
        assert alum_forces['resultant'] < steel_forces['resultant']
    
    def test_tool_wear_calculation(self):
        """Test tool wear calculation"""
        physics = MachiningPhysics("steel")
        params = MachiningParameters(
            feed_rate=500,
            spindle_speed=2000,  # High speed = more wear
            depth_of_cut=2,
            width_of_cut=5,
            tool_radius=5
        )
        
        wear = physics.calculate_tool_wear(params, cutting_time_minutes=60)
        
        assert wear >= 0
        assert wear <= 100
        assert wear > 0  # Should have some wear
    
    def test_surface_roughness(self):
        """Test surface roughness calculation"""
        physics = MachiningPhysics("steel")
        
        # Fine feed
        fine_params = MachiningParameters(
            feed_rate=200,  # mm/min
            spindle_speed=2000,  # RPM
            depth_of_cut=1,
            width_of_cut=5,
            tool_radius=5
        )
        
        # Coarse feed
        coarse_params = MachiningParameters(
            feed_rate=800,  # mm/min
            spindle_speed=2000,
            depth_of_cut=1,
            width_of_cut=5,
            tool_radius=5
        )
        
        fine_roughness = physics.calculate_surface_roughness(fine_params)
        coarse_roughness = physics.calculate_surface_roughness(coarse_params)
        
        # Coarse feed should give worse (higher) roughness
        assert coarse_roughness > fine_roughness


class TestMachiningParameters:
    """Test MachiningParameters dataclass"""
    
    def test_derived_parameters(self):
        """Test that derived parameters are calculated"""
        params = MachiningParameters(
            feed_rate=600,  # mm/min
            spindle_speed=1000,  # RPM
            depth_of_cut=2,
            width_of_cut=5,
            tool_radius=5
        )
        
        # Feed per tooth = 600 / (1000 * 4) = 0.15
        assert params.feed_per_tooth == pytest.approx(0.15, abs=0.01)
        
        # Cutting speed = 2 * pi * 5 * 1000 / 1000 = 31.4 m/min
        assert params.cutting_speed == pytest.approx(31.4, abs=0.5)
        
        # MRR = 5 * 2 * 600 = 6000 mm³/min
        assert params.material_removal_rate == 6000


class TestProcessSimulator:
    """Test process simulator"""
    
    def test_simulator_initialization(self):
        """Test simulator setup"""
        simulator = ProcessSimulator(stock_dims=[100, 100, 50])
        
        assert simulator.kernel is not None
        assert simulator.physics is not None
        assert simulator.total_time == 0
    
    def test_simulate_single_linear_move(self):
        """Test simulating single G1 move"""
        simulator = ProcessSimulator(stock_dims=[100, 100, 50])
        
        operations = [
            {
                'g_code': 'G1',
                'start': np.array([0, 0, 0]),
                'end': np.array([50, 0, 0]),
                'feed_rate': 500,
                'spindle_speed': 1000,
                'is_rapid': False
            }
        ]
        
        tool = ToolProfile(id="test_tool", radius=5, type="FLAT")
        metrics = simulator.simulate_operations(operations, tool)
        
        assert metrics.total_time_seconds > 0
        assert metrics.total_distance_mm == 50
        assert metrics.cutting_time_seconds > 0
        assert metrics.avg_cutting_force_n > 0
    
    def test_simulate_rapid_vs_cutting(self):
        """Test that rapid moves are faster than cutting"""
        simulator = ProcessSimulator(stock_dims=[100, 100, 50])
        
        operations = [
            {
                'g_code': 'G0',  # Rapid
                'start': np.array([0, 0, 10]),
                'end': np.array([50, 0, 10]),
                'feed_rate': 500,
                'spindle_speed': 1000,
                'is_rapid': True
            },
            {
                'g_code': 'G1',  # Cutting
                'start': np.array([50, 0, 10]),
                'end': np.array([100, 0, 10]),
                'feed_rate': 500,
                'spindle_speed': 1000,
                'is_rapid': False
            }
        ]
        
        tool = ToolProfile(id="test_tool", radius=5, type="FLAT")
        metrics = simulator.simulate_operations(operations, tool)
        
        # Rapid distance = 50mm at 10000 mm/min → 0.3 seconds
        # Cutting distance = 50mm at 500 mm/min → 6 seconds
        assert metrics.rapid_time_seconds < metrics.cutting_time_seconds
        assert metrics.rapid_distance_mm == 50
        assert metrics.cutting_distance_mm == 50
    
    def test_simulate_gcode_file(self):
        """Test simulating G-code from file"""
        # Create temp G-code file
        gcode = """G21
G90
G0 X0 Y0 Z10
G1 Z-1 F100
G1 X50 F500
G0 Z10
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.nc', delete=False) as f:
            f.write(gcode)
            temp_path = f.name
        
        try:
            metrics = simulate_machining_process(
                gcode_file=temp_path,
                stock_dims=[100, 100, 50],
                material="steel"
            )
            
            assert isinstance(metrics, ProcessMetrics)
            assert metrics.total_time_seconds > 0
            assert metrics.total_distance_mm > 0
            
        finally:
            os.unlink(temp_path)
    
    def test_different_materials(self):
        """Test that different materials produce different results"""
        operations = [
            {
                'g_code': 'G1',
                'start': np.array([0, 0, 0]),
                'end': np.array([50, 0, 0]),
                'feed_rate': 500,
                'spindle_speed': 1000,
                'is_rapid': False
            }
        ]
        
        tool = ToolProfile(id="test_tool", radius=5, type="FLAT")
        
        simulator_steel = ProcessSimulator(stock_dims=[100, 100, 50], material="steel")
        simulator_alum = ProcessSimulator(stock_dims=[100, 100, 50], material="aluminum")
        
        metrics_steel = simulator_steel.simulate_operations(operations, tool)
        metrics_alum = simulator_alum.simulate_operations(operations, tool)
        
        # Steel should have higher forces
        assert metrics_steel.avg_cutting_force_n > metrics_alum.avg_cutting_force_n
    
    def test_toolpath_visualization(self):
        """Test visualization data generation"""
        simulator = ProcessSimulator(stock_dims=[100, 100, 50])
        
        operations = [
            {
                'g_code': 'G1',
                'start': np.array([0, 0, 0]),
                'end': np.array([50, 0, 0]),
                'feed_rate': 500,
                'spindle_speed': 1000,
                'is_rapid': False
            }
        ]
        
        tool = ToolProfile(id="test_tool", radius=5, type="FLAT")
        simulator.simulate_operations(operations, tool)
        
        viz = simulator.get_toolpath_visualization()
        
        assert 'instructions' in viz
        assert 'total_time' in viz
        assert 'stock_dims' in viz
        assert len(viz['instructions']) > 0


class TestIntegration:
    """Integration tests with VMK kernel"""
    
    def test_kernel_geometry_tracking(self):
        """Test that VMK tracks geometry correctly during simulation"""
        simulator = ProcessSimulator(stock_dims=[100, 100, 50])
        
        # Create a pocketing operation (tool moves through material)
        operations = []
        for y in [10, 20, 30]:
            operations.append({
                'g_code': 'G1',
                'start': np.array([10, y, -5]),
                'end': np.array([40, y, -5]),
                'feed_rate': 500,
                'spindle_speed': 1000,
                'is_rapid': False
            })
        
        tool = ToolProfile(id="endmill_10", radius=5, type="FLAT")
        simulator.simulate_operations(operations, tool)
        
        # Check VMK state
        state = simulator.kernel.get_state()
        assert len(state['history']) == 3
        
        # Check SDF at a point that should have been cut by the middle operation
        # Tool has 5mm radius, path at y=20, z=-5 should cut a channel
        # After cutting, material at this point should be REMOVED
        test_point = np.array([25, 20, -5])  # Center of tool path
        sdf = simulator.kernel.get_sdf(test_point)
        
        # After cutting: SDF > 0 means material was removed (outside remaining stock)
        # Before cutting: SDF would be negative (inside stock)
        # The key test is that the SDF changed due to the operation
        assert sdf > 0, f"SDF at cut point should be positive (material removed), got {sdf}"
        
        # Check a point away from cuts should still be inside
        uncut_point = np.array([25, 45, -5])  # y=45, well away from cuts at y=10,20,30
        uncut_sdf = simulator.kernel.get_sdf(uncut_point)
        # SDF < 0 means inside material, 0 means on boundary
        assert uncut_sdf < -0.1, f"SDF at uncut point should be negative (material present), got {uncut_sdf}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
