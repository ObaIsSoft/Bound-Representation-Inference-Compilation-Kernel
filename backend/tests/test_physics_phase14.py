import unittest
import sys
import os
import math
import numpy as np

# Add backend to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from agents.physics_agent import PhysicsAgent, StructuralSolver

class TestPhysicsPhase14(unittest.TestCase):
    def setUp(self):
        self.agent = PhysicsAgent()
        self.structural = StructuralSolver()

    def test_free_fall(self):
        """Verify F=ma under gravity"""
        print("\n[Test] Free Fall (Gravity Only)")
        
        # Initial State: 100m alt, 0 vel
        state = {
            "altitude": 100.0,
            "velocity": 0.0,
            "position": {"x": 0, "y": 100, "z": 0},
            "velocity_vector": [0, 0, 0],
            "quaternion": [1, 0, 0, 0] # Upright
        }
        
        # Input: 0 Thrust, Standard Gravity, 1.0 Mass
        inputs = {
            "thrust": 0.0,
            "gravity": 9.81,
            "mass": 1.0,
            "drag_coeff": 0.0 # Remove drag for analytical check
        }
        
        dt = 1.0
        # Expected: d = 0.5 * g * t^2 = 0.5 * 9.81 * 1 = 4.905m drop
        # New Alt = 95.095
        # Vel = g*t = 9.81 m/s (down)
        
        res = self.agent.step(state, inputs, dt)
        new_state = res["state"]
        
        # Check Vert Position (y)
        # Note: Euler integration in previous step vs ODEint now. 
        # ODEint should be more accurate.
        print(f"  Initial Y: 100.0")
        print(f"  Final Y: {new_state['position']['y']:.4f}")
        print(f"  Final Vy: {new_state['velocity_vector'][1]:.4f}")
        
        self.assertTrue(new_state['position']['y'] < 100.0, "Object should fall")
        self.assertAlmostEqual(new_state['velocity_vector'][1], -9.81, delta=0.5, msg="Velocity should be approx -g*t")

    def test_inertia(self):
        """Verify Newton's First Law (Inertia)"""
        print("\n[Test] Inertia (Drift)")
        
        # Moving horizontally at 10m/s. No forces.
        state = {
            "position": {"x": 0, "y": 100, "z": 0},
            "velocity_vector": [10.0, 0.0, 0.0], # Moving X
            "quaternion": [1, 0, 0, 0]
        }
        
        inputs = {
            "thrust": 0.0,
            "gravity": 0.0, # Space
            "mass": 1.0,
            "drag_coeff": 0.0 # Vacuum
        }
        
        res = self.agent.step(state, inputs, dt=2.0)
        new_state = res["state"]
        
        print(f"  Final Vx: {new_state['velocity_vector'][0]:.4f}")
        
        self.assertAlmostEqual(new_state['velocity_vector'][0], 10.0, delta=0.1, msg="Velocity should persist (Inertia)")
        self.assertAlmostEqual(new_state['position']['x'], 20.0, delta=0.1, msg="Position should update to 20m")

    def test_structural_cantilever(self):
        """Verify Beam Bending calculation"""
        print("\n[Test] Structural Solver (Cantilever)")
        
        if not self.structural.ready:
            print("  Skipping: Structural Solver missing dependencies")
            return

        length = 1.0 # m
        area = 0.01 # 10cm x 10cm
        load = 1000.0 # N
        
        res = self.structural.solve_cantilever(length, area, load)
        
        print(f"  Deflection: {res['max_deflection_m']*1000:.2f} mm")
        print(f"  Stress: {res['max_stress_Pa']/1e6:.2f} MPa")
        print(f"  Safety Factor: {res['safety_factor_yield']:.2f}")
        
        self.assertTrue(res['max_deflection_m'] > 0)
        self.assertTrue(res['max_stress_Pa'] > 0)

if __name__ == '__main__':
    unittest.main()
