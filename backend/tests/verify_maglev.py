import sys
import os
import unittest
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.electronics_agent import ElectronicsAgent

class TestMaglevPhysics(unittest.TestCase):
    
    def setUp(self):
        self.agent = ElectronicsAgent()
        
    def test_lorentz_force_calculation(self):
        print("\n--- Testing Lorentz Force Calculation (F = ILB) ---")
        
        # Setup: 1T Field (Vertical Z) - effectively simplified magnitude
        # We pass vector [0, 0, 1] -> B=1.0 Tesla
        b_vec = [0.0, 0.0, 1.0]
        
        # Case 1: Standard Coil
        # I=10A, L=10m. F should be 100N
        comps = [{
            "name": "Maglev_Coil_1",
            "category": "coil",
            "current_a": 10.0,
            "wire_length_m": 10.0
        }]
        
        force = self.agent._calculate_lorentz_force(comps, b_vec)
        print(f"Force (1T, 10A, 10m): {force}N")
        self.assertAlmostEqual(force, 100.0)
        print("✅ Correct force for single coil.")
        
        # Case 2: Zero Field
        force_zero = self.agent._calculate_lorentz_force(comps, [0,0,0])
        print(f"Force (0T): {force_zero}N")
        self.assertEqual(force_zero, 0.0)
        print("✅ Zero force in zero field.")
        
        # Case 3: Non-Coil Component (Should be ignored)
        comps_mixed = [
            {
                "name": "Maglev_Coil_1",
                "category": "coil",
                "current_a": 10.0,
                "wire_length_m": 10.0
            },
            {
                "name": "Resistor",
                "category": "passive", # Not a coil
                "current_a": 5.0,
                "wire_length_m": 100.0
            }
        ]
        
        force_mixed = self.agent._calculate_lorentz_force(comps_mixed, b_vec)
        print(f"Force (Mixed): {force_mixed}N")
        self.assertAlmostEqual(force_mixed, 100.0) # Resistor should produce 0 propulsive force
        print("✅ Ignored non-propulsive components.")
        
    def test_run_integration(self):
        print("\n--- Testing Agent Run Integration ---")
        
        # Mock params with B-field
        params = {
            "components": [], # Will resolve to default empty or mocked active_components_data below if we could inject.
            # Since run() rebuilds active_components from component_ids, we can't easily mock valid components without DB.
            # However, we can inspect if 'mag_lift_n' key is present in output.
            "magnetic_field_vec_T": [0, 0.5, 0]
        }
        
        # We rely on the fact that default components (motor_2207) are NOT coils
        # So force should be 0, but key should exist.
        result = self.agent.run(params)
        
        self.assertIn("mag_lift_n", result)
        print(f"Result contains 'mag_lift_n': {result['mag_lift_n']}")
        
        # We can't easily force it to be >0 without adding a 'coil' to the SQLite DB or mocking the DB.
        # But verifying the pipeline key existence verifies the integration code path.
        print("✅ Integration successful (Key present).")

if __name__ == "__main__":
    unittest.main()
