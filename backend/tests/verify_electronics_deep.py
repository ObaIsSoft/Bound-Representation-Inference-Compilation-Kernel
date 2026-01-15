import sys
import os
import unittest
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.electronics_agent import ElectronicsAgent

class TestDeepElectronics(unittest.TestCase):
    
    def setUp(self):
        self.agent = ElectronicsAgent()
        
        # Mock Components
        self.comp_short = {
            "id": "mosfet_to220",
            "name": "Power MOSFET",
            "category": "component",
            "insulated": False,
            "mount": "chassis"
        }
        
        self.comp_insulated = {
            "id": "mosfet_iso",
            "name": "Isolated MOSFET",
            "category": "component",
            "insulated": True,
            "mount": "chassis"
        }
        
    def test_chassis_short_detection(self):
        print("\n--- Testing Chassis Short Circuit Detection ---")
        
        # Case 1: Conductive Chassis (Aluminum) + Uninsulated -> SHORT
        print("Test 1: Aluminum Chassis + Uninsulated Device")
        params = {
            "components": [], # We will mock resolved_components injection or override
            # ElectronicsAgent.run() re-resolves from DB, so it's hard to inject mocks directly via params if we use run().
            # Luckily, _check_chassis_shorts is a public-ish helper we can test directly or via side-channel.
            # But let's test the helper method directly for unit testing.
        }
        
        # Direct Helper Test
        issues = self.agent._check_chassis_shorts([self.comp_short], "Aluminum")
        print(f"Issues found: {issues}")
        self.assertTrue(len(issues) > 0)
        self.assertIn("SHORT_CIRCUIT_RISK", issues[0])
        self.assertIn("Aluminum", issues[0])
        print("✅ Correctly identified short on Aluminum.")
        
        # Case 2: Conductive Chassis + Insulated -> SAFE
        print("Test 2: Aluminum Chassis + Insulated Device")
        issues = self.agent._check_chassis_shorts([self.comp_insulated], "Aluminum")
        print(f"Issues found: {len(issues)}")
        self.assertEqual(len(issues), 0)
        print("✅ Correctly cleared insulated device.")
        
        # Case 3: Non-Conductive Chassis (PLA) + Uninsulated -> SAFE
        print("Test 3: Plastic Chassis + Uninsulated Device")
        issues = self.agent._check_chassis_shorts([self.comp_short], "PLA Plastic")
        print(f"Issues found: {len(issues)}")
        self.assertEqual(len(issues), 0)
        print("✅ Correctly ignored plastic chassis.")

    def test_emi_detection(self):
        print("\n--- Testing EMI/EMC Interference Detection ---")
        
        # Setup: Motor (Noisy) and Mag (Sensitive)
        comps = [
            {"name": "Motor_FL", "category": "motor"},
            {"name": "Compass_Main", "category": "sensor"} # 'compass' string match
        ]
        
        # Case 1: Too Close (<50mm)
        # Motor at 0,0,0. Compass at 10,0,0 (10mm dist)
        geo_bad = [
            {"name": "Motor_FL", "position": [0,0,0]},
            {"name": "Compass_Main", "position": [10,0,0]}
        ]
        
        issues = self.agent._check_emi_compatibility(comps, geo_bad)
        print(f"Close Proximity Issues: {issues}")
        self.assertTrue(len(issues) > 0)
        self.assertIn("EMI_INTERFERENCE", issues[0])
        print("✅ Detected EMI Interference (10mm).")
        
        # Case 2: Safe Distance (>50mm)
        # Compass at 100,0,0 (100mm dist)
        geo_good = [
            {"name": "Motor_FL", "position": [0,0,0]},
            {"name": "Compass_Main", "position": [100,0,0]}
        ]
        
        issues = self.agent._check_emi_compatibility(comps, geo_good)
        print(f"Safe Distance Issues: {len(issues)}")
        self.assertEqual(len(issues), 0)
        print("✅ Confirmed Safe Separation (100mm).")

if __name__ == "__main__":
    unittest.main()
