import sys
import os
import unittest
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.electronics_agent import ElectronicsAgent

class TestUniversalElectronics(unittest.TestCase):
    
    def test_mega_scale_arcing(self):
        print("\n--- Testing MEGA Scale (High Voltage Arcing) ---")
        agent = ElectronicsAgent()
        
        components = [
            {"name": "Grid_Substation_1", "category": "source", "power_peak_w": 1e6, "voltage_v": 115000},
            {"name": "City_Block_A", "category": "load", "power_peak_w": 0.8e6}
        ]
        
        params = {
            "scale": "MEGA",
            "resolved_components": components
        }
        
        res = agent.run(params)
        
        # Should flag arcing risk for 115kV component without explicit insulation
        issues = res["validation_issues"]
        print("Issues:", issues)
        arcing_found = any("ARCING_RISK" in i for i in issues)
        self.assertTrue(arcing_found, "Should detect High Voltage Arcing risk in MEGA scale")

    def test_nano_scale_tunneling(self):
        print("\n--- Testing NANO Scale (Quantum Tunneling) ---")
        agent = ElectronicsAgent()
        
        components = [
            {"name": "Nanobattery", "category": "source", "power_peak_w": 1e-8, "insulated": True},
            {"name": "Transistor_Gate", "category": "load", "power_peak_w": 1e-9, "feature_size_nm": 3.0, "insulated": True}
        ]
        
        params = {
            "scale": "NANO",
            "resolved_components": components,
            "chassis_material": "Silicon" 
        }
        
        res = agent.run(params)
        
        issues = res["validation_issues"]
        print("Issues:", issues)
        tunneling_found = any("QUANTUM_TUNNELING" in i for i in issues)
        self.assertTrue(tunneling_found, "Should detect Quantum Tunneling risk (<5nm) in NANO scale")

    def test_meso_power_balance(self):
        print("\n--- Testing MESO Scale (Standard Power Balance) ---")
        agent = ElectronicsAgent()
        
        components = [
            {"name": "Battery_Pack", "category": "battery", "power_peak_w": 500.0, "capacity_wh": 50.0, "insulated": True},
            {"name": "Motor_Array", "category": "load", "power_peak_w": 400.0, "insulated": True}
        ]
        
        params = {
            "scale": "MESO",
            "resolved_components": components,
            "chassis_material": "Carbon Fiber" # Conductive but we set insulated=True
        }
        
        res = agent.run(params)
        
        self.assertEqual(res["status"], "success")
        self.assertGreater(res["power_analysis"]["margin_w"], 0)
        print("Power Analysis:", res["power_analysis"])

if __name__ == "__main__":
    unittest.main()
