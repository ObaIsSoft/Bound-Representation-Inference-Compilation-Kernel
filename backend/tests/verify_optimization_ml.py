import sys
import os
import unittest
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.optimization_agent import OptimizationAgent

class TestMLOptimization(unittest.TestCase):
    
    def setUp(self):
        self.agent = OptimizationAgent()
        
    def test_bayesian_suggestion(self):
        print("\n--- Testing Bayesian Optimization Agent ---")
        
        # Initial Params
        params = {
            "radius": 10.0,
            "thickness": 2.0,
            "material": "Aluminum" # Non-numeric, should be ignored
        }
        
        flags = {"physics_safe": False}
        reasons = ["Stress too high"]
        
        # Run Agent
        result = self.agent.run(params, flags, reasons)
        
        print(f"Logs: {result['logs']}")
        new_params = result["new_parameters"]
        print(f"New Params: {new_params}")
        
        # Checks
        self.assertEqual(result["status"], "optimized")
        
        # 1. Check Numeric Update
        self.assertNotEqual(new_params["radius"], 10.0)
        self.assertNotEqual(new_params["thickness"], 2.0)
        
        # 2. Check Bounds (+/- 50%)
        self.assertTrue(5.0 <= new_params["radius"] <= 15.0)
        self.assertTrue(1.0 <= new_params["thickness"] <= 3.0)
        
        # 3. Check Non-Numeric Preservation
        self.assertEqual(new_params["material"], "Aluminum")
        
        print("✅ PASS: Optimizer suggested valid new parameters.")

if __name__ == "__main__":
    unittest.main()
