
import unittest
import json
from agents.manufacturing_agent import ManufacturingAgent
from agents.physics_agent import PhysicsAgent

class TestCritique(unittest.TestCase):
    def setUp(self):
        self.man_agent = ManufacturingAgent()
        self.phys_agent = PhysicsAgent()
        
    def test_manufacturing_critique_radius(self):
        # 0.5mm radius is < 1mm limit
        sketch = [
            {"x": 0, "y": 0, "radius": 0.0005}, 
            {"x": 1, "y": 1, "radius": 0.0005}
        ]
        
        critiques = self.man_agent.critique_sketch(sketch)
        
        self.assertTrue(len(critiques) > 0)
        self.assertEqual(critiques[0]["level"], "WARN")
        self.assertIn("too small", critiques[0]["message"])
        
    def test_physics_critique_mass(self):
        # Near zero mass
        geometry = [
            {"id": "test", "params": {"width": 10, "height": 10}, "mass_kg": 0.00001}
        ]
        
        critiques = self.phys_agent.critique_design(geometry)
        
        self.assertTrue(len(critiques) > 0)
        self.assertEqual(critiques[0]["agent"], "Physics")
        self.assertIn("near-zero mass", critiques[0]["message"])

if __name__ == '__main__':
    unittest.main()
