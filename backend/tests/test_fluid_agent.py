
import unittest
from agents.fluid_agent import FluidAgent

class TestFluidAgent(unittest.TestCase):
    def setUp(self):
        self.agent = FluidAgent()
        
    def test_potential_flow_drag(self):
        # Test Case: Simple Box
        geometry = [{"type": "box", "params": {"width": 1, "height": 1, "length": 1}}]
        context = {"velocity": 10.0, "density": 1.225}
        
        result = self.agent.run(geometry, context)
        
        # Expected:
        # Area = 1*1 = 1m2
        # Cd ~ 1.05 (Cube)
        # Drag = 0.5 * 1.225 * 100 * 1.05 * 1 = 64.3 N
        
        self.assertAlmostEqual(result["frontal_area_m2"], 1.0)
        self.assertGreater(result["drag_n"], 50.0)
        self.assertLess(result["drag_n"], 80.0)
        self.assertEqual(result["solver"], "Potential Flow (Heuristic)")

    def test_streamlined_body(self):
        # Long cylinder
        geometry = [{"type": "cylinder", "params": {"radius": 0.5, "length": 10.0}}]
        # Width=1, Length=10 -> Aspect=10 -> Low Cd
        
        context = {"velocity": 10.0}
        result = self.agent.run(geometry, context)
        
        self.assertLess(result["cd"], 0.5)

if __name__ == '__main__':
    unittest.main()
