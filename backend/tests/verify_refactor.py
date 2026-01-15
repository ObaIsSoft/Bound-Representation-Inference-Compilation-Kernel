import sys
import os
import unittest
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.designer_agent import DesignerAgent
from agents.cost_agent import CostAgent
from agents.codegen_agent import CodegenAgent
from agents.electronics_agent import ElectronicsAgent
from agents.geometry_agent import GeometryAgent

class TestRefactor(unittest.TestCase):
    
    def test_designer_db(self):
        print("\n--- Testing DesignerAgent DB ---")
        agent = DesignerAgent()
        res = agent.run({"style": "cyberpunk"})
        self.assertEqual(res["aesthetics"]["primary"], "#18181b")
        print("✅ DesignerAgent fetched 'cyberpunk' palette from DB.")

    def test_cost_db(self):
        print("\n--- Testing CostAgent DB ---")
        agent = CostAgent()
        res = agent.run({"material_name": "Aluminum 6061", "manufacturing_process": "cnc_milling"})
        self.assertIn("total_cost_usd", res)
        print("✅ CostAgent ran without error (DB/Fallback).")

    def test_codegen_db(self):
        print("\n--- Testing CodegenAgent DB ---")
        agent = CodegenAgent()
        comps = [{"name": "MyServo", "category": "servo", "id": "s1"}]
        res = agent.run({"resolved_components": comps})
        code = res["firmware_source_code"]
        self.assertIn("<Servo.h>", code)
        self.assertIn("Servo MYSERVO_0;", code)
        print("✅ CodegenAgent fetched 'Servo' library from DB.")

    def test_electronics_db(self):
        print("\n--- Testing ElectronicsAgent DB (Shorts) ---")
        agent = ElectronicsAgent()
        issues = agent._check_chassis_shorts([{"name": "Part", "mount": "chassis"}], "Aluminum")
        self.assertTrue(len(issues) > 0)
        print("✅ ElectronicsAgent used DB standards to detect short on Aluminum.")

    def test_geometry_db(self):
        print("\n--- Testing GeometryAgent DB (KCL) ---")
        agent = GeometryAgent()
        kcl = agent._append_component_placeholders("// base", [])
        self.assertIn("render_nema_stepper", kcl)
        print("✅ GeometryAgent fetched 'render_nema_stepper' from KCL Templates DB.")

if __name__ == "__main__":
    unittest.main()
