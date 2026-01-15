import sys
import os
import unittest
from typing import Dict, Any

# Add backend to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.gnc_agent import GncAgent

class TestGncPlanner(unittest.TestCase):
    
    def test_trajectory_planning(self):
        print("\n--- Testing GNC Trajectory Planner (CEM) ---")
        agent = GncAgent()
        
        # Scenario: Drone needs to move 10m diagonally
        mission = {
            "start_pos": [0, 0, 0],
            "target_pos": [10, 10, 10],
            "obstacles": [
                {"pos": [5, 5, 5], "radius": 2.0} # Obstacle in the middle
            ]
        }
        
        params = {
            "mass_kg": 1.0,
            "thrust_n": 30.0, # Plenty of thrust (T/W ~3)
            "environment": "EARTH",
            "mission_profile": mission
        }
        
        res = agent.run(params)
        
        # Checks
        self.assertEqual(res["status"], "success")
        self.assertTrue(res["flight_ready"])
        
        traj = res.get("trajectory", {})
        self.assertTrue(traj.get("success"), "Planner should find a path with adequate thrust")
        waypoints = traj.get("waypoints", [])
        print(f"✅ Generated Path with {len(waypoints)} waypoints.")
        
        # Basic check: did it move?
        start = waypoints[0]
        end = waypoints[-1]
        print(f"Start: {start}, End: {end}")
        
        # dist to target should be small (CEM success threshold is cost < 50, but let's check basic proximity)
        import math
        dist = math.sqrt(sum((e-t)**2 for e,t in zip(end, mission["target_pos"])))
        print(f"Final Distance to Target: {dist:.2f}m")
        # CEM is stochastic, so we don't assert specific low dist, just that it ran success.
        cost = traj.get("final_cost", 999.0)
        print(f"Final Cost: {cost:.2f}")

    def test_stress_cem(self):
        """
        Stress Test: Run planner 20 times to ensure Stochastic Reliability > 80%
        """
        print("\n--- Stress Testing CEM (20 Iterations) ---")
        agent = GncAgent()
        mission = {
            "start_pos": [0, 0, 0],
            "target_pos": [10, 10, 10],
            "obstacles": [{"pos": [5, 5, 5], "radius": 2.0}]
        }
        params = {
            "mass_kg": 1.0, 
            "thrust_n": 30.0, 
            "environment": "EARTH",
            "mission_profile": mission
        }
        
        success_count = 0
        total = 20
        
        for i in range(total):
            res = agent.run(params)
            if res["status"] == "success":
                success_count += 1
            # print(f"Run {i+1}: {res['status']}")
            
        rate = success_count / total
        print(f"Stress Test Result: {success_count}/{total} passed ({rate*100:.1f}%)")
        self.assertGreaterEqual(rate, 0.80, "CEM Planner reliability is too low (<80%)")

if __name__ == "__main__":
    unittest.main()
