import unittest
import logging
import sys
import os

# Fix path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.swarm_manager import SwarmManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SwarmTest")

class TestSwarmIntegration(unittest.TestCase):

    def test_heterogeneous_stigmergy(self):
        """
        Verify that VonNeumannAgents (Scouts) and ConstructionAgents (Builders)
        coordinate via Pheromones to build structures.
        """
        print("\n--- Starting Heterogeneous Swarm Test ---")
        
        manager = SwarmManager()
        config = {
            "initial_pop": 4, # 2 Scouts, 2 Builders
            "resource_count": 20,
            "agent_types": ["VonNeumannAgent", "ConstructionAgent"]
        }
        
        manager.init_simulation(config)
        
        # Run for sufficient ticks to allow finding > signaling > traveling > building
        metrics = manager.run_simulation(ticks=150)
        
        print(f"Final Metrics: {metrics}")
        
        # Assertions
        # 1. Structures Built (Proof of Builder Logic + Finding Ore)
        # Note: Builders need Ore (50) to build. Initial energy doesn't give ore.
        # They must harvest.
        # If Scouts found Ore and signaled, Builders should have found it faster.
        # We at least expect >0 structures if resources are plentiful.
        self.assertGreaterEqual(metrics["structures_built"], 1, "Swarm failed to build any structures.")
        
        # 2. Population Growth (Proof of Replication)
        self.assertGreater(metrics["population"], 4, "Swarm failed to reproduce.")
        
        # 3. Resource Depletion (Proof of consumption)
        self.assertLess(metrics["resources_remaining"], 20, "Swarm failed to consume resources.")

if __name__ == '__main__':
    unittest.main()
