import unittest
from typing import List, Dict
import logging
import sys
import os

# Fix path to allow importing from backend root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.environment_agent import EnvironmentAgent
from agents.von_neumann_agent import VonNeumannAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ReplicationTest")

class TestReplication(unittest.TestCase):

    def test_population_bloom(self):
        """
        Simulate a Von Neumann Probe spreading in a resource-rich environment.
        """
        print("\n--- Starting Swarm Simulation ---")
        
        # 1. Setup Environment
        env_agent = EnvironmentAgent()
        resources = env_agent.init_swarm_resources(width=100.0, height=100.0, count=20) # Plenty of resources
        pheromones = {}
        
        # 2. Setup Seed Agent
        # Lower threshold to ensure replication happens in test duration
        seed_genetics = {"replication_threshold": 210.0} 
        seed = VonNeumannAgent(initial_energy=200.0, genetics=seed_genetics)
        agents: List[VonNeumannAgent] = [seed]
        
        # 3. Simulation Loop
        ticks = 100
        population_history = []
        
        for t in range(ticks):
            current_pop = len(agents)
            population_history.append(current_pop)
            
            # Snapshot environment state for agents
            env_state = {
                "resources": resources,
                "pheromones": pheromones
            }
            
            new_babies = []
            dead_ids = []
            
            # Agent Decisions
            for agent in agents:
                res = agent.run(env_state)
                
                # Handle Harvest
                if req := res.get("harvest_request"):
                    # Find pile
                    harvested = env_agent.consume_resource(resources, agent.pos, req["amount"])
                    if harvested > 0:
                        agent.energy += harvested # Gain Energy
                        # Stigmergy: Deposit Pheromone
                        pheromones[req["target_id"]] = pheromones.get(req["target_id"], 0.0) + 1.0
                
                # Handle Replication
                if child_config := res.get("child"):
                    # Instantiate Child
                    baby = VonNeumannAgent(
                        genetics=child_config["genetics"],
                        initial_energy=child_config["energy_grant"]
                    )
                    baby.pos = list(agent.pos) # Start at parent location
                    new_babies.append(baby)
                
                # Handle Death
                if res["status"] == "dead":
                    dead_ids.append(agent.id)
            
            # Update Population
            agents = [a for a in agents if a.id not in dead_ids] + new_babies
            
            # Update Environment (Entropy)
            pheromones = env_agent.update_pheromones(pheromones)
            
            # Debug Print (every 10 ticks)
            if t % 10 == 0:
                print(f"Tick {t}: Pop {current_pop} | Resources {len([r for r in resources if r['amount'] > 1.0])} | Pheromones {len(pheromones)}")
            
            if len(agents) == 0:
                print("Extinction Event!")
                break
                
            if len(agents) > 20: 
                print("Carrying Capacity Reached (Simulated Stop).")
                break

        # 4. Assertions
        print(f"Final Population: {len(agents)}")
        self.assertGreater(len(agents), 1, "Population failed to grow from seed.")
        
        # Check Lineage
        child = agents[-1]
        print(f"Example Genome (Gen {child.genetics.generation}): {child.genetics.dict()}")
        self.assertGreaterEqual(child.genetics.generation, 0, "Generation counter broken.")
        
if __name__ == '__main__':
    unittest.main()
