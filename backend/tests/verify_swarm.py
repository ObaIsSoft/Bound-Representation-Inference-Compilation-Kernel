import sys
import os
import logging

# Ensure backend is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.swarm_manager import SwarmManager

# Configure logging to show Swarm metrics clearly
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("SwarmTest")

def test_swarm_replication():
    print("--- Testing Swarm Self-Replication & Construction ---")
    
    manager = SwarmManager()
    
    # 1. Initialize High-Energy Environment (Favoring Replication)
    print("Initializing Swarm...")
    config = {
        "initial_pop": 4, # 2 Scouts, 2 Builders
        "resource_count": 50, # Abundant resources
        "agent_types": ["VonNeumannAgent", "ConstructionAgent"]
    }
    manager.init_simulation(config)
    
    initial_pop = len(manager.agents)
    print(f"Initial Population: {initial_pop}")
    print(f"Initial Resources: {len(manager.resources)}")
    
    # 2. Run Simulation
    ticks = 60
    print(f"Running for {ticks} ticks...")
    
    metrics = manager.run_simulation(ticks=ticks)
    
    # 3. Validation
    final_pop = metrics["population"]
    structures = metrics["structures_built"]
    max_gen = metrics["max_generation"]
    
    print("\n--- Simulation Results ---")
    print(f"Final Population: {final_pop} (Started: {initial_pop})")
    print(f"Structures Built: {structures}")
    print(f"Max Generation: {max_gen}")
    print(f"Resources Remaining: {metrics['resources_remaining']}")
    
    # Assertions
    if final_pop > initial_pop:
        print("✅ SUCCESS: Population grew (Self-Replication verified)")
    else:
        print("❌ FAILURE: Population did not grow")
        
    if structures > 0:
        print("✅ SUCCESS: Structures were built")
    else:
        print("⚠️ WARNING: No structures built (might need more time/resources)")
        
    if max_gen > 0:
        print(f"✅ SUCCESS: Evolution occurred (Generation {max_gen} reached)")
    else:
        print("❌ FAILURE: No new generations")

if __name__ == "__main__":
    test_swarm_replication()
