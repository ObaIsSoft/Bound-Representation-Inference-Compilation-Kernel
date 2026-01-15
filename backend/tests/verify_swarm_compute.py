import sys
import os
import time
import logging

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from agents.swarm_manager import SwarmManager

def test_swarm_compute():
    print("--- Testing Distributed Compute Swarm ---")
    
    manager = SwarmManager()
    
    # 1. Initialize Compute Swarm
    # 10 Agents, 100 Tasks
    print("Initializing Compute Swarm...")
    config = {
        "initial_pop": 5, 
        "resource_count": 50,
        "agent_types": ["VonNeumannAgent"],
        "task_count": 100 # New config
    }
    manager.init_simulation(config)
    
    print(f"Agents: {len(manager.agents)}")
    print(f"Tasks Pending: {len(manager.task_queue)}")
    
    # 2. Run
    start_time = time.time()
    ticks = 0
    while len(manager.task_queue) > 0 and ticks < 200:
        manager.run_tick()
        ticks += 1
        pct = 100 - len(manager.task_queue)
        # print(f"Tick {ticks}: {len(manager.agents)} Agents | {len(manager.task_queue)} Tasks Remaining")

    duration = time.time() - start_time
    
    print("\n--- Results ---")
    print(f"Time Taken: {duration:.3f}s ({ticks} ticks)")
    print(f"Final Population: {len(manager.agents)}")
    print(f"Tasks Remaining: {len(manager.task_queue)}")
    
    if len(manager.task_queue) == 0:
        print("✅ SUCCESS: All tasks processed.")
    else:
        print("❌ FAILURE: Tasks unfinished.")
        
    # Efficiency Check:
    # 100 tasks / 5 agents = 20 ticks (ideal)
    # But agents need to harvest too.
    print(f"Throughput: {100/ticks:.2f} tasks/tick")

if __name__ == "__main__":
    test_swarm_compute()
