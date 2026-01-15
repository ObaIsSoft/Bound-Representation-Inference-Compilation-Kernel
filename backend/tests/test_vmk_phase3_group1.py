
import sys
import os
print("DEBUG: Starting Phase 3 Script")
sys.path.append(os.path.join(os.path.dirname(__file__), '../'))

from agents.von_neumann_agent import VonNeumannAgent
from agents.swarm_manager import SwarmManager
from agents.replicator_mixin import AgentGenetics

def test_phase3_group1_integration():
    print("--- VMK PHASE 3 (GROUP 1) INTEGRATION TEST ---")
    
    # 1. Von Neumann Agent (Replication Fidelity)
    print("\n[1] Testing VonNeumannAgent.verify_replication_fidelity...")
    
    # Setup Parent with High Energy
    parent = VonNeumannAgent(initial_energy=1000.0)
    # Set genetics to cause large radius
    parent.genetics.harvest_efficiency = 1.0 # Radius = 10.0
    parent.genetics.replication_threshold = 500.0
    
    # Run Agent Tick -> Should Trigger Reproduction
    state = parent.run({"resources": [], "pheromones": {}})
    
    logs = state.get("logs", [])
    print("Agent Logs:")
    for l in logs:
        print(f"  - {l}")
        
    # Check for "Gen " or "ABORTED"
    # Note: Mutation is random. It might pass or fail depending on tolerance.
    # We set tolerance=5.0 in Agent logic. Mutation rate 0.05.
    # Drift = 1 +/- 0.05. Variation ~0.5 units. Should PASS.
    
    if any("replicated" in l for l in logs) or any("ABORTED" in l for l in logs):
        print("Replication Logic & VMK Verification Triggered")
    else:
        print("FAILURE: No replication attempt detected.")
        
    # 2. Swarm Manager (Collision Check)
    print("\n[2] Testing SwarmManager.check_swarm_collisions...")
    swarm = SwarmManager()
    
    # Initialize normally
    swarm.init_simulation({"initial_pop": 0, "task_count": 0})
    
    # Add 2 overlapping agents
    # Agent Size ~ 10.0 radius (Eff=1.0 * 5 * 2? Wait, Agent logic uses Eff * 5 as radius?)
    # SwarmManager code: r = eff * 5.0. 
    # VonNeumann code: r = eff * 10.0 (in verify_replication_fidelity).
    # Inconsistent? SwarmManager used 5.0. Let's assume Radius=5.0.
    
    a1 = VonNeumannAgent(initial_energy=100)
    a1.pos = [0.0, 0.0]
    a1.genetics.harvest_efficiency = 1.0 # r=5
    
    a2 = VonNeumannAgent(initial_energy=100)
    a2.pos = [5.0, 0.0] # Distance 5.0. R1+R2 = 10.0. Overlap!
    a2.genetics.harvest_efficiency = 1.0 # r=5
    
    swarm.agents = [a1, a2]
    
    # Run Tick
    swarm.run_tick()
    
    # Check logs/stdout for warnings
    # SwarmManager logs warnings using `logging`.
    # We can check the return or if method works.
    # We will call check_swarm_collisions directly to see output
    
    collisions = swarm.check_swarm_collisions(swarm.agents)
    print(f"Collisions Detected: {len(collisions)}")
    for c in collisions:
        print(f"  - {c}")
        
    if len(collisions) > 0:
        print("Collision Detection Logic: SUCCESS")
    else:
        print("Collision Detection Logic: FAILURE")

if __name__ == "__main__":
    test_phase3_group1_integration()
