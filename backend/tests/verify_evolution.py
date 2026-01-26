import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agents.optimization_agent import OptimizationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_verification():
    print("=== Evolutionary Engine Verification ===")
    
    agent = OptimizationAgent()
    agent.generations = 5 # Short run for test
    agent.population_size = 10
    
    # Mock Input
    params = {
        "isa_state": {
            "constraints": {
                # Dummy constraints, currently ignored by seed logic but good to have
                "max_width": {"locked": True, "val": 10.0}
            }
        }
    }
    
    try:
        result = agent.run(params)
        
        if result['success']:
            print("\n[SUCCESS] Evolutionary Run Completed.")
            print(f"Generations: {len(result['evolution_history'])}")
            print(f"Best Fitness: {result['fitness_score']}")
            print(f"Topology JSON: {result['best_genome']}")
            
            # Basic validation
            nodes = result['best_genome']['nodes']
            print(f"Final Node Count: {len(nodes)}")
            if len(nodes) > 1:
                print("Topology evolved beyond root! (Mutation successful)")
            else:
                print("Warning: Topology is still just root.")
                
        else:
            print("[FAILED] Agent returned success=False")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_verification()
