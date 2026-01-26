import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agents.optimization_agent import OptimizationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_complex_verification():
    print("=== Complex Shape Evolution (Objective: SPREAD) ===")
    
    agent = OptimizationAgent()
    agent.generations = 10
    agent.population_size = 20
    
    # Complex Objective: Maximize Spread (Skeletal/Drone shape)
    params = {
        "isa_state": {
            "constraints": {
                "max_weight": {"locked": True, "val": 50.0}
            }
        },
        "objective": {
            "type": "SPREAD" 
        },
        "seed_config": {
            "type": "CYLINDER",
            "dimensions": {"radius": 0.5, "height": 5.0} 
        }
    }
    
    print(f"Goal: Evolve a structure that maximizes distance from origin (skeletal).")
    
    try:
        result = agent.run(params)
        
        if result['success']:
            print("\n[SUCCESS] Complex Run Completed.")
            print(f"Generations: {len(result['evolution_history'])}")
            print(f"Best Fitness: {result['fitness_score']}")
            
            nodes = result['best_genome']['nodes']
            print(f"Final Node Count: {len(nodes)}")
            
            # Analyze Spread
            max_dist = 0
            for n in nodes:
                 pos = n.get('transform', [0]*6)
                 dist = (pos[0]**2 + pos[1]**2 + pos[2]**2)**0.5
                 max_dist = max(max_dist, dist)
            print(f"Max Radius from Origin: {max_dist:.4f}")
            
            if max_dist > 2.0 and len(nodes) > 5:
                print("Result: Evolved complex spread-out topology!")
            else:
                print("Result: Still somewhat compact.")
                
        else:
            print("[FAILED] Agent returned success=False")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_complex_verification()
