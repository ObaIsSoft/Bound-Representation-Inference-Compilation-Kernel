import sys
import os
import logging
import random

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agents.optimization_agent import OptimizationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_robust_verification():
    print("=== Robust Evolution (Survival of the Fittest) ===")
    
    agent = OptimizationAgent()
    
    # Configuration
    # We want a decent number of generations to allow adaptation
    generations = 15
    pop_size = 20
    
    params = {
        "isa_state": {
            "constraints": {
                "max_weight": {"val": 100.0}, # Moderate load
                "material_strength": {"val": 1.0} 
            }
        },
        "config": {
            "population_size": pop_size,
            "generations": generations,
            "enable_red_team": True, # ACTIVATE THE ADVERSARY
            "available_assets": ["REINFORCED_JOINT", "STEEL_BEAM"]
        },
        "objective": {
            "type": "SPREAD" # Try to grow outwards
        }
    }
    
    print(f"Goal: Evolve a wide structure that survives Red Team attacks.")
    
    try:
        result = agent.run(params)
        
        if result['success']:
            print("\n[SUCCESS] Robust Evolution Completed.")
            best_genome = result['best_genome']
            fitness = result['fitness_score']
            
            print(f"Final Fitness: {fitness}")
            print(f"Strategy: {result['strategy_used']}")
            
            # Post-Run Stress Test verification
            # Let's see if the winner is actually robust
            from backend.agents.surrogate.pinn_model import MultiPhysicsPINN
            from backend.agents.critics.adversarial import RedTeamAgent
            
            # Reconstruct genome
            from backend.agents.evolution import GeometryGenome
            genome_obj = GeometryGenome.from_json(best_genome)
            nodes_data = [attr['data'].dict() for attr in genome_obj.graph.nodes.values()]
            
            pinn = MultiPhysicsPINN()
            red_team = RedTeamAgent(pinn)
            
            print("\n--- Final Stress Audit ---")
            audit = red_team.stress_test(nodes_data, params["isa_state"]["constraints"], trials=100)
            print(f"Failure Rate: {audit['failure_rate']*100:.1f}%")
            print(f"Robust: {audit['is_robust']}")
            
            if audit['is_robust']:
                print("Result: Evolution successfully adapted to the pressure!")
            else:
                print("Result: Design still failed audit. (Maybe needs more generations?)")
                
        else:
            print("[FAILED] Agent returned success=False")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_robust_verification()
