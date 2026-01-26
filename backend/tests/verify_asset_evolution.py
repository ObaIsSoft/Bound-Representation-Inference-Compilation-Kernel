import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from backend.agents.optimization_agent import OptimizationAgent

# Configure logging
logging.basicConfig(level=logging.INFO)

def run_asset_verification():
    print("=== Asset Integration Verification ===")
    
    agent = OptimizationAgent()
    agent.generations = 10
    agent.population_size = 20
    
    # Mock Input with Assets
    assets_to_test = ["NEMA17_MOTOR", "LIPO_BATTERY_3S", "CARBON_ROD"]
    
    params = {
        "isa_state": {
            "constraints": {
                "max_weight": {"locked": True, "val": 50.0}
            }
        },
        "config": {
            "available_assets": assets_to_test,
             # Boost topology add rate to ensure we get some mutations
            "topology_add_rate": 0.5 
        }
    }
    
    print(f"Goal: Evolve a design using library assets: {assets_to_test}")
    
    try:
        result = agent.run(params)
        
        if result['success']:
            print("\n[SUCCESS] Asset Evolutionary Run Completed.")
            best_genome = result['best_genome']
            nodes = best_genome['nodes']
            print(f"Final Node Count: {len(nodes)}")
            
            # Count Assets
            asset_count = 0
            found_assets = []
            for n in nodes:
                 if n['type'] == 'LIBRARY_ASSET':
                     asset_count += 1
                     if n.get('asset_id'):
                         found_assets.append(n['asset_id'])
            
            print(f"Library Assets Found: {asset_count}")
            print(f"Assets Used: {set(found_assets)}")
            
            if asset_count > 0:
                print("Result: Success! Evolution incorporated library assets.")
            else:
                print("Result: Failed. No assets found in best design (might need more generations or higher prob).")
                
        else:
            print("[FAILED] Agent returned success=False")
            
    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_asset_verification()
