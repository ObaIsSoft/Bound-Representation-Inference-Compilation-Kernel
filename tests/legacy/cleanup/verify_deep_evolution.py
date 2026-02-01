
import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath("."))

from backend.agents.material_agent import MaterialAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DeepEvolutionVerifier")

def verify_deep_evolution():
    logger.info("--- Starting Deep Evolution Verification (MaterialAgent) ---")
    
    agent = MaterialAgent()
    
    # 1. Test Inference (Before Training)
    logger.info("1. Testing Initial Inference...")
    result_initial = agent.run("Aluminum", temperature=200.0)
    logger.info(f"Initial Result: {result_initial['properties']['strength_factor']:.3f}")
    logger.info(f"Model Used: {result_initial['properties']['degradation_model']}")
    
    if "MaterialNet" not in result_initial['properties']['degradation_model']:
        logger.error("FAIL: MaterialAgent is not using MaterialNet!")
        # return
        
    initial_correction = result_initial['properties'].get('neural_correction', 0.0)
    logger.info(f"Initial Neural Correction: {initial_correction}")

    # 2. Test Evolution (Training)
    logger.info("\n2. Testing Evolution (Backpropagation)...")
    # Simulate a scenario: Real world says material is STRONGER than heuristic (Target Correction = +0.1)
    # Heuristic at 200C (excess 50C) -> 1.0 - (50 * 0.005) = 0.75
    # Reality says it should be 0.85
    # So we want Neural Correction -> +0.1
    
    input_vector = [200.0/1000.0, 276e6/1e9, 0.0, 7.0] # Scaled inputs
    target_output = [0.1, 0.0] # Target correction for [strength, stiffness]
    
    training_data = [(input_vector, target_output) for _ in range(50)] # 50 Epochs
    
    evolution_result = agent.evolve(training_data)
    logger.info(f"Evolution Result: {evolution_result}")
    
    # 3. Test Inference (After Training)
    logger.info("\n3. Testing Post-Training Inference...")
    result_post = agent.run("Aluminum", temperature=200.0)
    post_correction = result_post['properties'].get('neural_correction', 0.0)
    logger.info(f"Post-Training Correction: {post_correction}")
    
    if post_correction > initial_correction:
         logger.info("SUCCESS: Neural Network Learned to increase strength factor!")
    else:
         logger.warning("WARNING: Neural Network did not shift significantly (check learning rate?)")

    # 4. Persistence Check
    if os.path.exists("data/material_net.weights.json"):
        logger.info("\nSUCCESS: Weights file saved to disk.")
    else:
        logger.error("\nFAIL: Weights file NOT found.")
        
    # 5. Full Loop Test (Critic -> Agent)
    logger.info("\n5. Testing Full Critic Loop...")
    from backend.agents.critics.MaterialCritic import MaterialCritic
    critic = MaterialCritic()
    
    # Simulate history
    for _ in range(20):
        # Run agent
        res = agent.run("Aluminum", temperature=250.0) # High Temp
        # Observe
        critic.observe(
            {"material_name": "Aluminum", "temperature": 250.0},
            res
        )
        
    # Trigger Critic Evolution
    logger.info("Triggering Critic Evolution...")
    evol_stats = critic.evolve_agent(agent)
    logger.info(f"Critic Evolution Stats: {evol_stats}")
    
    if evol_stats and evol_stats.get("status") == "evolved":
        logger.info("SUCCESS: Critic successfully trained Agent!")
    else:
        logger.error("FAIL: Critic failed to evolve Agent.")

if __name__ == "__main__":
    verify_deep_evolution()

