
import sys
import os
import logging
import numpy as np

sys.path.append(os.path.abspath("."))

from backend.agents.chemistry_agent import ChemistryAgent
from backend.agents.critics.ChemistryCritic import ChemistryCritic

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ChemistryEvolutionVerifier")

def verify_chemistry_evolution():
    logger.info("--- Starting Deep Evolution Verification (ChemistryAgent) ---")
    
    agent = ChemistryAgent()
    critic = ChemistryCritic()
    
    # 1. Initial State
    # Run in Acidic Env (pH 4.0)
    # Default surrogate (random weights) might predict anything, but likely ~0.5 or 0 based on initialization/relu
    inputs = {
        "ph": 4.0,
        "temperature": 25.0,
        "humidity": 0.8,
        "material_type": "steel",
        "thickness_mm": 5.0
    }
    state = {"corrosion_depth": 0.0, "integrity": 1.0}
    
    res_initial = agent.step(state, inputs, dt=3600*24*30) # 1 Month Step
    initial_factor = res_initial['metrics'].get('neural_factor', 1.0)
    logger.info(f"Initial Neural Factor (pH 4.0): {initial_factor:.3f}")
    
    # Observe
    critic.observe({"environment_type": "ACIDIC", **inputs}, res_initial)
    
    # 2. Train (Critic drives Evolution)
    # The Critic logic currently sets Target=2.0 for pH < 5
    logger.info("Triggering Critic Evolution (Target Factor = 2.0)...")
    
    # We need a few more samples to enable training (Critic checks history len > 10)
    for _ in range(15):
        critic.observe({"environment_type": "ACIDIC", **inputs}, res_initial)
        
    evol_stats = critic.evolve_agent(agent)
    logger.info(f"Evolution Stats: {evol_stats}")
    
    # 3. Post-Training Check
    res_post = agent.step(state, inputs, dt=3600*24*30)
    post_factor = res_post['metrics'].get('neural_factor', 1.0)
    logger.info(f"Post-Training Neural Factor (pH 4.0): {post_factor:.3f}")
    
    if post_factor > initial_factor: # Should move towards 2.0
        logger.info("SUCCESS: Neural Network Learned Acid Acceleration!")
    else:
        logger.warning(f"Result did not increase (Target 2.0). Init: {initial_factor}, Post: {post_factor}")

    # 4. Persistence
    if os.path.exists("data/chemistry_surrogate.weights.json"):
        logger.info("Weights saved.")
    else:
        logger.error("Weights NOT saved.")

if __name__ == "__main__":
    verify_chemistry_evolution()
