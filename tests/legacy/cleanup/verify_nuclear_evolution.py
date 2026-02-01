
import sys
import os
import logging
import numpy as np

sys.path.append(os.path.abspath("."))

from backend.agents.physics_agent import PhysicsAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NuclearVerification")

def verify_nuclear_evolution():
    logger.info("--- Starting Deep Evolution Verification (Nuclear) ---")
    
    agent = PhysicsAgent()
    
    # 1. Initial Inference (Oracle)
    params = {
        "physics_domain": "NUCLEAR",
        "type": "FUSION",
        "density": 1e20,
        "temperature_kev": 15.0,
        "confinement_time": 2.0,
        "fuel": "DT"
    }
    
    # Mock environment/geometry as they are not used for pure physics query
    env = {}
    geo = []
    
    logger.info("1. Querying Agent (Expect Oracle/Teacher)...")
    res_initial = agent.run(env, geo, params)
    
    preds = res_initial.get("physics_predictions", {})
    source = preds.get("source", "Unknown")
    q_factor = preds.get("Q_factor", 0)
    
    logger.info(f"Source: {source}")
    logger.info(f"Q Factor: {q_factor}")
    
    # 2. Train (Simulate Evolution)
    logger.info("\n2. Training Nuclear Surrogate...")
    
    # Generate Synthetic Data (Simple relation: Q ~ T * tau)
    training_data = []
    for t in np.linspace(5, 50, 20):
        # Input: [log_n, T/100, tau/10, fuel]
        # density 1e20 -> log10=20 -> /20 = 1.0
        row_in = [1.0, t/100.0, 0.2, 1.0] # tau=2.0
        
        # Target: Q factor
        # Lawson: ~ n T tau. 
        # Let's just say Q = T * 0.5 for simple regression learning
        q_target = t * 0.5 
        
        # Output: [Q/10, Power/100]
        row_out = [q_target/10.0, 0.5]
        
        training_data.append((row_in, row_out))
        
    agent.evolve(training_data)
    
    # 3. Post-Training Inference (Surrogate)
    logger.info("\n3. Querying Agent (Expect Surrogate)...")
    
    # Use parameters from training set
    params['temperature_kev'] = 50.0 # Should give Q ~ 25
    
    res_post = agent.run(env, geo, params)
    preds_post = res_post.get("physics_predictions", {})
    source_post = preds_post.get("source", "Unknown")
    q_post = preds_post.get("Q_factor", 0)
    
    logger.info(f"Source: {source_post}")
    logger.info(f"Q Factor (Predicted): {q_post}")
    
    if "Student" in source_post:
        logger.info("SUCCESS: Switched to Nuclear Surrogate.")
    else:
        logger.warning(f"FAIL: Still using {source_post}")

    if q_post > 5.0: # Expect ~25
        logger.info("SUCCESS: Learned High Q Factor.")
    else:
        logger.warning("FAIL: Prediction too low (Learning failed?)")

if __name__ == "__main__":
    verify_nuclear_evolution()
