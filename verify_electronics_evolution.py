
import logging
import sys
import os
import numpy as np

# Ensure backend path is available
sys.path.append(os.path.abspath("."))

from backend.agents.electronics_agent import ElectronicsAgent
from backend.agents.critics.ElectronicsCritic import ElectronicsCritic

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ElectronicsVerification")

def verify_electronics_evolution():
    logger.info("--- Starting Deep Evolution Verification (Electronics) ---")
    
    # 0. Clean Slate: Remove existing weights to ensure fresh evolution
    # (Addresses user request to ensure no prior instance data is used)
    weights_path = "data/electronics_surrogate.weights.json"
    if os.path.exists(weights_path):
        os.remove(weights_path)
        logger.info("CLEANUP: Removed existing weights file to ensure fresh evolution.")
    
    # 1. Initialize Agent & Critic
    agent = ElectronicsAgent()
    critic = ElectronicsCritic(window_size=20)
    
    # Check if Surrogate loaded
    if not agent.surrogate:
        logger.error("FAILURE: ElectronicsSurrogate not initialized in Agent.")
        return

    # 2. Generate Synthetic Training Data (Oracle Runs)
    # Topology: Simple Buck Converter components
    # Good Design: L, C, S, D
    good_topology = {
        "components": ["Inductor", "Capacitor", "MOSFET", "Diode"],
        "v_in": 12.0,
        "v_out_target": 5.0
    }
    good_result = {"efficiency": 0.95, "ripple_mv": 20.0} # "Ground Truth"
    
    # Bad Design: Just Resistors
    bad_topology = {
        "components": ["Resistor", "Resistor"],
        "v_in": 12.0,
        "v_out_target": 5.0
    }
    bad_result = {"efficiency": 0.30, "ripple_mv": 500.0}
    
    # 3. Feed Data to Critic (Simulate Operation)
    logger.info("1. Feeding interactions to Critic...")
    for _ in range(10):
        critic.observe(
            input_state={"topology": good_topology},
            electronics_output=good_result, # Acting as Ground Truth for this test context
            actual_power_outcome=None
        )
        critic.observe(
            input_state={"topology": bad_topology},
            electronics_output=bad_result,
            actual_power_outcome=None
        )

    # 4. Trigger Evolution (Train Surrogate)
    logger.info("2. Triggering Evolution (Training)...")
    report = critic.evolve_agent(agent)
    logger.info(f"Evolution Report: {report}")
    
    if report.get("status") != "success":
        logger.error(f"FAILURE: Evolution failed with status {report.get('status')}")
        return

    # 5. Verify Learning (Inference)
    logger.info("3. Verifying Surrogate Predictions...")
    
    # Agent should now use Surrogate
    # Force usage just in case logic in evaluate_fitness is complex
    agent.use_surrogate = True 
    
    pred_good = agent.surrogate.predict_performance(good_topology)
    pred_bad = agent.surrogate.predict_performance(bad_topology)
    
    logger.info(f"Good Topology Prediction: Eff={pred_good['efficiency']:.2f}, Ripple={pred_good['ripple_mv']:.1f}")
    logger.info(f"Bad Topology Prediction:  Eff={pred_bad['efficiency']:.2f}, Ripple={pred_bad['ripple_mv']:.1f}")
    
    # Assertions
    if pred_good['efficiency'] > 0.8 and pred_bad['efficiency'] < 0.5:
        logger.info("SUCCESS: Surrogate learned distinction between Good and Bad topologies.")
    else:
        logger.error("FAILURE: Surrogate predictions do not match expected patterns.")

if __name__ == "__main__":
    verify_electronics_evolution()
