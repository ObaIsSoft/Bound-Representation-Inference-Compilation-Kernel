
import logging
import sys
import os

sys.path.append(os.path.abspath("."))

from backend.agents.cost_agent import CostAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CostValidation")

def verify_cost_evolution():
    logger.info("--- Starting Tier 5 Verification (Cost) ---")
    
    agent = CostAgent()
    if not agent.use_surrogate:
        logger.error("FAILURE: CostSurrogate not initialized.")
        return

    # Check that multiplier is used
    params = {
        "mass_kg": 10.0,
        "material_name": "Aluminum 6061", # Base cost $20/kg -> $200
        "processing_time_hr": 0.0 # Ignore process cost
    }
    
    result = agent.run(params)
    
    if result["status"] == "success":
        mat_cost = result["breakdown"]["material"]
        mult = result["breakdown"]["market_multiplier"]
        
        # Base cost should be 200 * multiplier
        expected = 200.0 * mult
        diff = abs(mat_cost - expected)
        
        logger.info(f"Multiplier: {mult:.3f}")
        logger.info(f"Material Cost: ${mat_cost:.2f} (Expected ~${expected:.2f})")
        
        if diff < 1.0:
            logger.info("SUCCESS: CostAgent correctly applied market multiplier.")
        else:
            logger.error(f"FAILURE: Math mismatch. {mat_cost} != {expected}")
    else:
        logger.error("FAILURE: Agent run failed.")

if __name__ == "__main__":
    verify_cost_evolution()
