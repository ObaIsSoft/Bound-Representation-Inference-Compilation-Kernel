
import logging
import sys
import os

# Ensure backend path is available
sys.path.append(os.path.abspath("."))

from backend.agents.mass_properties_agent import MassPropertiesAgent

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MassValidation")

def verify_mass_properties():
    logger.info("--- Starting Tier 5 Verification (Mass Properties) ---")
    
    agent = MassPropertiesAgent()
    
    if not agent.use_surrogate:
        logger.error("FAILURE: MassPropertiesSurrogate not initialized.")
        return

    # Test Case: Solid Cube (10x10x10cm, Aluminum)
    # Mass ~ 2.7kg. 
    # Ixx = m/12 * (0.1^2 + 0.1^2) = 2.7/12 * 0.02 = 0.225 * 0.02 = 0.0045 kg.m2
    
    params = {
        "volume_cm3": 1000.0,
        "material_density": 2.7,
        "bounding_box": [10.0, 10.0, 10.0]
    }
    
    logger.info("1. Running Analysis...")
    result = agent.run(params)
    
    if result["status"] == "success":
        m = result["mass"]["value"]
        I = result["inertia_tensor"]
        logger.info(f"Mass: {m} kg")
        logger.info(f"Inertia: {I}")
        logger.info(f"Logs: {result['logs']}")
        logger.info("SUCCESS: MassPropertiesAgent ran successfully.")
    else:
        logger.error(f"FAILURE: {result}")

if __name__ == "__main__":
    verify_mass_properties()
