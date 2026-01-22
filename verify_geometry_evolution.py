
import logging
import os
import json
from backend.agents.geometry_agent import GeometryAgent

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GeometryVerif")

def verify_geometry():
    # 0. Clean prior weights
    if os.path.exists("data/geometry_agent_weights.json"):
        os.remove("data/geometry_agent_weights.json")

    agent = GeometryAgent()
    
    # 1. Check Baseline
    settings_1 = agent._load_kernel_settings()
    res_1 = settings_1.get("sdf_resolution", 64)
    logger.info(f"Baseline Resolution: {res_1}")
    
    # 2. Simulate Failure -> Evolution Trigger
    logger.info("Triggering Resolution Increase (Simulated Failure)...")
    agent.update_kernel_settings("INCREASE_RESOLUTION")
    
    # 3. Check Evolution
    settings_2 = agent._load_kernel_settings()
    res_2 = settings_2.get("sdf_resolution", 64)
    logger.info(f"Evolved Resolution: {res_2}")
    
    if res_2 > res_1:
         logger.info(f"✅ GeometryAgent successfully evolved (Resolution {res_1} -> {res_2})")
    else:
         logger.error("❌ GeometryAgent failed to update resolution.")
         exit(1)
         
    # 4. Simulate Slow Performance -> Decrease
    logger.info("Triggering Resolution Decrease (Simulated Slow Perf)...")
    agent.update_kernel_settings("DECREASE_RESOLUTION")
    settings_3 = agent._load_kernel_settings()
    res_3 = settings_3.get("sdf_resolution", 64)
    logger.info(f"Optimized Resolution: {res_3}")
    
    if res_3 < res_2:
        logger.info("✅ GeometryAgent successfully optimized resolution downward.")
    else:
        logger.error("❌ GeometryAgent failed to decrease resolution.")
        exit(1)

if __name__ == "__main__":
    verify_geometry()
