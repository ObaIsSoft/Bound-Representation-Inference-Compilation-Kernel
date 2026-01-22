
import logging
import os
import json
from backend.agents.topological_agent import TopologicalAgent

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TopologyVerif")

def verify_topology():
    # 0. Clean prior weights
    if os.path.exists("data/topological_agent_weights.json"):
        os.remove("data/topological_agent_weights.json")

    agent = TopologicalAgent()
    params = {
        "elevation_data": [10, 20, 30], # Placeholder to trigger non-flat logic
        "preferences": {"prefer_speed": False} # Prefer Ground
    }
    
    # Mock internal methods to force specific terrain values for consistency
    # (Since _analyze_elevation is a stub returning Fixed 12.0 deg)
    # We will just test the calculation method directly or assume default stub values.
    # Stub returns slope=12.0, roughness=0.25.
    
    # 1. Baseline Check
    # Mode = GROUND (Gentle Hills)
    # Slope 12/45 = 0.266
    # Penalty = 0.266 * 0.6 + 0.25 * 0.4 = 0.16 + 0.1 = 0.26
    # Score ~ 0.74
    
    res_1 = agent.run(params)
    score_1 = res_1["traversability"]
    logger.info(f"Baseline Traversability Score: {score_1:.3f}")
    
    # 2. Simulate Crash -> "You were too confident!" -> Increase Penalty
    logger.info("Triggering Safety Update (Simulated Crash)...")
    agent.update_weights("INCREASE_PENALTY")
    
    # 3. Evolved Check
    res_2 = agent.run(params)
    score_2 = res_2["traversability"]
    logger.info(f"Evolved Traversability Score: {score_2:.3f}")
    
    # We expect Score 2 to be LOWER (Stricter) than Score 1
    if score_2 < score_1:
         logger.info(f"✅ TopologicalAgent successfully learned caution ({score_1:.3f} -> {score_2:.3f})")
    else:
         logger.error("❌ TopologicalAgent failed to update weights.")
         exit(1)

if __name__ == "__main__":
    verify_topology()
