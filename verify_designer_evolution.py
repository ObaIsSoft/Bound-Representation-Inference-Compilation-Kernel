
import logging
import os
import json
from backend.agents.designer_agent import DesignerAgent
from backend.agents.critics.DesignCritic import DesignCritic

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DesignerVerif")

def verify_designer():
    # 0. Clean prior weights
    if os.path.exists("data/designer_agent_weights.json"):
        os.remove("data/designer_agent_weights.json")

    agent = DesignerAgent()
    params = {"style": "modern"} # No base_color -> trigger generative
    
    # 1. Baseline Phase (Should be Random)
    logger.info("--- Baseline Phase (Random) ---")
    hues_1 = []
    for _ in range(5):
        res = agent.run(params)
        color = res["aesthetics"]["primary"]
        h, _, _ = agent._hex_to_hsv(color)
        hues_1.append(h)
        logger.info(f"Generated: {color} (Hue: {h:.2f})")
        
    # Check diversity (should be highish)
    import numpy as np
    std_1 = np.std(hues_1)
    logger.info(f"Baseline StdDev: {std_1:.2f}")

    # 2. Reinforce 'Red' (Hue ~0.0 or 1.0)
    logger.info("--- Training Phase (Reinforcing Red) ---")
    # Simulate user liking Red 5 times
    for _ in range(5):
        agent.update_preferences(0.0) # Red
        agent.update_preferences(0.99)# Red
        
    # 3. Evolved Phase (Should favor Red)
    logger.info("--- Evolved Phase (Should be Red-ish) ---")
    hues_2 = []
    red_count = 0
    for _ in range(10):
        res = agent.run(params)
        color = res["aesthetics"]["primary"]
        h, _, _ = agent._hex_to_hsv(color)
        hues_2.append(h)
        
        # Red is close to 0 or 1
        if h < 0.1 or h > 0.9:
            red_count += 1
            
        logger.info(f"Generated: {color} (Hue: {h:.2f}) { '[RED]' if (h<0.1 or h>0.9) else '' }")
        
    logger.info(f"Red Usage: {red_count}/10")
    
    if red_count >= 5: # Expect >50% exploitation (random exploration is only 20%)
        logger.info("✅ DesignerAgent successfully learned preference for RED!")
    else:
        logger.error("❌ DesignerAgent failed to learn preference.")
        exit(1)

if __name__ == "__main__":
    verify_designer()
