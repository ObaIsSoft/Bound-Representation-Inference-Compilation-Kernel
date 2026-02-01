
import logging
import time
from backend.agents.component_agent import ComponentAgent
try:
    from models.component import Component
except ImportError:
    from backend.models.component import Component

# Mock Data setup
# ComponentAgent normally checks Supabase. We will mock _fetch_candidates or use real checks if DB avail?
# DB is likely available (Tier 2 verification worked partially).
# But Component table might be empty.
# I will use a Mock or rely on fallback?
# ComponentAgent has NO fallback for _fetch_candidates. It returns [].
# I must Mock _fetch_candidates for this test to work without DB.

def mock_fetch(self, category=None):
    return [
        Component({"id": "C1", "name": "Part A", "category": "motor", "specs": {"power_w": 100}}),
        Component({"id": "C2", "name": "Part B", "category": "motor", "specs": {"power_w": 100}}),
        Component({"id": "C3", "name": "Part C", "category": "motor", "specs": {"power_w": 100}})
    ]

ComponentAgent._fetch_candidates = mock_fetch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ComponentVerif")

def verify_ranking():
    agent = ComponentAgent()
    params = {"requirements": {"min_power_w": 50}, "limit": 3}
    
    # 0. Reset Weights (optional, or just handle existing)
    agent.update_preferences("C1", 0.0) # Reset C1 logic effect roughly? No, 0.0 is bias? 
    # Logic: new = old + 0.1*reward.
    
    # 1. Baseline
    # All defaults 1.0. Order might be C1, C2, C3 (list order).
    res1 = agent.run(params)
    selection1 = res1["selection"]
    ids1 = [c["id"] for c in selection1]
    logger.info(f"Run 1 Order: {ids1}")
    
    # 2. Punish C1 (e.g. Critical Failure in Installation)
    # Reward = -5.0
    logger.info("Punishing C1...")
    agent.update_preferences("C1", -5.0) 
    
    # 3. Reward C3 (Good)
    logger.info("Rewarding C3...")
    agent.update_preferences("C3", 2.0)
    
    # 4. Run 2
    res2 = agent.run(params)
    selection2 = res2["selection"]
    ids2 = [c["id"] for c in selection2]
    logger.info(f"Run 2 Order: {ids2}")
    
    # Check if C3 is ahead of C1
    # C3 score should be > 1.0 (approx 1.2)
    # C1 score should be < 1.0 (approx 0.5)
    
    if ids2[0] == "C3" and "C1" in ids2 and ids2.index("C3") < ids2.index("C1"):
        logger.info("✅ ComponentAgent successfully evolved ranking (C3 promoted, C1 demoted)!")
    elif ids2[0] == "C3":
        logger.info("✅ C3 Promoted to top!")
    else:
        logger.error(f"❌ Ranking check failed. {ids2}")
        exit(1)

if __name__ == "__main__":
    verify_ranking()
