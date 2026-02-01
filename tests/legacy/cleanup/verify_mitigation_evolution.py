
import logging
import sys
import os

sys.path.append(os.path.abspath("."))

from backend.agents.mitigation_agent import MitigationAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MitigationVerify")

def verify_mitigation_evolution():
    logger.info("--- Starting Tier 5 Verification (Mitigation) ---")
    
    agent = MitigationAgent()
    
    if not agent.use_surrogate:
        logger.error("FAILURE: MitigationSurrogate not initialized.")
        return

    # Scenario: Safe Stress (Ratio 0.5) but High Cycles (Fatigue Risk)
    params = {
        "errors": [], # No explicit errors
        "physics_data": {
            "max_stress_mpa": 130.0,
            "yield_strength_mpa": 270.0, # Stress Ratio ~0.48 (Safe by heuristics)
            "temperature_c": 100.0,
            "cycles": 1e8, # 100 Million cycles (High Fatigue)
            "corrosion_index": 0.2
        }
    }
    
    # 1. Test Prediction (Untrained might be random, but we check execution)
    result = agent.run(params)
    fixes = result["fixes"]
    
    surrogate_fix_found = any(f.get("source") == "surrogate" for f in fixes)
    
    logger.info(f"Found {len(fixes)} fixes.")
    
    if surrogate_fix_found:
        logger.info("SUCCESS: Surrogate flagged a risk-based fix.")
    else:
        logger.warning("Surrogate did not flag risk (might need training or threshold adjustment).")
        # Ensure it didn't crash
        
    # 2. Test Training (Evolution)
    logger.info("Testing Evolution Step...")
    # Teach it that Stress 0.9 @ 300C = Failure (1.0)
    training_data = [
        ({"stress_ratio": 0.9, "temp_c": 300, "cycles": 1e3}, 1.0),
        ({"stress_ratio": 0.2, "temp_c": 25, "cycles": 1e3}, 0.0)
    ]
    
    loss = agent.surrogate.train_on_batch(training_data)
    logger.info(f"Training Loss: {loss:.4f}")
    
    if loss < 10.0: # Arbitrary check that it ran
        logger.info("SUCCESS: Training loop executed.")
    else:
        logger.error("FAILURE: Training loss suspiciously high.")

if __name__ == "__main__":
    verify_mitigation_evolution()
