
import sys
import os
import logging
import numpy as np
from collections import deque

sys.path.append(os.path.abspath("."))

from backend.agents.critics.PhysicsCritic import PhysicsCritic, CriticReport

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("UniversalPhysicsVerifier")

def verify_universal_physics():
    logger.info("--- Starting Universal Physics Critic Verification ---")
    
    critic = PhysicsCritic()
    
    # Test Cases: (Domain, Input, Prediction, ViolationExpected)
    test_cases = [
        (
            "THERMODYNAMICS (1st Law)", 
            {"physics_domain": "THERMODYNAMICS", "t_hot": 1000, "t_cold": 500}, 
            {"efficiency": 1.2}, 
            "1st Law"
        ),
        (
            "THERMODYNAMICS (2nd Law - Carnot)", 
            {"physics_domain": "THERMODYNAMICS", "t_hot": 1000, "t_cold": 500}, 
            {"efficiency": 0.6}, # Carnot = 1 - 500/1000 = 0.5. So 0.6 is a violation!
            "Carnot Limit"
        ),
        (
            "OPTICS", 
            {"physics_domain": "OPTICS"}, 
            {"reflectance": 0.5, "transmittance": 0.6, "absorbance": 0.0}, # Sum 1.1 
            "OPTICS VIOLATION"
        ),
        (
            "PLASMA",
            {"physics_domain": "PLASMA", "Z": 1},
            {"electron_density": 1e18, "ion_density": 1e19}, # Mismatch
            "Quasi-Neutrality"
        ),
        (
            "CIRCUITS",
            {"physics_domain": "CIRCUITS"},
            {"supply_power": 100, "demand_power": 80, "margin": 50}, # Actual margin 20
            "KCL Power Mismatch"
        ),
        (
            "QUANTUM", 
            {"physics_domain": "QUANTUM"}, 
            {"probabilities": [0.5, 0.2]}, # Sum 0.7 
            "QUANTUM VIOLATION"
        ),
        (
            "RELATIVITY", 
            {"physics_domain": "RELATIVITY"}, 
            {"velocity_m_s": 3e8 + 100}, # > c 
            "RELATIVITY VIOLATION"
        )
    ]
    
    success_count = 0
    
    for domain, inp, pred, expected_msg in test_cases:
        logger.info(f"\nTesting Domain: {domain}")
        
        # Reset Critic History for clarity (optional, but good for isolation)
        critic = PhysicsCritic() 
        
        # Observe (Mock values for non-critical args)
        # Note: observe() takes numpy array for input usually, but we modified critic to check input_history dict
        # Wait, PhysicsCritic.observe() expects input_state: np.ndarray.
        # But _check_conservation_laws retrieves self.input_history[-1] which is appended as input_state.copy().
        # If we pass a dict, numpy will complain or wrap it.
        # Let's see how observe handles it.
        # observe(input_state: np.ndarray ...) -> self.input_history.append(input_state.copy())
        
        # We need to pass the dict so checks can read 'physics_domain'.
        # Python lists/dicts are valid 'input_state' arguments in python (dynamic typing), 
        # unless type hinted strictly enforced by runtime (unlikely here).
        
        critic.observe(inp, pred, 0.0, 0.0) # prediction triggers append to history
        
        # Manually trigger check (normally called by analyze -> _detect_failure_modes -> _check_conservation_laws)
        report = critic.analyze()
        failures = report.failure_modes
        
        # Check if expected message is in failures
        found = any(expected_msg in f for f in failures)
        
        if found:
            logger.info(f"SUCCESS: Caught {expected_msg}")
            success_count += 1
        else:
            logger.error(f"FAILURE: Missed {expected_msg}. Got: {failures}")
            
    if success_count == len(test_cases):
        logger.info("\nALL UNIVERSAL CHECKS PASSED.")
    else:
        logger.warning(f"\nPassed {success_count}/{len(test_cases)} checks.")

if __name__ == "__main__":
    verify_universal_physics()
