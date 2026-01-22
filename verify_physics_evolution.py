
import sys
import os
import logging
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath("."))

from backend.agents.physics_agent import PhysicsAgent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PhysicsEvolutionVerifier")

def verify_physics_evolution():
    logger.info("--- Starting Deep Evolution Verification (PhysicsAgent) ---")
    
    agent = PhysicsAgent()
    
    # 1. Test Teacher Fallback (Pre-Training)
    logger.info("1. Testing Initial Inference (Expect Teacher)...")
    # Using unrealistic mass to force novel calculation
    env = {"gravity": 9.81, "fluid_density": 1.225, "regime": "AERIAL"}
    geo_tree = [{"params": {"width": 1000, "length": 1000}, "mass_kg": 500.0}] 
    params = {}
    
    res_initial = agent._solve_flight_dynamics(mass=500.0, g=9.81, rho=1.225, area=1.0)
    logger.info(f"Source: {res_initial.get('source')}")
    logger.info(f"Thrust: {res_initial.get('required_thrust_N')}")
    
    if "Teacher" not in res_initial.get('source', ''):
        logger.warning("Agent used Student too early? Or Logic Error.")
        
    # 2. Train the Student (Simulate Active Learning)
    logger.info("\n2. Training Student (Mimicking Oracle)...")
    
    # Generate Synthetic Training Data from "Teacher" Logic
    # Hover Thrust = Mass * G
    # Stall Speed = sqrt(2W / rho A Cl)
    
    training_data = []
    for m in np.linspace(100, 1000, 50):
        # Input: [mass/1000, g/9.81, rho/1.225, area/10, bias]
        inp = [m/1000.0, 1.0, 1.0, 0.1, 1.0]
        
        # Target: [Thrust/1000, Speed/100]
        thrust = m * 9.81
        speed =  np.sqrt((2 * thrust) / (1.225 * 1.0 * 0.5))
        target = [thrust/1000.0, speed/100.0]
        
        training_data.append((inp, target))
        
    evolution_res = agent.evolve(training_data)
    logger.info(f"Evolution Result: {evolution_res}")
    
    # 3. Test Student Graduation (Post-Training)
    logger.info("\n3. Testing Post-Training Inference (Expect Student)...")
    
    # Query within training distribution
    # Note: PhysicsSurrogate logic checks confidence. 
    # Since we don't have real dropout, confidence is mocked based on negative outputs.
    # Provided training was good, outputs should be positive -> Confidence High.
    
    res_post = agent._solve_flight_dynamics(mass=500.0, g=9.81, rho=1.225, area=1.0)
    logger.info(f"Source: {res_post.get('source')}")
    logger.info(f"Thrust: {res_post.get('required_thrust_N')}")
    
    if "Student" in res_post.get('source', ''):
        logger.info("SUCCESS: Agent switched to Neural Surrogate!")
        
        # Verify Accuracy
        # Teacher: 500 * 9.81 = 4905 N
        # Student: Should be close
        error = abs(res_post['required_thrust_N'] - 4905.0)
        logger.info(f"Prediction Error: {error:.2f} N")
        if error < 500.0: # <10% error
            logger.info("Accuracy: PASS")
        else:
            logger.warning("Accuracy: LOW (Need more training?)")
            
    else:
        logger.warning("FAIL: Agent still relying on Teacher (Confidence Low?)")
        
    # 4. Persistence
    if os.path.exists("data/physics_surrogate.weights.json"):
        logger.info("\nSUCCESS: Weights saved.")
    else:
        logger.error("\nFAIL: Weights not saved.")

if __name__ == "__main__":
    verify_physics_evolution()
