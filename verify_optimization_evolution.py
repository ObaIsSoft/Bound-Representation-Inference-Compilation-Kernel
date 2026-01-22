import logging
import time
import random
import sys
import os

# Add backend to path so 'import agents' works (legacy support)
sys.path.append(os.path.join(os.getcwd(), 'backend'))

from backend.agents.optimization_agent import OptimizationAgent, OptimizationStrategy
from backend.agents.critics.OptimizationCritic import OptimizationCritic

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("OptimizationVerification")

def verify_meta_learning():
    """
    Runs a loop to verify that the Agent learns to switch strategies.
    Scenario: Complex "Trap" Function where Gradient Descent fails.
    """
    logger.info("Starting Optimization Meta-Learning Verification...")
    
    agent = OptimizationAgent()
    critic = OptimizationCritic()
    
    # Force initial bias to Gradient Descent to prove it can unlearn it
    agent.selector.weights = {
        OptimizationStrategy.GRADIENT_DESCENT: 0.9,
        OptimizationStrategy.GENETIC_ALGORITHM: 0.05,
        OptimizationStrategy.SIMULATED_ANNEALING: 0.05
    }
    logger.info(f"Initial Weights: {agent.selector.weights}")
    
    # Simulation Loop
    for i in range(1, 15):
        logger.info(f"\n--- Iteration {i} ---")
        
        # 1. Setup Problem (High Constraints -> Trap for GD)
        # We assume GRADIENT DESCENT fails here (simulated)
        constraints = {
            "param1": {"val": {"value": 1.0}, "locked": False},
            "param2": {"val": {"value": 1.0}, "locked": False},
            # Dummy constraints to trigger "complex" context
            "c1": {"val": {"value": 0}, "locked": True},
            "c2": {"val": {"value": 0}, "locked": True}
        }
        
        payload = {
            "isa_state": {"constraints": constraints},
            "objective": {"id": "test_objective", "target": "MINIMIZE", "metric": "MASS"}
        }
        
        # 2. Run Agent
        result = agent.run(payload)
        strategy_used = result["strategy_used"]
        logger.info(f"Agent chose: {strategy_used}")
        
        # 3. Simulate Environment Feedback (The "Real World" Test)
        # If GD was used -> FAILURE (Stuck in trap)
        # If GA was used -> SUCCESS (Found global optimum)
        # If SA was used -> PARTIAL SUCCESS
        
        success = False
        efficiency = 0.0
        
        if strategy_used == OptimizationStrategy.GRADIENT_DESCENT.value:
            logger.info("❌ Gradient Descent stuck in local minima!")
            success = False
            efficiency = 0.1
        elif strategy_used == OptimizationStrategy.GENETIC_ALGORITHM.value:
            logger.info("✅ Genetic Algorithm found global optimum!")
            success = True
            efficiency = 0.8
        elif strategy_used == OptimizationStrategy.SIMULATED_ANNEALING.value:
            logger.info("⚠️ Annealing worked okay.")
            success = True
            efficiency = 0.5
            
        # 4. Update Agent (Self-Evolution)
        # Note: In real app, this happens via Critic triggering it, 
        # or Agent calling selector.update() directly.
        # In our refactor, Agent calls selector.update() internally based on runtime success.
        # But we need to feed "External Success" back into it for the verification to be meaningful.
        # The agent's internal run() success is just "did code run". 
        # We need to manually override the update with "Environmental Success".
        
        # Manually reinforcing for verification simulation
        strat_enum = OptimizationStrategy(strategy_used)
        agent.selector.update_policy(strat_enum, success, efficiency)
        
        # 5. Critic Monitoring
        critic.observe("OptimizationAgent", payload, {"strategy_used": strategy_used, "success": success, "mutations": result.get("mutations")}, {"timestamp": time.time()})
        report = critic.analyze()
        
        if report.recommendations:
            logger.info(f"Critic Recommendation: {report.recommendations[0]}")
            
        logger.info(f"Updated Weights: {agent.selector.weights}")
        
            
        logger.info(f"Updated Weights: {agent.selector.weights}")
        
        # Check convergence of weights
        # Success if we switched preferred strategy
        best_strat = max(agent.selector.weights, key=agent.selector.weights.get)
        if best_strat != OptimizationStrategy.GRADIENT_DESCENT:
             logger.info(f"\n🏆 SUCCESS: Agent evolved! New preferred strategy: {best_strat}")
             return
            
    logger.error("❌ FAILURE: Agent did not switch strategies in time.")
    exit(1)

if __name__ == "__main__":
    verify_meta_learning()
