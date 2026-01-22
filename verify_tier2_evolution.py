
import logging
import json
import os
from backend.agents.material_agent import MaterialAgent
from backend.agents.chemistry_agent import ChemistryAgent
from backend.agents.electronics_agent import ElectronicsAgent

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("Tier2Verification")

def verify_material_evolution():
    logger.info("\n--- Verifying MaterialAgent Evolution ---")
    agent = MaterialAgent()
    mat_name = "Aluminum 6061"
    
    # 1. Baseline Run (Temp > Max)
    res1 = agent.run(mat_name, temperature=200.0) # Max 150
    factor1 = res1["properties"]["strength_factor"]
    logger.info(f"Baseline (Heuristic) Factor: {factor1}")
    
    # 2. Evolve forces highly sensitive degradation (beta > 0)
    # Critic says: "Hey, actual strength was LOWER than predicted!"
    agent.update_learned_parameters(mat_name, {"degradation_beta": 0.01})
    
    # 3. Evolved Run (Same Temp)
    res2 = agent.run(mat_name, temperature=200.0)
    factor2 = res2["properties"]["strength_factor"]
    logger.info(f"Evolved (Hybrid) Factor: {factor2}")
    
    if factor2 < factor1:
        logger.info("✅ MaterialAgent successfully evolved (learned higher sensitivity)!")
    else:
        logger.error("❌ MaterialAgent failed to evolve.")

def verify_chemistry_evolution():
    logger.info("\n--- Verifying ChemistryAgent Evolution ---")
    agent = ChemistryAgent()
    mat_family = "steel"
    
    # 1. Baseline
    # We inspect internals or run a step
    # Let's peek at the calculated base_rate by mocking internal logic call
    # Or rely on run? run() is static check. step() is dynamic.
    # We'll use step()
    state = {"integrity": 1.0, "corrosion_depth": 0.0, "mass_loss": 0.0}
    inputs = {"material_type": "Steel", "temperature": 20.0, "humidity": 1.0}
    
    res1 = agent.step(state, inputs, dt=31536000) # 1 Year
    depth1 = res1["state"]["corrosion_depth"]
    logger.info(f"Baseline Corrosion (1 yr): {depth1:.4f} mm")
    
    # 2. Evolve (Double the rate)
    agent.update_learned_parameters(mat_family, {"corrosion_rate_factor": 2.0})
    
    res2 = agent.step(state, inputs, dt=31536000)
    depth2 = res2["state"]["corrosion_depth"]
    logger.info(f"Evolved Corrosion (1 yr): {depth2:.4f} mm")
    
    if depth2 > depth1 * 1.5:
        logger.info("✅ ChemistryAgent successfully evolved (learned higher rate)!")
    else:
        logger.error("❌ ChemistryAgent failed to evolve.")

def verify_electronics_evolution():
    logger.info("\n--- Verifying ElectronicsAgent Evolution ---")
    agent = ElectronicsAgent()
    
    # 1. Baseline (Eff=0.95, LoadFactor=1.10 from code default)
    params = {
        "resolved_components": [
            {"category": "source", "power_peak_w": 100.0},
            {"category": "load", "power_peak_w": 90.0}
        ]
    }
    
    res1 = agent.run(params)
    supply1 = res1["power_analysis"]["hybrid_supply_w"]
    demand1 = res1["power_analysis"]["hybrid_demand_w"]
    margin1 = res1["power_analysis"]["margin_w"]
    logger.info(f"Baseline Margin: {margin1:.2f}W (Supply {supply1} - Demand {demand1})")
    
    # 2. Evolve (Source is terrible, Loads are hungry)
    agent.update_learned_parameters({"source_efficiency": 0.50, "load_correction": 2.0})
    
    res2 = agent.run(params)
    supply2 = res2["power_analysis"]["hybrid_supply_w"]
    demand2 = res2["power_analysis"]["hybrid_demand_w"]
    margin2 = res2["power_analysis"]["margin_w"]
    logger.info(f"Evolved Margin: {margin2:.2f}W (Supply {supply2} - Demand {demand2})")
    
    if supply2 < supply1 and demand2 > demand1:
        logger.info("✅ ElectronicsAgent successfully evolved (adjusted efficiency factors)!")
    else:
        logger.error("❌ ElectronicsAgent failed to evolve.")

if __name__ == "__main__":
    verify_material_evolution()
    verify_chemistry_evolution()
    verify_electronics_evolution()
