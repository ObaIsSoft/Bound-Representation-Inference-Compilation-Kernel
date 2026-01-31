"""
Swarm and Construction Agent Configuration.
Externalizes constants for ConstructionAgent, SwarmManager, and related entities.
"""

# Construction Agent Defaults
CONSTRUCTION_DEFAULTS = {
    "initial_energy": 150.0,
    "metabolism_rate": 1.2,
    "replication_threshold": 600.0,
    "movement_step": 3.0,
    "harvest_amount": 10.0,
    "ore_storage_cap": 100.0,
    "build_cost_ore": 50.0,
    "build_reward_structure": 1
}

# Pheromone Thresholds
PHEROMONE_THRESHOLDS = {
    "build_trigger": 5.0,
    "harvest_attraction_multiplier": 10.0,
    "evaporation_rate": 0.99,
    "diffusion_rate": 0.1
}

# Genetic Parameters
GENETIC_DEFAULTS = {
    "mutation_rate": 0.05,
    "crossover_rate": 0.7,
    "energy_cost_replication": 0.7  # Fraction of threshold
}
