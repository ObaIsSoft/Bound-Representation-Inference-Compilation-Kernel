"""
Design Exploration Configuration.
Externalizes parameters for DesignExplorationAgent.
"""

EXPLORATION_DEFAULTS = {
    "num_samples": 50,
    "max_iterations": 100,
    "population_size": 20,
    "mutation_rate": 0.1,
    "crossover_rate": 0.8
}

SCORING_WEIGHTS = {
    "mass": -1.0,      # Minimize
    "cost": -2.0,      # Minimize (higher priority)
    "strength": 1.5,   # Maximize
    "efficiency": 1.0  # Maximize
}

SURROGATE_CONFIG = {
    "enabled": True,
    "model_type": "GaussianProcess", # or NeuralNet
    "confidence_threshold": 0.8
}

PARETO_CONFIG = {
    "max_front_size": 10,
    "epsilon": 0.01  # Dominance tolerance
}
