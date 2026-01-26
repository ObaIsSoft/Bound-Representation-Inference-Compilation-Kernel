import logging
import random
import copy
import math
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
from backend.agents.evolution import GeometryGenome, EvolutionaryMutator, EvolutionaryCrossover
from backend.agents.surrogate.pinn_model import MultiPhysicsPINN
from backend.agents.critics.adversarial import RedTeamAgent
from backend.agents.critics.scientist import ScientistAgent
from backend.agents.generative.latent_agent import LatentSpaceAgent

logger = logging.getLogger(__name__)

class OptimizationAgent:
    """
    The Evolutionary Designer.
    
    Replaces the old 'Gradient Descent' Tuner with a Population-Based
    Evolutionary Engine that explores Topological Novelty.
    
    Pipeline:
    1. Seed Population (from Constraints or Random)
    2. Evaluate Fitness (Physics/Surrogate)
    3. Select Parents (Tournament)
    4. Crossover (Sexual Reproduction)
    5. Mutate (Topological & Parametric)
    6. Repeat
    """
    
    def __init__(self):
        self.name = "OptimizationAgent"
        # Defaults (will be overridden by config in run)
        self.default_config = {
            "population_size": 50,
            "generations": 20, 
            "mutation_rate": 0.2,
            "crossover_rate": 0.7,
            "topology_add_rate": 0.2,
            "topology_remove_rate": 0.05,
            "param_mutation_strength": 0.2,
            "enable_red_team": True # New config toggle
        }
        self.judge = MultiPhysicsPINN(config=self.default_config) # The Physics Oracle
        self.red_team = RedTeamAgent(self.judge) # The Adversary
        self.scientist = ScientistAgent() # The Observer
        self.latent_agent = LatentSpaceAgent() # The Cartographer
        
    def run(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Evolutionary Loop.
        Input: Constraints (from Dreamer/ISA).
        Output: The Fittest GeometryGenome.
        """
        isa_state = params.get("isa_state", {})
        constraints = isa_state.get('constraints', {})
        
        # Load Config
        config = {**self.default_config, **params.get("config", {})}
        
        pop_size = config["population_size"]
        generations = config["generations"]
        mutation_rate = config["mutation_rate"]
        crossover_rate = config["crossover_rate"]
        possible_assets = config.get("available_assets", [])
        enable_red_team = config.get("enable_red_team", False)
        
        logger.info(f"[EVOLUTION] Starting Evolutionary Run. Constraints: {len(constraints)}. Config: {config}")
        
        # 1. Initialize Population
        population = self._initialize_population(constraints, pop_size, params)
        
        best_genome = None
        best_fitness = float('-inf')
        history_log = []
        red_team_logs = []
        
        # 2. Evolution Loop
        for gen in range(generations):
            # Evaluate (Fast Pass - PINN)
            scored_pop = []
            for genome in population:
                fitness = self._evaluate_fitness(genome, params)
                scored_pop.append((genome, fitness))
                
            # Sort by basic fitness
            scored_pop.sort(key=lambda x: x[1], reverse=True)
            
            # ELITE ADVERSARIAL CHECK
            # "What happens after adversary detects weakness?" -> We punish the weak.
            # Only test the top 20% to save compute
            if enable_red_team:
                elite_count = max(1, int(pop_size * 0.2))
                elites = scored_pop[:elite_count]
                
                # logger.info(f"[Gen {gen}] Red Team attacking top {elite_count} candidates...")
                
                new_scored_elites = []
                for genome, base_fitness in elites:
                    nodes_data = [attr['data'].dict() for attr in genome.graph.nodes.values()]
                    
                    # Run Stress Test (30 trials is enough for a quick check)
                    stress_result = self.red_team.stress_test(nodes_data, constraints, trials=30)
                    
                    # Capture Data for Scientist
                    if gen > generations * 0.5: # Only analyze mature designs
                        # Extract basic metrics
                        node_count = len(nodes_data)
                        # Calc max dimension (approx)
                        max_dim = 0
                        for n in nodes_data:
                            pos = n['transform']
                            max_dim = max(max_dim, abs(pos[0]), abs(pos[1]), abs(pos[2]))
                            
                        red_team_logs.append({
                            "node_count": float(node_count),
                            "max_dim": float(max_dim),
                            "failure_rate": float(stress_result['failure_rate']),
                            "fitness": float(base_fitness)
                        })
                    
                    if not stress_result['is_robust']:
                        # Penalize! 
                        # Fitness *= 0.1 (Severe penalty for failing Red Team)
                        # We want Robustness to be a hard gate.
                        new_fitness = base_fitness * 0.1
                        # logger.info(f"Genome penalized: {base_fitness:.2f} -> {new_fitness:.2f} (Failure: {stress_result['failure_rate']*100:.0f}%)")
                    else:
                        # Bonus for robustness?
                        new_fitness = base_fitness * 1.1 
                        
                    new_scored_elites.append((genome, new_fitness))
                
                # Update the population scores with new values
                scored_pop[:elite_count] = new_scored_elites
                # Re-sort after penalties
                scored_pop.sort(key=lambda x: x[1], reverse=True)

            # Update Best
            current_best, current_score = scored_pop[0]
            if current_score > best_fitness:
                best_fitness = current_score
                best_genome = current_best
            
            # Log progress
            avg_fitness = sum(s[1] for s in scored_pop) / len(scored_pop)
            history_log.append(f"Gen {gen}: Best={best_fitness:.4f}, Avg={avg_fitness:.4f}")
            
            # Selection (Elitism: Keep top 2)
            next_generation = [g for g, s in scored_pop[:2]]
            
            # Breeding
            while len(next_generation) < pop_size:
                parent_a = self._tournament_selection(scored_pop)
                parent_b = self._tournament_selection(scored_pop)
                
                # Crossover
                if random.random() < crossover_rate:
                    child = EvolutionaryCrossover.crossover(parent_a, parent_b)
                else:
                    child = parent_a.clone()
                
                # Mutation
                EvolutionaryMutator.mutate_topology(
                    child, 
                    add_prob=config["topology_add_rate"], 
                    remove_prob=config["topology_remove_rate"],
                    available_assets=possible_assets
                )
                child.mutate_parameter(
                    mutation_rate=mutation_rate, 
                    strength=config["param_mutation_strength"]
                )
                
                next_generation.append(child)
            
            population = next_generation
            
        logger.info(f"[EVOLUTION] Finished. Best Fitness: {best_fitness}")
        
        # SCIENTIST ANALYSIS
        scientific_insight = "No Data"
        if red_team_logs:
             scientific_insight = self.scientist.discover_law(red_team_logs, "failure_rate")
             logger.info(f"[SCIENTIST] Discovery: {scientific_insight}")
             
        # LATENT SPACE GENERATION
        morph_sequence = []
        try:
            # 1. Learn Manifold from final population
            self.latent_agent.learn_manifold(population)
            
            # 2. Interpolate between Top 2 (Winner vs Runner-Up)
            if len(population) >= 2:
                # Re-sort to find best two from final pop
                final_scored = [(g, self._evaluate_fitness(g, params)) for g in population]
                final_scored.sort(key=lambda x: x[1], reverse=True)
                
                winner = final_scored[0][0]
                runner_up = final_scored[1][0]
                
                morph_sequence = self.latent_agent.interpolate(runner_up, winner, steps=5) # Morph from 2nd to 1st
                logger.info("[LATENT] Generated Morph Sequence.")
        except Exception as e:
            logger.error(f"[LATENT] Failed to generate morph: {e}")
        
        return {
            "success": True,
            "strategy_used": "EVOLUTIONARY_TOPOLOGY_OPTIMIZATION + RED_TEAM_ADVERSARY",
            "best_genome": best_genome.to_json() if best_genome else None,
            "fitness_score": best_fitness,
            "evolution_history": history_log,
            "population_count": pop_size,
            "config_used": config,
            "scientific_insight": scientific_insight,
            "morph_sequence": morph_sequence
        }

    def _initialize_population(self, constraints: Dict[str, Any], pop_size: int, params: Dict[str, Any]) -> List[GeometryGenome]:
        """Seeds the population with random valid geometries, respecting user intent."""
        seed_config = params.get("seed_config", {})
        seed_type_str = seed_config.get("type", "CUBE").upper()
        seed_dims = seed_config.get("dimensions", {})
        
        # Map string to Enum
        try:
            from backend.agents.evolution import PrimitiveType
            seed_type = PrimitiveType[seed_type_str]
        except KeyError:
            seed_type = PrimitiveType.CUBE
            
        possible_assets = params.get("config", {}).get("available_assets", [])
            
        pop = []
        for _ in range(pop_size):
            # Create a seeded genome
            g = GeometryGenome(seed_params=seed_dims, seed_type=seed_type) 
            
            # Randomly grow it a bit initially
            for _ in range(random.randint(1, 5)):
                 EvolutionaryMutator._add_random_primitive(g, possible_assets)
            pop.append(g)
        return pop

    def _evaluate_fitness(self, genome: GeometryGenome, params: Dict[str, Any]) -> float:
        """
        The Judge.
        Uses MultiPhysicsPINN to validate the design against Conservation Laws.
        Also evaluates the specific Design Objective.
        """
        constraints = params.get("isa_state", {}).get("constraints", {})
        objective_type = params.get("objective", {}).get("type", "VOLUME")
        
        # Serialize genome nodes for the PINN
        nodes_data = [attr['data'].dict() for attr in genome.graph.nodes.values()]
        
        # Ask the Oracle (PINN) - Basic Check using Nominal Constraints
        result = self.judge.validate_design(nodes_data, constraints)
        physics_score = result['physics_score'] # 0.0 to 1.0
        
        # Objective Evaluation
        objective_score = 0.0
        
        if objective_type == "VOLUME":
            for n in nodes_data:
                p = n.get('params', {})
                if n['type'] == 'CUBE':
                    w = p.get('width', {}).get('value', 1)
                    h = p.get('height', {}).get('value', 1)
                    d = p.get('depth', {}).get('value', 1)
                    objective_score += w * h * d
                    
        elif objective_type == "SPREAD":
            # Reward wide/tall structures (e.g. Drone Arms, Skyscrapers)
            # Metric: Average distance from origin + Bounding Box Size
            for n in nodes_data:
                pos = n.get('transform', [0]*6)
                dist = math.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
                objective_score += dist
                
        elif objective_type == "HEIGHT":
             # Reward Y-axis growth (Skyscraper)
             for n in nodes_data:
                pos = n.get('transform', [0]*6)
                objective_score += pos[1] # Reward positive Y
        
        # Fitness = Validity * (1 + NormalizedObjective)
        # We assume Physics is the "Gatekeeper" (Multiplier).
        fitness = physics_score * (1.0 + objective_score)
        
        return fitness

    def _tournament_selection(self, scored_population: List[Any], k=3) -> GeometryGenome:
        """Selects the best individual from k random samples."""
        tournament = random.sample(scored_population, k)
        return max(tournament, key=lambda x: x[1])[0]
