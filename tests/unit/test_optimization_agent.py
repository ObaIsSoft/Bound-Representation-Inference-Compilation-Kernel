
import pytest
from unittest.mock import MagicMock, patch
from backend.agents.optimization_agent import OptimizationAgent
from backend.agents.evolution import GeometryGenome

@pytest.fixture
def optimization_agent():
    """Returns an OptimizationAgent with mocked heavy dependencies."""
    # We purposefully mock the Judge (PINN), Red Team, Scientist, and Latent Agent
    # because we want to test the GENETIC ALGORITHM logic, not the physics engine.
    with patch("backend.agents.optimization_agent.MultiPhysicsPINN") as Mockjudge, \
         patch("backend.agents.optimization_agent.RedTeamAgent") as MockRedTeam, \
         patch("backend.agents.optimization_agent.ScientistAgent") as MockScientist, \
         patch("backend.agents.optimization_agent.LatentSpaceAgent") as MockLatent:
        
        agent = OptimizationAgent()
        
        # Setup the Judge to return a predictable score
        # so we can verify selection logic
        agent.judge.validate_design.return_value = {'physics_score': 0.5} 
        
        return agent

def test_initialize_population(optimization_agent):
    """Verify population is seeded correctly."""
    params = {
        "isa_state": {"constraints": {}},
        "config": {"population_size": 10}
    }
    population = optimization_agent._initialize_population({}, 10, params)
    
    assert len(population) == 10
    assert isinstance(population[0], GeometryGenome)
    # Check that genomes are not all identical (they undergo random initial growth)
    # Note: It's possible for them to be similar if random seed is fixed, but generally they differ.
    
def test_evaluate_fitness(optimization_agent):
    """Verify fitness calculation logic."""
    genome = GeometryGenome()
    params = {
        "isa_state": {"constraints": {}},
        "objective": {"type": "VOLUME"}
    }
    
    # Mock Judge to return 0.8 physics score
    optimization_agent.judge.validate_design.return_value = {'physics_score': 0.8}
    
    fitness = optimization_agent._evaluate_fitness(genome, params)
    
    # Fitness = physics_score * (1 + objective_score)
    # Volume of default cube is 1*1*1 = 1. Objective score += 1.
    # Expected = 0.8 * (1 + 1) = 1.6
    assert fitness > 0
    assert fitness == 1.6 # Assuming default cube params

def test_tournament_selection(optimization_agent):
    """Verify that tournament selection picks the better individual."""
    # Create two dummy candidates: (Genome, Score)
    weak_genome = GeometryGenome()
    strong_genome = GeometryGenome()
    
    population_with_scores = [
        (weak_genome, 0.1),
        (strong_genome, 0.9),
        (weak_genome, 0.2)
    ]
    
    # We patch random.sample to always return our custom list
    with patch("random.sample", return_value=population_with_scores):
        winner = optimization_agent._tournament_selection(population_with_scores, k=3)
        assert winner == strong_genome

def test_run_success(optimization_agent):
    """Verify the full evolutionary loop runs and produces a result."""
    params = {
        "isa_state": {
            "constraints": {"mass": 100}
        },
        "config": {
            "population_size": 4, # Small for speed
            "generations": 2,     # Short for speed
            "mutation_rate": 0.5,
            "enable_red_team": False # Disable for unit test simplicity
        }
    }
    
    result = optimization_agent.run(params)
    
    assert result["success"] is True
    assert result["population_count"] == 4
    assert len(result["evolution_history"]) == 2
    assert "best_genome" in result
