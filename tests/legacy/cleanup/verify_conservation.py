
import logging
import unittest
from backend.agents.critics.ElectronicsCritic import ElectronicsCritic
from backend.agents.critics.ChemistryCritic import ChemistryCritic

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ConservationVerif")

class TestConservationLaws(unittest.TestCase):
    
    def test_electronics_power_creation(self):
        """Test detection of energy creation (First Law Violation)."""
        critic = ElectronicsCritic()
        
        # Scenario: Supply 100W, Demand 200W, Margin +50W (Energy from nothing)
        input_state = {"components": []}
        bad_output = {
            "power_analysis": {
                "hybrid_supply_w": 100.0,
                "hybrid_demand_w": 200.0,
                "margin_w": 50.0 
            },
            "validation_issues": []
        }
        
        # Loop to satisfy min_history requirement (usually 10)
        for _ in range(15):
             critic.observe(input_state, bad_output)
        
        report = critic.analyze()
        
        logger.info(f"Electronics Failures: {report.get('failure_modes', [])}")
        
        # Check for specific violation string
        modes = report.get("failure_modes", [])
        violation_found = any("Power Balance Mismatch" in m for m in modes)
        self.assertTrue(violation_found, f"Critic failed to detect Power Creation. Modes: {modes}")

    def test_chemistry_alchemy(self):
        """Test detection of element transmutation (Alchemy)."""
        critic = ChemistryCritic()
        
        # Scenario: Input is Steel. Output mentions Gold.
        input_state = {
            "materials": ["Steel", "Carbon"],
            "environment_type": "MARINE"
        }
        
        bad_output = {
            "chemical_safe": True,
            "logs": ["Surface passivated by Gold (Au) layer formation."] 
        }
        
        # Loop to satisfy min_history requirement
        for _ in range(15):
            critic.observe(input_state, bad_output)
            
        report = critic.analyze()
        
        logger.info(f"Chemistry Failures: {report.get('failure_modes', [])}")
        
        modes = report.get("failure_modes", [])
        violation_found = any("Alchemy detected" in m for m in modes)
        self.assertTrue(violation_found, f"Critic failed to detect Alchemy. Modes: {modes}")

if __name__ == "__main__":
    unittest.main()
