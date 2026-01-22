"""
BRICK OS Specialized Critics Module

This module contains specialized critic agents that monitor domain-specific agents and oracles:

- ChemistryCritic: Monitors ChemistryAgent for corrosion, safety, compatibility
- ElectronicsCritic: Monitors ElectronicsAgent for power balance, shorts, DRC
- MaterialCritic: Monitors MaterialAgent for degradation, mass accuracy, DB coverage
- ComponentCritic: Monitors ComponentAgent for selection quality, installations
- OracleCritic: Monitors ALL oracle systems for fundamental calculation correctness

Usage:
    from agents.critics import ChemistryCritic, OracleCritic
    
    # Initialize
    chemistry_critic = ChemistryCritic(window_size=100)
    
    # Observe agent behavior
    chemistry_critic.observe(
        input_state={"materials": ["Aluminum"], "environment_type": "MARINE"},
        chemistry_output=agent.run(...)
    )
    
    # Generate report
    report = chemistry_critic.analyze()
    
    # Check if evolution needed
    should_evolve, reason, strategy = chemistry_critic.should_evolve()
"""

from .ChemistryCritic import ChemistryCritic
from .ElectronicsCritic import ElectronicsCritic
from .MaterialCritic import MaterialCritic
from .ComponentCritic import ComponentCritic
from .OracleCritic import OracleCritic
from .PhysicsCritic import PhysicsCritic

__all__ = [
    'ChemistryCritic',
    'ElectronicsCritic',
    'MaterialCritic',
    'ComponentCritic',
    'OracleCritic',
    'PhysicsCritic'
]

# Version
__version__ = '1.0.0'
