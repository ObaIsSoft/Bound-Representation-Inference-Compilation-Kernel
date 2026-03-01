"""
BRICK OS Engineering Physics Module

Production-grade physics calculations for hardware design.

This module provides:
- Advanced fluid dynamics (FIX-101, FIX-102)
- Structural mechanics with stress concentration (FIX-103)
- Failure criteria (Von Mises, Tresca) (FIX-104)
- Comprehensive safety factors (FIX-105)
- Fatigue analysis with S-N curves (FIX-106)
- Buckling analysis (Euler, Johnson) (FIX-107)
- Thermal stress analysis (FIX-108, FIX-109)
"""

# Fluid dynamics
from .fluids_advanced import (
    AdvancedFluids,
    DragCorrelation,
    calculate_drag_coefficient,
    calculate_reynolds_number
)

# Structural mechanics
from .structures_advanced import (
    AdvancedStructures,
    StressState,
    FailureResult,
    von_mises_stress,
    calculate_safety_factor
)

# Fatigue analysis
from .fatigue import (
    FatigueAnalyzer,
    MaterialSNCurve,
    calculate_fatigue_life
)

# Thermal stress
from .thermal_stress import (
    ThermalStressAnalyzer,
    MaterialThermalProps,
    thermal_stress_simple
)

__version__ = "1.0.0"

__all__ = [
    # Fluid dynamics
    "AdvancedFluids",
    "DragCorrelation",
    "calculate_drag_coefficient",
    "calculate_reynolds_number",
    
    # Structural mechanics
    "AdvancedStructures",
    "StressState",
    "FailureResult",
    "von_mises_stress",
    "calculate_safety_factor",
    
    # Fatigue
    "FatigueAnalyzer",
    "MaterialSNCurve",
    "calculate_fatigue_life",
    
    # Thermal
    "ThermalStressAnalyzer",
    "MaterialThermalProps",
    "thermal_stress_simple"
]
