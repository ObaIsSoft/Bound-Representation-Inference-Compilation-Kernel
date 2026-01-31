"""
Manufacturing Standards and Capabilities Configuration.
Externalizes hardcoded values for ToleranceAgent and ManufacturingAgent.
"""

# ISO 286-1 Hole Basis Fits
# Usage: HOLE_BASIS_FITS["H7/g6"] -> {"type": "clearance", ...}
HOLE_BASIS_FITS = {
    "H7/g6": {
        "type": "clearance", 
        "min_clear": 0.010, 
        "max_clear": 0.040, 
        "description": "Locational Clearance - Precision sliding"
    },
    "H7/h6": {
        "type": "transition", 
        "min_clear": 0.000, 
        "max_clear": 0.020, 
        "description": "Locational Transition - Accurate location"
    },
    "H7/p6": {
        "type": "interference", 
        "min_clear": -0.040, 
        "max_clear": -0.010, 
        "description": "Locational Interference - Press fit"
    },
    "H8/f7": {
        "type": "clearance",
        "min_clear": 0.020,
        "max_clear": 0.060,
        "description": "Running Fit - Good lubrication"
    },
    "H11/c11": {
        "type": "clearance",
        "min_clear": 0.100,
        "max_clear": 0.300,
        "description": "Loose Running Fit - Large tolerance"
    }
}

# Manufacturing Process Capabilities (Tolerance in mm)
PROCESS_CAPABILITIES = {
    "CNC": {
        "tolerance_mm": 0.010,
        "surface_finish_ra": 1.6,
        "cost_modifier": 1.5
    },
    "3D_PRINT": {
        "tolerance_mm": 0.200,
        "surface_finish_ra": 12.5,
        "cost_modifier": 1.0,
        "notes": "Anisotropic properties, needs tolerance compensation"
    },
    "INJECTION_MOLD": {
        "tolerance_mm": 0.050,
        "surface_finish_ra": 0.8,
        "cost_modifier": 0.8,  # High setup, low unit
        "notes": "Draft angles required"
    },
    "MANUAL_MACHINING": {
        "tolerance_mm": 0.050,
        "surface_finish_ra": 3.2,
        "cost_modifier": 1.2
    }
}

# Default Fit Selection Strategy based on Nominal Diameter (mm)
def get_recommended_fit(diameter_mm: float) -> str:
    if diameter_mm <= 3:
        return "H7/g6"
    elif diameter_mm <= 50:
        return "H7/h6"
    else:
        return "H8/f7"
