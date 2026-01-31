"""
Validation Thresholds and Scoring Weights.
Externalizes constants for VisualValidatorAgent, VerificationAgent, and others.
"""

# Visual Validation Weights (Deductions from 1.0)
VISUAL_SCORING = {
    "watertightness_penalty": 0.3,
    "inverted_normals_penalty": 0.5,
    "degenerate_face_penalty": 0.1,
    "unlit_scene_penalty": 0.2,
    "min_face_area": 1e-9,  # Threshold for degenerate faces
    "min_quality_score": 0.7  # Passing score
}

# VMK Safety Verification Thresholds
SAFETY_THRESHOLDS = {
    "collision_margin_mm": -0.1,  # Negative SDF means inside material
    "rapid_clearance_mm": 5.0,    # Minimum height above stock for rapids
    "max_force_newtons": 1000.0,  # Generic force limit
    "tool_deflection_limit_mm": 0.05
}

# Verification Pass Criteria
VERIFICATION_CRITERIA = {
    "min_pass_rate": 1.0,  # All critical tests must pass
    "performance_tolerance": 0.05,  # +/- 5% deviation allowed from baseline
    "required_tests": [
        "requirement_satisfaction",
        "geometry_validity",
        "physics_feasibility"
    ]
}
