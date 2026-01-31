"""
Monitoring and Health Check Configuration.
Externalizes constants for DoctorAgent.
"""

HEALTH_CONFIG = {
    "failure_probability": 0.05,  # Simulation of random agent failure
    "check_interval_seconds": 60,
    "resource_thresholds": {
        "cpu_percent": 90.0,
        "memory_percent": 85.0
    }
}
