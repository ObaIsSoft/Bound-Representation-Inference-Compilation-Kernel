"""
Physics Engine Module for BRICK OS

This module provides the foundational physics kernel that grounds all operations
in BRICK OS. Physics is not optional - it's the core of the system.

Usage:
    from backend.physics import get_physics_kernel
    
    physics = get_physics_kernel()
    result = physics.calculate("mechanics", "stress", force=1000, area=0.01)
"""

from backend.physics.kernel import get_physics_kernel

__all__ = ["get_physics_kernel"]
