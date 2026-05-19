"""
BRICK OS Configuration Management

Centralized configuration for physics constants, materials database,
and agent settings to eliminate hardcoding.
"""

import os
import yaml
from typing import Dict, Any, Optional
from pathlib import Path

# Configuration cache
_config_cache: Dict[str, Any] = {}

def load_yaml_config(filename: str) -> Dict[str, Any]:
    """Load YAML configuration file"""
    global _config_cache
    
    if filename in _config_cache:
        return _config_cache[filename]
    
    config_path = Path(__file__).parent / filename
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    _config_cache[filename] = config
    return config

def get_physics_constants() -> Dict[str, Any]:
    """Get physics constants configuration"""
    return load_yaml_config('physics_constants.yaml')

def get_materials_database() -> Dict[str, Any]:
    """Get materials database"""
    return load_yaml_config('materials_database.yaml')

def get_material_properties(material_name: str) -> Optional[Dict[str, Any]]:
    """Get properties for a specific material"""
    db = get_materials_database()
    
    # Search in all categories
    for category, materials in db.items():
        if material_name.lower() in materials:
            return materials[material_name.lower()]
    
    return None

def get_physics_constant(path: str) -> Any:
    """
    Get a specific physics constant by dot-notation path.
    
    Examples:
        get_physics_constant('universal.stefan_boltzmann')
        get_physics_constant('air.specific_heat_cp')
        get_physics_constant('safety_factors.structural.yield')
    """
    config = get_physics_constants()
    keys = path.split('.')
    
    value = config
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise KeyError(f"Configuration path not found: {path}")
    
    return value

def clear_cache():
    """Clear configuration cache (useful for testing)"""
    global _config_cache
    _config_cache.clear()

# Convenience functions for common constants
def stefan_boltzmann() -> float:
    """Stefan-Boltzmann constant (W/m²K⁴)"""
    return get_physics_constant('universal.stefan_boltzmann')

def standard_pressure() -> float:
    """Standard atmospheric pressure (Pa)"""
    return get_physics_constant('standard_conditions.pressure_pa')

def standard_temperature() -> float:
    """Standard temperature (K)"""
    return get_physics_constant('standard_conditions.temperature_k')

def air_density() -> float:
    """Air density at standard conditions (kg/m³)"""
    return get_physics_constant('standard_conditions.density_air')

def gravity() -> float:
    """Gravitational acceleration (m/s²)"""
    return get_physics_constant('universal.gravitational_acceleration')


# =============================================================================
# Functional Agents Configuration
# =============================================================================

def get_functional_agent_config(agent_name: str, key: Optional[str] = None) -> Any:
    """
    Get configuration for functional agents.
    
    Args:
        agent_name: Name of the agent (e.g., 'component_agent', 'gnc_agent')
        key: Optional specific key within agent config (e.g., 'default_limit')
    
    Returns:
        Configuration dict or specific value if key provided
        
    Examples:
        get_functional_agent_config('component_agent', 'default_limit')  # Returns 5
        get_functional_agent_config('gnc_agent')  # Returns full GNC config dict
    """
    config = load_yaml_config('functional_agents.yaml')
    
    if agent_name not in config:
        raise KeyError(f"Agent configuration not found: {agent_name}")
    
    agent_config = config[agent_name]
    
    if key is None:
        return agent_config
    
    keys = key.split('.')
    value = agent_config
    for k in keys:
        if isinstance(value, dict) and k in value:
            value = value[k]
        else:
            raise KeyError(f"Configuration key not found: {agent_name}.{key}")
    
    return value


# Convenience functions for common agent configs
def component_config(key: Optional[str] = None) -> Any:
    """Get ComponentAgent configuration"""
    return get_functional_agent_config('component_agent', key)

def gnc_config(key: Optional[str] = None) -> Any:
    """Get GncAgent configuration"""
    return get_functional_agent_config('gnc_agent', key)

def manufacturing_config(key: Optional[str] = None) -> Any:
    """Get ManufacturingAgent configuration"""
    return get_functional_agent_config('manufacturing_agent', key)

def mesh_quality_config(key: Optional[str] = None) -> Any:
    """Get MeshQualityChecker configuration"""
    return get_functional_agent_config('mesh_quality', key)

def mitigation_config(key: Optional[str] = None) -> Any:
    """Get MitigationAgent configuration"""
    return get_functional_agent_config('mitigation_agent', key)
