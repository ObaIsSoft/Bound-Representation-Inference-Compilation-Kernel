"""
Component Agent Configuration.
Externalizes constants for ComponentAgent.
"""

COMPONENT_DEFAULTS = {
    "default_limit": 5,
    "default_volatility": 0.0,
    "db_table": "components",
    "weights_path": "data/component_agent_weights.json",
    "install_path": "data/components"
}

TEST_ASSETS = {
    "cube": "test_assets/test_cube.stl",
    "sphere": "test_assets/test_sphere.stl"
}

ATLAS_CONFIG = {
    "default_resolution": 64,
    "chunk_size": 8192
}
