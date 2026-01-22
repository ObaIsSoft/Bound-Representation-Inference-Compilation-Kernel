
from typing import List, Dict

class AgentProfile:
    def __init__(self, name: str, description: str, active_agents: List[str]):
        self.name = name
        self.description = description
        self.active_agents = active_agents

# Define Standard Profiles
PROFILES = {
    "aerospace": AgentProfile(
        name="Aerospace / Defense",
        description="High-fidelity physics, thermal analysis, and GNC.",
        active_agents=[
            "environment", "geometry", "surrogate_physics", "mass_properties",
            "thermal", "gnc", "structural_load", "material", "verification"
        ]
    ),
    "consumer_electronics": AgentProfile(
        name="Consumer Electronics",
        description="PCB design, enclosure aesthetics, and DfM.",
        active_agents=[
            "designer", "geometry", "electronics", "dfm", 
            "plastic_injection", "thermal", "cost", "visual_validator"
        ]
    ),
    "architecture": AgentProfile(
        name="Architecture / Civil",
        description="Structural integrity, zoning, and MEP routing.",
        active_agents=[
            "environment", "geometry", "zoning", "mep", 
            "structural_load", "cost", "standards"
        ]
    ),
    "rapid_prototyping": AgentProfile(
        name="Rapid Prototyping",
        description="Fast iterations for 3D printing.",
        active_agents=[
            "designer", "geometry", "slicer", "tolerance", "material"
        ]
    ),
    "full_suite": AgentProfile(
        name="Full BRICK OS",
        description="All systems active.",
        active_agents=[] # Special case: All
    )
}

# Essential Agents cannot be disabled
ESSENTIAL_AGENTS = ["documentation", "conversational", "nexus", "environment", "surrogate_physics"]

# Custom Profiles (In-Memory for now, should be DB/File backed)
CUSTOM_PROFILES = {}

def get_profile(key: str) -> Dict[str, any]:
    # Check standard
    profile = PROFILES.get(key)
    if profile:
        return {
            "name": profile.name, 
            "description": profile.description,
            "active_agents": list(set(profile.active_agents + ESSENTIAL_AGENTS)) # Enforce essentials
        }
    
    # Check custom
    c_profile = CUSTOM_PROFILES.get(key)
    if c_profile:
        return c_profile
        
    return None

def create_custom_profile(name: str, agents: List[str]) -> str:
    """Creates a new custom profile and returns its ID."""
    p_id = name.lower().replace(" ", "_")
    
    # Enforce essentials
    final_agents = list(set(agents + ESSENTIAL_AGENTS))
    
    CUSTOM_PROFILES[p_id] = {
        "name": name,
        "description": "User created profile",
        "active_agents": final_agents,
        "is_custom": True
    }
    return p_id

def list_profiles() -> List[Dict[str, str]]:
    standard = [{"id": k, "name": p.name, "is_custom": False} for k, p in PROFILES.items()]
    custom = [{"id": k, "name": p["name"], "is_custom": True} for k, p in CUSTOM_PROFILES.items()]
    return standard + custom

def get_essential_agents():
    return ESSENTIAL_AGENTS
