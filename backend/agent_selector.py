"""
Intelligent Agent Selection System

This module implements the intelligent agent selection logic that determines
which physics agents should run based on design requirements, user intent,
environment type, and design parameters.
"""

from typing import List, Dict, Any
import logging

logger = logging.getLogger(__name__)


def select_physics_agents(state: Dict[str, Any]) -> List[str]:
    """
    Intelligently select which physics agents to run based on design requirements.
    
    The system analyzes:
    - User intent keywords
    - Environment type
    - Design parameters
    
    Args:
        state: AgentState dictionary
        
    Returns:
        List of agent names to run (e.g., ["material", "chemistry", "thermal", "physics"])
    """
    intent = state.get("user_intent", "").upper()
    env = state.get("environment", {})
    params = state.get("design_parameters", {})
    env_type = env.get("type", "GROUND")
    
    # ALWAYS run these core agents (every design needs material/chemistry/thermal/physics)
    selected = ["material", "chemistry", "thermal", "physics"]
    
    print(f"DEBUG SELECTOR: Intent={intent}")
    print(f"DEBUG SELECTOR: Core={selected}")
    
    # === ELECTRONICS DETECTION ===
    electronics_keywords = [
        "POWER", "ELECTRIC", "BATTERY", "CIRCUIT", "LED", "MOTOR",
        "SENSOR", "PCB", "VOLTAGE", "CURRENT", "CAPACITOR", "RESISTOR",
        "TRANSISTOR", "MICROCONTROLLER", "ARDUINO", "RASPBERRY PI"
    ]
    if any(kw in intent for kw in electronics_keywords):
        selected.append("electronics")
        logger.info("🔌 Electronics detected: adding ElectronicsAgent")
    
    # === AUTONOMOUS/MOVING SYSTEM DETECTION ===
    autonomous_keywords = [
        "FLY", "DRIVE", "NAVIGATE", "AUTONOMOUS", "DRONE", "CAR",
        "ROBOT", "VEHICLE", "AIRCRAFT", "BOAT", "SUBMARINE", "ROVER",
        "AUTOPILOT", "SELF-DRIVING", "UAV", "UGV", "USV"
    ]
    if any(kw in intent for kw in autonomous_keywords) or env_type in ["AERIAL", "MARINE", "SPACE"]:
        selected.extend(["gnc", "control"])
        logger.info("Autonomous/moving system detected: adding GNC + Control")
    
    # === MANUFACTURING COMPLEXITY DETECTION ===
    num_components = params.get("num_components", 1)
    manufacturing_keywords = ["ASSEMBLY", "MANUFACTURE", "FABRICATE", "PRODUCTION"]
    if num_components > 5 or any(kw in intent for kw in manufacturing_keywords):
        selected.append("dfm")
        logger.info(f"🏭 Manufacturing complexity detected ({num_components} components): adding DFM")
    
    # === REGULATED INDUSTRY DETECTION ===
    regulated_keywords = [
        "MEDICAL", "AEROSPACE", "AIRCRAFT", "FDA", "FAA", "CE",
        "ISO", "CERTIFIED", "COMPLIANT", "REGULATORY", "CLINICAL",
        "AVIATION", "PHARMACEUTICAL"
    ]
    if any(kw in intent for kw in regulated_keywords):
        selected.extend(["compliance", "standards"])
        logger.info("📋 Regulated industry detected: adding Compliance + Standards")
    
    # === COMPLEX SYSTEM DETECTION ===
    # If many agents are needed OR explicitly marked as complex
    if len(selected) > 6 or params.get("complexity", "simple") == "complex":
        selected.append("diagnostic")
        logger.info("🔍 Complex system detected: adding Diagnostic")
    
    # Remove duplicates and return
    unique_selected = list(set(selected))
    logger.info(f"🎯 Final agent selection: {unique_selected}")
    
    return unique_selected


def get_agent_selection_summary(selected_agents: List[str]) -> Dict[str, Any]:
    """
    Generate a human-readable summary of agent selection.
    
    Args:
        selected_agents: List of selected agent names
        
    Returns:
        Dictionary with selection summary
    """
    agent_categories = {
        "core": ["material", "chemistry", "thermal", "physics"],
        "electronics": ["electronics"],
        "autonomous": ["gnc", "control"],
        "manufacturing": ["dfm"],
        "regulatory": ["compliance", "standards"],
        "diagnostics": ["diagnostic"]
    }
    
    summary = {
        "total_agents": len(selected_agents),
        "categories": {},
        "efficiency_gain": f"{100 - (len(selected_agents) / 11 * 100):.1f}%"
    }
    
    for category, agents in agent_categories.items():
        active = [a for a in agents if a in selected_agents]
        if active:
            summary["categories"][category] = active
    
    return summary
