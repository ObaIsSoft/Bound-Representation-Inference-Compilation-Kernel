"""
BRICK OS - Unified Agent API Router

This module aggregates all agent API endpoints into a single router.
Include this in your main FastAPI application.

Example:
    from fastapi import FastAPI
    from backend.api.agent_routes import router as agent_router
    
    app = FastAPI()
    app.include_router(agent_router, prefix="/api/v1")
"""

from fastapi import APIRouter, HTTPException
import logging

logger = logging.getLogger(__name__)

# Create main router
router = APIRouter(prefix="/agents", tags=["agents"])

# ============================================================================
# Import agent routers (if available)
# ============================================================================

# Core agents with API endpoints
agent_modules = [
    ("backend.agents.codegen_agent", "codegen", "/codegen"),
    ("backend.agents.devops_agent", "devops", "/devops"),
    ("backend.agents.multi_mode_agent", "multimode", "/multimode"),
    ("backend.agents.nexus_agent", "nexus", "/nexus"),
    ("backend.agents.review_agent", "review", "/review"),
    ("backend.agents.surrogate_agent", "surrogate", "/surrogate"),
    ("backend.agents.geometry_agent", "geometry", "/geometry"),
    ("backend.agents.electronics_agent", "electronics", "/electronics"),
    ("backend.agents.document_agent", "document", "/document"),
    ("backend.agents.thermal_agent", "thermal", "/thermal"),
    ("backend.agents.fluid_agent", "fluid", "/fluid"),
    ("backend.agents.structural_agent", "structural", "/structural"),
    ("backend.agents.manufacturing_agent", "manufacturing", "/manufacturing"),
    ("backend.agents.material_agent", "material", "/material"),
    ("backend.agents.dfm_agent", "dfm", "/dfm"),
    ("backend.agents.doctor_agent", "doctor", "/health"),
    ("backend.agents.control_agent", "control", "/control"),
    ("backend.agents.gnc_agent", "gnc", "/gnc"),
    ("backend.agents.safety_agent", "safety", "/safety"),
    ("backend.agents.compliance_agent", "compliance", "/compliance"),
    ("backend.agents.forensic_agent", "forensic", "/forensic"),
    ("backend.agents.verification_agent", "verification", "/verification"),
    ("backend.agents.chemistry_agent", "chemistry", "/chemistry"),
    ("backend.agents.physics_agent", "physics", "/physics"),
    ("backend.agents.cost_agent", "cost", "/cost"),
]

# Try to import and register each agent router
registered_agents = []
failed_agents = []

for module_name, agent_name, prefix in agent_modules:
    try:
        module = __import__(module_name, fromlist=["router"])
        if hasattr(module, "router") and module.router is not None:
            router.include_router(module.router, prefix=prefix)
            registered_agents.append({"name": agent_name, "prefix": prefix, "status": "registered"})
            logger.info(f"Registered agent router: {agent_name} at {prefix}")
        else:
            failed_agents.append({"name": agent_name, "reason": "No router found"})
    except Exception as e:
        failed_agents.append({"name": agent_name, "reason": str(e)})
        logger.warning(f"Failed to register agent {agent_name}: {e}")

# ============================================================================
# Agent Registry Endpoints
# ============================================================================

@router.get("/registry")
async def get_agent_registry():
    """Get registry of all available agents"""
    return {
        "status": "success",
        "registered_agents": registered_agents,
        "failed_agents": failed_agents,
        "total_available": len(registered_agents),
        "total_failed": len(failed_agents)
    }

@router.get("/registry/{agent_name}")
async def get_agent_info(agent_name: str):
    """Get information about a specific agent"""
    for agent in registered_agents:
        if agent["name"] == agent_name:
            return {
                "status": "success",
                "agent": agent,
                "endpoints": f"/api/v1/agents{agent['prefix']}"
            }
    raise HTTPException(status_code=404, detail=f"Agent '{agent_name}' not found")


# ============================================================================
# Health Check Endpoint
# ============================================================================

@router.get("/health")
async def agents_health_check():
    """Health check for all agent services"""
    return {
        "status": "healthy",
        "registered_agents_count": len(registered_agents),
        "agents": [a["name"] for a in registered_agents]
    }


# ============================================================================
# Quick Access Endpoints
# ============================================================================

@router.post("/run/{agent_name}")
async def run_agent(agent_name: str, params: dict):
    """Run a specific agent by name"""
    # Map agent names to their run functions
    agent_runners = {
        "codegen": "backend.agents.codegen_agent",
        "devops": "backend.agents.devops_agent",
        "multimode": "backend.agents.multi_mode_agent",
        "nexus": "backend.agents.nexus_agent",
        "review": "backend.agents.review_agent",
        "surrogate": "backend.agents.surrogate_agent",
        "geometry": "backend.agents.geometry_agent",
        "electronics": "backend.agents.electronics_agent",
        "document": "backend.agents.document_agent",
        "thermal": "backend.agents.thermal_agent",
        "fluid": "backend.agents.fluid_agent",
        "structural": "backend.agents.structural_agent",
        "manufacturing": "backend.agents.manufacturing_agent",
        "material": "backend.agents.material_agent",
        "dfm": "backend.agents.dfm_agent",
        "doctor": "backend.agents.doctor_agent",
        "control": "backend.agents.control_agent",
        "gnc": "backend.agents.gnc_agent",
        "safety": "backend.agents.safety_agent",
        "compliance": "backend.agents.compliance_agent",
        "forensic": "backend.agents.forensic_agent",
        "verification": "backend.agents.verification_agent",
        "chemistry": "backend.agents.chemistry_agent",
        "physics": "backend.agents.physics_agent",
    }
    
    if agent_name not in agent_runners:
        raise HTTPException(status_code=404, detail=f"Unknown agent: {agent_name}")
    
    try:
        module = __import__(agent_runners[agent_name], fromlist=["Agent"])

        # Standard naming: {AgentName}Agent, then bare Agent fallback
        agent_class = None
        for class_name in [f"{agent_name.title()}Agent", "Agent"]:
            if hasattr(module, class_name):
                agent_class = getattr(module, class_name)
                break
        
        if agent_class is None:
            raise HTTPException(status_code=500, detail=f"Agent class not found in module")
        
        # Instantiate and run
        agent = agent_class()
        
        # Check if run is async
        import asyncio
        if asyncio.iscoroutinefunction(agent.run):
            result = await agent.run(params)
        else:
            result = agent.run(params)
        
        return {
            "status": "success",
            "agent": agent_name,
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent execution failed: {str(e)}")
