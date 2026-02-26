# BRICK OS - Agents Guide

> **Single Source of Truth for All Agents**  
> **Version:** 2026.02.26  
> **Status:** Production-Ready (85%)

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Agent Registry](#agent-registry)
3. [Tier 1: Physics Agents](#tier-1-physics-agents)
4. [Tier 2: Manufacturing & Cost](#tier-2-manufacturing--cost)
5. [Tier 3: Design & Optimization](#tier-3-design--optimization)
6. [Tier 4: Systems & Control](#tier-4-systems--control)
7. [Tier 5: Critics & Oracles](#tier-5-critics--oracles)
8. [Production Status](#production-status)
9. [Architecture](#architecture)

---

## Quick Start

```python
import asyncio
from backend.agent_registry import registry

async def main():
    # Get any agent from the registry (lazy-loaded)
    structural = registry.get_agent("StructuralAgent")
    material = registry.get_agent("MaterialAgent")
    
    # Use the agent
    mat_data = await material.get_material("aluminum_6061_t6", temperature_c=20.0)
    result = await structural.analyze_beam_simple(
        length=1.0, width=0.05, height=0.1,
        elastic_modulus=mat_data["properties"]["elastic_modulus"]["value"],
        load=1000
    )
    print(f"Max stress: {result['max_stress']/1e6:.2f} MPa")

asyncio.run(main())
```

---

## Agent Registry

**File:** `backend/agent_registry.py`

- **100 agents registered** in `AVAILABLE_AGENTS`
- **Lazy-loading**: Agents instantiated only when requested
- **XAI wrapper**: Auto-injected for observability
- **Case-insensitive lookup**

```python
from backend.agent_registry import registry

# List all available agents
registry.list_known_agents()  # Returns 100 agent names

# Check if agent exists (without loading)
registry.is_agent_available("StructuralAgent")  # True

# Get agent instance (lazy-loaded)
agent = registry.get_agent("structural")  # Case-insensitive
```

---

## Tier 1: Physics Agents

### ProductionStructuralAgent
**File:** `backend/agents/structural_agent.py` (1,311 lines)

**Capabilities:**
- Multi-fidelity analysis (Analytical → Surrogate → ROM → FEA)
- CalculiX FEA integration with FRD parsing
- Physics-Informed Neural Operator (FNO) architecture
- POD Reduced Order Model (99% energy retention)
- Fatigue analysis (rainflow counting)
- Buckling eigenvalue analysis

**Usage:**
```python
from backend.agents.structural_agent import ProductionStructuralAgent, FidelityLevel

agent = ProductionStructuralAgent()
result = await agent.analyze(
    geometry_type="cantilever_beam",
    dimensions={"length": 1.0, "width": 0.1, "height": 0.1},
    material={"elastic_modulus": 70e9, "poisson_ratio": 0.33},
    loads={"tip_load": 1000},
    fidelity=FidelityLevel.AUTO  # or ANALYTICAL, ROM, FEA
)
```

**Status:** ✅ Production Ready  
**Dependencies:** CalculiX (optional), PyTorch (optional), Gmsh (optional)

---

### ProductionThermalAgent
**File:** `backend/agents/thermal_agent.py` (755 lines)

**Capabilities:**
- 3D steady/transient conduction (FiPy FVM)
- Convection correlations (Churchill-Chu, Gnielinski, Dittus-Boelter)
- CoolProp fluid properties integration
- Thermal-structural coupling
- View factor calculations for radiation

**Status:** ✅ Production Ready  
**Dependencies:** CoolProp (optional), FiPy (optional)

---

### ProductionGeometryAgent
**File:** `backend/agents/geometry_agent.py` (978 lines)

**Capabilities:**
- Multi-kernel CAD (OpenCASCADE + Manifold3D)
- STEP import/export (ISO 10303)
- Mesh generation (Gmsh integration)
- Feature-based parametric modeling
- GD&T engine (ASME Y14.5)

**Status:** ⚠️ Partial (Fillet/chamfer not in Manifold3D kernel)

---

### ProductionMaterialAgent
**File:** `backend/agents/material_agent.py` (758 lines)

**Capabilities:**
- 29 certified materials (NIST/ASTM)
- Temperature-dependent properties (polynomial models)
- Uncertainty quantification with confidence intervals
- Data provenance tracking (NIST_CERTIFIED, ASTM_CERTIFIED, etc.)
- Emergency fallback (3 materials: Al 6061-T6, Steel 4140, Ti-6Al-4V)

**Status:** ✅ Production Ready

---

## Tier 2: Manufacturing & Cost

### ManufacturingAgent
**File:** `backend/agents/manufacturing_agent.py` (491 lines)

**Capabilities:**
- Database-driven manufacturing rates
- Bill of Materials (BOM) generation
- Cost breakdown (material + machining + setup)
- Recursive sub-assembly aggregation
- Toolpath verification via VMK

**Status:** ✅ Production Ready

---

### CostAgent
**File:** `backend/agents/cost_agent.py` (359 lines)

**Capabilities:**
- Real-time pricing via Metals-API / Yahoo Finance
- Currency conversion (real-time rates)
- No hardcoded costs - fails fast if pricing unavailable
- Market dynamic adjustment via surrogate

**Status:** ✅ Production Ready

---

## Tier 3: Design & Optimization

### OptimizationAgent
**File:** `backend/agents/optimization_agent.py` (296 lines)

**Capabilities:**
- Evolutionary algorithm (population-based)
- Red Team adversarial testing
- MultiPhysicsPINN validation
- ScientistAgent integration (pattern discovery)
- Latent space morphing

**Status:** ⚠️ Partial (Scientist integration stubbed)

---

### DocumentAgent
**File:** `backend/agents/document_agent.py` (432 lines)

**Capabilities:**
- Orchestrates 5+ specialized agents
- Generates comprehensive design plans
- LLM synthesis (if available)
- Error handling without mock data
- PDF generation

**Status:** ✅ Production Ready

---

## Tier 4: Systems & Control

### PhysicsAgent
**File:** `backend/agents/physics_agent.py` (895 lines)

**Capabilities:**
- 6-DOF rigid body dynamics (scipy.odeint)
- Student-Teacher surrogate routing
- SDF collision detection
- Multi-regime support (AERIAL, MARINE, GROUND)
- Nuclear physics (fusion/fission) via kernel

**Status:** ✅ Production Ready

---

### ElectronicsAgent
**File:** `backend/agents/electronics_agent.py` (500+ lines)

**Capabilities:**
- Scale-aware (MEGA to NANO)
- Power network analysis (sources vs loads)
- Hybrid correction with learned parameters
- Chassis short detection

**Status:** ✅ Production Ready

---

### SafetyAgent
**File:** `backend/agents/safety_agent.py` (229 lines)

**Capabilities:**
- Material-specific stress limits
- Temperature safety margins
- Application-type specific checks (aerospace, medical, industrial)
- Database-driven safety factors

**Status:** ✅ Production Ready

---

## Tier 5: Critics & Oracles

### Critic System
**Files:** `backend/agents/critics/`

**Implemented Critics (10):**
1. BaseCriticAgent (ABC)
2. PhysicsCritic ✅
3. DesignCritic ✅
4. MaterialCritic
5. FluidCritic
6. OptimizationCritic
7. ComponentCritic
8. ChemistryCritic
9. ElectronicsCritic
10. GeometryCritic

**MetaCriticOrchestrator:**
- Weighted conflict resolution
- PhysicsCritic weight: 10.0 (highest)
- Hard stop on FAIL, soft stop on WARN accumulation

**Status:** ✅ Framework Ready

---

## Production Status

| Metric | Value |
|--------|-------|
| Total Agents Registered | 100 |
| Production Ready | 32 (32%) |
| Needs Attention | 59 (59%) |
| Critical Issues | 0 |

**Production Ready Agents:**
- ProductionStructuralAgent
- ProductionThermalAgent
- ProductionMaterialAgent
- CostAgent
- ManufacturingAgent
- DocumentAgent
- PhysicsAgent
- SafetyAgent
- ElectronicsAgent
- All 10 Critic Agents

**Known Limitations:**
1. Neural surrogates: Architecture ready, needs training data
2. CalculiX: External dependency (not pip-installable)
3. PyTorch: Optional for neural features

---

## Architecture

### Multi-Fidelity Physics
```
User Request
    ↓
[Analytical] < 1ms (Beam theory)
    ↓ (if needed)
[Surrogate] < 10ms (Neural Operator)
    ↓ (if needed)
[ROM] < 100ms (POD Reduced Order)
    ↓ (if needed)
[FEA] Minutes (CalculiX)
```

### Critic Loop
```
Agent Execution
    ↓
Critic Observation
    ↓
Analysis (Performance, Gate Alignment)
    ↓
Should Evolve? → Retraining Trigger
```

---

## See Also

- [API Documentation](../API.md) - Endpoint reference
- [Architecture Guide](./ARCHITECTURE.md) - System design
- [Testing Guide](../tests/README.md) - Validation suite

---

*Consolidated from: AGENTS_PRODUCTION_SUMMARY.md, AGENT_IMPLEMENTATION_RESEARCH.md, ALL_AGENTS_MASTER_SPEC.md, AGENT_AUDIT_REPORT.md, AGENTS_README.md*
