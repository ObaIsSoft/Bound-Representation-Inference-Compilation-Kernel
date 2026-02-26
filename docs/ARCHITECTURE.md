# BRICK OS - System Architecture

> **Single Source of Truth for System Design**  
> **Version:** 2026.02.26

---

## Table of Contents

1. [High-Level Architecture](#high-level-architecture)
2. [Orchestration Layer](#orchestration-layer)
3. [Agent Layer](#agent-layer)
4. [Physics Kernel](#physics-kernel)
5. [Data Flow](#data-flow)
6. [Technology Stack](#technology-stack)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Frontend Layer                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────┐  │
│  │   Next.js    │  │ Three.js/    │  │   React Three        │  │
│  │     UI       │  │   WebGL      │  │      Fiber           │  │
│  └──────┬───────┘  └──────┬───────┘  └──────────┬───────────┘  │
└─────────┼─────────────────┼─────────────────────┼───────────────┘
          │                 │                     │
          └─────────────────┼─────────────────────┘
                            │ HTTP/WebSocket
┌───────────────────────────▼─────────────────────────────────────┐
│                        API Layer                                 │
│                    FastAPI Server                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  /api/orchestrator/*  │  /api/agents/*  │  /ws/orchestrator│  │
│  └──────────────────────────────────────────────────────────┘  │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Orchestration Layer                           │
│                      LangGraph Workflow                          │
│                                                                  │
│  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │
│  │ Phase 1│→ │ Phase 2│→ │ Phase 3│→ │ Phase 4│→ │ Phase 5│   │
│  │Feasibility│ │Planning│ │Geometry│ │ Physics│ │Manufact│   │
│  └────────┘  └────────┘  └────────┘  └────────┘  └────────┘   │
│                                                     ↓           │
│                              ┌────────┐  ┌────────┐            │
│                              │ Phase 6│→ │ Phase 7│→ Phase 8    │
│                              │Validate│  │Document│            │
│                              └────────┘  └────────┘            │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                      Agent Layer                                 │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Physics   │  │  Geometry   │  │  Structural │              │
│  │    Agent    │  │    Agent    │  │    Agent    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Thermal   │  │   Material  │  │Manufacturing│              │
│  │    Agent    │  │    Agent    │  │    Agent    │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │Electronics  │  │    Cost     │  │  Critic     │              │
│  │    Agent    │  │    Agent    │  │  Agents     │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
│                                                                  │
│              + 70+ More Specialized Agents                       │
└───────────────────────────┬─────────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────────┐
│                    Physics Kernel                                │
│              (UnifiedPhysicsKernel)                              │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐           │
│  │Mechanics │ │  Fluids  │ │  Thermo  │ │  Nuclear │           │
│  └──────────┘ └──────────┘ └──────────┘ └──────────┘           │
│                                                                  │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐                        │
│  │  Electromagnetics   │  │  Quantum │                        │
│  └─────────────────────┘  └──────────┘                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## Orchestration Layer

**File:** `backend/orchestrator.py` (1,318 lines)

### 8-Phase Workflow

```python
from langgraph.graph import StateGraph, END

# Phase 1: Feasibility Check
stt_node → dreamer_node → geometry_estimator_node → cost_quick_estimate_node

# Phase 2: Planning & Review
document_plan_node → review_plan_node

# Phase 3: Geometry Kernel
designer_node → ldp_node → geometry_node → mass_properties_node → structural_node

# Phase 4: Multi-Physics
physics_mega_node (routes to Thermal, Electronics, etc.)

# Phase 5: Manufacturing
slicer_node → lattice_synthesis_node → manufacturing_node

# Phase 6: Validation
training_node → validation_node → optimization_node

# Phase 7: Sourcing & Deployment
asset_sourcing_node → component_node → devops_node → swarm_node → doctor_node → pvc_node

# Phase 8: Final Documentation
final_document_node → final_review_node
```

### Conditional Gates

```python
from backend.conditional_gates import (
    check_feasibility,      # Phase 1 → 2
    check_user_approval,    # Phase 2 → 3 (pauses for UI)
    check_fluid_needed,     # Phase 3 → 4
    check_manufacturing_type, # Phase 4 → 5
    check_lattice_needed,   # Phase 5
    check_validation        # Phase 6 → 3 (loop back) or 7
)
```

---

## Agent Layer

### Registry Pattern

```python
from backend.agent_registry import registry

# Lazy loading - agents instantiated on first request
agent = registry.get_agent("StructuralAgent")

# All agents wrapped with XAI for observability
```

### Agent Categories

| Tier | Category | Count | Examples |
|------|----------|-------|----------|
| 1 | Core Physics | 4 | ProductionStructuralAgent, ProductionThermalAgent |
| 2 | Manufacturing | 3 | ManufacturingAgent, CostAgent, DfmAgent |
| 3 | Design | 5 | DesignerAgent, OptimizationAgent, TemplateDesignAgent |
| 4 | Systems | 5 | ElectronicsAgent, ControlAgent, GncAgent, SafetyAgent |
| 5 | Critics | 10 | PhysicsCritic, DesignCritic, MaterialCritic |
| 6 | Support | 20 | DocumentAgent, ValidationAgent, STTAgent |

### Multi-Fidelity Physics

```
User Request
    ↓
[Analytical] < 1ms (Euler-Bernoulli)
    ↓ confidence < threshold
[Surrogate] < 10ms (Neural Operator)
    ↓ confidence < threshold
[ROM] < 100ms (POD/SVD)
    ↓ confidence < threshold
[FEA] Minutes (CalculiX)
    ↓
High-Fidelity Result
```

---

## Physics Kernel

**Access:** `backend/agents/physics_agent.py`

### Domains

1. **Mechanics**: Force balance, moment calculations
2. **Fluids**: Bernoulli, Navier-Stokes (simplified), drag calculations
3. **Thermodynamics**: Conduction, convection, radiation
4. **Nuclear**: Fusion Lawson criterion, fission kinetics
5. **Electromagnetism**: Basic circuit analysis
6. **Quantum**: Tunneling checks (nano-scale)

### Constants

```python
physics.get_constant("g")        # 9.80665 m/s²
physics.get_constant("c")        # 299792458 m/s
physics.get_constant("R")        # 8.314462618 J/(mol·K)
```

---

## Data Flow

### Design Request Flow

```
1. User: "Titanium drone under 2kg"
   ↓
2. ConversationalAgent (RLM)
   - Decomposes requirements
   - Queries material database
   - Validates feasibility
   ↓
3. Orchestrator (8-phase pipeline)
   - Phase 1: Feasibility check
   - Phase 2: Planning
   - Phase 3: Geometry generation
   - Phase 4: Physics validation
   - Phase 5: Manufacturing analysis
   - Phase 6: Validation
   - Phase 7: Component sourcing
   - Phase 8: Documentation
   ↓
4. Output: Design report + CAD files + Cost estimate
```

### State Management

```python
class AgentState(TypedDict):
    user_intent: str
    design_parameters: Dict
    geometry_tree: List
    physics_results: Dict
    cost_estimate: Dict
    validation_status: str
    errors: List
```

---

## Technology Stack

### Backend
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | FastAPI | API server |
| Orchestration | LangGraph | Workflow engine |
| Physics | CalculiX, SciPy | FEA + numerics |
| ML | PyTorch | Neural surrogates |
| CAD | OpenCASCADE, Manifold3D | Geometry kernel |
| Database | Supabase (PostgreSQL) | Data persistence |
| Cache | Redis (optional) | Session + results |

### Frontend
| Component | Technology | Purpose |
|-----------|------------|---------|
| Framework | Next.js 14 | React framework |
| Language | TypeScript | Type safety |
| Styling | TailwindCSS | UI styling |
| 3D | Three.js, React Three Fiber | 3D visualization |
| State | React Context | State management |

### DevOps
| Component | Technology | Purpose |
|-----------|------------|---------|
| Container | Docker | Deployment |
| Orchestration | Docker Compose | Local dev |
| CI/CD | GitHub Actions (planned) | Automation |

---

## Key Files

### Core
- `backend/main.py` - FastAPI entry point
- `backend/orchestrator.py` - LangGraph workflow
- `backend/agent_registry.py` - Agent registry
- `backend/schema.py` - State definitions

### Agents
- `backend/agents/*_agent.py` - Individual agents
- `backend/agents/critics/` - Critic agents

### Services
- `backend/services/supabase_service.py` - Database
- `backend/services/pricing_service.py` - Pricing APIs
- `backend/services/standards_service.py` - Engineering standards

### Configuration
- `backend/config/agent_config.py` - Agent settings
- `backend/config/validation_thresholds.py` - Critic thresholds

---

## Consolidated From

- docs/BIBLE.md
- docs/MASTER_ARCHITECTURE.md
- docs/SELF_EVOLUTION_ARCHITECTURE.md
- docs/PHYSICS_FIRST_ARCHITECTURE.md
- docs/CRITIC_AGENT.md
- docs/CRITIC_WORKFLOW.md

---

*This document consolidates 6 architecture-related files into a single reference.*
