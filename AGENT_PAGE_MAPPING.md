# BRICK OS - Agent to Page Mapping

Based on LangGraph 8-Phase Pipeline and Frontend Structure

---

## Frontend Pages

### 1. `/landing` - Landing Page
**Purpose**: Entry point, project selection, system status

**Agents**: None (Static UI)
- Boot sequence animation
- Project list
- System health dashboard (read-only)
- Navigation to Requirements

---

### 2. `/requirements` - Requirements Gathering Page
**Purpose**: Capture user intent, initial feasibility check

**LangGraph Phase**: Phase 1 - Feasibility

**Primary Agents**:
| Agent | Role | API Endpoint |
|-------|------|--------------|
| **ConversationalAgent** | Natural language understanding | POST /api/chat |
| **DocumentAgent** | Parse uploaded requirements docs | POST /api/agents/document |
| **GeometryEstimator** | Quick feasibility estimate | POST /api/agents/geometry/estimate |
| **CostAgent** | Quick cost estimate | POST /api/cost/estimate |
| **SafetyAgent** | Initial safety screening | POST /api/agents/safety |

**Critics**:
- DesignCritic (initial design validation)

**Panels**:
- Chat interface (ConversationalAgent)
- Document upload (DocumentAgent)
- Quick feasibility dashboard (GeometryEstimator + CostAgent)
- XAI Thought Stream (explainability)

**Workflow**:
1. User describes intent → ConversationalAgent
2. Upload documents → DocumentAgent
3. Quick feasibility → GeometryEstimator
4. Budget check → CostAgent
5. Safety pre-check → SafetyAgent
6. If feasible → Navigate to /planning

---

### 3. `/planning` - Planning Page
**Purpose**: Generate ISA, create execution plan, resource allocation

**LangGraph Phase**: Phase 2 - Planning

**Primary Agents**:
| Agent | Role | API Endpoint |
|-------|------|--------------|
| **PlanningAgent** | Generate execution plan | POST /api/orchestrator/plan |
| **DocumentAgent** | Generate plan documents | POST /api/agents/document |
| **FeasibilityAgent** | Full feasibility analysis | POST /api/agents/feasibility |
| **GenericAgent** | Fallback/utility | POST /api/agents/{name}/run |

**Critics**:
- OracleCritic (plan validation)
- SurrogateCritic (predict outcomes)

**Panels**:
- Plan visualization (ISA tree)
- Resource allocation view
- Approval workflow
- Plan comparison (if multiple options)

**Workflow**:
1. Generate ISA tree → PlanningAgent
2. Create plan document → DocumentAgent
3. Validate plan → OracleCritic
4. Predict outcomes → SurrogateCritic
5. User approval
6. If approved → Navigate to /workspace

---

### 4. `/workspace` - Main Workspace
**Purpose**: Execute plan, design, simulate, manufacture

**LangGraph Phases**: Phases 3-8 (Geometry → Multi-Physics → Manufacturing → Validation → Sourcing → Documentation)

---

#### **Workspace Panels and Agent Mapping**

##### **Panel: Agent Pods** (`agent-pods`)
**Icon**: Boxes
**Status**: Functional ✓

**Purpose**: Visual overview of all 64 agents, their status, and pod organization

**Agents Displayed**:
- All 64 agents in their ISA pods
- Real-time status (idle, running, complete, error)
- Quick actions per agent

**API**: WebSocket `/ws/orchestrator/{project_id}`

---

##### **Panel: Search** (`search`)
**Icon**: Search
**Status**: Functional ✓

**Purpose**: Search across ISA, components, materials, standards

**Agents**:
| Agent | Search Target |
|-------|---------------|
| **ComponentAgent** | Search component catalog |
| **StandardsAgent** | Search standards (ISO, ASTM, NASA) |
| **AssetSourcingAgent** | Search 3D models (NASA 3D, Sketchfab) |
| **MaterialsOracle** | Search material properties |

**API Endpoints**:
- GET /api/standards/search
- GET /api/components/catalog
- POST /api/agents/asset/sourcing

---

##### **Panel: Compile ISA** (`compile`)
**Icon**: FileCheck
**Status**: Planned

**Purpose**: Compile ISA tree to executable form

**Agents**:
| Agent | Role |
|-------|------|
| **OpenSCADAgent** | Compile geometry to OpenSCAD |
| **CodeGenAgent** | Generate fabrication code |
| **ComplianceAgent** | Pre-compile compliance check |

**API**: POST /api/openscad/compile, POST /api/isa/checkout

---

##### **Panel: Run & Debug** (`run-debug`)
**Icon**: Play
**Status**: Planned

**Purpose**: Execute simulation, debugging

**LangGraph Phase**: Phase 4 - Multi-Physics

**Agents**:
| Agent | Simulation Type |
|-------|-----------------|
| **PhysicsAgent** | General physics simulation |
| **StructuralAgent** | Structural analysis (FEA) |
| **FluidAgent** | CFD simulation |
| **ThermalAgent** | Thermal analysis |
| **ElectronicsAgent** | Circuit simulation |
| **ChemistryAgent** | Chemical processes |
| **ControlAgent** | Control systems |
| **GNCAgent** | Guidance/Navigation/Control |
| **ExplainableAgent** | XAI explanations |
| **DiagnosticAgent** | Error diagnosis |

**Critics**:
- PhysicsCritic (validate physics results)
- ComponentCritic (validate components)
- FluidCritic (validate CFD)

**API Endpoints**:
- POST /api/physics/solve
- POST /api/physics/validate
- POST /api/physics/step
- POST /api/chemistry/analyze
- POST /api/agents/{agent}/run

---

##### **Panel: Manufacturing** (`manufacturing`)
**Icon**: Factory
**Status**: Planned

**Purpose**: Manufacturing planning, cost estimation, DFM

**LangGraph Phase**: Phase 5 - Manufacturing

**Agents**:
| Agent | Role |
|-------|------|
| **ManufacturingAgent** | Main manufacturing planner |
| **DFMAgent** | Design for Manufacturing |
| **SlicerAgent** | 3D printing slicer |
| **LatticeSynthesisAgent** | Lattice generation |
| **ToleranceAgent** | Tolerance analysis |
| **OpenscadAgent** | Manufacturing geometry |
| **CostAgent** | Manufacturing cost |
| **SustainabilityAgent** | Carbon footprint |

**Critics**:
- MaterialCritic (material selection)
- Manufacturing constraint checking

**API Endpoints**:
- POST /api/cost/estimate
- POST /api/pricing/set-price
- GET /api/pricing/check
- POST /api/manufacturing/analyze

---

##### **Panel: Compliance** (implied from CompliancePanel.jsx)
**Icon**: FileCheck (variant)
**Status**: Exists

**Purpose**: Standards compliance, safety validation

**Agents**:
| Agent | Role |
|-------|------|
| **ComplianceAgent** | Standards compliance check |
| **SafetyAgent** | Safety validation |
| **StandardsAgent** | Standards lookup |
| **ForensicAgent** | Failure analysis |
| **ZoningAgent** | Regulatory zoning |

**Critics**:
- DesignCritic (design validation)
- SafetyCritic (safety checks)
- ChemistryCritic (chemical safety)
- ElectronicsCritic (electrical safety)

**API Endpoints**:
- POST /api/compliance/check
- POST /api/agents/safety
- GET /api/standards/sources
- GET /api/standards/status

---

##### **Panel: Export** (`export`)
**Icon**: Download
**Status**: Functional ✓

**Purpose**: Export designs, BOM, documentation

**LangGraph Phase**: Phase 8 - Final Documentation

**Agents**:
| Agent | Export Type |
|-------|-------------|
| **DocumentAgent** | Generate PDF docs |
| **CodeGenAgent** | Export code files |
| **GeometryAgent** | Export CAD (STL, STEP) |
| **OpenSCADAgent** | Export .scad files |
| **DevOpsAgent** | Export deployment configs |

**API Endpoints**:
- POST /api/geometry/export/stl
- POST /api/openscad/compile
- POST /api/project/export

---

##### **Panel: Version Control** (`version-control`)
**Icon**: GitBranch
**Status**: Planned

**Purpose**: Design versioning, branching, merging

**Agents**:
| Agent | Role |
|-------|------|
| **FeedbackAgent** | Version comparison feedback |
| **OptimizationAgent** | Optimize across versions |
| **MultiModeAgent** | Multi-modal version handling |

**API Endpoints**:
- POST /api/version/commit
- POST /api/version/branch/create
- POST /api/pods/merge
- POST /api/pods/unmerge

---

##### **Panel: Settings** (`settings`)
**Icon**: Settings
**Status**: Functional ✓

**Purpose**: System configuration, agent tuning

**Agents**: None (System UI)

**API Endpoints**:
- User preferences
- Agent configuration
- Theme settings

---

##### **Panel: Account** (`account`)
**Icon**: User
**Status**: Functional ✓

**Purpose**: User profile, API keys, billing

**Agents**: None (System UI)

---

##### **Panel: Docs** (`docs`)
**Icon**: BookOpen
**Status**: Functional ✓

**Purpose**: Documentation, help, tutorials

**Agents**: None (Static/Search)

---

### 5. Performance Dashboard (Overlay/Modal)
**Purpose**: Real-time system monitoring

**File**: `frontend/src/components/performance/PerformanceDashboard.jsx`

**Agents**: None (Telemetry from all agents)

**WebSocket**: `/ws/telemetry`

**Metrics**:
- CPU/RAM usage
- Agent latency (p95)
- Active agents count
- 64-dot agent grid status

---

## Agent-to-Page Summary Table

| Agent | Primary Page | Panel | Phase |
|-------|-------------|-------|-------|
| ConversationalAgent | /requirements | Chat | 1 |
| DocumentAgent | /requirements, /workspace | Chat, Export | 1, 8 |
| GeometryEstimator | /requirements | Feasibility | 1 |
| CostAgent | /requirements, /workspace | Feasibility, Manufacturing | 1, 5 |
| SafetyAgent | /requirements, /workspace | Feasibility, Compliance | 1, 6 |
| PlanningAgent | /planning | Plan view | 2 |
| FeasibilityAgent | /planning | Feasibility | 1-2 |
| ComponentAgent | /workspace | Search, Run & Debug | 7 |
| StandardsAgent | /workspace | Search, Compliance | 3-8 |
| AssetSourcingAgent | /workspace | Search | 7 |
| PhysicsAgent | /workspace | Run & Debug | 4 |
| StructuralAgent | /workspace | Run & Debug | 4 |
| FluidAgent | /workspace | Run & Debug | 4 |
| ThermalAgent | /workspace | Run & Debug | 4 |
| ElectronicsAgent | /workspace | Run & Debug | 4 |
| ChemistryAgent | /workspace | Run & Debug | 4 |
| ControlAgent | /workspace | Run & Debug | 4 |
| GNCAgent | /workspace | Run & Debug | 4 |
| ManufacturingAgent | /workspace | Manufacturing | 5 |
| DFMAgent | /workspace | Manufacturing | 5 |
| SlicerAgent | /workspace | Manufacturing | 5 |
| LatticeSynthesisAgent | /workspace | Manufacturing | 5 |
| ToleranceAgent | /workspace | Manufacturing | 5 |
| SustainabilityAgent | /workspace | Manufacturing | 5 |
| ComplianceAgent | /workspace | Compliance | 6 |
| ForensicAgent | /workspace | Compliance | 6 |
| OpenSCADAgent | /workspace | Compile, Export | 3, 5, 8 |
| CodeGenAgent | /workspace | Compile, Export | 3, 8 |
| GeometryAgent | /workspace | Compile, Export | 3 |
| ExplainableAgent | /workspace | XAI overlay | All |
| DiagnosticAgent | /workspace | Run & Debug | 4, 6 |

---

## Critics-to-Page Mapping

| Critic | Primary Page | Panel | Purpose |
|--------|-------------|-------|---------|
| ControlCritic | /workspace | Run & Debug | Validate control actions |
| PhysicsCritic | /workspace | Run & Debug | Validate physics results |
| DesignCritic | /requirements, /workspace | Feasibility, Compliance | Design validation |
| OracleCritic | /planning | Plan view | Plan validation |
| SurrogateCritic | /planning | Plan view | Predict outcomes |
| ComponentCritic | /workspace | Run & Debug | Component validation |
| MaterialCritic | /workspace | Manufacturing | Material selection |
| ChemistryCritic | /workspace | Compliance, Run & Debug | Chemical safety |
| ElectronicsCritic | /workspace | Compliance, Run & Debug | Electrical safety |

---

## API Endpoint Organization by Page

### /requirements
```
POST /api/chat                    - ConversationalAgent
POST /api/agents/document         - DocumentAgent
POST /api/agents/geometry/estimate - GeometryEstimator
POST /api/cost/estimate           - CostAgent
POST /api/agents/safety           - SafetyAgent
```

### /planning
```
POST /api/orchestrator/plan       - PlanningAgent
POST /api/agents/feasibility      - FeasibilityAgent
GET  /api/agents/thoughts         - XAI thought stream
```

### /workspace
```
# Search Panel
GET  /api/standards/search        - StandardsAgent
GET  /api/standards/sources       - StandardsAgent
GET  /api/components/catalog      - ComponentAgent
POST /api/agents/asset/sourcing   - AssetSourcingAgent

# Compile Panel
POST /api/openscad/compile        - OpenSCADAgent
POST /api/openscad/compile-stream - OpenSCADAgent
POST /api/isa/checkout            - ISA compilation

# Run & Debug Panel
POST /api/physics/solve           - PhysicsAgent
POST /api/physics/validate        - PhysicsAgent
POST /api/physics/compile         - Multi-physics
POST /api/chemistry/analyze       - ChemistryAgent
POST /api/agents/{name}/run       - Generic agent run

# Manufacturing Panel
POST /api/cost/estimate           - CostAgent
POST /api/pricing/set-price       - CostAgent
GET  /api/pricing/check           - CostAgent
POST /api/analyze/cost            - ManufacturingAgent

# Compliance Panel
POST /api/compliance/check        - ComplianceAgent
GET  /api/standards/status        - StandardsAgent

# Export Panel
POST /api/geometry/export/stl     - GeometryAgent
POST /api/project/export          - DocumentAgent

# Version Control
POST /api/version/commit          - Version control
POST /api/pods/merge              - Pod operations

# Global
WS   /ws/orchestrator/{id}        - Real-time updates
WS   /ws/telemetry                - Performance metrics
GET  /api/agents/thoughts         - XAI explanations
POST /api/agents/select           - Agent selection
```

---

## Implementation Priority

### Phase 1 (MVP) - Current
- ✓ /requirements with ConversationalAgent
- ✓ /workspace with basic panels
- ✓ CostAgent integration
- ✓ Standards search

### Phase 2 (Core)
- /planning with full PlanningAgent
- /workspace Agent Pods panel
- /workspace Run & Debug panel (basic physics)
- XAI thought stream

### Phase 3 (Advanced)
- /workspace Manufacturing panel
- /workspace Compliance panel
- Full 64-agent grid
- Performance dashboard

### Phase 4 (Polish)
- All panels fully functional
- Version control panel
- Export panel with all formats
- Real-time collaboration
