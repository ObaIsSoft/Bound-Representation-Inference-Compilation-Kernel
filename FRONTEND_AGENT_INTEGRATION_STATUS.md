# Frontend-Agent Integration Status

**Date**: 2026-02-09
**Phase**: Week 3 Complete → Week 4 Planning

---

## Current Implementation Status

### ✅ Fully Functional (MVP Ready)

| Component | Backend | Frontend | Integration |
|-----------|---------|----------|-------------|
| **CostAgent** | ✅ Pricing service + DB | ✅ Quick estimate UI | ✅ `/api/cost/estimate` |
| **StandardsAgent** | ✅ NIST/NASA connectors | ✅ Basic search | ✅ `/api/standards/search` |
| **ControlCritic** | ✅ DB thresholds | ❌ No UI yet | ⚠️ Needs panel |
| **ManufacturingAgent** | ✅ DB rates | ❌ No UI yet | ⚠️ Needs panel |
| **SafetyAgent** | ✅ Material props | ❌ No UI yet | ⚠️ Needs panel |
| **SustainabilityAgent** | ✅ Carbon data | ❌ No UI yet | ⚠️ Needs panel |

### 🔄 Backend Ready, Frontend Placeholder

| Agent | Backend Status | Panel Status | Action Needed |
|-------|---------------|--------------|---------------|
| StandardsAgent | ✅ Ready | Placeholder | Add search UI |
| ComponentAgent | ✅ Ready | Not built | Create ComponentPanel |
| PhysicsAgent | ✅ Ready | Not built | Create RunDebugPanel |
| OpenSCADAgent | ✅ Ready | Not built | Create CompilePanel |
| ManufacturingAgent | ✅ Ready | Placeholder | Add cost/DFM UI |

### 📋 Frontend Placeholder Only

All panels in `frontend/src/components/panels/` are placeholder divs:
```jsx
// Current state of all panels
export default function XxxPanel({ width }) {
    const { theme } = useTheme();
    return (
        <div style={{ width, backgroundColor: theme.colors.bg.secondary }} className="h-full" />
    );
}
```

---

## Panel-by-Panel Implementation Plan

### Panel: SearchPanel (High Priority)
**Current**: Empty div
**Backend**: ✅ Standards API ready
**Needed**:
```typescript
interface SearchPanelProps {
    onSelectStandard: (standard: Standard) => void;
    onSelectComponent: (component: Component) => void;
}

// Features:
// - Tab: Standards (NIST/NASA/ISO search)
// - Tab: Components (catalog search)
// - Tab: Materials (properties lookup)
// - Tab: Assets (3D models)
```

**API Endpoints to Wire**:
- `GET /api/standards/search?q={query}`
- `GET /api/standards/sources`
- `POST /api/agents/asset/sourcing`

---

### Panel: ManufacturingPanel (High Priority)
**Current**: Empty div
**Backend**: ✅ ManufacturingAgent, CostAgent, SustainabilityAgent ready
**Needed**:
```typescript
interface ManufacturingPanelProps {
    projectId: string;
    onCostUpdate: (cost: CostEstimate) => void;
}

// Features:
// - Cost estimation form
// - Manufacturing process selection
// - DFM feedback
// - Carbon footprint display
// - Sustainability rating
```

**API Endpoints to Wire**:
- `POST /api/cost/estimate`
- `POST /api/manufacturing/analyze` (needs creation)
- `POST /api/sustainability/analyze` (needs creation)

---

### Panel: RunDebugPanel (Medium Priority)
**Current**: Not built
**Backend**: ✅ PhysicsAgent, ControlAgent ready
**Needed**:
```typescript
interface RunDebugPanelProps {
    projectId: string;
    onSimulationComplete: (results: SimResults) => void;
}

// Features:
// - Simulation type selector
// - Physics parameters
// - Run/Stop controls
// - Results visualization
// - ControlCritic feedback
```

**API Endpoints to Wire**:
- `POST /api/physics/solve`
- `POST /api/physics/validate`
- `POST /api/agents/control/run`

---

### Panel: AgentPodsPanel (Medium Priority)
**Current**: Empty div
**Backend**: ✅ All 64 agents available
**Needed**:
```typescript
interface AgentPodsPanelProps {
    projectId: string;
    onAgentSelect: (agent: Agent) => void;
}

// Features:
// - 64-dot grid visualization
// - Agent status indicators
// - Pod grouping
// - Quick actions per agent
```

**WebSocket to Wire**:
- `WS /ws/orchestrator/{project_id}`

---

### Panel: CompilePanel (Medium Priority)
**Current**: Empty div
**Backend**: ✅ OpenSCADAgent ready
**Needed**:
```typescript
interface CompilePanelProps {
    projectId: string;
    isa: ISATree;
}

// Features:
// - ISA tree view
// - Compile options
// - Export format selection
// - Preview generated code
```

**API Endpoints to Wire**:
- `POST /api/openscad/compile`
- `POST /api/openscad/compile-stream`
- `POST /api/isa/checkout`

---

### Panel: ExportPanel (Low Priority - Basic)
**Current**: Empty div
**Backend**: Partial
**Needed**: Basic file export UI

---

### Panel: CompliancePanel (Medium Priority)
**Current**: Empty div
**Backend**: ✅ SafetyAgent, StandardsAgent ready
**Needed**:
```typescript
interface CompliancePanelProps {
    projectId: string;
}

// Features:
// - Safety checklist
// - Standards compliance
// - Forensic analysis (future)
```

**API Endpoints to Wire**:
- `POST /api/compliance/check`
- `POST /api/agents/safety`

---

## Recommended Next Steps

### Week 4 Focus: Manufacturing Panel
1. **Create ManufacturingPanel with CostAgent integration**
   - Cost estimation form (material, mass, complexity)
   - Real-time cost display
   - Region selector (affects hourly rates)

2. **Add SustainabilityAgent to ManufacturingPanel**
   - Carbon footprint calculator
   - Material comparison
   - Sustainability rating (A/B/C)

3. **Create API endpoints if missing**
   - `POST /api/manufacturing/estimate`
   - `POST /api/sustainability/analyze`

### Week 5 Focus: Search Panel
1. **Standards search with live results**
2. **Component catalog browser**
3. **Material properties lookup**

### Week 6 Focus: Run & Debug Panel
1. **Physics simulation controls**
2. **ControlCritic integration**
3. **Real-time results display**

---

## Wireframe: Manufacturing Panel

```
┌──────────────────────────────────────────────┐
│  Manufacturing                               │
│  ══════════════════════════════════════════  │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Cost Estimation                         │ │
│  │ ─────────────────────────────────────── │ │
│  │ Material: [Aluminum 6061-T6 ▼]         │ │
│  │ Mass:     [5.0 kg          ]           │ │
│  │ Complexity: [Moderate    ▼]            │ │
│  │ Region:   [USA/Global    ▼]            │ │
│  │                                         │ │
│  │ [Calculate Cost]                        │ │
│  │                                         │ │
│  │ ┌─────────────────────────────────────┐ │ │
│  │ │ Estimated Cost: $XX.XX USD          │ │ │
│  │ │ Lead Time: X days                   │ │ │
│  │ │ Confidence: 70%                     │ │ │
│  │ └─────────────────────────────────────┘ │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Sustainability                          │ │
│  │ ─────────────────────────────────────── │ │
│  │ CO2 Emissions: XX kg                    │ │
│  │ Rating: [A] 🟢                          │ │
│  │ Data Source: Ecoinvent 3.8              │ │
│  └─────────────────────────────────────────┘ │
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Manufacturing Options                   │ │
│  │ ─────────────────────────────────────── │ │
│  │ [ ] CNC Milling    ($75/hr)            │ │
│  │ [ ] 3D Printing    ($50/hr)            │ │
│  │ [ ] Sheet Metal    ($65/hr)            │ │
│  └─────────────────────────────────────────┘ │
│                                              │
└──────────────────────────────────────────────┘
```

---

## Wireframe: Search Panel

```
┌──────────────────────────────────────────────┐
│  Search                                      │
│  ══════════════════════════════════════════  │
│                                              │
│  [🔍 Search standards, components...       ] │
│                                              │
│  [Standards] [Components] [Materials] [Assets]│
│                                              │
│  ┌─────────────────────────────────────────┐ │
│  │ Results                                 │ │
│  │ ─────────────────────────────────────── │ │
│  │ 🔖 FIPS 140-3                           │ │
│  │    Security Requirements for...         │ │
│  │    [View PDF]                           │ │
│  │                                         │ │
│  │ 🔖 NASA-STD-5005                        │ │
│  │    Strength Analysis Requirements       │ │
│  │    [View Metadata] [Download PDF]       │ │
│  │                                         │ │
│  │ ...                                     │ │
│  └─────────────────────────────────────────┘ │
│                                              │
└──────────────────────────────────────────────┘
```
