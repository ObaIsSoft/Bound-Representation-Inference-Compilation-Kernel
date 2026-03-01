# DfmAgent Production Implementation Plan

**Agent:** Design for Manufacturing (DfM) Agent  
**Date:** 2026-02-26  
**Phase:** 1.3  
**Effort:** 2-3 weeks  

---

## Current State

**File:** `backend/agents/dfm_agent.py` (76 lines)  
**Status:** 🔴 Stub - Basic rule-based checks only  
**Issues:**
- Hardcoded limits (0.8mm wall for FDM, etc.)
- No feature recognition from CAD
- No process-specific analysis
- No modern DfAM capabilities

---

## Target State: ProductionDfMAgent

### Core Capabilities

| Feature | Method | Status |
|---------|--------|--------|
| **Feature Recognition** | Trimesh geometric analysis | NEW |
| **Manufacturability Scoring** | Boothroyd DFM + AI (2024) | NEW |
| **Process Selection** | Rule-based + ML recommendation | NEW |
| **DfAM Analysis** | Overhang detection, support volume | NEW |
| **CNC Machinability** | Undercut detection, tool access | NEW |
| **Injection Molding** | Draft analysis, wall thickness | NEW |

### Supported Processes

1. **CNC Machining** (Milling, Turning, EDM)
2. **Additive Manufacturing** (FDM, SLA, SLS, SLM)
3. **Injection Molding**
4. **Sheet Metal**
5. **Die Casting**

### Modern Research Integration

| Technique | Research | Implementation |
|-----------|----------|----------------|
| Feature Recognition | Deep learning CAD analysis (2023) | Trimesh + heuristic rules |
| DfAM Rules | DfAM framework (2023) | Overhang, support, warp detection |
| Manufacturability Score | Boothroyd (2011) + AI (2024) | Feature-based difficulty scoring |
| Part Decomposition | Assembly-based redesign (2019) | Split recommendations |

---

## Implementation Phases

### Week 1: Core Feature Recognition

**Day 1-2: Geometry Analysis**
- Wall thickness analysis (3D mesh)
- Aspect ratio detection
- Sharp corner detection
- Thin feature identification

**Day 3-4: Process-Specific Features**
- CNC: Undercuts, tool access, deep holes
- AM: Overhangs, support volume, bridging
- Molding: Draft angles, uniform walls

**Day 5: Integration & Testing**
- Connect to GeometryAgent
- Unit tests for feature detection

### Week 2: Manufacturability Analysis

**Day 1-2: Boothroyd DFM Scoring**
- Feature classification (simple, medium, complex)
- Handling difficulty scores
- Insertion/access difficulty

**Day 3-4: DfAM Analysis**
- Overhang angle detection
- Support volume estimation
- Warp risk (thermal analysis integration)

**Day 5: Process Recommendations**
- Multi-process comparison
- Cost/difficulty trade-offs

### Week 3: Advanced Features (Optional)

- Part decomposition suggestions
- Design rule validation
- ML-based manufacturability prediction

---

## Key Classes

```python
@dataclass
class ManufacturingFeature:
    """Detected manufacturing feature."""
    feature_type: str  # "hole", "slot", "pocket", "boss", "rib"
    dimensions: Dict[str, float]
    difficulty_score: float  # 0-100
    process_compatibility: Dict[str, float]  # {"cnc": 0.9, "fdm": 0.3}

@dataclass
class DfmReport:
    """Complete DfM analysis report."""
    manufacturability_score: float  # 0-100
    features: List[ManufacturingFeature]
    issues: List[DfmIssue]
    recommendations: List[str]
    process_recommendations: List[ProcessRecommendation]

@dataclass
class DfmIssue:
    """Manufacturability issue."""
    severity: str  # "critical", "warning", "info"
    category: str  # "wall_thickness", "overhang", "undercut"
    description: str
    location: Optional[Tuple[float, float, float]]
    suggestion: str
```

---

## External Dependencies

```
trimesh          # 3D geometry analysis
numpy            # Numerical operations
scipy            # Spatial analysis
```

## Configuration Files

```
backend/agents/config/
  ├── dfm_rules.json           # Process-specific rules
  ├── boothroyd_scores.json    # Handling/insertion scores
  └── dfam_guidelines.json     # AM-specific rules
```

---

## Success Criteria

1. ✅ Detect wall thickness violations from 3D mesh
2. ✅ Identify CNC undercuts and tool access issues
3. ✅ Analyze AM overhangs and support requirements
4. ✅ Generate manufacturability score (0-100)
5. ✅ Recommend best manufacturing process
6. ✅ Provide specific design improvement suggestions
7. ✅ All calculations verified with test cases

---

## Research References

1. Boothroyd, G. et al. (2011) - Product Design for Manufacture and Assembly
2. DfAM Framework (2023) - HAL Archives
3. Deep Learning Feature Recognition (2023)
4. Part Decomposition for AM (2019)
