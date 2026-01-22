# Self-Evolving Agent Architecture for BRICK OS

## The Big Picture

**Your question was spot-on**: No, the critic should NOT only watch physics agents. EVERY agent in BRICK OS can benefit from self-evolution.

Here's **why** and **how**:

---

## 1. The Problem with Current Approach

The current `CriticAgent` is **too specialized**:
- Requires gate values (only gated hybrid agents have these)
- Monitors prediction vs ground truth (not all agents have ground truth)
- Physics-centric metrics (gate alignment, turbulence regime)

**BRICK OS has 57+ agents** across domains:
- **Physics** (PhysicsAgent, ThermalAgent, StructuralAgent)
- **Design** (DesignerAgent, OptimizationAgent, GeometryAgent)
- **Manufacturing** (ManufacturingAgent, SlicerAgent, DfmAgent)
- **Machine Learning** (SurrogateAgent, TrainingAgent)
- **Analysis** (GncAgent, MassPropertiesAgent, MitigationAgent)
- **Documentation** (DocumentAgent, DiagnosticAgent)

Each has **different evolution needs**:
| Agent Type | What Can Evolve | How to Detect Degradation |
|-----------|----------------|--------------------------|
| **SurrogateAgent** | Neural weights | Compare predictions to PhysicsAgent |
| **DesignerAgent** | Color preferences, style heuristics | Track user acceptance rate |
| **OptimizationAgent** | Learning rate, mutation strategy | Monitor convergence speed |
| **GncAgent** | Control gains, stability margins | Validate against test scenarios |
| **PhysicsAgent** | Gate mechanism, turbulence models | Gate alignment + accuracy |

---

## 2. The Solution: Generalized Critic Framework

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                     BRICK OS ORCHESTRATOR                           │
│                                                                     │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │ Physics  │  │ Designer │  │Surrogate │  │   GNC    │  ...     │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent   │          │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬─────┘          │
│       │             │              │             │                 │
│       ▼             ▼              ▼             ▼                 │
│  ┌─────────────────────────────────────────────────────────┐      │
│  │            OBSERVATION LAYER                            │      │
│  │  (Captures: input, output, timestamp, metadata)         │      │
│  └─────────────────────────────────────────────────────────┘      │
└─────────────────────────────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                   META-CRITIC ORCHESTRATOR                          │
│                                                                     │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐       │
│  │ PhysicsCritic  │  │ DesignCritic   │  │SurrogateCritic │       │
│  │                │  │                │  │                │       │
│  │ • Gate align   │  │ • Diversity    │  │ • Pred drift   │       │
│  │ • Conservation │  │ • User prefs   │  │ • Uncertainty  │       │
│  │ • Regime detect│  │ • Convergence  │  │ • Active learn │       │
│  └────────┬───────┘  └────────┬───────┘  └────────┬───────┘       │
│           │                   │                   │                │
│           └───────────────────┼───────────────────┘                │
│                               ▼                                    │
│                    ┌──────────────────────┐                        │
│                    │  Conflict Detection  │                        │
│                    │  • Ping-pong loops   │                        │
│                    │  • Silent failures   │                        │
│                    │  • Cascading errors  │                        │
│                    └──────────┬───────────┘                        │
│                               ▼                                    │
│                    ┌──────────────────────┐                        │
│                    │  Evolution Queue     │                        │
│                    │  • Prioritized       │                        │
│                    │  • Safety-checked    │                        │
│                    │  • User-approved     │                        │
│                    └──────────┬───────────┘                        │
└────────────────────────────────┼────────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      TRAINING AGENT                                 │
│                   (Evolution Executor)                              │
│                                                                     │
│  Strategy 1: RETRAIN_SURROGATE                                     │
│    → Full retraining with new data                                 │
│                                                                     │
│  Strategy 2: TUNE_HEURISTIC                                        │
│    → Adjust hyperparameters (learning rate, thresholds)            │
│                                                                     │
│  Strategy 3: UPDATE_PRIORS                                         │
│    → Bayesian update (user preferences, material priors)           │
│                                                                     │
│  Strategy 4: EXPAND_RULES                                          │
│    → Add new rules (material compatibility, design constraints)    │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Which Agents Should Self-Evolve?

### Tier 1: HIGH Priority (Immediate ROI)

#### **SurrogateAgent** ⭐ BEST CANDIDATE
- **Why**: Already has neural network that needs retraining
- **What Evolves**: Model weights when drift detected
- **How**: `SurrogateCritic` compares predictions to `PhysicsAgent`
- **Trigger**: Prediction error > 15% over 100 samples
- **Evidence**: Your `SurrogateAgent` already has `validate_prediction()` method!

#### **PhysicsAgent** Sub-Agents (Thermal, Structural)
- **Why**: Operating conditions change (new materials, environments)
- **What Evolves**: Heuristic coefficients, regime boundaries
- **How**: `PhysicsCritic` monitors gate alignment + conservation laws
- **Trigger**: Gate misalignment < 70% or conservation violation

#### **OptimizationAgent**
- **Why**: Different design problems need different strategies
- **What Evolves**: Learning rate, mutation strength, convergence criteria
- **How**: Track iterations-to-convergence, solution quality trends
- **Trigger**: Convergence speed degrades by >30%

---

### Tier 2: MEDIUM Priority (User Experience)

#### **DesignerAgent**
- **Why**: User aesthetic preferences change over time
- **What Evolves**: Color harmony weights, style parameters
- **How**: `DesignCritic` tracks user acceptance/rejection rates
- **Trigger**: Acceptance rate < 70% over 20 designs

#### **GncAgent** (Control Systems)
- **Why**: Different vehicle masses/geometries need different gains
- **What Evolves**: PID/LQR gains, stability margins
- **How**: Monitor control effort, overshoot, settling time
- **Trigger**: Stability margin < safety threshold

---

### Tier 3: LOW Priority (Deterministic or Low Variability)

#### **DocumentAgent**, **DiagnosticAgent**
- **Why**: Mostly rule-based, less benefit from evolution
- **When**: Only evolve if template quality degrades

---

## 4. Cross-Agent Self-Evolution Examples

### Example 1: Surrogate-Physics Drift

**Scenario**: Environment changes (new materials introduced)

```
1. User designs with new composite material
2. SurrogateAgent predicts (fast, but trained on old materials)
3. PhysicsAgent simulates (slow, but accurate with new material)
4. SurrogateCritic observes 25% prediction error
5. Flags for retraining
6. TrainingAgent retrains surrogate with new material data
7. Next prediction: error drops to 5%
```

**Self-Evolution**: Surrogate automatically adapts to new domain

---

### Example 2: Designer-User Preference Learning

**Scenario**: User consistently rejects neon colors

```
1. DesignerAgent generates palette: #FF00FF (neon purple)
2. User rejects design
3. DesignCritic logs: "High saturation rejected"
4. After 10 rejections of high-saturation palettes:
5. DesignCritic detects pattern
6. TrainingAgent updates DesignerAgent priors:
   - saturation_max: 0.9 → 0.7
7. Future designs: more muted tones
8. Acceptance rate: 40% → 75%
```

**Self-Evolution**: Designer learns user preferences implicitly

---

### Example 3: Optimization Strategy Adaptation

**Scenario**: OptimizationAgent inefficient for large designs

```
1. Small design (10 params): converges in 15 iterations
2. Large design (100 params): stuck after 50 iterations
3. OptimizationCritic detects: "High-dimensional design not converging"
4. Recommends strategy change: Gradient descent → Genetic algorithm
5. TrainingAgent updates OptimizationAgent strategy map:
   - param_count > 50 → use genetic algorithm
6. Next large design: converges in 30 iterations
```

**Self-Evolution**: Optimization strategy adapts to problem complexity

---

### Example 4: Cross-Agent Conflict (Meta-Critic)

**Scenario**: Designer vs Structural ping-pong

```
1. DesignerAgent: "Use thin walls (aesthetic)"
2. StructuralAgent: "REJECT - too weak"
3. OptimizationAgent: "Increase thickness"
4. DesignerAgent: "REJECT - ugly"
5. [Loop 3 times]
6. MetaCriticOrchestrator detects ping-pong
7. Analyzes:
   - DesignerAgent aesthetic weight = 0.9 (too high)
   - StructuralAgent safety margin = 3.0 (too conservative)
8. Proposes mediation:
   - Reduce DesignerAgent aesthetic weight: 0.9 → 0.7
   - OR relax StructuralAgent margin: 3.0 → 2.0
9. User approves structural relaxation
10. Conflict resolved
```

**Self-Evolution**: System-level conflict resolution

---

## 5. Safety Constraints

### Critical vs Non-Critical

| Safety Level | Agents | Auto-Evolution |
|-------------|--------|----------------|
| **CRITICAL** | GncAgent, StructuralAgent, ComplianceAgent | ❌ User approval required |
| **STANDARD** | PhysicsAgent sub-agents | ⚠️ Auto if performance < 80% |
| **LOW** | DesignerAgent, OptimizationAgent | ✅ Auto-evolve freely |

### Rollback Capability

Every agent evolution creates a version snapshot:
```python
agent_registry = {
    "surrogate": {
        "v1.0": <original_model>,
        "v1.1": <after_retraining_2024_01_15>,
        "v1.2": <current>  # Active
    }
}
```

**If evolution degrades performance**:
```python
# One-click rollback
rollback_agent("surrogate", to_version="v1.1")
```

---

## 6. Your Questions Answered

### Q: "Is it only physics agents it will be watching?"

**A: No!** The generalized `BaseCriticAgent` can watch **any agent**:
- Physics agents (with specialized `PhysicsCritic`)
- Design agents (with `DesignCritic`)
- ML agents (with `SurrogateCritic`)
- Manufacturing agents (with `ManufacturingCritic` - future)

### Q: "Can it be better?"

**A: Yes!** Current improvements:
1. **Generic base class** - works for any agent without modification
2. **Multiple evolution strategies** - not just retraining
3. **Cross-agent coordination** - prevents cascading failures
4. **User preference learning** - implicit feedback loops
5. **Safety constraints** - critical agents require approval

### Q: "Should other agents be self-evolving?"

**A: Absolutely!** Priority:

**High Priority** (implement first):
- ✅ SurrogateAgent - clear drift metric
- ✅ PhysicsAgent sub-agents - gate alignment
- ✅ OptimizationAgent - convergence tracking

**Medium Priority** (phase 2):
- ⚠️ DesignerAgent - user preference learning
- ⚠️ GncAgent - stability margins
- ⚠️ ManufacturingAgent - learn from production outcomes

**Low Priority** (nice to have):
- 📝 DocumentAgent - template quality
- 📝 DiagnosticAgent - rule updates

---

## 7. Next Steps

**Immediate Actions:**
1. Review [implementation_plan.md](file:///Users/obafemi/.gemini/antigravity/brain/58157100-3470-4334-846d-2fcf86eedc73/implementation_plan.md)
2. Decide which agents to prioritize
3. Approve/modify safety constraints

**Development Sequence:**
1. Create `BaseCriticAgent` (foundation for all)
2. Implement `SurrogateCritic` (highest ROI)
3. Add `PhysicsCritic` for thermal/structural
4. Build `MetaCriticOrchestrator` for conflict detection
5. Create user dashboard for transparency

---

## Key Insight

**Self-evolution is not about making agents "smarter" - it's about making them ADAPTIVE.**

- SurrogateAgent adapts to new materials
- DesignerAgent adapts to user taste
- OptimizationAgent adapts to problem complexity
- The SYSTEM adapts to eliminate agent conflicts

This is the foundation for a **truly autonomous design system** that improves itself over time.

Would you like me to start implementation with `SurrogateAgent` evolution (lowest risk, highest ROI)?
