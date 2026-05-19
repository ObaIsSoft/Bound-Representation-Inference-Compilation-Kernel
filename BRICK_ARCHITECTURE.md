# BRICK OS — Architecture Specification
> Version: 1.0 | Date: 2026-03-29

---

## Core Principle: LLM Is the Interpreter, Not the Engine

BRICK is a **physics-based multi-agent design compiler**. The LLM has exactly one job at each end of the pipeline:

1. **Intake** — parse natural language intent into a structured `DesignSpec`
2. **Output** — translate structured physics results back into human language

Everything between those two points is **pure physics**: equations, solvers, simulations, database lookups, and validated correlations. No agent in the physics pipeline should ever call an LLM to answer a physics question. Physics answers come from physics — not from language model inference.

---

## Correct Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│  USER                                                           │
│  "Design me a 2kg surveillance drone for arctic operations"     │
└──────────────────────────┬──────────────────────────────────────┘
                           │ natural language
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ConversationalAgent  (LLM layer — ONLY place LLM runs)        │
│  • Extract intent and parameters                                │
│  • Ask clarifying questions for missing values                  │
│  • Validate completeness of DesignSpec                         │
│  • Return structured DesignSpec (Pydantic model)                │
└──────────────────────────┬──────────────────────────────────────┘
                           │ DesignSpec (structured JSON)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ProjectOrchestrator  (LangGraph pipeline — no LLM calls)      │
│                                                                 │
│  Phase 1 → GeometryAgent      OpenCASCADE / Manifold3D          │
│  Phase 2 → MaterialAgent      NIST / MatWeb database lookup     │
│  Phase 3 → StructuralAgent    CalculiX FEA / beam theory        │
│  Phase 4 → ThermalAgent       Nusselt correlations / FiPy       │
│  Phase 5 → FluidAgent         Empirical correlations / OpenFOAM │
│  Phase 6 → ElectronicsAgent   PySpice / KiCad                   │
│  Phase 7 → GNCAgent           T/W physics + CEM trajectory      │
│  Phase 8 → ManufacturingAgent Cost rates DB + BOM               │
│  Phase 9 → ComplianceAgent    Structured rules DB (JSON/YAML)   │
│                                                                 │
│  AgentState blackboard flows between all phases                 │
└──────────────────────────┬──────────────────────────────────────┘
                           │ PhysicsReport (all agent outputs)
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  ExplainabilityAgent  (LLM layer — second and final LLM call)  │
│  • Translate physics results → plain language                   │
│  • Surface key pass/fail decisions with citations               │
│  • Suggest design changes for failures                         │
│  • Generate XAI thought stream for UI                          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ ExplainedReport
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│  Frontend                                                       │
│  • Omniviewport: 3D geometry mesh from GeometryAgent           │
│  • Analysis panels: stress, thermal, fluid maps                 │
│  • BOM + cost table from ManufacturingAgent                     │
│  • Compliance checklist from ComplianceAgent                   │
│  • XAI thought stream from ExplainabilityAgent                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Agent Interface Contract

Every physics agent MUST implement this interface:

```python
class PhysicsAgent(ABC):
    name: str
    version: str
    input_schema: Type[BaseModel]   # Pydantic model — explicit input contract
    output_schema: Type[BaseModel]  # Pydantic model — explicit output contract

    @abstractmethod
    async def run(self, params: BaseModel, state: AgentState) -> BaseModel:
        """
        Execute physics analysis.
        - MUST NOT call LLM
        - MUST validate inputs via input_schema
        - MUST return typed output via output_schema
        - MUST raise, not swallow, unrecoverable errors
        - MAY use async for external solvers (CalculiX, OpenFOAM)
        """

    def explain(self, result: BaseModel) -> str:
        """
        Optional: human-readable one-line summary of the result.
        Used by ExplainabilityAgent as structured context.
        This IS allowed to use templating but NOT an LLM call.
        """
```

---

## Agent Fidelity Levels

Each physics agent should support multiple fidelity levels. Higher fidelity is slower but more accurate:

| Level | Speed | Who uses it | Example |
|---|---|---|---|
| **Analytical** | < 1 ms | UI preview, fast iteration | Beam theory, Nusselt correlations, Drag Cd tables |
| **Surrogate** | 1–100 ms | Design sweeps, optimization loops | FNO trained on FEM data, MLPRegressor on validated dataset |
| **Numerical** | 1 s – 1 min | Per-design validation | scipy/scikit-fem, FiPy finite volume |
| **Full solver** | 1 min – hours | Final sign-off | CalculiX FEA, OpenFOAM CFD |

The agent selects fidelity automatically based on `params.fidelity_level` or `FidelityLevel.AUTO`.

---

## Physics Agent Patterns (from SciML reference)

### Pattern A — Analytical / Empirical Correlations
Use for: fast iteration, preview, well-understood physics
```python
# Structural: Euler-Bernoulli beam theory
sigma_max = (F * L * c) / I

# Fluid: Moody chart correlation for friction factor
f = 0.316 * Re**(-0.25)  # Blasius (turbulent pipe, Re < 100k)

# Thermal: Dittus-Boelter Nusselt correlation
Nu = 0.023 * Re**0.8 * Pr**0.4
```
**Rule:** Only use validated, published correlations with cited sources. Never make up constants.

### Pattern B — Physics-Informed Surrogate (UDE)
Use for: design sweeps across parameter space
```python
# Known physics term (hardcoded — never changes)
def known_physics(x):
    return -k/m * x  # Spring restoring force

# Unknown correction term (learned from simulation data)
correction_net = MLP(...)  # Trained on FEM/CFD outputs

# Total: known physics + learned correction
dxdt = known_physics(x) + correction_net(x)
```
**Rule:** The NN corrects known physics — it doesn't replace it. Train on real solver data, validate against analytical solutions.

### Pattern C — Full Numerical Solver
Use for: final design validation
```python
# CalculiX FEA via subprocess
proc = await asyncio.create_subprocess_exec(
    "ccx", "-i", inp_file,
    stdout=asyncio.subprocess.PIPE,
    stderr=asyncio.subprocess.PIPE
)
stdout, stderr = await proc.communicate()
# Parse .frd output file — never fall back silently if ccx not found
```
**Rule:** Fail loudly if solver not available. Never fall back to a weaker method without telling the user.

### Pattern D — Fourier Neural Operator (FNO) Surrogate
Use for: fast thermal/structural field prediction after offline training
```python
# Generate training data: run FEM solver across parameter space
# Train FNO: maps (geometry, BCs) → solution field
# Inference: < 100ms per query vs minutes for FEM
```
**Rule:** FNO is only as good as its training data. Document training distribution. Refuse inference outside training domain.

---

## Design Knowledge Graph

BRICK's design space is a **knowledge graph** where nodes are design parameters and edges are physics dependencies. Changing one node propagates constraints through connected agents.

```
                    ┌─────────────┐
                    │  mass_kg    │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
       ┌──────────┐  ┌──────────┐  ┌──────────┐
       │ geometry │  │ material │  │  thrust  │
       └────┬─────┘  └────┬─────┘  └────┬─────┘
            │              │              │
            ▼              ▼              ▼
     ┌──────────┐   ┌───────────┐  ┌──────────────┐
     │  stress  │   │ thermal   │  │  T/W ratio   │
     │ analysis │   │ analysis  │  │  GNC stable? │
     └────┬─────┘   └─────┬─────┘  └──────┬───────┘
          │               │                │
          └───────────────┴────────────────┘
                          │
                          ▼
                  ┌───────────────┐
                  │  compliance   │
                  │  pass / fail  │
                  └───────────────┘
```

**Application of ML Knowledge Graph pattern** (inspired by the-palindrome.github.io/ml-knowledge-graph):

- Each **design parameter** is a node (mass, geometry, material, temperature, stress, drag coefficient, etc.)
- Each **physics relationship** is a directed edge (stress depends on geometry + material + load)
- Each **agent** owns a cluster of nodes (StructuralAgent owns: stress_xx, stress_yy, von_mises, safety_factor, displacement)
- **Cascading invalidation**: when mass changes → structural, thermal, GNC, and manufacturing nodes all invalidate → only those agents re-run
- **UI**: the Omniviewport can render this as an interactive force-directed graph. Users can:
  - Click a parameter node to see which agents it affects
  - Double-click an agent cluster to expand its physics outputs
  - Drag to change a parameter value and watch dependent nodes update in real time
  - Color nodes by: pass/fail status, confidence level, fidelity used
  - Size nodes by: sensitivity (how much does changing this node change the final design?)

This turns BRICK from a "run all agents once" pipeline into a **live, incremental constraint propagation system**: change one parameter, only the affected physics re-runs.

### Knowledge Graph Implementation

```python
@dataclass
class DesignNode:
    id: str                    # e.g. "mass_kg", "stress_xx"
    value: Any                 # current value
    unit: str                  # SI unit string
    owner_agent: str           # which agent computed this
    fidelity: FidelityLevel    # how it was computed
    valid: bool = True         # invalidated when deps change
    confidence: float = 1.0    # 0-1, based on fidelity + validation

@dataclass
class DesignEdge:
    source: str                # node id
    target: str                # node id
    relationship: str          # "determines", "constrains", "validates"
    physics_law: str           # citation: "Euler-Bernoulli", "Fourier's law"

class DesignKnowledgeGraph:
    nodes: Dict[str, DesignNode]
    edges: List[DesignEdge]

    def invalidate(self, node_id: str):
        """Propagate invalidation to all dependent nodes"""

    def get_rerun_set(self) -> Set[str]:
        """Return agent names that need to re-run due to invalidated nodes"""

    def to_frontend_json(self) -> dict:
        """Serialize for Three.js / force-directed graph rendering"""
```

---

## What the LLM Is Allowed to Do

| ✅ Allowed | ❌ Not Allowed |
|---|---|
| Parse user intent → structured params | Answer "what is the stress on this beam?" |
| Ask clarifying questions | Generate regulation text for compliance checks |
| Explain physics results in plain language | Choose material properties |
| Suggest design alternatives when physics fails | Run simulations |
| Generate XAI thought narrative | Estimate drag coefficients |
| Synthesize a final design report | Replace a database lookup |

---

## The Compliance Agent Rule: Structured Rules, Not LLM Generation

The compliance agent is the most dangerous place to use an LLM. LLM-generated regulation text:
- Can hallucinate citations
- Can fabricate thresholds (e.g., wrong FAA weight limit)
- Cannot be audited or versioned
- Will differ between runs on the same input

The correct implementation is a **structured rules database**:

```yaml
# backend/config/compliance_rules/FAA_Part107.yaml
regime: AERIAL
standard: FAA Part 107
rules:
  - id: FAR107_12
    name: Maximum Takeoff Weight
    field: mass_kg
    operator: "<="
    threshold: 24.95        # 55 lbs in kg
    violation_msg: "Exceeds FAA Part 107 max MTOW of 55 lbs (24.95 kg)"
    citation: "14 CFR § 107.12"
    official_link: "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107"

  - id: FAR107_51
    name: Maximum Altitude
    field: max_altitude_m
    operator: "<="
    threshold: 121.92       # 400 ft in meters
    violation_msg: "Exceeds FAA Part 107 max altitude of 400 ft (121.92 m)"
    citation: "14 CFR § 107.51(b)"
    official_link: "https://www.ecfr.gov/current/title-14/chapter-I/subchapter-F/part-107"
```

```python
class ComplianceAgent:
    def run(self, params, state):
        rules = self._load_rules(state.regime)  # from YAML, never from LLM
        results = []
        for rule in rules:
            value = state.design_params[rule.field]
            passed = OPERATORS[rule.operator](value, rule.threshold)
            results.append(ComplianceResult(
                rule_id=rule.id,
                passed=passed,
                actual_value=value,
                threshold=rule.threshold,
                citation=rule.citation
            ))
        return results
```

---

## Agent Registry Contract

All agents registered in `agent_registry.py` MUST:

1. Have a corresponding file on disk at the registered path
2. Implement `async def run(self, params, state) -> dict`
3. Not import LLM providers at module level
4. Use `backend.` prefixed imports (not bare module names)
5. Be listed in `backend/config/functional_agents.yaml` with fidelity capabilities

Stub agents that aren't implemented yet must return:
```python
def run(self, params, state):
    raise NotImplementedError(
        f"{self.name} is not yet implemented. "
        f"Required inputs: {self.input_schema.schema()}"
    )
```
Never return `None` silently. Never return mock data silently.

---

## XAI Thought Stream

The thought stream is how BRICK makes the physics pipeline visible to the user in real time. Every agent MUST emit thoughts at key decision points:

```python
# Good: physics decision made transparent
inject_thought(AgentThought(
    agent="StructuralAgent",
    phase="stress_analysis",
    thought="Von Mises stress peak at 187 MPa — safety factor 1.34x vs. required 1.5x. Margin insufficient.",
    confidence=0.95,
    physics_law="von Mises yield criterion: σ_vm = √(½[(σ₁-σ₂)²+(σ₂-σ₃)²+(σ₃-σ₁)²])"
))

# Bad: LLM narrative injected here
inject_thought("The structural analysis suggests potential issues...")
```

Thoughts must be **push-based via WebSocket**, not polled. The current polling implementation is a known deficiency (see BRICK_AUDIT.md A-03).

---

## File Organization (Target State)

```
backend/
  agents/
    base.py                  # PhysicsAgent ABC + AgentState
    structural_agent.py      # CalculiX + beam theory
    thermal_agent.py         # Nusselt + FiPy
    fluid_agent.py           # Correlations + OpenFOAM
    electronics_agent.py     # PySpice + KiCad
    gnc_agent.py             # CEM trajectory
    geometry_agent.py        # OpenCASCADE + Manifold3D
    material_agent.py        # Materials DB lookup
    manufacturing_agent.py   # Cost rates DB
    compliance_agent.py      # Structured rules engine (NO LLM)
    conversational_agent.py  # LLM: intent → DesignSpec (ONLY LLM in pipeline)
    explainability_agent.py  # LLM: PhysicsReport → plain language (ONLY LLM out)
  config/
    compliance_rules/        # YAML rules per regime (FAA, ISO, ASME, FCC...)
    materials_database.yaml  # Material properties from NIST/MatWeb
    physics_constants.yaml   # Physical constants (SI)
  knowledge_graph/
    design_graph.py          # DesignKnowledgeGraph: nodes, edges, invalidation
    serializer.py            # to_frontend_json() for Three.js rendering
  orchestrator.py            # LangGraph pipeline — no LLM calls
  schema.py                  # AgentState, DesignSpec, PhysicsReport (Pydantic)
```

---

## References

- SciML patterns: [matlab-deep-learning/SciML-and-Physics-Informed-Machine-Learning-Examples](https://github.com/matlab-deep-learning/SciML-and-Physics-Informed-Machine-Learning-Examples)
- Knowledge graph visualization pattern: [the-palindrome.github.io/ml-knowledge-graph](https://the-palindrome.github.io/ml-knowledge-graph/)
- Analytical correlations: Incropera & DeWitt (Heat Transfer), White (Fluid Mechanics), Roark's Formulas for Stress and Strain
- FEA: CalculiX documentation, NAFEMS benchmarks
- CFD: OpenFOAM validation cases, NASA Technical Reports
- Compliance: FAA 14 CFR Part 107, ASME Y14.5, ISO 10303, MIL-HDBK-310
