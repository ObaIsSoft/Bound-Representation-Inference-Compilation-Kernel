# RLM Conversation Flows: Post-Requirements Refinement & Iteration

## Overview

This document maps how the Recursive Language Model (RLM) handles the COMPLETE conversation lifecycle - not just initial requirements gathering, but all subsequent refinements, iterations, and explorations.

---

## Conversation Lifecycle Phases

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONVERSATION LIFECYCLE                                │
└─────────────────────────────────────────────────────────────────────────────┘

Phase 1: INITIAL GATHERING          Phase 2: PLANNING
┌─────────────────────────┐         ┌─────────────────────────┐
│ User: "Design a drone   │         │ Agent generates design  │
│ frame that can carry    │         │ plan with:              │
│ 2kg payload"            │         │ - Manufacturing steps   │
│                         │         │ - Cost breakdown        │
│ RLM:                    │         │ - Timeline              │
│ • DiscoveryNode         │         │                         │
│ • GeometryNode          │         │ User: "Looks good"      │
│ • MaterialNode          │         │ or "That's too expensive"│
│ • CostNode              │         └─────────────────────────┘
└─────────────────────────┘                    │
                                               ▼
Phase 3: WORKSPACE                     Phase 4: ITERATION
┌─────────────────────────┐         ┌─────────────────────────┐
│ - CAD viewing           │         │ User refinements:       │
│ - Manufacturing preview │         │ • "Make it 20% lighter" │
│ - Cost tracking         │         │ • "Use titanium instead"│
│ - Quality checks        │         │ • "Add mounting holes"  │
│                         │         │ • "What about carbon?"  │
│ User: "Can we optimize  │         │                         │
│ the infill pattern?"    │         │ RLM re-runs relevant    │
└─────────────────────────┘         │ nodes with new params   │
                                    └─────────────────────────┘
```

---

## RLM as Universal Router

The key insight: **RLM is not just for initial decomposition - it's a universal router for EVERY user input.**

```python
class ConversationalAgent:
    async def run(self, user_input: str, session_id: str) -> Response:
        # ALWAYS check if recursive reasoning needed
        if self._should_use_rlm(user_input):
            return await self._rlm_execute(user_input, session_id)
        else:
            return await self._single_pass_execute(user_input, session_id)
```

### Input Classification

| Input Type | Example | RLM Complexity | Nodes Invoked |
|------------|---------|----------------|---------------|
| **Greenfield** | "Design a drone frame" | High | Discovery→Geometry→Material→Cost→Safety |
| **Constraint Change** | "Make it lighter" | Medium | Geometry→Cost (re-run) |
| **Material Swap** | "Use titanium" | Medium | Material→Cost→Safety (re-run) |
| **Variant Compare** | "Compare Ti vs Al" | High | [MaterialA + CostA] ∥ [MaterialB + CostB] |
| **Feature Add** | "Add mounting holes" | Medium | Geometry→Cost (delta) |
| **Explanation** | "Why titanium?" | Low | Memory lookup only |
| **Narrowing** | "Only under $500" | Medium | Cost→Material (constrained search) |

---

## Post-Gathering Refinement Patterns

### Pattern 1: Parameter Adjustment

```
User: "Actually, make it lighter"
     ↓
RLM Analysis:
  ┌──────────────────────────────────────┐
  │ Intent: CONSTRAINT_MODIFICATION      │
  │ Target: mass_constraint              │
  │ Current: 2.5 kg                      │
  │ New: "as light as possible"          │
  └──────────────────────────────────────┘
     ↓
Decomposition:
  ┌─────────────────┐     ┌─────────────────┐
  │ GeometryNode    │     │ MaterialNode    │
  │ "Optimize for   │     │ "Suggest lighter│
  │  weight"        │     │  materials"     │
  └────────┬────────┘     └────────┬────────┘
           │                       │
           └───────────┬───────────┘
                       ▼
              ┌─────────────────┐
              │ CostNode        │
              │ "Re-calculate   │
              │  with new specs"│
              └────────┬────────┘
                       ▼
Synthesis: "I can reduce weight to 1.8kg using carbon fiber 
            instead of aluminum. Cost increases by $45."
```

**Context Handling**:
```python
# SCENE context contains accumulated design facts
facts = {
    "mission": "drone_frame",
    "payload_kg": 2.0,
    "previous_material": "aluminum_6061",
    "previous_mass_kg": 2.5,
    "manufacturing_process": "CNC_milling"
}

# New constraint added
new_constraint = {"optimize": "mass", "priority": "high"}

# RLM only re-runs affected nodes
affected_nodes = ["GeometryNode", "MaterialNode", "CostNode"]
```

### Pattern 2: Material/Process Swap

```
User: "What if we use titanium instead?"
     ↓
RLM Analysis:
  ┌──────────────────────────────────────┐
  │ Intent: MATERIAL_VARIANT             │
  │ Target: material                     │
  │ Current: aluminum_6061               │
  │ Proposed: titanium_grade5            │
  └──────────────────────────────────────┘
     ↓
Parallel Execution (Independent checks):
  ┌──────────────────┐  ┌──────────────────┐  ┌──────────────────┐
  │ MaterialNode     │  │ CostNode         │  │ SafetyNode       │
  │ "Verify Ti can   │  │ "Calculate Ti    │  │ "Check Ti        │
  │  handle loads"   │  │  material cost"  │  │  compatibility"  │
  └────────┬─────────┘  └────────┬─────────┘  └────────┬─────────┘
           │                     │                     │
           └─────────────────────┼─────────────────────┘
                                 ▼
Synthesis: "Titanium Grade 5 increases strength-to-weight 
            by 40% but costs 3x more ($127 vs $38). 
            Machining time doubles."
```

**Key Feature**: Previous design is PRESERVED, new variant is BRANCHED

```python
# ConversationManager branches the session
branch_id = await self.conversation_manager.branch_session(
    parent_id=current_session_id,
    variant_name="titanium_option"
)

# Run variant analysis in branch
variant_result = await self.rlm.execute_in_branch(
    branch_id=branch_id,
    modifications={"material": "titanium_grade5"}
)

# Present both options to user
response = {
    "current": original_design,
    "variant": variant_result,
    "comparison": generate_comparison_table(original_design, variant_result)
}
```

### Pattern 3: Multi-Variant Comparison

```
User: "Compare aluminum, titanium, and carbon fiber"
     ↓
RLM Analysis:
  ┌──────────────────────────────────────┐
  │ Intent: COMPARATIVE_ANALYSIS         │
  │ Variants: 3 materials                │
  │ Execution: PARALLEL                  │
  └──────────────────────────────────────┘
     ↓
Fork Execution:
  ┌─────────────────────────────────────────────────────────────────┐
  │                                                                 │
  │  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐        │
  │  │ Branch: Al   │   │ Branch: Ti   │   │ Branch: CF   │        │
  │  │              │   │              │   │              │        │
  │  │ MaterialNode │   │ MaterialNode │   │ MaterialNode │        │
  │  │ CostNode     │   │ CostNode     │   │ CostNode     │        │
  │  │ SafetyNode   │   │ SafetyNode   │   │ SafetyNode   │        │
  │  │              │   │              │   │              │        │
  │  │ Result: $38  │   │ Result: $127 │   │ Result: $89  │        │
  │  │ Mass: 2.5kg  │   │ Mass: 1.8kg  │   │ Mass: 1.4kg  │        │
  │  │ Strength: A  │   │ Strength: A+ │   │ Strength: A- │        │
  │  └──────────────┘   └──────────────┘   └──────────────┘        │
  │                                                                 │
  └─────────────────────────────────────────────────────────────────┘
     ↓
Comparative Synthesis:
  ┌─────────────────────────────────────────────────────────┐
  │ Material   │ Cost   │ Mass   │ Strength │ Machining    │
  │────────────│────────│────────│──────────│──────────────│
  │ Aluminum   │ $38    │ 2.5kg  │ A        │ Easy         │
  │ Titanium   │ $127   │ 1.8kg  │ A+       │ Moderate     │
  │ Carbon Fiber│ $89   │ 1.4kg  │ A-       │ Complex      │
  └─────────────────────────────────────────────────────────┘
  
  Recommendation: "Carbon fiber offers best weight savings at
  moderate cost increase. Titanium overkill for this use case."
```

### Pattern 4: Feature Addition (Delta Changes)

```
User: "Add mounting holes for a camera"
     ↓
RLM Analysis:
  ┌──────────────────────────────────────┐
  │ Intent: FEATURE_ADDITION             │
  │ Target: geometry                     │
  │ Delta: + mounting holes              │
  │ Impact assessment needed             │
  └──────────────────────────────────────┘
     ↓
Delta Analysis:
  ┌─────────────────┐
  │ GeometryNode    │ "Calculate hole placements,
  │ (Delta mode)    │  structural impact"
  └────────┬────────┘
           │
           ▼
  ┌─────────────────┐
  │ CostNode        │ "Add machining operations,
  │ (Delta mode)    │  tool changes"
  └────────┬────────┘
           │
           ▼
Synthesis: "Adding 4x M3 mounting holes increases:
            • Machining time: +15 minutes
            • Material cost: +$0 (negligible)
            • Tool changes: +1 (drill bit)
            Total additional cost: $12"
```

**Delta Mode Optimization**:
```python
class GeometryNode:
    async def execute(self, context, mode="full", previous_result=None):
        if mode == "delta" and previous_result:
            # Only compute changes
            delta_geom = self.calculate_delta(
                base=previous_result.geometry,
                changes=context.new_features
            )
            return DeltaResult(
                base=previous_result,
                changes=delta_geom,
                impact=self.assess_impact(delta_geom)
            )
        else:
            # Full calculation
            return await self.full_geometry_calc(context)
```

### Pattern 5: Constraint Narrowing

```
User: "It needs to be under $100 and done in 3 days"
     ↓
RLM Analysis:
  ┌──────────────────────────────────────┐
  │ Intent: CONSTRAINT_NARROWING         │
  │ Constraints:                         │
  │   • cost < $100                      │
  │   • lead_time < 3 days               │
  │ Current design: $127, 5 days         │
  │ Feasibility: VIOLATION               │
  └──────────────────────────────────────┘
     ↓
Constraint Satisfaction Search:
  ┌──────────────────────────────────────────────────────────┐
  │                                                          │
  │  Iteration 1: Reduce material grade?                     │
  │  ┌─────────────────┐  ┌─────────────────┐               │
  │  │ MaterialNode    │→ │ CostNode        │               │
  │  │ "Try Al 5052    │   │ "Cost: $89"     │ ✓ Cost OK     │
  │  │  vs 6061"       │   │                 │ ✗ Time: 5d    │
  │  └─────────────────┘  └─────────────────┘               │
  │                                                          │
  │  Iteration 2: Change process to sheet metal?             │
  │  ┌─────────────────┐  ┌─────────────────┐               │
  │  │ ProcessNode     │→ │ CostNode        │               │
  │  │ "Sheet metal    │   │ "Cost: $72"     │ ✓ Cost OK     │
  │  │  instead of CNC"│   │                 │ ✓ Time: 2d    │
  │  └─────────────────┘  └─────────────────┘ ✓ VIABLE      │
  │                                                          │
  └──────────────────────────────────────────────────────────┘
     ↓
Synthesis: "To meet your constraints, I suggest:
            • Switch to Aluminum 5052 (vs 6061)
            • Use sheet metal forming (vs CNC)
            • Result: $72, 2-day turnaround"
```

---

## Context Preservation Across Turns

### Turn 1: Initial Request

```yaml
Session: session_abc123
SCENE Context:
  mission: "drone_frame"
  payload_kg: 2.0
  max_budget: 500
  material: null  # Not yet decided
  
EPHEMERAL (Turn 1):
  - Extracted requirements
  - Generated initial design
  
Output: "Designing a drone frame for 2kg payload..."
```

### Turn 2: Refinement

```yaml
Session: session_abc123
SCENE Context (Preserved + Enriched):
  mission: "drone_frame"
  payload_kg: 2.0
  max_budget: 500
  material: "aluminum_6061"      # ← From Turn 1
  mass_kg: 2.5                    # ← From Turn 1
  cost_estimate: 127              # ← From Turn 1
  
EPHEMERAL (Turn 2):
  - User constraint: "make it lighter"
  - Re-run: MaterialNode (suggest CF)
  - Re-run: CostNode (new estimate)
  
Output: "I can reduce weight to 1.8kg using carbon fiber..."
```

### Turn 3: Comparison Request

```yaml
Session: session_abc123
SCENE Context (All previous facts preserved):
  # ... all previous facts ...
  
Branches Created:
  - branch_carbon: {material: "carbon_fiber", cost: 189}
  - branch_titanium: {material: "titanium", cost: 245}
  
EPHEMERAL (Turn 3):
  - Fork 3 variants
  - Parallel execution
  - Comparative synthesis
  
Output: "Comparison: CF saves 44% weight vs Ti saves 28%..."
```

---

## Complex Multi-Turn Example

### Full Conversation Trace

```
┌─────────────────────────────────────────────────────────────────┐
│ TURN 1: Initial Design                                          │
├─────────────────────────────────────────────────────────────────┤
│ USER: "I need a bracket to hold a 5kg motor"                    │
│                                                                 │
│ RLM: [Discovery→Geometry→Material→Cost]                         │
│   • Discovers: industrial, high-vibration                       │
│   • Calculates: 150x100x50mm envelope                           │
│   • Selects: steel_304 (vibration resistance)                   │
│   • Costs: $45, 2-day machining                                 │
│                                                                 │
│ BOT: "I'll design a steel bracket for 5kg load. Cost: $45."     │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 2: Constraint Addition                                     │
├─────────────────────────────────────────────────────────────────┤
│ USER: "It needs to be corrosion resistant too"                  │
│                                                                 │
│ RLM Analysis:                                                   │
│   • Current: steel_304 (good corrosion resistance) ✓           │
│   • No change needed                                            │
│   • BUT: Run SafetyNode to verify for specific environment     │
│                                                                 │
│ RLM: [SafetyNode only]                                          │
│   • Checks: marine vs industrial corrosion                      │
│   • Result: 304 sufficient, 316 overkill                        │
│                                                                 │
│ BOT: "Steel 304 already has good corrosion resistance. For      │
│      marine environments, I'd suggest 316L (+$12)."             │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 3: Design Variation                                        │
├─────────────────────────────────────────────────────────────────┤
│ USER: "What if we made it from aluminum to save weight?"        │
│                                                                 │
│ RLM Analysis:                                                   │
│   • Intent: MATERIAL_VARIANT                                    │
│   • Fork: Create branch_aluminum                                │
│                                                                 │
│ RLM Parallel:                                                   │
│   Branch Steel (preserved):                                     │
│     [Material→Cost] = $45, 3.2kg                                │
│   Branch Aluminum (new):                                        │
│     [Material→Geometry(reduced)→Cost] = $62, 1.8kg              │
│       ↑ Note: Geometry re-run needed (aluminum needs more       │
│         material for same strength)                             │
│                                                                 │
│ BOT: "Aluminum saves 44% weight (1.8kg vs 3.2kg) but costs      │
│      38% more ($62 vs $45). Wall thickness increases from       │
│      3mm to 5mm to maintain strength."                          │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 4: Feature Addition                                        │
├─────────────────────────────────────────────────────────────────┤
│ USER: "Let's go with aluminum. Add mounting holes for heat      │
│      sinks."                                                    │
│                                                                 │
│ RLM Analysis:                                                   │
│   • Select: aluminum branch                                     │
│   • Intent: FEATURE_ADDITION                                    │
│   • Target: geometry (delta)                                    │
│                                                                 │
│ RLM: [GeometryNode (delta mode)→CostNode (delta mode)]          │
│   • Delta: +4x M4 threaded holes                                │
│   • Impact: +$8 machining, +10g weight                          │
│                                                                 │
│ BOT: "Adding 4 heat sink mounting holes: +$8, negligible        │
│      weight. Total: $70, 1.81kg."                               │
└─────────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────────┐
│ TURN 5: Explanation Request                                     │
├─────────────────────────────────────────────────────────────────┤
│ USER: "Why did the aluminum version cost more?"                 │
│                                                                 │
│ RLM Analysis:                                                   │
│   • Intent: EXPLANATION                                         │
│   • No calculation needed!                                      │
│   • Retrieve from SCENE context                                 │
│                                                                 │
│ RLM: Memory Lookup                                              │
│   • Steel: Simple block, easy machining                         │
│   • Aluminum: Complex geometry (thicker walls), more ops        │
│                                                                 │
│ BOT: "Aluminum needs thicker walls (5mm vs 3mm) and more        │
│      complex geometry to match steel's strength. Machining      │
│      time is 40% longer."                                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## RLM State Machine

```
                         ┌─────────────────┐
                         │   USER INPUT    │
                         └────────┬────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │    INTENT CLASSIFICATION    │
                    └─────────────┬───────────────┘
                                  │
            ┌─────────────────────┼─────────────────────┐
            │                     │                     │
            ▼                     ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ NEW_DESIGN      │   │ REFINEMENT      │   │ EXPLANATION     │
   │                 │   │                 │   │                 │
   │ Full RLM        │   │ Partial RLM     │   │ Memory only     │
   │ Decomposition   │   │ (affected nodes)│   │ No recursion    │
   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
            │                     │                     │
            ▼                     ▼                     ▼
   ┌─────────────────┐   ┌─────────────────┐   ┌─────────────────┐
   │ All nodes       │   │ Identify        │   │ Retrieve from   │
   │ Discovery→Cost  │   │ dependencies    │   │ SCENE context   │
   │                 │   │                 │   │                 │
   │ Parallel where  │   │ Re-run affected │   │ Grounded in     │
   │ independent     │   │ nodes only      │   │ actual calcs    │
   └────────┬────────┘   └────────┬────────┘   └────────┬────────┘
            │                     │                     │
            └─────────────────────┼─────────────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │   SYNTHESIS & RESPONSE      │
                    └─────────────┬───────────────┘
                                  │
                                  ▼
                    ┌─────────────────────────────┐
                    │  UPDATE SCENE CONTEXT       │
                    │  (preserve facts)           │
                    └─────────────────────────────┘
```

---

## Implementation: Input Classifier

```python
class InputClassifier:
    """Determines how to route user input through RLM"""
    
    async def classify(self, user_input: str, context: Dict) -> IntentType:
        # Use lightweight LLM for fast classification
        prompt = f"""
        Classify the user input into one of these intents:
        - NEW_DESIGN: Starting fresh design
        - CONSTRAINT_CHANGE: Modifying existing constraints  
        - MATERIAL_VARIANT: Asking about different materials
        - COMPARATIVE: Compare multiple options
        - FEATURE_ADD: Add features to existing design
        - EXPLANATION: "Why", "explain", "how come"
        - NARROWING: Tightening constraints
        
        User input: "{user_input}"
        Current context: {json.dumps(context, indent=2)}
        
        Return JSON: {{"intent": "...", "confidence": 0.0-1.0}}
        """
        
        result = await self.llm.quick_classify(prompt)
        return IntentType(result["intent"])
    
    def get_execution_strategy(self, intent: IntentType) -> ExecutionStrategy:
        strategies = {
            IntentType.NEW_DESIGN: ExecutionStrategy(
                recursive=True,
                nodes=["Discovery", "Geometry", "Material", "Cost", "Safety"],
                parallel_groups=[
                    ["Material", "Safety"],  # Can run together
                    ["Geometry", "Cost"]     # Depends on material
                ]
            ),
            IntentType.CONSTRAINT_CHANGE: ExecutionStrategy(
                recursive=True,
                nodes=["Geometry", "Cost"],  # Only affected nodes
                affected_by="new_constraints"
            ),
            IntentType.EXPLANATION: ExecutionStrategy(
                recursive=False,  # No calculation needed
                use_memory=True   # Retrieve from SCENE
            ),
            # ... more strategies
        }
        return strategies[intent]
```

---

## Key Insights

### 1. SCENE Context is the "Design DNA"

Every turn enriches the SCENE context. It's the single source of truth for:
- What we're designing
- Current constraints
- Decisions made
- Facts established

```python
# SCENE context grows monotonically
session.scene_context = {
    # Turn 1
    "mission": "drone_frame",
    "payload_kg": 2.0,
    
    # Turn 2  
    "corrosion_resistance": "required",
    
    # Turn 3
    "selected_material": "aluminum_6061",
    "mass_kg": 2.5,
    "cost_usd": 127,
    
    # Turn 4
    "features": ["mounting_holes"],
    "final_cost_usd": 135,
    
    # Always preserved
}
```

### 2. EPHEMERAL is Turn-Local Working Memory

Each turn's calculations are isolated:
```python
# Turn 2 calculations
turn_2_ephemeral = {
    "temp_calculations": [...],
    "intermediate_results": [...],
    "discarded": True  # After synthesis
}

# Only synthesized facts promoted to SCENE
```

### 3. Branches Enable "What-If" Without Commitment

```python
# User can explore variants without losing original
branches = {
    "main": original_design,
    "titanium_variant": ti_design,
    "carbon_variant": cf_design
}

# User selects one
await conversation_manager.merge_branch(
    parent_id="main",
    branch_id="carbon_variant"
)
```

### 4. RLM is Adaptive, Not Just Initial

**Before (Linear)**:
- Turn 1: Discovery handles everything
- Turn 2+: Can't handle complexity

**After (Recursive)**:
- Turn 1: Full decomposition
- Turn 2: Targeted refinement
- Turn 3: Parallel comparison
- Turn 4+: Continues adapting

---

## Summary

The RLM handles post-gathering conversations by:

1. **Classifying intent** (change vs compare vs explain)
2. **Selecting strategy** (full RLM vs partial vs memory-only)
3. **Preserving SCENE context** (design DNA accumulates)
4. **Isolating EPHEMERAL** (temporary calculations per turn)
5. **Branching for variants** (explore without commitment)
6. **Delta optimization** (only re-run what changed)
7. **Grounded synthesis** (every claim tied to actual calculations)

This enables **genuine design conversations** - not just Q&A, but collaborative iteration with the agent understanding and building on previous context.
