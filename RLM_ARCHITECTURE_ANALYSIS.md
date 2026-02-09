# RLM Deep Architecture Analysis
## Global Memory, Discovery Manager & Context System Integration

---

## Current Architecture Overview

### 1. GlobalMemoryBank (Experience Ledger)
**File**: `backend/core/global_memory.py`

**Purpose**: Cross-session learning - remembers failures, optimizations, insights

```python
class GlobalMemoryBank:
    - Stores: failure modes, optimization shortcuts, cross-domain insights
    - Persistence: brain/memory_ledger.json
    - Methods: add_experience(), query(context_tag)
```

**RLM Integration**: 
- Recursive nodes can query GlobalMemory before executing
- Failed sub-tasks are logged with context for future avoidance
- Success patterns are cached for similar future queries

### 2. ConversationManager (Session State)
**File**: `backend/conversation_state.py`

**Purpose**: Manages multi-turn conversation state with branching/merging

```python
class ConversationManager:
    - conversations: Dict[str, ConversationState]
    - Features: branch_session(), merge_session()
    - Persistence: JSON file storage
```

**RLM Integration**:
- Each recursive sub-task can be a "micro-session" within the main session
- Sub-task results are stored in conversation.gathered_requirements
- Branches can explore alternative designs in parallel

### 3. DiscoveryManager (Requirements Phase)
**File**: `backend/agents/conversational_agent.py` (lines 76-330)

**Purpose**: Guides users through structured requirements gathering

```python
class DiscoveryManager:
    - Session-scoped state isolation
    - Configurable: min/max turns, required fields
    - Methods: check_completeness(), update_session()
    - Schema-driven extraction (mission, environment, constraints)
```

**RLM Integration**:
- DiscoveryManager becomes ONE recursive node type among many
- RLM can spawn Discovery sub-tasks for complex requirements
- Existing DiscoveryManager can be reused as-is

### 4. EnhancedContextManager (Hierarchical Memory)
**File**: `backend/context_manager.py`

**Purpose**: Production-grade context with 5-level hierarchy

```python
ContextScope:
    EPHEMERAL      # This turn only
    SCENE          # This design session
    CAMPAIGN       # This project
    AGENT_IDENTITY # Persistent personality
    UNIVERSAL      # Shared physics knowledge

class EnhancedContextManager:
    - HierarchicalSummarizer (L0-L3 compression)
    - VectorMemoryIndex (semantic retrieval)
    - build_prompt_context() with token budgeting
```

**RLM Integration** (CRITICAL):
- **EPHEMERAL**: Sub-task working memory (temporary calculations)
- **SCENE**: Recursive session context (accumulated facts)
- **CAMPAIGN**: Cross-design patterns (learned heuristics)
- Sub-tasks write to EPHEMERAL, synthesis promotes to SCENE

### 5. GlobalAgentRegistry (Agent Spawning)
**File**: `backend/agent_registry.py`

**Purpose**: Lazy-loading registry for 60+ agents

```python
GlobalAgentRegistry:
    - AVAILABLE_AGENTS: Dict[name -> (module, class)]
    - get_agent(): Lazy instantiation with XAI wrapping
    - 60+ agents registered
```

**RLM Integration**:
- RLM uses registry to spawn "thought worker" agents
- Each recursive node = one agent call via registry
- Registry ensures observability (XAI wrapping)

---

## Proposed RLM Architecture (Integrated)

### System Context Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            USER INTERFACE                                    │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CONVERSATIONAL AGENT                                  │
│                     (Universal Router + RLM Core)                            │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Recursive Task Executor                          │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────────────────┐  │   │
│  │  │ Decompose    │→│ Execute      │→│ Synthesize               │  │   │
│  │  │ (LLM)        │  │ (Parallel)   │  │ (Grounded)               │  │   │
│  │  └──────────────┘  └──────┬───────┘  └──────────────────────────┘  │   │
│  │                           │                                         │   │
│  │              ┌────────────┼────────────┐                           │   │
│  │              ▼            ▼            ▼                           │   │
│  │  ┌──────────────┐ ┌──────────┐ ┌──────────────┐                   │   │
│  │  │ Discovery    │ │ Geometry │ │ Standards    │  ...60+ nodes      │   │
│  │  │ Node         │ │ Node     │ │ Node         │                   │   │
│  │  └──────────────┘ └──────────┘ └──────────────┘                   │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
│                                                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Context Flow                                     │   │
│  │                                                                      │   │
│  │   User Input → EPHEMERAL (sub-task calc) → SCENE (accumulated)      │   │
│  │                      ↓                                               │   │
│  │   Sub-task Results → Synthesis → Facts (promoted to SCENE)          │   │
│  │                      ↓                                               │   │
│  │   Turn Complete → Summarize → CAMPAIGN (long-term patterns)         │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└──────────────────────────────┬──────────────────────────────────────────────┘
                               │
         ┌─────────────────────┼─────────────────────┐
         │                     │                     │
         ▼                     ▼                     ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ GlobalMemoryBank│  │  Agent Registry │  │   SessionStore  │
│                 │  │                 │  │                 │
│ • Failures      │  │ • 60+ Agents    │  │ • Discovery     │
│ • Optimizations │  │ • Lazy Loading  │  │ • Conversation  │
│ • Insights      │  │ • XAI Wrapping  │  │ • Context       │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Memory Hierarchy in RLM

```
Level 0: EPHEMERAL (Sub-task Working Memory)
├── Temporary calculations
├── Intermediate results
└── Auto-discarded after synthesis

Level 1: SCENE (Recursive Session Context)  
├── Accumulated facts from sub-tasks
├── User requirements (enriched)
└── Survives across turns

Level 2: CAMPAIGN (Cross-Design Patterns)
├── Learned heuristics
├── Material preferences
└── Survives across sessions

Level 3: AGENT_IDENTITY (Persistent)
├── Agent personality
└── Never evicted

Level 4: UNIVERSAL (Physics Knowledge)
├── Physical constants
└── Shared across all agents
```

---

## RLM + Existing System Integration

### 1. Integration with DiscoveryManager

**Current**: DiscoveryManager drives the entire conversation flow
**With RLM**: DiscoveryManager becomes ONE node in the recursive graph

```python
# Current Usage (Linear)
ConversationalAgent -> DiscoveryManager.check_completeness() -> Response

# RLM Usage (Recursive)
ConversationalAgent.run_recursive():
    1. Decompose: ["Gather basic requirements", "Validate constraints"]
    2. DiscoveryNode.execute():  # Reuses DiscoveryManager
         - Calls check_completeness()
         - Returns: {mission: "drone", environment: "aerospace"}
    3. ValidationNode.execute():
         - Checks feasibility
         - Returns: {feasible: True, concerns: ["weight"]}
    4. Synthesize: Combined response
```

**Benefit**: DiscoveryManager stays unchanged, becomes composable

### 2. Integration with EnhancedContextManager

**Key Insight**: Each recursive sub-task gets its own EPHEMERAL scope,
but shares the SCENE scope with siblings.

```python
class RecursiveTaskExecutor:
    def __init__(self, parent_context: EnhancedContextManager):
        self.parent_context = parent_context
        
    async def execute_subtask(self, task: SubTask) -> TaskResult:
        # Create isolated context for this sub-task
        sub_context = EnhancedContextManager(
            agent_id=f"subtask_{task.id}",
            max_tokens=4000
        )
        
        # Copy relevant SCENE-level context from parent
        sub_context.plan = self.parent_context.plan
        sub_context.constraints = self.parent_context.constraints
        
        # Execute
        result = await task.execute(sub_context)
        
        # Promote key findings to parent SCENE
        for fragment in sub_context.working_memory:
            if "critical" in fragment.tags:
                await self.parent_context.add_message(
                    role="system",
                    content=fragment.content,
                    scope=ContextScope.SCENE,
                    tags={"recursive_fact", task.type}
                )
        
        return result
```

### 3. Integration with GlobalMemoryBank

**Pattern**: Cache expensive recursive queries

```python
class RecursiveTaskExecutor:
    def __init__(self):
        self.global_memory = GlobalMemoryBank()
        
    async def execute_with_memory(self, task: SubTask) -> TaskResult:
        # Check if we've done this before
        cache_key = f"{task.type}:{hash(task.params)}"
        cached = self.global_memory.query(cache_key)
        
        if cached and cached[0]["outcome"] == "SUCCESS":
            logger.info(f"Using cached result for {task.type}")
            return TaskResult.from_cache(cached[0])
        
        # Execute fresh
        result = await task.execute()
        
        # Store outcome
        self.global_memory.add_experience(
            agent_id="RLM",
            context=cache_key,
            outcome="SUCCESS" if result.success else "FAILURE",
            data={"result": result.to_dict()}
        )
        
        return result
```

### 4. Integration with ConversationManager

**Pattern**: Recursive branching for design exploration

```python
# User: "What if I used carbon fiber instead?"

ConversationalAgent:
    1. Recognizes this as "design variant exploration"
    2. Creates branch in ConversationManager
    3. Spawns parallel sub-tasks:
       - Current: Analyze existing design (Titanium)
       - Branch 1: Analyze variant (Carbon Fiber)
       - Branch 2: Analyze variant (Aluminum)
    4. Compares results
    5. Presents: "Carbon fiber saves 30% weight but costs 2x more"
```

---

## Implementation Strategy (Revised)

### Phase 1: Foundation (Week 1)
**Goal**: Minimal viable RLM that integrates with existing systems

**Tasks**:
1. Create `RecursiveTaskExecutor` class
2. Implement `SubTask` dataclass with context passing
3. Wire into `ConversationalAgent._handle_design_flow()`
4. Add recursion depth limits (max 2 for MVP)

**Deliverable**: RLM can decompose simple queries into 2-3 sub-tasks

### Phase 2: Node Refactoring (Week 1-2)
**Goal**: Make existing agents callable as recursive nodes

**Tasks**:
1. Create `BaseRecursiveNode` abstract class
2. Refactor `GeometryEstimator` as `GeometryRecursiveNode`
3. Refactor `CostAgent` as `CostRecursiveNode`
4. Refactor `DiscoveryManager` as `DiscoveryRecursiveNode`

**Key Change**: Each node accepts `EnhancedContextManager` and returns structured `NodeResult`

### Phase 3: Memory Integration (Week 2)
**Goal**: Full context hierarchy and global memory

**Tasks**:
1. Implement EPHEMERAL/SCENE promotion in sub-tasks
2. Wire GlobalMemoryBank for caching
3. Add conversation branching for design variants
4. Implement synthesis with grounded generation

### Phase 4: Polish (Week 3)
**Goal**: Production readiness

**Tasks**:
1. Add detailed tracing/logging for each recursion level
2. Cost tracking and budget enforcement
3. Performance optimization (parallel execution)
4. Fallbacks for failed recursion

---

## Critical Design Decisions

### 1. Context Isolation vs Sharing

**Decision**: Sub-tasks get isolated EPHEMERAL context but share SCENE

**Rationale**:
- Isolation prevents sub-task pollution
- Sharing enables fact accumulation
- SCENE persistence across turns

### 2. Synchronous vs Parallel Execution

**Decision**: Parallel for independent sub-tasks, sequential for dependent

```python
# Independent tasks (parallel)
[MaterialQuery, StandardsQuery, GeometryCalc]  # All can run at once

# Dependent tasks (sequential)
MaterialQuery → GeometryCalc  # Need density before calculating mass
```

### 3. When to Recurse

**Decision**: Triage based on complexity indicators

```python
async def _should_use_rlm(self, intent: str, context: Dict) -> bool:
    indicators = [
        len(intent.split()) > 10,           # Complex description
        "and" in intent.lower(),            # Multiple entities
        "compare" in intent.lower(),        # Comparative analysis
        "optimize" in intent.lower(),       # Optimization request
        "if" in intent.lower(),             # Conditional logic
        context.get("has_constraints", False)  # Active constraints
    ]
    
    return sum(indicators) >= 2  # Threshold
```

### 4. Failure Handling

**Decision**: Graceful degradation to single-pass

```python
async def run(self, params, session_id):
    try:
        if self._should_use_rlm(...):
            return await self._rlm_execute(...)
    except RecursionError as e:
        logger.warning(f"RLM failed: {e}, falling back to single-pass")
    
    return await self._single_pass_execute(...)  # Fallback
```

---

## Benefits Summary

| Capability | Without RLM | With RLM |
|------------|-------------|----------|
| **Hardcoding** | Possible (values in code) | Impossible (must query) |
| **Context Accumulation** | Session only | Session + Campaign |
| **Multi-turn Reasoning** | Linear | Recursive depth 3 |
| **Explainability** | Black box | Full trace tree |
| **Design Variants** | Sequential | Parallel branches |
| **Cost Control** | Per-request | Budget + caching |

---

## Conclusion

The current BRICK OS architecture is **ideally suited** for RLM integration:

1. **DiscoveryManager** → Becomes recursive node
2. **EnhancedContextManager** → Provides hierarchical memory
3. **GlobalMemoryBank** → Enables cross-session learning
4. **GlobalAgentRegistry** → Spawns thought workers
5. **ConversationManager** → Handles branching

**Implementation complexity**: LOW (reuses 80% existing code)
**Capability gain**: HIGH (10x reasoning depth, explainability)
**Risk**: LOW (graceful fallback to current behavior)

**Recommendation**: Proceed with Phase 1 immediately.
