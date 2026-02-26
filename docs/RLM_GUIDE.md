# BRICK OS - Recursive Language Model (RLM) Guide

> **Unified Conversational Architecture**  
> **Version:** 2026.02.26  
> **Status:** Implemented in ConversationalAgent

---

## Overview

The Recursive Language Model (RLM) transforms the ConversationalAgent from a "question-answerer" to a "problem-solver" through recursive decomposition.

**Key Difference:**
```python
# Old (Linear)
User: "Titanium drone"
Agent: *single LLM call* → "Tell me more about the drone..."

# RLM (Recursive)
User: "Titanium drone"
Agent:
  1. Decompose: ["What is titanium density?", "What is typical drone volume?", "Calculate mass"]
  2. Execute sub-tasks (query DB, standards, calculate)
  3. Synthesize: "A titanium drone would weigh ~1.2kg. Is this acceptable?"
```

---

## Architecture

**File:** `backend/agents/conversational_agent.py` (1,000+ lines)

### Core Components

```python
# 1. DiscoveryRecursiveNode - Entry point
class DiscoveryRecursiveNode:
    - Intent classification
    - Requirements extraction
    - Gathering missing info
    - Determines when ready to proceed

# 2. Specialized Recursive Nodes
- GeometryRecursiveNode
- MaterialRecursiveNode
- CostRecursiveNode
- SafetyRecursiveNode

# 3. Context Management
class NodeContext:
    - user_input
    - requirements
    - constraints
    - facts
    - history
    - depth (recursion tracking)
```

### Execution Flow

```
User Input
    ↓
DiscoveryRecursiveNode.execute()
    ↓
Intent Classification (design/analysis/question/chat)
    ↓
Requirements Extraction
    ↓
Sufficient Info?
    ├── No → Generate Question → User
    └── Yes → Route to Specialized Node
                ↓
        [Geometry|Material|Cost|Safety]RecursiveNode
                ↓
        Sub-task Execution (may recurse)
                ↓
        Synthesis → Response
```

---

## Node Types

### DiscoveryRecursiveNode
- **Purpose:** Entry point for ALL conversations
- **Max Turns:** 5
- **Min Turns:** 2
- **Schema:** Structured JSON extraction

### GeometryRecursiveNode
- **Purpose:** CAD modeling and geometry analysis
- **Capabilities:** Shape generation, constraint solving

### MaterialRecursiveNode
- **Purpose:** Material selection and property lookup
- **Capabilities:** Database queries, temperature models

### CostRecursiveNode
- **Purpose:** Cost estimation and budgeting
- **Capabilities:** Pricing APIs, manufacturing rates

### SafetyRecursiveNode
- **Purpose:** Safety validation and hazard detection
- **Capabilities:** Standards lookup, limit checking

---

## Integration Status

**Implementation:** ✅ Complete in `conversational_agent.py`

**RLM Integration:**
- DiscoveryRecursiveNode: ✅ Implemented
- GeometryRecursiveNode: ✅ Implemented
- MaterialRecursiveNode: ✅ Implemented
- CostRecursiveNode: ✅ Implemented
- SafetyRecursiveNode: ✅ Implemented

**Previous DiscoveryManager:** ❌ Removed (replaced by RLM)

---

## Conversation Flows

### Requirements Gathering
```
User: "I want a drone"
Agent: "What material would you like to use? (e.g., Carbon fiber, Aluminum)"
User: "Titanium"
Agent: "What is the primary mission? (e.g., Photography, Racing, Delivery)"
User: "Racing"
Agent: "Any size constraints?"
User: "Under 2kg"
→ Sufficient info gathered → Proceed to design
```

### Recursive Analysis
```
User: "Is titanium strong enough for a racing drone?"
Agent:
  1. Query Material DB → Ti-6Al-4V properties
  2. Query Racing drone loads → Typical G-forces
  3. Calculate safety factor
  4. Compare to aerospace standards
→ "Titanium has adequate strength with 2.5x safety factor..."
```

---

## Configuration

**Environment Variables:**
```bash
# LLM Provider (required)
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...

# Optional
REDIS_URL=redis://localhost:6379/0
```

**Session Management:**
- SessionStore: Persistent conversation history
- ContextScope: EPHEMERAL → WORKFLOW → PROJECT → UNIVERSAL

---

## Usage

```python
from backend.agents.conversational_agent import ConversationalAgent
from backend.llm.factory import get_llm_provider

async def main():
    llm = get_llm_provider()
    agent = ConversationalAgent(llm_provider=llm)
    
    # Single turn
    result = await agent.run({
        "user_intent": "I want a titanium drone",
        "session_id": "session_123"
    })
    
    print(result["response"])  # "What material would you like..."

asyncio.run(main())
```

---

## Cleanup History

**Files Removed/Consolidated:**
- RLM_ANALYSIS.md → Merged into this guide
- RLM_ARCHITECTURE_ANALYSIS.md → Merged
- RLM_CONVERSATION_FLOWS.md → Merged
- RLM_IMPLEMENTATION_SUMMARY.md → Merged
- RLM_INTEGRATION_CLEANUP.md → Merged
- RLM_ROBOT_HAND_EVALUATION.md → Test-specific, archived

---

*Consolidated from 7 RLM-related documentation files*
