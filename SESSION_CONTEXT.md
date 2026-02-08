# BRICK OS - Session Context

## Current Status: Phase 1 Critical Fixes COMPLETE

**Date:** 2026-02-08  
**Branch:** phase1-critical-fixes  
**Next:** Phase 4 (Testing & CI/CD) and Phase 5 (Monitoring/WebSockets)

---

## Completed Phase 1 Critical Fixes

### 1. Silent Agent Loading Fix ✅
**File:** `backend/agent_registry.py`  
**Issue:** `_lazy_load()` returned `None` on import failure, causing silent downstream failures  
**Fix:** Now raises `RuntimeError` with descriptive message + `AgentNotFoundError` exception class added

```python
# Before: return None
# After:
raise RuntimeError(f"Critical agent {name} failed to load: {e}") from e
```

### 2. Async Context Fix ✅
**File:** `backend/agents/geometry_agent.py`  
**Issue:** `asyncio.run()` inside async methods caused "cannot be called from a running event loop"  
**Fix:** Converted `run()` to async, using `await self.engine.compile()` instead of `asyncio.run()`

```python
# Before: result = asyncio.run(self.engine.compile(...))
# After:  result = await self.engine.compile(...)
```

### 3. Redis Session Persistence ✅
**File:** `backend/session_store.py` (NEW FILE)  
**Issue:** Session data lost on server restart (was in-memory only)  
**Fix:** Implemented session store abstraction with Redis + InMemory + Factory pattern

```python
# Usage:
store = create_session_store()  # Auto-detects REDIS_URL env var
await store.save_session(session_id, data)
data = await store.load_session(session_id)
```

### 4. Circular Import Fix ✅
**File:** `backend/xai_stream.py` (NEW FILE)  
**Issue:** `main.py` ↔ `orchestrator.py` circular dependency via XAI thought streaming  
**Fix:** Created centralized `xai_stream.py` module with `THOUGHT_STREAM` deque

```python
# Both modules now import from xai_stream:
from xai_stream import inject_thought, get_thoughts
```

### 5. State Mutation Fix ✅
**File:** `backend/orchestrator.py`  
**Issue:** `physics_node` mutated original state dict instead of building new state  
**Fix:** Create NEW validation_flags dict with copied values

```python
# Before: Mutated original flags dict
# After:
validation_flags = {
    **state.get("validation_flags", {}),
    "physics_safe": len(failure_reasons) == 0,
    "reasons": list(original_flags.get("reasons", []))
}
```

### 6. API Standardization ✅
**Files:** `backend/main.py`, `frontend/src/pages/Landing.tsx`, `frontend/src/pages/RequirementsGatheringPage.jsx`  
**Issue:** `/chat/requirements` and `/orchestrator/*` endpoints accepted FormData instead of JSON  
**Fix:** All endpoints now accept Pydantic JSON models (except `/stt/transcribe` which correctly uses FormData for binary audio)

```python
# Backend:
@app.post("/api/chat/requirements")
async def chat_requirements_endpoint(req: ChatRequirementsRequest):
    # Access via req.message, req.user_intent, etc.

# Frontend:
const payload = { message, user_intent, session_id };
await apiClient.post('/chat/requirements', payload);  // JSON, not FormData
```

### 7. Async Agent Handling in physics_node ✅
**File:** `backend/orchestrator.py`  
**Issue:** 6 agents called without async check (Material, Chemistry, Electronics, Thermal, Structural, Control)  
**Fix:** Added proper async/sync handling pattern for all XAI-wrapped agent calls

```python
# Pattern applied to all 6 agent calls:
if hasattr(agent, "run") and asyncio.iscoroutinefunction(agent.run):
    result = await agent.run(params)
else:
    result = agent.run(params)
    if asyncio.iscoroutine(result):
        result = await result
```

---

## Files Modified

### Backend
- `backend/agent_registry.py` - Added AgentNotFoundError, raise on load failure
- `backend/agents/geometry_agent.py` - Converted to async
- `backend/agents/explainable_agent.py` - Updated to import from xai_stream
- `backend/orchestrator.py` - Fixed state mutation, async agent handling
- `backend/main.py` - API standardization to Pydantic models

### New Files
- `backend/session_store.py` - Session persistence abstraction
- `backend/xai_stream.py` - Centralized XAI thought streaming

### Frontend
- `frontend/src/pages/Landing.tsx` - Updated to send JSON
- `frontend/src/pages/RequirementsGatheringPage.jsx` - Updated to send JSON

---

## Voice/Audio Flow (Preserved)

The voice flow correctly preserves FormData for binary audio:

```
Audio File → /stt/transcribe (FormData) → Transcript → Landing state → 
RequirementsGatheringPage → /chat/requirements (JSON)
```

---

## Architecture Notes

### Agent Registry Pattern
- All 57 agents registered in `AVAILABLE_AGENTS`
- Agents wrapped with `ExplainableAgent` for XAI (except XAI itself)
- `ExplainableAgent.run()` is async - orchestrator nodes must await
- `AgentNotFoundError` raised for missing agents with suggestions

### Session Store Pattern
- Factory `create_session_store()` auto-detects `REDIS_URL` env var
- Falls back to `InMemorySessionStore` if Redis unavailable
- Interface: `save_session()`, `load_session()`, `delete_session()`

### Async/Sync Agent Compatibility
All orchestrator nodes use this pattern:
```python
if hasattr(agent, "run") and asyncio.iscoroutinefunction(agent.run):
    result = await agent.run(params)
else:
    result = agent.run(params)
    if asyncio.iscoroutine(result):
        result = await result
```

---

## Next Phase: Phase 4 & 5

### Phase 4: Testing & CI/CD (Day 20-24)
1. Integration tests for agent chains
2. Frontend component tests
3. GitHub Actions CI/CD pipeline
4. Test coverage reporting

### Phase 5: Real-time Monitoring (Day 25-28)
1. WebSocket endpoint for live updates
2. Frontend subscription handling
3. Agent progress streaming
4. Performance metrics dashboard

---

## Quick Start

```bash
# Backend
cd backend
pip install -r requirements.txt
export REDIS_URL=redis://localhost:6379  # Optional
python main.py

# Frontend
cd frontend
npm install
npm run dev
```

---

## Verification Commands

```bash
# Syntax check all modified files
python -m py_compile backend/orchestrator.py backend/agent_registry.py \
  backend/agents/geometry_agent.py backend/session_store.py \
  backend/xai_stream.py backend/main.py

# Import test (limited - full test requires all dependencies)
cd backend && python -c "from agent_registry import registry, AgentNotFoundError"
```

---

## Known Limitations

1. **Frontend Stubs:** CompilePanel, Workspace remain stubbed per user request
2. **Agent Async Migration:** 57 agents total - some still sync (works via XAI wrapper)
3. **LangGraph Dependency:** Full import test requires `pip install langgraph`
4. **Redis Optional:** Falls back to in-memory if REDIS_URL not set

---

## Context Preservation

This file serves as the "boot sequence" for new sessions. Load this first to understand:
- What was fixed in Phase 1
- Current architecture patterns
- Next priorities (Phase 4/5)
- How to verify changes
