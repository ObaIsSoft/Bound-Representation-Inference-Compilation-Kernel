# BRICK OS - Technical Debt & Implementation Tasks

> **Generated from Comprehensive Codebase Analysis**
> **Status**: Tier 6 Backend / Tier 2 Frontend
> **Last Updated**: 2026-02-08

---

## 🚨 CRITICAL - Fix Immediately (Production Blockers)

### C1. Silent Agent Loading Failures
**File**: `backend/agent_registry.py` (line ~160)
**Issue**: `_lazy_load()` returns `None` on import failure, causing downstream null pointer exceptions
**Risk**: System appears to work but agents fail silently, leading to undefined behavior
**Fix**:
```python
def _lazy_load(self, name: str) -> Optional[Any]:
    try:
        # ... import logic ...
    except Exception as e:
        logger.error(f"Failed to load agent {name}: {e}")
        raise RuntimeError(f"Critical agent {name} failed to load: {e}") from e
```
**Acceptance**: Registry raises explicit error on agent load failure

---

### C2. Async Context Violation in GeometryAgent
**File**: `backend/agents/geometry_agent.py` (line ~158)
**Issue**: `asyncio.run()` called inside synchronous `run()` method - crashes when already in async context
**Risk**: Runtime crash during geometry compilation
**Fix**: Convert `GeometryAgent.run()` to async or use thread-safe async execution
```python
async def run(self, params, intent, ...):  # Make async
    result = await self.engine.compile(geometry_tree, format="glb")
```
**Acceptance**: GeometryAgent runs without asyncio errors in orchestrator

---

### C3. No Persistent Session Storage
**File**: `backend/context_manager.py`, `backend/agents/conversational_agent.py`
**Issue**: `InMemorySessionStore` used - all session data lost on server restart
**Risk**: Production restart = lost user conversations and design state
**Fix**: Implement Redis persistence
```python
class RedisSessionStore(SessionStore):
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis = aioredis.from_url(redis_url)
```
**Acceptance**: Sessions survive server restart, TTL still enforced

---

### C4. Circular Import Between Main and Orchestrator
**Files**: `backend/main.py`, `backend/orchestrator.py`
**Issue**: 
- `orchestrator.py` imports `from main import inject_thought`
- `main.py` imports `from orchestrator import run_orchestrator`
**Risk**: Import-time crashes, unpredictable module loading
**Fix**: Move `inject_thought` to separate module (e.g., `backend/xai_stream.py`)
**Acceptance**: No cross-imports between main.py and orchestrator.py

---

### C5. State Mutation Bug in Physics Node
**File**: `backend/orchestrator.py` (physics_node, lines 640-700)
**Issue**: Direct mutation of `flags` and `reasons` dicts while iterating
**Risk**: Runtime errors, inconsistent state
**Fix**: Create new dict objects instead of mutating in place
**Acceptance**: Physics node completes without mutation errors

---

## 🔴 HIGH - Major Functionality Broken/Missing

### H1. Frontend Components Are Stubs
**Files**: 
- `frontend/src/components/panels/CompilePanel.jsx` (9 lines - empty div)
- `frontend/src/pages/Workspace.jsx` (58 lines - placeholder text)
- `frontend/src/components/panels/*.jsx` (many have empty handlers)

**Issue**: Frontend is non-functional beyond requirements gathering
**Fix Priority**:
1. Implement actual 3D viewer in Workspace (Three.js already in deps)
2. Connect CompilePanel to `/api/orchestrator/run` endpoint
3. Add agent status dashboard (Phase 11.3 - SystemHealth)
4. Implement design parameter adjustment UI
**Acceptance**: User can view 3D models, trigger compilation, see agent progress

---

### H2. Mixed Async/Sync Agent Interfaces
**Files**: All files in `backend/agents/`
**Issue**: Some agents have `def run()`, others `async def run()` - requires boilerplate checking
**Risk**: Blocking in async context, complex error-prone calling code
**Fix**: Standardize all agents to async interface
```python
class BaseAgent(ABC):
    @abstractmethod
    async def run(self, params: Dict) -> Dict:
        pass
```
**Acceptance**: Remove all `hasattr(agent, "run") and asyncio.iscoroutinefunction(agent.run)` checks

---

### H3. No Docker/Containerization
**Missing**: `Dockerfile`, `docker-compose.yml`
**Issue**: Complex dependencies (FEniCS, CoolProp, physics libraries) hard to deploy
**Fix**: Multi-stage Dockerfile
```dockerfile
FROM python:3.12-slim as base
# Install system deps for physics libs
RUN apt-get update && apt-get install -y libopenmpi-dev petsc-dev
# ... pip install ...
```
**Acceptance**: `docker-compose up` starts full stack (backend + redis + frontend)

---

### H4. Test Quality - Excessive Mocking
**Files**: `tests/unit/test_physics.py`, `tests/unit/test_*.py`
**Issue**: Tests mock 30+ modules - testing mock configuration, not actual physics
**Risk**: False confidence, bugs in real physics code not caught
**Fix**: Integration tests with real physics kernel, mocked only at LLM/API boundaries
**Acceptance**: Tests verify actual physics calculations against known values

---

### H5. Frontend-Backend Type Mismatch
**Files**: `frontend/src/pages/RequirementsGatheringPage.jsx`, `backend/main.py`
**Issue**: Frontend sends `FormData`, backend expects JSON/Pydantic - inconsistent handling
**Fix**: Standardize all API endpoints to accept JSON, update frontend to send JSON
**Acceptance**: All endpoints have consistent Pydantic models, no manual FormData parsing

---

### H6. Missing CI/CD Pipeline
**Missing**: `.github/workflows/`, `.gitlab-ci.yml`
**Issue**: No automated testing, no deployment pipeline
**Fix**: GitHub Actions workflow
```yaml
- Lint (ruff, mypy)
- Unit tests (pytest)
- Integration tests (docker-compose)
- Frontend build (vite build)
- Deploy to staging
```
**Acceptance**: PRs trigger automated checks, main branch auto-deploys

---

## 🟡 MEDIUM - Code Quality & Technical Debt

### M1. AgentState God Object
**File**: `backend/schema.py` (AgentState TypedDict)
**Issue**: 40+ fields - violates Single Responsibility Principle
**Risk**: Hard to track changes, race conditions, merge conflicts
**Fix**: Split into domain-specific sub-objects
```python
class AgentState(TypedDict):
    core: CoreState
    design: DesignState
    geometry: GeometryState
    physics: PhysicsState
    manufacturing: ManufacturingState
```
**Acceptance**: No state object has more than 10 fields

---

### M2. Backup Files in Repository
**File**: `backend/orchestrator.py.backup`
**Issue**: Should not be in version control
**Fix**: `git rm backend/orchestrator.py.backup`, add `*.backup` to `.gitignore`
**Acceptance**: No backup files in repo

---

### M3. Bare Except Pass (3 occurrences)
**Files**: Found in 3 Python files
**Issue**: `except Exception: pass` masks critical errors
**Fix**: Explicit exception handling with logging
```python
except SpecificException as e:
    logger.error(f"Context: {e}")
    raise  # or handle appropriately
```
**Acceptance**: Zero `except: pass` patterns in codebase

---

### M4. Debug Scripts Pollution
**Files**: `backend/debug_*.py`, `backend/tests/legacy/cleanup/`
**Issue**: Unclear which scripts are current vs legacy
**Fix**: 
1. Move debug scripts to `scripts/debug/`
2. Add README explaining each script
3. Archive or delete legacy cleanup scripts
**Acceptance**: Only actively used scripts in main directories

---

### M5. Hardcoded Values
**Files**: Multiple agent files
**Issue**: Magic numbers scattered throughout
```python
THERMAL_LOAD_W = POWER_REQ_W * 0.15  # Why 0.15?
max_questions = 5  # In multiple places
```
**Fix**: Centralize configuration
```python
from config import THERMAL_EFFICIENCY_DEFAULT, MAX_DISCOVERY_QUESTIONS
```
**Acceptance**: Configuration values in `backend/config/` directory

---

### M6. Missing Health Check Endpoint
**File**: `backend/main.py` (has `/api/health` but basic)
**Issue**: Health check doesn't verify LLM providers, agents, physics kernel
**Fix**: Comprehensive health check
```python
@app.get("/api/health")
async def health_check():
    return {
        "status": "healthy",
        "llm_provider": check_llm(),  # Test actual call
        "physics_kernel": check_physics(),  # Verify providers
        "agents_loaded": len(registry._agents),
        "materials_api": check_materials_api()
    }
```
**Acceptance**: Health check fails if any critical dependency unavailable

---

## 🟢 LOW - Nice to Have / Polish

### L1. Frontend Three.js Integration
**Missing**: 3D viewer component
**Issue**: Three.js in package.json but unused
**Fix**: Create `frontend/src/components/viewers/ModelViewer.jsx`
- Load GLB from `/api/geometry/model/{model_id}`
- Display physics telemetry overlay
- Allow parameter adjustment
**Acceptance**: User can view 3D models with physics data overlay

---

### L2. WebSocket for Real-time Updates
**Files**: `backend/main.py`, Frontend
**Issue**: Frontend polls for agent thoughts (`/api/agents/thoughts`)
**Fix**: WebSocket endpoint for real-time agent status
```python
@app.websocket("/ws/orchestrator/{project_id}")
async def orchestrator_ws(websocket: WebSocket, project_id: str):
    # Stream agent progress, thoughts, completions
```
**Acceptance**: Real-time progress bar during orchestration

---

### L3. Documentation Sync
**Files**: `docs/MASTER_ARCHITECTURE.md`, `README.md`
**Issue**: Documentation may drift from implementation
**Fix**: Add docs-as-code checks in CI
**Acceptance**: Architecture diagrams auto-generated from code

---

### L4. Performance Monitoring
**Files**: `backend/monitoring/latency.py`
**Issue**: Basic latency tracking, no alerting
**Fix**: Integrate with Prometheus/Grafana or Sentry Performance
**Acceptance**: Dashboard showing agent execution times, bottleneck identification

---

### L5. Environment Variable Validation
**File**: `backend/main.py` (startup)
**Issue**: No validation that required env vars are set
**Fix**: Pydantic Settings validation on startup
```python
class Settings(BaseSettings):
    groq_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    # At least one must be set
    
    @validator
    def at_least_one_llm(cls, v, values):
        if not any([values.get('groq_api_key'), values.get('openai_api_key')]):
            raise ValueError("At least one LLM provider required")
```
**Acceptance**: Clear error message on startup if config invalid

---

## 📋 Implementation Priority Roadmap

### Week 1: Critical Fixes (Production Blockers)
- [ ] C1: Fix silent agent loading failures
- [ ] C2: Fix async context violation in GeometryAgent
- [ ] C3: Implement Redis session persistence
- [ ] C4: Fix circular import (main ↔ orchestrator)
- [ ] C5: Fix state mutation bug in physics_node
- [ ] M2: Remove backup files from repo
- [ ] M3: Fix bare except pass patterns

### Week 2: Frontend MVP
- [ ] H1: Implement Workspace 3D viewer (Three.js)
- [ ] H1: Connect CompilePanel to orchestrator API
- [ ] H1: Add agent status/progress UI
- [ ] H5: Standardize frontend to JSON API calls
- [ ] L1: Add physics telemetry overlay to 3D viewer

### Week 3: Backend Hardening
- [ ] H2: Standardize all agents to async interface
- [ ] H3: Create Docker containerization
- [ ] M1: Refactor AgentState god object
- [ ] M6: Implement comprehensive health checks
- [ ] M4: Clean up debug scripts

### Week 4: Testing & CI/CD
- [ ] H4: Rewrite tests with real physics (not mocks)
- [ ] H6: Set up GitHub Actions CI/CD
- [ ] M5: Centralize configuration
- [ ] L5: Add environment validation

### Month 2: Advanced Features
- [ ] L2: WebSocket real-time updates
- [ ] L4: Performance monitoring dashboard
- [ ] L3: Auto-generated architecture docs
- [ ] Integration tests with full pipeline

---

## 🎯 Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Silent failures | 3+ | 0 |
| Frontend stub components | 15+ | 0 |
| Test coverage (real code) | ~20% | 70% |
| Async/sync boilerplate per agent | 6 lines | 0 lines |
| Session persistence | None | Redis |
| Docker deployment | None | Working |
| CI/CD | None | Automated |

---

## 📝 Notes

### Patterns to Follow
- **Lazy Loading**: Continue using `GlobalAgentRegistry` pattern
- **De-Mocking**: Never add mocks back - always use real data/providers
- **Physics-First**: Maintain validation before generation approach
- **Critic Integration**: Keep observer pattern for self-evolution

### Anti-Patterns to Avoid
- Silent failures (return None on error)
- Mixed async/sync interfaces
- God objects (40+ field state)
- Excessive mocking in tests
- Bare except: pass
- Hardcoded magic numbers
