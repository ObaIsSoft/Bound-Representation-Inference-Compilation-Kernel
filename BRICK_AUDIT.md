# BRICK OS — Full Codebase Audit
> Generated: 2026-03-28 | Auditor: Claude Sonnet 4.6 | Deep scan of all backend + frontend

---

## System Overview

**What it is:** Multi-agent AI hardware design compiler. User describes intent → 50+ specialized agents run physics, geometry, thermal, structural, fluid, electronics, manufacturing, compliance analysis → outputs validated design with BOM and 3D geometry.

**Stack:**
- Backend: FastAPI + LangGraph (Python 3.10+)
- Frontend: React 18 / Vite (JSX + one .tsx outlier), Three.js / R3F, Tauri desktop wrapper
- Physics: scipy, scikit-fem, CoolProp, manifold3d, cadquery/OCP, FiPy (unlisted in requirements)
- LLM: Groq, OpenAI, Gemini, Ollama, Kimi, HuggingFace

**Self-assessed production readiness: 4/10** (from task.md)

---

## 🔴 CRITICAL — Will crash or silently corrupt data

### C-01: Five Agent Files Registered But Do Not Exist
`backend/agent_registry.py` registers these → lazy load will throw `ImportError` (unhandled), bubbling to 500 on any request that triggers these phases:

| Registry Key | Expected Path | Impact |
|---|---|---|
| `DesignerAgent` | `agents/designer_agent.py` | 500 on design finalization phase |
| `DesignExplorationAgent` | `agents/design_exploration_agent.py` | 500 on exploration routes |
| `ConstructionAgent` | `agents/construction_agent.py` | 500 on construction phase |
| `MepAgent` | `agents/mep_agent.py` | 500 on MEP analysis (arch path) |
| `ZoningAgent` | `agents/zoning_agent.py` | 500 on zoning analysis |

`config/design_exploration_config.py` even exists (config for a missing agent).

---

### C-02: `structural_agent_fixed.py` Referenced But Does Not Exist
`backend/physics/domains/structures.py:37,74` lazy-imports:
```python
from backend.agents.structural_agent_fixed import ProductionStructuralAgent
from backend.agents.structural_agent_fixed import FidelityLevel
```
File: `agents/structural_agent_fixed.py` — **does not exist**. Only `structural_agent.py` exists.
The `_get_structural_agent()` method silently sets `self._structural_agent = False` on failure and falls back to analytical mode — so this fails quietly without warning, always using the weaker analytical path.

---

### C-03: `AgentVersionRegistry` Class Does Not Exist Anywhere
`main.py:489` and `main.py:989`:
```python
from backend.core.agent_registry import AgentVersionRegistry
registry = AgentVersionRegistry()
return {"metrics": registry.get_all_metrics()}
```
`backend/core/` contains `agent_executor.py` (not `agent_registry.py`). `AgentVersionRegistry` does not exist anywhere in the codebase. `/api/agents/metrics` throws `ModuleNotFoundError` on every call.

---

### C-04: Duplicate `ChatRequest` Pydantic Model — Wrong Schema Active
`main.py:515` defines `ChatRequest` with `(message, session_id, ai_model, context)`.
`main.py:2414` redefines `ChatRequest` with `(message, context, aiModel, conversation_id, language, focusedPodId)`.

Python's second definition silently replaces the first. The `/api/chat` endpoint at line 522 now uses the wrong model. `session_id` and `ai_model` are absent from the active schema — clients sending these fields get silent key mismatches. The `ai_model` field is never passed to the LLM factory anyway (see C-09), but `session_id` lookup breaks.

---

### C-05: File Attachment Upload Is Completely Broken
`GlobalInputConsole.jsx:98-118`:
```javascript
if (filesToSend.length > 0) {
    const formData = new FormData();  // ← built, never used
    formData.append('message', userMsg);
    formData.append('session_id', activeSessionId);
    filesToSend.forEach(file => formData.append('files', file));
    // ↓ sends JSON, not formData — files never transmitted
    responseData = await apiClient.post('/chat', { message: userMsg, ... });
}
```
The `formData` object is constructed and immediately abandoned. Both the `if` and `else` branches send identical JSON. File attachments are silently dropped every time.

---

### ~~C-06: RETRACTED — `GncAgent.run()` Is Actually Fine~~
~~Originally reported: `gnc_agent.py:21` has a stray `return None` making `run()` unreachable.~~
**Correction:** Line 21 is inside an `except ImportError` fallback function `get_physics_kernel()`, not inside `GncAgent.run()`. The GNC agent's `run()` method (line 45) works correctly — it performs real physics: thrust-to-weight ratio, stability checks, CEM trajectory optimization. No fix needed.

---

### ~~C-07: RETRACTED — `.env` Was Never Committed~~
Verified via `git log --all --full-history -- backend/.env` → 0 commits. The `.env` is correctly listed in `.gitignore` and has never entered git history. No action needed on this finding.

Note: the `.env` does contain many live API keys (OpenAI, Gemini, Groq, Supabase, Nexar, Materials Project, etc.). This is normal for local dev. The risk would only arise if the file were accidentally committed — which it has not been.

---

### C-08: Import Path Inconsistency Creates Runtime Split-Brain
Two competing import styles coexist and both are active:

**Pattern A** (`backend.`-prefixed, used in `orchestrator.py`, `main.py`):
```python
from backend.agent_registry import registry
from backend.llm.factory import get_llm_provider
from backend.xai_stream import inject_thought
```

**Pattern B** (bare module name, used in `new_nodes.py`, `document_agent.py`, `conversational_agent.py`, tests):
```python
from agent_registry import registry
from llm.factory import get_llm_provider
from xai_stream import inject_thought
```

`orchestrator.py` imports `new_nodes` as `from backend.new_nodes import ...`, but `new_nodes.py` itself uses bare imports. This works only because `main.py` manually injects the backend directory into `sys.path` at startup. If any module is imported before `main.py` runs that path setup (e.g., in tests or standalone scripts), bare imports fail. This is the root cause of all "works in server, fails in tests" bugs.

---

## 🟠 HIGH — Broken functionality, silent wrong results

### H-01: LLM Provider Selection Is Completely Ignored
`GlobalInputConsole` has a provider dropdown (defaults to 'groq') and sends `ai_model: 'groq'` in the request body.
`/api/chat` endpoint receives `req.ai_model` but the comment at line 536 literally says:
```python
# agent = ConversationalAgent(model_name=req.ai_model) -> Use global conversational_agent
```
The global `conversational_agent` is always used regardless of what the user selects. The LLM dropdown in the UI **does nothing**. The user has no actual control over which model processes their request.

Additionally, `SettingsContext.jsx` manages an `aiModel` state (defaults to `'mock'`) but it is **never imported or used** by `GlobalInputConsole`. The settings UI and the chat UI are completely disconnected. Changing the model in Settings has zero effect.

---

### H-02: LangGraph Full Pipeline Never Triggered From Chat UI
The 8-phase agent pipeline (`run_orchestrator()` in `orchestrator.py`) is never called from any frontend-facing chat endpoint:
- `/api/chat` → calls `conversational_agent.run()` (single LLM call only)
- `/api/chat/requirements` → calls `conversational_agent.chat()` + 3 quick estimator agents
- `/api/chat/discovery` → calls `conversational_agent.run()` (single LLM call only)

The full pipeline (geometry agent, physics agent, thermal, structural, GNC, manufacturing, compliance, etc.) is only reachable via `/api/v1/orchestrator/projects` (the `ProjectOrchestrator` API), which has **no frontend UI whatsoever**. The 50-agent system is effectively unreachable through normal use.

---

### H-03: 3D Viewport Never Receives Geometry
`Omniviewport.jsx:50-51` extracts geometry from WebSocket messages:
```javascript
const geometryStream = messages.find(m => m.type === 'geometry')?.payload || null;
```
The backend WebSocket at `/ws/orchestrator/{project_id}` broadcasts agent progress, thoughts, and performance metrics — **it never emits a `type: 'geometry'` message**. The viewport always renders nothing from live design runs. The 3D canvas remains empty regardless of what agents produce.

---

### H-04: `main.py` is a 4,821-Line God Module
A single file responsible for: session management, 3 overlapping chat endpoints with different schemas, all 50+ agent-specific HTTP endpoints, WebSocket handlers, user management, plan review, simulation control, shell execution, design genome API, performance monitoring, system health, CORS config, startup lifecycle. This creates:
- Circular import pressure (any submodule that imports `main` causes a cascade)
- Impossible to test individual endpoints in isolation
- Route registration order determines behavior (CORS at line 1017 after routes at line 69)
- The `api/` router split was started but barely used (only `orchestrator_api.py`)

---

### H-05: Three Overlapping Chat Endpoints With Different Schemas
| Endpoint | Model | Session Lookup | LLM Call |
|---|---|---|---|
| `/api/chat` | (broken - wrong schema active) | `conversation_manager` | `conversational_agent.run()` |
| `/api/chat/requirements` | `ChatRequirementsRequest` | `conversation_manager` | `conversational_agent.chat()` |
| `/api/chat/discovery` | Form-based (not JSON) | bare `from conversation_state import` | `conversational_agent.run()` |

The discovery endpoint uses a bare `from conversation_state import conversation_manager` (line 2437) — no `backend.` prefix. If the path isn't set up, this fails with `ModuleNotFoundError`. It also uses `Form()` parameters (multipart) but the frontend sends JSON — the endpoint would always receive empty strings.

---

### H-06: `new_nodes.py` Import Style Breaks When Run as Module
`new_nodes.py` uses:
```python
from xai_stream import inject_thought   # bare, no backend. prefix
from agent_registry import registry     # bare, no backend. prefix
```
But it's imported BY `orchestrator.py` as `from backend.new_nodes import ...`. If Python resolves `new_nodes` in package context (`backend.new_nodes`), its internal bare imports will look for `backend.xai_stream` and `backend.agent_registry` as siblings — which don't resolve. This only works because `main.py` hacks `sys.path`. Any import of `new_nodes` before `sys.path` is patched fails.

---

### H-07: `detect_environment()` vs `run()` API Inconsistency
`EnvironmentAgent` exposes two public methods:
- `run(user_intent)` — standard agent interface, called by orchestrator nodes
- `detect_environment(user_intent)` — same logic, different name

`main.py:731` calls `env_agent.detect_environment(updated_intent)` in the requirements chat handler — inconsistent with how every other agent is called. The orchestrator calls `agent.run(intent)`. This duplicated surface area creates maintenance risk.

---

### H-08: Neural Network max_iter=1 in ThermalAgent
From task.md (confirmed in audit trail): `MLPRegressor(max_iter=1, warm_start=True)` — the model is initialized but never meaningfully trained. Every thermal prediction using the neural path is effectively random noise from an untrained model. The CoolProp analytical path is correct, but the ML path (used for surrogate prediction) returns garbage.

---

### H-09: Multiple Agents Return `None` on Critical Paths
These `None` returns propagate into `AgentState` and produce silent downstream corruption:

| File | Line | Issue |
|---|---|---|
| `gnc_agent.py` | — | ~~C-06 retracted~~ — line 21 is in fallback func, not `run()` |
| `geometry_agent.py` | 518, 586, 1162, 1173, 1281, 1289 | Multiple geometry generation paths |
| `fluid_agent.py` | 1163, 1197 | CFD result paths |
| `lattice_synthesis_agent.py` | 263-265 | Comment: "not implemented, return None" |
| `control_agent.py` | 314, 402 | Control solution paths |
| `mitigation_agent.py` | 104 | Risk mitigation path |
| `component_agent.py` | 232 | Component matching |
| `chemistry_agent.py` | 207 | Corrosion check path |
| `openscad_agent.py` | 48 | Entire compile path if CLI missing |

Orchestrator nodes do not consistently null-check these. A `None` result merged into state via `state.update(result or {})` silently drops the entire agent's contribution.

---

### H-10: `'mock'` AI Model Has No Backend Handler
`SettingsContext.jsx` defaults `aiModel = 'mock'`. `llm/factory.py` has no mock provider. If this setting were ever correctly wired to the backend, it would fall through the entire provider chain and either hit Ollama (if installed) or raise `RuntimeError("No working LLM API Keys...")`. Frontend settings default creates a server error condition.

---

### H-11: `BiologyAgent` Silently Mapped to `ChemistryAgent`
`agent_registry.py:67`:
```python
"BiologyAgent": ("agents.chemistry_agent", "ChemistryAgent"),
```
Any system requesting biology analysis gets chemistry analysis with zero indication of the substitution.

---

## 🟡 MEDIUM — Technical debt, fragile code, wrong assumptions

### M-01: Hardcoded Physics Constants Still Present (Despite Refactoring Claims)

The task.md says hardcodes were removed. They weren't — they were moved in some agents but remain in others:

| Location | Hardcoded Value | Wrong Because |
|---|---|---|
| `fluid_agent.py:56` | `density: float = 1.225` | Only valid at sea level, 15°C |
| `fluid_agent.py:1366` | `rho = 1.225` | Same — ignores altitude/temperature |
| `fluid_agent.py:326` | `Cd_base = 1.05` (cube), `0.3` (general), `0.6` (cylinder) | No source citation, geometry-type lookup hardcoded |
| `thermal_agent.py:152` | `T_ref = 300.0` | Assumed reference temperature |
| `structural_agent.py:789` | `Kt = 1.0` (no stress concentration) | Should compute from geometry |
| `structural_agent.py:814` | `Kt = 1.8` (fillet) | Mid-range guess, range is 1.5-2.5 |
| `gnc_agent.py:411,417` | `g_earth = 9.80665` | Duplicated from config, bypasses config system |
| `fluid_agent.py:873-874` | `k = 0.01 * v**2`, `omega = v / 0.1` | Turbulence intensity pulled from thin air |
| `chemistry_agent.py:271` | `density = 2.7` (aluminum fallback) | Wrong for alloys, no units documented |

---

### M-02: `feedback_agent.py` — Docstring Contains Import Example Inside Its Own Module

`feedback_agent.py:356` — inside the `analyze_failure()` function's docstring:
```python
    """
    Usage:
        from backend.agents.feedback_agent import analyze_failure
    """
```
This is a docstring (not live code), so it doesn't actually create a circular import. But it's misleading when grepping for import issues.

---

### M-03: Leftover Dev Artifacts Throughout Repository

**Duplicate agent files (should pick one and delete the rest):**
- `environment_agent.py`, `environment_agent_original.py`, `environment_agent_refactored.py`
- `thermal_agent.py`, `thermal_agent_original.py`, `thermal_agent_refactored.py`

**Backup file:**
- `backend/orchestrator.py.backup`

**Debug scripts (should be in a `dev/` folder or removed entirely):**
- `backend/debug_openscad.py`, `debug_orchestrator.py`, `debug_pricing.py`, `debug_scooter.py`, `debug_trigger.py`
- `backend/reproduce_500.py`, `verify_compliance.py`, `verify_forensic_wiring.py`, `verify_kimi.py`, `verify_pods.py`, `verify_safety.py`

**Test run JSON dumps committed to repo:**
- `backend/h2o_combined_res.json`, `h2o_decoupled_res.json`, `h2o_final_res.json`, `h2o_parallel_res.json`, `h2o_res.json`, `h2o_safe_res.json`, `h2o_sim_res.json`
- `backend/evolve_req.json`, `evolve_verify.json`, `explore.json`, `explore_res.json`, `res.json`, `interpret_res.json`, `final_verify_res.json`

**Root-level status/report MDs cluttering project root (20+):**
- `AGENTS_API_IMPLEMENTATION_SUMMARY.md`, `AGENTS_COMPLETE_FIX_REPORT.md`, `AGENTS_COVERAGE_ANALYSIS.md`, `AGENTS_FINAL_REPORT.md`, `AGENTS_IMPLEMENTATION_COMPLETE.md`, `AGENTS_REFACTOR_PROGRESS_REPORT.md`, `AGENT_COMPLETION_STATUS.md`, `AGENT_IMPLEMENTATION_RESEARCH_GUIDE.md`, `AGENT_SUITE_IMPLEMENTATION.md`, `BRICK_OS_MASTER_GUIDE.md`, `CONSOLIDATION_SUMMARY.md`, `FUNCTIONAL_AGENTS_REFACTOR_COMPLETE.md`, `IMPLEMENTATION_MASTER_PLAN.md`, `IMPLEMENTATION_STATUS.md`, `NETWORK_AGENT_IMPLEMENTATION.md`, `P0_COMPLETION_SUMMARY.md`, `P0_HARDCODED_VALUES_AUDIT.md`, `PRODUCTION_AGENTS_SUMMARY.md`, `ROADMAP_STATUS.md`, `SDF_INTEGRATION_SUMMARY.md`, `TEST_REPORT.md`

These should be archived or deleted. They inflate the repo and confuse onboarding.

---

### M-04: No Session Persistence Across Restarts
`conversation_manager` is an in-memory singleton with `_auto_save()` writing to disk (JSON file presumably). On server restart: all active sessions are lost. If the auto-save file isn't found, users lose conversation history. No database (Supabase is listed in requirements but not used for sessions), no Redis, no persistent store. The frontend fetches sessions on mount — after a restart, it gets an empty list.

---

### M-05: Mixed `.tsx` / `.jsx` Without TypeScript Config
`frontend/src/pages/Landing.tsx` is TypeScript. Every other file (App.jsx, Workspace.jsx, PanelContext.jsx, etc.) is JSX. There is no `tsconfig.json` in `frontend/`. Vite handles `.tsx` via the React plugin transpilation only — TypeScript type checking is completely disabled. The `.tsx` extension provides zero type safety. Either:
- Add `tsconfig.json` + convert all `.jsx` files → consistent TypeScript
- Convert `Landing.tsx` back to `.jsx` → consistent JavaScript

---

### M-06: `Workspace.jsx` Uses `window.location.reload()` for "New Chat"
`Workspace.jsx:28`:
```javascript
const handleNewChat = () => {
    window.location.reload();
};
```
A full page reload to reset chat state is architecturally wrong in a React SPA. It destroys all React state, re-runs boot sequence check (though `sessionStorage` prevents re-showing the boot screen), forces re-fetching of all sessions, and disconnects any active WebSocket. The correct approach is to call `createNewSession()` from `PanelContext`.

---

### M-07: `App.jsx` Has Placeholder Comment in Production Code
`App.jsx:18`:
```jsx
// ... imports ...

const AppContent = () => {
```
A dev-time placeholder comment left in the production component file.

---

### M-08: `leftPanelRequest` State Never Used
`PanelContext.jsx:25`:
```javascript
const [leftPanelRequest, setLeftPanelRequest] = useState(null);
```
Neither `leftPanelRequest` nor `setLeftPanelRequest` appears in the context `value` prop or anywhere else in the frontend codebase. Dead state.

---

### M-09: `MarkdownViewer` Falls Back to Fake Content on Load Failure
`MarkdownViewer.jsx:29-30`:
```javascript
// Mock content for demo if file doesn't exist
setContent(`# ${fileName}\n\nThis is a preview... placeholder preview.`);
```
When the backend can't serve the file, the user sees fabricated markdown content labeled as their design artifact. No error state is shown. The user may not notice they're looking at a placeholder. Additionally, the endpoint `/api/files/read` does not exist in `main.py`.

---

### M-10: `GenomeViewer` Renders Nothing
`GenomeViewer.jsx:23`:
```javascript
// For now, if no url is provided, we show a professional placeholder
```
The viewer shows a placeholder when no URL is provided. No mechanism exists to populate the URL from a real genome run.

---

### M-11: `ai_model` Parameter Is Received But Never Used
In all three chat endpoints, the `ai_model` / `aiModel` field from the request body is received but the backend always uses the global `conversational_agent` instantiated at startup with `get_llm_provider()` (whichever provider is first available from env vars). Per-request model switching is architecturally stubbed but never implemented.

---

### M-12: LLM Factory Doesn't Document `GROQ_API_KEY` in `.env.example`
`backend/.env.example` only documents `OPENAI_API_KEY` and `GEMINI_API_KEY`. But:
- `GlobalInputConsole` defaults to `'groq'`
- `llm/factory.py` tries Groq first (if key present)
- `GROQ_API_KEY` not in `.env.example`

New developers following the setup guide won't set the Groq key, then wonder why the frontend's default provider doesn't work.

---

### M-13: Async/Sync Agent Interface Fragile Triple-Check Pattern
Throughout `orchestrator.py`, every agent call uses:
```python
if hasattr(agent, "run") and asyncio.iscoroutinefunction(agent.run):
    result = await agent.run(...)
else:
    result = agent.run(...)
    if asyncio.iscoroutine(result):
        result = await result
```
This triple check (is it async? is the result a coroutine?) is repeated for every single agent call (~20 occurrences). The correct approach is to enforce a consistent interface on all agents (either all sync or all async). The current pattern silently breaks for `asyncio.coroutine`-decorated functions (deprecated) and returns wrong results for agents that return awaitable non-coroutines.

---

### M-14: `document_agent.py` Uses Bare `from agent_registry import registry`
`document_agent.py:4`:
```python
from agent_registry import registry
```
No `backend.` prefix. Works only when `sys.path` is pre-configured by `main.py`. Also uses `from llm.provider import LLMProvider` — same bare import pattern. If `DocumentAgent` is ever imported before `main.py` runs (unit tests, CLI tools), both fail.

---

### M-15: `requirements.txt` Missing Key Dependencies
- **`pyyaml`** — used in `environment_agent.py`, `backend/config/__init__.py` via `import yaml`. Not listed. (It may be a transitive dep, but should be explicit.)
- **`fipy`** — imported in `thermal_agent.py` (`from fipy import Grid3D...`). Not listed in `requirements.txt`. FiPy requires a complex install (PETSc, trilinos).
- **`numpy==2.2.6`** — pinned exactly while `cadquery`, `pymatgen`, `scipy`, `scikit-fem` all have their own numpy version ranges. Version conflicts are likely.
- **`CalculiX`** — there's a full CalculiX source tree in `/CalculiX/` but no install step. Users don't know to compile it.
- **`GROQ_API_KEY`** — not in `.env.example` (covered in M-12).

---

### M-16: `optimization_agent.py` Top-Level Imports Will Fail Without Correct sys.path
`optimization_agent.py:7-11`:
```python
from agents.evolution import GeometryGenome, EvolutionaryMutator, EvolutionaryCrossover
from agents.surrogate.pinn_model import MultiPhysicsPINN
from agents.critics.adversarial import RedTeamAgent
from agents.critics.scientist import ScientistAgent
from agents.generative.latent_agent import LatentSpaceAgent
```
These are bare `agents.` imports (not `backend.agents.`). The files exist, but this agent won't import in any context where `backend/agents` isn't directly on `sys.path`. Since it's a top-level import (not wrapped in try/except), `OptimizationAgent` will fail to import entirely in those environments.

---

### M-17: `swarm_manager.py` Uses Bare Import of `EnvironmentAgent`
`swarm_manager.py:3`:
```python
from agents.environment_agent import EnvironmentAgent
```
Same bare import issue. Fails unless `backend/agents` is on `sys.path`.

---

### M-18: 17 Broad `except Exception` Catches Across Agents
Found 17 `except Exception as e:` or `except Exception:` blocks (bare catch-all) that log the error but continue execution with potentially broken state. Each is a location where a real failure (network timeout, malformed data, OOM) is silently swallowed and the agent returns a degraded/partial result that looks valid to callers.

Notable: `chemistry_agent.py:482` uses bare `except Exception:` with no logging at all.

---

## 🔵 ARCHITECTURE / VISION GAPS

### A-01: `ProjectOrchestrator` API Is Orphaned
`backend/api/orchestrator_api.py` and `backend/core/project_orchestrator.py` implement a clean, production-grade project lifecycle API (`POST /api/v1/orchestrator/projects`, phase-by-phase execution, WebSocket streaming, checkpoints). It's registered in `main.py`. **Zero frontend components call these endpoints.** The most complete part of the backend is completely unreachable from the UI.

---

### A-02: Planning Page Has No Backend Connection
`/planning` route exists in React Router. `PlanningPage.jsx` renders a session list and "generate plan" button. The plan generation button calls... nothing. There's no `apiClient.post` in `PlanningPage.jsx` for plan generation. The `document_plan_node` in the orchestrator exists but is only reachable through the LangGraph pipeline (itself unreachable per H-02).

---

### A-03: Thought Streaming Is Polling, Not Push
`useThoughtStream.js` polls `/api/agents/thoughts` on an interval (destructive read). The backend `get_thoughts(clear=True)` clears the buffer on read. If two tabs are open, they race to consume thoughts and each gets a partial view. WebSocket push (already wired in `ws_manager`) should be used instead.

---

### A-04: `xai_stream.py` Uses a Global Deque — Not Thread-Safe in Production
`THOUGHT_STREAM: deque = deque(maxlen=50)` is a module-level global. Under uvicorn with multiple workers (`uvicorn main:app --workers 4`), each worker has its own memory space and its own deque. Thoughts injected in worker A are never visible to worker B. The polling endpoint might return empty while work is happening. Fine for single-worker dev, broken for any production deployment.

---

### A-05: WebSocket `/ws/telemetry` Endpoint — Unprotected Shell-Like Access
`main.py:1481-1498` — the `/ws/telemetry` endpoint sends system metrics (CPU, RAM, process info) over WebSocket with no authentication. This is an information disclosure in any non-localhost deployment.

---

### A-06: Shell Execution Endpoint Has No Input Sanitization
`main.py` has a shell agent endpoint that runs commands via `subprocess`. While it presumably uses `subprocess.run` with a list (not shell=True), the allowed command set and input validation are not apparent from the code. In a multi-tenant or networked context this is a critical security risk.

---

### A-07: No Authentication on Any Endpoint
Zero authentication middleware. All `main.py` endpoints are publicly accessible. The `/api/user/profile`, `/api/plans/*/comments`, `/api/agents/explain`, shell execution — all reachable without credentials. This is fine for localhost dev, catastrophic for any deployment.

---

---

### C-09: `ComplianceAgent` Uses LLM to Generate Regulation Text ✅ FIXED
`backend/agents/compliance_agent.py` — **FIXED 2026-03-29**:
- Removed LLM from `__init__` entirely (was loaded unconditionally even though `run()` never used it)
- `discover_regulations()` LLM fallback removed — now falls back to checking `data/regulatory_rules.json`, then returns empty with a warning
- Fixed syntax error in FastAPI router: `Field(...), description=` → `Field(..., description=`
- `run()` was already JSON/YAML-based — no LLM used in the compliance evaluation path

---

### C-10: LLM → Physics Pipeline Connection Is Missing (Full Stack Never Runs) ✅ FIXED
**FIXED 2026-03-29** — full end-to-end wiring:

**Backend (`main.py`):**
- `_run_physics_pipeline(project_id, user_intent, entities)` background task added
- `/api/chat` endpoint now checks `discovery_complete == True && intent == "design_request"` in `ConversationalAgent` result
- Fires `asyncio.create_task(_run_physics_pipeline(...))` — non-blocking
- Returns `project_id` and `discovery_complete` in response

**Backend (`orchestrator.py`):**
- `geometry_node` now emits `broadcast_state_update(project_id, {"type": "geometry", "payload": {...}})` after successful geometry synthesis (H-03 fix bundled here)

**Frontend (`PanelContext.jsx`):**
- Added `activeProjectId` / `setActiveProjectId` state and context value

**Frontend (`GlobalInputConsole.jsx`):**
- Fixed C-05: file upload now sends `formData` directly (not JSON body when files attached)
- Handles `project_id` in response: calls `setActiveProjectId` + `setIsAgentProcessing(true)`

**Frontend (`Workspace.jsx`):**
- Uses `activeProjectId` from context as primary project source
- `handleNewChat` uses `createNewSession()` instead of `window.location.reload()`

---

## Priority Fix Order

### Immediate (crashes / security / architecture)
1. ~~**Rotate the Groq API key** — RETRACTED, .env was never committed (C-07)~~
2. **Create the 5 missing agent stub files** (C-01) — prevents 500s on agent lazy-load
3. ~~**Fix `GncAgent.run()` return None** (C-06) — RETRACTED, was false positive~~
4. **Remove `AgentVersionRegistry` references** or create the class (C-03)
5. **Fix duplicate `ChatRequest`** — rename second one to `DiscoveryChatRequest` (C-04)
6. ~~**Replace LLM regulation generation in `ComplianceAgent`** with structured YAML rules DB (C-09)~~ ✅ DONE
7. ~~**Wire `ConversationalAgent` → `ProjectOrchestrator`** so physics pipeline is reachable from UI (C-10)~~ ✅ DONE

### High (broken UX features)
8. ~~**Fix FormData file upload** in `GlobalInputConsole` — actually pass `formData` (C-05)~~ ✅ DONE (bundled with C-10)
9. **Wire `ai_model` to `get_llm_provider(preferred=req.ai_model)`** (H-01, M-11)
10. **Connect SettingsContext `aiModel` to `GlobalInputConsole`** via `useSettings()` (H-01)
11. ~~**Emit `type: 'geometry'` messages from WebSocket** so viewport renders results (H-03)~~ ✅ DONE (bundled with C-10)

### Medium (code quality)
11. ~~**Standardize all imports to `backend.` prefix** — eliminate bare imports (C-08, M-14, M-16, M-17)~~ ✅ DONE (production runtime files fixed; test scripts excluded)
12. **Split `main.py`** into routers under `backend/api/` (H-04)
13. **Add `GROQ_API_KEY` to `.env.example`** (M-12)
14. **Add `pyyaml` and `fipy` to `requirements.txt`** (M-15)
15. **Create `structural_agent_fixed.py` or fix the import** to point to `structural_agent.py` (C-02)
16. **Delete dev artifacts**: `_original.py`, `_refactored.py`, `.backup`, debug scripts, JSON dumps (M-03)
17. **Fix `Workspace.jsx` `handleNewChat`** — use `createNewSession()` not `window.reload()` (M-06)
18. **Add tsconfig.json or convert `Landing.tsx` to `.jsx`** (M-05)
19. **Implement `/api/files/read` endpoint** or fix `MarkdownViewer` error state (M-09)
20. **Remove `leftPanelRequest` dead state** from PanelContext (M-08)

### Architecture
21. **Add auth middleware** before any production deployment (A-07)
22. **Push thoughts over WebSocket** instead of polling (A-03)
23. **Wire `PlanningPage` plan generation** to actual endpoint (A-02)
24. **Connect frontend to `ProjectOrchestrator` API** (A-01)

---

### M-19: `Omniviewport.jsx:57` — Hardcoded Mock `activeAgents` Array
`Omniviewport.jsx` line 57 defines a hardcoded `activeAgents` array used to render agent activity indicators in the 3D viewport. This is static mock data — it never receives real agent status from the backend. The viewport always shows the same fake agent activity regardless of what's actually running.

---

### M-20: `Landing.tsx` Also Uses `window.location.reload()` Pattern
`Landing.tsx` uses `window.location.reload()` for navigation/state reset, same anti-pattern as `Workspace.jsx` (M-06). Both should use React state management (`createNewSession()` from PanelContext) instead of full page reloads that destroy all in-memory state.

---

## File Count Summary

| Category | Count |
|---|---|
| Python files in backend | ~431 |
| Agent files with `run()` | ~98 |
| Agents fully implemented | ~40 |
| Agents returning None / stub | ~8+ directly, ~40 partially |
| Missing agent files referenced in registry | 5 |
| Missing non-agent files referenced in code | 2 (`structural_agent_fixed.py`, `agents/designer_agent.py` etc.) |
| Hardcoded physics values remaining | 15+ |
| TODOs / FIXMEs in agents | 25+ |
| Root-level status MD files to clean | 20+ |
| Debug JSON files to clean from backend/ | 10+ |
| Frontend components with mock/placeholder behavior | 4 |
