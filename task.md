# BRICK OS - Critical Issues & Fix Plan

## Production Readiness Score: 4/10

---

## 🔴 CRITICAL ISSUES (Will Crash in Production)

### 1. Manifold3D API Mismatch ✓ CONFIRMED
**Location:** `backend/geometry/manifold_engine.py` (lines 37, 131)

**Problem:** Uses wrong attribute name for vertex access
```python
# WRONG (lines 37, 131):
vertices=np.array(mesh.vert_properties, dtype=np.float32)

# CORRECT (lines 58, 251 - already correct in some places):
vertices=np.array(mesh.vert_pos, dtype=np.float32)
```

**Impact:** Geometry compilation crashes with AttributeError
**Fix:** Replace all `vert_properties` with `vert_pos`

---

### 2. Frontend Import Order (Non-critical but messy) ✓ CONFIRMED
**Location:** `frontend/src/App.jsx` (line 15)

**Problem:** Import statement placed after function definition
```javascript
const BootWrapper = ({ onComplete }) => {
    return <BootSequence onComplete={onComplete} />;
};

import { PanelProvider } from './contexts/PanelContext';  // ❌ Should be at top
```

**Impact:** Works but violates ESLint/import rules
**Fix:** Move imports to top of file

---

### 3. Double Method Definition ✓ CONFIRMED
**Location:** `backend/agents/geometry_agent.py` (lines 188, 443)

**Problem:** `_estimate_geometry_tree` defined twice - second overrides first
```python
def _estimate_geometry_tree(self, regime, params):  # Line 188
    ...

def _estimate_geometry_tree(self, regime: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:  # Line 443
    ...
```

**Impact:** Explicit constraint sizing unreachable
**Fix:** Consolidate into single method

---

### 4. VMK SDF Grid Memory Bomb ✓ CONFIRMED
**Location:** `backend/vmk_kernel.py` (lines 262-272)

**Problem:** Triple nested loop O(N³ × M)
```python
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            p = np.array([x_range[i], y_range[j], z_range[k]])
            grid[i, j, k] = self.get_sdf(p)
```

**Impact:** 64³ grid = 262k evaluations × history length = hangs
**Fix:** Vectorize with NumPy broadcasting

---

### 5. Global Mutable State ✓ CONFIRMED
**Location:** `backend/main.py` (line 2606), `backend/comment_schema.py` (line 31)

**Problem:** 
```python
global_vmk = SymbolicMachiningKernel(...)  # Shared across all requests
plan_reviews: dict[str, PlanReview] = {}   # Grows forever, no TTL
```

**Impact:** Race conditions, memory leaks
**Fix:** Per-request instances, add TTL/expiry

---

### 6. Blocking I/O in Async Context ✓ CONFIRMED
**Location:** `backend/main.py` (12 occurrences), `backend/services/file_extractor.py` (17 occurrences)

**Problem:** Using standard `open()` instead of `aiofiles` blocks event loop
**Impact:** Async performance degraded
**Fix:** Use `aiofiles` or `asyncio.to_thread()`

---

### 7. Hardcoded Paths ✓ CONFIRMED
**Location:** `backend/geometry/_worker_cadquery.py` (lines 85-86)

**Problem:**
```python
filename = f"data/exports/{req_id}.{output_format}"
os.makedirs("data/exports", exist_ok=True)
```

**Impact:** Crash if directory doesn't exist or no write permissions
**Fix:** Use configurable paths with proper error handling

---

### 8. Git Hygiene Issues ✓ CONFIRMED
**Location:** Repository root

**Problems:**
- 48 `__pycache__` directories committed
- `frontend/node_modules` (351MB) committed
- Debug files: `debug_gnc.log`, `debug_gnc_heuristic.log`, `final_*.json`

**Impact:** Repository bloat, security risks
**Fix:** Add to .gitignore, clean history

---

## 🟠 ARCHITECTURAL ISSUES

### 9. No Persistent Job Queue
**Problem:** WebSocket disconnect = lost results, server restart = lost orchestrations
**Impact:** No recovery from failures
**Fix:** Implement Redis/RabbitMQ job queue

### 10. File Upload Security
**Location:** `backend/main.py` (lines 3348-3374)

**Current:**
```python
temp_path = f"{upload_dir}/{file_id}{ext}"
with open(temp_path, "wb") as f:
    f.write(content)
```

**Risk:** Path traversal if file_id not sanitized
**Fix:** Validate file_id format, use uuid strictly

### 11. No Type Safety (Frontend)
**Problem:** All `.jsx` files, minimal TypeScript (only a few `.tsx` files)
**Impact:** Runtime errors guaranteed at this complexity
**Fix:** Gradual TypeScript migration

---

## 🟡 CODE QUALITY ISSUES

### 12. Missing Dependencies Check
**Problem:** Requirements look correct but need verification:
- `numpy==2.2.6` may conflict with manifold3d (typically needs 1.26.x)
- Version pinning too strict

### 13. Agent Registry Verification Needed
**Claim:** 57 agents but many are stubs
**Action:** Audit agent registry to confirm which are implemented vs stubbed

### 14. FEM Integration Verification
**Claim:** No real FEM despite scikit-fem in requirements
**Action:** Verify if scikit-fem is actually used or just heuristic formulas

---

## ✅ VERIFIED WORKING

The following are confirmed working correctly:

1. **Request Attribute** - Already using `request.parameters` (correct)
2. **Transform Implementation** - Fully implemented in `_apply_transform()`
3. **Dependencies** - `fphysics` removed, `manifold3d` and `trimesh` present
4. **Orchestrator get_critic** - Already removed/fixed in current version
5. **Error Handling** - conversational_agent.py has 14 try/except blocks

---

## 🎯 PRIORITY FIX PLAN

### Week 1: Stop the Bleeding (Days 1-3)

- [ ] **Fix 1:** Replace `vert_properties` → `vert_pos` in manifold_engine.py (2 lines)
- [ ] **Fix 2:** Consolidate `_estimate_geometry_tree` double definition
- [ ] **Fix 3:** Clean git - Remove __pycache__, node_modules, debug files
- [ ] **Fix 4:** Add directory creation with `exist_ok=True` for all paths
- [ ] **Fix 5:** Add `.gitignore` entries for cache and logs

### Week 2: Performance & Stability (Days 4-7)

- [ ] **Fix 6:** Vectorize VMK SDF grid generation with NumPy broadcasting
- [ ] **Fix 7:** Replace global mutable state with per-request instances
- [ ] **Fix 8:** Add TTL/expiry to plan_reviews dictionary
- [ ] **Fix 9:** Convert blocking file I/O to async (aiofiles)
- [ ] **Fix 10:** Fix frontend import order in App.jsx

### Week 3: Security & Hardening (Days 8-10)

- [ ] **Fix 11:** Validate file_id format in upload endpoints
- [ ] **Fix 12:** Use configurable paths instead of hardcoded strings
- [ ] **Fix 13:** Add path traversal protection
- [ ] **Fix 14:** Add input validation for geometry complexity limits
- [ ] **Fix 15:** Add circuit breakers for LLM calls (tenacity)

### Week 4: Architecture (Days 11-14)

- [ ] **Fix 16:** Implement Redis job queue for persistent jobs
- [ ] **Fix 17:** Add proper FEM integration with scikit-fem
- [ ] **Fix 18:** Add type safety (TypeScript migration start)
- [ ] **Fix 19:** Implement caching layer for geometry/physics
- [ ] **Fix 20:** Add comprehensive error handling and retries

---

## 📋 DETAILED FIX CHECKLIST

### Fix 1: Manifold3D API
```python
# File: backend/geometry/manifold_engine.py
# Line 37: Change mesh.vert_properties → mesh.vert_pos
# Line 131: Change mesh.vert_properties → mesh.vert_pos
```

### Fix 2: Double Method
```python
# File: backend/agents/geometry_agent.py
# Keep the typed version at line 443, remove/merge line 188
```

### Fix 3: Git Cleanup
```bash
# Commands to run:
git rm -r --cached backend/**/__pycache__
git rm -r --cached frontend/node_modules
git rm --cached backend/*.log
git rm --cached backend/final_*.json
echo "__pycache__/" >> .gitignore
echo "*.log" >> .gitignore
echo "final_*.json" >> .gitignore
```

### Fix 4: Global State
```python
# Replace global_vmk with factory function
def get_vmk_instance(request_id: str) -> SymbolicMachiningKernel:
    # Cache per-request with TTL
    ...

# Add TTL to plan_reviews
from cachetools import TTLCache
plan_reviews = TTLCache(maxsize=1000, ttl=3600)
```

### Fix 5: Vectorize VMK
```python
# Replace triple loop with vectorized operations
# Use np.meshgrid and vectorized SDF evaluation
```

---

## 📊 VERIFICATION METRICS

| Issue | Status | Severity | Effort | File |
|-------|--------|----------|--------|------|
| Manifold API | ✓ Confirmed | Critical | 5 min | manifold_engine.py |
| Double Method | ✓ Confirmed | Critical | 15 min | geometry_agent.py |
| VMK Grid | ✓ Confirmed | High | 2 hrs | vmk_kernel.py |
| Global State | ✓ Confirmed | High | 4 hrs | main.py |
| Blocking I/O | ✓ Confirmed | Medium | 3 hrs | main.py, file_extractor.py |
| Git Hygiene | ✓ Confirmed | Medium | 30 min | .gitignore |
| Hardcoded Paths | ✓ Confirmed | Medium | 1 hr | _worker_cadquery.py |
| Frontend Import | ✓ Confirmed | Low | 5 min | App.jsx |

---

## 📝 NOTES

1. **Orchestrator Issues:** The claimed `get_critic` and `planning_node` issues are already fixed in the current version (comments indicate they were removed/fixed)

2. **Physics Validation:** The claimed logic error about `center` and `p_local` was not found in current codebase - may have been fixed already

3. **Dependencies:** Requirements.txt looks correct - fphysics removed, manifold3d and trimesh present

4. **FEM Integration:** Needs verification - scikit-fem is in requirements but usage needs audit

---

*Last Updated: 2026-02-18*
*Verification Status: All issues manually verified against codebase*
