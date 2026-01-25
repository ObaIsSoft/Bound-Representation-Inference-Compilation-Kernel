# Physics-First Architecture: Validated Master Documentation
**Project:** BRICK OS  
**Status:** Production Ready  
**Date:** 2026-01-25  

---

## 1. Core Philosophy: The Unified Physics Kernel
**Location:** `backend/physics/kernel.py`

The heart of BRICK OS is the Unified Physics Kernel. Unlike traditional CAD tools that add simulation as an after-thought, BRICK OS validates every operation against physical laws in real-time.

### 1.1 Architecture
The kernel operates as a singleton service (`get_physics_kernel()`) that orchestrates:
1.  **Providers**: Low-level libraries (PhysiPy, SciPy, CoolProp) that perform the raw calculations.
2.  **Domains**: High-level modules (Mechanics, Thermodynamics, Fluids) that expose domain-specific APIs.
3.  **Validation**: A layer that checks conservation laws and feasibility constraints.
4.  **Intelligence**: AI routing for equation retrieval and multi-fidelity selection.

### 1.2 Strict Compliance Mode
**Crucial Implementation Detail:**
The kernel operates in **Strict Compliance Mode**. All fallbacks and approximations have been removed. 
- **Initialization:** Fails immediately if dependencies (like `physipy` or `coolprop`) are missing.
- **Constants:** Retrieved strictly from the `FPhysicsProvider`.
- **Units:** Conversions delegated strictly to `PhysiPyProvider`.

---

## 2. Phase Walkthroughs & Implementation Details

### Phase 1: Backend Physics Infrastructure
**Key Components:**
- **Kernel:** The central orchestrator.
- **Providers:** Abstraction layer for `scipy`, `sympy`, `physipy`.
- **Domains:** `fluids.py`, `mechanics.py`, `structures.py`.

**Verification:**
Unit tests in `backend/tests/test_physics_kernel.py` verify the kernel initializes correctly and enforces strict compliance.

### Phase 2: Physics Library Integration
**Key Integrations:**
- **PhysiPy:** For rigorous unit handling (e.g., `1 * m/s + 5 * km/h`).
- **CoolProp:** For accurate material properties (e.g., fluid specifications).
- **SymPy:** For symbolic equation solving.

### Phase 6: Materials System
**Data Source:** Materials Project API & CoolProp.
**Functionality:**
- Query properties like `density`, `yield_strength`, `thermal_conductivity`.
- Real-time lookup of temperature-dependent properties.

### Phase 9: Progressive Assembly Rendering (Visual Physics)
**Problem:** Monolithic OpenSCAD compilation was too slow for complex assemblies (60s+).
**Solution:**
- **Parallel Compilation:** A custom parser (`openscad_parser.py`) breaks code into independent modules.
- **Streaming:** Parts are compiled in parallel (Thread Pool) and streamed to the frontend via SSE.
- **Result:** Compilation time reduced to <15s for F-22 assembly level complexity.

---

## 3. End-to-End Workflow

The system validates a design through the following pipeline:

1.  **Design Intent**: User creates geometry or imports a component.
2.  **Material Assignment**: User assigns a material (e.g., "Aluminum 6061").
3.  **Physics validation**:
    *   Kernel calculates Volume & Mass.
    *   Kernel queries Material Agent for `density` & `yield_strength`.
    *   Kernel validates self-weight feasibility (`stress < yield`).
4.  **Simulation**:
    *   Thermal analysis runs using thermodynamic domain.
    *   Structural analysis runs using mechanics domain.
5.  **Renderer**: SDF (Signed Distance Field) or Progressive Mesh visualizes the result.

---

## 4. Developer Reference

### Key Commands
*   **Run Backend:** `python3 -m uvicorn main:app --reload`
*   **Run Unit Tests:** `python3 -m pytest backend/tests/`
*   **Verify System:** `python3 -m backend.tests.verify_full_system`

### Critical Files
*   `backend/physics/kernel.py`: The brain of the operation.
*   `backend/tests/verify_full_system.py`: The single source of truth for system health.
*   `docs/progressive_rendering.md`: Details on the rendering pipeline.

---
*Created via Aggregation of Phases 1, 2, 6, and 9 documentation.*
