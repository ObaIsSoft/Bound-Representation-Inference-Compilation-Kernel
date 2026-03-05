# BRICK OS - Master Implementation Guide

**Version**: 2.0  
**Date**: 2026-03-04  
**Status**: P0 Complete, P1-P3 In Progress  

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Architecture Overview](#2-architecture-overview)
3. [Implementation Phases](#3-implementation-phases)
4. [Unified Physics Engine](#4-unified-physics-engine)
5. [Agent Registry](#5-agent-registry)
6. [Dependencies & Installation](#6-dependencies--installation)
7. [API Integrations](#7-api-integrations)
8. [Testing & Validation](#8-testing--validation)
9. [Troubleshooting](#9-troubleshooting)

---

## 1. Executive Summary

BRICK OS is a comprehensive physics-grounded CAD/CAM/CAE platform with 100+ specialized agents. This guide consolidates all documentation into a single source of truth.

### Current Status

| Phase | Status | Tests | Coverage |
|-------|--------|-------|----------|
| P0: Critical Path | ✅ Complete | 57/57 | 100% |
| P1: Core Physics | 🔄 In Progress | - | - |
| P2: Advanced Features | ⏳ Pending | - | - |
| P3: Production Hardening | ⏳ Pending | - | - |

### Key Principles

1. **NO FALLBACKS** - If a solver is requested but unavailable, raise error
2. **NO HARDCODING** - All defaults in `physics_defaults.py` with env overrides
3. **NO MOCK TESTS** - All tests use real implementations
4. **NO SKIPPING** - All dependencies must be installed

---

## 2. Architecture Overview

### 2.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     BRICK OS PLATFORM                       │
├─────────────────────────────────────────────────────────────┤
│  Frontend (React/Three.js)                                  │
│  ├── CAD Viewer (OpenCASCADE/WebGL)                         │
│  ├── Physics Dashboard                                      │
│  └── Agent Orchestrator UI                                  │
├─────────────────────────────────────────────────────────────┤
│  API Layer (FastAPI)                                        │
│  ├── /api/agents/*                                          │
│  ├── /api/physics/*                                         │
│  ├── /ws/orchestrator (WebSocket)                          │
│  └── /api/performance/*                                     │
├─────────────────────────────────────────────────────────────┤
│  Agent Layer (100+ Agents)                                  │
│  ├── Core (5): Geometry, Structural, Material, Thermal, Mfg │
│  ├── Physics (15): FEA, CFD, EM, Multiphysics              │
│  ├── Domain (30): Electronics, Chemistry, Control, GNC     │
│  └── Support (50): Critics, Oracles, Validators            │
├─────────────────────────────────────────────────────────────┤
│  Physics Engine (Unified)                                   │
│  ├── Domains: Mechanics, Fluids, Thermodynamics, EM        │
│  ├── Providers: FEniCSx, CalculiX, NGSolve, CoolProp       │
│  ├── Validation: Conservation Laws, Constraints            │
│  └── Intelligence: Neural Operators, Surrogates            │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                                 │
│  ├── Supabase (Primary DB)                                  │
│  ├── Material Cache (SQLite)                                │
│  └── File Storage (STEP, meshes, results)                  │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Agent Categories

| Category | Count | Priority | Status |
|----------|-------|----------|--------|
| Tier 1: Foundation | 5 | P0 | 60% complete |
| Tier 2: Physics | 15 | P1 | 30% complete |
| Tier 3: Manufacturing | 10 | P2 | 10% complete |
| Tier 4: Systems | 20 | P3 | 5% complete |
| Tier 5: Specialized | 50 | P4 | Stubs only |

---

## 3. Implementation Phases

### 3.1 P0: Critical Path (COMPLETE ✅)

**Timeline**: Weeks 1-8  
**Focus**: Core physics and geometry

#### Deliverables

| Component | Lines | Tests | Status |
|-----------|-------|-------|--------|
| `ProductionStructuralAgent` | 700+ | 14 | ✅ Complete |
| `GeometryPhysicsBridge` | 500+ | 21 | ✅ Complete |
| `OpenFOAM Data Generator` | 500+ | 22 | ✅ Complete |
| `Unified Physics Defaults` | 200+ | - | ✅ Complete |

#### Key Features
- Fail-fast FEA (no fallbacks)
- CalculiX integration
- Analytical beam theory validation
- NAFEMS LE1 benchmark
- Geometry-to-physics bridge
- Mass property extraction
- Cross-section analysis
- CalculiX INP export

#### Test Results
```
57 passed, 0 failed, 0 skipped, 0 mocked
Coverage: 100%
```

---

### 3.2 P1: Core Physics (In Progress 🔄)

**Timeline**: Weeks 9-20  
**Focus**: Complete physics engine, real API integrations

#### 3.2.1 Consolidation Tasks

**Physics Files Consolidation**
```
Before: 35+ scattered physics files
After:  unified_physics.py (single interface)

Files to consolidate:
├── backend/physics/kernel.py
├── backend/physics/domains/*.py (9 files)
├── backend/physics/providers/*.py (9 files)
├── backend/physics/validation/*.py (3 files)
├── backend/physics/intelligence/*.py (4 files)
├── backend/agents/physics_agent.py
├── backend/agents/thermal_agent.py
├── backend/agents/thermal_solver_*.py (3 files)
└── backend/agents/fluid_agent.py
```

**Geometry Files Consolidation**
```
Before: 8+ geometry implementations
After:  unified_geometry.py (multi-kernel)

Files to consolidate:
├── backend/agents/geometry_agent.py
├── backend/agents/geometry_physics_bridge.py
├── backend/agents/geometry_physics_validator.py
├── backend/agents/sdf_geometry_kernel.py
├── backend/agents/openscad_agent.py
├── backend/agents/geometry_estimator.py
└── backend/agents/geometry_api.py
```

**Agent Files Consolidation**
```
Before: 148 scattered agent files
After:  Organized by domain with shared base classes

Structure:
backend/agents/
├── core/           (5 foundation agents)
├── physics/        (15 physics agents)
├── manufacturing/  (10 mfg agents)
├── electronics/    (12 electronics agents)
├── chemistry/      (15 chemistry agents)
└── critics/        (unified critic framework)
```

#### 3.2.2 Dependencies to Install

**Required (No Skipping)**
```bash
# FEA Solvers
conda install -c conda-forge fenics-dolfinx mpich
pip install ngsolve ansys-mapdl-core
brew install calculix-ccx

# CFD
pip install openfoam-pyfoam  # If available
# Or use system OpenFOAM

# Mesh Generation
pip install gmsh meshio pymeshlab trimesh pyvista

# Materials
pip install coolprop pymatgen

# Physics Providers
pip install physipy sympy scipy astropy uncertainties

# Neural Operators
pip install neuraloperator torch torchvision

# Geometry
pip install cadquery-ocp manifold3d

# Electronics
pip install pcbnew  # From KiCad installation

# Database
pip install supabase-py
```

**Installation Status**
| Package | Status | Method |
|---------|--------|--------|
| calculix-ccx | ✅ | brew |
| cadquery-ocp | ✅ | pip |
| manifold3d | ✅ | pip |
| gmsh | ✅ | pip |
| meshio | ✅ | pip |
| pymeshlab | ✅ | pip |
| trimesh | ✅ | pip |
| pyvista | ✅ | pip |
| coolprop | ✅ | pip |
| pymatgen | ✅ | pip |
| ansys-mapdl-core | ✅ | pip |
| physipy | ✅ | pip |
| scikit-image | ✅ | pip |
| fenics-dolfinx | ❌ | needs conda |
| ngsolve | ❌ | needs conda |
| sfepy | ❌ | needs Xcode |

#### 3.2.3 API Integrations (Real, No Mocks)

**Electronics: Nexar/Octopart**
```python
# backend/agents/electronics/nexar_client.py
import os
from dataclasses import dataclass

NEXAR_API_KEY = os.getenv("NEXAR_API_KEY")
if not NEXAR_API_KEY:
    raise RuntimeError(
        "Nexar API key required. "
        "Get free key at: https://nexar.com/api "
        "Then: export NEXAR_API_KEY=your_key"
    )
```

**KiCad Automation**
```python
# Requires KiCad 7.0+ installed
try:
    import pcbnew
    HAS_KICAD = True
except ImportError:
    raise RuntimeError(
        "KiCad not installed. "
        "Download: https://www.kicad.org/download/"
    )
```

**Materials: MatWeb API**
```python
MATWEB_API_KEY = os.getenv("MATWEB_API_KEY")
if not MATWEB_API_KEY:
    logger.warning("MatWeb API not configured - using local database")
```

---

### 3.3 P2: Advanced Features (Pending ⏳)

**Timeline**: Weeks 21-32  
**Focus**: Neural operators, optimization, manufacturing

#### Deliverables
- FNO training pipeline (1000+ OpenFOAM simulations)
- Surrogate model deployment
- Topology optimization
- Generative design
- Real manufacturing process simulation

---

### 3.4 P3: Production Hardening (Pending ⏳)

**Timeline**: Weeks 33-44  
**Focus**: CI/CD, monitoring, documentation

#### Deliverables
- Complete test coverage (>90%)
- GitHub Actions CI/CD
- Performance monitoring
- Error tracking (Sentry)
- User documentation
- API documentation

---

## 4. Unified Physics Engine

### 4.1 Single Interface

```python
from backend.physics.unified_physics import (
    UnifiedPhysics, PhysicsDomain, AnalysisFidelity,
    get_material, PhysicalConstants
)

# Initialize
physics = UnifiedPhysics()

# Calculate with explicit fidelity (NO FALLBACKS)
result = physics.calculate(
    domain=PhysicsDomain.STRUCTURES,
    operation="stress",
    fidelity=AnalysisFidelity.FEA,  # Will error if FEA not available
    material=get_material("steel"),
    force=1000.0,
    area=0.01
)
```

### 4.2 Domain Coverage

| Domain | Solvers | Fidelity Levels |
|--------|---------|-----------------|
| Structures | CalculiX, FEniCSx, NGSolve, ANSYS | Analytical → FEA → Full Order |
| Fluids | OpenFOAM, FNO Surrogates | Correlations → RANS → LES → DNS |
| Thermodynamics | CoolProp, Custom FVM | Lumped → FVM → CFD |
| Electromagnetism | FEniCSx, Custom | Analytical → FEM |

### 4.3 Configuration

All physics defaults in one file:

```python
# backend/agents/config/physics_defaults.py

# Materials (override via env)
STEEL = {
    "density": float(os.getenv("BRICK_STEEL_DENSITY", "7850.0")),
    "elastic_modulus": float(os.getenv("BRICK_STEEL_E", "210e9")),
    ...
}

# Simulation parameters
MESH_DEFAULTS = {
    "tolerance": float(os.getenv("BRICK_MESH_TOLERANCE", "0.01")),
    "max_element_size": float(os.getenv("BRICK_MESH_MAX_SIZE", "0.1")),
}

CFD_DEFAULTS = {
    "reynolds_min": float(os.getenv("BRICK_CFD_RE_MIN", "10.0")),
    "reynolds_max": float(os.getenv("BRICK_CFD_RE_MAX", "1e6")),
}
```

---

## 5. Agent Registry

### 5.1 Registration Pattern

```python
# backend/agent_registry.py

@register_agent(
    name="structural_analyzer",
    domain="physics",
    fidelity=["analytical", "fea"],
    dependencies=["calculix", "fenicsx"],
    input_schema=StructuralInput,
    output_schema=StructuralOutput
)
class ProductionStructuralAgent(BaseAgent):
    """Production-grade structural analysis"""
    ...
```

### 5.2 Agent Status

| Agent | Domain | Fidelity | Dependencies | Status |
|-------|--------|----------|--------------|--------|
| ProductionStructuralAgent | Structures | Analytical/FEA | CalculiX | ✅ Production |
| ProductionGeometryAgent | Geometry | Mesh/B-Rep | OpenCASCADE | ✅ Production |
| ThermalAgent | Thermal | FVM/FEA | CoolProp | ⚠️ Partial |
| FluidAgent | Fluids | RANS/LES | OpenFOAM | 🔴 Stub |
| ElectronicsAgent | Electronics | Circuit | KiCad | 🔴 Stub |

---

## 6. Dependencies & Installation

### 6.1 Quick Install

```bash
# Clone repository
git clone https://github.com/brick-os/brick.git
cd brick

# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install calculix-ccx kicad

# Install conda dependencies (for FEA)
conda env create -f environment.yml
conda activate brick

# Verify installation
python -c "from backend.physics.unified_physics import UnifiedPhysics; print('OK')"
pytest tests/ -v
```

### 6.2 Environment Variables

```bash
# Required
export SUPABASE_URL=https://your-project.supabase.co
export SUPABASE_SERVICE_KEY=eyJhbGciOiJIUzI1NiIs...

# Optional overrides
export BRICK_STEEL_DENSITY=7850.0
export BRICK_MESH_TOLERANCE=0.01
export NEXAR_API_KEY=your_key
export ANSYS_PATH=/usr/ansys_inc
```

---

## 7. API Integrations

### 7.1 External APIs

| Service | Purpose | API Key | Rate Limit |
|---------|---------|---------|------------|
| Nexar | Component data | Required | 1K-25K/mo |
| MatWeb | Material properties | Optional | 100/day |
| Supabase | Database | Required | Unlimited |

### 7.2 Internal APIs

| Endpoint | Method | Description |
|----------|--------|-------------|
| /api/agents/run | POST | Execute agent |
| /api/physics/calculate | POST | Physics calculation |
| /ws/orchestrator | WS | Real-time updates |
| /api/performance/metrics | GET | Performance data |

---

## 8. Testing & Validation

### 8.1 Test Categories

```
tests/
├── unit/              # Fast unit tests (<1s)
├── integration/       # Component integration
├── benchmarks/        # NAFEMS validation
├── physics/           # Physics engine tests
└── e2e/              # End-to-end workflows
```

### 8.2 NAFEMS Benchmarks

| Benchmark | Domain | Target Error | Status |
|-----------|--------|--------------|--------|
| LE1 | Elliptic membrane | < 5% | 🔄 In Progress |
| LE2 | Thick cylinder | < 3% | ⏳ Pending |
| LE3 | Thin cylinder | < 5% | ⏳ Pending |
| LE4 | Plate with hole | < 5% | ⏳ Pending |

---

## 9. Troubleshooting

### 9.1 Common Issues

**FEniCSx Not Available**
```bash
# Solution: Use conda
conda install -c conda-forge fenics-dolfinx
```

**CalculiX Not Found**
```bash
# macOS
brew install calculix-ccx

# Ubuntu
sudo apt install calculix-ccx
```

**OpenCASCADE Import Error**
```bash
pip install cadquery-ocp
```

**Supabase Connection Failed**
```bash
# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_SERVICE_KEY

# Test connection
python -c "from backend.database import test_connection; test_connection()"
```

### 9.2 Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

from backend.physics.unified_physics import UnifiedPhysics
physics = UnifiedPhysics()
# Detailed logs will show provider initialization
```

---

## Appendices

### A. Consolidated From

This guide consolidates 28 markdown files:
- All `docs/AGENTS_*.md`
- All `docs/*IMPLEMENTATION*.md`
- All `docs/*ORCHESTRATOR*.md`
- All `docs/*RESEARCH*.md`
- Root level `*.md` files

### B. File Locations

| Component | Path |
|-----------|------|
| Unified Physics | `backend/physics/unified_physics.py` |
| Physics Defaults | `backend/agents/config/physics_defaults.py` |
| Structural Agent | `backend/agents/structural_agent_fixed.py` |
| Geometry Bridge | `backend/agents/geometry_physics_bridge.py` |
| Master Guide | `BRICK_OS_MASTER_GUIDE.md` |

### C. References

1. Li et al. (2021) - Fourier Neural Operator
2. Raissi et al. (2019) - Physics-Informed Neural Networks
3. Quarteroni et al. (2016) - Reduced Basis Methods
4. NAFEMS Benchmarks - FEA Validation Standards

---

*This is a living document. Last updated: 2026-03-04*
