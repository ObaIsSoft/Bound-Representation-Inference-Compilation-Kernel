# BRICK OS Tier 1 Agent Dependencies

This document describes the dependencies required for full 3D functionality of the Tier 1 Core Agents.

## Overview

| Agent | 3D Capability | Heavy Dependencies | Install Size |
|-------|--------------|-------------------|--------------|
| Structural | 3D FEA | Gmsh, CalculiX | ~300MB |
| Geometry | CAD B-rep | OpenCASCADE | ~600MB |
| Thermal | 3D Conduction | FiPy | ~100MB |
| Material | API-based | None (web APIs) | ~0MB |

**Total for full 3D:** ~1GB additional dependencies

---

## Quick Install

```bash
# Run the installation script
./scripts/install_dependencies.sh
```

Or install manually by category:

---

## Core Dependencies (Required)

All agents require these:

```bash
pip install pydantic numpy scipy
```

---

## Structural Agent Dependencies

### For 3D Finite Element Analysis (FEA)

#### Gmsh - Mesh Generation

**Purpose:** 3D mesh generation from CAD geometry

**Installation:**
```bash
pip install gmsh-sdk
```

**Size:** ~200MB

**Verification:**
```python
from backend.agents.structural_agent import HAS_GMSH
print(HAS_GMSH)  # True if available
```

#### CalculiX - FEA Solver

**Purpose:** Solve linear/nonlinear structural equations

**Installation:**

**Ubuntu/Debian:**
```bash
sudo apt-get install calculix-ccx
```

**macOS:**
```bash
brew install calculix
```

**Windows:**
Download from https://www.calculix.de/download/

**Size:** ~100MB

**Verification:**
```bash
ccx -v
```

```python
from backend.agents.structural_agent import HAS_CALCULIX
print(HAS_CALCULIX)  # True if available
```

---

## Geometry Agent Dependencies

### For CAD-Quality B-rep Modeling

#### OpenCASCADE

**Purpose:** Industrial-grade CAD kernel (STEP/IGES, fillets, boolean operations)

**Installation:**

**Ubuntu/Debian:**
```bash
sudo apt-get install libocct-foundation-dev libocct-modeling-dev libocct-data-exchange-dev
pip install pythonocc-core
```

**macOS:**
```bash
brew install opencascade
pip install pythonocc-core
```

**Size:** ~500MB

**Verification:**
```python
from backend.agents.geometry_agent import HAS_OPENCASCADE
print(HAS_OPENCASCADE)  # True if available
```

**Note:** If OpenCASCADE is not available, the agent falls back to Manifold3D (which is always available but has limited CAD capabilities).

---

## Thermal Agent Dependencies

### For 3D Heat Transfer Analysis

#### CoolProp

**Purpose:** Thermophysical fluid properties

**Installation:**
```bash
pip install CoolProp
```

**Size:** ~50MB

#### FiPy

**Purpose:** 3D finite volume thermal solver

**Installation:**
```bash
pip install fipy
```

**Size:** ~50MB

**Note:** May conflict with SciPy versions. If issues occur:
```bash
pip install fipy --no-deps
pip install pysparse
```

**Verification:**
```python
from backend.agents.thermal_agent import HAS_FIPY, HAS_COOLPROP
print(f"FiPy: {HAS_FIPY}, CoolProp: {HAS_COOLPROP}")
```

---

## Material Agent Dependencies

### No Heavy Dependencies!

The Material Agent uses web APIs and local caching:
- **MatWeb API** - Requires API key (web-based)
- **NIST Ceramics** - Web API
- **Materials Project** - Web API

Local caching uses SQLite (built into Python).

**Optional API Keys:**
```bash
export MATWEB_API_KEY="your_key_here"
export MATERIALS_PROJECT_API_KEY="your_key_here"
```

---

## Docker Installation (Recommended for Production)

To avoid dependency conflicts, use Docker:

```dockerfile
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    calculix-ccx \
    libocct-foundation-dev \
    libocct-modeling-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install \
    pydantic numpy scipy \
    gmsh-sdk \
    pythonocc-core \
    CoolProp \
    fipy

# Copy BRICK OS
COPY . /app
WORKDIR /app
```

Build and run:
```bash
docker build -t brick-os .
docker run -it brick-os
```

---

## Graceful Degradation

All agents work with limited functionality if dependencies are missing:

| Agent | Without Heavy Deps | Capabilities |
|-------|-------------------|--------------|
| Structural | No Gmsh/CalculiX | 1D analytical (beams) only |
| Geometry | No OpenCASCADE | Mesh generation only (no CAD export) |
| Thermal | No FiPy | 1D heat conduction only |
| Material | No API keys | 3 fallback materials only |

---

## Troubleshooting

### Gmsh Import Error

**Problem:** `ModuleNotFoundError: No module named 'gmsh'`

**Solution:**
```bash
pip install gmsh-sdk
# Verify installation location
python -c "import gmsh; print(gmsh.__file__)"
```

### CalculiX Not Found

**Problem:** `HAS_CALCULIX = False`

**Solution:**
```bash
# Check if ccx is in PATH
which ccx

# If not, add to PATH
export PATH="/usr/local/bin:$PATH"

# Or specify path in code
solver = CalculiXSolver(executable="/path/to/ccx")
```

### OpenCASCADE Import Error

**Problem:** `ModuleNotFoundError: No module named 'OCC'`

**Solution:**
```bash
# Reinstall pythonocc-core
pip uninstall pythonocc-core -y
pip install --no-cache-dir pythonocc-core

# Verify
python -c "from OCC.Core import gp; print('OK')"
```

### FiPy Version Conflicts

**Problem:** Import errors with scipy/numpy

**Solution:**
```bash
# Use conda instead of pip
conda install -c conda-forge fipy

# Or install compatible versions
pip install scipy==1.10.0 numpy==1.23.0 fipy
```

---

## Performance Notes

| Dependency | Import Time | Memory Usage | Notes |
|------------|-------------|--------------|-------|
| Gmsh | ~1s | ~100MB | Lazy-loaded only when meshing |
| CalculiX | N/A | ~50MB | External binary, spawned as needed |
| OpenCASCADE | ~3s | ~300MB | Heavy initialization |
| FiPy | ~2s | ~100MB | Depends on mesh size |

**Recommendation:** Use lazy imports and check availability before initializing agents.

---

## License Considerations

| Dependency | License | Commercial Use |
|------------|---------|----------------|
| Gmsh | GPL | ✓ (with attribution) |
| CalculiX | GPL | ✓ (with attribution) |
| OpenCASCADE | LGPL | ✓ |
| FiPy | BSD | ✓ |
| CoolProp | MIT | ✓ |

**Note:** GPL dependencies (Gmsh, CalculiX) require source code distribution if you distribute BRICK OS as part of a larger work. Consider using Docker to isolate these dependencies.

---

## See Also

- [Installation Script](../scripts/install_dependencies.sh)
- [Docker Setup](../docker/Dockerfile)
- [Agent Documentation](./agents/README.md)
