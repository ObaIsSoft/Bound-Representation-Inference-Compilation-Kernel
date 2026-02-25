# CalculiX Installation Guide for macOS

## Status

CalculiX (ccx) is **NOT available** via:
- ❌ Homebrew (no formula)
- ❌ PyPI (not a Python package)
- ❌ Pre-built binaries for macOS (Linux only)

## Installation Options

### Option 1: Build from Source (Full Native)

Run the provided build script:

```bash
chmod +x scripts/install_calculix.sh
./scripts/install_calculix.sh
```

**Requirements:**
- Xcode Command Line Tools: `xcode-select --install`
- gfortran: `brew install gcc`
- ~30 minutes build time

**What it builds:**
1. spooles (linear solver library)
2. ARPACK (eigenvalue solver)
3. CalculiX ccx (FEA solver)

**After installation:**
```bash
# Add to ~/.zshrc or ~/.bashrc:
export PATH="$HOME/.local/bin:$PATH"

# Verify:
ccx -v
```

### Option 2: Docker (Immediate Use)

Full 3D environment with everything pre-built:

```bash
docker-compose -f docker-compose.3d.yml up
```

**Includes:**
- ✅ CalculiX (ccx)
- ✅ OpenCASCADE (pythonocc-core)
- ✅ All pip dependencies

### Option 3: Conda (Local Full Install)

```bash
# Install miniconda from https://docs.conda.io

# Create environment
conda env create -f environment-3d.yml
conda activate brick-3d

# Build CalculiX separately
./scripts/install_calculix.sh
```

## Verification

```bash
# Check if ccx is available
which ccx
ccx -v

# Should output: ccx_2.X
```

## Without CalculiX (Fallback Mode)

The Structural Agent **works without CalculiX** using analytical beam theory:

```python
# Euler-Bernoulli solutions
δ = PL³/(3EI)           # Deflection
σ = My/I                # Bending stress
τ = VQ/(It)             # Shear stress
```

**When to use full FEA:**
- Complex geometries (non-prismatic)
- Stress concentrations
- Buckling analysis
- Non-linear materials
- Dynamic/transient loads

**When analytical works:**
- Standard beams, columns
- Static loads
- Elastic materials
- Quick estimates
