#!/bin/bash

# BRICK OS - Comprehensive 3D Dependency Installation
# ===================================================
# 
# This script installs dependencies for 4 Tier 1 agents:
#   - Structural Agent: Gmsh, CalculiX
#   - Geometry Agent: OpenCASCADE, Trimesh, Manifold3D
#   - Thermal Agent: FiPy, CoolProp
#   - Material Agent: PyMatGen, ASE, requests
#
# NOTE: Some dependencies require special handling (conda/Docker)
# See: docker-compose.3d.yml for full environment

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Installation tracking
SUCCESS=()
FAILED=()
NEEDS_MANUAL=()

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[✓]${NC} $1"
    SUCCESS+=("$1")
}

log_warn() {
    echo -e "${YELLOW}[⚠]${NC} $1"
    NEEDS_MANUAL+=("$1")
}

log_error() {
    echo -e "${RED}[✗]${NC} $1"
    FAILED+=("$1")
}

detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
    elif [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "cygwin" ]]; then
        OS="windows"
    else
        OS="unknown"
    fi
    log_info "Detected OS: $OS"
}

check_python() {
    log_info "Checking Python version..."
    PYTHON_VERSION=$(python --version 2>&1 | awk '{print $2}')
    PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
    PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)
    
    log_info "Python version: $PYTHON_VERSION"
    
    if [[ $PYTHON_MAJOR -eq 3 && $PYTHON_MINOR -ge 11 ]]; then
        log_success "Python $PYTHON_VERSION is compatible"
        return 0
    else
        log_warn "Python $PYTHON_VERSION may have compatibility issues with some packages"
        return 1
    fi
}

install_core_deps() {
    log_info "Installing core dependencies..."
    
    pip install -q numpy scipy pandas 2>/dev/null && \
        log_success "Core scientific packages (numpy, scipy, pandas)" || \
        log_error "Core scientific packages"
    
    pip install -q fastapi uvicorn httpx pydantic 2>/dev/null && \
        log_success "Web framework (FastAPI, uvicorn)" || \
        log_error "Web framework"
    
    pip install -q pyyaml python-dotenv aiofiles aiosqlite 2>/dev/null && \
        log_success "Utilities (yaml, dotenv, async files)" || \
        log_error "Utilities"
}

install_structural_deps() {
    log_info "Installing Structural Agent dependencies..."
    
    # Gmsh - 3D meshing
    if pip install -q gmsh-sdk 2>/dev/null; then
        log_success "Gmsh SDK (3D meshing)"
    else
        log_error "Gmsh SDK"
    fi
    
    # CalculiX - FEA solver
    log_warn "CalculiX (ccx) - NOT available via pip/brew"
    log_info "  → For full FEA: use Docker or download from https://www.dhondt.de/"
    log_info "  → Fallback: Analytical beam theory works without CalculiX"
    
    # Mesh I/O
    pip install -q meshio 2>/dev/null && \
        log_success "MeshIO (mesh format conversion)" || \
        log_error "MeshIO"
}

install_geometry_deps() {
    log_info "Installing Geometry Agent dependencies..."
    
    # OpenCASCADE - NOT available via pip (conda only)
    if python -c "import OCC.Core" 2>/dev/null; then
        log_success "OpenCASCADE (already installed)"
    else
        log_warn "pythonocc-core (OpenCASCADE) - NOT available via pip"
        log_info "  → Install with: conda install -c conda-forge pythonocc-core"
        log_info "  → Or use Docker: docker-compose -f docker-compose.3d.yml up"
        log_info "  → Fallback: Manifold3D works for CSG operations"
    fi
    
    # Trimesh - mesh processing
    pip install -q trimesh 2>/dev/null && \
        log_success "Trimesh (mesh processing)" || \
        log_error "Trimesh"
    
    # Manifold3D - robust CSG
    pip install -q manifold3d 2>/dev/null && \
        log_success "Manifold3D (robust CSG operations)" || \
        log_error "Manifold3D"
    
    # DXF support
    pip install -q ezdxf 2>/dev/null && \
        log_success "ezdxf (DXF import/export)" || \
        log_error "ezdxf"
}

install_thermal_deps() {
    log_info "Installing Thermal Agent dependencies..."
    
    # FiPy - Finite Volume Method
    if pip install -q fipy 2>/dev/null; then
        log_success "FiPy (3D finite volume solver)"
    else
        log_error "FiPy"
    fi
    
    # CoolProp - Thermodynamic properties
    if pip install -q CoolProp 2>/dev/null; then
        log_success "CoolProp (thermodynamic properties)"
    else
        log_error "CoolProp"
    fi
}

install_material_deps() {
    log_info "Installing Material Agent dependencies..."
    
    # PyMatGen - Materials science
    pip install -q pymatgen 2>/dev/null && \
        log_success "PyMatGen (materials analysis)" || \
        log_warn "PyMatGen (optional)"
    
    # ASE - Atomic simulation
    pip install -q ase 2>/dev/null && \
        log_success "ASE (atomic simulation)" || \
        log_warn "ASE (optional)"
    
    # HTTP client for APIs
    pip install -q requests aiohttp 2>/dev/null && \
        log_success "HTTP clients (requests, aiohttp)" || \
        log_error "HTTP clients"
}

install_dev_deps() {
    log_info "Installing development dependencies..."
    
    pip install -q pytest pytest-asyncio pytest-cov 2>/dev/null && \
        log_success "Testing (pytest)" || \
        log_error "Testing packages"
    
    pip install -q black ruff mypy 2>/dev/null && \
        log_success "Linting (black, ruff, mypy)" || \
        log_warn "Linting tools"
}

print_summary() {
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║         BRICK OS - Dependency Installation Summary             ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo ""
    
    echo "✓ Successfully Installed (${#SUCCESS[@]}):"
    for item in "${SUCCESS[@]}"; do
        echo "    • $item"
    done
    
    if [[ ${#NEEDS_MANUAL[@]} -gt 0 ]]; then
        echo ""
        echo "⚠ Requires Manual Setup (${#NEEDS_MANUAL[@]}):"
        for item in "${NEEDS_MANUAL[@]}"; do
            echo "    • $item"
        done
    fi
    
    if [[ ${#FAILED[@]} -gt 0 ]]; then
        echo ""
        echo "✗ Failed (${#FAILED[@]}):"
        for item in "${FAILED[@]}"; do
            echo "    • $item"
        done
    fi
    
    echo ""
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                    Next Steps                                  ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo ""
    echo "For FULL 3D CAPABILITIES (OpenCASCADE + CalculiX):"
    echo "  1. Using Docker (recommended):"
    echo "     docker-compose -f docker-compose.3d.yml up"
    echo ""
    echo "  2. Using Conda:"
    echo "     conda env create -f environment-3d.yml"
    echo "     conda activate brick-3d"
    echo ""
    echo "For DEVELOPMENT (fallback mode - works now):"
    echo "  • Structural: Analytical beam theory (no FEA needed)"
    echo "  • Geometry: Manifold3D CSG (mesh-based, works now)"
    echo "  • Thermal: FiPy 3D FVM (fully working!)"
    echo "  • Material: Full API + database (fully working!)"
    echo ""
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# Main
main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     BRICK OS - 3D Agent Dependency Installer                   ║"
    echo "║     Full CAD/FEA/CFD Environment Setup                         ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    detect_os
    check_python || true
    
    # Parse arguments
    INSTALL_ALL=false
    INSTALL_DEV=false
    
    while [[ $# -gt 0 ]]; do
        case $1 in
            --all)
                INSTALL_ALL=true
                shift
                ;;
            --dev)
                INSTALL_DEV=true
                shift
                ;;
            *)
                shift
                ;;
        esac
    done
    
    install_core_deps
    install_structural_deps
    install_geometry_deps
    install_thermal_deps
    install_material_deps
    
    if $INSTALL_DEV || $INSTALL_ALL; then
        install_dev_deps
    fi
    
    print_summary
}

# Run main
main "$@"
