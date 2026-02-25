#!/bin/bash
# CalculiX Installation Script for macOS
# ======================================
# 
# CalculiX (ccx) is a finite element analysis solver required for
# full 3D structural analysis in BRICK OS.
#
# Since CalculiX is not available via pip or homebrew, this script
# downloads and builds it from source.

set -e

CCX_VERSION="2.21"
INSTALL_DIR="${HOME}/.local"
BUILD_DIR="/tmp/calculix_build_$$"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[✓]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[⚠]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

check_dependencies() {
    log_info "Checking build dependencies..."
    
    local missing=()
    
    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        missing+=("gcc/clang compiler")
    fi
    
    if ! command -v gfortran &> /dev/null; then
        missing+=("gfortran")
    fi
    
    if ! command -v make &> /dev/null; then
        missing+=("make")
    fi
    
    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Install with: xcode-select --install"
        log_info "And: brew install gcc (for gfortran)"
        exit 1
    fi
    
    log_success "Build dependencies satisfied"
}

build_spooles() {
    log_info "Building spooles (linear solver library)..."
    
    cd "$BUILD_DIR"
    
    if [ ! -f "spooles.2.2.tgz" ]; then
        curl -L -o spooles.2.2.tgz "http://www.netlib.org/linalg/spooles/spooles.2.2.tgz"
    fi
    
    tar -xzf spooles.2.2.tgz
    cd spooles.2.2
    
    # Fix for macOS
    find . -name "*.c" -exec sed -i.bak 's/#include <malloc.h>/#include <stdlib.h>/g' {} \;
    
    # Build
    make lib 2>&1 | tail -20
    
    mkdir -p "$INSTALL_DIR/lib"
    cp spooles.a "$INSTALL_DIR/lib/libspooles.a"
    
    log_success "spooles built and installed"
}

build_arpack() {
    log_info "Building ARPACK (eigenvalue solver)..."
    
    cd "$BUILD_DIR"
    
    # Download ARPACK
    if [ ! -d "ARPACK" ]; then
        curl -L -o arpack96.tar.gz "https://www.caam.rice.edu/software/ARPACK/SRC/arpack96.tar.gz"
        tar -xzf arpack96.tar.gz
    fi
    
    cd ARPACK
    
    # Set up make.inc for macOS
    cat > make.inc << 'EOF'
SHELL = /bin/sh
FC = gfortran
FFLAGS = -O -fno-second-underscore
LDR = $(FC)
LDFLAGS =
AR = ar
ARFLAGS = rv
RANLIB = ranlib
BLAS = -framework Accelerate
LAPACK =
EOF
    
    make all 2>&1 | tail -20
    
    cp libarpack.a "$INSTALL_DIR/lib/"
    
    log_success "ARPACK built and installed"
}

build_calculix() {
    log_info "Building CalculiX $CCX_VERSION..."
    
    cd "$BUILD_DIR"
    
    # Download CalculiX
    if [ ! -f "ccx_${CCX_VERSION}.src.tar.bz2" ]; then
        curl -L -o "ccx_${CCX_VERSION}.src.tar.bz2" "http://www.dhondt.de/ccx_${CCX_VERSION}.src.tar.bz2"
    fi
    
    tar -xjf "ccx_${CCX_VERSION}.src.tar.bz2"
    cd "CalculiX/ccx_${CCX_VERSION}/src"
    
    # Modify Makefile for macOS
    cat > Makefile.mac << EOF
# CalculiX Makefile for macOS

CFLAGS = -O2 -I${INSTALL_DIR}/include -DARCH=Linux \
    -I${BUILD_DIR}/spooles.2.2 \
    -I/usr/include \
    -mmacosx-version-min=10.15

FFLAGS = -O2 -fno-automatic -fbounds-check
LDFLAGS = -L${INSTALL_DIR}/lib \
    -lspooles \
    -larpack \
    -framework Accelerate \
    -lm -lgfortran

CC = gcc
FC = gfortran

# Source files
SCCX = $(ls *.c 2>/dev/null | xargs)

OBJ = \$(SCCX:.c=.o)

ccx: \$(OBJ)
	\$(FC) -o ccx \$(OBJ) \$(LDFLAGS)

%.o: %.c
	\$(CC) \$(CFLAGS) -c $< -o $@

clean:
	rm -f *.o ccx
EOF
    
    # Try to build
    if make -f Makefile.mac 2>&1 | tee build.log | tail -30; then
        log_success "CalculiX built successfully"
        
        mkdir -p "$INSTALL_DIR/bin"
        cp ccx "$INSTALL_DIR/bin/ccx"
        chmod +x "$INSTALL_DIR/bin/ccx"
        
        log_success "CalculiX installed to $INSTALL_DIR/bin/ccx"
    else
        log_error "CalculiX build failed - see $BUILD_DIR/CalculiX/ccx_${CCX_VERSION}/src/build.log"
        exit 1
    fi
}

verify_installation() {
    log_info "Verifying installation..."
    
    if [ -f "$INSTALL_DIR/bin/ccx" ]; then
        export PATH="$INSTALL_DIR/bin:$PATH"
        if ccx -v 2>&1 | head -5; then
            log_success "CalculiX is working!"
            return 0
        else
            log_error "CalculiX binary exists but doesn't run"
            return 1
        fi
    else
        log_error "CalculiX binary not found"
        return 1
    fi
}

main() {
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     CalculiX (ccx) Installation for macOS                      ║"
    echo "║     Finite Element Analysis Solver                             ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo ""
    
    mkdir -p "$BUILD_DIR"
    mkdir -p "$INSTALL_DIR/bin" "$INSTALL_DIR/lib"
    
    check_dependencies
    
    log_info "Build directory: $BUILD_DIR"
    log_info "Install directory: $INSTALL_DIR"
    echo ""
    
    # Build dependencies
    build_spooles || log_warn "spooles build may have issues"
    build_arpack || log_warn "ARPACK build may have issues"
    
    # Build CalculiX
    build_calculix
    
    # Verify
    verify_installation
    
    # Cleanup
    log_info "Cleaning up build directory..."
    rm -rf "$BUILD_DIR"
    
    echo ""
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║     Installation Complete                                      ║"
    echo "╠════════════════════════════════════════════════════════════════╣"
    echo "║                                                                ║"
    echo "║  Add to your shell profile (.zshrc/.bashrc):                   ║"
    echo "║    export PATH=\"\$HOME/.local/bin:\$PATH\"                       ║"
    echo "║                                                                ║"
    echo "║  Verify:                                                       ║"
    echo "║    ccx -v                                                      ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
}

# Allow running specific steps
if [ "$1" == "clean" ]; then
    rm -rf /tmp/calculix_build_*
    log_success "Cleaned up build directories"
elif [ "$1" == "verify" ]; then
    export PATH="$HOME/.local/bin:$PATH"
    verify_installation
else
    main
fi
