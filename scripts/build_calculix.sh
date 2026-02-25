#!/bin/bash
# Build CalculiX CCX from source for macOS
# ========================================
# Builds: SPOOLES → ARPACK → CalculiX CCX
# Requires: gcc, gfortran, make (brew install gcc)

set -e

CCX_VERSION="2.22"
CALCULIX_DIR="/Users/obafemi/Documents/dev/brick/CalculiX"
INSTALL_DIR="${HOME}/.local"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m'

log() { echo -e "${BLUE}[BUILD]${NC} $1"; }
success() { echo -e "${GREEN}[✓]${NC} $1"; }
error() { echo -e "${RED}[✗]${NC} $1"; }

# Check dependencies
log "Checking build dependencies..."
if ! command -v gfortran &> /dev/null; then
    error "gfortran not found. Install with: brew install gcc"
    exit 1
fi
if ! command -v cc &> /dev/null && ! command -v gcc &> /dev/null; then
    error "C compiler not found. Install Xcode command line tools"
    exit 1
fi
success "Build dependencies OK (gfortran: $(gfortran --version | head -1))"

# Create directories
mkdir -p "${CALCULIX_DIR}"
cd "${CALCULIX_DIR}"

# ===== STEP 1: Build SPOOLES =====
if [ ! -f "SPOOLES.2.2/spooles.a" ]; then
    log "Downloading and building SPOOLES..."
    
    if [ ! -f "spooles.2.2.tgz" ]; then
        curl -L -o spooles.2.2.tgz "http://www.netlib.org/linalg/spooles/spooles.2.2.tgz" || {
            error "Failed to download SPOOLES"
            exit 1
        }
    fi
    
    mkdir -p SPOOLES.2.2
    cd SPOOLES.2.2
    tar -xzf ../spooles.2.2.tgz
    
    # Fix for macOS - replace malloc.h with stdlib.h
    find . -name "*.c" -exec sed -i.bak 's/#include <malloc.h>/#include <stdlib.h>/g' {} \; 2>/dev/null || true
    find . -name "*.bak" -delete 2>/dev/null || true
    
    # Fix Make.inc for macOS (use cc instead of potentially missing gcc)
    if [ -f "Make.inc" ]; then
        sed -i.bak 's|CC = /usr/lang-4.0/bin/cc|CC = cc|g' Make.inc
        sed -i.bak 's|CC = gcc|CC = cc|g' Make.inc
        rm -f Make.inc.bak
    fi
    
    make lib 2>&1 | tail -5
    
    if [ ! -f "spooles.a" ]; then
        error "SPOOLES build failed"
        exit 1
    fi
    success "SPOOLES built"
    cd ..
else
    success "SPOOLES already built"
fi

# ===== STEP 2: Build ARPACK =====
if [ ! -f "ARPACK/libarpack_SUN4.a" ] && [ ! -f "ARPACK/libarpack.a" ]; then
    log "Downloading and building ARPACK..."
    
    if [ ! -f "arpack96.tar.gz" ]; then
        curl -L -o arpack96.tar.gz "https://www.caam.rice.edu/software/ARPACK/SRC/arpack96.tar.gz" || {
            # Fallback mirror
            curl -L -o arpack96.tar.gz "https://github.com/opencollab/arpack-ng/releases/download/3.9.1/arpack-ng-3.9.1.tar.gz"
        }
    fi
    
    # Also get the patch
    if [ ! -f "patch.tar.gz" ]; then
        curl -L -o patch.tar.gz "https://www.caam.rice.edu/software/ARPACK/SRC/patch.tar.gz" 2>/dev/null || true
    fi
    
    tar -xzf arpack96.tar.gz 2>/dev/null || true
    if [ -f "patch.tar.gz" ]; then
        tar -xzf patch.tar.gz 2>/dev/null || true
    fi
    
    cd ARPACK
    
    # Configure ARmake.inc for macOS
    cat > ARmake.inc << 'AREOF'
home = $(shell pwd)
PLAT = SUN4
FC = gfortran
FFLAGS = -O2 -fallow-argument-mismatch
LDFLAGS =
CD = cd
ECHO = echo
LN = ln
LNFLAGS = -s
MAKE = make
RM = rm
RMFLAGS = -f
SHELL = /bin/sh
AR = ar
ARFLAGS = rv
ARPACKLIB = $(home)/libarpack_$(PLAT).a
LAPACKLIB =
BLASLIB = -framework Accelerate
RANLIB = ranlib
AREOF
    
    # Build
    make lib 2>&1 | tail -5
    
    ARPACK_LIB=""
    if [ -f "libarpack_SUN4.a" ]; then
        ARPACK_LIB="libarpack_SUN4.a"
    elif [ -f "libarpack.a" ]; then
        ARPACK_LIB="libarpack.a"
    fi
    
    if [ -z "$ARPACK_LIB" ]; then
        error "ARPACK build failed"
        exit 1
    fi
    success "ARPACK built ($ARPACK_LIB)"
    cd ..
else
    success "ARPACK already built"
fi

# Find the ARPACK library
ARPACK_LIB_PATH=""
if [ -f "${CALCULIX_DIR}/ARPACK/libarpack_SUN4.a" ]; then
    ARPACK_LIB_PATH="${CALCULIX_DIR}/ARPACK/libarpack_SUN4.a"
elif [ -f "${CALCULIX_DIR}/ARPACK/libarpack.a" ]; then
    ARPACK_LIB_PATH="${CALCULIX_DIR}/ARPACK/libarpack.a"
fi

# ===== STEP 3: Download CalculiX =====
if [ ! -d "ccx_${CCX_VERSION}" ]; then
    log "Downloading CalculiX CCX ${CCX_VERSION}..."
    
    if [ ! -f "ccx_${CCX_VERSION}.src.tar.bz2" ]; then
        curl -L -o "ccx_${CCX_VERSION}.src.tar.bz2" "http://www.dhondt.de/ccx_${CCX_VERSION}.src.tar.bz2" || {
            error "Failed to download CalculiX source"
            exit 1
        }
    fi
    
    tar -xjf "ccx_${CCX_VERSION}.src.tar.bz2"
    
    # The archive extracts to CalculiX/ccx_VERSION - move it up
    if [ -d "CalculiX/ccx_${CCX_VERSION}" ]; then
        mv "CalculiX/ccx_${CCX_VERSION}" .
        rm -rf CalculiX 2>/dev/null || true
    fi
    
    success "CalculiX source extracted"
fi

# ===== STEP 4: Build CalculiX =====
log "Building CalculiX CCX ${CCX_VERSION}..."
cd "ccx_${CCX_VERSION}/src"

SPOOLES_DIR="${CALCULIX_DIR}/SPOOLES.2.2"

# Create macOS-compatible Makefile
cat > Makefile.macos << MKEOF
CFLAGS = -Wall -O2 -I${SPOOLES_DIR} -DARCH="Linux" -DSPOOLES -DARPACK -DMATRIXSTORAGE -DNETWORKOUT
FFLAGS = -Wall -O2 -cpp -fallow-argument-mismatch

CC = cc
FC = gfortran

.c.o :
	\$(CC) \$(CFLAGS) -c \$<
.f.o :
	\$(FC) \$(FFLAGS) -c \$<

include Makefile.inc

SCCXMAIN = ccx_${CCX_VERSION}.c

OCCXF = \$(SCCXF:.f=.o)
OCCXC = \$(SCCXC:.c=.o)
OCCXMAIN = \$(SCCXMAIN:.c=.o)

LIBS = \\
       ${SPOOLES_DIR}/spooles.a \\
       ${ARPACK_LIB_PATH} \\
       -framework Accelerate -lpthread -lm

ccx_${CCX_VERSION}: \$(OCCXMAIN) ccx_${CCX_VERSION}.a \$(LIBS)
	\$(FC) -Wall -O2 -o \$@ \$(OCCXMAIN) ccx_${CCX_VERSION}.a \$(LIBS)

ccx_${CCX_VERSION}.a: \$(OCCXF) \$(OCCXC)
	ar vr \$@ \$?

clean:
	rm -f *.o ccx_${CCX_VERSION} ccx_${CCX_VERSION}.a
MKEOF

# Clean previous builds
make -f Makefile.macos clean 2>/dev/null || true

# Build (using available cores)
NCPU=$(sysctl -n hw.ncpu 2>/dev/null || echo 4)
log "Compiling with ${NCPU} cores (this may take 5-10 minutes)..."
if make -f Makefile.macos -j${NCPU} 2>&1 | tee build.log | tail -5; then
    success "CalculiX CCX built successfully"
else
    error "Build failed - check ${CALCULIX_DIR}/ccx_${CCX_VERSION}/src/build.log"
    echo "Last 30 lines of build log:"
    tail -30 build.log
    exit 1
fi

# ===== STEP 5: Install =====
mkdir -p "${INSTALL_DIR}/bin"
cp "ccx_${CCX_VERSION}" "${INSTALL_DIR}/bin/ccx"
chmod +x "${INSTALL_DIR}/bin/ccx"
success "CalculiX installed to ${INSTALL_DIR}/bin/ccx"

# ===== STEP 6: Verify =====
if "${INSTALL_DIR}/bin/ccx" -v 2>&1 | head -3; then
    success "CalculiX is working!"
else
    error "Installation verification failed"
    exit 1
fi

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║     CalculiX CCX ${CCX_VERSION} - Installation Complete              ║"
echo "╠════════════════════════════════════════════════════════════════╣"
echo "║                                                                ║"
echo "║  Ensure ~/.local/bin is in PATH:                               ║"
echo "║    export PATH=\"\$HOME/.local/bin:\$PATH\"                       ║"
echo "║                                                                ║"
echo "║  Verify: ccx -v                                                ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
