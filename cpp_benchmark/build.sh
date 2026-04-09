#!/usr/bin/env bash
# build.sh — Configures and compiles the C++ benchmark.
#
# Uses the LibTorch that ships inside the conda environment's PyTorch
# installation, so no separate LibTorch download is needed.
#
# Usage:
#   cd cpp_benchmark/
#   chmod +x build.sh && ./build.sh
#   ./build/benchmark

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Locate LibTorch cmake config inside the active PyTorch install ─────────────
# Try to locate TorchConfig.cmake by:
#   1. Running python to ask torch where it lives (needs LD_LIBRARY_PATH set)
#   2. Falling back to a direct find in known conda/venv locations
PYTHON="${PYTHON:-/home/suleiman/miniconda3/envs/go2-convex-mpc/bin/python3}"

TORCH_PREFIX="$("$PYTHON" -c \
    "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'share', 'cmake', 'Torch'))" \
    2>/dev/null || true)"

if [[ -z "$TORCH_PREFIX" ]] || [[ ! -f "$TORCH_PREFIX/TorchConfig.cmake" ]]; then
    # Python couldn't import torch (likely missing LD_LIBRARY_PATH).
    # Fall back: search site-packages for the cmake config directly.
    SITE_PKG="$(dirname "$("$PYTHON" -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || \
                          "$PYTHON" -m site --user-site 2>/dev/null)" 2>/dev/null || true)"
    TORCH_PREFIX="$(find /home/suleiman/miniconda3/envs \
                         /home/suleiman/miniconda3 \
                         /usr/local/lib \
                         /usr/lib \
                    -name "TorchConfig.cmake" -maxdepth 12 2>/dev/null \
                    | head -1 | xargs -I{} dirname {})"
fi

if [[ -z "$TORCH_PREFIX" ]] || [[ ! -f "$TORCH_PREFIX/TorchConfig.cmake" ]]; then
    echo "ERROR: Could not find TorchConfig.cmake."
    echo "  Set TORCH_PREFIX= to the directory containing TorchConfig.cmake, e.g.:"
    echo "  TORCH_PREFIX=/home/suleiman/miniconda3/envs/go2-convex-mpc/lib/python3.10/site-packages/torch/share/cmake/Torch ./build.sh"
    exit 1
fi

echo "Using LibTorch cmake config: $TORCH_PREFIX"

# Eigen3 is installed system-wide (apt install libeigen3-dev)
BUILD_DIR="$SCRIPT_DIR/build"
mkdir -p "$BUILD_DIR"

# Derive the torch install prefix from the cmake config location:
#   .../torch/share/cmake/Torch → .../torch
TORCH_INSTALL_PREFIX="$(dirname "$(dirname "$(dirname "$TORCH_PREFIX")")")"

cmake -S "$SCRIPT_DIR" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE=Release \
      -DTORCH_INSTALL_PREFIX="$TORCH_INSTALL_PREFIX"

cmake --build "$BUILD_DIR" -- -j"$(nproc)"

echo ""
echo "Build complete.  Binary: $BUILD_DIR/benchmark"
echo ""
# Derive the torch lib dir from the cmake config path (../../../lib relative to cmake/Torch/)
TORCH_LIB_DIR="$(dirname "$(dirname "$(dirname "$TORCH_PREFIX")")")/lib"
echo "To run (from cpp_benchmark/):"
echo "  LD_LIBRARY_PATH=\"$TORCH_LIB_DIR:\$LD_LIBRARY_PATH\" ./build/benchmark"
