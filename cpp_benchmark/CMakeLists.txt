cmake_minimum_required(VERSION 3.18)
project(kan_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ── Release build with full optimisation ─────────────────────────────────────
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# ── LibTorch — manual linking to bypass Caffe2's CUDA requirement ─────────────
# find_package(Torch) fails when CUDA libs are absent even for CPU-only builds.
# Instead we point directly at the torch package inside the conda env.
# Pass -DTORCH_INSTALL_PREFIX=... to cmake, or set the env var TORCH_INSTALL_PREFIX.
if(NOT TORCH_INSTALL_PREFIX)
  if(DEFINED ENV{TORCH_INSTALL_PREFIX})
    set(TORCH_INSTALL_PREFIX "$ENV{TORCH_INSTALL_PREFIX}")
  else()
    # Default: conda env used by this project
    set(TORCH_INSTALL_PREFIX
      "/home/suleiman/miniconda3/envs/go2-convex-mpc/lib/python3.10/site-packages/torch")
  endif()
endif()

if(NOT EXISTS "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include/torch/torch.h")
  message(FATAL_ERROR
    "torch/torch.h not found under TORCH_INSTALL_PREFIX=${TORCH_INSTALL_PREFIX}\n"
    "Set -DTORCH_INSTALL_PREFIX=/path/to/site-packages/torch")
endif()

set(TORCH_INCLUDE_DIRS
  "${TORCH_INSTALL_PREFIX}/include"
  "${TORCH_INSTALL_PREFIX}/include/torch/csrc/api/include"
)
set(TORCH_LIB_DIR "${TORCH_INSTALL_PREFIX}/lib")

# Only the CPU-side libs are needed; libtorch_cuda*.so are not linked.
find_library(LIBTORCH      NAMES torch      PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(LIBTORCH_CPU  NAMES torch_cpu  PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)
find_library(LIBC10        NAMES c10        PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH REQUIRED)

message(STATUS "TORCH_INSTALL_PREFIX : ${TORCH_INSTALL_PREFIX}")
message(STATUS "LIBTORCH             : ${LIBTORCH}")
message(STATUS "LIBTORCH_CPU         : ${LIBTORCH_CPU}")
message(STATUS "LIBC10               : ${LIBC10}")

# ── Eigen3 ────────────────────────────────────────────────────────────────────
# Ubuntu: sudo apt install libeigen3-dev
find_package(Eigen3 REQUIRED NO_MODULE)

# ── nlohmann/json (header-only, fetched automatically) ───────────────────────
include(FetchContent)
FetchContent_Declare(
  nlohmann_json
  URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz
  URL_HASH SHA256=d6c65aca6b1ed68e7a182f4757257b107ae403032760ed6ef121c9d55e81757d
)
FetchContent_MakeAvailable(nlohmann_json)

# ── Benchmark executable ──────────────────────────────────────────────────────
add_executable(benchmark
  benchmark.cpp
  sym_eval.cpp
)

target_include_directories(benchmark PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TORCH_INCLUDE_DIRS}
)

target_link_directories(benchmark PRIVATE "${TORCH_LIB_DIR}")

target_link_libraries(benchmark PRIVATE
  "${LIBTORCH}"
  "${LIBTORCH_CPU}"
  "${LIBC10}"
  Eigen3::Eigen
  nlohmann_json::nlohmann_json
)

# torch_cpu requires _GLIBCXX_USE_CXX11_ABI=1 on Linux (matches conda build)
target_compile_definitions(benchmark PRIVATE _GLIBCXX_USE_CXX11_ABI=1)

target_compile_options(benchmark PRIVATE -O3 -Wall -Wextra)

# Runtime: set LD_LIBRARY_PATH to TORCH_LIB_DIR before running the binary.
message(STATUS "Build type   : ${CMAKE_BUILD_TYPE}")
message(STATUS "Eigen3       : ${EIGEN3_VERSION_STRING}")
message(STATUS "Run with     : LD_LIBRARY_PATH=${TORCH_LIB_DIR}:\$LD_LIBRARY_PATH ./benchmark")
