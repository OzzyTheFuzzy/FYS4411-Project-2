cmake_minimum_required(VERSION 3.25)

project(project_1_fys4411)

set(CMAKE_CXX_STANDARD 17)  # LibTorch recommends C++17

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

# ── LibTorch ──────────────────────────────────────────────────────────────────
# Tell CMake where to find TorchConfig.cmake. You can pass this at configure
# time instead:  cmake -DCMAKE_PREFIX_PATH=/path/to/libtorch ..
# or if you installed via pip:
#   cmake -DCMAKE_PREFIX_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)') ..
find_package(Torch REQUIRED)
# ─────────────────────────────────────────────────────────────────────────────

include_directories(
    .
    Hamiltonians
    InitialStates
    Math
    WaveFunctions
    Solvers
)

file(
    GLOB SOURCES "*.cpp"
    GLOB SOURCES "Hamiltonians/*.cpp"
    GLOB SOURCES "InitialStates/*.cpp"
    GLOB SOURCES "Math/*.cpp"
    GLOB SOURCES "Solvers/*.cpp"
    GLOB SOURCES "WaveFunctions/*.cpp"
)

find_package(OpenMP REQUIRED)

add_executable(vmc ${SOURCES})

target_link_libraries(vmc
    PRIVATE OpenMP::OpenMP_CXX
    PRIVATE "${TORCH_LIBRARIES}"   # adds libtorch, libc10, etc.
)

# LibTorch sets its own compiler flags (ABI flags etc.) — apply them to the target
target_compile_options(vmc PRIVATE ${TORCH_CXX_FLAGS})