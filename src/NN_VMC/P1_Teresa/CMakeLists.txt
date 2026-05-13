cmake_minimum_required(VERSION 3.25)

project(project_1_fys4411)

set(CMAKE_CXX_STANDARD 17)  # LibTorch recommends C++17

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Debug)
endif()

set(CMAKE_CXX_FLAGS "-Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_CXX_FLAGS_RELEASE "-O3")

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
)
