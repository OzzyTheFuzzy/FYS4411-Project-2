#!/bin/bash

# Create build-directory (ignores if it already exists)
mkdir -p build_debug

# Move into the build-directory
cd build_debug

# Run CMake (Defaults to Debug based on your CMakeLists.txt)
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=$PWD/../libtorch ../

# Generate documentation using Doxygen
make doc > /dev/null

# Make the Makefile using eight threads
make -j4

# Move and rename the executable to the top-directory
mv vmc ../vmc_debug

echo "Debug build complete. Run with: ./vmc_debug"