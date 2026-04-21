#!/bin/bash
# Clean all build artifacts from the build directory, preserving CSVs and scripts

BUILD_DIR="$(dirname "$0")/build"

rm -rf "$BUILD_DIR/CMakeFiles"
rm -f "$BUILD_DIR/CMakeCache.txt" \
      "$BUILD_DIR/cmake_install.cmake" \
      "$BUILD_DIR/Makefile" \
      "$BUILD_DIR/spmm_bell"

echo "Build artifacts cleaned. CSVs and scripts preserved."
