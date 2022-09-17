#!/bin/bash

# Default GPU architecture, compatible with RTX 30 series
arch=86
build_type=Release
should_clean=0

for arg in "$@"
do
  case "$arg" in
  Debug)
    build_type=Debug
    ;;
  Release)
    build_type=Release
    ;;
  --clean)
    should_clean=1
    ;;
  *)
    arch=$arg
    ;;
  esac
done

if (($should_clean)); then
  if [ -d "build/$build_type" ]; then
    rm -rf build/$build_type
  fi
fi

echo "Building $build_type"\
  && mkdir -p build/$build_type\
  && cd build/$build_type\
  && conan install ../.. --build missing -s build_type=$build_type\
  && cmake -DCMAKE_CUDA_ARCHITECTURES=$arch ../..\
  && cmake --build . --config $build_type