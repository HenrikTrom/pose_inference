#!/bin/bash
set -e

rm -rf build; mkdir build

cd build

opencv_dir="$(find /usr/local -path '*/opencv4/OpenCVConfig.cmake' -printf '%h\n' -quit)"
cmake_args=()
if [ -n "${opencv_dir}" ]; then
    cmake_args+=("-DOpenCV_DIR=${opencv_dir}")
fi

cmake "${cmake_args[@]}" ..

make -j$(nproc)

sudo make install
