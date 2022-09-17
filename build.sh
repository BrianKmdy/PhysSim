#!/bin/bash

echo 'Building Release'\
    && rm -rf build\
    && mkdir build\
    && cd build\
    && conan install .. --build missing -c tools.system.package_manager:mode=install -s compiler.libcxx=libstdc++11 -s build_type=Release\
    && cmake ..\
    && cmake --build . --config Release