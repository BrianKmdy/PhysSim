FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

# Install required tools and libraries
RUN apt update\
    && apt install -y python3 python3-pip g++ cmake pkg-config vim\
       libglu1-mesa-dev libegl-dev libopengl-dev
# nvidia-cuda-dev
RUN python3 -m pip install --upgrade pip\
    && pip install conan

# Update default conan profiel settings
RUN conan profile new default --detect
RUN conan profile update settings.compiler.libcxx=libstdc++11 default
RUN conan profile update conf.tools.system.package_manager:mode=install default

WORKDIR /project