FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN apt update\
 && apt -y install g++ cmake libyaml-cpp-dev libspdlog-dev libcxxopts-dev

WORKDIR /project/build
COPY . .
CMD cmake .. && cmake --build .