FROM nvidia/cuda:11.7.0-devel-ubuntu22.04

RUN apt update\
 && apt -y install g++ cmake libyaml-cpp-dev libspdlog-dev libcxxopts-dev

COPY . /project
WORKDIR /project/build
RUN cmake -DCMAKE_CUDA_ARCHITECTURES=$cuda_arch .. && cmake --build .

WORKDIR /simulation
CMD /project/build/simulator