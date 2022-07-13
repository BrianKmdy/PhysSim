FROM ubuntu:latest

RUN apt update\
 && apt -y install g++ cmake libyaml-cpp-dev

WORKDIR /project/build
CMD cmake .. && cmake --build .