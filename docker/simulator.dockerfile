FROM build-linux:latest AS builder

WORKDIR /project
COPY . .
RUN ./build.sh --clean Release


FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04

RUN apt update &&\
    apt install -y vim

WORKDIR /app
COPY --from=builder /project/build/Release/bin/simulator .