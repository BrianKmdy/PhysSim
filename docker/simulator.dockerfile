FROM build-linux:latest AS builder

WORKDIR /project
COPY . .
RUN ./build.sh


FROM nvidia/cuda:11.7.0-runtime-ubuntu22.04

WORKDIR /app
COPY --from=builder /project/build/bin/simulator .